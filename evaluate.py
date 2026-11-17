"""Ewaluacja wytrenowanych polityk (te same metryki i seedy co baseline_eval.py).

Pojedyncze porównanie:
    python evaluate.py --pred runs/predator_mappo_adv3_spd1.0_s0 --prey random
    python evaluate.py --pred runs/predator_mappo_adv3_spd1.0_s0 --prey random --prey-speed 1.25
    python evaluate.py --pred runs/predator_mappo_adv3_spd1.0_s0 --prey runs/prey_mappo_..._s0

`--pred` / `--prey`: `random`, katalog runu (bierze final.pt, a gdy go brak – latest.pt)
albo konkretny plik checkpointu, np. runs/.../ckpt_1000k.pt.

Pełny zestaw (wszystkie runy w runs/ -> results/trained_eval.csv):
    python evaluate.py --suite

Metodologia: 500 epizodów na porównanie, seedy 10000..10499 (identyczne pozycje startowe dla
wszystkich polityk), polityka stochastyczna (próbkowanie z rozkładu aktora; --greedy = argmax),
średnia ± błąd standardowy po epizodach.
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from ppo import ENV_KEYS, EVAL_SEED0, TeamPolicy, resolve_ckpt
from tag_utils import make_env, random_policy, run_episode

METRICS = ["adv_return", "prey_return", "collisions", "caught", "caught_steps", "first_catch",
           "boundary_pen", "out_steps", "mean_min_dist"]
SPEED_GRID = (0.75, 1.0, 1.25)


def load(spec):
    return None if spec == "random" else TeamPolicy(spec)


def env_config(pred, prey, overrides):
    """Konfiguracja środowiska: z checkpointu (o ile jest) + jawne nadpisania."""
    kw = {}
    for pol in (pred, prey):
        if pol is not None:
            kw.update(pol.cfg.env_kwargs())
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return kw


def evaluate(pred_spec, prey_spec, env_overrides=None, episodes=500, greedy=False):
    torch.set_num_threads(1)
    pred, prey = load(pred_spec), load(prey_spec)
    for pol, team in ((pred, "predator"), (prey, "prey")):
        if pol is not None and pol.team != team:
            raise ValueError(f"{pol.path} to polityka drużyny {pol.team}, nie {team}")
    kw = env_config(pred, prey, env_overrides or {})
    adv_fn = random_policy if pred is None else pred.as_policy_fn(greedy)
    prey_fn = random_policy if prey is None else prey.as_policy_fn(greedy)
    env = make_env(**kw)
    # obserwacje muszą pasować do sieci (np. liczba drapieżników zmienia ich wymiar)
    for pol, name in ((pred, "adversary_0"), (prey, "agent_0")):
        if pol is not None and env.observation_space(name).shape[0] != pol.obs_dim:
            raise ValueError(f"{pol.path}: wymiar obserwacji {pol.obs_dim} nie pasuje do {kw}")
    rows = [run_episode(env, adv_fn, prey_fn, EVAL_SEED0 + i) for i in range(episodes)]
    env.close()
    out = dict(kw)
    for k in METRICS:
        v = np.array([r[k] for r in rows], dtype=float)
        ok = ~np.isnan(v)
        out[k] = float(v[ok].mean()) if ok.any() else float("nan")
        out[k + "_se"] = float(v[ok].std() / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan")
    return out


# ----------------------------------------------------------------------------- pełny zestaw

def _job(job):
    meta, pred, prey, overrides, episodes = job
    try:
        return {**meta, "pred": pred, "prey": prey, **evaluate(pred, prey, overrides, episodes)}
    except Exception as e:  # jeden zepsuty run nie powinien zatrzymać całej ewaluacji
        print(f"BŁĄD {meta}: {e}")
        return None


def build_suite(runs_dir, episodes):
    jobs, configs = [], set()
    for d in sorted(Path(runs_dir).iterdir()):
        if not (d / "final.pt").exists():
            continue
        pol = TeamPolicy(d / "final.pt")
        c = pol.cfg
        meta = dict(run=d.name, team=c.team, algo=c.algo, seed=c.seed,
                    train_num_adversaries=c.num_adversaries, train_prey_speed=c.prey_speed)
        configs.add((c.num_adversaries, c.prey_speed))
        if c.team == "predator":
            # drapieżniki vs losowa ofiara: konfiguracja treningowa + inne prędkości ofiary
            for sp in sorted(set(SPEED_GRID) | {c.prey_speed}):
                jobs.append(({**meta, "kind": "predator"}, str(d / "final.pt"), "random",
                             {"prey_speed": sp}, episodes))
                configs.add((c.num_adversaries, sp))
        else:
            opp = str(resolve_ckpt(c.opponent))
            jobs.append(({**meta, "kind": "prey"}, opp, str(d / "final.pt"), {}, episodes))
            # punkt odniesienia: losowa ofiara przeciw tym samym zamrożonym drapieżnikom
            jobs.append(({**meta, "kind": "prey_ref_random"}, opp, "random", {}, episodes))
    for n_adv, sp in sorted(configs):
        meta = dict(run="random", team="baseline", algo="random", seed=-1,
                    train_num_adversaries=n_adv, train_prey_speed=sp, kind="baseline")
        jobs.append((meta, "random", "random", {"num_adversaries": n_adv, "prey_speed": sp},
                     episodes))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default="random")
    ap.add_argument("--prey", default="random")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--greedy", action="store_true", help="argmax zamiast próbkowania")
    for k in ENV_KEYS:
        ap.add_argument("--" + k.replace("_", "-"), type=float if k == "prey_speed" else int)
    ap.add_argument("--suite", action="store_true", help="ewaluacja wszystkich runów")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="results/trained_eval.csv")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.suite:
        jobs = build_suite(args.runs_dir, args.episodes)
        print(f"{len(jobs)} ewaluacji po {args.episodes} epizodów...")
        with Pool(args.workers) as pool:
            rows = [r for r in pool.imap(_job, jobs) if r is not None]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"zapisano {args.out} ({len(rows)} wierszy)")
        return

    overrides = {k: getattr(args, k) for k in ENV_KEYS}
    r = evaluate(args.pred, args.prey, overrides, args.episodes, args.greedy)
    print(f"drapieżniki: {args.pred}\nofiara:      {args.prey}")
    print("środowisko:  " + ", ".join(f"{k}={r[k]}" for k in ENV_KEYS if k in r))
    print(f"({args.episodes} epizodów, polityka {'argmax' if args.greedy else 'stochastyczna'})")
    labels = dict(collisions="kolizje / epizod", caught="% epizodów ze złapaniem",
                  first_catch="krok 1. złapania", caught_steps="kroki z kontaktem",
                  mean_min_dist="dystans ofiara–najbliższy drapieżnik",
                  out_steps="kroki ofiary poza planszą", adv_return="zwrot drapieżników",
                  prey_return="zwrot ofiary")
    for k, lab in labels.items():
        scale = 100 if k == "caught" else 1
        print(f"  {lab:38s} {scale * r[k]:8.2f} ± {scale * r[k + '_se']:.2f}")


if __name__ == "__main__":
    main()
