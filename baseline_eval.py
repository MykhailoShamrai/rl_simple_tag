"""Ewaluacja polityk bazowych (losowej i heurystycznych) w simple_tag.

1) macierz polityk drapieżnik × ofiara w konfiguracji domyślnej,
2) wpływ parametrów środowiska (liczba drapieżników, prędkość ofiary, przeszkody).

Wyniki: results/baseline_episodes.csv, results/baseline_summary.csv, results/*.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tag_utils import POLICIES, make_env, run_episode

METRICS = ["adv_return", "prey_return", "collisions", "caught", "first_catch",
           "boundary_pen", "mean_min_dist"]


def evaluate(env_kwargs, adv, prey, episodes, seed0=10_000):
    env = make_env(**env_kwargs)
    rows = [run_episode(env, POLICIES[adv], POLICIES[prey], seed0 + i) for i in range(episodes)]
    env.close()
    return rows


def summarize(rows):
    out = {}
    for k in METRICS:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = np.nanmean(v) if not np.all(np.isnan(v)) else np.nan
        out[k + "_se"] = np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v))) if out[k] == out[k] else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--max-cycles", type=int, default=25)
    args = ap.parse_args()
    Path("results").mkdir(exist_ok=True)
    mc = args.max_cycles

    experiments = []  # (grupa, etykieta, env_kwargs, adv, prey)
    for adv in ["random", "greedy"]:
        for prey in ["random", "flee"]:
            experiments.append(("policy", f"{adv} vs {prey}", dict(max_cycles=mc), adv, prey))
    for adv, prey in [("random", "random"), ("greedy", "flee")]:
        for n in [1, 2, 3, 4]:
            experiments.append(("num_adversaries", n, dict(max_cycles=mc, num_adversaries=n), adv, prey))
        for sp in [0.5, 0.75, 1.0, 1.25]:
            experiments.append(("prey_speed", sp, dict(max_cycles=mc, prey_speed=sp), adv, prey))
        for n in [0, 2, 4]:
            experiments.append(("num_obstacles", n, dict(max_cycles=mc, num_obstacles=n), adv, prey))

    summary, episodes = [], []
    for group, label, kw, adv, prey in experiments:
        rows = evaluate(kw, adv, prey, args.episodes)
        s = summarize(rows)
        s.update(group=group, value=label, adv_policy=adv, prey_policy=prey)
        summary.append(s)
        episodes += [dict(r, group=group, value=label, adv_policy=adv, prey_policy=prey) for r in rows]
        print(f"{group:16s} {str(label):18s} {adv:>6s}/{prey:<6s} "
              f"R_adv={s['adv_return']:7.2f}  R_prey={s['prey_return']:8.2f}  "
              f"catch%={100 * s['caught']:5.1f}  coll={s['collisions']:5.2f}  "
              f"t_first={s['first_catch']:5.1f}  bound={s['boundary_pen']:6.2f}")

    for path, data in [("results/baseline_summary.csv", summary), ("results/baseline_episodes.csv", episodes)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    plot(summary, args.episodes, mc)


def plot(summary, n_ep, mc):
    colors = {("random", "random"): "#2a78d6", ("greedy", "flee"): "#e8590c"}
    titles = {"num_adversaries": "Liczba drapieżników",
              "prey_speed": "Prędkość ofiary (× domyślnej)",
              "num_obstacles": "Liczba przeszkód"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (group, title) in zip(axes, titles.items()):
        for pol, c in colors.items():
            rows = [s for s in summary if s["group"] == group and (s["adv_policy"], s["prey_policy"]) == pol]
            x = [s["value"] for s in rows]
            y = np.array([100 * s["caught"] for s in rows])
            e = np.array([100 * s["caught_se"] for s in rows])
            ax.plot(x, y, "o-", color=c, label=f"{pol[0]} vs {pol[1]}")
            ax.fill_between(x, y - 1.96 * e, y + 1.96 * e, color=c, alpha=0.15)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"% epizodów z ≥1 złapaniem ({mc} kroków)")
    axes[0].legend(title="drapieżnik vs ofiara")
    fig.suptitle(f"Polityki bazowe w simple_tag ({n_ep} epizodów / punkt, 95% CI)")
    fig.tight_layout()
    fig.savefig("results/baseline_configs.png", dpi=150)

    rows = [s for s in summary if s["group"] == "policy"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (k, t) in zip(axes, [("adv_return", "Zwrot drużyny drapieżników"),
                                 ("caught", "% epizodów ze złapaniem"),
                                 ("boundary_pen", "Kara ofiary za opuszczenie obszaru")]):
        scale = 100 if k == "caught" else 1
        ax.bar([s["value"] for s in rows], [scale * s[k] for s in rows],
               yerr=[1.96 * scale * s[k + "_se"] for s in rows], color="#2a78d6", capsize=4)
        ax.set_title(t)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Macierz polityk bazowych (konfiguracja domyślna: 3 drapieżniki, 1 ofiara, 2 przeszkody)")
    fig.tight_layout()
    fig.savefig("results/baseline_policies.png", dpi=150)


if __name__ == "__main__":
    main()
