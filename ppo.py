"""PPO ze współdzieleniem parametrów w drużynie – warianty IPPO i MAPPO (własna implementacja).

Jedna sieć aktora i jedna sieć krytyka na całą drużynę (drapieżniki albo ofiary); do wejścia
dokładany jest one-hot numeru agenta w drużynie. Warianty różnią się WYŁĄCZNIE wejściem krytyka:

  IPPO  – krytyk V(o_i): własna obserwacja agenta,
  MAPPO – krytyk V(s):   stan globalny = konkatenacja obserwacji wszystkich agentów (CTDE).

Aktor w obu wariantach widzi tylko własną obserwację (zdecentralizowane wykonanie).

Uczy się jedna drużyna, przeciwnik jest stały: polityka losowa albo zamrożony checkpoint.
Plan projektu: (1) drapieżniki vs losowa ofiara, (2) ofiara vs zamrożone drapieżniki z (1).

Przykłady:
    # nowy trening drapieżników (MAPPO) przeciw losowej ofierze
    python ppo.py --team predator --algo mappo --seed 0 --total-steps 3000000

    # kontynuacja przerwanego treningu (albo wydłużenie: podaj większe --total-steps)
    python ppo.py --resume runs/predator_mappo_adv3_spd1.0_s0 --total-steps 5000000

    # trening ofiary przeciw zamrożonym drapieżnikom
    python ppo.py --team prey --algo mappo --opponent runs/predator_mappo_adv3_spd1.0_s0/final.pt

Ctrl+C (albo SIGTERM) zapisuje checkpoint `latest.pt` i kończy program – trening można potem
wznowić przez --resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from tag_utils import make_env, random_policy, run_episode

ENV_KEYS = ("num_adversaries", "num_good", "num_obstacles", "max_cycles", "prey_speed")
EVAL_SEED0 = 10_000            # te same seedy co w baseline_eval.py -> porównywalne wyniki
TRAIN_SEED0 = 1_000_000_000    # seedy resetów w treningu: rozłączne z ewaluacyjnymi


@dataclass
class Config:
    # co i jak uczymy
    team: str = "predator"            # predator | prey
    algo: str = "mappo"               # ippo | mappo
    opponent: str = "random"          # random | ścieżka do checkpointu (.pt) lub katalogu runu
    seed: int = 0
    total_steps: int = 3_000_000      # kroki środowiska (jeden krok = ruch wszystkich agentów)
    # środowisko
    num_adversaries: int = 3
    num_good: int = 1
    num_obstacles: int = 2
    max_cycles: int = 25
    prey_speed: float = 1.0
    # PPO
    n_envs: int = 16
    rollout_len: int = 125            # kroków na środowisko na iterację (5 epizodów po 25)
    lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatches: int = 4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    reward_scale: float = 0.1         # nagrody ±10 za kolizję -> ±1
    # logowanie i checkpointy
    eval_every: int = 25              # iteracji
    eval_episodes: int = 100
    save_every: int = 10              # iteracji (latest.pt)
    snapshot_every: int = 500_000     # kroków (ckpt_<k>k.pt – "zamrożone" stany pośrednie)
    threads: int = 1
    run_dir: str = ""

    def env_kwargs(self):
        return {k: getattr(self, k) for k in ENV_KEYS}


# ----------------------------------------------------------------------------- pomocnicze

def team_agents(agents, team):
    prefix = "adversary" if team == "predator" else "agent"
    return [a for a in agents if a.startswith(prefix)]


def other_team(team):
    return "prey" if team == "predator" else "predator"


def resolve_ckpt(path):
    """Katalog runu -> final.pt (jeśli jest) albo latest.pt; plik -> bez zmian."""
    p = Path(path)
    if p.is_dir():
        for name in ("final.pt", "latest.pt"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"brak final.pt/latest.pt w {p}")
    return p


def mlp(inp, out, hidden, out_gain):
    layers = [nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
              nn.Linear(hidden, out)]
    for i, layer in enumerate(l for l in layers if isinstance(l, nn.Linear)):
        nn.init.orthogonal_(layer.weight, out_gain if i == 2 else np.sqrt(2))
        nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, n_team, n_actions, hidden):
        super().__init__()
        self.net = mlp(obs_dim + n_team, n_actions, hidden, 0.01)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = mlp(in_dim, 1, hidden, 1.0)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def actor_input(team_obs):
    """team_obs: (B, K, obs_dim) -> (B, K, obs_dim + K) z one-hot numeru agenta."""
    B, K, _ = team_obs.shape
    eye = np.broadcast_to(np.eye(K, dtype=np.float32), (B, K, K))
    return np.concatenate([team_obs, eye], axis=-1)


def critic_input(algo, team_obs, state):
    """IPPO: własna obserwacja + id; MAPPO: stan globalny (B, S) + id."""
    if algo == "ippo":
        return actor_input(team_obs)
    B, K, _ = team_obs.shape
    eye = np.broadcast_to(np.eye(K, dtype=np.float32), (B, K, K))
    return np.concatenate([np.broadcast_to(state[:, None, :], (B, K, state.shape[-1])), eye], -1)


# ----------------------------------------------------------------------------- zamrożona polityka

class TeamPolicy:
    """Wczytana z checkpointu polityka drużyny (tylko aktor) – do ewaluacji i jako przeciwnik."""

    def __init__(self, ckpt_path):
        self.path = str(resolve_ckpt(ckpt_path))
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        self.cfg = Config(**ck["config"])
        self.team = self.cfg.team
        self.n_team, self.obs_dim = ck["n_team"], ck["obs_dim"]
        self.env_steps = ck["env_steps"]
        self.actor = Actor(self.obs_dim, self.n_team, ck["n_actions"], self.cfg.hidden)
        self.actor.load_state_dict(ck["actor"])
        self.actor.eval()

    @torch.no_grad()
    def act_batch(self, team_obs, greedy=False):
        logits = self.actor(torch.as_tensor(actor_input(team_obs)))
        if greedy:
            return logits.argmax(-1).numpy()
        return Categorical(logits=logits).sample().numpy()

    def as_policy_fn(self, greedy=False):
        """Funkcja w formacie tag_utils.run_episode: (env, agent, rng, obs) -> akcja."""

        @torch.no_grad()
        def fn(env, agent, rng, obs):
            idx = int(agent.rsplit("_", 1)[1])
            x = np.zeros((1, self.obs_dim + self.n_team), np.float32)
            x[0, :self.obs_dim] = obs
            x[0, self.obs_dim + idx] = 1.0
            logits = self.actor(torch.as_tensor(x))[0]
            if greedy:
                return int(logits.argmax())
            p = torch.softmax(logits, -1).numpy().astype(np.float64)
            return int(rng.choice(len(p), p=p / p.sum()))

        return fn


def policy_fns(team, team_fn, opp_fn):
    """Zwraca (adv_policy, prey_policy) dla run_episode."""
    return (team_fn, opp_fn) if team == "predator" else (opp_fn, team_fn)


# ----------------------------------------------------------------------------- trener

class Trainer:
    def __init__(self, cfg: Config, resume_ckpt=None):
        self.cfg = cfg
        torch.set_num_threads(cfg.threads)
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.envs = [make_env(**cfg.env_kwargs()) for _ in range(cfg.n_envs)]
        env0 = self.envs[0]
        self.agents = list(env0.possible_agents)
        self.team = team_agents(self.agents, cfg.team)
        self.opp = team_agents(self.agents, other_team(cfg.team))
        self.K = len(self.team)
        self.obs_dim = env0.observation_space(self.team[0]).shape[0]
        self.n_actions = env0.action_space(self.team[0]).n
        self.state_dim = sum(env0.observation_space(a).shape[0] for a in self.agents)
        critic_dim = (self.obs_dim if cfg.algo == "ippo" else self.state_dim) + self.K

        self.actor = Actor(self.obs_dim, self.K, self.n_actions, cfg.hidden)
        self.critic = Critic(critic_dim, cfg.hidden)
        self.optim = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=cfg.lr, eps=1e-5)

        # przeciwnik
        self.opp_policy = None
        if cfg.opponent != "random":
            self.opp_policy = TeamPolicy(cfg.opponent)
            oc = self.opp_policy.cfg
            if oc.team != other_team(cfg.team):
                raise ValueError(f"przeciwnik {cfg.opponent} trenował drużynę {oc.team}")
            if oc.env_kwargs() != cfg.env_kwargs():
                raise ValueError(f"inna konfiguracja środowiska przeciwnika: {oc.env_kwargs()}")

        self.iteration, self.env_steps, self.resets = 0, 0, 0
        self.best_eval = -np.inf
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        if resume_ckpt is not None:
            self._load(resume_ckpt)

        self.stop_requested = False
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    # ---------------------------------------------------------------- checkpointy
    def _state(self):
        return dict(config=asdict(self.cfg), actor=self.actor.state_dict(),
                    critic=self.critic.state_dict(), optim=self.optim.state_dict(),
                    iteration=self.iteration, env_steps=self.env_steps, resets=self.resets,
                    best_eval=self.best_eval, n_team=self.K, obs_dim=self.obs_dim,
                    n_actions=self.n_actions, np_rng=self.rng.bit_generator.state,
                    torch_rng=torch.get_rng_state())

    def save(self, name):
        tmp = self.run_dir / (name + ".tmp")
        torch.save(self._state(), tmp)
        tmp.replace(self.run_dir / name)

    def _load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.optim.load_state_dict(ck["optim"])
        for g in self.optim.param_groups:
            g["lr"] = self.cfg.lr
        self.iteration, self.env_steps = ck["iteration"], ck["env_steps"]
        self.resets, self.best_eval = ck["resets"], ck["best_eval"]
        self.rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        # usuń z logów wiersze zapisane po tym checkpoincie (np. po twardym przerwaniu)
        for log in ("log.csv", "eval.csv"):
            p = self.run_dir / log
            if p.exists():
                with open(p) as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    keep = [r for r in rows if int(r["env_steps"]) <= self.env_steps]
                    with open(p, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        w.writeheader()
                        w.writerows(keep)
        print(f"wznowiono z {path}: iteracja {self.iteration}, {self.env_steps} kroków")

    def _on_signal(self, signum, frame):
        if self.stop_requested:          # drugi Ctrl+C – wyjdź od razu
            raise KeyboardInterrupt
        print("\nprzerwanie – zapisuję checkpoint po bieżącej iteracji (ponowny Ctrl+C = wyjście)")
        self.stop_requested = True

    # ---------------------------------------------------------------- środowisko
    def _reset(self, i):
        obs, _ = self.envs[i].reset(seed=TRAIN_SEED0 + self.cfg.seed * 10_000_000 + self.resets)
        self.resets += 1
        return obs

    def _team_obs(self, obs_list, names):
        return np.stack([np.stack([o[a] for a in names]) for o in obs_list]).astype(np.float32)

    def _state_vec(self, obs_list):
        return np.stack([np.concatenate([o[a] for a in self.agents]) for o in obs_list]).astype(
            np.float32)

    def _inputs(self, obs_list):
        tobs = self._team_obs(obs_list, self.team)
        return actor_input(tobs), critic_input(self.cfg.algo, tobs, self._state_vec(obs_list))

    @torch.no_grad()
    def _value(self, c_in):
        return self.critic(torch.as_tensor(c_in)).numpy()

    def _opp_actions(self, obs_list):
        n, k = len(obs_list), len(self.opp)
        if self.opp_policy is None:
            return self.rng.integers(self.n_actions, size=(n, k))
        return self.opp_policy.act_batch(self._team_obs(obs_list, self.opp))

    # ---------------------------------------------------------------- zbieranie danych
    def collect(self, obs_list, ep_stats):
        c, T, N, K = self.cfg, self.cfg.rollout_len, self.cfg.n_envs, self.K
        a_in, c_in = self._inputs(obs_list)
        buf = {k: np.zeros((T, N, K) + s, np.float32) for k, s in
               [("a_in", a_in.shape[2:]), ("c_in", c_in.shape[2:]), ("act", ()), ("logp", ()),
                ("val", ()), ("rew", ()), ("done", ()), ("final_val", ())]}

        for t in range(T):
            a_in, c_in = self._inputs(obs_list)
            with torch.no_grad():
                dist = Categorical(logits=self.actor(torch.as_tensor(a_in)))
                act = dist.sample()
                logp = dist.log_prob(act).numpy()
            act = act.numpy()
            val = self._value(c_in)
            opp_act = self._opp_actions(obs_list)

            final_obs, final_idx = [], []
            for i, env in enumerate(self.envs):
                actions = {a: int(act[i, j]) for j, a in enumerate(self.team)}
                actions.update({a: int(opp_act[i, j]) for j, a in enumerate(self.opp)})
                obs, rew, term, trunc, _ = env.step(actions)
                r = np.array([rew[a] for a in self.team], np.float32)
                buf["rew"][t, i] = r * c.reward_scale
                ep_stats["ret"][i] += r[0] if c.team == "predator" else r.sum()
                ep_stats["coll"][i] += rew[self.agents[0]] / 10.0   # adversary_0: 10 / kolizja
                if not env.agents:                                   # koniec epizodu
                    terminal = any(term.values())
                    buf["done"][t, i] = 1.0
                    if not terminal:                                 # obcięcie czasem -> bootstrap
                        final_obs.append(obs)
                        final_idx.append(i)
                    ep_stats["done_ret"].append(ep_stats["ret"][i])
                    ep_stats["done_coll"].append(ep_stats["coll"][i])
                    ep_stats["ret"][i] = ep_stats["coll"][i] = 0.0
                    obs = self._reset(i)
                obs_list[i] = obs
            if final_obs:
                _, fc_in = self._inputs(final_obs)
                buf["final_val"][t, final_idx] = self._value(fc_in)

            buf["a_in"][t], buf["c_in"][t] = a_in, c_in
            buf["act"][t], buf["logp"][t], buf["val"][t] = act, logp, val

        self.env_steps += T * N
        _, c_in = self._inputs(obs_list)
        last_val = self._value(c_in)

        # GAE(λ); na końcu epizodu wartość następnego stanu = V(stan końcowy) (0, gdy terminal)
        adv = np.zeros((T, N, K), np.float32)
        gae = np.zeros((N, K), np.float32)
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else buf["val"][t + 1]
            done = buf["done"][t]
            next_val = np.where(done > 0, buf["final_val"][t], next_val)
            delta = buf["rew"][t] + c.gamma * next_val - buf["val"][t]
            gae = delta + c.gamma * c.gae_lambda * (1.0 - done) * gae
            adv[t] = gae
        buf["adv"], buf["ret"] = adv, adv + buf["val"]
        return buf

    # ---------------------------------------------------------------- aktualizacja PPO
    def update(self, buf):
        c = self.cfg
        data = {k: torch.as_tensor(v.reshape((-1,) + v.shape[3:])) for k, v in buf.items()}
        n = data["act"].shape[0]
        mb = n // c.minibatches
        stats = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}
        for _ in range(c.epochs):
            perm = torch.randperm(n)
            for s in range(0, n - mb + 1, mb):
                idx = perm[s:s + mb]
                dist = Categorical(logits=self.actor(data["a_in"][idx]))
                logp = dist.log_prob(data["act"][idx].long())
                ratio = torch.exp(logp - data["logp"][idx])
                a = data["adv"][idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * a).mean()
                v_loss = ((self.critic(data["c_in"][idx]) - data["ret"][idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg + c.vf_coef * v_loss - c.ent_coef * ent
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    c.max_grad_norm)
                self.optim.step()
                with torch.no_grad():
                    stats["pg_loss"].append(pg.item())
                    stats["v_loss"].append(v_loss.item())
                    stats["entropy"].append(ent.item())
                    stats["approx_kl"].append(((ratio - 1) - torch.log(ratio)).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > c.clip).float().mean().item())
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        y, v = data["ret"].numpy(), data["val"].numpy()
        out["explained_var"] = float(1 - np.var(y - v) / (np.var(y) + 1e-8))
        return out

    # ---------------------------------------------------------------- ewaluacja
    def evaluate(self, episodes):
        """Te same metryki i seedy co ewaluacja baseline'u (tag_utils.run_episode)."""
        frozen = TeamPolicy.__new__(TeamPolicy)       # bieżący aktor w opakowaniu TeamPolicy
        frozen.actor, frozen.n_team, frozen.obs_dim = self.actor, self.K, self.obs_dim
        opp_fn = random_policy if self.opp_policy is None else self.opp_policy.as_policy_fn()
        adv_fn, prey_fn = policy_fns(self.cfg.team, frozen.as_policy_fn(), opp_fn)
        env = make_env(**self.cfg.env_kwargs())
        rows = [run_episode(env, adv_fn, prey_fn, EVAL_SEED0 + i) for i in range(episodes)]
        env.close()
        return {k: float(np.nanmean([r[k] for r in rows])) if not all(
            np.isnan(r[k]) for r in rows) else float("nan") for k in rows[0]}

    # ---------------------------------------------------------------- pętla główna
    def _append_csv(self, name, row):
        p = self.run_dir / name
        new = not p.exists()
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def _do_eval(self, t_start):
        m = self.evaluate(self.cfg.eval_episodes)
        self._append_csv("eval.csv", dict(iteration=self.iteration, env_steps=self.env_steps,
                                          wall_s=round(time.time() - t_start, 1), **m))
        score = m["adv_return"] if self.cfg.team == "predator" else m["prey_return"]
        if score > self.best_eval:
            self.best_eval = score
            self.save("best.pt")
        return m

    def train(self):
        c = self.cfg
        (self.run_dir / "config.json").write_text(json.dumps(asdict(c), indent=2))
        obs_list = [self._reset(i) for i in range(c.n_envs)]
        ep_stats = {"ret": np.zeros(c.n_envs), "coll": np.zeros(c.n_envs),
                    "done_ret": [], "done_coll": []}
        t_start = time.time()
        if self.iteration == 0:
            self._do_eval(t_start)                     # punkt startowy krzywej uczenia
        next_snapshot = (self.env_steps // c.snapshot_every + 1) * c.snapshot_every

        while self.env_steps < c.total_steps and not self.stop_requested:
            t0 = time.time()
            buf = self.collect(obs_list, ep_stats)
            st = self.update(buf)
            self.iteration += 1
            coll = np.array(ep_stats["done_coll"])
            row = dict(iteration=self.iteration, env_steps=self.env_steps,
                       wall_s=round(time.time() - t_start, 1),
                       ep_return=float(np.mean(ep_stats["done_ret"])),
                       collisions=float(coll.mean()), catch_rate=float((coll > 0).mean()),
                       **st, fps=round(c.rollout_len * c.n_envs / (time.time() - t0)))
            ep_stats["done_ret"], ep_stats["done_coll"] = [], []
            self._append_csv("log.csv", row)

            if self.iteration % c.eval_every == 0:
                m = self._do_eval(t_start)
                print(f"[{self.run_dir.name}] it {self.iteration:5d}  kroki {self.env_steps:>9,}"
                      f"  ret {row['ep_return']:7.2f}  kolizje/ep {row['collisions']:5.2f}  "
                      f"| eval: kolizje/ep {m['collisions']:5.2f}  złapania "
                      f"{100 * m['caught']:5.1f}%  | {row['fps']} kroków/s", flush=True)
            if self.iteration % c.save_every == 0:
                self.save("latest.pt")
            if self.env_steps >= next_snapshot:
                self.save(f"ckpt_{next_snapshot // 1000}k.pt")
                next_snapshot += c.snapshot_every

        self.save("latest.pt")
        if self.env_steps >= c.total_steps:
            if self.iteration % c.eval_every:
                self._do_eval(t_start)
            self.save("final.pt")
            print(f"[{self.run_dir.name}] koniec: {self.env_steps:,} kroków, "
                  f"{time.time() - t_start:.0f} s -> {self.run_dir / 'final.pt'}")
        else:
            print(f"[{self.run_dir.name}] zatrzymano po {self.env_steps:,} krokach; wznów: "
                  f"python ppo.py --resume {self.run_dir}")


# ----------------------------------------------------------------------------- CLI

def default_run_name(cfg: Config):
    if cfg.team == "predator":
        opp = "" if cfg.opponent == "random" else "_vs-" + Path(cfg.opponent).parent.name
    else:
        opp = "_vs-random" if cfg.opponent == "random" else \
            "_vs-" + resolve_ckpt(cfg.opponent).parent.name
    return f"{cfg.team}_{cfg.algo}_adv{cfg.num_adversaries}_spd{cfg.prey_speed}{opp}_s{cfg.seed}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", help="katalog runu (wznawia z latest.pt) lub plik .pt")
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        typ = type(f.default)
        if f.name == "total_steps":
            typ = lambda s: int(float(s))      # pozwala pisać 3e6
        ap.add_argument(flag, type=typ, default=None,
                        help=f"(domyślnie: {f.default})")
    args = ap.parse_args()
    given = {f.name: getattr(args, f.name) for f in fields(Config)
             if getattr(args, f.name) is not None}

    if args.resume:
        p = Path(args.resume)
        ckpt = p / "latest.pt" if p.is_dir() else p
        cfg = Config(**torch.load(ckpt, map_location="cpu", weights_only=False)["config"])
        for k, v in given.items():            # przy wznowieniu można zmienić tylko te pola
            if k in ("total_steps", "eval_every", "save_every", "snapshot_every", "threads"):
                setattr(cfg, k, v)
            elif getattr(cfg, k) != v:
                print(f"uwaga: przy --resume ignoruję --{k.replace('_', '-')} "
                      f"(zostaje {getattr(cfg, k)})")
        Trainer(cfg, resume_ckpt=ckpt).train()
        return

    cfg = Config(**given)
    if cfg.team not in ("predator", "prey") or cfg.algo not in ("ippo", "mappo"):
        ap.error("--team predator|prey, --algo ippo|mappo")
    if cfg.opponent != "random":
        cfg.opponent = str(resolve_ckpt(cfg.opponent))
        # konfiguracja środowiska domyślnie taka jak u zamrożonego przeciwnika
        oc = Config(**torch.load(cfg.opponent, map_location="cpu", weights_only=False)["config"])
        for k in ENV_KEYS:
            if k not in given:
                setattr(cfg, k, getattr(oc, k))
    if not cfg.run_dir:
        cfg.run_dir = str(Path("runs") / default_run_name(cfg))
    if (Path(cfg.run_dir) / "latest.pt").exists():
        ap.error(f"{cfg.run_dir} już istnieje – użyj --resume {cfg.run_dir} albo innego --run-dir")
    print(f"trening -> {cfg.run_dir}")
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
