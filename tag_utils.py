"""Wspólne narzędzia dla środowiska MPE2 simple_tag: tworzenie środowiska,
polityki bazowe (losowa, heurystyczna) oraz uruchamianie epizodu z metrykami."""

from __future__ import annotations

import numpy as np
from mpe2 import simple_tag_v3

# Akcje dyskretne: 0 = brak, 1 = lewo (-x), 2 = prawo (+x), 3 = dół (-y), 4 = góra (+y)
NOOP, LEFT, RIGHT, DOWN, UP = range(5)


def make_env(render_mode=None, prey_speed=1.0, **kwargs):
    """Tworzy środowisko simple_tag w wersji Parallel API.

    `prey_speed` skaluje max_speed (1.3) i accel (4.0) ofiar. Robimy to ręcznie, bo w mpe2 1.1.1
    tryb `curriculum=True` jest zepsuty: BaseScenario definiuje property `curriculum_stage`
    z pustym setterem, więc set_curriculum_stage() nic nie zmienia i ofiara ma zawsze 50% prędkości.
    """
    env = simple_tag_v3.parallel_env(render_mode=render_mode, **kwargs)
    for a in env.unwrapped.world.agents:
        if not a.adversary:
            a.max_speed *= prey_speed
            a.accel *= prey_speed
    return env


def _move_towards(delta: np.ndarray) -> int:
    """Dyskretna akcja przybliżająca do wektora `delta` (wzdłuż dominującej osi)."""
    if abs(delta[0]) >= abs(delta[1]):
        return RIGHT if delta[0] > 0 else LEFT
    return UP if delta[1] > 0 else DOWN


def random_policy(env, agent, rng):
    return int(rng.integers(env.action_space(agent).n))


def greedy_chase_policy(env, agent, rng):
    """Heurystyka drapieżnika: biegnij prosto do najbliższej ofiary."""
    world = env.unwrapped.world
    me = next(a for a in world.agents if a.name == agent)
    prey = [a for a in world.agents if not a.adversary]
    target = min(prey, key=lambda p: np.linalg.norm(p.state.p_pos - me.state.p_pos))
    return _move_towards(target.state.p_pos - me.state.p_pos)


def flee_policy(env, agent, rng):
    """Heurystyka ofiary: uciekaj od najbliższego drapieżnika, ale nie wychodź poza obszar."""
    world = env.unwrapped.world
    me = next(a for a in world.agents if a.name == agent)
    advs = [a for a in world.agents if a.adversary]
    nearest = min(advs, key=lambda a: np.linalg.norm(a.state.p_pos - me.state.p_pos))
    away = me.state.p_pos - nearest.state.p_pos
    # przy krawędzi dodaj składową ciągnącą do środka (kara za wyjście poza |x| > 0.9)
    away = away - np.where(np.abs(me.state.p_pos) > 0.8, me.state.p_pos * 2.0, 0.0)
    return _move_towards(away)


POLICIES = {
    "random": random_policy,
    "greedy": greedy_chase_policy,
    "flee": flee_policy,
}


def run_episode(env, adv_policy, prey_policy, seed, frames=None, coll_trace=None):
    """Rozgrywa jeden epizod i zwraca słownik metryk.

    Jeśli podano listę `frames`, dopisywane są do niej klatki renderu; jeśli `coll_trace`,
    to skumulowana liczba kolizji po każdej klatce (do podpisów w filmie).

    Metryki:
      adv_return      – zwrot drużyny drapieżników (nagroda wspólna, liczona dla jednego adwersarza)
      prey_return     – suma zwrotów ofiar
      collisions      – liczba par (drapieżnik, ofiara) w kolizji, zsumowana po krokach
      caught_steps    – liczba kroków, w których co najmniej jedna ofiara była dotykana
      first_catch     – krok pierwszego złapania (NaN, gdy brak)
      caught          – 1, jeśli w epizodzie doszło do złapania
      boundary_pen    – kara ofiar za wychodzenie poza obszar (dodatnia wartość)
      mean_min_dist   – średni dystans ofiara–najbliższy drapieżnik
    """
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    scen, world = env.unwrapped.scenario, env.unwrapped.world
    advs = [a for a in world.agents if a.adversary]
    prey = [a for a in world.agents if not a.adversary]
    adv_name = advs[0].name

    m = dict(adv_return=0.0, prey_return=0.0, collisions=0, caught_steps=0,
             first_catch=np.nan, boundary_pen=0.0, dist_sum=0.0, steps=0)
    if frames is not None:
        frames.append(env.render())
    if coll_trace is not None:
        coll_trace.append(0)

    while env.agents:
        actions = {
            a: (adv_policy if a.startswith("adversary") else prey_policy)(env, a, rng)
            for a in env.agents
        }
        _, rew, term, trunc, _ = env.step(actions)
        m["steps"] += 1
        m["adv_return"] += rew[adv_name]
        m["prey_return"] += sum(rew[p.name] for p in prey)

        n_coll = sum(scen.is_collision(p, a) for p in prey for a in advs)
        m["collisions"] += n_coll
        if n_coll:
            m["caught_steps"] += 1
            if np.isnan(m["first_catch"]):
                m["first_catch"] = m["steps"]
        # kara brzegowa = nagroda ofiary bez części za kolizje
        for p in prey:
            coll_pen = 10 * sum(scen.is_collision(a, p) for a in advs)
            m["boundary_pen"] += -(rew[p.name] + coll_pen)
            m["dist_sum"] += min(np.linalg.norm(p.state.p_pos - a.state.p_pos) for a in advs)
        if frames is not None:
            frames.append(env.render())
        if coll_trace is not None:
            coll_trace.append(m["collisions"])

    m["caught"] = int(m["collisions"] > 0)
    m["mean_min_dist"] = m.pop("dist_sum") / (m["steps"] * len(prey))
    return m
