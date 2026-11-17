"""Pokaz na żywo: dwie plansze obok siebie, ten sam seed (te same pozycje startowe) –
domyślnie losowe drapieżniki (lewa) vs wytrenowane (prawa), ofiara losowa.

    python demo.py --right-pred runs/predator_mappo_adv3_spd1.0_s0
    python demo.py --left-pred runs/predator_ippo_adv3_spd1.0_s0 \
                   --right-pred runs/predator_mappo_adv3_spd1.0_s0          # IPPO vs MAPPO
    python demo.py --left-pred runs/predator_mappo_adv3_spd1.0_s0/ckpt_500k.pt \
                   --right-pred runs/predator_mappo_adv3_spd1.0_s0          # w trakcie vs koniec
    python demo.py --left-pred runs/P --right-pred runs/P \
                   --right-prey runs/prey_mappo_..._s0                      # losowa vs wyuczona ofiara
    python demo.py --left-pred random                                       # jedna plansza

Specyfikacja polityki: `random`, katalog runu (final.pt / latest.pt) albo plik checkpointu.
Klawisze: spacja – pauza, → / n – następny epizod, ↑/↓ – szybciej/wolniej, Esc / q – wyjście.
"""

from __future__ import annotations

import argparse

import numpy as np
import pygame

from evaluate import env_config, load
from tag_utils import make_env, random_policy, run_episode

PANEL = 520           # rozmiar planszy w pikselach
HEADER = 64           # pasek z opisem nad planszą
FOOTER = 40           # pasek z podsumowaniem pod planszami


def describe(pred, prey):
    def name(pol):
        if pol is None:
            return "losowe"
        return f"{pol.cfg.algo.upper()} ({pol.env_steps / 1e6:.1f} mln kroków)"
    return f"drapieżniki: {name(pred)}", f"ofiara: {name(prey)}"


class Side:
    def __init__(self, pred_spec, prey_spec, greedy):
        self.pred, self.prey = load(pred_spec), load(prey_spec)
        self.adv_fn = random_policy if self.pred is None else self.pred.as_policy_fn(greedy)
        self.prey_fn = random_policy if self.prey is None else self.prey.as_policy_fn(greedy)
        self.lines = describe(self.pred, self.prey)
        self.total_coll, self.caught_eps, self.eps = 0, 0, 0

    def make(self, kw):
        self.kw = kw
        self.env = make_env(render_mode="rgb_array", **self.kw)

    def play(self, seed):
        frames, coll = [], []
        m = run_episode(self.env, self.adv_fn, self.prey_fn, seed, frames, coll)
        self.total_coll += m["collisions"]
        self.caught_eps += m["caught"]
        self.eps += 1
        return frames, coll


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left-pred", default="random")
    ap.add_argument("--left-prey", default="random")
    ap.add_argument("--right-pred", help="brak = tylko jedna plansza")
    ap.add_argument("--right-prey", default="random")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--prey-speed", type=float)
    ap.add_argument("--max-cycles", type=int, help="dłuższe epizody do pokazu (trening: 25)")
    ap.add_argument("--greedy", action="store_true", help="argmax zamiast próbkowania")
    args = ap.parse_args()

    overrides = {"prey_speed": args.prey_speed, "max_cycles": args.max_cycles}
    sides = [Side(args.left_pred, args.left_prey, args.greedy)]
    if args.right_pred:
        sides.append(Side(args.right_pred, args.right_prey, args.greedy))
    # wspólna konfiguracja środowiska: z checkpointów (losowe polityki jej nie mają) + nadpisania
    configs = {tuple(sorted(env_config(s.pred, s.prey, {}).items())) for s in sides} - {()}
    if len(configs) > 1:
        raise SystemExit(f"modele trenowane w różnych konfiguracjach środowiska: {configs}")
    kw = dict(next(iter(configs), ()))
    kw.update({k: v for k, v in overrides.items() if v is not None})
    for s in sides:
        s.make(kw)

    pygame.init()
    screen = pygame.display.set_mode((PANEL * len(sides), HEADER + PANEL + FOOTER))
    pygame.display.set_caption("MPE2 simple_tag – pokaz")
    font = pygame.font.SysFont("dejavusans", 17)
    big = pygame.font.SysFont("dejavusans", 20, bold=True)
    clock = pygame.time.Clock()
    ink, muted, hit = (11, 11, 11), (82, 81, 78), (200, 40, 40)

    fps, seed, paused = args.fps, args.seed, False
    max_steps = sides[0].kw.get("max_cycles", 25)
    while True:
        episodes = [s.play(seed) for s in sides]
        n_frames = len(episodes[0][0])
        t, skip = 0, False
        while t < n_frames + fps and not skip:          # + krótka pauza po epizodzie
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and
                                             e.key in (pygame.K_ESCAPE, pygame.K_q)):
                    pygame.quit()
                    return
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_SPACE:
                        paused = not paused
                    elif e.key in (pygame.K_RIGHT, pygame.K_n):
                        skip = True
                    elif e.key == pygame.K_UP:
                        fps = min(fps + 2, 60)
                    elif e.key == pygame.K_DOWN:
                        fps = max(fps - 2, 2)
            k = min(t, n_frames - 1)
            screen.fill((252, 252, 251))
            for i, (side, (frames, coll)) in enumerate(zip(sides, episodes)):
                x0 = i * PANEL
                img = np.ascontiguousarray(frames[k].transpose(1, 0, 2))
                surf = pygame.transform.smoothscale(pygame.surfarray.make_surface(img),
                                                    (PANEL, PANEL))
                screen.blit(surf, (x0, HEADER))
                screen.blit(big.render(side.lines[0], True, ink), (x0 + 12, 8))
                screen.blit(font.render(side.lines[1], True, muted), (x0 + 12, 36))
                c = coll[k]
                screen.blit(big.render(f"kolizje: {c}", True, hit if c else ink),
                            (x0 + PANEL - 140, HEADER + 8))
                rate = 100 * side.caught_eps / side.eps
                screen.blit(font.render(
                    f"łącznie: {side.total_coll / side.eps:.2f} kolizji/ep, złapania w "
                    f"{rate:.0f}% z {side.eps} ep.", True, muted), (x0 + 12, HEADER + PANEL + 10))
                if i:
                    pygame.draw.line(screen, (228, 227, 223), (x0, 0), (x0, screen.get_height()), 2)
            status = f"epizod {seed - args.seed + 1}  krok {min(k, max_steps)}/{max_steps}"
            status += "  [PAUZA]" if paused else ""
            screen.blit(font.render(status, True, muted), (12, HEADER + 8))
            pygame.display.flip()
            clock.tick(fps)
            if not paused:
                t += 1
        seed += 1


if __name__ == "__main__":
    main()
