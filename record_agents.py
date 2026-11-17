"""Film z działania wytrenowanych (lub losowych) agentów.

    python record_agents.py --pred runs/predator_mappo_adv3_spd1.0_s0 --out media/mappo_vs_random.mp4
    python record_agents.py --pred runs/predator_mappo_adv3_spd1.0_s0 \
                            --prey runs/prey_mappo_adv3_spd1.0_vs-..._s0 --out media/rl_vs_rl.mp4

`--pred` / `--prey`: `random`, katalog runu albo plik checkpointu (np. ckpt_500k.pt – można
pokazać, jak zachowanie zmieniało się w trakcie treningu).
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image

from evaluate import env_config, load
from record_random import annotate
from tag_utils import make_env, random_policy, run_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="random")
    ap.add_argument("--prey", default="random")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--prey-speed", type=float)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="media/trained_agents.mp4")
    args = ap.parse_args()

    pred, prey = load(args.pred), load(args.prey)
    kw = env_config(pred, prey, {"prey_speed": args.prey_speed})
    env = make_env(render_mode="rgb_array", **kw)
    adv_fn = random_policy if pred is None else pred.as_policy_fn(args.greedy)
    prey_fn = random_policy if prey is None else prey.as_policy_fn(args.greedy)
    name = lambda pol: "losowe" if pol is None else f"{pol.cfg.algo.upper()} " \
                                                   f"({pol.env_steps // 1000}k kroków)"
    title = f"drapieżniki: {name(pred)} | ofiara: {name(prey)}"
    print(title)

    video = []
    for ep in range(args.episodes):
        frames, coll = [], []
        m = run_episode(env, adv_fn, prey_fn, args.seed + ep, frames, coll)
        print(f"epizod {ep}: kolizje {m['collisions']}, 1. złapanie w kroku {m['first_catch']}")
        for t, (f, c) in enumerate(zip(frames, coll)):
            video.append(annotate(annotate(f, title), "\n" + f"epizod {ep + 1}  krok {t}/"
                                                        f"{kw.get('max_cycles', 25)}  kolizje: {c}"))
        video.extend([video[-1]] * args.fps)
    env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, video, fps=args.fps, macro_block_size=1)
    gif = [Image.fromarray(f).resize((350, 350)) for f in video]
    gif[0].save(out.with_suffix(".gif"), save_all=True, append_images=gif[1:],
                duration=1000 // args.fps, loop=0)
    print(f"zapisano {out} i {out.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
