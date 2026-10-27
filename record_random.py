"""Nagrywa film z działania losowego agenta (wszyscy agenci wybierają akcje losowo).

Użycie:
    python record_random.py --episodes 4 --max-cycles 50 --out media/random_agent.mp4
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tag_utils import POLICIES, make_env, run_episode


def annotate(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 10), text, fill=(0, 0, 0), font=font)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--max-cycles", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="media/random_agent.mp4")
    args = ap.parse_args()

    env = make_env(render_mode="rgb_array", max_cycles=args.max_cycles)
    video = []
    for ep in range(args.episodes):
        frames = []
        m = run_episode(env, POLICIES["random"], POLICIES["random"], args.seed + ep, frames)
        print(f"epizod {ep}: {m}")
        for t, f in enumerate(frames):
            video.append(annotate(
                f, f"Losowy agent | epizod {ep + 1}  krok {t}/{args.max_cycles}  "
                   f"kolizje w epizodzie: {m['collisions']}"))
        video.extend([video[-1]] * args.fps)  # krótka pauza między epizodami
    env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, video, fps=args.fps, macro_block_size=1)
    gif = [Image.fromarray(f).resize((350, 350)) for f in video]
    gif[0].save(out.with_suffix(".gif"), save_all=True, append_images=gif[1:],
                duration=1000 // args.fps, loop=0)
    print(f"zapisano {out} i {out.with_suffix('.gif')} ({len(video)} klatek)")


if __name__ == "__main__":
    main()
