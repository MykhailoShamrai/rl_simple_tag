"""Uruchamia cały zestaw eksperymentów (np. na noc), równolegle na kilku procesach.

Faza 1 – drapieżniki (IPPO i MAPPO) przeciw losowej ofierze:
          konfiguracja domyślna (3 drapieżniki, prędkość ofiary 1.0) + osie trudności:
          2 / 4 drapieżniki, prędkość ofiary 0.75 / 1.25; po 3 seedy.
Faza 2 – zamrażamy najlepsze drapieżniki z konfiguracji domyślnej (najwyższa liczba kolizji
          w ostatniej ewaluacji) i trenujemy przeciw nim ofiarę (IPPO i MAPPO, 3 seedy).
Faza 3 – ewaluacja wszystkich runów (evaluate.py --suite) i wykresy (plot_results.py).

Skrypt jest idempotentny: ukończone runy (final.pt) pomija, przerwane wznawia z latest.pt.
Można go więc po prostu uruchomić ponownie po przerwaniu (Ctrl+C zapisuje checkpointy).

    python run_experiments.py --dry-run          # pokaż, co zostanie uruchomione
    python run_experiments.py                    # wszystko (fazy 1-3)
    python run_experiments.py --phase 2          # tylko ofiara (wymaga ukończonej fazy 1)
    python run_experiments.py --quick            # szybki test całego potoku (runs_quick/)
"""

from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

PY = sys.executable
SEEDS = (0, 1, 2)
ALGOS = ("ippo", "mappo")
DEFAULT = dict(num_adversaries=3, prey_speed=1.0)
VARIANTS = [DEFAULT,
            dict(num_adversaries=2, prey_speed=1.0), dict(num_adversaries=4, prey_speed=1.0),
            dict(num_adversaries=3, prey_speed=0.75), dict(num_adversaries=3, prey_speed=1.25)]

children: list[subprocess.Popen] = []


def run_name(team, algo, env, seed, opp=None):
    base = f"{team}_{algo}_adv{env['num_adversaries']}_spd{env['prey_speed']}"
    return f"{base}_vs-{opp}_s{seed}" if opp else f"{base}_s{seed}"


def finished(run_dir, total):
    p = run_dir / "final.pt"
    return p.exists() and torch.load(p, map_location="cpu", weights_only=False)[
        "env_steps"] >= total


def command(run_dir, total, extra):
    if (run_dir / "latest.pt").exists():
        return [PY, "ppo.py", "--resume", str(run_dir), "--total-steps", str(total)]
    return [PY, "ppo.py", "--run-dir", str(run_dir), "--total-steps", str(total)] + extra


def run_pool(jobs, n_parallel, dry):
    """jobs: lista (run_dir, cmd). Uruchamia po n_parallel naraz, logi -> run_dir/stdout.log."""
    pending = list(jobs)
    if dry:
        for _, cmd in pending:
            print("  " + " ".join(cmd[1:]))
        return
    running = []
    while pending or running:
        while pending and len(running) < n_parallel:
            run_dir, cmd = pending.pop(0)
            run_dir.mkdir(parents=True, exist_ok=True)
            log = open(run_dir / "stdout.log", "a")
            # osobna sesja: Ctrl+C w terminalu trafia tylko do tego skryptu, który przekazuje
            # treningom pojedynczy SIGINT (zapis checkpointu), a nie dwa (natychmiastowe wyjście)
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            children.append(p)
            running.append((run_dir, p, time.time()))
            print(f"{time.strftime('%H:%M:%S')} start  {run_dir.name}", flush=True)
        time.sleep(5)
        for item in list(running):
            run_dir, p, t0 = item
            if p.poll() is not None:
                running.remove(item)
                status = "OK" if p.returncode == 0 else f"BŁĄD (kod {p.returncode}), " \
                                                        f"zob. {run_dir / 'stdout.log'}"
                print(f"{time.strftime('%H:%M:%S')} koniec {run_dir.name}  "
                      f"{(time.time() - t0) / 60:.0f} min  {status}", flush=True)


def last_eval(run_dir, key):
    with open(run_dir / "eval.csv") as f:
        return float(list(csv.DictReader(f))[-1][key])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all", choices=["1", "2", "3", "all"])
    ap.add_argument("--parallel", type=int, default=10, help="liczba równoległych treningów")
    ap.add_argument("--pred-steps", type=float, default=3e6)
    ap.add_argument("--prey-steps", type=float, default=5e6)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quick", action="store_true", help="mini-test potoku (runs_quick/)")
    args = ap.parse_args()

    extra = []
    if args.quick:
        args.runs_dir, args.pred_steps, args.prey_steps = "runs_quick", 20_000, 20_000
        extra = ["--eval-every", "5", "--eval-episodes", "10", "--snapshot-every", "10000"]
    root = Path(args.runs_dir)
    pred_total, prey_total = int(args.pred_steps), int(args.prey_steps)

    def on_signal(signum, frame):
        print("\nprzerwanie – czekam, aż treningi zapiszą checkpointy...", flush=True)
        for p in children:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in children:
            p.wait()
        sys.exit(1)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # ------------------------------------------------------------------ faza 1
    if args.phase in ("1", "all"):
        jobs = []
        for env in VARIANTS:
            for algo in ALGOS:
                for seed in SEEDS:
                    d = root / run_name("predator", algo, env, seed)
                    if finished(d, pred_total):
                        continue
                    cmd = command(d, pred_total, ["--team", "predator", "--algo", algo,
                                                  "--seed", str(seed),
                                                  "--num-adversaries", str(env["num_adversaries"]),
                                                  "--prey-speed", str(env["prey_speed"])] + extra)
                    jobs.append((d, cmd))
        print(f"== faza 1: drapieżniki vs losowa ofiara – {len(jobs)} treningów do wykonania")
        run_pool(jobs, args.parallel, args.dry_run)

    # ------------------------------------------------------------------ faza 2
    if args.phase in ("2", "all"):
        cands = [root / run_name("predator", a, DEFAULT, s) for a in ALGOS for s in SEEDS]
        cands = [d for d in cands if (d / "final.pt").exists()]
        if not cands:
            if args.dry_run:
                print("== faza 2: (po fazie 1) ofiara IPPO/MAPPO × 3 seedy vs najlepsze drapieżniki")
                return
            sys.exit("faza 2: brak ukończonych drapieżników w konfiguracji domyślnej")
        best = max(cands, key=lambda d: last_eval(d, "collisions"))
        note = root / "frozen_predator.txt"
        note.write_text(
            "Zamrożone drapieżniki dla fazy 2 (najwięcej kolizji/epizod w ostatniej ewaluacji):\n"
            + "\n".join(f"{d.name}: {last_eval(d, 'collisions'):.2f}" for d in cands)
            + f"\n\nwybrano: {best / 'final.pt'}\n")
        print(f"== faza 2: zamrożone drapieżniki = {best.name} "
              f"({last_eval(best, 'collisions'):.2f} kolizji/ep)")
        jobs = []
        for algo in ALGOS:
            for seed in SEEDS:
                d = root / run_name("prey", algo, DEFAULT, seed, opp=best.name)
                if finished(d, prey_total):
                    continue
                cmd = command(d, prey_total, ["--team", "prey", "--algo", algo, "--seed",
                                              str(seed), "--opponent",
                                              str(best / "final.pt")] + extra)
                jobs.append((d, cmd))
        print(f"   {len(jobs)} treningów ofiary do wykonania")
        run_pool(jobs, args.parallel, args.dry_run)

    # ------------------------------------------------------------------ faza 3
    if args.phase in ("3", "all") and not args.dry_run:
        print("== faza 3: ewaluacja i wykresy", flush=True)
        eps = ["--episodes", "50"] if args.quick else []
        out = "results_quick" if args.quick else "results"
        subprocess.run([PY, "evaluate.py", "--suite", "--runs-dir", str(root),
                        "--out", f"{out}/trained_eval.csv"] + eps, check=True)
        subprocess.run([PY, "plot_results.py", "--runs-dir", str(root), "--out-dir", out],
                       check=True)
    print("gotowe.")


if __name__ == "__main__":
    main()
