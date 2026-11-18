"""Krzywe uczenia i porównania z wyników treningu (runs/*/log.csv, eval.csv) i ewaluacji
(results/trained_eval.csv z `evaluate.py --suite`).

    python plot_results.py                       # -> results/*.png, results/summary.md

Wykresy:
  learning_curves_default.png    IPPO vs MAPPO (3 drapieżniki, prędkość 1.0): kolizje, złapania,
                                 zwrot w treningu, entropia; średnia ± odch. std. po seedach
  learning_curves_variants.png   to samo (kolizje / epizod) dla każdej konfiguracji środowiska
  learning_curves_prey.png       trening ofiary przeciw zamrożonym drapieżnikom
  compare_variants.png           końcowa ewaluacja (500 epizodów): losowe vs IPPO vs MAPPO
  speed_generalization.png       drapieżniki z treningu przy prędkości 1.0 testowane przy innych
Uwaga: krzywe z eval.csv to 100 epizodów co 25 iteracji; tabela końcowa – 500 epizodów.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"random": "#8a8984", "ippo": "#2a78d6", "mappo": "#eb6834"}
LABELS = {"random": "losowe", "ippo": "IPPO", "mappo": "MAPPO"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "font.size": 9.5, "legend.frameon": False, "lines.linewidth": 2,
})


def load_runs(runs_dir):
    runs = []
    for d in sorted(Path(runs_dir).glob("*")):
        if not (d / "config.json").exists() or not (d / "eval.csv").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        log = pd.read_csv(d / "log.csv") if (d / "log.csv").exists() else None
        runs.append(dict(name=d.name, cfg=cfg, eval=pd.read_csv(d / "eval.csv"), log=log,
                         done=(d / "final.pt").exists()))
    return runs


def select(runs, team, n_adv=3, speed=1.0, algo=None):
    return [r for r in runs if r["cfg"]["team"] == team and r["cfg"]["num_adversaries"] == n_adv
            and r["cfg"]["prey_speed"] == speed and (algo is None or r["cfg"]["algo"] == algo)]


def band(ax, runs, source, key, algo, scale=1.0, smooth=1):
    """Średnia ± odch. std. po seedach, na wspólnej siatce env_steps."""
    series = []
    for r in runs:
        df = r[source]
        if df is None or key not in df:
            continue
        s = df.set_index("env_steps")[key] * scale
        if smooth > 1:
            s = s.rolling(smooth, min_periods=1).mean()
        series.append(s)
    if not series:
        return
    m = pd.concat(series, axis=1).dropna()
    x = m.index.values / 1e6
    mu, sd = m.mean(axis=1).values, m.std(axis=1).fillna(0).values
    ax.fill_between(x, mu - sd, mu + sd, color=COLORS[algo], alpha=0.18, lw=0)
    ax.plot(x, mu, color=COLORS[algo], label=f"{LABELS[algo]} (n={len(series)})")


def hline(ax, y, text):
    if y is None or np.isnan(y):
        return
    ax.axhline(y, color=COLORS["random"], ls="--", lw=1.5)
    ax.annotate(text, (1, y), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                textcoords="offset points", ha="right", va="bottom", color=MUTED, fontsize=8.5)


def baseline(ev, n_adv, speed, key):
    if ev is None:
        return None
    b = ev[(ev.kind == "baseline") & (ev.train_num_adversaries == n_adv) &
           (ev.train_prey_speed == speed)]
    return float(b[key].iloc[0]) if len(b) else None


# ----------------------------------------------------------------------------- wykresy

def plot_default(runs, ev, out):
    sel = select(runs, "predator")
    if not sel:
        return
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    panels = [("eval", "collisions", 1, 1, "Kolizje / epizod (ewaluacja)", "collisions"),
              ("eval", "caught", 100, 1, "% epizodów ze złapaniem (ewaluacja)", "caught"),
              ("log", "ep_return", 1, 20, "Zwrot drużyny w treningu (wygładzony)", None),
              ("log", "entropy", 1, 20, "Entropia polityki", None)]
    for ax, (src, key, scale, sm, title, bkey) in zip(axs.flat, panels):
        for algo in ("ippo", "mappo"):
            band(ax, select(runs, "predator", algo=algo), src, key, algo, scale, sm)
        if bkey:
            b = baseline(ev, 3, 1.0, bkey)
            hline(ax, None if b is None else b * scale, "losowe drapieżniki")
        ax.set_title(title, loc="left")
        ax.set_xlabel("kroki środowiska [mln]")
    axs[0, 0].legend(loc="lower right", bbox_to_anchor=(1, 0.1))
    fig.suptitle("Drapieżniki vs losowa ofiara – 3 drapieżniki, prędkość ofiary 1.0×",
                 color=INK, fontweight="bold", x=0.01, ha="left")
    fig.savefig(out / "learning_curves_default.png", dpi=150)
    plt.close(fig)


VARIANTS = [(2, 1.0), (3, 1.0), (4, 1.0), (3, 0.75), (3, 1.25)]


def plot_variants(runs, ev, out):
    present = [v for v in VARIANTS if select(runs, "predator", *v)]
    if not present:
        return
    fig, axs = plt.subplots(1, len(present), figsize=(3.3 * len(present), 3.4),
                            sharey=True, constrained_layout=True, squeeze=False)
    for ax, (n, sp) in zip(axs[0], present):
        for algo in ("ippo", "mappo"):
            band(ax, select(runs, "predator", n, sp, algo), "eval", "collisions", algo)
        hline(ax, baseline(ev, n, sp, "collisions"), "losowe")
        ax.set_title(f"{n} drapieżniki, ofiara {sp}×", loc="left")
        ax.set_xlabel("kroki [mln]")
    axs[0, 0].set_ylabel("kolizje / epizod")
    axs[0, 0].legend(loc="lower right", bbox_to_anchor=(1, 0.1))
    fig.savefig(out / "learning_curves_variants.png", dpi=150)
    plt.close(fig)


def plot_prey(runs, ev, out):
    sel = [r for r in runs if r["cfg"]["team"] == "prey"]
    if not sel:
        return
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    ref = None
    if ev is not None:
        r = ev[ev.kind == "prey_ref_random"]
        ref = r if len(r) else None
    panels = [("eval", "collisions", 1, "Kolizje / epizod (mniej = lepiej)"),
              ("eval", "out_steps", 1, "Kroki ofiary poza planszą"),
              ("eval", "prey_return", 1, "Zwrot ofiary (ewaluacja)")]
    for ax, (src, key, scale, title) in zip(axs, panels):
        for algo in ("ippo", "mappo"):
            band(ax, [r for r in sel if r["cfg"]["algo"] == algo], src, key, algo, scale)
        if ref is not None:
            hline(ax, float(ref[key].mean()), "losowa ofiara")
        ax.set_title(title, loc="left")
        ax.set_xlabel("kroki środowiska [mln]")
    axs[0].legend(loc="upper right")
    opp = Path(sel[0]["cfg"]["opponent"]).parent.name
    fig.suptitle(f"Ofiara vs zamrożone drapieżniki ({opp})", color=INK, fontweight="bold",
                 x=0.01, ha="left")
    fig.savefig(out / "learning_curves_prey.png", dpi=150)
    plt.close(fig)


def per_algo(df, key):
    """Średnia i odch. std. po seedach (każdy seed = średnia z 500 epizodów)."""
    g = df.groupby("algo")[key]
    return g.mean(), g.std().fillna(0)


def plot_compare(ev, out):
    own = ev[(ev.kind.isin(["predator", "baseline"])) &
             (ev.prey_speed == ev.train_prey_speed)]
    present = [v for v in VARIANTS if len(own[(own.train_num_adversaries == v[0]) &
                                              (own.train_prey_speed == v[1]) &
                                              (own.kind == "predator")])]
    if not present:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    w = 0.26
    for ax, key, scale, title in [(axs[0], "collisions", 1, "Kolizje / epizod"),
                                  (axs[1], "caught", 100, "% epizodów ze złapaniem")]:
        for i, (n, sp) in enumerate(present):
            sub = own[(own.train_num_adversaries == n) & (own.train_prey_speed == sp)]
            mu, sd = per_algo(sub, key)
            for j, algo in enumerate(("random", "ippo", "mappo")):
                if algo not in mu:
                    continue
                ax.bar(i + (j - 1) * w, mu[algo] * scale, w - 0.03, yerr=sd[algo] * scale,
                       color=COLORS[algo], label=LABELS[algo] if i == 0 else None,
                       error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
        ax.set_xticks(range(len(present)), [f"{n} drap.\nofiara {sp}×" for n, sp in present])
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axs[0].legend(loc="upper left")
    fig.suptitle("Końcowa ewaluacja drapieżników vs losowa ofiara (500 epizodów, "
                 "słupek błędu = odch. std. po seedach)", color=INK, fontweight="bold",
                 x=0.01, ha="left")
    fig.savefig(out / "compare_variants.png", dpi=150)
    plt.close(fig)


def plot_speed(ev, out):
    pr = ev[(ev.kind == "predator") & (ev.train_num_adversaries == 3)]
    base = ev[(ev.kind == "baseline") & (ev.train_num_adversaries == 3)]
    if pr.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)
    b = base.groupby("prey_speed")["collisions"].mean()
    ax.plot(b.index, b.values, color=COLORS["random"], ls="--", marker="o", ms=6,
            label="losowe drapieżniki")
    for algo in ("ippo", "mappo"):
        a = pr[pr.algo == algo]
        fixed = a[a.train_prey_speed == 1.0].groupby("prey_speed")["collisions"]
        if len(fixed):
            m = fixed.mean()
            ax.errorbar(m.index, m.values, yerr=fixed.std().fillna(0), color=COLORS[algo],
                        marker="o", ms=6, capsize=3, label=f"{LABELS[algo]} trenowane przy 1.0×")
        matched = a[a.train_prey_speed == a.prey_speed].groupby("prey_speed")["collisions"]
        if matched.ngroups > 1:
            m = matched.mean()
            ax.plot(m.index, m.values, color=COLORS[algo], ls=":", marker="s", ms=6,
                    label=f"{LABELS[algo]} trenowane przy tej prędkości")
    ax.set_xlabel("prędkość ofiary w teście (× domyślnej)")
    ax.set_ylabel("kolizje / epizod")
    ax.set_ylim(bottom=0)
    ax.set_title("Uogólnianie na prędkość ofiary (3 drapieżniki)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    fig.savefig(out / "speed_generalization.png", dpi=150)
    plt.close(fig)


def write_summary(ev, out):
    cols = ["collisions", "caught", "first_catch", "mean_min_dist", "out_steps"]
    lines = ["# Wyniki ewaluacji (500 epizodów, seedy 10000-10499)", "",
             "Wartości: średnia po seedach (± odch. std. po seedach); każdy seed = średnia z 500 "
             "epizodów.", ""]

    def table(df, group_cols, title):
        nonlocal lines
        if df.empty:
            return
        g = df.groupby(group_cols)
        mu, sd, n = g[cols].mean(), g[cols].std().fillna(0), g.size()
        lines += [f"## {title}", "",
                  "| " + " | ".join(group_cols) + " | seedy | kolizje/ep | % złapań | krok 1. złap. |"
                  " dystans | kroki poza planszą |",
                  "|" + "---|" * (len(group_cols) + 6)]
        for idx in mu.index:
            key = idx if isinstance(idx, tuple) else (idx,)
            m, s = mu.loc[idx], sd.loc[idx]
            lines.append("| " + " | ".join(str(k) for k in key) + f" | {n.loc[idx]} | "
                         f"{m.collisions:.2f} ± {s.collisions:.2f} | "
                         f"{100 * m.caught:.1f} ± {100 * s.caught:.1f} | "
                         f"{m.first_catch:.1f} | {m.mean_min_dist:.3f} | {m.out_steps:.1f} |")
        lines.append("")

    own = ev[ev.kind.isin(["predator", "baseline"]) & (ev.prey_speed == ev.train_prey_speed)]
    trained = set(zip(own[own.kind == "predator"].train_num_adversaries,
                      own[own.kind == "predator"].train_prey_speed))
    own = own[[c in trained for c in zip(own.train_num_adversaries, own.train_prey_speed)]]
    table(own, ["train_num_adversaries", "train_prey_speed", "algo"],
          "Drapieżniki vs losowa ofiara (konfiguracja treningowa)")
    gen = ev[(ev.kind == "predator") & (ev.train_num_adversaries == 3) &
             (ev.train_prey_speed == 1.0)]
    table(gen, ["prey_speed", "algo"], "Drapieżniki trenowane przy 1.0× testowane przy innych "
                                       "prędkościach ofiary")
    # wiersz odniesienia (losowa ofiara vs te same drapieżniki) powtarza się dla każdego runu ofiary
    ref = ev[ev.kind == "prey_ref_random"].drop_duplicates("pred").assign(algo="random")
    prey = pd.concat([ev[ev.kind == "prey"], ref])
    table(prey, ["algo"], "Ofiara vs zamrożone drapieżniki")
    (out / "summary.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    runs = load_runs(args.runs_dir)
    ev_path = out / "trained_eval.csv"
    ev = pd.read_csv(ev_path) if ev_path.exists() else None
    print(f"{len(runs)} runów, ewaluacja: {'tak' if ev is not None else 'brak (tylko krzywe)'}")
    plot_default(runs, ev, out)
    plot_variants(runs, ev, out)
    plot_prey(runs, ev, out)
    if ev is not None:
        plot_compare(ev, out)
        plot_speed(ev, out)
        write_summary(ev, out)
    print(f"zapisano wykresy w {out}/")


if __name__ == "__main__":
    main()
