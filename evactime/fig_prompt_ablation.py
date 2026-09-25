"""Figure: what the task description adds to the exit claim (neutral-prompt ablation, C6).

Replaces the audit figure, whose content the audit table now carries. The same recordings were
annotated under the original task description, which stated that the participant leaves the room
and required an exit timestamp, and under a neutral one that did neither. Every point is a paired
rate on the same model, participant, condition and repeat.

  a  per cell, the rate under the original and the neutral description, with the paired
     participant-clustered interval of the neutral rate
  b  Gemini 3 Flash on the same route, video with audio against frames, under both descriptions
  c  one verbatim sentence written under the neutral description, looked up at build time

Run: py evactime/fig_prompt_ablation.py   (after C6_neutral_prompt.py)
"""
import json, pathlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
C = fs.C
CELLS = [("openai/gpt-5.2", "B"), ("anthropic/claude-opus-4.8", "B"),
         ("qwen/qwen3-vl-235b-a22b-instruct", "B"), ("gemini-3-flash-preview", "B"),
         ("gemini-3-flash-preview", "A")]
COND = {"A": "video and audio", "B": "frames"}
QUOTE = ("gemini-3-flash-preview", "A", "exit the room into a hallway")


def boot_rate(g, col, n=5000, seed=17):
    rng = np.random.default_rng(seed)
    groups = [x[col].values for _, x in g.groupby("pid")]
    vals = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
            for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def build():
    d = pd.read_csv(OUT / "C6_neutral_prompt.csv")
    hit = d[(d.model == QUOTE[0]) & (d.cond == QUOTE[1])
            & d.narrative_v3.str.contains(QUOTE[2], regex=False)]
    if hit.empty:
        raise SystemExit(f"quote not found verbatim under the neutral description: {QUOTE}")
    full = hit.narrative_v3.iloc[0]
    sent = next(s for s in full.replace("\n", " ").split(". ") if QUOTE[2] in s).strip()
    sent = sent if sent.endswith(".") else sent + "."

    fig = plt.figure(figsize=(7.4, 3.72))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.30], wspace=0.18,
                          left=0.215, right=0.975, top=0.88, bottom=0.31)

    # ---- (a) paired rates per cell
    ax = fig.add_subplot(gs[0])
    rows = []
    for m, c in CELLS:
        g = d[(d.model == m) & (d.cond == c)]
        if len(g):
            rows.append((m, c, g.narr_v2.mean() * 100, g.narr_v3.mean() * 100,
                         boot_rate(g, "narr_v3") * 100, len(g)))
    rows.sort(key=lambda r: r[3])
    for i, (m, c, v2, v3, ci, n) in enumerate(rows):
        ax.plot([v3, v2], [i, i], color=C["vlm_dim"], lw=3.0, solid_capstyle="round", zorder=2)
        ax.plot(ci, [i, i], color=C["vlm"], lw=1.0, zorder=3)
        ax.scatter(v2, i, s=46, color=C["vlm"], zorder=4, edgecolor="white", lw=0.8)
        ax.scatter(v3, i, s=46, facecolor="white", edgecolor=C["vlm"], lw=1.6, zorder=5)
        ax.text(max(v2, v3) + 3.5, i, f"{v2:.0f} → {v3:.0f}", va="center", fontsize=6.9,
                color=C["vlm"], fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{fs.MODEL_SHORT.get(m, m)}, {COND[c]}" + (", OpenRouter" if m.startswith("gemini") else "")
                        for m, c, *_ in rows],
                       fontsize=7.0)
    ax.set_xlim(0, 118)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100])
    fs.axis(ax, "Narratives asserting an exit (%)", None)
    fs.panel_title(ax, "a", "Paired exit-claim rates", pad=12)
    fs.despine(ax, left=False)
    fs.faint_grid(ax, axis="x")
    ax.scatter([], [], s=40, color=C["vlm"], label="exit stated and required")
    ax.scatter([], [], s=40, facecolor="white", edgecolor=C["vlm"], lw=1.6, label="neutral")
    ax.legend(loc="upper left", ncol=1, frameon=False, fontsize=6.8, handletextpad=0.3,
              borderaxespad=0.2)

    # ---- (b) Gemini 3 Flash, video against frames, both descriptions, both routes (same day)
    ax = fig.add_subplot(gs[1])
    res = json.load(open(OUT / "C6_neutral_prompt.json", encoding="utf-8"))
    g = d[d.model == "gemini-3-flash-preview"]
    series = [("OpenRouter", "stated", [g[g.cond == c].narr_v2.mean() * 100 for c in ("B", "A")]),
              ("OpenRouter", "neutral", [g[g.cond == c].narr_v3.mean() * 100 for c in ("B", "A")])]
    nat = res.get("native", {}).get("cells", {})
    if {"A", "B"} <= set(nat):
        series += [("native", "stated", [nat[c]["v2"] * 100 for c in ("B", "A")]),
                   ("native", "neutral", [nat[c]["v3"] * 100 for c in ("B", "A")])]
    x = [0, 1]
    for route, lab, y in series:
        style = dict(color=C["vlm"] if route == "native" else C["vlm_dim"],
                     ls="-" if route == "native" else (0, (3, 2)),
                     mfc="white" if lab == "neutral" else None)
        style = {k: v for k, v in style.items() if v is not None}
        ax.plot(x, y, marker="o", ms=5.5, lw=1.6, mew=1.4, **style, zorder=3)
        ax.text(1.08, y[1], f"{lab} {y[1]:.0f}", va="center", fontsize=6.3,
                color=style["color"], fontweight="bold")
    ax.plot([], [], color=C["vlm"], lw=1.6, label="native")
    ax.plot([], [], color=C["vlm_dim"], lw=1.6, ls=(0, (3, 2)), label="OpenRouter")
    ax.legend(loc="lower left", frameon=False, fontsize=6.4, handlelength=2.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Frames", "Video and\naudio"], fontsize=7.0)
    ax.set_xlim(-0.25, 1.75)
    ax.set_ylim(0, 105)
    fs.axis(ax, None, "Asserting an exit (%)")
    fs.panel_title(ax, "b", "Exit-claim rates by input and service route", pad=12)
    fs.despine(ax)
    fs.faint_grid(ax)

    # ---- (c) a sentence written under the neutral description
    qa = fig.add_axes([0.215, 0.010, 0.76, 0.085])
    qa.axis("off")
    fig.text(0.215, 0.112, "(c) Neutral-description narrative", fontsize=9.2,
             fontweight="regular", color=C["ink"], ha="left", va="bottom")
    qa.add_patch(Rectangle((0, 0), 1, 1, fc="white", ec=C["grid"], lw=0.6,
                           transform=qa.transAxes))
    qa.add_patch(Rectangle((0, 0), 0.008, 1, fc=C["vlm"], ec="none", transform=qa.transAxes))
    qa.text(0.025, 0.68, f"“{sent}”", transform=qa.transAxes, fontsize=7.2,
            style="italic", va="center", color=C["ink"])
    qa.text(0.025, 0.15, "Gemini 3 Flash, video and audio, neutral task description. No hallway "
            "appears in any recording.", transform=qa.transAxes, fontsize=6.6,
            fontweight="bold", color=C["vlm"], va="center")

    fs.save(fig, "q1_fig4_prompt")
    print("  wrote q1_fig4_prompt")


if __name__ == "__main__":
    fs.setup()
    build()
