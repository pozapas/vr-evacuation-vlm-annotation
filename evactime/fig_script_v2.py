"""Figure: exit claims, with the recording itself on the page.

The paper's most striking finding is about something visible on screen, so the figure shows it.
Five frames from one participant's recording (P1) run across the top: the alarm sounding, the walk
to the door, the hand on the handle, the cut, and the same room 38 s later. Under them are the
words three models used to describe THIS recording, quoted verbatim from the corpus. A reader who
sees the room reload unchanged does not need to take the 73.8% on trust.

Below, the per-model correction and input-condition response cards. Each card gives the rate, a
common-scale microbar, and the change from frames-only input, so the reader can inspect the
condition pattern without decoding a table.

Every quote is looked up by exact string in C4_script_completion.csv at build time and the build
fails if it is not found, so the figure cannot carry a paraphrase.
"""
import json, pathlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.image import imread

import figstyle as fs
from palette import ramp

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
C = fs.C

FRAMES = [("a_alarm", "alarm sounding"), ("b_door", "walking to the door"),
          ("c_handle", "hand on the handle"), ("d_reset", "the scene reloads"),
          ("e_later", "still the same room")]

# (model, condition, verbatim sentence). Each must occur in a P1 narrative, or the build stops.
QUOTES = [
    ("gemini-3.1-pro-preview", "A",
     "They then walk directly to the exit door, open it, and leave the room."),
    ("qwen/qwen3-vl-235b-a22b-instruct", "B",
     "They open the door and exit the room, ending the scene."),
    ("anthropic/claude-opus-4.8", "B",
     "The scene then resets to a wider room view, indicating the participant passed through "
     "the exit."),
]
COND_WORD = {"A": "video with audio", "A0": "video, muted", "B": "frames"}


def check_quotes(d):
    p1 = d[d.pid == "P1"]
    for model, cond, q in QUOTES:
        hit = p1[(p1.model == model) & (p1.cond == cond) & p1.narrative.str.contains(q, regex=False)]
        if hit.empty:
            raise SystemExit(f"quote not found verbatim in P1 / {model} / {cond}: {q!r}")
        if not hit.transit.any():
            raise SystemExit(f"quote is not classified as asserting an exit: {q!r}")


def wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return "\n".join(lines)


def build():
    d = pd.read_csv(OUT / "C4_script_completion.csv")
    check_quotes(d)
    meta = json.load(open(OUT / "filmstrip" / "frames.json", encoding="utf-8"))

    fig = plt.figure(figsize=(7.4, 6.05))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.95, 0.40, 1.55], hspace=0.22,
                             left=0.035, right=0.985, top=0.955, bottom=0.075)

    # ================= (a) the recording
    strip = outer[0].subgridspec(1, len(FRAMES), wspace=0.045)
    cut_x = first_frame = None
    for k, (key, cap) in enumerate(FRAMES):
        ax = fig.add_subplot(strip[k])
        ax.imshow(imread(OUT / "filmstrip" / f"{key}.png"))
        ax.set_xticks([]); ax.set_yticks([])
        # the whole strip is the recording, i.e. what the engine logged, so it is all slate;
        # ember is kept for what the models SAID, in the quote row underneath
        edge = C["engine"]
        for s in ax.spines.values():
            s.set_edgecolor(edge); s.set_linewidth(1.6)
        t = meta["frames"][key]["t_rel_cut"]
        ax.text(0.5, 1.035, f"{t:+.1f} s" if abs(t) < 1 else f"{t:+.0f} s",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7.6,
                fontweight="bold", color=edge, clip_on=False)
        ax.text(0.5, -0.09, cap, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.0, color=C["ink"])
        if k == 0:
            first_frame = ax
        if key == "d_reset":
            cut_x = ax
    # the cut, drawn between the two frames that straddle it
    bb = cut_x.get_position()
    fig.add_artist(plt.Line2D([bb.x0 - 0.0065] * 2, [bb.y0 - 0.012, bb.y1 + 0.028],
                              transform=fig.transFigure, color=C["ink"], lw=1.2,
                              ls=(0, (3, 2))))
    fig.text(bb.x0 - 0.0065, bb.y1 + 0.034, "scene cut", ha="center", va="bottom",
             fontsize=7.0, fontweight="bold", color=C["ink"])
    # The strip shares its exact left and right bounds with the lower panels: its first frame
    # begins at panel (b)'s y-axis ticks and its final frame ends at panel (c)'s right edge.
    first_box = first_frame.get_position()
    fig.text(first_box.x0, first_box.y1 + 0.038, "(a) Recorded sequence around the scene cut",
             ha="left", va="bottom", fontsize=9.2, fontweight="regular", color=C["ink"])

    # ================= the three quotes
    qrow = outer[1].subgridspec(1, len(QUOTES), wspace=0.06)
    quote_shift = 0.035
    for k, (model, cond, q) in enumerate(QUOTES):
        ax = fig.add_subplot(qrow[k]); ax.axis("off")
        q_box = ax.get_position()
        ax.set_position([q_box.x0, q_box.y0 + quote_shift, q_box.width, q_box.height])
        ax.add_patch(Rectangle((0.0, 0.04), 1.0, 0.92, fc="white", ec=C["grid"], lw=0.6,
                               transform=ax.transAxes, zorder=1))
        ax.add_patch(Rectangle((0.0, 0.04), 0.022, 0.92, fc=C["vlm"], ec="none",
                               transform=ax.transAxes, zorder=2))
        ax.text(0.07, 0.90, f"“{wrap(q, 44)}”", transform=ax.transAxes, fontsize=7.1,
                va="top", ha="left", color=C["ink"], style="italic", linespacing=1.3, zorder=3)
        ax.text(0.07, 0.10, f"{fs.MODEL_SHORT[model]}, {COND_WORD[cond]}",
                transform=ax.transAxes, fontsize=6.7, fontweight="bold", color=C["vlm"],
                va="bottom", ha="left", zorder=3)

    # ================= (b) the correction, per model
    low = outer[2].subgridspec(1, 2, width_ratios=[1.22, 1.26], wspace=0.20)
    ax = fig.add_subplot(low[0])
    g = (d.groupby("model").agg(new=("transit", "mean"), old=("transit_old", "mean"))
         .sort_values("new"))
    rows = list(g.iterrows()) + [("__all__", pd.Series({"new": d.transit.mean(),
                                                        "old": d.transit_old.mean()}))]
    for i, (m, r) in enumerate(rows):
        allrow = m == "__all__"
        y = i + (0.55 if allrow else 0)
        ax.plot([0, r.new * 100], [y, y], color=C["vlm_dim"], lw=2.6,
                solid_capstyle="round", zorder=2)
        ax.scatter(r.new * 100, y, s=64 if allrow else 40, color=C["vlm"], zorder=4,
                   edgecolor="white", linewidth=0.8)
        ax.text(r.new * 100 + 3.0, y, f"{r.new * 100:.0f}", ha="left", va="center",
                fontsize=6.9, color=C["vlm"], fontweight="bold")
    ax.axhline(len(rows) - 1 + 0.02, color=C["grid"], lw=0.8)
    labels = [fs.MODEL_SHORT.get(m, m) for m, _ in rows[:-1]] + ["All eight models"]
    ax.set_yticks([i + (0.55 if m == "__all__" else 0) for i, (m, _) in enumerate(rows)])
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(0, 110)
    ax.set_ylim(-0.5, len(rows) + 0.2)
    fs.axis(ax, "Narratives asserting an exit (%)", None)
    ax.xaxis.set_label_coords(0.5, -0.17)
    b_label_figure_y = ax.get_position().y0 - 0.17 * ax.get_position().height
    fs.panel_title(ax, "b", "Exit-claim rates by model", pad=12)
    fs.despine(ax, left=False)
    fs.faint_grid(ax, axis="x")

    # ================= (c) the input-condition response cards
    ax = fig.add_subplot(low[1])
    c_box = ax.get_position()
    c_stretch = 0.020
    ax.set_position([c_box.x0, c_box.y0 - c_stretch, c_box.width, c_box.height + c_stretch])
    goog = [m for m in ["gemini-3.1-pro-preview", "gemini-3-flash-preview",
                        "gemini-3.1-flash-lite"]]
    conds = ["B", "A0", "A"]
    M = np.array([[d[(d.model == m) & (d.cond == c)].transit.mean() * 100 for c in conds]
                  for m in goog])
    mean = d[d.model.isin(goog)].groupby("cond").transit.mean().reindex(conds).values * 100
    grid = np.vstack([M, mean])
    cmap = ramp("HEAT")
    header = ["Frames\n1 per s", "Video\nmuted", "Video\n+ audio"]
    card_gap, card_h = 0.075, 0.86
    ax.add_patch(Rectangle((-0.12, -0.10), 3.24, 1.00, fc="#F6F8FA", ec="none", zorder=0))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            y = 3 - i if i < 3 else -0.08
            card_color = cmap(v / 100)
            is_mean = i == 3
            ax.add_patch(Rectangle((j + card_gap, y + 0.08), 1 - 2 * card_gap, card_h,
                                   fc=card_color, ec=C["ink"] if is_mean else "white",
                                   lw=1.1 if is_mean else 0.65, zorder=1))
            ax.add_patch(Rectangle((j + card_gap, y + 0.84), 1 - 2 * card_gap, 0.06,
                                   fc=(*mpl.colors.to_rgb("white"), 0.36), ec="none", zorder=2))
            ax.text(j + 0.5, y + 0.62, f"{v:.0f}%", ha="center", va="center",
                    fontsize=9.0 if is_mean else 8.4, fontweight="bold",
                    color="white" if v > 58 else C["ink"], zorder=3)
            ax.add_patch(Rectangle((j + 0.20, y + 0.35), 0.60, 0.075,
                                   fc=(*mpl.colors.to_rgb("white"), 0.60), ec="none", zorder=2))
            ax.add_patch(Rectangle((j + 0.20, y + 0.35), 0.60 * v / 100, 0.075,
                                   fc="white" if v > 58 else C["ink"], ec="none", zorder=3))
            note = "Reference" if j == 0 else f"{v - grid[i, 0]:+.0f} pp"
            ax.text(j + 0.5, y + 0.18, note, ha="center", va="center", fontsize=6.4,
                    fontweight="bold" if is_mean and j > 0 else "regular",
                    color="white" if v > 58 else C["ink"], zorder=3)
    for j, label in enumerate(header):
        ax.text(j + 0.5, 4.08, label, ha="center", va="bottom", fontsize=7.0,
                fontweight="bold", color=C["ink"], linespacing=1.05)
        ax.add_patch(Rectangle((j + 0.22, 4.00), 0.56, 0.035, fc=C["ink"], ec="none"))
    ax.set_xlim(0, 3); ax.set_ylim(0, 4.57)
    ax.set_xticks([])
    ax.set_yticks([3.49, 2.49, 1.49, 0.33])
    ax.set_yticklabels([fs.MODEL_SHORT[m].replace("Gemini ", "") for m in goog]
                       + ["Mean"], fontsize=7.2)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(length=0, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    fs.panel_title(ax, "c", "Exit-claim rates by input format", pad=12)
    fs.axis(ax, "Input, from least to most of the recording", None)
    c_label_axes_y = (b_label_figure_y - ax.get_position().y0) / ax.get_position().height
    ax.xaxis.set_label_coords(0.5, c_label_axes_y)
    ax.text(1.50, -0.040, "Increasing recording context  →", ha="center", va="top",
            transform=ax.get_xaxis_transform(), clip_on=False,
            fontsize=6.6, color=C["sub"], style="italic")

    fs.save(fig, "q1_fig3_script")
    print("  wrote q1_fig3_script")


if __name__ == "__main__":
    fs.setup()
    build()
