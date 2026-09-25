"""Q1 journal figures - scene geometry, plus two exhibits that no longer ship.

The attention raster is diverted to superseded/ purely on the journal's five-figure limit, not
because anything is wrong with it. Adding the method pipeline as Figure 1 made six, and the
raster is the only one of the six that neither claim rests on: it shows the time course, while
the two results are about WHICH object is named and WHAT ending is asserted. It belongs in the
Zenodo deposit, since IJDRS accepts no appendices.


F1  The room decides what is resolvable : plan view + angular-separation matrix
F2  The resolution curve                : angular error distribution, the headline
F3  Attention raster                    : engine reference vs model claim, phase-aligned

Everything here is built from data already on disk. Nothing depends on the human audit.
Run: py evactime/Q1_figures.py
"""
import json, pathlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
DS = ROOT / "vlm dataset"
C = fs.C

# one fixed colour per AOI super-class, used identically in every figure
# object colors come from the shared palette so this figure cannot drift from the others
# (an earlier hard-coded set carried a purple NPC after the palette dropped purple)
AOI_C = dict(fs.OBJECT, misc="#9AA3AE")
SUPER = {-1: "none", 0: "none", 1: "misc", 2: "misc", 3: "sign", 4: "alarm", 10: "alarm",
         5: "door", 7: "door", 12: "door", 13: "door", 6: "display", 11: "equipment",
         8: "npc", 9: "npc"}
NAMES = {3: "Exit sign", 4: "Alarm (visual)", 10: "Alarm (audible)", 6: "Queue display",
         13: "Exit door", 12: "Doctor's door", 8: "NPC 1", 9: "NPC 2", 11: "Extinguisher"}
EYE = np.array([2.356, 1.223, -1.387])          # canonical seated viewpoint (published constant)


def geom():
    g = json.load(open(OUT / "scene_geometry.json"))
    pos = {int(k): np.array(v["pos"]) for k, v in g.items() if v.get("pos")}
    pos[6] = np.array([3.71000004, 2.58500004, 2.95700002])
    return pos


def ang(a, b, eye=EYE):
    va, vb = a - eye, b - eye
    return float(np.degrees(np.arccos(np.clip(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)),
                                              -1, 1))))


# ----------------------------------------------------------------- F1
def fig1(pos):
    fig = plt.figure(figsize=(fs.FULL_W, 3.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=-0.10,
                          left=0.075, right=0.985, top=0.88, bottom=0.23)

    # (a) plan view, true scale. The wall objects are co-located by design, so they are
    # numbered rather than labelled in place - which is the figure making the paper's point.
    ax = fig.add_subplot(gs[0])
    ax.add_patch(Rectangle((-3.75, -3.2), 7.5, 6.4, fc="#FFFFFF", ec=C["ink"], lw=0.9, zorder=1))
    order = [3, 4, 10, 6, 13, 11, 9, 12, 8]
    for oid in order:
        ax.plot([EYE[0], pos[oid][0]], [EYE[2], pos[oid][2]], lw=0.4, color=C["sub"],
                alpha=0.40, zorder=2)
    # the five wall objects sit within ~2 m of each other; offset their callouts on a fan so
    # all nine are legible, with leaders back to the true position
    OFFS = {3: (-1.05, 0.62), 4: (-1.95, 0.30), 10: (-1.55, 0.48), 6: (0.55, 0.60),
            11: (1.05, 0.22)}
    for n, oid in enumerate(order, 1):
        p_ = pos[oid]
        ox, oz = OFFS.get(oid, (0.0, 0.0))
        cx, cz = p_[0] + ox, p_[2] + oz
        ax.scatter(p_[0], p_[2], s=13, c=AOI_C[SUPER[oid]], ec="none", zorder=4)
        if (ox, oz) != (0.0, 0.0):
            ax.plot([p_[0], cx], [p_[2], cz], lw=0.5, color=C["sub"], zorder=3)
        ax.scatter(cx, cz, s=118, c=AOI_C[SUPER[oid]], ec="white", lw=1.0, zorder=5)
        ax.annotate(str(n), (cx, cz), fontsize=6.2, color="white", fontweight="bold",
                    ha="center", va="center", zorder=6)
    ax.scatter(*EYE[[0, 2]], marker="X", s=78, c=C["ink"], zorder=5)
    ax.annotate("seated viewpoint", (EYE[0], EYE[2] - 0.30), fontsize=6.6, ha="center",
                va="top", color=C["ink"])
    ax.set_xlim(-4.1, 5.0); ax.set_ylim(-3.5, 4.2); ax.set_aspect("equal")
    ax.set_anchor("N")
    fs.axis(ax, "X (m)", "Z (m)")
    fs.panel_title(ax, "a", "Virtual waiting-room layout")

    # (b) angular separation matrix
    ax = fig.add_subplot(gs[1])
    ks = [3, 4, 10, 6, 13, 11, 9, 12, 8]
    M = np.array([[ang(pos[i], pos[j]) if i != j else np.nan for j in ks] for i in ks])
    from palette import ramp
    im = ax.imshow(M, cmap=ramp("DEPTH"), vmin=0, vmax=90)   # engine geometry: the slate ramp, no purple
    ax.set_anchor("N")
    ax.set_xticks(range(len(ks))); ax.set_yticks(range(len(ks)))
    lab = [NAMES[k] for k in ks]
    ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=6.4)
    ax.set_yticklabels(lab, fontsize=6.4)
    for a in range(len(ks)):
        for b in range(len(ks)):
            if a != b:
                v = M[a, b]
                ax.text(b, a, f"{v:.0f}", ha="center", va="center", fontsize=5.6,
                        color="white" if v > 45 else C["ink"])
    # ring the sub-20 deg block
    # the five cues mounted on the back wall, which the models confuse with one another
    ax.add_patch(Rectangle((-0.5, -0.5), 5, 5, fill=False, ec=C["ink"], lw=1.2, zorder=5))
    ax.annotate("cues on the back wall", (2.0, -0.95), fontsize=6.6, ha="center",
                color=C["ink"], fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fs.colorbar_label(cb, "Angular separation (°)")
    cb.ax.tick_params(labelsize=6.6)
    fs.panel_title(ax, "b", "Angular separation between targets")

    # A full-width key prevents an empty lower-right area and preserves a clean gap below the
    # map x-axis label. The numbered circles exactly repeat the map symbols.
    key_ax = fig.add_axes([0.075, 0.000, 0.91, 0.065])
    key_ax.axis("off")
    key_labels = {3: "Exit sign", 4: "Visual alarm", 10: "Audible alarm", 6: "Queue display",
                  13: "Exit door", 11: "Extinguisher", 9: "NPC 2", 12: "Doctor's door",
                  8: "NPC 1"}
    # Center the complete row as a single unit. A fixed inter-item gap keeps the distance
    # between each complete circle-plus-label group constant.
    labels = [key_labels[oid] for oid in order]
    item_widths = [0.020 + 0.0062 * len(label) for label in labels]
    item_gap = 0.017
    # The rendered row has a small left-side marker overhang. This offset centers its visible
    # bounds on the complete figure rather than its text-only bounds.
    circle_x = (1.0 - sum(item_widths) - item_gap * (len(order) - 1)) / 2 + 0.050
    for n, (oid, label, item_width) in enumerate(zip(order, labels, item_widths), 1):
        key_ax.scatter(circle_x, 0.38, s=86, color=AOI_C[SUPER[oid]], edgecolor="white", linewidth=0.8,
                       transform=key_ax.transAxes, zorder=2)
        key_ax.text(circle_x, 0.38, str(n), ha="center", va="center", fontsize=5.3, fontweight="bold",
                    color="white", transform=key_ax.transAxes, zorder=3)
        key_ax.text(circle_x + 0.020, 0.38, label, ha="left", va="center", fontsize=5.5,
                    color=C["ink"], transform=key_ax.transAxes)
        circle_x += item_width + item_gap
    fs.save(fig, "q1_fig1_room")
    print("  wrote q1_fig1_room")


# ----------------------------------------------------------------- F2
def fig2(d):
    fig = plt.figure(figsize=(fs.FULL_W, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.34)

    # (a) tolerance curve: share of calls within x degrees
    ax = fig.add_subplot(gs[0])
    xs = np.linspace(0, 60, 400)
    for t, lab, col in [("look_at_npc", "NPC", AOI_C["npc"]),
                        ("look_at_alarm", "Alarm", AOI_C["alarm"]),
                        ("look_at_exit_sign", "Exit sign", AOI_C["sign"])]:
        e = d[d.type == t].sep_deg.values
        ax.plot(xs, [(e <= x).mean() for x in xs], color=col, lw=1.7, label=lab)
    e = d.sep_deg.values
    ax.plot(xs, [(e <= x).mean() for x in xs], color=C["ink"], lw=1.1, ls=(0, (4, 2)), label="All")
    ax.axhline(0.8, color=C["sub"], lw=0.6, ls=":")
    p80 = np.quantile(e, .8)
    ax.axvline(p80, color=C["vlm"], lw=0.9)
    ax.annotate(f"80% coverage\nneeds {p80:.0f}°", (p80 + 1.5, 0.36), fontsize=6.8,
                color=C["vlm"], fontweight="bold")
    ax.set_xlim(0, 60); ax.set_ylim(0, 1.02)
    fs.axis(ax, "Tolerance (°)", "Share of calls within tolerance")
    fs.faint_grid(ax); fs.despine(ax)
    ax.legend(loc="lower right", fontsize=6.8)
    fs.title(ax, "Resolution tolerance")
    fs.panel_label(ax, "a")

    # (b) error distribution per assertion type
    ax = fig.add_subplot(gs[1])
    types = ["look_at_npc", "look_at_alarm", "look_at_exit_sign"]
    labs = ["NPC", "Alarm", "Exit sign"]
    for i, t in enumerate(types):
        e = d[d.type == t].sep_deg.values
        jit = (np.random.default_rng(7).random(len(e)) - .5) * 0.55
        ax.scatter(e, np.full(len(e), i) + jit, s=3.2, alpha=0.22,
                   color=AOI_C[{"look_at_npc": "npc", "look_at_alarm": "alarm",
                                "look_at_exit_sign": "sign"}[t]], lw=0)
        ax.plot([np.median(e)] * 2, [i - .34, i + .34], color=C["ink"], lw=2.0, zorder=5)
    ax.axvline(20.1, color=C["vlm"], lw=0.9, ls="--")
    ax.annotate("20°", (21, 2.42), fontsize=6.8, color=C["vlm"], fontweight="bold")
    ax.set_yticks(range(3)); ax.set_yticklabels(labs, fontsize=7.6)
    ax.set_xlim(-2, 125)
    fs.axis(ax, "Angular error (°)", None)
    fs.despine(ax); fs.faint_grid(ax, axis="x")
    fs.title(ax, "Error by assertion type")
    fs.panel_label(ax, "b")

    # (c) the limit is a property of the scene, not the model
    ax = fig.add_subplot(gs[2])
    SHORT = {"gemini-3-flash-preview": "Gemini 3 Flash", "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
             "gemini-3.1-flash-lite": "Gemini 3.1 Lite", "openai/gpt-5.2": "GPT-5.2",
             "anthropic/claude-opus-4.8": "Claude Opus 4.8",
             "anthropic/claude-sonnet-5": "Claude Sonnet 5",
             "qwen/qwen3-vl-235b-a22b-instruct": "Qwen3-VL 235B",
             "meta-llama/llama-4-maverick": "Llama 4 Maverick"}
    g = (d.groupby("model").sep_deg.agg(n="size", med="median",
                                        p80=lambda s: s.quantile(.8)).reset_index())
    g["lab"] = g.model.map(SHORT).fillna(g.model)
    g = g.sort_values("p80")
    y = np.arange(len(g))
    ax.hlines(y, g.med, g.p80, color=C["sub"], lw=1.1, zorder=2)
    ax.scatter(g.med, y, s=22, color=C["engine"], zorder=3, label="median")
    ax.scatter(g.p80, y, s=30, color=C["vlm"], marker="D", zorder=3, label="80th pct")
    ax.axvline(20.1, color=C["vlm"], lw=0.9, ls="--", zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(g.lab, fontsize=6.6)
    ax.set_xlim(0, 30); ax.set_ylim(-0.7, len(g) - 0.3)
    fs.axis(ax, "Angular error (°)", None)
    fs.despine(ax); fs.faint_grid(ax, axis="x")
    ax.legend(loc="upper left", fontsize=6.6, bbox_to_anchor=(0.02, 0.30))
    fs.title(ax, "Same limit for every model")
    fs.panel_label(ax, "c")

    fs.save(fig, "q1_fig2_resolution", superseded=True)
    print("  wrote q1_fig2_resolution")


# ----------------------------------------------------------------- F3
def fig3():
    a = pd.read_csv(OUT / "A1_assertions_scored.csv")
    a = a[a.timed & a.t_unity.notna()]
    h = pd.read_csv(DS / "participant_hitobject_timeseries.csv", sep=";")
    h["pid"] = "P" + h.Participant.str.extract(r"P0*(\d+)")[0]
    h["cls"] = h.HitObjectID.map(SUPER).fillna("misc")
    mk = pd.read_csv(DS / "participant_marker_times.csv", sep=";")
    mk["pid"] = "P" + mk.Participant.str.extract(r"P0*(\d+)")[0]
    mk = mk.set_index("pid")

    pids = sorted(a.pid.unique(), key=lambda p: int(p[1:]))
    fig, ax = plt.subplots(figsize=(fs.FULL_W, 4.5))
    for i, pid in enumerate(pids):
        m = mk.loc[pid]
        g = h[(h.pid == pid) & (h.UnityTime >= m.ALARM_TRIGGERED - 30) &
              (h.UnityTime <= m.OutroScene)]
        t = g.UnityTime.values - m.ALARM_TRIGGERED
        if len(t) < 2:
            continue
        cls = g.cls.values
        # engine band
        start = 0
        for j in range(1, len(t) + 1):
            if j == len(t) or cls[j] != cls[start]:
                ax.add_patch(Rectangle((t[start], i + .16), t[j - 1] - t[start] + .02, .34,
                                       fc=AOI_C.get(cls[start], "#CCC"), lw=0, zorder=2))
                start = j
        # model claims
        for _, r in a[a.pid == pid].iterrows():
            cl = {"look_at_alarm": "alarm", "look_at_exit_sign": "sign",
                  "look_at_npc": "npc"}[r.type]
            ax.plot([r.t_unity - m.ALARM_TRIGGERED] * 2, [i - .30, i + .04],
                    color=AOI_C[cl], lw=0.75, alpha=0.85, zorder=3)
    ax.axvline(0, color=C["ink"], lw=1.0, zorder=6)
    ax.annotate("alarm", (0.6, len(pids) + .2), fontsize=7.2, fontweight="bold", color=C["ink"])
    ax.set_yticks(range(len(pids)))
    ax.set_yticklabels(pids, fontsize=5.6)
    ax.set_ylim(-1, len(pids) + .6); ax.set_xlim(-31, 46)
    fs.axis(ax, "Time from alarm onset (s)", "Participant")
    fs.despine(ax, left=False)
    handles = [plt.Line2D([], [], color=AOI_C[k], lw=5, label=v) for k, v in
               [("sign", "Exit sign"), ("alarm", "Alarm"), ("npc", "NPC"), ("door", "Door"),
                ("display", "Queue display"), ("equipment", "Extinguisher"), ("none", "Untagged")]]
    ax.legend(handles=handles, ncol=7, fontsize=6.4, loc="lower center",
              bbox_to_anchor=(0.5, 1.02))
    ax.annotate("upper band: eye-tracking reference     lower ticks: model claims",
                (-30, -0.82), fontsize=6.6, color=C["sub"])
    fs.save(fig, "q1_fig3_raster", superseded=True)
    print("  wrote q1_fig3_raster")


if __name__ == "__main__":
    fs.setup()
    pos = geom()
    d = pd.read_csv(OUT / "C2_resolution.csv")
    print("building figures:")
    fig1(pos)
    fig2(d)
    fig3()
    print("done ->", ROOT / "evactime" / "figures")
