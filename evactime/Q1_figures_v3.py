"""Q1 journal figures, rebuilt for the corrected analysis.

Replaces the resolution-limit figure set. What changed and why:

  F2 (was "the resolution curve" and "invariance") now shows what the annotation actually
     resolves: a confusion matrix, region against object, and an explicit chance baseline. The
     old panels plotted an angular quantile that is zero by construction when the class is right
     and a fixed room distance when it is wrong, so they could only ever show a scene constant.
  F3 (was "the deliverable") now shows script completion at the corrected rate, the size of the
     correction, and the input-condition effect, whose direction reversed once the classifier
     stopped counting the noun phrase "exit door".
  F4 is new. The human audit previously existed only as a table, and its most important feature
     is the shape of the coder disagreement, which a table cannot show.

Run: py evactime/Q1_figures_v3.py
"""
import json, pathlib, collections
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, PathPatch
from matplotlib.path import Path

import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
C = fs.C

CLS_LABEL = {"alarm": "Alarm", "sign": "Exit sign", "display": "Queue display",
             "door": "Door", "npc": "NPC", "equipment": "Extinguisher", "none": "Nothing tagged"}
TYPE_LABEL = {"look_at_npc": "NPC", "look_at_alarm": "Alarm", "look_at_exit_sign": "Exit sign"}


def flow_ribbons(ax, M, claim_keys, ref_keys, claim_labels, ref_labels,
                 gap=0.012, pad_out=0.055):
    """Claim on the left, reference on the right, one ribbon per cell.

    A confusion matrix answers "what fraction of row i went to column j", which the reader has to
    integrate cell by cell. The question this paper asks is directional and unequal: 1306
    assertions were made about three things, and the gaze was on six. A flow reads that directly.
    Ribbon thickness is the number of assertions, so the eye gets the volumes without arithmetic,
    and the correct flows are the only ones drawn in the verdict colour.

    Ribbons are cubic Beziers with a flat mid-section, drawn back to front by size so the large
    confusions do not bury the small correct ones.
    """
    # Order the right-hand blocks by the weighted mean position of the claims feeding them
    # (the barycentre heuristic). Sorting them alphabetically or by size leaves ribbons crossing
    # for no reason, which reads as visual noise rather than as information.
    src_pos = np.arange(M.shape[0], dtype=float)
    bary = (M * src_pos[:, None]).sum(axis=0) / np.maximum(M.sum(axis=0), 1e-9)
    ordr = np.argsort(bary)
    M = M[:, ordr]
    ref_keys = [ref_keys[i] for i in ordr]
    ref_labels = [ref_labels[i] for i in ordr]

    tot = M.sum()
    # Use the complete width of the panel cell. This gives the flow body the same visible
    # horizontal scale as the lower-left chart, while leaving a compact margin for direct labels.
    xL, xR, w = 0.02, 0.98, 0.045

    # stack heights, normalised to a unit column with small gaps between blocks
    def stack(vals):
        n = len(vals)
        h = (1 - pad_out * 2 - gap * (n - 1)) * np.asarray(vals) / tot * (tot / max(sum(vals), 1))
        h = (1 - pad_out * 2 - gap * (n - 1)) * np.asarray(vals) / max(sum(vals), 1)
        y, out = pad_out, []
        for v in h:
            out.append((y, y + v))
            y += v + gap
        return out

    Lblocks = stack(M.sum(axis=1))
    Rblocks = stack(M.sum(axis=0))

    # draw ribbons largest first so small correct flows stay visible on top
    cells = [(i, k, M[i, k]) for i in range(M.shape[0]) for k in range(M.shape[1]) if M[i, k] > 0]
    cells.sort(key=lambda c: (c[0], c[1]))      # by source, then by destination
    Lcur = {i: Lblocks[i][0] for i in range(M.shape[0])}
    Rcur = {k: Rblocks[k][0] for k in range(M.shape[1])}
    # consume each block in the same (sorted) order on both sides so ribbons do not cross twice
    for i, k, v in cells:
        hL = (Lblocks[i][1] - Lblocks[i][0]) * v / M[i].sum()
        hR = (Rblocks[k][1] - Rblocks[k][0]) * v / M[:, k].sum()
        y0a, y0b = Lcur[i], Lcur[i] + hL
        y1a, y1b = Rcur[k], Rcur[k] + hR
        Lcur[i] += hL
        Rcur[k] += hR
        correct = claim_keys[i] == ref_keys[k]
        col = C["good"] if correct else C["vlm"]
        alpha = 0.92 if correct else 0.34
        mx = (xL + xR) / 2
        verts = [(xL + w, y0a),
                 (mx, y0a), (mx, y1a), (xR - w, y1a),
                 (xR - w, y1b),
                 (mx, y1b), (mx, y0b), (xL + w, y0b),
                 (xL + w, y0a)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=col, edgecolor="none",
                               alpha=alpha, zorder=3 if correct else 2))

    # end blocks, labelled directly
    for i, (lo, hi) in enumerate(Lblocks):
        ax.add_patch(Rectangle((xL, lo), w, hi - lo, facecolor=C["vlm"], edgecolor="none",
                               zorder=5))
        ax.text(xL - 0.02, (lo + hi) / 2, f"{claim_labels[i]}\n{int(M[i].sum())}", ha="right",
                va="center", fontsize=7.0, color=C["ink"], linespacing=1.25)
    # Small blocks sit almost on top of each other, so their labels need de-overlapping. Walk
    # down the column pushing each label clear of the one above and draw a leader when the label
    # has moved off its block.
    MINSEP, prev = 0.052, None
    for k in range(len(Rblocks) - 1, -1, -1):
        lo, hi = Rblocks[k]
        ax.add_patch(Rectangle((xR - w, lo), w, hi - lo, facecolor=OBJ_COL(ref_keys[k]),
                               edgecolor="none", zorder=5))
        ty = (lo + hi) / 2
        if prev is not None and prev - ty < MINSEP:
            ty = prev - MINSEP
        prev = ty
        if abs(ty - (lo + hi) / 2) > 0.004:
            ax.plot([xR + 0.004, xR + 0.017], [(lo + hi) / 2, ty], color=C["sub"], lw=0.6,
                    zorder=4, clip_on=False)
        ax.text(xR + 0.022, ty, f"{ref_labels[k]}  {int(M[:, k].sum())}", ha="left",
                va="center", fontsize=7.0, color=C["ink"])

    ax.text(xL + w / 2, 1.0, "Model assertions", ha="center", va="bottom", fontsize=6.8,
            fontweight="bold", color=C["vlm"])
    ax.text(xR - w / 2, 1.0, "Recorded gaze target", ha="center", va="bottom", fontsize=6.8,
            fontweight="bold", color=C["engine"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.06)
    ax.axis("off")


def OBJ_COL(key):
    return fs.OBJECT.get(key, C["sub"])


# ================================================================== F2
def fig_resolves():
    d = pd.read_csv(OUT / "C2_confusion.csv")
    j = json.load(open(OUT / "C2_confusion.json", encoding="utf-8"))

    fig = plt.figure(figsize=(7.4, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                          hspace=0.52, wspace=0.46,
                          left=0.085, right=0.975, top=0.94, bottom=0.105)

    # ---- (a) where the claim went, and where the gaze actually was
    ax = fig.add_subplot(gs[0, 0])
    # This panel has no x-axis title. Extend its lower edge into the reserved row gap so the
    # final NPC band reaches the visual baseline of panel (b)'s x-axis title.
    a_box = ax.get_position()
    a_extension = 0.085
    ax.set_position([a_box.x0, a_box.y0 - a_extension, a_box.width, a_box.height + a_extension])
    claim_order = ["npc", "alarm", "sign"]
    ref_order = ["npc", "alarm", "sign", "display", "door", "equipment"]
    cm = pd.crosstab(d.claimed_cls, d.actual_cls)
    M = np.array([[cm.loc[r, c] if (r in cm.index and c in cm.columns) else 0
                   for c in ref_order] for r in claim_order], dtype=float)
    flow_ribbons(ax, M, claim_order, ref_order,
                 [CLS_LABEL[c] for c in claim_order], [CLS_LABEL[c] for c in ref_order])
    fs.panel_title(ax, "a", "Model assertions and recorded gaze targets", pad=12)

    # ---- (b) region versus object, with chance
    ax = fig.add_subplot(gs[0, 1])
    order = ["look_at_npc", "look_at_alarm", "look_at_exit_sign"]
    y = np.arange(len(order))[::-1]
    for i, t in zip(y, order):
        g = d[d.type == t]
        ax.barh(i + .18, g.region_hit.mean() * 100, height=.34,
                color=C["engine"], alpha=.30, zorder=2)
        ax.barh(i - .18, g.correct.mean() * 100, height=.34, color=C["vlm"], zorder=3)
        ax.text(g.region_hit.mean() * 100 + 1.5, i + .18, f"{g.region_hit.mean() * 100:.0f}",
                va="center", fontsize=7.2, color=C["sub"])
        ax.text(g.correct.mean() * 100 + 1.5, i - .18, f"{g.correct.mean() * 100:.0f}",
                va="center", fontsize=7.2, fontweight="bold", color=C["vlm"])
    ax.axvline(j["chance"] * 100, color=C["vlm"], lw=1.0, ls=(0, (3, 2)), zorder=6)
    import matplotlib.transforms as mtrans
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(j["chance"] * 100, 1.0, "object chance", transform=tr, fontsize=6.2,
            color=C["vlm"], ha="center", va="bottom")
    ax.axvline(j["region_chance"] * 100, color=C["engine"], lw=1.0, ls=(0, (3, 2)), zorder=6)
    ax.text(j["region_chance"] * 100, 1.0, "region chance", transform=tr, fontsize=6.2,
            color=C["engine"], ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([TYPE_LABEL[t] for t in order], fontsize=7.6)
    ax.set_xlim(0, 104)
    fs.axis(ax, "Agreement with the reference (%)", None)
    fs.panel_title(ax, "b", "Region-level versus object-level agreement", pad=12)
    fs.despine(ax, left=False)
    ax.annotate("right region", xy=(0.50, 0.12), xycoords="axes fraction", fontsize=7.0,
                color=C["engine"], fontweight="bold")
    ax.annotate("exact object", xy=(0.50, 0.02), xycoords="axes fraction", fontsize=7.0,
                color=C["vlm"], fontweight="bold")

    # ---- (c) the positive control, stated as a separability contrast
    ax = fig.add_subplot(gs[1, 0])
    groups = [("NPC\n(free-standing)", j["by_type"]["look_at_npc"], C["good"]),
              ("Alarm\n(wall cluster)", j["by_type"]["look_at_alarm"], C["mid"]),
              ("Exit sign\n(wall cluster)", j["by_type"]["look_at_exit_sign"], C["bad"])]
    for i, (lab, v, col) in enumerate(groups):
        ax.bar(i, v * 100, width=.62, color=col, zorder=3)
        ax.text(i, v * 100 + 2.5, f"{v * 100:.1f}%", ha="center", fontsize=7.6,
                fontweight="bold", color=col)
    ax.axhline(j["chance"] * 100, color=C["ink"], lw=1.0, ls=(0, (3, 2)), zorder=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.2)
    ax.set_ylim(0, 104)
    fs.axis(ax, None, "Exact object named (%)")
    fs.panel_title(ax, "c", "Object identification by target isolation", pad=12)
    fs.despine(ax)
    fs.faint_grid(ax)

    # ---- (d) the scoring rule, across the matching tolerance. The earlier panel here showed the
    # atoms of the withdrawn angular measure; the scoring-rule result is a finding of the paper
    # and had no figure.
    ax = fig.add_subplot(gs[1, 1])
    a = pd.read_csv(OUT / "A1_assertions_scored.csv")
    a = a[a.timed & a.t_unity.notna()]
    tols = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    # observed values from A2, which scores both rules over tagged samples exactly as Equation 2
    _n = json.load(open(OUT / "A2_scoring_null.json", encoding="utf-8"))
    anyv = [_n[f"{t:g}"]["any"] * 100 for t in tols]
    modv = [_n[f"{t:g}"]["modal"] * 100 for t in tols]
    nul = json.load(open(OUT / "A2_scoring_null.json", encoding="utf-8"))
    anyn = [nul[f"{t:g}"]["any_null"] * 100 for t in tols]
    modn = [nul[f"{t:g}"]["modal_null"] * 100 for t in tols]
    # the gap between each rule and its own chance level is what the rule actually measures
    ax.fill_between(tols, anyn, anyv, color=C["vlm"], alpha=.12, lw=0, zorder=1)
    ax.fill_between(tols, modn, modv, color=C["engine"], alpha=.12, lw=0, zorder=1)
    ax.plot(tols, anyn, color=C["vlm"], lw=1.1, ls=(0, (3, 2)), zorder=2)
    ax.plot(tols, modn, color=C["engine"], lw=1.1, ls=(0, (3, 2)), zorder=2)
    ax.text(tols[-1] + .08, anyn[-1], "its chance", fontsize=6.2, color=C["vlm"], va="center")
    ax.plot(tols, anyv, color=C["vlm"], lw=1.8, marker="o", ms=3.6, zorder=3)
    ax.plot(tols, modv, color=C["engine"], lw=1.8, marker="o", ms=3.6, zorder=3)
    ax.text(tols[-1] + .08, anyv[-1], f"any sample\n{anyv[-1]:.0f}%", fontsize=6.8,
            color=C["vlm"], va="center", fontweight="bold", linespacing=1.1)
    ax.text(tols[-1] + .08, modv[-1], f"modal\n{modv[-1]:.0f}%", fontsize=6.8,
            color=C["engine"], va="center", fontweight="bold", linespacing=1.1)
    ax.set_xlim(0, 3.75)
    ax.set_ylim(0, 80)
    ax.set_xticks(tols[1:])
    fs.axis(ax, "Matching window, ± s", "Agreement (%)")
    fs.panel_title(ax, "d", "Agreement across matching windows", pad=12)
    fs.despine(ax)
    fs.faint_grid(ax)

    fs.save(fig, "q1_fig2_resolves")
    print("  wrote q1_fig2_resolves")


# ================================================================== F3
def fig_script_completion():
    d = pd.read_csv(OUT / "C4_script_completion.csv")

    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], height_ratios=[1, 1],
                          hspace=0.75, wspace=0.40,
                          left=0.095, right=0.98, top=0.845, bottom=0.115)

    # ---- (a) the corrected rate, against the measurement it replaces
    ax = fig.add_subplot(gs[0, 0])
    new, old = d.transit.mean() * 100, d.transit_old.mean() * 100
    ax.bar(0, old, width=.52, color=C["grid"], zorder=2)
    ax.bar(0, new, width=.52, color=C["vlm"], zorder=3)
    ax.annotate("", xy=(0.40, new), xytext=(0.40, old),
                arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=0.9))
    ax.text(0.46, (new + old) / 2, f"{old - new:.0f} pts\nwere the\nphrase\n\"exit door\"",
            fontsize=6.6, va="center", color=C["ink"], linespacing=1.3)
    ax.text(0, new - 7, f"{new:.1f}%", ha="center", fontsize=9.5, fontweight="bold", color="white")
    ax.text(0, old + 2.5, f"{old:.1f}% withdrawn", ha="center", fontsize=7.0, color=C["sub"])
    ax.set_xlim(-0.5, 1.35)
    ax.set_ylim(0, 104)
    ax.set_xticks([])
    fs.axis(ax, None, "Narratives asserting an exit (%)")
    fs.title(ax, "The corrected rate")
    fs.panel_label(ax, "a", dx=-0.19)
    fs.despine(ax, bottom=False)
    fs.faint_grid(ax)

    # ---- (b) by model, corrected against withdrawn
    ax = fig.add_subplot(gs[:, 1])
    g = (d.groupby("model").agg(new=("transit", "mean"), old=("transit_old", "mean"),
                                n=("transit", "size")).sort_values("new"))
    y = np.arange(len(g))
    for i, (m, r) in enumerate(g.iterrows()):
        ax.plot([r.new * 100, r.old * 100], [i, i], color=C["grid"], lw=2.6,
                solid_capstyle="round", zorder=2)
        ax.scatter(r.old * 100, i, s=22, color=C["sub"], zorder=3)
        ax.scatter(r.new * 100, i, s=42, color=C["vlm"], zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([fs.MODEL_SHORT.get(m, m)[:22] for m in g.index], fontsize=7.4)
    ax.set_xlim(0, 106)
    fs.axis(ax, "Narratives asserting an exit (%)", None)
    fs.title(ax, "The correction is not uniform across models", pad=10)
    fs.panel_label(ax, "b", dx=-0.02, dy=1.045)
    fs.despine(ax, left=False)
    # the left half of this panel is empty for every row, so the key goes there
    ax.scatter(6, 5.15, s=22, color=C["sub"])
    ax.annotate("withdrawn measure", xy=(10, 5.15), fontsize=6.9, color=C["sub"], va="center")
    ax.scatter(6, 4.55, s=42, color=C["vlm"])
    ax.annotate("corrected", xy=(10, 4.55), fontsize=6.9, color=C["vlm"],
                fontweight="bold", va="center")

    # ---- (c) the reversal: video is worse than frames, within model
    ax = fig.add_subplot(gs[1, 0])
    goog = d[d.model.str.startswith("gemini")]
    piv = goog.groupby(["model", "cond"]).transit.mean().unstack() * 100
    xs = {"B": 0, "A0": 1, "A": 2}
    for m, row in piv.iterrows():
        pts = [(xs[c], row[c]) for c in ("B", "A0", "A") if c in row and not np.isnan(row[c])]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", lw=1.5, ms=4.2,
                color=fs.MODEL.get(m, C["sub"]), zorder=3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Frames\n1 fps", "Video\nmuted", "Video\n+ audio"], fontsize=7.0)
    ax.set_xlim(-0.35, 2.55)
    ax.set_ylim(0, 108)
    fs.axis(ax, None, "Asserting an exit (%)")
    fs.title(ax, "Richer input makes it worse")
    fs.panel_label(ax, "c", dx=-0.19)
    fs.despine(ax)
    fs.faint_grid(ax)
    ax.annotate("each line is one model\nrun on all three inputs", xy=(0.04, 0.06),
                xycoords="axes fraction", fontsize=6.4, color=C["sub"], linespacing=1.3)

    fs.save(fig, "q1_fig3_script")
    print("  wrote q1_fig3_script")


# ================================================================== F4
def fig_audit():
    a = json.load(open(OUT / "audit_final.json", encoding="utf-8"))
    rates = sorted(a["exit_rate"].values())
    maj = [r for r in rates if r <= 0.5]
    dis = [r for r in rates if r > 0.5]

    fig = plt.figure(figsize=(7.4, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.55,
                          left=0.075, right=0.985, top=0.775, bottom=0.215)

    # ---- (a) the faction structure
    ax = fig.add_subplot(gs[0, 0])
    # Four coders sit at 0% and would print as one dot. Stack ties vertically so the panel
    # shows ten coders, which is the whole point of the panel.
    seen = collections.Counter()
    for r in rates:
        k = round(r * 100)
        dy = seen[k] * 0.16
        seen[k] += 1
        col = C["bad"] if r > 0.5 else C["engine"]
        ax.scatter(r * 100, dy, s=78, color=col, alpha=.88, zorder=3,
                   edgecolor="white", linewidth=0.9)
    ax.axvline(50, color=C["ink"], lw=0.8, ls=(0, (3, 2)))
    ax.set_ylim(-0.35, 1.15)
    ax.set_xlim(-6, 106)
    ax.set_yticks([])
    ax.annotate(f"{len(maj)} coders agree\nwith the engine", xy=(4, 0.30), fontsize=6.9,
                color=C["engine"], linespacing=1.3)
    ax.annotate(f"{len(dis)} read it as an exit", xy=(54, 0.90), fontsize=6.9,
                color=C["bad"], linespacing=1.3)
    fs.axis(ax, "Clips called an exit (%)", None)
    fs.title(ax, "Coders split into two groups")
    fs.panel_label(ax, "a", dx=-0.05, dy=1.06)
    fs.despine(ax, left=False)

    # ---- (b) corroboration holds inside both groups
    ax = fig.add_subplot(gs[0, 1])
    vals = [("Majority\ngroup", a["corroboration_majority"], C["engine"]),
            ("Dissenting\ngroup", a["corroboration_dissent"], C["bad"])]
    for i, (lab, v, col) in enumerate(vals):
        ax.bar(i, v * 100, width=.56, color=col, alpha=.85, zorder=3)
        ax.text(i, v * 100 + 2.5, f"{v * 100:.0f}%", ha="center", fontsize=8.2,
                fontweight="bold", color=col)
    ax.axhline(50, color=C["ink"], lw=0.8, ls=(0, (3, 2)), zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([v[0] for v in vals], fontsize=7.0)
    ax.set_ylim(0, 112)
    fs.axis(ax, None, "Judged wrong (%)")
    fs.title(ax, "Both groups agree the\nnarratives are wrong")
    fs.panel_label(ax, "b", dx=-0.24, dy=1.06)
    fs.despine(ax)
    fs.faint_grid(ax)

    # ---- (c) agreement, reported honestly
    ax = fig.add_subplot(gs[0, 2])
    ks = [("Ending\n(did they exit)", a["alpha_q1"]), ("Description\n(is it correct)", a["alpha_q2"])]
    for i, (lab, v) in enumerate(ks):
        ax.bar(i, v, width=.5, color=C["sub"], zorder=3)
        ax.text(i, v + (0.03 if v >= 0 else -0.05), f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.6, fontweight="bold", color=C["ink"])
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.axhspan(0.67, 1.0, color=C["good"], alpha=.10, zorder=1)
    ax.annotate("conventionally\nacceptable", xy=(0.5, 0.80), fontsize=6.3, ha="center",
                color=C["good"], linespacing=1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([k[0] for k in ks], fontsize=7.0)
    ax.set_ylim(-0.25, 1.0)
    fs.axis(ax, None, "Krippendorff's alpha")
    fs.title(ax, "Individual agreement is\nnear zero, and reported")
    fs.panel_label(ax, "c", dx=-0.26, dy=1.06)
    fs.despine(ax)

    fs.save(fig, "q1_fig4_audit", superseded=True)   # content now in the audit table
    print("  wrote q1_fig4_audit")


if __name__ == "__main__":
    fs.setup()
    fig_resolves()
    # fig_script_completion() superseded by fig_script_v2.py (filmstrip + tiles)
    fig_audit()
