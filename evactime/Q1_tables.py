"""Q1 journal tables. LaTeX, booktabs, TRB-style short captions.

T1  Corpus and reference channels
T2  (superseded) What the attention annotation resolves. Every number it carried now appears in
    the 'what it resolves' figure (flow ribbon, region against object, chance baseline), and
    IJDRS allows three tables, so the slot goes to the literature positioning table.
T3  (superseded) Resolution limit by model and input condition
T4  (superseded) Scene geometry: pairwise angular separation

T2 replaced an angular "resolution limit" table that was withdrawn after review: the angle is zero
by construction whenever the class is right, the error variable has sixteen atoms so the 80th
percentile sits on one of them, and the participant bootstrap printed [18.9, 18.9]. T3 and T4 are
still generated, into superseded/, because IJDRS allows three tables and the figures already carry
their content. T3's caption states the withdrawn claim and must not be reinstated.

None of these depends on the human audit. Writes to paper/tables/.
Run: py evactime/Q1_tables.py
"""
import json, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
DS = ROOT / "vlm dataset"
TAB = ROOT / "paper" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

EYE = np.array([2.356, 1.223, -1.387])
NAMES = {3: "Exit sign", 4: "Alarm (visual)", 10: "Alarm (audible)", 6: "Queue display",
         13: "Exit door", 12: "Doctor's door", 8: "NPC 1", 9: "NPC 2", 11: "Extinguisher"}
SHORT = {"gemini-3-flash-preview": "Gemini 3 Flash", "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
         "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "openai/gpt-5.2": "GPT-5.2",
         "anthropic/claude-opus-4.8": "Claude Opus 4.8",
         "anthropic/claude-sonnet-5": "Claude Sonnet 5",
         "qwen/qwen3-vl-235b-a22b-instruct": "Qwen3-VL 235B",
         "meta-llama/llama-4-maverick": "Llama 4 Maverick"}
COND = {"A": "Video with audio", "A0": "Video, muted", "B": "Frames at 1\\,fps"}
LABEL = {"look_at_alarm": "Alarm", "look_at_exit_sign": "Exit sign", "look_at_npc": "NPC"}


def ang(a, b, eye=EYE):
    va, vb = a - eye, b - eye
    return float(np.degrees(np.arccos(np.clip(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)),
                                              -1, 1))))


def boot_ci(df, col="sep_deg", stat=np.median, n=2000, seed=11):
    """Participant-clustered bootstrap. Resampling individual assertions understates the
    uncertainty because assertions from one participant are not independent, and with a
    discrete error scale it collapses the interval to a single value."""
    rng = np.random.default_rng(seed)
    groups = [g[col].values for _, g in df.groupby("pid")]
    k = len(groups)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        vals.append(stat(np.concatenate([groups[i] for i in idx])))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def wrap(body, caption, label, colspec, header, note=None):
    out = ["\\begin{table}[!ht]", "\\centering",
           f"\\caption{{{caption}}}\\label{{{label}}}",
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header, "\\midrule",
           body, "\\bottomrule", "\\end{tabular}"]
    if note:
        out.append(f"\\begin{{tablenotes}}\\footnotesize\\item {note}\\end{{tablenotes}}")
    out.append("\\end{table}")
    return "\n".join(out)


def t1():
    mk = pd.read_csv(DS / "participant_marker_times.csv", sep=";")
    h = pd.read_csv(DS / "participant_hitobject_timeseries.csv", sep=";")
    h["pid"] = "P" + h.Participant.str.extract(r"P0*(\d+)")[0]
    win = pd.read_csv(OUT / "asset_windows.csv")
    inc = win[win.include]
    c2 = pd.read_csv(OUT / "C2_resolution.csv")
    import sys
    sys.path.insert(0, str(ROOT / "evactime"))
    from vlm_records import load_records
    _, st, _ = load_records()
    # achieved reference rate, measured rather than quoted (the headset's nominal rate is higher)
    rates = []
    for _, g in h.groupby("pid"):
        dt = np.diff(np.sort(g.UnityTime.values))
        dt = dt[(dt > 0) & (dt < 5)]
        if len(dt):
            rates.append(1.0 / dt.mean())
    hz = float(np.median(rates))
    n = lambda v: f"{v:,}".replace(",", "{,}")
    import tabviz as tv
    _, _, per_model = load_records()
    ref = pd.read_csv(OUT / "reference_engine_all.csv")
    m = inc.merge(ref[["pid", "outro", "gaze_hz"]], on="pid")
    alarm = m.pad_before_alarm_s.values
    cut = (m.outro - m.window_start_s).values
    post = (m.window_end_s - m.outro).values
    gaze = m.gaze_hz.values
    # per-model usable share, in the order the Method lists the models
    order = ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite",
             "openai/gpt-5.2", "anthropic/claude-opus-4.8", "anthropic/claude-sonnet-5",
             "qwen/qwen3-vl-235b-a22b-instruct", "meta-llama/llama-4-maverick"]
    usable = [per_model[k]["kept"] / per_model[k]["on_disk"] for k in order]
    worst = min(zip(usable, order))
    typ = c2.type.value_counts()
    shares = [typ.get(t, 0) / len(c2) for t in ("look_at_alarm", "look_at_exit_sign", "look_at_npc")]
    nin, nall = len(inc), 31
    rows = [
        tv.group("Engine record, per recording", 3),
        f"Recordings analyzed & {nin} of {nall} & \\\\",
        f"Alarm and scene cut in the 75\\,s window & alarm {alarm.min():.0f}--{alarm.max():.0f}\\,s, "
        f"cut {cut.min():.0f}--{cut.max():.0f}\\,s & "
        + tv.rug([(alarm, "tvEmber", "alarm"), (cut, "tvInk", "cut")], 0, 75, ticks=(0, 25, 50, 75))
        + r" \\[3pt]",
        f"Footage after the cut & median {np.median(post):.1f}\\,s & "
        + tv.strip(post, 0, 60, ticks=(0, 30, 60), unit="s") + r" \\[3pt]",
        f"Gaze raycast & {n(len(h))} samples, median {hz:.0f}\\,Hz & "
        + tv.strip(gaze, 40, 55, ticks=(40, 45, 50, 55), unit="Hz") + r" \\[3pt]",
        "Tagged scene objects & 11 & \\\\",
        tv.group("Model annotation", 3),
        "Models and input formats & 8 models, 5 developers, 3 formats & \\\\",
        f"Usable calls, per model & {n(st['kept'])} of {n(st['calls_on_disk'])}, lowest "
        f"{100 * worst[0]:.0f}\\% & "
        + tv.strip([100 * u for u in usable], 70, 100, ticks=(70, 85, 100), unit=r"\%",
                   r_dot=0.04, median=False) + r" \\[9pt]",
        f"Attention assertions, by object named & {n(len(c2))} & "
        + tv.stack(list(zip(shares, ["alarm", "sign", "NPC"]))) + r" \\[6pt]",
    ]
    # ---- the human audit, merged here so the paper keeps three tables and gains the model table
    au = json.load(open(OUT / "audit_final.json", encoding="utf-8"))
    pc = lambda x: f"{100 * x:.0f}\\%"
    rates = sorted(au["exit_rate"].values())
    maj = [x for x in rates if x <= 0.5]
    dis = [x for x in rates if x > 0.5]
    lo, hi = au["ci"]
    (a1l, a1h), (a2l, a2h) = au["alpha_q1_ci"], au["alpha_q2_ci"]
    (ml, mh), (dl, dh) = au["corroboration_majority_ci"], au["corroboration_dissent_ci"]
    aref = [(0, ""), (0.667, "0.67")]
    P = dict(lo=0, hi=100, ticks=(0, 50, 100))
    rows += [
        tv.group("Human audit", 3),
        # 58 items were shown: 56 narratives and 2 attention checks. Two narratives on the clip of
        # the participant who never moved were dropped, which leaves the 54 that were scored.
        f"Coders and items & {au['n_coders']} coders, {au['n_items']} scored of 58 shown & \\\\",
        f"Doorway transit reported, per coder & {len(maj)} at {pc(min(maj))}--{pc(max(maj))}, "
        f"{len(dis)} at {pc(min(dis))}--{pc(max(dis))} & "
        + tv.strip([100 * x for x in rates], 0, 100, color="tvViolet", ticks=(0, 50, 100),
                   unit=r"\%", r_dot=0.04, median=False) + r" \\[3pt]",
        f"Krippendorff's $\\alpha$, ending & ${au['alpha_q1']:+.2f}$ [${a1l:+.2f}$, ${a1h:+.2f}$] & "
        + tv.ptci(au["alpha_q1"], a1l, a1h, -1, 1, color="tvViolet", refs=aref, ticks=(-1, 0, 1))
        + r" \\[4pt]",
        f"Krippendorff's $\\alpha$, description & ${au['alpha_q2']:+.2f}$ [${a2l:+.2f}$, ${a2h:+.2f}$] & "
        + tv.ptci(au["alpha_q2"], a2l, a2h, -1, 1, color="tvViolet", refs=aref, ticks=(-1, 0, 1))
        + r" \\[4pt]",
        f"Flagged items judged wrong, per item & {au['regex_transit_n']} items, median "
        f"{pc(float(np.nanmedian(au['item_share_wrong'])))} of coders & "
        + tv.hist([100 * x for x in au["item_share_wrong"]], 0, 100, 10, color="tvViolet",
                  ticks=(0, 50, 100)) + r" \\[3pt]",
        f"Judged wrong by majority, all coders & {pc(au['corroboration'])} [{pc(lo)}, {pc(hi)}] & "
        + tv.ptci(100 * au["corroboration"], 100 * lo, 100 * hi, **P, color="tvViolet") + r" \\[3pt]",
        f"\\quad majority group, {len(maj)} coders & {pc(au['corroboration_majority'])} "
        f"[{pc(ml)}, {pc(mh)}] & "
        + tv.ptci(100 * au["corroboration_majority"], 100 * ml, 100 * mh, **P) + r" \\[3pt]",
        f"\\quad dissenting group, {len(dis)} coders & {pc(au['corroboration_dissent'])} "
        f"[{pc(dl)}, {pc(dh)}] & "
        + tv.ptci(100 * au["corroboration_dissent"], 100 * dl, 100 * dh, **P) + r" \\",
    ]
    return "\n".join([
        "% generated by evactime/Q1_tables.py; do not edit",
        tv.preamble(),
        r"\begin{table}[!ht]", r"\centering\footnotesize", r"\begin{threeparttable}",
        r"\caption{Data, Reference Channels and Human Audit}\label{tab:corpus}",
        r"\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.3}",
        r"\begin{tabular}{@{}l l l@{}}", r"\toprule",
        r"\textbf{Item} & \textbf{Value [95\% interval]} & \textbf{Distribution} \\", r"\midrule",
        "\n".join(rows), r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]\scriptsize",
        r"\item[] \textit{Note:} Dots are recordings, models or coders; bars mark the median; whiskers are 95\% "
        r"intervals bootstrapped over coders. The dotted line at 0.67 is the lowest value of "
        r"$\alpha$ Krippendorff accepts for tentative conclusions.",
        r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"])


def t2(d):
    """What the annotation resolves. Replaces the withdrawn angular-error table.

    The previous version reported a median angular error with a bootstrapped CI and an 80th
    percentile called a resolution limit. It was withdrawn for three reasons, all confirmed
    against the data: the angle is zero by construction whenever the class is right, the variable
    has sixteen atoms so the 80th percentile sits on one of them, and the CI printed as
    [18.9, 18.9] because resampling participants cannot move a median off an atom. A degenerate
    interval in a published table is the most visible symptom and would be the first thing a
    reviewer asked about.
    """
    import json
    c = json.load(open(OUT / "C2_confusion.json", encoding="utf-8"))
    conf = pd.read_csv(OUT / "C2_confusion.csv")
    order = ["look_at_npc", "look_at_alarm", "look_at_exit_sign"]
    rows = []
    for t in order:
        g = conf[conf.type == t]
        wrong = g[~g.correct].actual_cls.value_counts()
        top = ", ".join(f"{k} ({v})" for k, v in wrong.head(2).items()) or "---"
        rows.append(f"{LABEL[t]} & {len(g)} & {g.correct.mean() * 100:.1f} & "
                    f"{g.region_hit.mean() * 100:.1f} & {top}")
    rows.append("\\midrule\nAll & %d & %.1f & %.1f & "
                % (len(conf), 100 * c["exact"], 100 * c["region"])
                + "\\textit{chance %.1f [%.1f, %.1f]}"
                % (100 * c["chance"], 100 * c["chance_ci"][0], 100 * c["chance_ci"][1]))
    body = " \\\\\n".join(rows) + " \\\\"
    return wrap(body, "What the Attention Annotation Resolves", "tab:resolution",
                "lrrrl",
                "Assertion & $n$ & Exact object (\\%) & Right region (\\%) & "
                "Confused with \\\\",
                "Exact object is agreement with the object the reference records. Right region "
                "scores a wall-cue claim as correct when gaze was on any back-wall object (exit "
                "sign, both alarm components, queue display, extinguisher, exit door) and an NPC "
                "claim as correct when gaze was on either NPC; the doctor's door is in neither "
                "group. The chance "
                "row permutes each participant's claims against their own reference, preserving "
                "both marginal distributions and destroying only the association "
                "($p<0.001$). The NPCs are a positive control: free-standing objects are named "
                "correctly, cues mounted together on one wall are not.")


def t3(d):
    rows = []
    for m, g in sorted(d.groupby("model"), key=lambda kv: kv[1].sep_deg.quantile(.8)):
        rows.append(f"{SHORT.get(m, m)} & {len(g)} & {g.sep_deg.median():.1f} & "
                    f"{g.sep_deg.quantile(.8):.1f} & {g.correct.mean() * 100:.1f}")
    rows.append("\\midrule")
    for c, g in d.groupby("cond"):
        rows.append(f"\\textit{{{COND[c]}}} & {len(g)} & {g.sep_deg.median():.1f} & "
                    f"{g.sep_deg.quantile(.8):.1f} & {g.correct.mean() * 100:.1f}")
    body = " \\\\\n".join(rows) + " \\\\"
    return wrap(body, "The Limit Does Not Move With the Model or the Input", "tab:bymodel",
                "lrrrr",
                "Model / condition & $n$ & Median (\\textdegree) & 80th pct & Correct (\\%) \\\\",
                "Eight models from five vendors and three input formats give an 80th percentile "
                "between 19.0 and 20.1\\textdegree, so the limit is a property of the scene "
                "rather than of the annotator.")


def t4():
    g = json.load(open(OUT / "scene_geometry.json"))
    pos = {int(k): np.array(v["pos"]) for k, v in g.items() if v.get("pos")}
    # NOTE: 6_CanvasQueueDisplay used to be hand-patched here because S1 read a
    # RectTransform's m_LocalPosition instead of its m_AnchoredPosition and put the object
    # at the room origin. S1 now resolves it, so the released geometry and this analysis
    # agree and the paper reproduces from the deposited artefacts.
    ks = [3, 4, 10, 6, 13, 11, 9, 12, 8]
    hdr = "Object & " + " & ".join(f"{i}" for i in range(1, len(ks) + 1)) + " \\\\"
    rows = []
    for i, oi in enumerate(ks, 1):
        cells = []
        for oj in ks:
            cells.append("--" if oi == oj else f"{ang(pos[oi], pos[oj]):.0f}")
        rows.append(f"{i} {NAMES[oi]} & " + " & ".join(cells))
    body = " \\\\\n".join(rows) + " \\\\"
    return wrap(body, "Pairwise Angular Separation of Tagged Objects", "tab:geometry",
                "l" + "r" * len(ks), hdr,
                "Degrees, from the seated viewpoint. The first five objects lie within "
                "20\\textdegree\\ of one another, below the measured resolution limit.")


if __name__ == "__main__":
    d = pd.read_csv(OUT / "C2_resolution.csv")
    # IJDRS allows three tables. t3 and t4 are still generated, but into superseded/, because
    # Figure 3 already shows the per-model p80 and Figure 1 already shows the angular-separation
    # matrix. Writing them to the main folder would break the venue check in Q1_numbers_audit.py.
    SUP = TAB / "superseded"
    SUP.mkdir(exist_ok=True)
    for name, tex, dst in [("q1_tab_corpus", t1(), TAB), ("q1_tab_resolution", t2(d), SUP),
                           ("q1_tab_audit_placeholder", None, None),
                           ("q1_tab_bymodel", t3(d), SUP), ("q1_tab_geometry", t4(), SUP)]:
        if tex is None:
            continue
        (dst / f"{name}.tex").write_text(tex, encoding="utf-8")
        print("wrote", dst / f"{name}.tex")
    print("note: q1_tab_audit.tex is built by Q1_table_audit.py, which needs the Prolific data")
    print("\n--- T2 preview ---\n")
    print(t2(d))
