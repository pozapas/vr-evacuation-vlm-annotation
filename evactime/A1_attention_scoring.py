"""A1 - score VLM `look_at_*` assertions against the eye-tracking reference, cohort-wide.

It uses only the 1,105 parsed calls on disk and the
Zenodo deposit (`vlm dataset/`). No new model calls.

Reference channel is the `hitObject` series. Per the locked scope, it is used ONLY to score model
assertions - never to report where attention went.

Clock chain (the part that is easy to get wrong):
    assertion t   : seconds from the start of the 75 s analysis clip
    video time    : window_start_s + t              (video t=0 == 'Scene changed to WaitingRoom')
    Unity time    : video time + WaitingRoom marker  <- the clock the hitObject series uses

Outputs (evactime/outputs/):
    A1_assertions_scored.csv   one row per timestamped look_at_* assertion
    A1_summary.csv             agreement by event type, condition and tolerance
    A1_support.csv             reference support per super-class (denominators only)
    A1_report.txt              the readable summary
"""
import json, pathlib, collections
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
DS = ROOT / "vlm dataset"

# hitObject id -> semantic super-class (locked in the implementation plan, PART 2)
SUPER = {-1: "none", 0: "none", 1: "misc", 2: "misc", 3: "sign", 4: "alarm", 10: "alarm",
         5: "door", 7: "door", 12: "door", 13: "door", 6: "display", 11: "equipment",
         8: "npc", 9: "npc"}

# which super-class each assertion type claims
CLAIM = {"look_at_alarm": "alarm", "look_at_exit_sign": "sign", "look_at_npc": "npc"}

TOLERANCES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)   # 1 and 2 s are reported; the rest trace the curve


def events(parsed):
    """The runners emitted events/optional_events as either a list or a type-keyed dict."""
    out = []
    for key in ("events", "optional_events"):
        v = parsed.get(key)
        if isinstance(v, list):
            out += [e for e in v if isinstance(e, dict)]
        elif isinstance(v, dict):
            out += [e for e in v.values() if isinstance(e, dict)]
    return out


def load_assertions():
    """Assertions from the same 1,164 usable calls as every other analysis.

    Review, 25 Sep 2026: this loader used to read the raw files and skip any call whose stored
    `parsed` field was not a dict, so the 62 calls that vlm_records recovers with raw_decode (55
    of them Gemini 3.1 Pro) entered the narrative analysis but not the attention analysis. One
    parsing path now serves both.
    """
    from vlm_records import load_records
    recs, _, _ = load_records()
    rows = []
    for d in recs:
        p = d["parsed"]
        if True:
            for e in events(p):
                t = e.get("type", "")
                if t in CLAIM:
                    # Condition B models emit `frame`, not `t`. frame_hz is 1.0, so the frame
                    # index IS seconds from clip start. Missing this silently drops ~920
                    # assertions and makes the timestamp yield look far worse than it is.
                    tt = e.get("t")
                    if not isinstance(tt, (int, float)):
                        tt = e.get("frame")
                    rows.append(dict(model=d["model"], pid=d["pid"], cond=d["cond"], rep=d["rep"],
                                     type=t, t=tt, t_source="t" if e.get("t") is not None else
                                     ("frame" if e.get("frame") is not None else "none"),
                                     conf=e.get("confidence")))
    return pd.DataFrame(rows)


def load_reference():
    h = pd.read_csv(DS / "participant_hitobject_timeseries.csv", sep=";")
    h["pid"] = "P" + h.Participant.str.extract(r"P0*(\d+)")[0]
    h["cls"] = h.HitObjectID.map(SUPER).fillna("misc")
    return h[["pid", "UnityTime", "HitObjectID", "cls"]].sort_values(["pid", "UnityTime"])


def main():
    win = pd.read_csv(OUT / "asset_windows.csv")
    mk = pd.read_csv(DS / "participant_marker_times.csv", sep=";")
    mk["pid"] = "P" + mk.Participant.str.extract(r"P0*(\d+)")[0]

    # clip -> Unity offset, per participant
    off = (win[["pid", "include", "window_start_s"]]
           .merge(mk[["pid", "WaitingRoom", "ALARM_TRIGGERED", "OutroScene"]], on="pid"))
    off["clip_to_unity"] = off.window_start_s + off.WaitingRoom
    included = set(off.loc[off.include, "pid"])

    a = load_assertions()
    n_all = len(a)
    a = a[a.pid.isin(included)].copy()
    a["timed"] = a.t.apply(lambda x: isinstance(x, (int, float)) and np.isfinite(x))

    ref = load_reference()
    ref_by_pid = {p: g for p, g in ref.groupby("pid")}

    a = a.merge(off[["pid", "clip_to_unity", "ALARM_TRIGGERED", "OutroScene"]], on="pid", how="left")
    a["t_unity"] = a.t.where(a.timed) + a.clip_to_unity

    # Score each timed assertion against the reference within +/- tol, THREE ways. The three
    # differ in how permissive they are, and reporting only the first overstated agreement.
    #
    #   hit_<tol>s        ANY sample in the window carries the claimed class.
    #   dwell_<tol>s      the FRACTION of samples carrying it.
    #   modal_<tol>s      the class the participant looked at MOST in the window is the claimed one.
    #
    # `any` is an existence test, not a measurement. The reference runs at ~47 Hz, so a +/-1 s
    # window holds about 94 samples and a single spurious sample scores agreement. The tell is
    # that agreement rises monotonically as the tolerance widens, which is what an existence test
    # over a growing window does regardless of whether the claim is right. The modal scoring is
    # the one comparable to C2, which already resolves the reference by dwell.
    for tol in TOLERANCES:
        hit, dwell, modal = [], [], []
        for _, r in a.iterrows():
            if not r.timed or ref_by_pid.get(r.pid) is None:
                hit.append(np.nan); dwell.append(np.nan); modal.append(np.nan); continue
            g = ref_by_pid[r.pid]
            w = g[(g.UnityTime >= r.t_unity - tol) & (g.UnityTime <= r.t_unity + tol)]
            if not len(w):
                # An empty window is a gap in the REFERENCE, not a wrong answer by the model.
                # Scoring it False charged the logger's silence to the model; P005 alone has
                # 36.4 s of dead time. It is now excluded from the denominator.
                hit.append(np.nan); dwell.append(np.nan); modal.append(np.nan); continue
            want = CLAIM[r.type]
            share = float((w.cls == want).mean())
            hit.append(bool(share > 0))
            dwell.append(share)
            top = w.cls.mode()
            modal.append(bool(len(top) and top.iloc[0] == want))
        a[f"hit_{tol:g}s"] = hit
        a[f"dwell_{tol:g}s"] = dwell
        a[f"modal_{tol:g}s"] = modal

    # is the assertion inside the evacuation window, or in the pre-alarm padding?
    a["in_evac"] = (a.t_unity >= a.ALARM_TRIGGERED) & (a.t_unity <= a.OutroScene)
    a["phase"] = np.where(a.t_unity < a.ALARM_TRIGGERED, "pre-alarm",
                 np.where(a.t_unity <= a.OutroScene, "evacuation", "post-outro"))
    a.to_csv(OUT / "A1_assertions_scored.csv", index=False)

    # Coverage (review, 25 Sep 2026). The agreement scores are conditional on a model choosing to
    # report an attention event, so they are precision-like. How often each model reports one at
    # all is written here, per usable call in the cohort, and reported beside the agreement.
    from vlm_records import load_records
    recs, _, _ = load_records()
    calls = pd.DataFrame([dict(pid=r["pid"], model=r["model"], cond=r["cond"], rep=r["rep"])
                          for r in recs if r["pid"] in included])
    cnt = (a[a.timed].groupby(["pid", "model", "cond", "rep"]).size()
           .rename("n_timed").reset_index())
    calls = calls.merge(cnt, on=["pid", "model", "cond", "rep"], how="left").fillna({"n_timed": 0})
    calls.to_csv(OUT / "A1_calls.csv", index=False)

    # ---- summaries
    lines = []
    P = lines.append
    P("A1 - VLM attention assertions vs the eye-tracking reference")
    P("=" * 66)
    P(f"cohort: {len(included)} participants (frozen asset set; P2 and P22 excluded)")
    P(f"assertions: {n_all} total, {len(a)} in cohort, {int(a.timed.sum())} carry a timestamp "
      f"({a.timed.mean():.1%})")
    P("")
    P("TIMESTAMP YIELD - the model names an attention event without locating it")
    P(f"  {'type':22s}{'asserted':>10}{'timed':>8}{'yield':>8}")
    for t, g in a.groupby("type"):
        P(f"  {t:22s}{len(g):10d}{int(g.timed.sum()):8d}{g.timed.mean():8.1%}")
    P("")
    P("AGREEMENT - of timed assertions, share where the reference shows the claimed class")
    P(f"  {'type':22s}{'cond':>6}{'n':>6}{'+/-1s':>9}{'+/-2s':>9}")
    rows = []
    for (t, c), g in a[a.timed].groupby(["type", "cond"]):
        h1, h2 = g["hit_1s"].mean(), g["hit_2s"].mean()
        P(f"  {t:22s}{c:>6}{len(g):6d}{h1:9.1%}{h2:9.1%}")
        rows.append(dict(type=t, cond=c, n=len(g), hit_1s=h1, hit_2s=h2))
    P("")
    P(f"  {'type':22s}{'ALL':>6}{'n':>6}{'+/-1s':>9}{'+/-2s':>9}")
    for t, g in a[a.timed].groupby("type"):
        h1, h2 = g["hit_1s"].mean(), g["hit_2s"].mean()
        P(f"  {t:22s}{'-':>6}{len(g):6d}{h1:9.1%}{h2:9.1%}")
        rows.append(dict(type=t, cond="ALL", n=len(g), hit_1s=h1, hit_2s=h2))
    pd.DataFrame(rows).to_csv(OUT / "A1_summary.csv", index=False)

    # ---- reference support (denominators only, never reported as behaviour)
    sup = []
    for pid, g in ref.groupby("pid"):
        if pid not in included:
            continue
        o = off[off.pid == pid].iloc[0]
        w = g[(g.UnityTime >= o.ALARM_TRIGGERED) & (g.UnityTime <= o.OutroScene)]
        for cls, gg in w.groupby("cls"):
            sup.append(dict(pid=pid, cls=cls, n_samples=len(gg)))
    sup = pd.DataFrame(sup)
    tot = (sup.groupby("cls").agg(n_samples=("n_samples", "sum"),
                                  n_participants=("pid", "nunique")).sort_values("n_samples",
                                                                                 ascending=False))
    sup.to_csv(OUT / "A1_support.csv", index=False)
    P("")
    P("PHASE SPLIT - where the assertions actually fall")
    P(f"  {'type':22s}{'pre-alarm':>11}{'evacuation':>12}{'post-outro':>12}")
    for t, g in a[a.timed].groupby("type"):
        v = g.phase.value_counts()
        P(f"  {t:22s}{v.get('pre-alarm',0):11d}{v.get('evacuation',0):12d}{v.get('post-outro',0):12d}")
    P("")
    # ---- how much of the agreement is an artefact of the scoring rule
    t = a[a.timed == True]
    P("SCORING RULE MATTERS MORE THAN THE TOLERANCE")
    P(f"  {'assertion':20s}{'n':>6}{'any 1s':>9}{'any 2s':>9}{'modal 1s':>10}{'modal 2s':>10}")
    for ty, g in t.groupby("type"):
        P(f"  {ty:20s}{int(g['hit_1s'].notna().sum()):6d}{g['hit_1s'].mean():9.1%}"
          f"{g['hit_2s'].mean():9.1%}{g['modal_1s'].mean():10.1%}{g['modal_2s'].mean():10.1%}")
    P(f"  {'ALL':20s}{int(t['hit_1s'].notna().sum()):6d}{t['hit_1s'].mean():9.1%}"
      f"{t['hit_2s'].mean():9.1%}{t['modal_1s'].mean():10.1%}{t['modal_2s'].mean():10.1%}")
    P("")
    P(f"  widening the window from 1 s to 2 s moves 'any' by "
      f"{t['hit_2s'].mean() - t['hit_1s'].mean():+.1%} and 'modal' by "
      f"{t['modal_2s'].mean() - t['modal_1s'].mean():+.1%}.")
    P("  A statistic that improves as the tolerance loosens is an existence test over a growing")
    P("  window, not a measurement converging on a value. The reference runs at about 47 Hz, so a")
    P("  +/-1 s window holds roughly 94 samples and one spurious sample scores agreement under")
    P("  'any'. MODAL is the number to report: it asks what the participant looked at MOST, which")
    P("  is the same question C2 asks, so the two analyses are finally comparable.")
    P(f"  {int(a[a.timed == True]['hit_1s'].isna().sum())} assertions fall in a reference gap and")
    P("  are excluded rather than scored as model error; P005 alone has 36.4 s of dead time.")
    P("")
    P("AGREEMENT BY PHASE (+/-1s) - the pre-alarm padding is not the evacuation")
    P(f"  {'type':22s}{'phase':>12}{'n':>6}{'+/-1s':>9}")
    for (t, ph), g in a[a.timed].groupby(["type", "phase"]):
        P(f"  {t:22s}{ph:>12}{len(g):6d}{g['hit_1s'].mean():9.1%}")
    P("")
    P("REFERENCE SUPPORT in the evacuation window (denominators for C2/C4, not findings)")
    P(tot.to_string())

    txt = "\n".join(lines)
    (OUT / "A1_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
