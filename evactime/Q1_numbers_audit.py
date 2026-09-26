"""Recompute every numeric claim the Q1 paper makes, from the artefacts.

The TRB paper's credibility came from an audit of this kind: 132 checks, 0 failures. This is the
same instrument for the journal paper. It exists because hand-carried numbers drift. Three separate
values for the post-cut footage duration (38.2, 38.5, 38.6) were in circulation before this script
was written, one of them hardcoded in a docstring, and nothing would have caught that.

Rules this enforces:
  - one denominator convention, read from config.yaml, never typed into prose;
  - every claim traceable to an artefact, not to a previous draft;
  - any figure-generator convention (filters, exclusions, aggregation level) read out of the
    generator rather than assumed.

Run: py evactime/Q1_numbers_audit.py
Exit code is non-zero if any check fails, so it can gate a build.
"""
import json, pathlib, re, sys
import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
CFG = yaml.safe_load(open(ROOT / "evactime" / "config.yaml", encoding="utf-8"))

CHECKS = []


def check(claim, got, want, tol=0.0):
    """Record one claim. `want` may be a number, a string, or a set/list for membership."""
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    CHECKS.append((ok, claim, got, want))
    return ok


# ----------------------------------------------------------------- corpus
def corpus():
    c = CFG["corpus"]
    w = pd.read_csv(OUT / "asset_windows.csv")
    inc = sorted(w[w.include].pid, key=lambda p: int(p[1:]))
    check("corpus: participants recorded", 31, c["n_participants_recorded"])
    check("corpus: participants analysed", len(inc), c["n_participants_analysed"])
    check("corpus: excluded set", sorted(w[~w.include].pid), sorted(c["excluded"]))
    check("corpus: config list matches artefact", inc, list(c["participants"]))
    # the trap: the companion EEG cohort is also 29 but a different set
    eng = pd.read_csv(OUT / "engine_raw.csv")
    eng = set(eng[eng.movement_unity.notna()].pid) - {"P22"}
    check("corpus: EEG cohort is a DIFFERENT 29", len(eng), 29)
    check("corpus: EEG cohort includes P2 (paper's does not)", "P2" in eng, True)


# ------------------------------------------------------------- resolution
def resolution():
    d = pd.read_csv(OUT / "C2_resolution.csv")
    check("C2: assertion count", len(d), CFG["resolution"]["n_assertions"])

    # ---- the claim now made: region resolved, object not, and better than chance.
    c = json.load(open(OUT / "C2_confusion.json", encoding="utf-8"))
    check("C2: exact-object accuracy", round(100 * c["exact"], 1), 24.2)
    check("C2: right-region accuracy", round(100 * c["region"], 1), 91.8)
    check("C2: object chance as written", round(100 * c["chance"], 1), 17.6)
    check("C2: region is resolved far better than object",
          bool(c["region"] - c["exact"] > 0.50), True)
    check("C2: accuracy exceeds the permutation baseline",
          bool(c["exact"] > c["chance_ci"][1]), True)
    check("C2: baseline test is significant", bool(c["p"] < 0.01), True)
    # the positive control has to behave, or the contrast means nothing
    check("C2: NPC positive control above 80%", bool(c["by_type"]["look_at_npc"] > 0.80), True)
    check("C2: exit-sign naming near zero", bool(c["by_type"]["look_at_exit_sign"] < 0.05), True)
    # The region result is only a finding against its own null: gaze sits on the back wall in
    # most windows, so region agreement is high even for a claim made without looking.
    check("C2: region chance reported", round(100 * c["region_chance"], 1), 80.9)
    check("C2: region beats its own chance", bool(c["region"] > c["region_chance_ci"][1]), True)
    check("C2: chance-corrected region far above chance-corrected object",
          bool(c["kappa_region"] > 4 * c["kappa_object"]), True)
    check("C2: chance-corrected values as written",
          (round(100 * c["kappa_object"], 1), round(100 * c["kappa_region"], 1)), (8.0, 57.0))
    check("C2: back-wall share as written", round(100 * c["backwall_share"], 1), 82.3)
    # review, 25 Sep 2026: participant-bootstrap intervals, per-target and per-model chance
    p1 = lambda x: round(100 * x, 1)
    b = c["boot_ci"]
    check("C2: bootstrap intervals as written",
          ([p1(x) for x in b["object"]], [p1(x) for x in b["kappa_object"]],
           [p1(x) for x in b["kappa_region"]]), ([18.3, 30.1], [4.0, 12.1], [37.5, 72.5]))
    s = c["sensitivity"]
    check("C2: sensitivity nulls bracket as written",
          (p1(s["call"]["kappa_object"]), p1(s["participant"]["kappa_object"]),
           p1(s["call"]["kappa_region"]), p1(s["participant"]["kappa_region"])),
          (6.8, 9.7, 53.4, 63.2))
    t = c["per_target"]
    check("C2: characters against own chance as written",
          (t["npc"]["n"], p1(t["npc"]["observed"]), p1(t["npc"]["chance"]), p1(t["npc"]["kappa"])),
          (154, 88.3, 39.9, 80.5))
    check("C2: alarm against own chance as written",
          (t["alarm"]["n"], p1(t["alarm"]["observed"]), p1(t["alarm"]["chance"])), (705, 26.2, 24.2))
    check("C2: exit sign at chance as written",
          (t["sign"]["n"], p1(t["sign"]["observed"]), p1(t["sign"]["chance"]), round(t["sign"]["p"], 2)),
          (505, 1.8, 1.6, 0.44))
    rk = lambda v: p1((v["region"] - v["region_chance"]) / (1 - v["region_chance"]))
    check("C2: wall-cue region kappa as written", (rk(t["alarm"]), rk(t["sign"])), (48.9, 36.7))
    check("C2: wall cues together as written", p1(c["wall"]["kappa"]), 1.5)
    check("C2: correct matches from characters", (c["npc_correct"], c["n_correct"]), (136, 330))
    ex = d[(d.type == "look_at_exit_sign") & ~d.correct].actual.value_counts()
    check("C2: exit-sign confusions as written",
          (int(ex.get(13, 0)), int(ex.get(4, 0) + ex.get(10, 0)), int(ex.get(6, 0))), (193, 130, 118))
    pm = c["per_model"]
    check("C2: coverage range as written",
          (p1(min(v["coverage"] for v in pm.values())),
           sum(v["coverage"] == 1.0 for v in pm.values())), (19.5, 3))
    check("C2: per-model object kappa range as written",
          (p1(min(v["kappa_object"] for v in pm.values())),
           p1(max(v["kappa_object"] for v in pm.values()))), (-0.8, 19.6))
    zero = [m for m, v in pm.items() if v["npc_claim_share"] == 0]
    check("C2: four models name only the back wall, region kappa zero",
          (len(zero), all(abs(pm[m]["kappa_region"]) < 1e-9 for m in zero)), (4, True))
    check("C2: region kappa range of the other four as written",
          (p1(min(v["kappa_region"] for m, v in pm.items() if m not in zero)),
           p1(max(v["kappa_region"] for m, v in pm.items() if m not in zero))), (31.0, 74.7))
    a1 = pd.read_csv(OUT / "A1_assertions_scored.csv")
    check("A1: timed attention assertions", int(a1.timed.sum()), 1466)

    # ---- guards on the WITHDRAWN framing. These exist so the resolution-limit claim cannot be
    # reintroduced without failing the build. Internal review showed that the angular
    # quantile is a scene constant: sep is 0 whenever the class is right and a fixed room distance
    # whenever it is wrong, so no accuracy-versus-angle curve exists.
    correct_max = float(d[d.correct].sep_deg.max())
    wrong_min = float(d[~d.correct].sep_deg.min())
    check("C2: sep is zero by construction when the class is right",
          bool(correct_max < 1e-4), True)
    check("C2: sep jumps straight to a room distance when wrong", bool(wrong_min > 5.0), True)
    check("C2: the error variable is a handful of atoms, not a continuum",
          bool(d.sep_deg.round(2).nunique() <= 20), True)
    check("C2: no scene object pair lies between 0 and 6 degrees",
          int(((d.sep_deg > 0.01) & (d.sep_deg < 6.0)).sum()), 0)


# -------------------------------------------------------- script completion
def script_completion():
    d = pd.read_csv(OUT / "C4_script_completion.csv")
    check("C4: narrative count", len(d), 1164)
    sm = json.load(open(OUT / "C4_summary.json", encoding="utf-8"))
    check("C4: frames-only rate as written",
          (sm["frames_transit"], sm["frames_n"], round(100 * sm["frames_rate"], 1)), (453, 691, 65.6))
    check("C4: scene-change language as written",
          (sm["scene_change_language"], sm["scene_change_language"] - sm["scene_change_without_exit"],
           sm["scene_change_without_exit"], round(100 * sm["scene_change_without_exit"] / sm["n"], 1)),
          (332, 215, 117, 10.1))

    # ---- VALIDITY, not consistency. The previous version of this file asserted
    # `transit.mean() == 92.0`, which is the number it was supposed to test, and it passed while
    # the classifier was counting the noun phrase "exit door". These checks interrogate the
    # instrument instead, so a regression in the classifier fails the build.
    sys.path.insert(0, str(ROOT / "evactime"))
    from transit_classifier import asserts_transit, run_self_test
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        failures = run_self_test()
    check("C4: transit classifier passes its own unit tests", failures, 0)
    check("C4: the object's NAME is not a transit",
          asserts_transit("The participant looks at the exit door."), False)
    check("C4: a negated exit is not a transit",
          asserts_transit("The participant does not exit the room."), False)
    check("C4: an NPC leaving is not the participant leaving",
          asserts_transit("An NPC exits the room. The participant stays seated."), False)
    check("C4: moving inside the room is not a transit",
          asserts_transit("She walks through the room toward the door."), False)
    check("C4: a real transit is still detected",
          asserts_transit("They open the door and walk through into the corridor."), True)
    check("C4: an accurate reset is not a transit",
          asserts_transit("The view resets to a fresh view of the waiting room."), False)

    # ---- development check: agreement with the human audit labels. These labels were used while
    # the rules were written (review, 25 Sep 2026), so this is a development result, not a
    # validation; the blind test is S3_blind_validation_sheet.py.
    key = {k["id"]: k for k in json.load(open(OUT / "audit_item_key.json", encoding="utf-8"))}
    a = pd.read_csv(OUT / "audit_final.csv")
    a = a[a.verdict_ending.isin(["correct", "not correct"])]
    hum = (a.verdict_ending == "not correct").to_numpy(dtype=bool)
    got = np.array([asserts_transit(key[i]["narrative"]) for i in a.id])
    fp = int((got & ~hum).sum())
    recall = float((got & hum).sum() / max(hum.sum(), 1))
    check("C4: no narrative humans call correct is flagged as a transit", fp, 0)
    check("C4: recall against human labels at least 0.80", bool(recall >= 0.80), True)

    # ---- the corrected headline, and the direction of the condition effect
    rate = round(100 * d.transit.mean(), 1)
    check("C4: corrected transit rate", rate, 74.7, tol=0.05)
    check("C4: transit count as written", int(d.transit.sum()), 870)
    goog = d[d.model.str.startswith("gemini")].groupby("cond").transit.mean()
    check("C4: native Gemini by format as written", tuple(round(100 * goog[c], 1) for c in ("B", "A0", "A")), (48.8, 88.2, 88.1))
    check("C4: corrected rate is far below the withdrawn 92%", bool(rate < 80), True)
    check("C4: actual ending described", round(100 * d.actual.mean(), 1), 28.5, tol=0.15)
    by = d.groupby("cond").transit.mean()
    check("C4: video input yields MORE script completion than frames",
          bool(min(by["A"], by["A0"]) > by["B"] + 0.15), True)
    bym = d.groupby("model").transit.mean()
    check("C4: n models", len(bym), 8)
    check("C4: at least six of eight models above 60%", int((bym > 0.60).sum()) >= 6, True)

    # ---- the task description ASSERTS the exit. It tells every model that the participant "leaves
    # the room, which ends the waiting-room scene", and makes exit_reached a required field. The
    # narrative rate therefore measures deference to the task description against the visual
    # evidence, not invention from nowhere, and the paper must say so. These checks keep any draft
    # from describing the prompt as neutral, and pin the evidence that the prompt cannot explain
    # the finding: it is constant, yet the free-text rate spans 14% to 99% across models.
    prompt = (ROOT / "evactime" / "prompts" / "common_task_v2.md").read_text(encoding="utf-8")
    check("C4: prompt asserts the participant leaves the room",
          "leaves the room" in prompt and "passes through the exit door" in prompt, True)
    fv = pd.read_csv(OUT / "C4_field_vs_narrative.csv")
    check("C4: every model filled the mandatory exit field", bool(fv.field_given.all()), True)
    per_model = fv.groupby("model").narr.mean()
    check("C4: free-text exit rate spans a wide range under one constant prompt",
          bool(per_model.max() - per_model.min() > 0.6), True)
    declined = float((fv.field_given & ~fv.narr).mean())
    check("C4: a quarter of calls fill the field yet decline the exit in their own words",
          bool(0.20 < declined < 0.32), True)

    # ---- call retention. 116 calls were previously dropped in silence, 114 of them from one
    # model, which made the per-model comparison meaningless. Recovery is now part of the loader
    # and the shortfall is reported rather than hidden.
    from vlm_records import load_records
    _, st_, per = load_records()
    check("C4: calls on disk", st_["calls_on_disk"], 1218)
    check("C4: usable calls after recovery", st_["kept"], 1164)
    check("C4: retention at least 95%", bool(st_["kept"] / st_["calls_on_disk"] >= 0.95), True)
    worst = min(per[m]["kept"] / per[m]["on_disk"] for m in per)
    check("C4: no model below 75% retention", bool(worst >= 0.75), True)
    check("C4: remaining losses are API errors, not parse failures",
          bool(st_["api_error"] > st_["unparseable"] * 10), True)
    # the post-cut duration, computed not quoted
    sys.path.insert(0, str(ROOT / "evactime"))
    from C4_script_completion import post_cut_seconds
    med, inside, n = post_cut_seconds()
    check("C4: cut inside window for all of the cohort", inside, n)
    check("C4: cohort size for post-cut stat", n, CFG["corpus"]["n_participants_analysed"])
    check("C4: median post-cut footage", round(med, 1), 38.5, tol=0.05)
    check("C4: models saw ~10x the coders' 4 s", round(med / 4.0) == 10, True)


# ------------------------------------------------------------------ audit
def audit():
    a = json.load(open(OUT / "audit_final.json", encoding="utf-8"))
    c = CFG["audit"]
    check("audit: coders retained", a["n_coders"], c["n_coders_retained"])
    check("audit: items scored", a["n_items"], c["n_items_scored"])
    check("audit: dropped items", sorted(a["dropped_items"]), sorted(c["dropped_items"]))
    check("audit: one coder excluded on attention checks", len(a["excluded_attention"]), 1)
    check("audit: alpha Q1 binary", round(a["alpha_q1"], 3), c["alpha_q1_binary"], tol=0.0005)
    check("audit: alpha Q2", round(a["alpha_q2"], 3), c["alpha_q2"], tol=0.0005)
    check("audit: alpha is near zero, not acceptable agreement",
          bool(abs(a["alpha_q1"]) < 0.10), True)
    # faction structure
    rates = sorted(a["exit_rate"].values())
    maj = [x for x in rates if x <= 0.5]
    dis = [x for x in rates if x > 0.5]
    check("audit: majority faction size", len(maj), 7)
    check("audit: dissenting faction size", len(dis), 3)
    check("audit: factions are separated, not a continuum",
          bool(min(dis) - max(maj) > 0.4), True)
    # corroboration
    check("audit: transit-asserting narratives in the sample", a["regex_transit_n"], 44)
    check("audit: corroboration, all coders", round(100 * a["corroboration"]), 95, tol=0.6)
    check("audit: corroboration, majority faction",
          round(100 * a["corroboration_majority"]), 98, tol=0.6)
    check("audit: corroboration, dissenting faction",
          round(100 * a["corroboration_dissent"]), 73, tol=0.6)
    lo, hi = a["ci"]
    check("audit: CI excludes 50%", bool(lo > 0.50), True)
    # review, 25 Sep 2026: three-level coding and the crossed interval
    pl = a["q2_plurality_flagged"]
    check("audit: plurality answers as written",
          (pl.get("correct", 0), pl.get("partly", 0), pl.get("incorrect", 0), pl.get("tie", 0)),
          (3, 21, 16, 4))
    check("audit: majority 'incorrect' as written", a["q2_majority_incorrect"], 7)
    check("audit: crossed interval as written", [round(100 * x) for x in a["ci_crossed"]], [66, 100])
    check("audit: three-category ending alpha as written", round(a["alpha_q1_cat3"], 2), -0.05)
    # the audit sample must be drawn only from the frozen corpus
    key = json.load(open(OUT / "audit_item_key.json", encoding="utf-8"))
    pids = {k["pid"] for k in key if not k.get("catch")}
    check("audit: no participant outside the frozen corpus",
          sorted(pids - set(CFG["corpus"]["participants"])), [])
    # the corpus-level figure the audit corroborates
    d = pd.read_csv(OUT / "C4_script_completion.csv")
    # the audit sample is stratified over model x condition, so its transit rate need not equal
    # the corpus rate exactly; it must be in the same region, and both must use the SAME
    # classifier. Before the correction the audit scored with the old regex and the corpus with
    # the new one, and this check caught that.
    check("audit: sample transit rate is close to the corpus rate",
          bool(abs(a["regex_transit_n"] / a["n_items"] - d.transit.mean()) < 0.12), True)


# ------------------------------------------------------------------ scope
def scope():
    """Part 8 item 6. Nothing excluded under the data agreement may appear in any artefact the paper cites."""
    banned = ["alarm rating", "ranking", "likert", "eeg", "eda", "physiolog",
              "co-registration", "hearing", "disabilit"]
    hits = []
    for f in sorted((ROOT / "paper" / "tables").glob("q1_*.tex")):
        t = f.read_text(encoding="utf-8").lower()
        hits += [(f.name, b) for b in banned if b in t]
    check("scope: no excluded topic appears in any Q1 table", hits, [])

    # A table must never print a rate the data does not support. The audit table's note once
    # carried a typed-in corpus rate that went stale when the classifier was corrected.
    import re as _re
    note = (ROOT / "paper" / "tables" / "q1_tab_corpus.tex").read_text(encoding="utf-8")
    m = _re.search(r"corroborates is ([0-9.]+)", note)
    live = 100 * pd.read_csv(OUT / "C4_script_completion.csv").transit.mean()
    check("tables: audit table quotes no corpus rate, or only the live one",
          bool(m is None or abs(float(m.group(1)) - live) < 0.06), True)


# ----------------------------------------------------------------- venue
def palette_and_venue():
    """The palette is part of the deliverable, so its safety is checked like any other number."""
    sys.path.insert(0, str(ROOT / "evactime"))
    import io, contextlib
    from palette import verify as verify_palette
    with contextlib.redirect_stdout(io.StringIO()):
        ok = verify_palette()
    check("palette: every co-occurring pair survives dichromatic vision", ok, True)

    v = CFG["venue"]
    # count only what the paper actually ships; superseded/ is excluded by the glob
    figs = sorted((ROOT / "paper" / "figures").glob("q1_*.pdf"))
    tabs = sorted((ROOT / "paper" / "tables").glob("q1_tab_*.tex"))
    check("venue: figure count within the IJDRS limit", len(figs) <= v["max_figures"], True)
    check("venue: exactly the five intended figures", len(figs), 5)
    # the withdrawn exhibits must not reappear in the shipped set
    names = {f.stem for f in figs}
    gone = {"q1_fig2_resolution", "q1_fig2_field", "q1_fig3_invariance",
            "q1_fig4_deliverable", "q1_fig3_raster", "q1_fig4_audit"}
    check("venue: no withdrawn figure in the paper set", sorted(names & gone), [])
    check("venue: table count within the IJDRS limit", len(tabs) <= v["max_tables"], True)


# ------------------------------------------------------------- reference scoring
def a1_scoring():
    """Guards on how an assertion is scored against the reference.

    The published scoring was `any sample in the window carries the claimed class`. At ~47 Hz a
    +/-1 s window holds about 94 samples, so one spurious sample scored agreement, and the
    statistic rose as the tolerance widened, which is what an existence test does regardless of
    whether the claim is right. Modal scoring asks what was looked at MOST and is the same
    question C2 asks.
    """
    a = pd.read_csv(OUT / "A1_assertions_scored.csv")
    a = a[a.timed == True]
    for c in ("hit_1s", "hit_2s", "modal_1s", "modal_2s", "dwell_1s"):
        check(f"A1: {c} present", c in a.columns, True)
    any_drift = float(a["hit_2s"].mean() - a["hit_1s"].mean())
    modal_drift = float(a["modal_2s"].mean() - a["modal_1s"].mean())
    check("A1: 'any' scoring inflates as the window widens", bool(any_drift > 0.10), True)
    check("A1: modal scoring is stable across tolerance", bool(abs(modal_drift) < 0.05), True)
    check("A1: 'any' overstates agreement relative to modal",
          bool(a["hit_1s"].mean() > 2 * a["modal_1s"].mean()), True)
    # a reference gap is not a model error
    check("A1: reference gaps excluded rather than scored False",
          bool(a["hit_1s"].isna().sum() > 0), True)
    check("A1: gaps are a small minority", bool(a["hit_1s"].isna().mean() < 0.05), True)
    # modal agreement and C2 exact accuracy ask the same question, so they must be in the
    # same region. They were 48.6% and 23.9% under the old scoring, which was the tell.
    c = json.load(open(OUT / "C2_confusion.json", encoding="utf-8"))
    check("A1 modal and C2 exact are now comparable",
          bool(abs(a["modal_1s"].mean() - c["exact"]) < 0.15), True)
    # the tolerance curve quoted in the Results and drawn in Figure 2d
    a2 = a[a.t_unity.notna()]
    curve = {t: (round(100 * a2[f"hit_{t:g}s"].mean(), 1), round(100 * a2[f"modal_{t:g}s"].mean(), 1))
             for t in (0.25, 0.5, 1.0, 3.0)}
    # the positive control has to survive the stricter rule
    npc = a[a.type == "look_at_npc"]
    check("A1: NPC control still the best class under modal scoring",
          bool(npc["modal_1s"].mean() > a[a.type != "look_at_npc"]["modal_1s"].mean()), True)


def neutral_prompt():
    """C6, the neutral-prompt ablation after Amendments 1 and 2 and the classifier extension.
    Every rate quoted in the Results, Discussion, Conclusion and Abstract."""
    r = json.load(open(OUT / "C6_neutral_prompt.json", encoding="utf-8"))
    pct = lambda x: round(100 * x, 1)
    check("C6: paired calls", r["pooled"]["n"], 435)
    check("C6: pooled rate v2 -> v3 as written", (pct(r["pooled"]["v2"]), pct(r["pooled"]["v3"])),
          (74.5, 55.6))
    check("C6: pooled intervals as written",
          ([pct(x) for x in r["pooled"]["v2_ci"]], [pct(x) for x in r["pooled"]["v3_ci"]]),
          ([71.5, 77.5], [49.0, 61.6]))
    check("C6: McNemar discordant pairs as written",
          (r["mcnemar"]["v2_only"], r["mcnemar"]["v3_only"]), (104, 22))
    check("C6: McNemar significant", bool(r["mcnemar"]["p"] < 0.001), True)
    want = {"openai/gpt-5.2|B": (16.1, 2.3, True), "anthropic/claude-opus-4.8|B": (86.2, 66.7, True),
            "qwen/qwen3-vl-235b-a22b-instruct|B": (83.9, 72.4, False),
            "gemini-3-flash-preview|A": (90.8, 54.0, True), "gemini-3-flash-preview|B": (95.4, 82.8, True)}
    for k, (w2, w3, sig) in want.items():
        c = r["cells"][k]
        check(f"C6: {k} as written", (pct(c["v2"]), pct(c["v3"])), (w2, w3))
        check(f"C6: {k} falls", bool(c["diff"] < 0), True)
        check(f"C6: {k} interval excludes zero as written", bool(c["ci"][1] < 0), sig)
    g = r["gemini_video_minus_frames"]
    check("C6: Gemini video minus frames as written", (pct(g["v2"]), pct(g["v3"])), (-4.6, -28.7))
    check("C6: its interval as written", [pct(-x) for x in reversed(g["v3_ci"])], [14.9, 44.8])
    rc = r["route_check"]
    check("C6: route check as written",
          (pct(rc["B"]["native"]), pct(rc["B"]["openrouter"]), pct(rc["B"]["agree"]),
           pct(rc["A"]["native"]), pct(rc["A"]["openrouter"])), (58.6, 95.4, 56.3, 81.6, 90.8))
    dc = r["drift_check"]
    check("C6: drift check as written",
          (pct(dc["qwen/qwen3-vl-235b-a22b-instruct"]["july"]), pct(dc["qwen/qwen3-vl-235b-a22b-instruct"]["now"]),
           pct(dc["anthropic/claude-opus-4.8"]["july"]), pct(dc["anthropic/claude-opus-4.8"]["now"]),
           pct(dc["openai/gpt-5.2"]["july"]), pct(dc["openai/gpt-5.2"]["now"])),
          (96.6, 83.9, 86.2, 86.2, 13.8, 16.1))
    check("C6: largest drift as written in the Conclusion",
          round(max(abs(v["july"] - v["now"]) for v in dc.values()) * 100, 1), 12.6, tol=0.05)
    check("C6: hallway counts as written", (r["hallway_with_claim"]["v2"],
                                            r["hallway_with_claim"]["v3"]), (16, 86))
    prompt = (ROOT / "evactime" / "prompts" / "common_task_v3_neutral.md").read_bytes()
    import hashlib
    check("C6: neutral prompt unchanged since the plan was fixed",
          hashlib.sha256(prompt).hexdigest(),
          "93e8c6f9edb6b219aebfcac67799984663bbb1773bdb2d2b18ce605279fe5f17")
    fvn = pd.read_csv(OUT / "C4_field_vs_narrative.csv")
    check("C4: field filled but narrative declined", int((fvn.field_given & ~fvn.narr).sum()), 294)
    check("C4: exit narratives citing the scene change",
          round(100 * fvn[fvn.narr].evid_scene.mean(), 1), 78.0)


def scoring_null_and_model():
    """A2 permutation nulls of both scoring rules, and the added GEE model of the prompt effect."""
    raw = json.load(open(OUT / "A2_scoring_null.json", encoding="utf-8"))
    n = {k: v for k, v in raw.items() if k[0].isdigit()}
    pct = lambda x: round(100 * x, 1)
    check("A2: fixed cohort size as written", raw["cohort"]["fixed_n"], 1151)
    check("A2: every window scored on the fixed cohort", {v["n"] for v in n.values()}, {1151})
    check("A2: permissive and modal curves as written",
          (pct(n["0.25"]["any"]), pct(n["1"]["any"]), pct(n["3"]["any"]),
           pct(n["0.25"]["modal"]), pct(n["3"]["modal"])), (37.4, 56.2, 75.0, 27.0, 24.8))
    check("A2: modal range across windows as written",
          (pct(min(v["modal"] for v in n.values())), pct(max(v["modal"] for v in n.values()))),
          (24.8, 27.6))
    check("A2: open cohort at 1 s reproduces Equation 3 (C2) exactly",
          (n["1"]["n_open"], pct(n["1"]["modal_open"])), (1364, 24.2))
    check("A2: permissive-rule chance level at 0.25 s and 3 s as written",
          (pct(n["0.25"]["any_null"]), pct(n["3"]["any_null"])), (29.8, 63.6))
    check("A2: 'three times' at 3 s", bool(n["3"]["any"] / n["3"]["modal"] >= 3), True)
    rise_obs = n["3"]["any"] - n["0.25"]["any"]
    rise_null = n["3"]["any_null"] - n["0.25"]["any_null"]
    check("A2: chance accounts for 34 of the 38 points as written",
          (round(100 * rise_null), round(100 * rise_obs)), (34, 38))
    check("A2: permissive chance-corrected range",
          (pct(n["0.25"]["kappa_any"]), pct(n["3"]["kappa_any"])), (10.8, 31.2))
    km = [v["kappa_modal"] for v in n.values()]
    check("A2: modal chance-corrected range as written", (pct(min(km)), pct(max(km))), (7.5, 8.9))
    check("A2: permissive rule never decreases with the window (the stated property)",
          all(n[a]["any"] <= n[b]["any"] + 1e-12 for a, b in zip(list(n)[:-1], list(n)[1:])), True)
    check("A2: dwell sensitivity as written",
          (round(100 * raw["median_modal_share_1s"]), pct(raw["dwell_1s"]["0.5"]["kappa"])), (70, 7.1))
    r = json.load(open(OUT / "C6_neutral_prompt.json", encoding="utf-8"))
    g = r["gee_pooled_bc"]
    check("C6 GEE: pooled odds ratio, bias-reduced, as written",
          (round(g["OR"], 2), round(g["ci"][0], 2), round(g["ci"][1], 2)), (0.25, 0.15, 0.39))
    ors = [v["OR"] for v in r["gee_cells"].values()]
    check("C6 GEE: cell odds-ratio range as written", (round(min(ors), 2), round(max(ors), 2)),
          (0.12, 0.50))
    check("C6 GEE: cells do not differ, as written", round(r["gee_cells_joint_p"], 2), 0.35)
    sg = r["participant_sign"]
    check("C6: participant sign test as written", (sg["down"], sg["up"], bool(sg["p"] < 0.001)),
          (22, 3, True))
    dq = r["drift_check"]["qwen/qwen3-vl-235b-a22b-instruct"]
    check("C6: Qwen date drift as written", round(100 * (dq["july"] - dq["now"]), 1), 12.6)


def native_gemini():
    """Amendment 3: Gemini 3 Flash on the native interface, both descriptions on one day."""
    r = json.load(open(OUT / "C6_neutral_prompt.json", encoding="utf-8"))["native"]
    pct = lambda x: round(100 * x, 1)
    c = r["cells"]
    check("native: all four cells complete", (c["A"]["n"], c["B"]["n"]), (87, 87))
    check("native: video v2 -> v3 as written", (pct(c["A"]["v2"]), pct(c["A"]["v3"])), (81.6, 65.5))
    check("native: video fall interval as written", [pct(-x) for x in reversed(c["A"]["ci"])], [2.3, 29.9])
    check("native: frames v2 -> v3 as written", (pct(c["B"]["v2"]), pct(c["B"]["v3"])), (59.8, 63.2))
    check("native: frames change not significant", bool(c["B"]["ci"][0] < 0 < c["B"]["ci"][1]), True)
    v2, v3 = r["video_minus_frames_v2"], r["video_minus_frames_v3"]
    check("native: video minus frames as written",
          (pct(v2["est"]), [pct(x) for x in v2["ci"]], pct(v3["est"]), [pct(x) for x in v3["ci"]]),
          (21.8, [8.0, 35.6], 2.3, [-11.5, 18.4]))
    rs = r["route_sameday"]["v2|B"]
    check("native: same-day route on frames as written",
          (pct(rs["native"]), pct(rs["openrouter"]), pct(rs["agree"])), (59.8, 95.4, 64.4))
    july = json.load(open(OUT / "C6_neutral_prompt.json", encoding="utf-8"))["route_check"]
    check("native: stable against July as written",
          (pct(july["A"]["native"]), pct(july["B"]["native"])), (81.6, 58.6))


def main():
    for fn in (corpus, resolution, script_completion, audit, a1_scoring, scope,
               palette_and_venue, neutral_prompt, scoring_null_and_model, native_gemini):
        try:
            fn()
        except Exception as e:
            CHECKS.append((False, f"{fn.__name__}: raised {type(e).__name__}", str(e), "no error"))
    bad = [c for c in CHECKS if not c[0]]
    for ok, claim, got, want in CHECKS:
        if not ok:
            print(f"  FAIL  {claim}\n          got {got!r}\n          want {want!r}")
    print(f"\n{len(CHECKS)} checks, {len(bad)} failures")
    if not bad:
        print("every numeric claim traces to an artefact")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
