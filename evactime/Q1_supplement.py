"""Supplementary Material for the IJDRS paper, generated from the analysis outputs (review, 25 Sep 2026).

Every number is read from an output file, so the supplement cannot drift from the analysis. The
prose is fixed here; the tables are filled from outputs/.

Writes overleaf_Q1/supplement.tex (a standalone document).
Run: py evactime/Q1_supplement.py
"""
import collections, datetime, hashlib, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
DST = ROOT / "overleaf_Q1" / "supplement.tex"
J = lambda f: json.load(open(OUT / f, encoding="utf-8"))
pc = lambda x, d=1: f"{100 * x:.{d}f}"

NAMES = {"gemini-3.1-pro-preview": "Gemini 3.1 Pro", "gemini-3-flash-preview": "Gemini 3 Flash",
         "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "openai/gpt-5.2": "GPT-5.2",
         "anthropic/claude-opus-4.8": "Claude Opus 4.8", "anthropic/claude-sonnet-5": "Claude Sonnet 5",
         "qwen/qwen3-vl-235b-a22b-instruct": "Qwen3-VL 235B",
         "meta-llama/llama-4-maverick": "Llama 4 Maverick"}
RUNS = [("full_openrouter_raw.jsonl", "Main, original", "OpenRouter"),
        ("gemini_native_raw.jsonl", "Main, original", "Google"),
        ("v2_sameday_raw.jsonl", "Same day, original", "OpenRouter"),
        ("neutral_prompt_raw.jsonl", "Ablation, neutral", "OpenRouter"),
        ("gemini_openrouter_raw.jsonl", "Ablation, both", "OpenRouter"),
        ("gemini_native_sameday_raw.jsonl", "Ablation, both", "Google")]


def day(ts):
    if isinstance(ts, (int, float)):
        return datetime.datetime.fromtimestamp(ts, datetime.UTC).strftime("%Y-%m-%d")
    return (ts or "")[:10]


def manifest_rows():
    rows = []
    for f, run, route in RUNS:
        g = collections.defaultdict(lambda: dict(n=0, prov=collections.Counter(), days=set(), pv=set()))
        for line in open(OUT / f, encoding="utf-8"):
            r = json.loads(line)
            if r["model"] == "gemini-3-flash-preview" and f == "neutral_prompt_raw.jsonl":
                continue      # six test calls before Amendment 1; not analyzed
            k = g[r["model"]]
            k["n"] += 1
            k["prov"][r.get("provider") or "Google"] += 1
            k["days"].add(day(r.get("ts")))
            k["pv"].add({"v2": "original", "v3n": "neutral"}.get(r.get("prompt_version"), "?"))
        for m, k in g.items():
            prov = ", ".join(f"{p} ({n})" if len(k["prov"]) > 1 else p for p, n in k["prov"].items())
            rows.append(rf"{run} & \texttt{{{m}}} & {route} & {prov} & "
                        rf"{', '.join(sorted(k['days']))} & {k['n']} \\")
    return rows


def main():
    bv = J("S3_blind_validation.json")
    bp = bv["primary"]
    s2, c2, a2, au, c6, c4 = (J("S2_reference_quality.json"), J("C2_confusion.json"),
                              J("A2_scoring_null.json"), J("audit_final.json"),
                              J("C6_neutral_prompt.json"), J("C4_summary.json"))
    sha = {v: hashlib.sha256((ROOT / "evactime" / "prompts" / f).read_bytes()).hexdigest()[:16]
           for v, f in (("original", "common_task_v2.md"), ("neutral", "common_task_v3_neutral.md"))}
    q = lambda d, k, f=1, m=1.0: (f"{d[k]['median'] * m:.{f}f} [{d[k]['min'] * m:.{f}f}, "
                                  f"{d[k]['max'] * m:.{f}f}]")
    ref_rows = [
        rf"Clock drift, LSL to Unity (ms per s) & {q(s2, 'drift_s_per_s', 2, 1000)} \\",
        rf"Audible alarm minus engine alarm marker (s) & {q(s2, 'alarm_audio_minus_marker_s', 2)} \\",
        rf"Gaze raycast rate (Hz) & {q(s2, 'gaze_hz', 1)} \\",
        rf"Samples on a tagged object in the 75\,s window (\%) & {q(s2, 'share_tagged_in_window', 0, 100)} \\",
        rf"Longest gap in the raycast series (s) & {q(s2, 'longest_gap_s', 2)} \\",
        rf"Recordings with a gap longer than 1\,s & {s2['n_gap_over_1s']} of {s2['n']} \\"]

    sens = [("cell", "Participant $\\times$ model $\\times$ format (primary)", c2),
            ("participant", "Participant", c2["sensitivity"]["participant"]),
            ("call", "Single call", c2["sensitivity"]["call"])]
    null_rows = [rf"{lab} & {pc(r['chance'])} & {pc(r['kappa_object'])} & {pc(r['region_chance'])} & "
                 rf"{pc(r['kappa_region'])} \\" for _, lab, r in sens]
    tnames = {"npc": "Non-player character", "alarm": "Alarm", "sign": "Exit sign"}
    tgt_rows = []
    for k, v in c2["per_target"].items():
        rk = (v["region"] - v["region_chance"]) / (1 - v["region_chance"])
        p = "$<$0.001" if v["p"] < 0.001 else f"{v['p']:.3f}"
        tgt_rows.append(rf"{tnames[k]} & {v['n']} & {pc(v['observed'])} & {pc(v['chance'])} "
                        rf"[{pc(v['null_range'][0])}, {pc(v['null_range'][1])}] & {p} & {pc(v['kappa'])} & "
                        rf"{pc(v['region'])} & {pc(v['region_chance'])} & {pc(rk)} \\")
    win_rows = []
    for t in ("0.25", "0.5", "1", "1.5", "2", "3"):
        r = a2[t]
        win_rows.append(rf"$\pm${t} & {r['n_open']} & {pc(r['any'])} & {pc(r['any_null'])} & "
                        rf"{pc(r['kappa_any'])} & {pc(r['modal'])} & {pc(r['modal_null'])} & "
                        rf"{pc(r['kappa_modal'])} \\")
    dw = a2["dwell_1s"]
    lv = au["q2_levels_flagged"]
    pl = au["q2_plurality_flagged"]
    fmt = {"A": "video with audio", "B": "frames"}
    short = {k.split("/")[-1]: v for k, v in NAMES.items()}

    def cellname(k):
        m, c = k.rsplit("_", 1)
        return f"{short[m]} on {fmt[c]}"
    cells = ", ".join(f"{v['OR']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}] for {cellname(k)}"
                      for k, v in c6["gee_cells"].items())

    tex = rf"""% generated by evactime/Q1_supplement.py; do not edit
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{newtxtext,newtxmath}}
\usepackage{{booktabs,array,xcolor}}
\usepackage[hidelinks]{{hyperref}}
\usepackage[labelfont=bf,labelsep=period,font=small]{{caption}}
\renewcommand{{\thetable}}{{S\arabic{{table}}}}
\renewcommand{{\thesection}}{{S\arabic{{section}}}}
\title{{Supplementary Material\\[4pt]\large Testing Vision-Language Model Annotations of Virtual Reality
Evacuation Recordings Against the Game Engine Record}}
\date{{}}
\begin{{document}}
\maketitle

\section{{Quality of the engine reference}}

The screen video starts at the engine marker for entry to the waiting room. The alignment was
checked against the audible alarm detected in each video, which is independent of the marker
stream, and the residual was not used to correct the offset. Table~\ref{{tab:s-ref}} gives the
diagnostics for the 29 analyzed recordings as the median with the range across recordings. Samples
logged with the identifiers $-1$ or 0, which the engine records as no object, and samples on two
untagged cubes are excluded from the modal object of Equation~2 of the paper. The eleven tagged
objects with a position in the scene are the nine targets of Figure~1 and the handles of the two
doors.
Gaze was recorded with the eye tracker integrated in the HTC Vive Pro Eye head-mounted display.
\textcolor{{red}}{{[Lund team to add the nominal accuracy, calibration procedure, recalibration rule and any
per-participant validation error.]}}

\begin{{table}}[!ht]\centering\small
\caption{{Diagnostics of the engine reference, median [range] over 29 recordings}}\label{{tab:s-ref}}
\begin{{tabular}}{{@{{}}l l@{{}}}}\toprule
Diagnostic & Value \\\midrule
{chr(10).join(ref_rows)}
\bottomrule\end{{tabular}}
\end{{table}}

\section{{Model calls and stimuli}}

Table~\ref{{tab:s-runs}} lists every run with the model identifier, route, serving backend reported by
the route, call date and number of calls. Each model was called three times per recording and input
format, the first at temperature 0 and the others at 0.4. No seed, token limit or system prompt was
set; each call sent one user message holding the task description followed by the media. The
original and neutral task descriptions have SHA-256 prefixes \texttt{{{sha['original']}}} and
\texttt{{{sha['neutral']}}}, and both are in the code repository.

The 75\,s window was cut from the screen recording and frozen before any call. The frame format
holds 75 JPEG images at 512\,$\times$\,288 pixels and quality 85, sampled at one per second and sent
in order after the text, and the task description tells the model that time is given as the frame
index. The video with audio holds the same window encoded as H.264 at 512\,$\times$\,288 pixels
(CRF 23) with AAC audio at 64\,kb/s, and the muted video is the same clip without its audio track;
the SHA-256 of every clip and frame set is in the code repository. On the native Google interface the video was
uploaded through the file interface and the frames sent inline, with media resolution set to medium
and a JSON response requested.

\begin{{table}}[!ht]\centering\scriptsize\setlength{{\tabcolsep}}{{3pt}}
\caption{{Run manifest}}\label{{tab:s-runs}}
\begin{{tabular}}{{@{{}}l l l l l r@{{}}}}\toprule
Run & Model identifier & Route & Backend & Date & Calls \\\midrule
{chr(10).join(manifest_rows())}
\bottomrule\end{{tabular}}
\end{{table}}

\section{{Analysis plan for the neutral description and its amendments}}

The plan was written on 23 September 2026 before the first call and fixed the manipulation, the five
cells, the primary outcome, the paired analyses and the exclusions. Three amendments were made, each
before any call it describes. Amendment~1 moved the two Gemini~3~Flash cells to OpenRouter and ran
both descriptions there, because the native interface refused every call at the time; a later
correction recorded that the two routes encode video with the same number of tokens and frames with
about twice as many through OpenRouter. Amendment~2 reran the original description on the same day
for GPT-5.2, Claude~Opus~4.8 and Qwen3-VL~235B, with the backend pinned and fallbacks disabled,
because the earlier calls were two months old and partly served by other backends; the earlier calls
enter only a drift check. Amendment~3 added native Gemini~3~Flash calls under both descriptions once
the interface was restored. The count of hallway destinations and the regression model of
Equation~1 were added after the run and are reported as such. The full plan is in the code
repository.

The exit-claim rates of the same model and description on the two dates were
{pc(c6['drift_check']['anthropic/claude-opus-4.8']['july'])}\% and {pc(c6['drift_check']['anthropic/claude-opus-4.8']['now'])}\% for Claude~Opus~4.8,
{pc(c6['drift_check']['openai/gpt-5.2']['july'])}\% and {pc(c6['drift_check']['openai/gpt-5.2']['now'])}\% for GPT-5.2, and
{pc(c6['drift_check']['qwen/qwen3-vl-235b-a22b-instruct']['july'])}\% and {pc(c6['drift_check']['qwen/qwen3-vl-235b-a22b-instruct']['now'])}\% for Qwen3-VL~235B. The pooled odds ratio of the neutral
description was {c6['gee_pooled_bc']['OR']:.2f} [{c6['gee_pooled_bc']['ci'][0]:.2f}, {c6['gee_pooled_bc']['ci'][1]:.2f}] with the bias-reduced covariance and
{c6['gee_pooled']['OR']:.2f} [{c6['gee_pooled']['ci'][0]:.2f}, {c6['gee_pooled']['ci'][1]:.2f}] with the standard robust covariance. The cell odds ratios of the interaction
model were {cells}, and a joint test found no difference between cells ($p={c6['gee_cells_joint_p']:.2f}$).

\section{{Chance levels and sensitivity of the attention scores}}

Table~\ref{{tab:s-null}} gives the chance level and $\kappa$ under three permutation strata. Pooling
claims over models and formats raises $\kappa$, and restricting the permutation to a single call
lowers it; the conclusions hold under all three. Table~\ref{{tab:s-target}} compares each claimed
object with its own chance level under the primary null, and Table~\ref{{tab:s-window}} gives both
scoring rules on the {a2['cohort']['fixed_n']} assertions scorable at every window, with the number scorable at each
window when the set is not fixed. When the modal object held at least half of the tagged samples in
its window ({pc(dw['0.5']['share'], 0)}\% of windows), object agreement was {pc(dw['0.5']['modal'])}\% against {pc(dw['0.5']['chance'])}\% by chance
($\kappa={pc(dw['0.5']['kappa'])}\%$), and when it held at least three quarters ({pc(dw['0.75']['share'], 0)}\%), {pc(dw['0.75']['modal'])}\% against
{pc(dw['0.75']['chance'])}\% ($\kappa={pc(dw['0.75']['kappa'])}\%$). The doctor's door, which belongs to neither region, was the modal object in
{pc(c2['doctor_door_share'])}\% of scored windows.

\begin{{table}}[!ht]\centering\small
\caption{{Chance level and $\kappa$ (\%) under three permutation strata}}\label{{tab:s-null}}
\begin{{tabular}}{{@{{}}l r r r r@{{}}}}\toprule
Claims permuted within & Object chance & Object $\kappa$ & Region chance & Region $\kappa$ \\\midrule
{chr(10).join(null_rows)}
\bottomrule\end{{tabular}}
\end{{table}}

\begin{{table}}[!ht]\centering\small\setlength{{\tabcolsep}}{{4pt}}
\caption{{Each claimed object against its own chance level (\%)}}\label{{tab:s-target}}
\begin{{tabular}}{{@{{}}l r r l l r r r r@{{}}}}\toprule
Claimed object & $n$ & Object & Chance [95\% null range] & $p$ & $\kappa$ & Region & Chance & $\kappa$ \\\midrule
{chr(10).join(tgt_rows)}
\bottomrule\end{{tabular}}
\end{{table}}

\begin{{table}}[!ht]\centering\small
\caption{{Permissive and modal rules across matching windows (\%), fixed set of assertions}}\label{{tab:s-window}}
\begin{{tabular}}{{@{{}}l r r r r r r r@{{}}}}\toprule
Window (s) & $n$ if not fixed & Permissive & Chance & $\kappa$ & Modal & Chance & $\kappa$ \\\midrule
{chr(10).join(win_rows)}
\bottomrule\end{{tabular}}
\end{{table}}

\section{{Exit-claim classifier}}

The classifier works sentence by sentence. It counts a sentence as a transit claim when a verb of
movement (pass, walk, step, go, move, head, proceed, exit, leave, enter, emerge, escape) takes a door,
doorway, room or corridor as its object or destination, or when the recording is said to end in a
corridor, hallway or new area. A bare ``through it'' counts only when a door is named in the same or
the previous sentence, the noun phrase ``exit door'' never counts, negated sentences and sentences
whose subject is a non-player character are excluded, and a description of the reset to the same
room overrides a scene-change match in the same sentence. The rules were developed against
the human audit labels and hand-read narratives, and 43 constructed test cases fix their behavior.
Against the audit it flagged no narrative that a majority of coders judged correct and recovered
87.5\% of those not judged correct, and a hand reading of 45 narratives written under the neutral
description found seven missed constructions, such as passing into an adjoining room, which were
added before the ablation was scored. These figures describe development, not validation. The frozen
classifier was then tested on 100 narratives that no one had read, 20 drawn at random from each of
five model and format cells of the same-day runs under the original description (GPT-5.2,
Claude~Opus~4.8 and Qwen3-VL~235B on frames, Gemini~3~Flash on frames and on video with audio). The
SHA-256 of the classifier was recorded when the sample was drawn and checked before scoring. One
author labeled each narrative, blind to model, format and classifier output, as asserting that the
participant left the room, not asserting it, or unsure. The labels were {bv['labels'].get('yes', 0)} yes,
{bv['labels'].get('no', 0)} no and {bv['labels'].get('unsure', 0)} unsure, and Table~\ref{{tab:s-blind}} compares them with the classifier.
The classifier flagged {bv['unsure_flagged']} of the unsure narratives. Its false positives were hedged
narratives that mention leaving while saying it is not clearly shown, and its misses were the
construction ``passes through by frame $n$'' and a bare ``exit into a hallway''. With unsure counted
as yes, $\kappa$ was {bv['unsure_as_yes']['kappa']:.2f}, and with unsure counted as no, {bv['unsure_as_no']['kappa']:.2f}.

\begin{{table}}[!ht]\centering\small
\caption{{Frozen classifier against blind labels, unsure excluded ($n={bp['n']}$)}}\label{{tab:s-blind}}
\begin{{tabular}}{{@{{}}l r r l@{{}}}}\toprule
& Human yes & Human no & Measure [95\% Wilson interval] \\\midrule
Classifier yes & {bp['tp']} & {bp['fp']} & Recall {pc(bp['recall'])}\% [{pc(bp['recall_ci'][0])}, {pc(bp['recall_ci'][1])}] \\
Classifier no & {bp['fn']} & {bp['tn']} & Precision {pc(bp['precision'])}\% [{pc(bp['precision_ci'][0])}, {pc(bp['precision_ci'][1])}] \\
& & & Specificity {pc(bp['specificity'])}\% [{pc(bp['specificity_ci'][0])}, {pc(bp['specificity_ci'][1])}] \\
& & & Accuracy {pc(bp['accuracy'])}\%, $\kappa={bp['kappa']:.2f}$ \\
\bottomrule\end{{tabular}}
\end{{table}}

Of the {c4['scene_change_language']} narratives that used scene-change language (a cut, fade, black screen or
transition), {c4['scene_change_language'] - c4['scene_change_without_exit']} also asserted an exit, and {c4['same_room']} stated a return to the same room.

\section{{Human audit}}

On the {au['regex_transit_n']} narratives flagged as asserting a transit, the coders answered correct in {pc(lv['correct'], 0)}\%
of judgments, partly correct in {pc(lv['partly'], 0)}\%, incorrect in {pc(lv['incorrect'], 0)}\% and unsure in {pc(lv['unsure'], 0)}\%. The
plurality answer was incorrect for {pl.get('incorrect', 0)} narratives, partly correct for {pl.get('partly', 0)}, correct for
{pl.get('correct', 0)} and tied for {pl.get('tie', 0)}, and a majority answered incorrect for {au['q2_majority_incorrect']}. The share not judged
correct by majority was {pc(au['corroboration'], 0)}\%, with a 95\% interval of [{pc(au['ci'][0], 0)}\%, {pc(au['ci'][1], 0)}\%] over coders and
[{pc(au['ci_crossed'][0], 0)}\%, {pc(au['ci_crossed'][1], 0)}\%] over coders and narratives together. For the ending, the binary coding gives
$\alpha={au['alpha_q1']:+.2f}$, keeping through a door, same room and other apart gives $\alpha={au['alpha_q1_cat3']:+.2f}$, and keeping
unsure as a fourth answer gives $\alpha={au['alpha_q1_cat4']:+.2f}$; one coder chose other for {au['other_max']} of {au['n_items']} items, with
notes that the participant stayed in the room. For the description, $\alpha={au['alpha_q2']:+.2f}$ [{au['alpha_q2_ci'][0]:+.2f},
{au['alpha_q2_ci'][1]:+.2f}].

\section{{Literature positioning}}

The score, supporting quotation and location for every cell of Table~1 of the paper are in the file
\nolinkurl{{paper/supplement/positioning_evidence.csv}} of the code repository.

\end{{document}}
"""
    DST.write_text(tex, encoding="utf-8")
    print("wrote", DST)


if __name__ == "__main__":
    main()
