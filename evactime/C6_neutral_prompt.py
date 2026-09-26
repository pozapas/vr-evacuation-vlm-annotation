"""C6 - neutral-prompt ablation, analysed exactly as fixed in docs/NEUTRAL_PROMPT_PREREG.md.

The v2 task description asserted that the participant leaves the room and required an exit
timestamp. The v3 description removes both and changes nothing else. Every usable call is scored
with the same transit classifier and the same record rules as C4, and each v3 call is paired with
the v2 call for the same model, participant, condition and repeat.

Analyses (numbered as in the plan)
  1. exit-claim rate per cell, v3 against v2, paired difference with a participant-clustered
     bootstrap interval, and an exact McNemar test on the discordant pairs pooled over cells
  2. pooled v3 rate with the same interval
  3. Gemini 3 Flash, video minus frames under v3 against the same difference under v2
  4. share of v3 calls that fill exit_reached, and its agreement with the narrative claim

Outputs: C6_neutral_prompt.csv (one row per paired call), C6_neutral_prompt.json, C6_report.txt
Run: py evactime/C6_neutral_prompt.py
"""
import json, pathlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from transit_classifier import asserts_transit
from vlm_records import load_records, FILES

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
V3_FILES = ("neutral_prompt_raw.jsonl",)
GEM_OR = ("gemini_openrouter_raw.jsonl",)   # Amendment 1: both prompts on the same route
SAMEDAY = ("v2_sameday_raw.jsonl",)          # Amendment 2: v2 on the same day and backend as v3
CELLS = [("openai/gpt-5.2", "B"), ("anthropic/claude-opus-4.8", "B"),
         ("qwen/qwen3-vl-235b-a22b-instruct", "B"),
         ("gemini-3-flash-preview", "A"), ("gemini-3-flash-preview", "B")]
NBOOT, SEED = 5000, 17


def frame(recs, version):
    rows = []
    for r in recs:
        if (r["model"], r["cond"]) not in CELLS:
            continue
        ev = (r["parsed"].get("events") or {})
        ex = ev.get("exit_reached")
        rows.append(dict(model=r["model"], cond=r["cond"], pid=r["pid"], rep=r["rep"],
                         version=version, narr=bool(asserts_transit(r["narrative"])),
                         field=isinstance(ex, dict) and any(k in ex for k in ("t", "frame")),
                         narrative=r["narrative"]))
    return pd.DataFrame(rows)


def boot(df, fn, n=NBOOT, seed=SEED):
    """Participant-clustered bootstrap of a statistic computed on a paired frame."""
    rng = np.random.default_rng(seed)
    groups = {p: g for p, g in df.groupby("pid")}
    keys = list(groups)
    vals = []
    for _ in range(n):
        pick = rng.choice(keys, len(keys), replace=True)
        vals.append(fn(pd.concat([groups[k] for k in pick])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    v2, _, _ = load_records(files=FILES)
    v3, st3, _ = load_records(files=V3_FILES)
    gor, stg, _ = load_records(files=GEM_OR)

    def gem(r):
        return r["model"].startswith("gemini")
    # Gemini cells pair v3 against v2 on the SAME route (OpenRouter). The native v2 Gemini calls
    # enter only the route check. The other cells pair against the original v2 calls.
    # Amendment 2: the non-Gemini cells pair against v2 rerun on the same day and backend.
    same, _, _ = load_records(files=SAMEDAY)
    a = pd.concat([frame(same, "v2"),
                   frame([r for r in gor if r["prompt_version"] == "v2"], "v2")])
    b = pd.concat([frame([r for r in v3 if not gem(r)], "v3"),
                   frame([r for r in gor if r["prompt_version"] == "v3n"], "v3")])
    for k in ("calls_on_disk", "kept", "api_error", "recovered"):
        st3[k] += stg[k]
    key = ["model", "cond", "pid", "rep"]
    pair = a.merge(b, on=key, suffixes=("_v2", "_v3"))
    pair.drop(columns=["version_v2", "version_v3"]).to_csv(OUT / "C6_neutral_prompt.csv", index=False)

    res, lines = {"cells": {}}, []
    lines.append(f"v3 calls on disk {st3['calls_on_disk']}, usable {st3['kept']}, "
                 f"429 {st3['api_error']}, recovered {st3['recovered']}; paired with v2 {len(pair)}")
    lines.append("\n1. EXIT-CLAIM RATE, v2 -> v3, paired on model x participant x condition x repeat")
    for m, c in CELLS:
        g = pair[(pair.model == m) & (pair.cond == c)]
        if g.empty:
            lines.append(f"   {m:34s} {c:2s} no paired calls yet")
            continue
        r2, r3 = g.narr_v2.mean(), g.narr_v3.mean()
        lo, hi = boot(g, lambda d: d.narr_v3.mean() - d.narr_v2.mean())
        share = 1 - r3 / r2 if r2 else float("nan")
        res["cells"][f"{m}|{c}"] = dict(n=len(g), v2=r2, v3=r3, diff=r3 - r2, ci=[lo, hi],
                                        share_explained=share)
        lines.append(f"   {m:34s} {c:2s} n={len(g):3d}  v2 {r2:6.1%}  v3 {r3:6.1%}  "
                     f"diff {100*(r3-r2):+6.1f} pts [{100*lo:+.1f}, {100*hi:+.1f}]  "
                     f"share of v2 rate removed {share:.0%}")
    up = int((~pair.narr_v2 & pair.narr_v3).sum())
    down = int((pair.narr_v2 & ~pair.narr_v3).sum())
    p = binomtest(min(up, down), up + down, 0.5).pvalue if up + down else float("nan")
    res["mcnemar"] = dict(v2_only=down, v3_only=up, p=p)
    lines.append(f"   McNemar exact, pooled: claim under v2 only {down}, under v3 only {up}, p = {p:.2g}")

    lines.append("\n2. POOLED")
    r2, r3 = pair.narr_v2.mean(), pair.narr_v3.mean()
    lo, hi = boot(pair, lambda d: d.narr_v3.mean())
    lo2, hi2 = boot(pair, lambda d: d.narr_v2.mean())
    res["pooled"] = dict(n=len(pair), v2=r2, v2_ci=[lo2, hi2], v3=r3, v3_ci=[lo, hi])
    lines.append(f"   same calls, v2 {r2:.1%} [{lo2:.1%}, {hi2:.1%}]  ->  v3 {r3:.1%} [{lo:.1%}, {hi:.1%}]")

    lines.append("\n3. GEMINI 3 FLASH, VIDEO MINUS FRAMES")
    gf = pair[pair.model == "gemini-3-flash-preview"]
    if set(gf.cond) >= {"A", "B"}:
        def did(d, v):
            return d[d.cond == "A"][f"narr_{v}"].mean() - d[d.cond == "B"][f"narr_{v}"].mean()
        d2, d3 = did(gf, "v2"), did(gf, "v3")
        lo, hi = boot(gf, lambda d: did(d, "v3") - did(d, "v2"))
        lo3, hi3 = boot(gf, lambda d: did(d, "v3"))
        res["gemini_video_minus_frames"] = dict(v2=d2, v3=d3, v3_ci=[lo3, hi3], did=d3 - d2,
                                                did_ci=[lo, hi])
        lines.append(f"   v2 {100*d2:+.1f} pts, v3 {100*d3:+.1f} pts [{100*lo3:+.1f}, {100*hi3:+.1f}], "
                     f"difference {100*(d3-d2):+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]")
    else:
        lines.append("   both conditions not yet available")

    lines.append("\nROUTE CHECK (Amendment 1): Gemini 3 Flash, v2 prompt, OpenRouter against native API")
    nat = frame([r for r in v2 if gem(r)], "v2")
    orv2 = frame([r for r in gor if r["prompt_version"] == "v2"], "v2")
    rc = nat.merge(orv2, on=key, suffixes=("_nat", "_or"))
    res["route_check"] = {}
    for c in ("A", "B"):
        g = rc[rc.cond == c]
        if len(g):
            res["route_check"][c] = dict(n=len(g), native=g.narr_nat.mean(),
                                         openrouter=g.narr_or.mean(),
                                         agree=(g.narr_nat == g.narr_or).mean())
            lines.append(f"   {c}: n={len(g)}  native {g.narr_nat.mean():.1%}  OpenRouter "
                         f"{g.narr_or.mean():.1%}  same verdict in "
                         f"{(g.narr_nat == g.narr_or).mean():.1%} of paired calls")

    lines.append("\nDRIFT CHECK (Amendment 2): v2 prompt, July calls against same-day calls")
    july = frame([r for r in v2 if not gem(r)], "v2")
    sd = frame(same, "v2")
    dc = july.merge(sd, on=key, suffixes=("_jul", "_now"))
    res["drift_check"] = {}
    for m in sorted(dc.model.unique()):
        g = dc[dc.model == m]
        res["drift_check"][m] = dict(n=len(g), july=g.narr_jul.mean(), now=g.narr_now.mean(),
                                     agree=(g.narr_jul == g.narr_now).mean())
        lines.append(f"   {m:34s} n={len(g):3d}  July {g.narr_jul.mean():.1%}  today "
                     f"{g.narr_now.mean():.1%}  same verdict in {(g.narr_jul == g.narr_now).mean():.1%}")

    lines.append("\n4. FIELD BEHAVIOR UNDER v3")
    f3 = pair.field_v3.mean()
    agree = (pair.field_v3 == pair.narr_v3).mean()
    res["field_v3"] = dict(filled=f3, agree_with_narrative=agree,
                           filled_no_claim=int((pair.field_v3 & ~pair.narr_v3).sum()),
                           claim_no_field=int((~pair.field_v3 & pair.narr_v3).sum()))
    lines.append(f"   exit_reached filled in {f3:.1%} of v3 calls (v2 {pair.field_v2.mean():.1%}); "
                 f"field and narrative agree in {agree:.1%}")
    for m, c in CELLS:
        g = pair[(pair.model == m) & (pair.cond == c)]
        if len(g):
            lines.append(f"   {m:34s} {c:2s} field filled {g.field_v3.mean():6.1%}  "
                         f"narrative claim {g.narr_v3.mean():6.1%}")
    lines.append("\nADDITIONAL, NOT IN THE PLAN: a hallway or corridor named together with an exit claim")
    hall = {}
    for v in ("v2", "v3"):
        h = pair[f"narrative_{v}"].str.contains(r"hallway|corridor", case=False, regex=True) & pair[f"narr_{v}"]
        hall[v] = int(h.sum())
    res["hallway_with_claim"] = hall
    lines.append(f"   v2 {hall['v2']} of {len(pair)}, v3 {hall['v3']} of {len(pair)}")
    # Amendment 3: Gemini 3 Flash on the native interface, both descriptions on the same day.
    nat_path = OUT / "gemini_native_sameday_raw.jsonl"
    if nat_path.exists():
        nrec, _, _ = load_records(files=("gemini_native_sameday_raw.jsonl",))
        n2 = frame([r for r in nrec if r["prompt_version"] == "v2"], "v2")
        n3 = frame([r for r in nrec if r["prompt_version"] == "v3n"], "v3")
        npair = n2.merge(n3, on=key, suffixes=("_v2", "_v3"))
        res["native"] = {"cells": {}}
        lines.append("\nAMENDMENT 3: Gemini 3 Flash, NATIVE interface, both descriptions same day")
        for c in ("A", "B"):
            g = npair[npair.cond == c]
            if not len(g):
                continue
            lo, hi = boot(g, lambda d: d.narr_v3.mean() - d.narr_v2.mean())
            res["native"]["cells"][c] = dict(n=len(g), v2=g.narr_v2.mean(), v3=g.narr_v3.mean(),
                                             ci=[lo, hi])
            lines.append(f"   {c}: n={len(g)}  v2 {g.narr_v2.mean():.1%}  v3 {g.narr_v3.mean():.1%}  "
                         f"diff {100*(g.narr_v3.mean()-g.narr_v2.mean()):+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]")
        if set(npair.cond) >= {"A", "B"}:
            def vmf(d, v):
                return d[d.cond == "A"][f"narr_{v}"].mean() - d[d.cond == "B"][f"narr_{v}"].mean()
            for v in ("v2", "v3"):
                lo, hi = boot(npair, lambda d: vmf(d, v))
                res["native"][f"video_minus_frames_{v}"] = dict(est=vmf(npair, v), ci=[lo, hi])
                lines.append(f"   video minus frames under {v}: {100*vmf(npair, v):+.1f} pts "
                             f"[{100*lo:+.1f}, {100*hi:+.1f}]")
        # same-day route comparison, each description, each format
        orp = pair[pair.model == "gemini-3-flash-preview"]
        res["native"]["route_sameday"] = {}
        for v in ("v2", "v3"):
            m = npair[key + [f"narr_{v}"]].merge(orp[key + [f"narr_{v}"]], on=key,
                                                    suffixes=("_nat", "_or"))
            for c in ("A", "B"):
                g = m[m.cond == c]
                if len(g):
                    a_, b_ = g[f"narr_{v}_nat"].mean(), g[f"narr_{v}_or"].mean()
                    agr = float((g[f"narr_{v}_nat"] == g[f"narr_{v}_or"]).mean())
                    res["native"]["route_sameday"][f"{v}|{c}"] = dict(n=len(g), native=a_, openrouter=b_,
                                                                       agree=agr)
                    lines.append(f"   same-day route, {v} {c}: native {a_:.1%}  OpenRouter {b_:.1%}")

    # ADDITIONAL, NOT IN THE PLAN: the prompt effect as a model estimate. A logistic model for the
    # exit claim with a fixed effect per cell and a neutral-prompt term, fitted by generalized
    # estimating equations with an exchangeable working correlation within participant, because
    # the three repeats and both prompts for one recording are not independent. The pooled model
    # gives a common odds ratio; the interaction model gives one per cell.
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    long = pd.concat([
        pair.assign(y=pair.narr_v2.astype(int), neutral=0),
        pair.assign(y=pair.narr_v3.astype(int), neutral=1)])
    long["cell"] = long.model.str.split("/").str[-1] + "_" + long.cond
    fam, cov = sm.families.Binomial(), sm.cov_struct.Exchangeable()
    m1 = smf.gee("y ~ C(cell) + neutral", "pid", long, family=fam, cov_struct=cov).fit()
    b, se = m1.params["neutral"], m1.bse["neutral"]
    res["gee_pooled"] = dict(OR=float(np.exp(b)), ci=[float(np.exp(b - 1.96 * se)),
                             float(np.exp(b + 1.96 * se))], p=float(m1.pvalues["neutral"]),
                             rho=float(m1.cov_struct.dep_params))
    lines.append("\nADDITIONAL, NOT IN THE PLAN: GEE logistic, clustered on participant")
    lines.append(f"   pooled neutral-prompt OR {np.exp(b):.2f} [{np.exp(b - 1.96 * se):.2f}, "
                 f"{np.exp(b + 1.96 * se):.2f}], p = {m1.pvalues['neutral']:.2g}, "
                 f"within-participant correlation {float(m1.cov_struct.dep_params):.2f}")
    # Review, 25 Sep 2026. (1) With 29 clusters the robust sandwich is biased downward, so the
    # pooled model is refitted with the Mancl-DeRouen bias-reduced covariance, and that fit is the
    # primary test. (2) The cell odds ratios come from ONE model with a prompt-by-cell interaction
    # (Equation 1 with cell-specific prompt terms), not from separate fits, with a joint Wald test
    # of whether the cells differ. (3) A participant-level sign test, which assumes nothing about
    # the dependence between calls: for each participant, did the neutral description lower or
    # raise that participant's count of exit claims over all paired calls?
    from scipy import stats as sps
    mb = smf.gee("y ~ C(cell) + neutral", "pid", long, family=fam, cov_struct=cov).fit(
        cov_type="bias_reduced")
    b, se = mb.params["neutral"], mb.bse["neutral"]
    res["gee_pooled_bc"] = dict(OR=float(np.exp(b)), ci=[float(np.exp(b - 1.96 * se)),
                                float(np.exp(b + 1.96 * se))], p=float(mb.pvalues["neutral"]))
    lines.append(f"   bias-reduced covariance: OR {np.exp(b):.2f} [{np.exp(b - 1.96 * se):.2f}, "
                 f"{np.exp(b + 1.96 * se):.2f}], p = {mb.pvalues['neutral']:.2g}")
    mi = smf.gee("y ~ C(cell) * neutral", "pid", long, family=fam, cov_struct=cov).fit(
        cov_type="bias_reduced")
    names_ = list(mi.params.index)
    V = mi.cov_params()
    base = sorted(long.cell.unique())[0]
    res["gee_cells"] = {}
    for c in sorted(long.cell.unique()):
        w = np.zeros(len(names_))
        w[names_.index("neutral")] = 1
        if c != base:
            w[names_.index(f"C(cell)[T.{c}]:neutral")] = 1
        est = float(w @ mi.params.values)
        sd = float(np.sqrt(w @ V.values @ w))
        res["gee_cells"][c] = dict(OR=float(np.exp(est)), ci=[float(np.exp(est - 1.96 * sd)),
                                   float(np.exp(est + 1.96 * sd))])
        lines.append(f"   interaction model {c:36s} OR {np.exp(est):.2f} "
                     f"[{np.exp(est - 1.96 * sd):.2f}, {np.exp(est + 1.96 * sd):.2f}]")
    inter = [i for i, n in enumerate(names_) if n.endswith(":neutral")]
    R = np.zeros((len(inter), len(names_)))
    for r_, i in enumerate(inter):
        R[r_, i] = 1
    wald = mi.wald_test(R, scalar=True)
    res["gee_cells_joint_p"] = float(wald.pvalue)
    lines.append(f"   cells differ? joint Wald p = {float(wald.pvalue):.2f}")
    net = pair.groupby("pid").apply(lambda g: int(g.narr_v3.sum()) - int(g.narr_v2.sum()))
    down, up = int((net < 0).sum()), int((net > 0).sum())
    sp = float(sps.binomtest(down, down + up, 0.5).pvalue)
    res["participant_sign"] = dict(down=down, up=up, tied=int((net == 0).sum()), p=sp)
    lines.append(f"   participant sign test: {down} down, {up} up, {int((net == 0).sum())} tied, p = {sp:.2g}")
    (OUT / "C6_neutral_prompt.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    txt = "\n".join(lines)
    (OUT / "C6_report.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
