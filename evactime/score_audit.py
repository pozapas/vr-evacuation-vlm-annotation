"""Score the returned human-audit files.

Usage:  py evactime/score_audit.py audit_returns/*.json

Submissions arrive either as files the coder downloaded, or from the Vercel Blob store the
Prolific version writes to. Either way, drop the .json files in audit_returns/ and run this.

Attention-check items (ids beginning with C) are screened first: any coder who fails one is
excluded entirely before agreement is computed, and the checks themselves never enter the
statistics.

Joins each coder's blinded answers back to `outputs/audit_item_key.json` by item id, then:
  - Krippendorff's alpha (nominal) per question, across however many coders returned files
  - majority-vote consensus per item
  - agreement between the human consensus and the regex used for the 92% figure

The regex is what the paper currently rests on. The point of the audit is to find out whether
people agree with it, so that comparison is the headline output here.
"""
import json, sys, glob, pathlib, itertools, collections
import numpy as np
import pandas as pd

from transit_classifier import asserts_transit

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Deployment probes and pilot runs write to the same Blob store as real submissions, so they
# have to be named and dropped. A co-author ran the study once on 8 Sep 2026 to check it worked
# before publishing it; that run is a test of the app, not a coding attempt, and must not be read
# as evidence about what naive coders do.
# P15's clip is degenerate and its items must not be scored. The scene cut lands 1.2 s into
# the 12 s clip instead of the intended ~7.5 s, so coders saw almost no pre-cut footage, and P15
# is also the one participant who never left the room. Every other clip cuts within 0.8 s of
# target (median 7.48 s). One coder caught it unprompted, noting "didnt get off chair".
BAD_ITEMS = {"I016": "P15, cut at 1.2 s", "I030": "P15, cut at 1.2 s"}

NOT_PARTICIPANTS = {
    "PILOT01": "co-author pilot run, 8 Sep 2026",
    "HEALTHCHECK": "deployment probe",
    "LIVECHECK01": "deployment probe",
    "POSTDEPLOY": "deployment probe",
    "FINALCHECK": "deployment probe",
}

OUT = ROOT / "evactime" / "outputs"

# same pattern as C4_script_completion.py
import re
TRANSIT = re.compile(
    r"pass(?:es|ed|ing)?\s+through|walk(?:s|ed|ing)?\s+through|step(?:s|ped|ping)?\s+(?:out|through)"
    r"|exits?\s+(?:the\s+)?(?:room|door|building)|goes?\s+through|through\s+the\s+door"
    r"|into\s+the\s+(?:corridor|hallway)|leaves?\s+the\s+room|to\s+safety|escap(?:es|ed|ing)",
    re.I)


def krippendorff_nominal(units):
    """units: list of lists of labels (one list per item, one label per coder who rated it).
    Standard nominal-scale alpha. Items with fewer than two ratings contribute nothing."""
    units = [[v for v in u if v is not None] for u in units]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    vals = sorted({v for u in units for v in u})
    idx = {v: i for i, v in enumerate(vals)}
    k = len(vals)
    Do = 0.0
    n_total = 0
    coincidence = np.zeros((k, k))
    for u in units:
        m = len(u)
        for a, b in itertools.permutations(u, 2):
            coincidence[idx[a], idx[b]] += 1.0 / (m - 1)
        n_total += m
    n = coincidence.sum()
    Do = (n - np.trace(coincidence)) / n
    marg = coincidence.sum(axis=1)
    De = 1.0 - (marg * (marg - 1)).sum() / (n * (n - 1))
    return 1.0 - Do / De if De > 0 else float("nan")


def main(paths):
    key = {k["id"]: k for k in json.load(open(OUT / "audit_item_key.json", encoding="utf-8"))}
    subs = []
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        who = d.get("coder") or (d.get("prolific") or {}).get("pid") or "anon"
        if who in NOT_PARTICIPANTS:
            print(f"skipping {who}: {NOT_PARTICIPANTS[who]}")
            continue
        for iid, a in d["answers"].items():
            if iid in BAD_ITEMS:
                continue

            subs.append(dict(coder=who, id=iid, **{q: a.get(q) for q in ("q1", "q2", "q3")},
                             note=a.get("note", "")))
    df = pd.DataFrame(subs)
    if df.empty:
        print("no answers found"); return

    # ---- attention checks first. Paid raters move faster than colleagues, so screen before
    # scoring anything. A coder who misses either check is dropped entirely, and the check items
    # never enter the statistics.
    catch = {k["id"]: k.get("expected_q2") for k in key.values() if k.get("catch")}
    keep = []
    if catch:
        print("ATTENTION CHECKS")
        for c, g in df[df.id.isin(catch)].groupby("coder"):
            hits = sum(1 for _, r in g.iterrows() if r.q2 == catch.get(r.id))
            ok = hits == len(catch)
            print(f"  {c:26s} {hits}/{len(catch)} passed   {'keep' if ok else 'EXCLUDE'}")
            if ok:
                keep.append(c)
        dropped = sorted(set(df.coder) - set(keep))
        if dropped:
            print(f"  dropping {len(dropped)}: {', '.join(dropped)}")
        df = df[df.coder.isin(keep) & ~df.id.isin(catch)]
        if df.empty:
            print("\nno usable submissions after screening"); return


    # ---- q1 is reported, never screened on. Two things make a screen wrong here.
    # First, `other` is a legitimate answer: the clip ends with the scene resetting the
    # participant to their start position, and the most careful coder in the first six chose
    # `other` on 29 of 56 items while writing notes like "does not exit" and "ends up back
    # on chair". Only `through_door` actually contradicts the engine. Second, and more
    # important, excluding coders who disagree would let the audit only ever confirm the
    # regex it exists to test. So the split is reported and left in.
    print("\nQ1: DOES THE CODER SEE AN EXIT? (the engine says no transit occurs)")
    for c, g in df.groupby("coder"):
        n = len(g)
        ex = int((g.q1 == "through_door").sum())
        print(f"  {c:26s} sees an exit on {ex:3d}/{n:3d} = {ex / n:5.1%}")

    coders = sorted(df.coder.unique())
    print(f"\ncoders retained: {len(coders)}  ({', '.join(coders)})")
    print(f"items rated: {df.id.nunique()}   ratings: {len(df)}\n")

    print("KRIPPENDORFF'S ALPHA (nominal)")
    for q, lab in [("q1", "what happens at the end"), ("q2", "does the description match")]:
        units = [g[q].tolist() for _, g in df.groupby("id")]
        print(f"  {q}  {lab:32s} alpha = {krippendorff_nominal(units):.3f}")

    # majority vote
    cons = {}
    for iid, g in df.groupby("id"):
        for q in ("q1", "q2"):
            c = collections.Counter([v for v in g[q] if v])
            top, n = (c.most_common(1) or [(None, 0)])[0]
            cons[(iid, q)] = top if n > len(g) / 2 else "tied"

    print("\nHUMAN CONSENSUS")
    for q in ("q1", "q2"):
        c = collections.Counter(cons[(i, q)] for i in df.id.unique())
        print(f"  {q}: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))

    # the comparison that matters: humans vs the regex behind the 92%
    rows = []
    for iid in df.id.unique():
        k = key.get(iid)
        if not k:
            continue
        regex_says_transit = asserts_transit(k["narrative"])
        human = cons[(iid, "q2")]
        rows.append(dict(id=iid, model=k["model"], cond=k["cond"], pid=k["pid"],
                         regex_transit=regex_says_transit, human_q2=human,
                         human_end=cons[(iid, "q1")]))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "audit_scored.csv", index=False)

    print("\nREGEX vs HUMAN CONSENSUS  (the 92% figure is the regex)")
    both = r[r.human_q2.isin(["correct", "partly", "incorrect"])]
    tab = pd.crosstab(both.regex_transit, both.human_q2)
    print(tab.to_string())
    agree = both[(both.regex_transit) & (both.human_q2 == "incorrect")]
    if len(both):
        print(f"\n  of {int(both.regex_transit.sum())} narratives the regex flags as asserting a "
              f"transit,\n  {len(agree)} were judged 'describes something that does not happen' "
              f"by the coders ({len(agree) / max(1, int(both.regex_transit.sum())):.0%}).")
    print(f"\nwrote {OUT / 'audit_scored.csv'}")


if __name__ == "__main__":
    args = sys.argv[1:] or glob.glob(str(ROOT / "audit_returns" / "*.json"))
    if not args:
        print("no input files; put the coders' JSON files in audit_returns/")
    else:
        main(args)
