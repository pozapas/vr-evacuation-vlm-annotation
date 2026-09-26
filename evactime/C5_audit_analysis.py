"""C5: the human audit of narrative endings.

Reads the returned Prolific submissions and answers one question: do naive viewers agree with the
game engine that the recordings do not end with the participant leaving the room, and does that
agreement corroborate the regex behind the 92% script-completion figure?

Three decisions are baked in here rather than left to the caller, because each of them changes the
answer and each needs to be defensible in review.

1. Q1 is recoded to a binary. The engine says no transit occurs. Of the four options offered,
   only `through_door` contradicts that: `same_room` and `other` both encode "did not leave", and
   the most careful coder in the set chose `other` on 29 of 54 items while writing notes like
   "does not exit" and "ends up back on chair". Scoring `other` as an error would have excluded
   the best annotator in the study.

2. Coders are screened on the attention checks only, never on whether they agree. Excluding the
   dissenting faction would let the audit confirm only the regex it exists to test.

3. Confidence intervals are bootstrapped over CODERS, not items. The disagreement in this study is
   between people, not between clips (see the response-set test below), so resampling items would
   understate it badly.
"""
import json, pathlib, collections, itertools, re
import numpy as np
import pandas as pd

from score_audit import krippendorff_nominal, NOT_PARTICIPANTS, BAD_ITEMS
from transit_classifier import asserts_transit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
RET = ROOT / "audit_returns"
B = 5000
RNG = np.random.default_rng(20260914)


def load():
    key = {k["id"]: k for k in json.load(open(OUT / "audit_item_key.json", encoding="utf-8"))}
    catch = {k["id"]: k.get("expected_q2") for k in key.values() if k.get("catch")}
    coders, failed = {}, []
    for f in sorted(RET.glob("*.json")):
        who = f.stem
        if who in NOT_PARTICIPANTS:
            continue
        a = json.load(open(f, encoding="utf-8")).get("answers", {})
        if len(a) < 58:
            continue
        if not all(a.get(c, {}).get("q2") == e for c, e in catch.items()):
            failed.append(who)
            continue
        coders[who] = a
    ids = [i for i in key if not key[i].get("catch") and i not in BAD_ITEMS]
    return key, ids, coders, failed


def main():
    key, ids, coders, failed = load()
    names = sorted(coders)
    P = lambda *a: print(*a)
    P(f"coders retained {len(names)}, excluded on attention checks {len(failed)} {failed}")
    P(f"items scored {len(ids)} (dropped {sorted(BAD_ITEMS)}: {set(BAD_ITEMS.values())})\n")

    exit_ = lambda v: None if v == "unsure" else (v == "through_door")
    ok_ = lambda v: None if v == "unsure" else (v == "correct")

    rate = {c: np.mean([coders[c][i]["q1"] == "through_door" for i in ids]) for c in names}
    order = sorted(names, key=lambda c: rate[c])
    P("FACTION STRUCTURE (share of items called 'goes through a doorway')")
    for c in order:
        P(f"  {c:26s} {rate[c]:6.1%}  {'#' * int(round(rate[c] * 40))}")
    dis = [c for c in names if rate[c] > 0.5]
    maj = [c for c in names if rate[c] <= 0.5]
    P(f"\n  majority faction {len(maj)}, dissenting faction {len(dis)}")

    a1 = krippendorff_nominal([[exit_(coders[c][i]["q1"]) for c in names] for i in ids])
    a2 = krippendorff_nominal([[ok_(coders[c][i]["q2"]) for c in names] for i in ids])
    P(f"\nKRIPPENDORFF ALPHA   Q1 binary 'did they exit' {a1:+.3f}   Q2 'ending correct' {a2:+.3f}")

    def verdicts(pool):
        v1, v2 = {}, {}
        for i in ids:
            u = [exit_(coders[c][i]["q1"]) for c in pool]
            u = [x for x in u if x is not None]
            v1[i] = "exit" if sum(u) > len(u) / 2 else ("no exit" if sum(u) < len(u) / 2 else "tied")
            w = [ok_(coders[c][i]["q2"]) for c in pool]
            w = [x for x in w if x is not None]
            v2[i] = "correct" if sum(w) > len(w) / 2 else ("not correct" if sum(w) < len(w) / 2 else "tied")
        return v1, v2

    v1, v2 = verdicts(names)
    P(f"\nMAJORITY VERDICT, all {len(names)} coders")
    P(f"  Q1 did they exit      : {dict(collections.Counter(v1.values()))}")
    P(f"  Q2 ending correct     : {dict(collections.Counter(v2.values()))}")

    tr = [i for i in ids if asserts_transit(key[i]["narrative"])]
    def frac(pool):
        _, w = verdicts(pool)
        return np.mean([w[i] == "not correct" for i in tr])
    pt = frac(names)
    boot = []
    for _ in range(B):
        s = list(RNG.choice(names, len(names), replace=True))
        boot.append(frac(s))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    P(f"\nCORROBORATION OF THE REGEX")
    P(f"  narratives the regex flags as asserting a transit: {len(tr)} of {len(ids)}"
      f" ({len(tr)/len(ids):.0%})   [paper's figure: 92%]")
    P(f"  of those, judged NOT correct by majority vote: {pt:.0%}"
      f"   95% CI over coders [{lo:.0%}, {hi:.0%}]")
    P(f"  same figure using only the majority faction (n={len(maj)}): {frac(maj):.0%}")
    P(f"  same figure using only the dissenting faction (n={len(dis)}): {frac(dis):.0%}")

    if dis:
        p = float(np.mean([rate[c] for c in dis]))
        cnt = collections.Counter(sum(coders[c][i]["q1"] == "through_door" for c in dis) for i in ids)
        P(f"\nIS THE DISSENT ITEM-DRIVEN OR A CODER RESPONSE SET? (n={len(dis)}, mean rate {p:.0%})")
        from math import comb
        for k in range(len(dis) + 1):
            e = comb(len(dis), k) * p ** k * (1 - p) ** (len(dis) - k) * len(ids)
            P(f"  {k} of {len(dis)} call exit on the same item: observed {cnt.get(k,0):3d}   independent {e:5.1f}")
        P("  matching the independent column means coder-level responding, not clip ambiguity")

    rows = [dict(id=i, pid=key[i]["pid"], model=key[i]["model"], cond=key[i]["cond"],
                 regex_transit=asserts_transit(key[i]["narrative"]),
                 verdict_exit=v1[i], verdict_ending=v2[i],
                 n_exit=sum(coders[c][i]["q1"] == "through_door" for c in names)) for i in ids]
    pd.DataFrame(rows).to_csv(OUT / "audit_final.csv", index=False)
    # Intervals for the agreement coefficients, bootstrapped over coders like every other interval
    # here, and the per-item share of coders judging each flagged narrative wrong, which the audit
    # table draws as a distribution rather than a single pooled percentage.
    a_boot1, a_boot2 = [], []
    for _ in range(2000):
        s = list(RNG.choice(names, len(names), replace=True))
        a_boot1.append(krippendorff_nominal([[exit_(coders[c][i]["q1"]) for c in s] for i in ids]))
        a_boot2.append(krippendorff_nominal([[ok_(coders[c][i]["q2"]) for c in s] for i in ids]))
    a1_ci = [float(x) for x in np.nanpercentile(a_boot1, [2.5, 97.5])]
    a2_ci = [float(x) for x in np.nanpercentile(a_boot2, [2.5, 97.5])]
    P(f"  alpha intervals over coders: Q1 [{a1_ci[0]:+.2f}, {a1_ci[1]:+.2f}]  Q2 [{a2_ci[0]:+.2f}, {a2_ci[1]:+.2f}]")
    item_wrong = []
    for i in tr:
        w = [ok_(coders[c][i]["q2"]) for c in names]
        w = [x for x in w if x is not None]
        item_wrong.append(float(np.mean([not x for x in w])) if w else float("nan"))

    def frac_ci(pool):
        bs = [frac(list(RNG.choice(pool, len(pool), replace=True))) for _ in range(2000)]
        return [float(x) for x in np.percentile(bs, [2.5, 97.5])]
    maj_ci, dis_ci = frac_ci(maj), frac_ci(dis)

    # ---- review, 25 Sep 2026: report the codings, not only the binary ones
    # Q2 had three substantive answers. "Judged wrong" above means "not Correct", so a "Partly"
    # counts as wrong. The three-level result is reported beside it: the plurality answer per
    # flagged narrative, and the share for which a majority answered "Incorrect".
    lvl = ("correct", "partly", "incorrect")
    plur, maj_incorrect = collections.Counter(), 0
    for i in tr:
        w = [coders[c][i]["q2"] for c in names if coders[c][i]["q2"] in lvl]
        cnt = collections.Counter(w)
        top = max(lvl, key=lambda v: cnt[v])
        plur["tie" if sum(cnt[v] == cnt[top] for v in lvl) > 1 else top] += 1
        maj_incorrect += cnt["incorrect"] > len(w) / 2
    q2_levels = {v: float(np.mean([coders[c][i]["q2"] == v for c in names for i in tr]))
                 for v in lvl + ("unsure",)}
    P(f"  Q2 answers on flagged narratives: " + ", ".join(f"{k} {v:.0%}" for k, v in q2_levels.items()))
    P(f"  plurality answer per flagged narrative: {dict(plur)}")
    P(f"  majority 'Incorrect': {maj_incorrect} of {len(tr)} ({maj_incorrect / len(tr):.0%})")

    # Q1 alpha under the original categories, to show the binary recoding does not drive it
    q1_3 = lambda v: None if v == "unsure" else ("left" if v == "through_door" else "stayed")
    a1_cat3 = krippendorff_nominal([[None if coders[c][i]["q1"] == "unsure" else coders[c][i]["q1"]
                                     for c in names] for i in ids])
    a1_cat4 = krippendorff_nominal([[coders[c][i]["q1"] for c in names] for i in ids])
    other_max = max(sum(coders[c][i]["q1"] == "other" for i in ids) for c in names)
    P(f"  Q1 alpha, three categories {a1_cat3:+.3f}, four categories {a1_cat4:+.3f};"
      f" most 'other' answers by one coder {other_max} of {len(ids)}")

    # crossed bootstrap, coders and flagged items resampled together
    cross = []
    for _ in range(B):
        s = list(RNG.choice(names, len(names), replace=True))
        its = list(RNG.choice(tr, len(tr), replace=True))
        _, w = verdicts(s)
        cross.append(np.mean([w[i] == "not correct" for i in its]))
    cross_ci = [float(x) for x in np.percentile(cross, [2.5, 97.5])]
    P(f"  judged not correct {pt:.0%}, crossed coder x item bootstrap [{cross_ci[0]:.0%}, {cross_ci[1]:.0%}]")
    json.dump(dict(n_coders=len(names), n_items=len(ids), excluded_attention=failed,
                   alpha_q1_ci=a1_ci, alpha_q2_ci=a2_ci, item_share_wrong=item_wrong,
                   corroboration_majority_ci=maj_ci, corroboration_dissent_ci=dis_ci,
                   dropped_items=sorted(BAD_ITEMS), alpha_q1=a1, alpha_q2=a2,
                   exit_rate={c: rate[c] for c in order}, regex_transit_n=len(tr),
                   corroboration=pt, ci=[lo, hi], ci_crossed=cross_ci,
                   q2_levels_flagged=q2_levels, q2_plurality_flagged=dict(plur),
                   q2_majority_incorrect=int(maj_incorrect),
                   alpha_q1_cat3=a1_cat3, alpha_q1_cat4=a1_cat4, other_max=int(other_max),
                   # within-faction figures are the ones that carry the claim, because the
                   # pooled majority verdict tracks whichever faction is larger by construction
                   corroboration_majority=frac(maj), corroboration_dissent=frac(dis)),
              open(OUT / "audit_final.json", "w"), indent=1)
    P(f"\nwrote {OUT/'audit_final.csv'} and audit_final.json")


if __name__ == "__main__":
    main()
