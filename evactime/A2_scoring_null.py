"""A2 - the chance level of each scoring rule at every matching window.

The permissive ("any") rule scores an attention claim correct when any reference sample in the
window W_i(D) = [t_i - D, t_i + D] carries the claimed class. Because W_i(D) is contained in
W_i(D') for D < D', the rule cannot turn a correct assertion into a wrong one as the window widens.
A rising curve is therefore not evidence of accuracy. The test is against its own null: the same
windows scored against claims permuted within the cell (participant x model x input format), which
keeps each participant's gaze and each model's claim mix and removes only the link between them.
The modal rule is given the same null, so both rules are compared on chance-corrected agreement.

Review, 25 Sep 2026:
  * One fixed cohort. An assertion enters only if its window holds a tagged sample at EVERY
    width, so the denominator no longer changes with the window (it used to run from 1,098 to
    1,376, and part of the modal decline was that change). The per-window denominators of the
    unrestricted scoring are still written, for the record.
  * The null is stratified by cell, as in C2, with 5,000 permutations (was 2,000, participant).
  * The mean of the null has a closed form (the mean over the stratum of the hit of each claim
    against every window in the stratum); permutations give the ranges and p-values.
  * Dwell sensitivity. The modal object of a +/-1 s window can hold a small share of the samples.
    Agreement is also given for windows in which the modal object holds at least half, and at
    least three quarters, of the tagged samples, as a check that short glances do not drive it.

Outputs: A2_scoring_null.json, A2_scoring_null.csv
Run: py evactime/A2_scoring_null.py
"""
import json, pathlib
import numpy as np
import pandas as pd

import A1_attention_scoring as A1

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
TOLS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
NPERM, SEED = 5000, 20260923
CLASSES = ("alarm", "npc", "sign")
STRATUM = ["pid", "model", "cond"]


def permuted_means(H, claim_ix, sid, rng, n_perm):
    """Mean hit under within-stratum permutations of the claims. H[i, c] = hit if claim were c."""
    n = len(claim_ix)
    order = np.argsort(sid, kind="stable")
    out = np.empty(n_perm)
    rows = np.arange(n)
    for b in range(n_perm):
        perm = np.lexsort((rng.random(n), sid))
        src = np.empty(n, dtype=int)
        src[order] = perm
        out[b] = H[rows, claim_ix[src]].mean()
    return out


def exact_chance(H, claim_ix, sid):
    """Closed-form mean of the permutation null: within a stratum, claim j meets window i w.p. 1/n."""
    e = np.empty(len(claim_ix))
    for s in np.unique(sid):
        ix = np.where(sid == s)[0]
        e[ix] = H[np.ix_(ix, claim_ix[ix])].mean(axis=1)
    return e


def main():
    a = pd.read_csv(OUT / "A1_assertions_scored.csv")
    a = a[a.timed & a.t_unity.notna()].copy().reset_index(drop=True)
    ref = A1.load_reference()
    # Same rule as Equation 2 and C2: only samples that struck a tagged object with a position in
    # the parsed scene enter the window, and the modal object is taken by identifier, then mapped
    # to its class.
    geom = json.load(open(OUT / "scene_geometry.json", encoding="utf-8"))
    POS = {int(k) for k, v in geom.items() if v.get("pos")}
    ref = ref[ref.HitObjectID.isin(POS)]
    by = {p: g for p, g in ref.groupby("pid")}
    a["want"] = a.type.map(A1.CLAIM)
    claim_ix = a.want.map({c: i for i, c in enumerate(CLASSES)}).values
    sid = a.groupby(STRATUM).ngroup().values

    n = len(a)
    ok = {t: np.zeros(n, bool) for t in TOLS}
    Hany = {t: np.zeros((n, 3), bool) for t in TOLS}
    Hmod = {t: np.zeros((n, 3), bool) for t in TOLS}
    mshare = np.full(n, np.nan)
    for i, r in a.iterrows():
        g = by[r.pid]
        for t in TOLS:
            w = g[(g.UnityTime >= r.t_unity - t) & (g.UnityTime <= r.t_unity + t)]
            if not len(w):
                continue
            ok[t][i] = True
            present = set(w.cls)
            # pandas mode() breaks ties toward the lower identifier, the same rule as C2
            mid = int(w.HitObjectID.mode().iloc[0])
            mcls = A1.SUPER.get(mid)
            for c, cname in enumerate(CLASSES):
                Hany[t][i, c] = cname in present
                Hmod[t][i, c] = cname == mcls
            if t == 1.0:
                mshare[i] = float((w.HitObjectID == mid).mean())

    rng = np.random.default_rng(SEED)
    fixed = np.logical_and.reduce([ok[t] for t in TOLS])
    rows, res = [], {"cohort": dict(fixed_n=int(fixed.sum()),
                                    open_n={f"{t:g}": int(ok[t].sum()) for t in TOLS})}
    k = lambda o, e: (o - e) / (1 - e)
    cix, sfx = claim_ix[fixed], sid[fixed]
    for t in TOLS:
        R = {}
        for rule, H in (("any", Hany[t]), ("modal", Hmod[t])):
            Hf = H[fixed]
            obs = float(Hf[np.arange(len(cix)), cix].mean())
            e = float(exact_chance(Hf, cix, sfx).mean())
            nl = permuted_means(Hf, cix, sfx, rng, NPERM)
            R[rule] = obs
            R[f"{rule}_null"] = e
            R[f"{rule}_null_ci"] = [float(x) for x in np.percentile(nl, [2.5, 97.5])]
            R[f"kappa_{rule}"] = k(obs, e)
            # the open cohort, as previously reported, for comparison only
            Ho = H[ok[t]]
            R[f"{rule}_open"] = float(Ho[np.arange(ok[t].sum()), claim_ix[ok[t]]].mean())
        R["n"] = int(fixed.sum())
        R["n_open"] = int(ok[t].sum())
        res[f"{t:g}"] = R
        rows.append(dict(tol=t, **{kk: v for kk, v in R.items() if not kk.endswith("_ci")}))
        print(f"+/-{t:<4g} n={fixed.sum():4d} (open {ok[t].sum():4d})  any {R['any']:.1%} "
              f"(null {R['any_null']:.1%}) kappa {R['kappa_any']:.1%}   modal {R['modal']:.1%} "
              f"(null {R['modal_null']:.1%}) kappa {R['kappa_modal']:.1%}")

    # dwell sensitivity at +/-1 s, on the open cohort of that window (the C2 set)
    dw = {}
    base = ok[1.0]
    for thr in (0.0, 0.5, 0.75):
        m = base & (np.nan_to_num(mshare) >= thr)
        Hm = Hmod[1.0][m]
        o = float(Hm[np.arange(m.sum()), claim_ix[m]].mean())
        e = float(exact_chance(Hm, claim_ix[m], sid[m]).mean())
        dw[f"{thr:g}"] = dict(n=int(m.sum()), share=float(m.sum() / base.sum()), modal=o,
                              chance=e, kappa=k(o, e))
        print(f"modal share >= {thr:.2f}: n={m.sum():4d} ({m.sum() / base.sum():.0%})  modal {o:.1%}"
              f"  chance {e:.1%}  kappa {k(o, e):.1%}")
    res["dwell_1s"] = dw
    res["median_modal_share_1s"] = float(np.nanmedian(mshare[base]))
    print(f"median share of the modal object in a +/-1 s window: {res['median_modal_share_1s']:.0%}")
    pd.DataFrame(rows).to_csv(OUT / "A2_scoring_null.csv", index=False)
    (OUT / "A2_scoring_null.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
