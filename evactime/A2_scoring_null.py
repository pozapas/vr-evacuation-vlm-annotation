"""A2 - the chance level of each scoring rule at every matching window.

The permissive ("any") rule scores an attention claim correct when any reference sample in the
window W_i(D) = [t_i - D, t_i + D] carries the claimed class. Because W_i(D) is contained in
W_i(D') for D < D', its agreement cannot decrease as the window widens, whatever the claims are.
A rising curve is therefore not evidence of accuracy. The test is against its own null: the same
windows scored against claims permuted within participant, which keeps each participant's gaze and
each model's output rates and removes only the link between them. The modal rule is given the
same null, so both rules are compared on chance-corrected agreement.

Gaze samples at 48 Hz are strongly autocorrelated, so no independence-based formula for the
inflation is fitted. The permutation null is exact for the data as they are.

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
NPERM, SEED = 2000, 20260923


def main():
    a = pd.read_csv(OUT / "A1_assertions_scored.csv")
    a = a[a.timed & a.t_unity.notna()].copy()
    ref = A1.load_reference()
    # Same rule as Equation 2 and C2: only samples that struck a tagged object with a position in
    # the parsed scene enter the window, and the modal object is taken by identifier, then mapped
    # to its class. Scoring over all samples instead gave a different modal value at 1 s (15.6%
    # against 23.9%), so the paper would have reported two numbers for one defined rule.
    geom = json.load(open(OUT / "scene_geometry.json", encoding="utf-8"))
    POS = {int(k) for k, v in geom.items() if v.get("pos")}
    ref = ref[ref.HitObjectID.isin(POS)]
    by = {p: g for p, g in ref.groupby("pid")}
    a["want"] = a.type.map(A1.CLAIM)

    # per assertion and window, the set of classes present and the modal class
    present, modal = {t: [] for t in TOLS}, {t: [] for t in TOLS}
    for _, r in a.iterrows():
        g = by[r.pid]
        for t in TOLS:
            w = g[(g.UnityTime >= r.t_unity - t) & (g.UnityTime <= r.t_unity + t)]
            present[t].append(frozenset(w.cls) if len(w) else None)
            m = w.HitObjectID.mode()
            modal[t].append(A1.SUPER.get(int(m.iloc[0])) if len(w) and len(m) else None)

    rng = np.random.default_rng(SEED)
    pids = a.pid.values
    want = a.want.values
    groups = [np.where(pids == p)[0] for p in np.unique(pids)]
    rows, res = [], {}
    for t in TOLS:
        ok = np.array([s is not None for s in present[t]])
        P = [present[t][i] for i in range(len(a))]
        M = [modal[t][i] for i in range(len(a))]

        def score(claims):
            anyv = np.array([ok[i] and claims[i] in P[i] for i in range(len(a))])[ok]
            modv = np.array([ok[i] and claims[i] == M[i] for i in range(len(a))])[ok]
            return anyv.mean(), modv.mean()

        obs_any, obs_mod = score(want)
        null_any, null_mod = [], []
        # permute only among the windows scored at this tolerance, so the null keeps the claim
        # marginals of exactly the assertions it is compared with
        vgroups = [idx[ok[idx]] for idx in groups]
        for _ in range(NPERM):
            perm = want.copy()
            for idx in vgroups:
                perm[idx] = rng.permutation(perm[idx])
            x, y = score(perm)
            null_any.append(x)
            null_mod.append(y)
        na, nm = np.array(null_any), np.array(null_mod)
        k = lambda o, n: (o - n.mean()) / (1 - n.mean())
        res[f"{t:g}"] = dict(n=int(ok.sum()), any=obs_any, any_null=na.mean(),
                             any_null_ci=list(np.percentile(na, [2.5, 97.5])),
                             modal=obs_mod, modal_null=nm.mean(),
                             modal_null_ci=list(np.percentile(nm, [2.5, 97.5])),
                             kappa_any=k(obs_any, na), kappa_modal=k(obs_mod, nm))
        rows.append(dict(tol=t, **{kk: v for kk, v in res[f"{t:g}"].items() if not kk.endswith("_ci")}))
        print(f"+/-{t:<4g} n={ok.sum():4d}  any {obs_any:.1%} (null {na.mean():.1%})  "
              f"kappa {k(obs_any, na):.1%}   modal {obs_mod:.1%} (null {nm.mean():.1%})  "
              f"kappa {k(obs_mod, nm):.1%}")
    pd.DataFrame(rows).to_csv(OUT / "A2_scoring_null.csv", index=False)
    (OUT / "A2_scoring_null.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
