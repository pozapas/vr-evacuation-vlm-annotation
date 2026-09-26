"""C2 (reframed) - what VLM attention annotation actually resolves.

WHY THIS REPLACES THE "RESOLUTION LIMIT" FRAMING
------------------------------------------------
The earlier analysis converted each categorical error into an angular distance and called the 80th
percentile of that distribution a "resolution limit of 19-20 degrees, invariant across models,
conditions and phases". Internal review rejected it, and the data agree:

  1. Circular by construction. `sep` is 0 whenever the class is right and a fixed room distance
     whenever it is wrong (max among correct rows 1.2e-6; min among incorrect 6.12). So accuracy
     against angle is a step function at zero by definition and no threshold curve exists.
  2. It is a scene constant. The variable has 17 atoms, three of which are 47% of the corpus, and
     percentiles 80, 85 and 90 are all 20.1. Holding the error distribution fixed and varying only
     accuracy, p80 stays in 19-20 degrees for any accuracy from 0% to 70%. Every model falls in
     that range, so "invariance across models" is forced and carries no information.
  3. No object pair in the scene lies between 0 and 6.12 degrees, so no threshold in that range is
     locatable even in principle.

What survives is the idea, not the number: the models identify the REGION of the scene attended to
far better than they identify the OBJECT. This file establishes that against an explicit chance
baseline, which is the evidence the angular framing never supplied.

Outputs: C2_confusion.csv, C2_confusion_report.txt
"""
import json, pathlib, collections
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
RNG = np.random.default_rng(20260916)
B = 5000


def main():
    d = pd.read_csv(OUT / "C2_resolution.csv")
    L = []
    P = L.append
    P("C2 - what the annotation resolves: region, not object")
    P("=" * 70)
    P(f"assertions scored: {len(d)}   participants: {d.pid.nunique()}   models: {d.model.nunique()}")
    P("")

    # ---- the positive control and the confusion set, side by side
    P("EXACT-CLASS ACCURACY BY ASSERTION TYPE")
    P(f"  {'assertion':22s}{'n':>6}{'correct':>10}   what it is confused with")
    for t, g in sorted(d.groupby("type"), key=lambda kv: -kv[1].correct.mean()):
        wrong = collections.Counter(g[~g.correct].actual_cls).most_common(3)
        tail = ", ".join(f"{k} {v}" for k, v in wrong) or "nothing"
        P(f"  {t:22s}{len(g):6d}{g.correct.mean():9.1%}   {tail}")
    P("")
    P("  The NPCs are the positive control: free-standing, far from anything else, and named")
    P("  correctly most of the time. The alarm and the exit sign sit in a cluster of wall-mounted")
    P("  cues and are not distinguished from one another.")
    P("")

    # ---- the confusion matrix itself
    P("CLASS CONFUSION MATRIX (rows: claimed, columns: reference)")
    cm = pd.crosstab(d.claimed_cls, d.actual_cls)
    P(cm.to_string())
    P("")
    P("PER-CLASS PRECISION (of the assertions naming this class, how many were right)")
    for c in cm.index:
        n = cm.loc[c].sum()
        hit = cm.loc[c, c] if c in cm.columns else 0
        P(f"  {c:12s} {hit:5d} / {n:5d} = {hit / n:6.1%}")
    P("")

    # ---- region versus object
    # The region must be defined by OBJECT ID, not by class label. The class "door" covers both
    # the exit door on the back wall and the doctor's door on the opposite side of the room, so
    # treating the label as a region would count a doctor's-door gaze as a back-wall hit and
    # inflate the result. Back wall = exit sign, alarm (visual and audible), queue display,
    # extinguisher, exit door. Outside it = doctor's door and the two NPCs.
    BACK_WALL = {3, 4, 6, 10, 11, 13}
    NPC_IDS = {8, 9}
    d = d.copy()
    # "Right part of the scene" depends on what was claimed. A model asserting an NPC is in the
    # right region if gaze was on either NPC; a model asserting a wall cue is in the right region
    # if gaze was on any back-wall object. Keying this on object-ID equality was wrong: an NPC
    # assertion can be class-correct while naming the other NPC, which scored the positive control
    # at 0% wall accuracy and would have been obvious nonsense in the published table.
    d["region_hit"] = [
        (int(a) in NPC_IDS) if c == "npc" else (int(a) in BACK_WALL)
        for a, c in zip(d.actual, d.claimed_cls)]
    P("REGION VERSUS OBJECT")
    P(f"  names the exact object              {d.correct.mean():6.1%}")
    P(f"  names the right part of the scene   {d.region_hit.mean():6.1%}")
    P("  (right part = the back wall carrying the exit sign, both alarm components, the queue")
    P("   display, the extinguisher and the exit door, for a claim about a wall cue; either NPC")
    P("   for a claim about an NPC. The doctor's door is in neither.)")
    outside = d[~d.actual.isin(BACK_WALL)]
    P(f"  assertions where gaze was NOT on the back wall: {len(outside)}"
      f"  ({len(outside) / len(d):.0%}), exact accuracy {outside.correct.mean():.1%}")
    P("")

    # ---- the chance baseline, which is what makes 'better than nothing' a claim and not a hope
    #
    # Review, 25 Sep 2026. The earlier null permuted claims within participant only, which moved
    # claims between models, input formats and repeats. The primary null now permutes within the
    # cell (participant x model x input format), so each model keeps its own claim mix for each
    # participant and format and only the link between claim and gaze window is removed. The
    # participant-only null and a stricter within-call null are kept as sensitivity results.
    #
    # The mean of a permutation null has a closed form. Within a stratum of n assertions, a
    # permuted claim i meets reference j with probability 1/n, so its expected hit is
    # e_i = (1/n) sum_j hit(claim_i, ref_j). Chance is the mean of e_i, and because e_i depends
    # only on the assertion's own stratum it can be carried through a participant bootstrap.
    # Permutations are still drawn, for the p-values and the null ranges.
    NPC_IDS_ARR = np.array(sorted(NPC_IDS))
    BW_ARR = np.array(sorted(BACK_WALL))

    def hit_obj(claims, ref_ids, ref_cls):
        return claims == ref_cls

    def hit_reg(claims, ref_ids, ref_cls):
        return np.where(claims == "npc", np.isin(ref_ids, NPC_IDS_ARR), np.isin(ref_ids, BW_ARR))

    STRATA = {"cell": ["pid", "model", "cond"], "participant": ["pid"],
              "call": ["pid", "model", "cond", "rep"]}

    def expected(d, keys, hit):
        e = np.empty(len(d))
        for _, g in d.groupby(keys):
            idx = d.index.get_indexer(g.index)
            c = g.claimed_cls.to_numpy(dtype=object)
            h = hit(c[:, None], g.actual.to_numpy()[None, :], g.actual_cls.to_numpy(dtype=object)[None, :])
            e[idx] = h.mean(axis=1)
        return e

    def permute(d, keys, hit, n_perm, rng):
        """Hit vectors under n_perm within-stratum permutations of the reference."""
        sid = d.groupby(keys).ngroup().values
        ids, cls = d.actual.to_numpy(), d.actual_cls.to_numpy(dtype=object)
        claims = d.claimed_cls.to_numpy(dtype=object)
        out = np.empty((n_perm, len(d)), dtype=bool)
        for b in range(n_perm):
            perm = np.lexsort((rng.random(len(d)), sid))
            order = np.argsort(sid, kind="stable")
            src = np.empty(len(d), dtype=int)
            src[order] = perm
            out[b] = hit(claims, ids[src], cls[src])
        return out

    d = d.reset_index(drop=True)
    obs = float(d.correct.mean())
    robs = float(d.region_hit.mean())
    res = {}
    for name, keys in STRATA.items():
        eo = expected(d, keys, hit_obj)
        er = expected(d, keys, hit_reg)
        res[name] = dict(chance=float(eo.mean()), region_chance=float(er.mean()),
                         kappa_object=float((obs - eo.mean()) / (1 - eo.mean())),
                         kappa_region=float((robs - er.mean()) / (1 - er.mean())))
        d[f"e_obj_{name}"], d[f"e_reg_{name}"] = eo, er

    prim = "cell"
    po = permute(d, STRATA[prim], hit_obj, B, RNG)
    pr = permute(d, STRATA[prim], hit_reg, B, RNG)
    null, rnull = po.mean(axis=1), pr.mean(axis=1)
    lo, hi = np.percentile(null, [2.5, 97.5])
    rlo, rhi = np.percentile(rnull, [2.5, 97.5])
    ch, rch = res[prim]["chance"], res[prim]["region_chance"]
    kappa_obj, kappa_reg = res[prim]["kappa_object"], res[prim]["kappa_region"]
    P("IS THIS BETTER THAN CHANCE?  (null: claims permuted within participant x model x format)")
    P(f"  object, observed {obs:6.1%}   chance {ch:6.1%}  null 95% range [{lo:.1%}, {hi:.1%}]"
      f"  p = {float((null >= obs).mean()):.4f}   kappa {kappa_obj:.1%}")
    P(f"  region, observed {robs:6.1%}   chance {rch:6.1%}  null 95% range [{rlo:.1%}, {rhi:.1%}]"
      f"  p = {float((rnull >= robs).mean()):.4f}   kappa {kappa_reg:.1%}")
    for name in ("participant", "call"):
        r = res[name]
        P(f"  sensitivity, within {name:11s}: object chance {r['chance']:.1%} kappa "
          f"{r['kappa_object']:.1%}; region chance {r['region_chance']:.1%} kappa {r['kappa_region']:.1%}")
    P(f"  gaze on the back wall in {np.isin(d.actual, list(BACK_WALL)).mean():.1%} of windows;"
      f" on the doctor's door (in neither region) in {(d.actual == 12).mean():.1%}")
    P("")

    # ---- participant bootstrap: 29 independent participants, correlated assertions within them
    pid_idx = [np.where(d.pid.values == p)[0] for p in d.pid.unique()]
    boot = collections.defaultdict(list)
    eo, er = d[f"e_obj_{prim}"].values, d[f"e_reg_{prim}"].values
    co, cr = d.correct.values.astype(float), d.region_hit.values.astype(float)
    for _ in range(2000):
        ix = np.concatenate([pid_idx[i] for i in RNG.integers(0, len(pid_idx), len(pid_idx))])
        a_o, a_r, c_o, c_r = co[ix].mean(), cr[ix].mean(), eo[ix].mean(), er[ix].mean()
        boot["object"].append(a_o)
        boot["region"].append(a_r)
        boot["kappa_object"].append((a_o - c_o) / (1 - c_o))
        boot["kappa_region"].append((a_r - c_r) / (1 - c_r))
    ci = {k: [float(x) for x in np.percentile(v, [2.5, 97.5])] for k, v in boot.items()}
    P("PARTICIPANT-BOOTSTRAP 95% CONFIDENCE INTERVALS (2,000 resamples of the 29 participants)")
    for k, v in ci.items():
        P(f"  {k:14s} [{v[0]:.1%}, {v[1]:.1%}]")
    P("")

    # ---- each target against its own null. A pooled chance line is the null of the pooled claim
    # mix, not of any one target, so each claimed class is compared with the chance of that class.
    tgt = {}
    P("EACH TARGET AGAINST ITS OWN CHANCE")
    for c in ("npc", "alarm", "sign"):
        m = d.claimed_cls.values == c
        nm = po[:, m].mean(axis=1)
        o = float(d.correct.values[m].mean())
        e = float(eo[m].mean())
        tl, th = np.percentile(nm, [2.5, 97.5])
        tgt[c] = dict(n=int(m.sum()), observed=o, chance=e, null_range=[float(tl), float(th)],
                      p=float((nm >= o).mean()), kappa=float((o - e) / (1 - e)),
                      region=float(cr[m].mean()), region_chance=float(er[m].mean()))
        P(f"  {c:6s} n={m.sum():5d}  observed {o:6.1%}  chance {e:6.1%}  null [{tl:.1%}, {th:.1%}]"
          f"  p = {tgt[c]['p']:.4f}  kappa {tgt[c]['kappa']:.1%}")
    wall = d.claimed_cls.isin(["alarm", "sign"]).values
    wo, we = float(d.correct.values[wall].mean()), float(eo[wall].mean())
    wn = po[:, wall].mean(axis=1)
    P(f"  wall cues together n={wall.sum()}  observed {wo:.1%}  chance {we:.1%}  kappa "
      f"{(wo - we) / (1 - we):.1%}  p = {float((wn >= wo).mean()):.4f}")
    npc_correct = int(d.correct.values[~wall].sum())
    P(f"  correct matches from character claims: {npc_correct} of {int(d.correct.sum())}")
    P("")

    # ---- each model against its own null, with how often it reports attention at all
    calls = pd.read_csv(OUT / "A1_calls.csv")
    cover = calls.assign(any=calls.n_timed > 0).groupby("model")["any"].mean()
    per_model = {}
    P("EACH MODEL AGAINST ITS OWN CHANCE (object and region), with coverage")
    P(f"  {'model':34s}{'n':>5}{'cover':>7}{'obj':>7}{'chance':>8}{'kappa':>7}"
      f"{'reg':>7}{'chance':>8}{'kappa':>7}{'npc':>6}")
    for mdl, g in d.groupby("model"):
        ix = g.index.values
        o_, e_ = co[ix].mean(), eo[ix].mean()
        r_, f_ = cr[ix].mean(), er[ix].mean()
        ko = (o_ - e_) / (1 - e_) if e_ < 1 else 0.0
        kr = (r_ - f_) / (1 - f_) if f_ < 1 else 0.0
        per_model[mdl] = dict(n=len(g), coverage=float(cover.get(mdl, np.nan)), object=float(o_),
                              object_chance=float(e_), kappa_object=float(ko), region=float(r_),
                              region_chance=float(f_), kappa_region=float(kr),
                              npc_claim_share=float((g.claimed_cls == "npc").mean()))
        P(f"  {mdl[:33]:34s}{len(g):5d}{cover.get(mdl, np.nan):7.1%}{o_:7.1%}{e_:8.1%}{ko:7.1%}"
          f"{r_:7.1%}{f_:8.1%}{kr:7.1%}{(g.claimed_cls == 'npc').mean():6.1%}")
    P("")

    # ---- the scene, described rather than treated as a measured threshold
    g = json.load(open(OUT / "scene_geometry.json", encoding="utf-8"))
    P("THE SCENE, AS A DESCRIPTION AND NOT A THRESHOLD")
    allv = d.sep_deg.round(2)
    seps = sorted(allv.unique())
    top3 = allv.value_counts().head(3)
    P(f"  the angular errors take {len(seps)} distinct values, because both the claimed and the")
    P("  gazed object come from a fixed set of nine objects viewed from one frozen viewpoint.")
    vals = ", ".join(f"{v:.1f}" for v in top3.index)
    P(f"  the three commonest values ({vals} deg) are {top3.sum() / len(d):.0%} of all rows.")
    P(f"  smallest non-zero separation in the scene: {min(s for s in seps if s > 0.01):.1f} deg")
    P("  No object pair lies between 0 and that value, so this design cannot locate a threshold")
    P("  below it. The wall-mounted cues are 0.5 to 1.1 m apart; that is a property of this room.")
    P("")
    P("  These angles are measured from one frozen viewpoint and are NOT stable. Recomputed from")
    P("  each participant's own seated position, the alarm-to-display separation spans 15.9 to")
    P("  27.1 deg (5th-95th percentile) against 24.9 at the canonical viewpoint, and the two NPCs")
    P("  span 78 to 127 deg against 113.5. Participants also stand and walk, which changes the")
    P("  subtense further. Quote these numbers as a description of the room from a seated start,")
    P("  never as a property of the annotator.")
    P("")
    P("READING")
    P("  VLM attention annotation is usable at the level of an entity that stands alone in the")
    P("  scene, and unusable for telling apart cues mounted together on one wall. That is a")
    P("  statement about what the annotation can be asked for. It is not a measured angular")
    P("  threshold, and it should not be reported as one.")

    d.to_csv(OUT / "C2_confusion.csv", index=False)
    json.dump(dict(n=len(d), exact=obs, region=robs, null="cell (participant x model x format)",
                   chance=ch, chance_ci=[float(lo), float(hi)],
                   p=float((null >= obs).mean()),
                   region_chance=rch, region_chance_ci=[float(rlo), float(rhi)],
                   region_p=float((rnull >= robs).mean()),
                   kappa_object=float(kappa_obj), kappa_region=float(kappa_reg),
                   sensitivity={k: v for k, v in res.items() if k != prim},
                   boot_ci=ci, per_target=tgt,
                   wall=dict(n=int(wall.sum()), observed=wo, chance=we,
                             kappa=(wo - we) / (1 - we)),
                   npc_correct=npc_correct, n_correct=int(d.correct.sum()),
                   per_model=per_model,
                   backwall_share=float(np.isin(d.actual, list(BACK_WALL)).mean()),
                   doctor_door_share=float((d.actual == 12).mean()),
                   by_type={t: float(g2.correct.mean()) for t, g2 in d.groupby("type")},
                   n_distinct_separations=len(seps)),
              open(OUT / "C2_confusion.json", "w"), indent=1)
    txt = "\n".join(L)
    (OUT / "C2_confusion_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
