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


def chance_baseline(d, n_boot=B):
    """Accuracy if the claim were drawn independently of where gaze actually was.

    The claim and the reference are permuted against each other WITHIN participant, so the
    baseline preserves each participant's gaze distribution and each model's claim distribution
    and destroys only the association between them. That is the null the paper needs: "the model
    names the right class more often than its own output rate would give by chance".
    """
    # Pooled over assertions, the same estimand as the observed agreement it is compared with.
    # An earlier version averaged per-participant accuracies, which weights participants equally
    # while the observed value weights assertions equally.
    got = []
    groups = [g for _, g in d.groupby("pid")]
    for _ in range(n_boot):
        hits = 0
        for g in groups:
            shuffled = RNG.permutation(g.actual_cls.values)
            hits += int((g.claimed_cls.values == shuffled).sum())
        got.append(hits / len(d))
    return np.array(got)


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
    null = chance_baseline(d)
    obs = float(d.correct.mean())
    lo, hi = np.percentile(null, [2.5, 97.5])
    P("IS THIS BETTER THAN CHANCE?")
    P(f"  observed exact-class accuracy      {obs:6.1%}")
    P(f"  chance, claims permuted within participant  {null.mean():6.1%}  95% [{lo:.1%}, {hi:.1%}]")
    P(f"  p = {float((null >= obs).mean()):.4f}")

    # The region result needs its own null. About 83% of gaze windows are on the back wall, so a
    # model that always named a wall cue would score high on region without looking. Permuting
    # the reference object within participant and re-scoring the region gives that baseline.
    def region_of(claims, actual):
        return np.where(claims == "npc", np.isin(actual, list(NPC_IDS)),
                        np.isin(actual, list(BACK_WALL)))
    rnull = []
    for _ in range(B):
        hits = sum(int(region_of(g.claimed_cls.values, RNG.permutation(g.actual.values)).sum())
                   for _, g in d.groupby("pid"))
        rnull.append(hits / len(d))
    rnull = np.array(rnull)
    robs = float(d.region_hit.mean())
    rlo, rhi = np.percentile(rnull, [2.5, 97.5])
    kappa_obj = (obs - null.mean()) / (1 - null.mean())
    kappa_reg = (robs - rnull.mean()) / (1 - rnull.mean())
    P(f"  region, observed                   {robs:6.1%}")
    P(f"  region chance, same permutation    {rnull.mean():6.1%}  95% [{rlo:.1%}, {rhi:.1%}]"
      f"  p = {float((rnull >= robs).mean()):.4f}")
    P(f"  gaze on the back wall in {np.isin(d.actual, list(BACK_WALL)).mean():.1%} of windows")
    P(f"  chance-corrected agreement (observed - chance) / (1 - chance):"
      f" object {kappa_obj:.1%}, region {kappa_reg:.1%}")
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
    json.dump(dict(n=len(d), exact=obs, region=float(d.region_hit.mean()),
                   chance=float(null.mean()), chance_ci=[float(lo), float(hi)],
                   p=float((null >= obs).mean()),
                   region_chance=float(rnull.mean()), region_chance_ci=[float(rlo), float(rhi)],
                   region_p=float((rnull >= robs).mean()),
                   kappa_object=float(kappa_obj), kappa_region=float(kappa_reg),
                   backwall_share=float(np.isin(d.actual, list(BACK_WALL)).mean()),
                   by_type={t: float(g2.correct.mean()) for t, g2 in d.groupby("type")},
                   n_distinct_separations=len(seps)),
              open(OUT / "C2_confusion.json", "w"), indent=1)
    txt = "\n".join(L)
    (OUT / "C2_confusion_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
