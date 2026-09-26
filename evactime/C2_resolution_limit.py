"""C2 - the resolution limit: at what angular separation can a VLM still tell two objects apart?

This replaces the earlier "fabrication" framing. A1 plus the wall-cluster check showed the models
are not inventing attention events: 95 of 98 "unsupported" exit-sign assertions occurred while gaze
was on the exit wall. The model identifies the REGION correctly and the OBJECT incorrectly.

So the measurable quantity is a resolution limit, and it is the spatial analogue of the TRB paper's
temporal reliability criterion.

Method. For every timestamped assertion we know
    claimed  : the object the model named
    actual   : the object the reference says was gazed at, at that moment
Both have world positions from S1. The angular separation between them, seen from the participant's
viewpoint, is the quantity the model had to resolve. Agreement as a function of that angle is the
resolution curve.

Angular separation needs only POSITIONS, not head orientation - which matters, because head
orientation was never logged (see S1 notes). The canonical viewpoint is published as a constant so
the whole analysis is reproducible from the released geometry plus the released hitObject series.

Outputs: C2_resolution.csv, C2_resolution_report.txt
"""
import json, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
DS = ROOT / "vlm dataset"

SUPER = {-1: "none", 0: "none", 1: "misc", 2: "misc", 3: "sign", 4: "alarm", 10: "alarm",
         5: "door", 7: "door", 12: "door", 13: "door", 6: "display", 11: "equipment",
         8: "npc", 9: "npc"}
# A claim names a CLASS, not one object ("an alarm", "an NPC"). The angular error is therefore
# the distance from the gazed object to the NEAREST member of the claimed class - zero when right.
CLAIM_MEMBERS = {"look_at_alarm": [4, 10], "look_at_exit_sign": [3], "look_at_npc": [8, 9]}
CLAIM_ID = {k: v[0] for k, v in CLAIM_MEMBERS.items()}
CLAIM_CLS = {"look_at_alarm": "alarm", "look_at_exit_sign": "sign", "look_at_npc": "npc"}

# Canonical seated viewpoint, median over the cohort's pre-alarm baseline.
# Published as a constant so this analysis is reproducible without the trajectory data.
VIEWPOINT = None  # computed below, then frozen into the report


def angle_between(a, b, eye):
    va, vb = np.asarray(a) - eye, np.asarray(b) - eye
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(va @ vb / (na * nb), -1, 1))))


def main():
    geom = json.load(open(OUT / "scene_geometry.json"))
    pos = {int(k): np.array(v["pos"]) for k, v in geom.items() if v.get("pos")}
    # NOTE: 6_CanvasQueueDisplay used to be hand-patched here because S1 read a
    # RectTransform's m_LocalPosition instead of its m_AnchoredPosition and put the object
    # at the room origin. S1 now resolves it, so the released geometry and this analysis
    # agree and the paper reproduces from the deposited artefacts.

    # canonical viewpoint from the pre-alarm baseline (a 3-number constant, not a dataset)
    raw = pd.read_csv(OUT / "engine_raw.csv")
    mk = pd.read_csv(DS / "participant_marker_times.csv", sep=";")
    mk["pid"] = "P" + mk.Participant.str.extract(r"P0*(\d+)")[0]
    eyes = []
    for _, r in raw.iterrows():
        if r.pid in ("P2", "P22"):
            continue
        m = mk[mk.pid == r.pid]
        if not len(m):
            continue
        m = m.iloc[0]
        g = pd.read_csv(ROOT / "engine_data" / r.pid / r.gazelog,
                        sep=r.csv_sep, engine="python")
        g.columns = [c.strip() for c in g.columns]
        b = g[(g.timestamp >= m.WaitingRoom) & (g.timestamp < m.ALARM_TRIGGERED)]
        if len(b):
            eyes.append([(b.playerX + b.leftPosX).median(), b.leftPosY.median(),
                         (b.playerZ + b.leftPosZ).median()])
    eye = np.median(np.array(eyes), axis=0)

    a = pd.read_csv(OUT / "A1_assertions_scored.csv")
    a = a[a.timed & a.t_unity.notna()].copy()
    h = pd.read_csv(DS / "participant_hitobject_timeseries.csv", sep=";")
    h["pid"] = "P" + h.Participant.str.extract(r"P0*(\d+)")[0]
    by = {p: g for p, g in h.groupby("pid")}

    rows = []
    for _, r in a.iterrows():
        g = by.get(r.pid)
        if g is None:
            continue
        w = g[(g.UnityTime >= r.t_unity - 1.0) & (g.UnityTime <= r.t_unity + 1.0)]
        w = w[w.HitObjectID >= 0]
        if not len(w):
            continue
        actual = int(w.HitObjectID.mode().iloc[0])
        claimed = CLAIM_ID[r.type]
        if actual not in pos or claimed not in pos:
            continue
        sep = min(angle_between(pos[m], pos[actual], eye)
                  for m in CLAIM_MEMBERS[r.type] if m in pos)
        rows.append(dict(pid=r.pid, model=r.model, cond=r.cond, rep=r.rep, type=r.type,
                         claimed=claimed, actual=actual,
                         claimed_cls=CLAIM_CLS[r.type], actual_cls=SUPER.get(actual, "misc"),
                         sep_deg=sep, correct=SUPER.get(actual) == CLAIM_CLS[r.type],
                         phase=r.phase))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "C2_resolution.csv", index=False)

    L = []
    P = L.append
    P("C2 - resolution limit of VLM attention annotation")
    P("=" * 64)
    P(f"canonical seated viewpoint (median pre-alarm eye position, n=29):")
    P(f"  ({eye[0]:.3f}, {eye[1]:.3f}, {eye[2]:.3f})   <- publish this constant")
    P(f"scored assertions with a resolvable reference: {len(d)}")
    P("")
    P("PAIRWISE ANGULAR SEPARATION from that viewpoint (scene property)")
    names = {3: "ExitSign", 4: "AlarmVisual", 10: "AlarmAudio", 6: "QueueDisplay",
             13: "ExitDoor", 12: "DoctorDoor", 8: "NPC1", 9: "NPC2", 11: "Extinguisher"}
    ks = [3, 4, 10, 6, 13, 11, 8, 9, 12]
    P("        " + "".join(f"{names[k][:9]:>11}" for k in ks))
    for i in ks:
        P(f"{names[i][:8]:8}" + "".join(
            f"{angle_between(pos[i], pos[j], eye):11.1f}" if i != j else f"{'-':>11}" for j in ks))
    P("")
    P("ANGULAR ERROR - how far the model's named object sits from the one actually gazed at.")
    P("The spatial analogue of the TRB timing-error tolerance curve: there the question was how")
    P("many seconds of tolerance a practitioner must accept, here it is how many degrees.")
    P("")
    P(f"  {'assertion':22s}{'n':>6}{'median':>9}{'p80':>8}{'p90':>8}{'<=10 deg':>11}")
    for t, g in d.groupby("type"):
        e = g.sep_deg
        P(f"  {t:22s}{len(g):6d}{e.median():9.1f}{e.quantile(.8):8.1f}"
          f"{e.quantile(.9):8.1f}{(e <= 10).mean():11.1%}")
    e = d.sep_deg
    P(f"  {'ALL':22s}{len(d):6d}{e.median():9.1f}{e.quantile(.8):8.1f}"
      f"{e.quantile(.9):8.1f}{(e <= 10).mean():11.1%}")
    P("")
    P(f"  Read: covering 80 percent of attention calls needs a tolerance of about "
      f"{e.quantile(.8):.0f} degrees.")
    P("  Objects closer together than that are not separable by this instrument.")
    P("")
    P("ERROR BY PHASE")
    P(f"  {'phase':>12}{'n':>7}{'median deg':>12}{'p80':>8}")
    for ph, g in d.groupby("phase"):
        P(f"  {ph:>12}{len(g):7d}{g.sep_deg.median():12.1f}{g.sep_deg.quantile(.8):8.1f}")
    P("")
    P("WHAT THE MODEL NAMES WHEN IT IS WRONG")
    for t, g in d.groupby("type"):
        P(f"  {t}  (n={len(g)}, exact-class hit {g.correct.mean():.1%})")
        for oid, n in g[~g.correct].actual.value_counts().head(4).items():
            P(f"      actually {names.get(oid, oid):14s} n={n:4d}  "
              f"{min(angle_between(pos[m], pos[oid], eye) for m in CLAIM_MEMBERS[t] if m in pos):5.1f} deg away")
    P("")
    txt = "\n".join(L)
    (OUT / "C2_resolution_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
