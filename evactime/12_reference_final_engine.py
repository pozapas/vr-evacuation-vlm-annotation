"""STEPS 3-4 - Engine reference in VIDEO time, plus the frozen analysis cohort.

Alignment
---------
video t=0 is taken as the `Scene changed to WaitingRoom` marker, so for any engine event
    t_video = t_unity - t_unity(WaitingRoom)
This was validated independently against the acoustic alarm detector: the residual
(audio alarm - predicted alarm) is within 1 s for 29/31 participants, median -0.45 s. The residual
is NOT used to correct the offset (that would make the alarm evaluation circular); it is reported
as alignment uncertainty and used only to flag participants whose trim rule evidently differs.

QC flags
--------
  clock_unstable       LSL<->Unity offset is not constant (restarted/merged sessions)
  alignment_uncertain  |audio - predicted alarm| > 2 s, i.e. the trim rule does not hold
  no_movement          player position never changes between alarm and egress
  event_outside_video  an engine event falls outside the compressed recording
"""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"

SPREAD_MAX = 1.0      # s, LSL<->Unity offset stability
RESID_MAX = 2.0       # s, alignment validation tolerance


def main():
    e = pd.read_csv(OUT / "engine_raw.csv")
    audio = pd.read_csv(OUT / "reference_alarm.csv")[["pid", "alarm_audio"]]
    man = pd.read_csv(OUT / "corpus_manifest.csv")[["pid", "duration_s", "resolution"]]
    d = e.merge(audio, on="pid", how="left").merge(man, on="pid", how="left")

    off = d.waitingroom_unity
    d["offset_video_to_unity_s"] = off
    for src, dst in [("alarm_unity", "alarm"), ("movement_unity", "movement"),
                     ("egress_unity", "egress"), ("outro_unity", "outro")]:
        d[dst] = d[src] - off

    d["alarm_resid_s"] = d.alarm_audio - d.alarm
    d["pre_movement"] = d.movement - d.alarm
    d["travel"] = d.egress - d.movement
    d["total"] = d.egress - d.alarm

    flags = []
    for _, r in d.iterrows():
        f = []
        if not np.isfinite(r.offset_spread_s) or r.offset_spread_s > SPREAD_MAX:
            f.append("clock_unstable")
        if np.isfinite(r.alarm_resid_s) and abs(r.alarm_resid_s) > RESID_MAX:
            f.append("alignment_uncertain")
        if not np.isfinite(r.movement):
            f.append("no_movement")
        for ev in ("alarm", "movement", "egress"):
            v = r[ev]
            if np.isfinite(v) and np.isfinite(r.duration_s) and not (0 <= v <= r.duration_s):
                f.append(f"{ev}_outside_video")
        flags.append(";".join(f))
    d["flags"] = flags

    d["include"] = d["flags"].apply(lambda s: not any(
        k in s for k in ("clock_unstable", "alignment_uncertain", "outside_video")))
    d["include_premovement"] = d["include"] & d.movement.notna()

    keep = ["pid", "offset_video_to_unity_s", "alarm", "movement", "egress", "outro",
            "pre_movement", "travel", "total", "alarm_audio", "alarm_resid_s",
            "duration_s", "resolution", "offset_spread_s", "offset_drift", "gaze_hz",
            "n_markers", "csv_sep", "gazelog", "flags", "include", "include_premovement"]
    d["n"] = d.pid.str.lstrip("P").astype(int)
    d = d.sort_values("n")[keep]
    d.to_csv(OUT / "reference_engine_all.csv", index=False)

    pd.set_option("display.width", 220)
    print(d[["pid", "alarm", "movement", "egress", "pre_movement", "travel", "total",
             "alarm_resid_s", "include", "flags"]].to_string(index=False))

    inc = d[d["include"]]
    incp = d[d["include_premovement"]]
    print(f"\n=== frozen cohort ===")
    print(f"  total participants        : {len(d)}")
    print(f"  included (alarm/egress)   : {len(inc)}  -> {sorted(set(d[~d["include"]].pid), key=lambda x:int(x[1:]))} excluded")
    print(f"  included (pre-movement)   : {len(incp)}")
    print("\n=== engine reference distributions (included) ===")
    for c in ["alarm", "pre_movement", "travel", "total"]:
        s = (incp if c in ("pre_movement", "travel") else inc)[c].dropna()
        print(f"  {c:13} n={len(s):2}  mean {s.mean():6.2f}  sd {s.std():5.2f}  "
              f"median {s.median():6.2f}  range {s.min():.1f}-{s.max():.1f}")
    r = d.alarm_resid_s.dropna()
    print(f"\nalignment residual: median {r.median():+.3f} s, "
          f"IQR {r.quantile(.25):+.2f}..{r.quantile(.75):+.2f}, "
          f"|resid|<=1 s for {int((r.abs()<=1).sum())}/{len(r)}")
    print("\nwrote", OUT / "reference_engine_all.csv")


if __name__ == "__main__":
    main()
