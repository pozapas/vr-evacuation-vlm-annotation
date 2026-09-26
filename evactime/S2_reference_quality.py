"""S2 - quality of the engine reference for the 29 analyzed recordings (review, 25 Sep 2026).

The engine record is the reference standard, so its own quality is reported: how well the three
clocks (screen video, Unity, LSL) were aligned, how much the alignment drifted, how the audible
alarm in the video lines up with the engine's alarm marker, the rate of the gaze raycast, the share
of raycast samples that struck a tagged object, and the gaps in the raycast series.

Outputs: S2_reference_quality.json
Run: py evactime/S2_reference_quality.py
"""
import json, pathlib
import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
DS = ROOT / "vlm dataset"


def tagged_shares():
    """Per-recording share of raycast samples on a tagged object, as written by main()."""
    return json.load(open(OUT / "S2_reference_quality.json", encoding="utf-8"))["tagged_per_recording"]


def main():
    cohort = set(yaml.safe_load(open(ROOT / "evactime" / "config.yaml", encoding="utf-8"))
                 ["corpus"]["participants"])
    r = pd.read_csv(OUT / "reference_engine_all.csv")
    r = r[r.pid.isin(cohort)]
    w = pd.read_csv(OUT / "asset_windows.csv").set_index("pid")
    mk = pd.read_csv(DS / "participant_marker_times.csv", sep=";")
    mk["pid"] = "P" + mk.Participant.str.extract(r"P0*(\d+)")[0]
    mk = mk.set_index("pid")
    h = pd.read_csv(DS / "participant_hitobject_timeseries.csv", sep=";")
    h["pid"] = "P" + h.Participant.str.extract(r"P0*(\d+)")[0]

    tagged, gaps = [], []
    for pid, g in h[h.pid.isin(cohort)].groupby("pid"):
        # the 75 s analysis window on the Unity clock
        t0 = w.loc[pid, "window_start_s"] + mk.loc[pid, "WaitingRoom"]
        g = g[(g.UnityTime >= t0) & (g.UnityTime <= t0 + 75)].sort_values("UnityTime")
        tagged.append(float((g.HitObjectID > 0).mean()))
        gaps.append(float(np.diff(g.UnityTime.values).max()) if len(g) > 1 else 75.0)
    q = lambda s: dict(median=float(np.median(s)), min=float(np.min(s)), max=float(np.max(s)))
    res = dict(n=len(r),
               lsl_unity_offset_spread_s=q(r.offset_spread_s),
               drift_s_per_s=q(r.offset_drift.abs()),
               alarm_audio_minus_marker_s=q(r.alarm_resid_s),
               gaze_hz=q(r.gaze_hz),
               share_tagged_in_window=q(tagged),
               longest_gap_s=q(gaps),
               n_gap_over_1s=int(sum(x > 1 for x in gaps)),
               tagged_per_recording=tagged)
    (OUT / "S2_reference_quality.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
