"""STEP 2 - Engine ground truth for all 31 participants, and the video-to-engine alignment.

Clocks
------
Three clocks are in play:
  LSL time    : the XDF wall-ish clock every stream is timestamped on
  Unity time  : the in-game clock; the GazeLog CSV `timestamp` column is ALREADY on this clock
  video time  : seconds from the first frame of the trimmed compressed MP4

The XDF Unity.TobiiGaze stream carries BOTH an LSL timestamp and Unity's own `timestamp` channel,
so LSL->Unity is recovered as their difference. The first sample alone is fragile, so we take the
MEDIAN over all samples and report the residual spread and the drift slope as QC.

video -> Unity is a per-participant constant because each MP4 was hand-trimmed. The hypothesis
under test (see 11b) is that every video was trimmed to the `Scene changed to WaitingRoom` marker,
which would make that marker's Unity time the offset. That is validated INDEPENDENTLY, against the
acoustic alarm onset, so no target event is consumed by the alignment.

Engine events (Unity time)
--------------------------
  alarm     : ALARM_TRIGGERED marker
  movement  : first sustained change in player planar position after the alarm
  egress    : the position-reset discontinuity at the scene transition (fallback: OutroScene marker)
"""
import json, pathlib, re, sys, warnings
import numpy as np, pandas as pd
import pyxdf

warnings.filterwarnings("ignore")
STAGE = pathlib.Path("D:/evac_engine_data")
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"

EPS = 1e-4          # player displacement per sample counted as motion
SUSTAIN = 0.2       # s it must persist
JUMP = 1.0          # position discontinuity marking a scene reset


# ----------------------------------------------------------------- xdf helpers
def stream_name(s):
    try:
        return s["info"]["name"][0]
    except Exception:
        return ""


def load_xdf(path):
    streams, _ = pyxdf.load_xdf(str(path), dejitter_timestamps=False,
                                synchronize_clocks=False, verbose=False)
    return streams


def gaze_offset(streams):
    """median(LSL - Unity) over the gaze stream, plus QC on spread and drift."""
    g = next((s for s in streams if stream_name(s) == "Unity.TobiiGaze"), None)
    if g is None:
        return None
    ts = np.asarray(g["time_stamps"], float)
    labels = []
    try:
        labels = [c["label"][0] for c in g["info"]["desc"][0]["channels"][0]["channel"]]
    except Exception:
        pass
    arr = np.asarray(g["time_series"], dtype=object)
    if "timestamp" not in labels:
        return None
    col = labels.index("timestamp")
    uni = pd.to_numeric(pd.Series(arr[:, col]), errors="coerce").values
    m = np.isfinite(uni) & np.isfinite(ts)
    if m.sum() < 100:
        return None
    d = ts[m] - uni[m]
    slope = np.polyfit(uni[m] - uni[m][0], d, 1)[0]
    return {"offset": float(np.median(d)), "spread_s": float(np.percentile(d, 97.5) - np.percentile(d, 2.5)),
            "drift_s_per_s": float(slope), "n": int(m.sum()),
            "unity_start": float(np.nanmin(uni[m])), "unity_end": float(np.nanmax(uni[m]))}


def markers(streams, off):
    s = next((x for x in streams if stream_name(x) == "Unity.Markers"), None)
    if s is None:
        return pd.DataFrame(columns=["unity_time_s", "marker"])
    ts = np.asarray(s["time_stamps"], float)
    vals = [(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v) for v in s["time_series"]]
    df = pd.DataFrame({"lsl_time_s": ts, "marker": [str(v).strip() for v in vals]})
    df["unity_time_s"] = df.lsl_time_s - off
    return df


# ----------------------------------------------------------------- gazelog
def pick_gazelog(pid, stage_df):
    """Largest GazeLog for the participant (P22 recorded several partial sessions)."""
    c = stage_df[(stage_df.pid == pid) & (stage_df.kind == "gazelog")]
    return c.sort_values("bytes", ascending=False).iloc[0] if len(c) else None


def read_gazelog(path):
    """GazeLog delimiter is locale-dependent: P1 and P2 exported with ';', the rest with ','."""
    head = open(path, encoding="utf-8", errors="ignore").readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    # documented export bug: leftControllerY appears twice; the second is really leftControllerZ
    if "leftControllerY.1" in df.columns:
        df = df.rename(columns={"leftControllerY.1": "leftControllerZ"})
    for c in ("timestamp", "playerX", "playerZ"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["timestamp", "playerX", "playerZ"]), sep


def engine_events(gz, alarm_u):
    t = gz["timestamp"].values
    px, pz = gz["playerX"].values, gz["playerZ"].values
    d = np.hypot(np.diff(px), np.diff(pz)); tm = t[1:]
    dt = float(np.median(np.diff(t))) or 0.02
    need = max(1, int(round(SUSTAIN / dt)))
    moving = d > EPS

    mv = None
    idx = np.where((tm > alarm_u) & moving)[0]
    for i in idx:
        if moving[i:i + need].mean() > 0.6:
            mv = float(tm[i]); break
    if mv is None and len(idx):
        mv = float(tm[idx[0]])

    eg = None
    j = np.where((tm > alarm_u) & (d > JUMP))[0]
    if len(j):
        eg = float(tm[j[0]])
    return mv, eg


def main():
    stage = pd.read_csv(OUT / "stage_manifest.csv")
    rows, marker_rows = [], []
    for i in range(1, 32):
        pid = f"P{i}"
        xs = stage[(stage.pid == pid) & (stage.kind == "xdf")].sort_values("bytes", ascending=False)
        rec = {"pid": pid}
        if not len(xs):
            rec["error"] = "no xdf"; rows.append(rec); continue
        try:
            streams = load_xdf(xs.iloc[0].staged_path)
            go = gaze_offset(streams)
            if go is None:
                rec["error"] = "no gaze stream"; rows.append(rec); continue
            mk = markers(streams, go["offset"])
        except Exception as e:
            rec["error"] = f"xdf: {str(e)[:80]}"; rows.append(rec); continue

        rec.update({"lsl_unity_offset": go["offset"], "offset_spread_s": go["spread_s"],
                    "offset_drift": go["drift_s_per_s"], "n_gaze": go["n"],
                    "n_markers": len(mk)})
        for _, m in mk.iterrows():
            marker_rows.append({"pid": pid, "unity_time_s": m.unity_time_s, "marker": m.marker})

        def find(pat, first=True):
            h = mk[mk.marker.str.contains(pat, case=False, na=False, regex=True)]
            if not len(h):
                return None
            return float(h.unity_time_s.iloc[0] if first else h.unity_time_s.iloc[-1])

        alarm_u = find(r"ALARM_TRIGGERED")
        wr_u = find(r"WaitingRoom")
        outro_u = find(r"OutroScene")
        rec.update({"alarm_unity": alarm_u, "waitingroom_unity": wr_u, "outro_unity": outro_u})

        gzr = pick_gazelog(pid, stage)
        if gzr is not None and alarm_u is not None:
            gz, sep = read_gazelog(gzr.staged_path)
            mv, eg = engine_events(gz, alarm_u)
            rec.update({"movement_unity": mv, "egress_unity": eg,
                        "gaze_t0": float(gz.timestamp.min()), "gaze_t1": float(gz.timestamp.max()),
                        "gaze_hz": float(len(gz) / max(gz.timestamp.max() - gz.timestamp.min(), 1e-9)),
                        "csv_sep": sep, "gazelog": pathlib.Path(gzr.staged_path).name})
        rows.append(rec)
        print(f"  {pid:4} markers={rec.get('n_markers')} alarm={alarm_u} "
              f"WR={wr_u} move={rec.get('movement_unity')} egress={rec.get('egress_unity')}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "engine_raw.csv", index=False)
    pd.DataFrame(marker_rows).to_csv(OUT / "engine_markers_all.csv", index=False)

    print("\n=== marker vocabulary across the cohort ===")
    mv = pd.DataFrame(marker_rows)
    vocab = mv.marker.value_counts()
    for k, v in vocab.items():
        print(f"  {v:4}  {k[:70]}")
    print("\nwrote engine_raw.csv, engine_markers_all.csv")


if __name__ == "__main__":
    main()
