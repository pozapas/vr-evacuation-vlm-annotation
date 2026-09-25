"""STEP 5 - Frozen, hashed model inputs for both experimental conditions.

Two conditions share ONE analysis window per participant so that results are directly comparable:

  Condition A (native)      : window re-encoded as an MP4 WITH audio -> video-capable models
  Condition B (frames-only) : the same window sampled at 1 fps as JPEGs -> every model

Window design
-------------
The window is anchored on the ENGINE alarm, but the pre-alarm padding is randomised per participant
(deterministically, from a fixed seed) over [PAD_MIN, PAD_MAX]. If the padding were constant the
alarm would sit at the same index in every item and a model could score well by guessing the
position rather than seeing the event. Window LENGTH is constant so token cost is uniform.

Everything is SHA-256 hashed and written to an immutable manifest. Once this runs, the bytes sent to
every model are fixed; any later change to framing invalidates the spend and must bump ASSET_VERSION.
"""
import hashlib, json, pathlib, subprocess, sys
import numpy as np, pandas as pd, cv2
import imageio_ffmpeg

ASSET_VERSION = "v1"
SEED = 20260723
PAD_MIN, PAD_MAX = 10.0, 25.0     # s of pre-alarm padding (randomised)
WINDOW_S = 75.0                   # s total, constant
FPS_SAMPLE = 1.0                  # frames per second for Condition B
FRAME_W, FRAME_H = 512, 288
JPEG_Q = 85

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
VID = ROOT / "compressed_videos" / "compressed_videos"
ASSETS = ROOT / "assets" / ASSET_VERSION
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def sha256(p, blocks=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(blocks)):
            h.update(b)
    return h.hexdigest()


def main():
    ref = pd.read_csv(OUT / "reference_engine_all.csv")
    rng = np.random.default_rng(SEED)
    ASSETS.mkdir(parents=True, exist_ok=True)

    # deterministic padding, drawn once in participant order
    ref = ref.sort_values(ref.pid.str.lstrip("P").astype(int).name if False else "pid")
    ref["n"] = ref.pid.str.lstrip("P").astype(int)
    ref = ref.sort_values("n")
    pads = rng.uniform(PAD_MIN, PAD_MAX, size=len(ref))

    part_rows, frame_rows = [], []
    for pad, (_, r) in zip(pads, ref.iterrows()):
        pid = r.pid
        src = VID / f"{pid}_Compressed.mp4"
        if not src.exists() or not np.isfinite(r.alarm):
            print(f"  {pid}: skip (no video or no alarm)"); continue

        t0 = max(0.0, float(r.alarm) - float(pad))
        t1 = min(float(r.duration_s), t0 + WINDOW_S)
        t0 = max(0.0, t1 - WINDOW_S)          # keep length constant near the tail

        pdir = ASSETS / pid
        (pdir / "frames").mkdir(parents=True, exist_ok=True)

        # ---------- Condition B: frames ----------
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        times = np.arange(t0, t1 - 1e-6, 1.0 / FPS_SAMPLE)
        n_ok = 0
        for k, t in enumerate(times):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.resize(fr, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
            fp = pdir / "frames" / f"f{k:03d}.jpg"
            cv2.imwrite(str(fp), fr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            frame_rows.append({
                "pid": pid, "frame_index": k, "file": str(fp),
                "video_time_s": round(float(t), 3),
                "unity_time_s": round(float(t) + float(r.offset_video_to_unity_s), 3),
                "sha256": sha256(fp)})
            n_ok += 1
        cap.release()

        # ---------- Condition A: clip with audio ----------
        clip = pdir / f"{pid}_window.mp4"
        cmd = [FFMPEG, "-y", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(src),
               "-t", f"{t1-t0:.3f}", "-vf", f"scale={FRAME_W}:{FRAME_H}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-c:a", "aac", "-b:a", "64k", str(clip)]
        rc = subprocess.run(cmd, capture_output=True).returncode

        part_rows.append({
            "pid": pid, "include": bool(r["include"]),
            "include_premovement": bool(r["include_premovement"]),
            "window_start_s": round(t0, 3), "window_end_s": round(t1, 3),
            "pad_before_alarm_s": round(float(r.alarm) - t0, 3),
            "n_frames": n_ok, "frame_hz": FPS_SAMPLE,
            "alarm_video_s": r.alarm, "movement_video_s": r.movement, "egress_video_s": r.egress,
            "alarm_frame": round((float(r.alarm) - t0) * FPS_SAMPLE, 2),
            "clip": str(clip) if rc == 0 else None,
            "clip_sha256": sha256(clip) if rc == 0 and clip.exists() else None,
            "clip_bytes": clip.stat().st_size if rc == 0 and clip.exists() else None})
        print(f"  {pid:4} window {t0:7.2f}-{t1:7.2f}  pad {float(r.alarm)-t0:5.2f}s  "
              f"frames {n_ok:3}  clip {'ok' if rc==0 else 'FAIL'}", flush=True)

    pdf = pd.DataFrame(part_rows)
    fdf = pd.DataFrame(frame_rows)
    pdf.to_csv(OUT / "asset_windows.csv", index=False)
    fdf.to_csv(OUT / "asset_frames.csv", index=False)

    manifest = {
        "asset_version": ASSET_VERSION, "seed": SEED,
        "window_s": WINDOW_S, "pad_range_s": [PAD_MIN, PAD_MAX],
        "frame_hz": FPS_SAMPLE, "frame_size": [FRAME_W, FRAME_H], "jpeg_quality": JPEG_Q,
        "source": "compressed_videos/compressed_videos/*_Compressed.mp4 (720p, 30 fps)",
        "alignment": "video t=0 == 'Scene changed to WaitingRoom' marker",
        "n_participants": int(len(pdf)),
        "n_included": int(pdf["include"].sum()),
        "n_frames_total": int(len(fdf)),
        "frames_sha256_of_hashes": hashlib.sha256(
            "".join(sorted(fdf.sha256)).encode()).hexdigest(),
    }
    (OUT / "asset_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print("\n=== assets ===")
    print(json.dumps(manifest, indent=1))
    print(f"\nalarm frame index: min {pdf.alarm_frame.min():.1f} "
          f"max {pdf.alarm_frame.max():.1f} (randomised, so position is not guessable)")
    print("wrote asset_windows.csv, asset_frames.csv, asset_manifest.json ->", ASSETS)


if __name__ == "__main__":
    main()
