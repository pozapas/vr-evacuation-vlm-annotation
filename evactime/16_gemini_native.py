"""Google models via the NATIVE Gemini API (not OpenRouter).

OpenRouter billed 83k prompt tokens for the same 75 images that cost Qwen 11.7k and Anthropic 17k,
and gave no control over image tokenisation. The native API exposes `media_resolution`, so Google
models are run here instead, on the identical frozen assets.

Supports both experimental conditions:
  A  native video clip WITH audio      (only Google/video-capable models can do this)
  B  the same window as 75 JPEG frames (identical bytes to the OpenRouter runs)

Usage:
  python 16_gemini_native.py --cond A --models gemini-3-flash-preview --pids P5 P20 P4
  python 16_gemini_native.py --cond B --reps 3 --go
"""
import argparse, json, pathlib, time, concurrent.futures as cf
import pandas as pd
from google import genai
from google.genai import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
PROMPTS = ROOT / "evactime" / "prompts"
CACHE = OUT / "gemini_native_raw.jsonl"
PROMPT_VERSION = "v2"

env = {}
for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1); env[k.strip()] = v.strip()
client = genai.Client(api_key=env["GEMINI_API_KEY"])

MEDIA_RES = {"low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
             "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
             "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH}


def build_prompt(cond, n_frames, dur):
    common = (PROMPTS / "common_task_v2.md").read_text(encoding="utf-8")
    if cond in ("A", "A0"):
        common = common.replace("{UNIT}", "SECONDS from the clip start").replace("{TKEY}", "t")
        tf = (PROMPTS / "condition_A_native_v2.md").read_text(encoding="utf-8")
        tf = tf.replace("{COMMON}", "").replace("{DUR}", f"{dur:.0f}")
        if cond == "A0":
            tf += ("\nThis clip has NO AUDIO TRACK. Judge the alarm from the visual change "
                   "alone (the room flashes red).\n")
    else:
        common = common.replace("{UNIT}", "a FRAME INDEX").replace("{TKEY}", "frame")
        tf = (PROMPTS / "condition_B_frames_v2.md").read_text(encoding="utf-8")
        tf = tf.replace("{COMMON}", "").replace("{N}", str(n_frames)).replace("{NMAX}", str(n_frames - 1))
    return common + "\n" + tf


def done_set():
    s = set()
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                s.add((r["model"], r["pid"], r["cond"], r["rep"], r.get("media_res", "medium")))
            except Exception:
                pass
    return s


def parse_json(txt):
    if not txt:
        return None
    t = txt.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    i, j = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[i:j + 1]) if i >= 0 and j > i else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["A", "A0", "B"], required=True)  # A0 = video, muted
    ap.add_argument("--models", nargs="+", default=["gemini-3-flash-preview"])
    ap.add_argument("--pids", nargs="+", default=None)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--media-res", default="medium", choices=list(MEDIA_RES))
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()

    win = pd.read_csv(OUT / "asset_windows.csv").set_index("pid")
    frames = pd.read_csv(OUT / "asset_frames.csv")
    ref = pd.read_csv(OUT / "reference_engine_all.csv").set_index("pid")
    pids = a.pids or [p for p in win.index if bool(ref.loc[p, "include"])]

    jobs = [(m, p, r) for m in a.models for p in pids for r in range(a.reps)]
    have = done_set()
    jobs = [j for j in jobs if (j[0], j[1], a.cond, j[2], a.media_res) not in have]
    print(f"condition {a.cond}: {len(a.models)} models x {len(pids)} participants x {a.reps} reps "
          f"= {len(jobs)} calls (media_res={a.media_res})")
    if not a.go:
        print("dry run - add --go"); return

    def run(job):
        model, pid, rep = job
        rec = {"model": model, "pid": pid, "cond": a.cond, "rep": rep,
               "media_res": a.media_res, "prompt_version": PROMPT_VERSION, "ts": time.time()}
        try:
            w = win.loc[pid]
            dur = float(w.window_end_s - w.window_start_s)
            if a.cond in ("A", "A0"):
                clip = str(w["clip"]) if a.cond == "A" else                     pd.read_csv(OUT / "asset_clips_noaudio.csv").set_index("pid").loc[pid, "clip_noaudio"]
                fh = client.files.upload(file=str(clip))
                t0 = time.time()
                while fh.state.name == "PROCESSING" and time.time() - t0 < 600:
                    time.sleep(2); fh = client.files.get(name=fh.name)
                parts = [fh]
                nf = 0
            else:
                sub = frames[frames.pid == pid].sort_values("frame_index")
                parts = [types.Part.from_bytes(data=pathlib.Path(f).read_bytes(),
                                               mime_type="image/jpeg") for f in sub.file]
                nf = len(sub)
            prompt = build_prompt(a.cond, nf, dur)
            t0 = time.time()
            resp = client.models.generate_content(
                model=model, contents=parts + [prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0 if rep == 0 else 0.4,
                    media_resolution=MEDIA_RES[a.media_res],
                    response_mime_type="application/json"))
            rec["latency_s"] = time.time() - t0
            um = getattr(resp, "usage_metadata", None)
            if um is not None:
                rec["in_tok"] = getattr(um, "prompt_token_count", None)
                rec["out_tok"] = getattr(um, "candidates_token_count", None)
            rec["raw"] = resp.text
            rec["parsed"] = parse_json(resp.text)
            if a.cond in ("A", "A0"):
                try:
                    client.files.delete(name=fh.name)
                except Exception:
                    pass
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:250]}"
        return rec

    with cf.ThreadPoolExecutor(max_workers=4) as ex, CACHE.open("a", encoding="utf-8") as fh:
        for rec in ex.map(run, jobs):
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            ok = "OK " if rec.get("parsed") else ("ERR" if rec.get("error") else "BAD")
            print(f"  {ok} {rec['model']:28} {rec['pid']:4} {a.cond} rep{rec['rep']} "
                  f"tok={rec.get('in_tok')}/{rec.get('out_tok')} "
                  f"lat={rec.get('latency_s') or 0:.1f}s {(rec.get('error') or '')[:80]}", flush=True)
    print("\nwrote", CACHE)


if __name__ == "__main__":
    main()
