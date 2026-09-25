"""Gemini 3 Flash on the NATIVE interface under both task descriptions, same day (Amendment 3 of
docs/NEUTRAL_PROMPT_PREREG.md).

Four cells: {v2, v3n} x {A video with audio, B frames}, 29 participants x 3 repeats, media
resolution medium, temperatures as in v2 (rep 0 at 0.0, reps 1 and 2 at 0.4). Prompts are built
by the same functions as the v2 native run and the v3 run, so only the description differs.

Usage:
  py evactime/22_gemini_native_both.py          dry run
  py evactime/22_gemini_native_both.py --go     run, resumable
"""
import argparse, importlib.util, json, pathlib, time, threading
import concurrent.futures as cf
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evactime"
OUT = EV / "outputs"
CACHE = OUT / "gemini_native_sameday_raw.jsonl"
MODEL = "gemini-3-flash-preview"
REPS = 3


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, EV / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gem = load("gemrun", "16_gemini_native.py")   # native client, v2 prompt builder, parser
n19 = load("n19", "19_neutral_prompt.py")     # v3 prompt builders with the frozen-hash check
types = gem.types


def prompt(version, cond, n, dur):
    if version == "v2":
        return gem.build_prompt(cond, n, dur)
    return n19.prompt_A(dur) if cond == "A" else n19.prompt_B(n)


def done_set():
    s = set()
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("parsed"):
                s.add((r["prompt_version"], r["cond"], r["pid"], r["rep"]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--pids", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    win = pd.read_csv(OUT / "asset_windows.csv").set_index("pid")
    frames = pd.read_csv(OUT / "asset_frames.csv")
    ref = pd.read_csv(OUT / "reference_engine_all.csv").set_index("pid")
    pids = a.pids or [p for p in win.index if bool(ref.loc[p, "include"])]
    have = done_set()
    jobs = [(v, c, p, r) for v in ("v2", "v3n") for c in ("A", "B") for p in pids
            for r in range(REPS) if (v, c, p, r) not in have]
    print(f"native Gemini 3 Flash, both descriptions: {len(jobs)} calls to run ({len(have)} cached)")
    n19.common("x", "y")
    if not a.go:
        print("dry run, add --go")
        return

    def run(job):
        version, cond, pid, rep = job
        w = win.loc[pid]
        dur = float(w.window_end_s - w.window_start_s)
        rec = {"model": MODEL, "route": "native", "pid": pid, "cond": cond, "rep": rep,
               "prompt_version": version, "media_res": "medium", "ts": time.time()}
        for attempt in range(5):
            try:
                if cond == "A":
                    fh = gem.client.files.upload(file=str(w["clip"]))
                    t0 = time.time()
                    while fh.state.name == "PROCESSING" and time.time() - t0 < 600:
                        time.sleep(2)
                        fh = gem.client.files.get(name=fh.name)
                    parts, n = [fh], 0
                else:
                    sub = frames[frames.pid == pid].sort_values("frame_index")
                    parts = [types.Part.from_bytes(data=pathlib.Path(f).read_bytes(),
                                                   mime_type="image/jpeg") for f in sub.file]
                    n = len(sub)
                t0 = time.time()
                resp = gem.client.models.generate_content(
                    model=MODEL, contents=parts + [prompt(version, cond, n, dur)],
                    config=types.GenerateContentConfig(
                        temperature=0.0 if rep == 0 else 0.4,
                        media_resolution=gem.MEDIA_RES["medium"],
                        response_mime_type="application/json"))
                rec["latency_s"] = time.time() - t0
                um = getattr(resp, "usage_metadata", None)
                if um is not None:
                    rec["in_tok"] = getattr(um, "prompt_token_count", None)
                    rec["out_tok"] = getattr(um, "candidates_token_count", None)
                rec.update(raw=resp.text, parsed=gem.parse_json(resp.text))
                rec.pop("error", None)
                if cond == "A":
                    try:
                        gem.client.files.delete(name=fh.name)
                    except Exception:
                        pass
                if rec.get("parsed"):
                    break
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(20 * (attempt + 1) if "429" in rec["error"] else 5)
        return rec

    k = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex, CACHE.open("a", encoding="utf-8") as fh:
        for rec in ex.map(run, jobs):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            k += 1
            ok = "OK " if rec.get("parsed") else ("ERR" if rec.get("error") else "BAD")
            if k % 10 == 0 or ok != "OK ":
                print(f"  [{k}/{len(jobs)}] {ok} {rec['prompt_version']:3} {rec['cond']} {rec['pid']:4} "
                      f"r{rec['rep']} tok={rec.get('in_tok')} {(rec.get('error') or '')[:70]}", flush=True)
    print(f"\nDONE {k} calls -> {CACHE}")


if __name__ == "__main__":
    main()
