"""Gemini 3 Flash through OpenRouter, under BOTH task descriptions (Amendment 1 of
docs/NEUTRAL_PROMPT_PREREG.md).

The native Gemini API refused every call, and OpenRouter encodes video at a different resolution
from the native v2 runs. Running v2 and v3 on the same route keeps the task description as the only
manipulated variable. Four cells: {v2, v3n} x {A video with audio, B frames}, 29 participants x
3 repeats, temperatures as in v2 (rep 0 at 0.0, reps 1 and 2 at 0.4).

Usage:
  py evactime/20_gemini_openrouter.py          dry run
  py evactime/20_gemini_openrouter.py --go     run, resumable
"""
import argparse, base64, importlib.util, json, pathlib, sys, time, threading
import concurrent.futures as cf
import urllib.error, urllib.request
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evactime"
OUT = EV / "outputs"
PROMPTS = EV / "prompts"
CACHE = OUT / "gemini_openrouter_raw.jsonl"
MODEL = "google/gemini-3-flash-preview"
REPS = 3
EST = {"A": 0.006, "B": 0.045}       # $/call from the test call and the v2 OpenRouter pilot
MAX_SPEND_USD = 15.0


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, EV / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


n19 = load("n19", "19_neutral_prompt.py")     # v3 prompt builders, with the frozen-hash check
orr = load("orrun", "17_run_full_openrouter.py")


def common_v2(unit, tkey):
    return ((PROMPTS / "common_task_v2.md").read_text(encoding="utf-8")
            .replace("{UNIT}", unit).replace("{TKEY}", tkey))


def prompt(version, cond, n=None, dur=None):
    if version == "v3n":
        return n19.prompt_A(dur) if cond == "A" else n19.prompt_B(n)
    if cond == "A":
        tf = (PROMPTS / "condition_A_native_v2.md").read_text(encoding="utf-8")
        return common_v2("SECONDS from the clip start", "t") + "\n" + \
            tf.replace("{COMMON}", "").replace("{DUR}", f"{dur:.0f}")
    tf = (PROMPTS / "condition_B_frames_v2.md").read_text(encoding="utf-8")
    return common_v2("a FRAME INDEX", "frame") + "\n" + \
        tf.replace("{COMMON}", "").replace("{N}", str(n)).replace("{NMAX}", str(n - 1))


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
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    win = pd.read_csv(OUT / "asset_windows.csv").set_index("pid")
    frames = pd.read_csv(OUT / "asset_frames.csv")
    ref = pd.read_csv(OUT / "reference_engine_all.csv").set_index("pid")
    pids = a.pids or [p for p in win.index if bool(ref.loc[p, "include"])]
    have = done_set()
    jobs = [(v, c, p, r) for v in ("v2", "v3n") for c in ("A", "B") for p in pids
            for r in range(REPS) if (v, c, p, r) not in have]
    est = sum(EST[c] for _, c, _, _ in jobs)
    print(f"Gemini 3 Flash via OpenRouter: 2 prompts x 2 conditions x {len(pids)} participants x {REPS}")
    print(f"  {len(jobs)} calls to run ({len(have)} cached), estimated ${est:.2f} (guard ${MAX_SPEND_USD})")
    n19.common("x", "y")
    if not a.go:
        print("dry run, add --go")
        return
    if est > MAX_SPEND_USD:
        sys.exit("estimate exceeds guard")

    cache = {}

    def media(pid, cond):
        if (pid, cond) not in cache:
            if cond == "A":
                b = base64.b64encode(pathlib.Path(str(win.loc[pid, "clip"])).read_bytes()).decode()
                cache[(pid, cond)] = ([{"type": "video_url",
                                        "video_url": {"url": "data:video/mp4;base64," + b}}], 0)
            else:
                sub = frames[frames.pid == pid].sort_values("frame_index")
                cache[(pid, cond)] = ([{"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(
                        pathlib.Path(f).read_bytes()).decode()}} for f in sub.file], len(sub))
        return cache[(pid, cond)]

    spent = [0.0]
    lock = threading.Lock()

    def run(job):
        version, cond, pid, rep = job
        w = win.loc[pid]
        dur = float(w.window_end_s - w.window_start_s)
        parts, n = media(pid, cond)
        text = prompt(version, cond, n=n, dur=dur)
        rec = {"model": "gemini-3-flash-preview", "route": "openrouter", "pid": pid, "cond": cond,
               "rep": rep, "prompt_version": version, "ts": time.time()}
        for attempt in range(3):
            try:
                d = orr.call(MODEL, [{"type": "text", "text": text}] + parts,
                             0.0 if rep == 0 else 0.4)
                msg = (d.get("choices") or [{}])[0].get("message", {})
                txt = msg.get("content")
                if isinstance(txt, list):
                    txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict))
                rec.update(raw=txt, parsed=orr.parse_json(txt), usage=d.get("usage"),
                           latency_s=d.get("_latency_s"), provider=d.get("provider"))
                rec.pop("error", None)
                if rec.get("parsed"):
                    break
            except urllib.error.HTTPError as e:
                rec["error"] = f"HTTP {e.code}: {e.read()[:200].decode(errors='ignore')}"
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(5 + 10 * attempt)
        with lock:
            spent[0] += ((rec.get("usage") or {}).get("cost") or 0.0)
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
                      f"r{rec['rep']} ${spent[0]:.2f} {(rec.get('error') or '')[:70]}", flush=True)
    print(f"\nDONE {k} calls, ${spent[0]:.2f} -> {CACHE}")


if __name__ == "__main__":
    main()
