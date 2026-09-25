"""Neutral-prompt ablation (analysis plan fixed in docs/NEUTRAL_PROMPT_PREREG.md before any call).

Reruns five cells with evactime/prompts/common_task_v3_neutral.md, which removes the v2 sentence
asserting that the participant leaves the room and makes `exit_reached` conditional. Everything
else is taken from the v2 runners unchanged, including the call functions, the frozen assets,
the temperatures and the media resolution, so the only manipulated variable is the task
description.

Cells: GPT-5.2, Claude Opus 4.8, Qwen3-VL 235B on frames (OpenRouter), Gemini 3 Flash on video
with audio and on frames (native API). 29 participants x 3 repeats each.

Usage:
  py evactime/19_neutral_prompt.py            dry run, prints the plan and the cost estimate
  py evactime/19_neutral_prompt.py --go       runs, resumable (cached calls are skipped)
"""
import argparse, base64, hashlib, importlib.util, json, pathlib, sys, time, threading
import concurrent.futures as cf
import urllib.error
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evactime"
OUT = EV / "outputs"
PROMPTS = EV / "prompts"
CACHE = OUT / "neutral_prompt_raw.jsonl"
CACHE_GEMINI = OUT / "neutral_prompt_raw_gemini.jsonl"   # separate file, so two runs can append safely
PROMPT_FILE = PROMPTS / "common_task_v3_neutral.md"
PROMPT_SHA = "93e8c6f9edb6b219aebfcac67799984663bbb1773bdb2d2b18ce605279fe5f17"
PROMPT_VERSION = "v3n"
REPS = 3
OR_MODELS = [("openai/gpt-5.2", 0.0514), ("anthropic/claude-opus-4.8", 0.1078),
             ("qwen/qwen3-vl-235b-a22b-instruct", 0.0039)]
GEMINI = [("gemini-3-flash-preview", "A"), ("gemini-3-flash-preview", "B")]
GEMINI_EST = 0.02          # per call, generous; the native API does not report cost
MAX_SPEND_USD = 25.0


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, EV / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def common(unit, tkey):
    raw = PROMPT_FILE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROMPT_SHA:
        sys.exit("neutral prompt changed after the analysis plan was fixed")
    return raw.decode("utf-8").replace("{UNIT}", unit).replace("{TKEY}", tkey)


def prompt_B(n):
    tf = (PROMPTS / "condition_B_frames_v2.md").read_text(encoding="utf-8")
    tf = tf.replace("{COMMON}", "").replace("{N}", str(n)).replace("{NMAX}", str(n - 1))
    return common("a FRAME INDEX", "frame") + "\n" + tf


def prompt_A(dur):
    tf = (PROMPTS / "condition_A_native_v2.md").read_text(encoding="utf-8")
    tf = tf.replace("{COMMON}", "").replace("{DUR}", f"{dur:.0f}")
    return common("SECONDS from the clip start", "t") + "\n" + tf


def done_set():
    s = set()
    for f in (CACHE, CACHE_GEMINI):
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("parsed"):
                s.add((r["model"], r["pid"], r["cond"], r["rep"]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--pids", nargs="+", default=None)
    ap.add_argument("--api", choices=["openrouter", "gemini", "all"], default="all")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    win = pd.read_csv(OUT / "asset_windows.csv").set_index("pid")
    frames = pd.read_csv(OUT / "asset_frames.csv")
    ref = pd.read_csv(OUT / "reference_engine_all.csv").set_index("pid")
    pids = a.pids or [p for p in win.index if bool(ref.loc[p, "include"])]
    cells = [(m, "B", "openrouter") for m, _ in OR_MODELS] + [(m, c, "gemini") for m, c in GEMINI]
    have = done_set()
    jobs = [(m, c, api, p, r) for m, c, api in cells for p in pids for r in range(REPS)
            if (m, p, c, r) not in have and a.api in ("all", api)]
    cost = dict(OR_MODELS)
    est = sum(cost.get(m, GEMINI_EST) for m, *_ in jobs)
    print(f"neutral-prompt ablation: {len(cells)} cells x {len(pids)} participants x {REPS} reps")
    print(f"  {len(jobs)} calls to run ({len(have)} cached), estimated ${est:.2f} (guard ${MAX_SPEND_USD})")
    common("x", "y")  # hash check before anything is sent
    if not a.go:
        print("dry run, add --go")
        return
    if est > MAX_SPEND_USD:
        sys.exit("estimate exceeds guard")

    orr = load("orrun", "17_run_full_openrouter.py")
    gem = load("gemrun", "16_gemini_native.py")
    types = gem.types
    b64 = {}

    def or_payload(pid):
        if pid not in b64:
            sub = frames[frames.pid == pid].sort_values("frame_index")
            b64[pid] = ([{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                          base64.b64encode(pathlib.Path(f).read_bytes()).decode()}}
                         for f in sub.file], len(sub))
        return b64[pid]

    spent = [0.0]
    lock = threading.Lock()

    def run(job):
        model, cond, api, pid, rep = job
        temp = 0.0 if rep == 0 else 0.4
        rec = {"model": model, "pid": pid, "cond": cond, "rep": rep, "api": api,
               "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA, "ts": time.time()}
        for attempt in range(6 if api == "gemini" else 3):
            try:
                if api == "openrouter":
                    imgs, nf = or_payload(pid)
                    d = orr.call(model, [{"type": "text", "text": prompt_B(nf)}] + imgs, temp)
                    msg = (d.get("choices") or [{}])[0].get("message", {})
                    txt = msg.get("content")
                    if isinstance(txt, list):
                        txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict))
                    rec.update(raw=txt, parsed=orr.parse_json(txt), usage=d.get("usage"),
                               latency_s=d.get("_latency_s"), provider=d.get("provider"))
                else:
                    w = win.loc[pid]
                    dur = float(w.window_end_s - w.window_start_s)
                    rec["media_res"] = "medium"
                    if cond == "A":
                        fh = gem.client.files.upload(file=str(w["clip"]))
                        t0 = time.time()
                        while fh.state.name == "PROCESSING" and time.time() - t0 < 600:
                            time.sleep(2)
                            fh = gem.client.files.get(name=fh.name)
                        parts, prompt = [fh], prompt_A(dur)
                    else:
                        sub = frames[frames.pid == pid].sort_values("frame_index")
                        parts = [types.Part.from_bytes(data=pathlib.Path(f).read_bytes(),
                                                       mime_type="image/jpeg") for f in sub.file]
                        prompt = prompt_B(len(sub))
                    t0 = time.time()
                    resp = gem.client.models.generate_content(
                        model=model, contents=parts + [prompt],
                        config=types.GenerateContentConfig(
                            temperature=temp, media_resolution=gem.MEDIA_RES["medium"],
                            response_mime_type="application/json"))
                    rec["latency_s"] = time.time() - t0
                    um = getattr(resp, "usage_metadata", None)
                    if um is not None:
                        rec["in_tok"] = getattr(um, "prompt_token_count", None)
                        rec["out_tok"] = getattr(um, "candidates_token_count", None)
                    rec.update(raw=resp.text, parsed=gem.parse_json(resp.text))
                    if cond == "A":
                        try:
                            gem.client.files.delete(name=fh.name)
                        except Exception:
                            pass
                rec.pop("error", None)
                if rec.get("parsed"):
                    break
            except urllib.error.HTTPError as e:
                rec["error"] = f"HTTP {e.code}: {e.read()[:200].decode(errors='ignore')}"
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            quota = "429" in (rec.get("error") or "") and api == "gemini"
            time.sleep(35 * (attempt + 1) if quota else 5 + 10 * attempt)
        with lock:
            spent[0] += ((rec.get("usage") or {}).get("cost") or 0.0)
        return rec

    n = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex, (CACHE_GEMINI if a.api == "gemini" else CACHE).open("a", encoding="utf-8") as fh:
        for rec in ex.map(run, jobs):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n += 1
            ok = "OK " if rec.get("parsed") else ("ERR" if rec.get("error") else "BAD")
            if n % 10 == 0 or ok != "OK ":
                print(f"  [{n}/{len(jobs)}] {ok} {rec['model']:34} {rec['cond']:2} {rec['pid']:4} "
                      f"r{rec['rep']} OpenRouter ${spent[0]:.2f} {(rec.get('error') or '')[:70]}",
                      flush=True)
    print(f"\nDONE {n} calls, OpenRouter ${spent[0]:.2f} -> {CACHE}")


if __name__ == "__main__":
    main()
