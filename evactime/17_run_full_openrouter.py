"""STEP 9a - Full matrix, non-Google models, Condition B (frames), rich schema.

Google models are NOT run here: OpenRouter's image tokenisation for Gemini is ~2x the native API's
and gives no `media_resolution` control, so Google is run through 16_gemini_native.py instead.

Rich schema only - the pilot showed it does not cost timing accuracy and it carries the Q1 payload.
"""
import argparse, base64, json, pathlib, sys, time, concurrent.futures as cf
import urllib.request, urllib.error
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
PROMPTS = ROOT / "evactime" / "prompts"
CACHE = OUT / "full_openrouter_raw.jsonl"
URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "v2"

# (slug, measured $/call from the pilot rich variant)
MODELS = [
    ("qwen/qwen3-vl-235b-a22b-instruct", 0.0043),
    ("meta-llama/llama-4-maverick",      0.0087),
    ("anthropic/claude-sonnet-5",        0.0570),
    ("openai/gpt-5.2",                   0.0600),
    ("anthropic/claude-opus-4.8",        0.1100),
]
MAX_SPEND_USD = 40.0

env = {}
for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1); env[k.strip()] = v.strip()
KEY = env["OPENROUTER_API_KEY"]


def build_prompt(n_frames):
    common = (PROMPTS / "common_task_v2.md").read_text(encoding="utf-8")
    common = common.replace("{UNIT}", "a FRAME INDEX").replace("{TKEY}", "frame")
    tf = (PROMPTS / "condition_B_frames_v2.md").read_text(encoding="utf-8")
    tf = tf.replace("{COMMON}", "").replace("{N}", str(n_frames)).replace("{NMAX}", str(n_frames - 1))
    return common + "\n" + tf


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


def call(model, content, temp, timeout=420):
    body = json.dumps({"model": model, "temperature": temp,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/evactime", "X-Title": "EvacTime"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    d["_latency_s"] = time.time() - t0
    return d


def done_set():
    s = set()
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if r.get("parsed"):
                    s.add((r["model"], r["pid"], r["rep"]))
            except Exception:
                pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()

    frames = pd.read_csv(OUT / "asset_frames.csv")
    ref = pd.read_csv(OUT / "reference_engine_all.csv")
    pids = list(ref.loc[ref["include"], "pid"])

    have = done_set()
    jobs = [(m, p, r) for m, _ in MODELS for p in pids for r in range(a.reps)
            if (m, p, r) not in have]
    est = sum(c for _, c in MODELS) * len(pids) * a.reps
    print(f"full matrix (OpenRouter, condition B, rich): {len(MODELS)} models x {len(pids)} "
          f"participants x {a.reps} reps = {len(MODELS)*len(pids)*a.reps} calls")
    print(f"  {len(jobs)} still to do ({len(have)} cached)")
    print(f"  ESTIMATED ${est:.2f}  (guard ${MAX_SPEND_USD})")
    if not a.go:
        print("\ndry run - add --go"); return
    if est > MAX_SPEND_USD:
        sys.exit("estimate exceeds guard")

    payload = {}
    for p in pids:
        sub = frames[frames.pid == p].sort_values("frame_index")
        payload[p] = ([{"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(
                pathlib.Path(f).read_bytes()).decode()}} for f in sub.file], len(sub))
    print(f"  {len(payload)} participants staged\n", flush=True)

    spent = [0.0]
    lock = __import__("threading").Lock()

    def run(job):
        model, pid, rep = job
        imgs, nf = payload[pid]
        content = [{"type": "text", "text": build_prompt(nf)}] + imgs
        rec = {"model": model, "pid": pid, "rep": rep, "cond": "B",
               "prompt_version": PROMPT_VERSION, "ts": time.time()}
        for attempt in range(3):
            try:
                d = call(model, content, 0.0 if rep == 0 else 0.4)
                msg = (d.get("choices") or [{}])[0].get("message", {})
                txt = msg.get("content")
                if isinstance(txt, list):
                    txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict))
                rec.update({"raw": txt, "parsed": parse_json(txt), "usage": d.get("usage"),
                            "latency_s": d.get("_latency_s"), "provider": d.get("provider")})
                rec.pop("error", None)
                break
            except urllib.error.HTTPError as e:
                rec["error"] = f"HTTP {e.code}: {e.read()[:200].decode(errors='ignore')}"
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except Exception as e:
                rec["error"] = str(e)[:200]
            time.sleep(4 + 6 * attempt)
        with lock:
            spent[0] += ((rec.get("usage") or {}).get("cost") or 0.0)
        return rec

    n = 0
    with cf.ThreadPoolExecutor(max_workers=5) as ex, CACHE.open("a", encoding="utf-8") as fh:
        for rec in ex.map(run, jobs):
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            n += 1
            ok = "OK " if rec.get("parsed") else ("ERR" if rec.get("error") else "BAD")
            if n % 10 == 0 or ok != "OK ":
                print(f"  [{n}/{len(jobs)}] {ok} {rec['model']:34} {rec['pid']:4} r{rec['rep']} "
                      f"${spent[0]:.2f} {(rec.get('error') or '')[:70]}", flush=True)
    print(f"\nDONE {n} calls, ${spent[0]:.2f} spent -> {CACHE}")


if __name__ == "__main__":
    main()
