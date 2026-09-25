"""Original (v2) task description, rerun on the same day and backend as the neutral-prompt calls
(Amendment 2 of docs/NEUTRAL_PROMPT_PREREG.md).

The July v2 calls for GPT-5.2, Claude Opus 4.8 and Qwen3-VL 235B were served two months earlier and
partly by different backends, so date and backend were confounded with the prompt. This reruns v2
with the backend pinned to the one each model used for its v3 calls and fallbacks disabled.

Usage:
  py evactime/21_v2_sameday.py          dry run
  py evactime/21_v2_sameday.py --go     run, resumable
"""
import argparse, base64, importlib.util, json, pathlib, sys, time, threading
import concurrent.futures as cf
import urllib.error, urllib.request
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evactime"
OUT = EV / "outputs"
CACHE = OUT / "v2_sameday_raw.jsonl"
REPS = 3
# model -> (backend used for its v3 calls, $/call estimate)
CELLS = {"openai/gpt-5.2": ("OpenAI", 0.0514),
         "anthropic/claude-opus-4.8": ("Claude Platform on AWS", 0.1078),
         "qwen/qwen3-vl-235b-a22b-instruct": ("Alibaba", 0.0039)}
MAX_SPEND_USD = 20.0


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, EV / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


orr = load("orrun", "17_run_full_openrouter.py")    # v2 prompt builder, parser, key


def call(model, content, temp, backend, timeout=420):
    body = json.dumps({"model": model, "temperature": temp,
                       "provider": {"order": [backend], "allow_fallbacks": False},
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(orr.URL, data=body, headers={
        "Authorization": f"Bearer {orr.KEY}", "Content-Type": "application/json",
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
            except Exception:
                continue
            if r.get("parsed"):
                s.add((r["model"], r["pid"], r["rep"]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--pids", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    frames = pd.read_csv(OUT / "asset_frames.csv")
    ref = pd.read_csv(OUT / "reference_engine_all.csv")
    pids = a.pids or list(ref.loc[ref["include"], "pid"])
    have = done_set()
    jobs = [(m, p, r) for m in CELLS for p in pids for r in range(REPS) if (m, p, r) not in have]
    est = sum(CELLS[m][1] for m, _, _ in jobs)
    print(f"v2 same-day rerun: {len(CELLS)} models x {len(pids)} x {REPS}; {len(jobs)} to run, "
          f"estimated ${est:.2f} (guard ${MAX_SPEND_USD})")
    if not a.go:
        print("dry run, add --go")
        return
    if est > MAX_SPEND_USD:
        sys.exit("estimate exceeds guard")
    payload = {}
    for p in pids:
        sub = frames[frames.pid == p].sort_values("frame_index")
        payload[p] = ([{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                        base64.b64encode(pathlib.Path(f).read_bytes()).decode()}}
                       for f in sub.file], len(sub))
    spent, lock = [0.0], threading.Lock()

    def run(job):
        model, pid, rep = job
        imgs, nf = payload[pid]
        rec = {"model": model, "pid": pid, "rep": rep, "cond": "B", "prompt_version": "v2",
               "route": "openrouter", "backend_pinned": CELLS[model][0], "ts": time.time()}
        for attempt in range(3):
            try:
                d = call(model, [{"type": "text", "text": orr.build_prompt(nf)}] + imgs,
                         0.0 if rep == 0 else 0.4, CELLS[model][0])
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
                print(f"  [{k}/{len(jobs)}] {ok} {rec['model']:34} {rec['pid']:4} r{rec['rep']} "
                      f"{rec.get('provider')} ${spent[0]:.2f} {(rec.get('error') or '')[:70]}",
                      flush=True)
    print(f"\nDONE {k} calls, ${spent[0]:.2f} -> {CACHE}")


if __name__ == "__main__":
    main()
