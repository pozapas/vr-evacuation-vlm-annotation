"""STEP 9b - Unified scoring of the full matrix against engine ground truth.

Sources
  full_openrouter_raw.jsonl   non-Google models, condition B (frames)
  gemini_native_raw.jsonl     Google models, conditions A (video+audio) and B (frames)

Time conversion
  condition A : value is SECONDS from clip start  -> t_video = window_start + v
  condition B : value is a FRAME INDEX at 1 fps   -> t_video = window_start + v
(identical arithmetic; the units coincide by construction because the frame rate is 1 fps)
"""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
EV = {"alarm_onset": "alarm", "movement_start": "movement", "exit_reached": "egress"}
EVENTS = ["alarm", "movement", "egress"]
GROSS = 15.0
RNG = np.random.default_rng(20260723)


import re

_EVRE = {k: re.compile(r'"%s"\s*:\s*\{[^{}]*?"(?:t|frame|time|seconds)"\s*:\s*(-?\d+(?:\.\d+)?)' % k,
                       re.S) for k in EV}


def salvage(raw):
    """Recover the three events from TRUNCATED JSON.

    Gemini 3.1 Pro repeatedly hit its output cap partway through the rich schema, leaving the
    object unclosed. The three required events are emitted first and are intact, so they are
    regex-recovered rather than re-purchased. Every salvaged call is flagged in the output so the
    paper can report how many were affected.
    """
    got = {}
    if not raw:
        return got
    for k, rx in _EVRE.items():
        m = rx.search(raw)
        if m:
            got[EV[k]] = float(m.group(1))
    return got


def extract(p):
    """Pull the three required events out of either JSON shape."""
    got = {}
    if not isinstance(p, dict):
        return got
    ev = p.get("events")
    def val(d):
        return next((d[c] for c in ("frame", "t", "time", "seconds")
                     if isinstance(d.get(c), (int, float))), None)
    if isinstance(ev, dict):
        for k, v in ev.items():
            if k in EV:
                x = val(v) if isinstance(v, dict) else (v if isinstance(v, (int, float)) else None)
                if x is not None:
                    got.setdefault(EV[k], float(x))
    elif isinstance(ev, list):
        for e in ev:
            if isinstance(e, dict) and e.get("type") in EV:
                x = val(e)
                if x is not None:
                    got.setdefault(EV[e["type"]], float(x))
    return got


def load():
    win = pd.read_csv(OUT / "asset_windows.csv").set_index("pid")
    ref = pd.read_csv(OUT / "reference_engine_all.csv").set_index("pid")
    rows = []
    for f, src in [(OUT / "full_openrouter_raw.jsonl", "openrouter"),
                   (OUT / "gemini_native_raw.jsonl", "gemini_native")]:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("prompt_version") != "v2":
                continue
            pid = r["pid"]
            if pid not in ref.index or not bool(ref.loc[pid, "include"]):
                continue
            u = r.get("usage") or {}
            rec = {"src": src, "model": r["model"], "pid": pid, "rep": r.get("rep", 0),
                   "cond": r.get("cond", "B"),
                   "in_tok": r.get("in_tok") or u.get("prompt_tokens"),
                   "out_tok": r.get("out_tok") or u.get("completion_tokens"),
                   "cost": u.get("cost"), "latency_s": r.get("latency_s"),
                   "ok": bool(r.get("parsed")), "error": r.get("error")}
            p = r.get("parsed")
            got = extract(p)
            rec["salvaged"] = False
            if len(got) < 3:                       # truncated / malformed output
                sal = salvage(r.get("raw"))
                if len(sal) > len(got):
                    got = sal
                    rec["salvaged"] = True
                    rec["ok"] = len(got) == 3
            if isinstance(p, dict):
                rec["n_optional"] = len(p.get("optional_events") or [])
                rec["n_phases"] = len(p.get("phases") or [])
                rec["has_narrative"] = bool(p.get("narrative"))
                rec["n_timeline"] = len(p.get("timeline") or [])
            w0 = win.loc[pid, "window_start_s"]
            for k in EVENTS:
                v = got.get(k)
                t = ref.loc[pid, k]
                rec[f"{k}_pred"] = (w0 + v) if v is not None else np.nan
                rec[f"{k}_err"] = (w0 + v - t) if (v is not None and np.isfinite(t)) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows), ref, win


def agg(d, label_cols):
    out = []
    for keys, g in d.groupby(label_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(label_cols, keys))
        row["n_calls"] = len(g)
        row["parse_ok"] = g.ok.mean()
        for k in EVENTS:
            e = g[f"{k}_err"].dropna()
            fail = e.abs() > GROSS
            ek = e[~fail]
            row[f"{k}_n"] = len(e)
            row[f"{k}_gross"] = fail.mean() if len(e) else np.nan
            row[f"{k}_bias"] = ek.mean() if len(ek) else np.nan
            row[f"{k}_sd"] = ek.std(ddof=1) if len(ek) > 1 else np.nan
            row[f"{k}_mae"] = ek.abs().mean() if len(ek) else np.nan
            row[f"{k}_w1"] = (e.abs() <= 1).mean() if len(e) else np.nan
            row[f"{k}_w2"] = (e.abs() <= 2).mean() if len(e) else np.nan
        row["cost_per_call"] = g.cost.mean()
        row["in_tok"] = g.in_tok.mean()
        row["latency_s"] = g.latency_s.median()
        out.append(row)
    return pd.DataFrame(out)


def intervals(d, ref):
    """Derived intervals per call, and their error vs engine truth."""
    rows = []
    for _, r in d.iterrows():
        t = ref.loc[r.pid]
        pm_p = r.movement_pred - r.alarm_pred
        tv_p = r.egress_pred - r.movement_pred
        to_p = r.egress_pred - r.alarm_pred
        rows.append({**{c: r[c] for c in ("src", "model", "pid", "rep", "cond")},
                     "pre_movement_err": pm_p - (t.movement - t.alarm),
                     "travel_err": tv_p - (t.egress - t.movement),
                     "total_err": to_p - (t.egress - t.alarm)})
    return pd.DataFrame(rows)


def main():
    d, ref, win = load()
    if d.empty:
        print("no results yet"); return
    d.to_csv(OUT / "full_scored_calls.csv", index=False)
    pd.set_option("display.width", 250)

    print(f"=== coverage ===\n  {len(d)} calls | "
          f"{d.model.nunique()} models | {d.pid.nunique()} participants | "
          f"parse ok {d.ok.mean()*100:.1f}%")
    print(d.groupby(["src", "cond", "model"]).size().to_string())

    a = agg(d, ["cond", "model"]).sort_values(["cond", "alarm_mae"])
    a.to_csv(OUT / "full_by_model.csv", index=False)
    print("\n=== per-event accuracy by model and condition (s) ===")
    cols = ["cond", "model", "n_calls", "parse_ok"] + \
           [f"{k}_{s}" for k in EVENTS for s in ("bias", "sd", "w1", "gross")]
    print(a[cols].round(3).to_string(index=False))

    iv = intervals(d, ref)
    iv.to_csv(OUT / "full_scored_intervals.csv", index=False)
    print("\n=== derived intervals: bias / MAE (gross-filtered) ===")
    rows = []
    for keys, g in iv.groupby(["cond", "model"]):
        row = {"cond": keys[0], "model": keys[1]}
        for c in ["pre_movement", "travel", "total"]:
            e = g[f"{c}_err"].dropna(); ek = e[e.abs() <= GROSS]
            row[f"{c}_bias"] = ek.mean() if len(ek) else np.nan
            row[f"{c}_mae"] = ek.abs().mean() if len(ek) else np.nan
            row[f"{c}_gross"] = (e.abs() > GROSS).mean() if len(e) else np.nan
        rows.append(row)
    ivs = pd.DataFrame(rows).sort_values(["cond", "total_mae"])
    ivs.to_csv(OUT / "full_intervals_by_model.csv", index=False)
    print(ivs.round(2).to_string(index=False))

    # ---- condition A vs B, Google models only (the audio / native-video ablation)
    g = d[d.model.str.startswith("gemini")]
    if g.cond.nunique() > 1:
        print("\n=== CONDITION A (video+audio) vs B (frames, no audio) - Google models ===")
        ab = agg(g, ["model", "cond"]).sort_values(["model", "cond"])
        print(ab[["model", "cond", "n_calls"] +
                 [f"{k}_{s}" for k in EVENTS for s in ("mae", "w1")] +
                 ["in_tok", "latency_s"]].round(3).to_string(index=False))

    # ---- test-retest (reproducibility across repeats)
    print("\n=== test-retest SD across repeats (s) ===")
    tr = d.groupby(["cond", "model", "pid"])[[f"{k}_pred" for k in EVENTS]].std(ddof=1)
    print(tr.groupby(["cond", "model"]).mean().round(2).to_string())

    # ---- spend
    tot = d.cost.dropna().sum()
    print(f"\n=== spend (OpenRouter-billed only) ===  ${tot:.2f}")
    print(d.groupby("model").cost.agg(["mean", "sum"]).round(4).to_string())
    print("\nwrote full_scored_calls.csv, full_by_model.csv, full_intervals_by_model.csv")


if __name__ == "__main__":
    main()
