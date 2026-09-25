"""Load the cached VLM responses, recovering the ones whose JSON was almost valid.

Why this exists. The run scripts stored `parsed: null` whenever `json.loads` raised, and every
consumer then silently skipped those records. That dropped 116 of 1218 calls, and the loss was not
spread evenly: 114 of them belonged to gemini-3.1-pro-preview, whose keep rate was 56.3% against
99%+ for every other model. A model with 44% of its output missing cannot be compared with the
others, and the loss appeared nowhere in the paper.

Two separate causes, and only one is a real failure:

  * 49 calls returned HTTP 429 RESOURCE_EXHAUSTED. Those are genuinely missing and stay missing.
  * 65 calls returned a complete, well-formed response with ONE extra closing brace appended.
    `json.loads` refuses the trailing byte; `JSONDecoder.raw_decode` stops at the end of the first
    valid object and returns it. 62 of those carry a narrative of 20 or more words and are as
    usable as any other record.

Recovering them is not a judgement call about content. The object is parsed exactly as the model
emitted it; only trailing garbage after a syntactically complete object is discarded.

`load_records()` returns every record with a usable narrative, and reports what it recovered and
what remains missing, so the shortfall is stated rather than silent.
"""
import json, pathlib, re
import collections

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
FILES = ("gemini_native_raw.jsonl", "full_openrouter_raw.jsonl")
MIN_WORDS = 20

_DEC = json.JSONDecoder()
_FENCE = re.compile(r"^```(?:json)?\s*")


def _parse(rec):
    """Return the response object, recovering from trailing garbage. None if unrecoverable."""
    p = rec.get("parsed")
    if isinstance(p, dict):
        return p
    raw = rec.get("raw")
    if not raw:
        return None
    try:
        obj, _ = _DEC.raw_decode(_FENCE.sub("", raw.strip()))
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None


def load_records(min_words=MIN_WORDS, report=False, files=FILES):
    """All calls with a usable narrative, plus an accounting of everything else."""
    kept, stats = [], collections.Counter()
    per_model = collections.defaultdict(collections.Counter)
    for f in files:
        path = OUT / f
        if not path.exists():
            continue
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            m = rec["model"]
            stats["calls_on_disk"] += 1
            per_model[m]["on_disk"] += 1
            if "error" in rec and rec.get("parsed") is None and not rec.get("raw"):
                stats["api_error"] += 1
                per_model[m]["api_error"] += 1
                continue
            obj = _parse(rec)
            if obj is None:
                stats["unparseable"] += 1
                per_model[m]["unparseable"] += 1
                continue
            nar = (obj.get("narrative") or "").strip()
            if len(nar.split()) < min_words:
                stats["narrative_too_short"] += 1
                per_model[m]["short"] += 1
                continue
            if not isinstance(rec.get("parsed"), dict):
                stats["recovered"] += 1
                per_model[m]["recovered"] += 1
            kept.append(dict(pid=rec["pid"], model=m, cond=rec["cond"], rep=rec.get("rep"),
                             narrative=nar, parsed=obj,
                             prompt_version=rec.get("prompt_version", "v2"),
                             route=rec.get("route", "native" if m.startswith("gemini") else "openrouter")))
            per_model[m]["kept"] += 1
    stats["kept"] = len(kept)
    if report:
        print(f"calls on disk {stats['calls_on_disk']}, usable {stats['kept']} "
              f"({stats['kept'] / max(stats['calls_on_disk'], 1):.1%})")
        print(f"  recovered by raw_decode : {stats['recovered']}")
        print(f"  lost to API 429 errors  : {stats['api_error']}")
        print(f"  unparseable             : {stats['unparseable']}")
        print(f"  narrative under {min_words} words: {stats['narrative_too_short']}")
        print(f"  {'model':34s}{'on disk':>9}{'kept':>7}{'rate':>8}{'recovered':>11}{'429':>6}")
        for m in sorted(per_model, key=lambda k: per_model[k]["kept"] / per_model[k]["on_disk"]):
            c = per_model[m]
            print(f"  {m[:33]:34s}{c['on_disk']:9d}{c['kept']:7d}"
                  f"{c['kept'] / c['on_disk']:8.1%}{c['recovered']:11d}{c['api_error']:6d}")
    return kept, stats, per_model


if __name__ == "__main__":
    load_records(report=True)
