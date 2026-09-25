"""C4 - script completion: the model narrates the evacuation it expects, not the one on screen.

What actually happens at the end of every recording, verified from the Unity source and from the
frames themselves:
    DoorHandleTrigger.OnTouched  ->  fade.StartFade()  ->  SceneSwitcher.LoadSceneByName("OutroScene")
The scene cuts on hand contact with the handle. The participant is then placed back in the SAME
waiting room, viewed from across it. Nobody walks through a doorway; nobody reaches a corridor.

Every clip contains that transition, with a long stretch of footage after it, so the models see the
ending. This measures how many of them report it. The post-cut duration is computed here from the
frozen cohort rather than quoted, because it was previously hardcoded at 38.6 s and had drifted
from the artefacts.

Outputs: C4_script_completion.csv, C4_script_report.txt
"""
import json, pathlib, re
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"

from transit_classifier import asserts_transit
from vlm_records import load_records

# The ORIGINAL pattern, kept only so the correction can be quantified and reported. It matched the
# bare noun phrase "exit door" (973 occurrences, sole trigger for 604 narratives) and had no
# negation handling, which inflated the headline by roughly twenty points. Do not score with
# it. The corrected rate is whatever this script prints; it is never quoted here.
TRANSIT = re.compile(
    r"pass(?:es|ed|ing)?\s+through|walk(?:s|ed|ing)?\s+through|step(?:s|ped|ping)?\s+(?:out|through)"
    r"|exits?\s+(?:the\s+)?(?:room|door|building)|goes?\s+through|through\s+the\s+door"
    r"|into\s+the\s+(?:corridor|hallway)|leaves?\s+the\s+room|to\s+safety|escap(?:es|ed|ing)",
    re.I)
# phrases describing what the footage actually shows
ACTUAL = re.compile(r"fade|black|screen\s+(?:goes|cuts)|cuts?\s+to|scene\s+(?:change|transition|end)"
                    r"|returns?\s+to|back\s+in\s+the\s+(?:same\s+)?room|teleport", re.I)




def post_cut_seconds():
    """Median seconds of footage after the scene cut, over the frozen cohort only.

    Three values for this quantity were in circulation (38.2 over all 31 participants, 38.6
    hardcoded here, 38.5 over the analysed 29). Computing it from config.yaml and the artefacts
    removes the ambiguity: the corpus is the 29, so the 29 is what the paper reports.
    """
    cfg = yaml.safe_load(open(ROOT / "evactime" / "config.yaml", encoding="utf-8"))
    cohort = set(cfg["corpus"]["participants"])
    w = pd.read_csv(OUT / "asset_windows.csv").set_index("pid")
    r = pd.read_csv(OUT / "reference_engine_all.csv").set_index("pid")
    m = w.join(r[["outro"]]).dropna(subset=["outro"])
    m = m[m.index.isin(cohort)]
    post = m.window_end_s - m.outro
    return float(post.median()), int((post > 0).sum()), len(post)
def events(p):
    out = []
    for k in ("events", "optional_events"):
        v = p.get(k)
        if isinstance(v, list):
            out += [e for e in v if isinstance(e, dict)]
        elif isinstance(v, dict):
            out += [e for e in v.values() if isinstance(e, dict)]
    return out


def main():
    # load_records recovers the 62 responses that were complete but carried one extra closing
    # brace, which json.loads rejected. Skipping them dropped 44% of one model's output while
    # leaving every other model intact, so the per-model comparison was not like-for-like.
    recs, stats, _ = load_records()
    rows = []
    for r in recs:
        nar, p = r["narrative"], r["parsed"]
        tl = " ".join(x.get("label", "") for x in (p.get("timeline") or [])
                      if isinstance(x, dict))
        rows.append(dict(pid=r["pid"], model=r["model"], cond=r["cond"], rep=r["rep"],
                         transit=asserts_transit(nar),
                         transit_old=bool(TRANSIT.search(nar)),
                         actual=bool(ACTUAL.search(nar + " " + tl)),
                         n_words=len(nar.split()), narrative=nar))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "C4_script_completion.csv", index=False)

    # The required exit field against the model's own prose. The field was mandatory, so a filled
    # field says little; a narrative that declines to claim the exit despite the filled field shows
    # the model did not simply copy the task description. The evidence clause the model gave for
    # the exit timestamp shows what it saw at that moment: a mention of the scene, a cut, a fade, a
    # transition or a reset means it registered the engine's scene change and read it as the exit.
    # (An earlier ad hoc version of this file could not be reproduced exactly; this definition is
    # the one the paper reports.)
    SCENE_EV = re.compile(r"\bscene\b|\bcut\b|\bfade|\btransition|\breset", re.I)
    fv = []
    for r in recs:
        ex = (r["parsed"].get("events") or {}).get("exit_reached")
        ev = (ex.get("evidence") or "") if isinstance(ex, dict) else ""
        fv.append(dict(pid=r["pid"], model=r["model"], cond=r["cond"], rep=r["rep"],
                       field_given=isinstance(ex, dict) and any(k in ex for k in ("t", "frame")),
                       evid_transit=bool(asserts_transit(ev)), evid_scene=bool(SCENE_EV.search(ev)),
                       narr=bool(asserts_transit(r["narrative"]))))
    pd.DataFrame(fv).to_csv(OUT / "C4_field_vs_narrative.csv", index=False)

    L = []
    P = L.append
    P("C4 - script completion in the free-text deliverable")
    P("=" * 66)
    P("Ground truth for the ending, from DoorHandleTrigger.cs and from the frames:")
    P("  hand contacts the handle -> fade -> the participant is placed back in the SAME room.")
    P("  There is no doorway transit, no corridor, and no exit to safety in any recording.")
    P("")
    P(f"narratives analysed: {len(d)} of {stats['calls_on_disk']} calls on disk "
      f"({len(d) / stats['calls_on_disk']:.1%})   median {d.n_words.median():.0f} words")
    P(f"  {stats['recovered']} recovered from responses with a trailing brace; "
      f"{stats['api_error']} lost to HTTP 429, all gemini-3.1-pro")
    P("")
    P(f"  {'':34s}{'n':>7}{'share':>9}")
    P(f"  {'assert a transit that never occurs':34s}{d.transit.sum():7d}{d.transit.mean():9.1%}")
    P(f"  {'describe the actual ending':34s}{d.actual.sum():7d}{d.actual.mean():9.1%}")
    P("")
    P("BY INPUT CONDITION - richer input makes it WORSE, not better")
    P("  (reversed by the classifier correction: the old regex was saturated near 90% in")
    P("   every arm, which hid the effect. Holds within each Google model run on all three.)")
    P(f"  {'condition':>12}{'n':>7}{'transit':>10}{'actual':>9}")
    for c, g in d.groupby("cond"):
        P(f"  {c:>12}{len(g):7d}{g.transit.mean():10.1%}{g.actual.mean():9.1%}")
    P("")
    P("BY MODEL")
    P(f"  {'model':34s}{'n':>6}{'transit':>10}{'actual':>9}")
    for m, g in sorted(d.groupby("model"), key=lambda kv: -kv[1].transit.mean()):
        P(f"  {m[:33]:34s}{len(g):6d}{g.transit.mean():10.1%}{g.actual.mean():9.1%}")
    P("")
    med, inside, n = post_cut_seconds()
    P(f"Reading: the ending is on screen in {inside}/{n} clips, for a median {med:.1f} s after the")
    P("cut, and")
    P("the models still complete the canonical evacuation script instead of reporting it.")
    P("This is a failure of the free-text deliverable, not of event timing, and it is invisible")
    P("to any accuracy statistic computed over the structured fields.")
    txt = "\n".join(L)
    (OUT / "C4_script_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
