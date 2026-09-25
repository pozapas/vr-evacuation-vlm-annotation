"""Figure 2 (study design) built on the TRB framework figure: same geometry, fonts and icons.

Copies paper/figures/fig1_pipeline.drawio (the TRB 2027 framework figure) and replaces only the
text of each cell, keyed by cell id, plus the one pink fill in the TRB file, which becomes the TRB
file's own peach so the paper carries no pink. Positions, sizes, font sizes, icons and arrows are
untouched. Numbers are read from the analysis outputs.

Writes paper/figures/q1_fig1_framework.drawio; export with draw.io to q1_fig1_pipeline.pdf.
Run: py evactime/make_framework_from_trb.py
"""
import json, pathlib
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
OUT = ROOT / "evactime" / "outputs"
SRC, DST = FIG / "fig1_pipeline.drawio", FIG / "q1_fig1_framework.drawio"


def bullets(title, items):
    body = "".join(f'<div style="text-align: left;">&#8226; {it}</div>' for it in items)
    return f'<div style="line-height: 180%;"><b>{title}</b><br>{body}</div>'


def stage_body(title, items):
    return ('<div style="line-height: 180%;"><b>' + title + "</b><br>"
            + "<br>".join("&#8226; " + it for it in items) + "</div>")


def main():
    c2 = json.load(open(OUT / "C2_confusion.json", encoding="utf-8"))
    ko, kr = c2["kappa_object"], c2["kappa_region"]
    text = {
        "stage1_header": "<b>Stage 1</b><br><b>Study corpus</b>",
        "stage1_body": stage_body("29 VR evacuation recordings",
                                  ["31 recorded, 2 excluded", "First-person view, 75 s window",
                                   "Alarm, walk and scene cut", "Shared engine timebase"]),
        "stage2_header": "<b>Stage 2</b><br><b>Engine reference channels</b>",
        "stage2_a": "<b>Engine markers</b><br>Waiting room, alarm and scene cut",
        "stage2_b": "<b>Gaze raycast</b><br>Tagged object struck at 48 Hz",
        "stage2_c": "<b>Scene geometry</b><br>Object positions and active state",
        "stage4_header": "<b>Stage 4</b><br><b>VLM annotation</b>",
        "stage4_body": stage_body("8 vision-language models",
                                  ["Five model developers", "Three repeated calls",
                                   "Attention and exit claims", "Original and neutral prompts"]),
        "engine_ref": bullets("Engine reference", ["Modal gaze object within &#177;1 s",
                                                   "Back-wall region and characters",
                                                   "Scene cut logged in all 29"]),
        "comparison": bullets("Model-engine comparison", ["Object and region agreement",
                                                          "Chance level and &#954; by permutation",
                                                          "Exit claims in narratives"]),
        "outputs": bullets("Analytical outputs", ["Scoring-rule sensitivity",
                                                  "Task-description effect (OR)",
                                                  "Human audit, Krippendorff &#945;"]),
        "cf_sal": "<b>Cue layout</b><br>lone vs grouped",
        "cf_mod": "<b>Annotation setup</b><br>prompt, route, input",
        "cf_err": "Attention and exit claims&nbsp; <b>c<sub>i</sub></b>",
        "cf_prop": "Engine check<br>modal gaze object, logged scene cut",
        "cf_rel": ("Agreement&nbsp; <b>&#954; = (A &#8722; A<sub>0</sub>) / "
                   "(1 &#8722; A<sub>0</sub>)</b>"),
        "cf_ok": '<div style="line-height: 100%;">Automate region</div>',
        "cf_no": "Instrument object",
    }
    recolor = {"#F8CECC": "#F7E8DB", "#B85450": "#C6753D"}   # the TRB file's pink -> its peach
    tree = ET.parse(SRC)
    seen = set()
    for c in tree.getroot().iter("mxCell"):
        cid = c.get("id")
        if cid in text:
            c.set("value", text[cid])
            seen.add(cid)
        st = c.get("style")
        if st:
            for a, b in recolor.items():
                st = st.replace(a, b)
            c.set("style", st)
    missing = set(text) - seen
    if missing:
        raise SystemExit(f"cells not found in the TRB file: {sorted(missing)}")
    tree.write(DST, encoding="utf-8", xml_declaration=False)
    print("wrote", DST)


if __name__ == "__main__":
    main()
