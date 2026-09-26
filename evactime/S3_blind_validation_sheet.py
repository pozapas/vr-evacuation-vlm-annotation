"""S3 - blind validation sample for the exit-claim classifier (review, 25 Sep 2026).

The classifier's rules were developed against the human-audit labels and hand-read narratives, so
those labels cannot validate it. This script draws a fresh sample from narratives that nobody has
read: the same-day repeat under the original task description (GPT-5.2, Claude Opus 4.8 and
Qwen3-VL 235B on frames) and the native same-day Gemini 3 Flash calls on video and on frames.
P1 is excluded because its narratives were read for Figure 4.

Twenty narratives are drawn from each of the five cells (seed fixed), shuffled, and written to a
labeling sheet without model, condition or classifier output. The classifier file is hashed now
and the hash is stored with the key, so the version validated is the version frozen before any
label was seen. S3b scores the returned labels.

Outputs: docs/classifier_blind_validation/blind_sheet.xlsx, key_do_not_open.json
"""
import hashlib, json, pathlib
import numpy as np
import pandas as pd

from vlm_records import _parse

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
DST = ROOT / "docs" / "classifier_blind_validation"
PER_CELL, SEED = 20, 20260925


def main():
    DST.mkdir(parents=True, exist_ok=True)
    if (DST / "key_do_not_open.json").exists():
        raise SystemExit("the sample already exists; drawing again would break the blind")
    rows = []
    for f in ("v2_sameday_raw.jsonl", "gemini_native_sameday_raw.jsonl"):
        for line in open(OUT / f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("prompt_version") != "v2" or r["pid"] == "P1":
                continue
            p = _parse(r)
            nar = (p or {}).get("narrative") or ""
            if len(nar.split()) < 20:
                continue
            rows.append(dict(file=f, model=r["model"], cond=r["cond"], pid=r["pid"], rep=r.get("rep"),
                             narrative=nar.strip()))
    d = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    pick = pd.concat([g.iloc[rng.choice(len(g), PER_CELL, replace=False)]
                      for _, g in d.groupby(["model", "cond"])])
    pick = pick.iloc[rng.permutation(len(pick))].reset_index(drop=True)
    pick["id"] = [f"V{i + 1:03d}" for i in range(len(pick))]

    sha = hashlib.sha256((ROOT / "evactime" / "transit_classifier.py").read_bytes()).hexdigest()
    key = dict(classifier_sha256=sha, seed=SEED, per_cell=PER_CELL, pool_size=len(d),
               items=pick[["id", "file", "model", "cond", "pid", "rep"]].to_dict("records"))
    (DST / "key_do_not_open.json").write_text(json.dumps(key, indent=1), encoding="utf-8")

    with pd.ExcelWriter(DST / "blind_sheet.xlsx", engine="openpyxl") as xw:
        pd.DataFrame({"Instructions": [
            "Read each narrative and answer one question in column C.",
            "Does the narrative state or clearly imply that the participant went through the door, "
            "left the room, or reached a place outside the room (a hallway, corridor, outside, safety)?",
            "Answer yes, no or unsure. Reaching or touching the door or handle alone is no.",
            "A scene cut, fade or reset described WITHOUT saying the participant left is no.",
            "Do not look up the recording or any model output. Label from the text only.",
        ]}).to_excel(xw, sheet_name="instructions", index=False)
        pick[["id", "narrative"]].assign(left_the_room="").to_excel(xw, sheet_name="label", index=False)
        ws = xw.sheets["label"]
        ws.column_dimensions["A"].width = 7
        ws.column_dimensions["B"].width = 110
        ws.column_dimensions["C"].width = 16
        from openpyxl.styles import Alignment
        from openpyxl.worksheet.datavalidation import DataValidation
        for row in ws.iter_rows(min_row=2, max_col=2):
            row[1].alignment = Alignment(wrap_text=True, vertical="top")
        dv = DataValidation(type="list", formula1='"yes,no,unsure"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"C2:C{len(pick) + 1}")
        xw.sheets["instructions"].column_dimensions["A"].width = 120
    print(f"pool {len(d)}, drew {len(pick)}; classifier sha256 {sha[:16]}...")


if __name__ == "__main__":
    main()
