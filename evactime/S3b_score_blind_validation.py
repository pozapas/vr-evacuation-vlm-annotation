"""S3b - score the frozen exit-claim classifier against the blind labels (review, 25-26 Sep 2026).

The labels were given by one author, blind to model, format and classifier output, on 100
narratives that no one had read (S3_blind_validation_sheet.py). The classifier is checked against
the SHA-256 recorded when the sample was drawn, so the version scored is the frozen one.

Primary analysis: unsure labels excluded. Sensitivity: unsure counted as yes, and as no.
Outputs: docs/classifier_blind_validation/labels.csv, outputs/S3_blind_validation.json
Run: py evactime/S3b_score_blind_validation.py <returned sheet.xlsx>
"""
import hashlib, json, pathlib, sys
import numpy as np
import pandas as pd

from vlm_records import _parse
from transit_classifier import asserts_transit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "evactime" / "outputs"
DST = ROOT / "docs" / "classifier_blind_validation"


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def metrics(y, g):
    tp, fp = int((y & g).sum()), int((~y & g).sum())
    fn, tn = int((y & ~g).sum()), int((~y & ~g).sum())
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return dict(n=n, tp=tp, fp=fp, fn=fn, tn=tn,
                recall=tp / (tp + fn), recall_ci=wilson(tp, tp + fn),
                precision=tp / (tp + fp), precision_ci=wilson(tp, tp + fp),
                specificity=tn / (tn + fp), specificity_ci=wilson(tn, tn + fp),
                accuracy=po, accuracy_ci=wilson(tp + tn, n), kappa=(po - pe) / (1 - pe),
                human_rate=(tp + fn) / n, classifier_rate=(tp + fp) / n)


def main(sheet):
    key = json.load(open(DST / "key_do_not_open.json", encoding="utf-8"))
    sha = hashlib.sha256((ROOT / "evactime" / "transit_classifier.py").read_bytes()).hexdigest()
    if sha != key["classifier_sha256"]:
        raise SystemExit("the classifier changed after the sample was drawn; the blind test is void")
    lab = pd.read_excel(sheet, sheet_name="label")
    lab["label"] = lab.left_the_room.astype(str).str.strip().str.lower()
    if set(lab.label) - {"yes", "no", "unsure"} or len(lab) != len(key["items"]):
        raise SystemExit("labels incomplete or malformed")
    items = pd.DataFrame(key["items"])
    d = items.merge(lab[["id", "narrative", "label"]], on="id")
    d.to_csv(DST / "labels.csv", index=False)

    # re-derive each narrative from the raw record and check it is the text that was labeled
    raw = {}
    for f in set(d.file):
        for line in open(OUT / f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("prompt_version") == "v2":
                raw[(f, r["model"], r["cond"], r["pid"], r.get("rep"))] = (_parse(r) or {}).get("narrative", "").strip()
    d["source"] = [raw[(r.file, r.model, r.cond, r.pid, r.rep)] for r in d.itertuples()]
    if not (d.source == d.narrative).all():
        raise SystemExit("a labeled narrative does not match its source record")
    d["pred"] = d.narrative.map(asserts_transit)

    res = dict(classifier_sha256=sha, labels=d.label.value_counts().to_dict())
    sure = d[d.label != "unsure"]
    res["primary"] = metrics((sure.label == "yes").values, sure.pred.values)
    res["unsure_as_yes"] = metrics(d.label.isin(["yes", "unsure"]).values, d.pred.values)
    res["unsure_as_no"] = metrics((d.label == "yes").values, d.pred.values)
    res["unsure_flagged"] = int(d[d.label == "unsure"].pred.sum())
    res["by_cell"] = {f"{m}|{c}": dict(n=len(g), human_yes=int((g.label == "yes").sum()),
                                       unsure=int((g.label == "unsure").sum()),
                                       flagged=int(g.pred.sum()))
                      for (m, c), g in d.groupby(["model", "cond"])}
    (OUT / "S3_blind_validation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    p = res["primary"]
    print(f"labels {res['labels']}")
    print(f"primary (unsure excluded, n={p['n']}): TP {p['tp']} FP {p['fp']} FN {p['fn']} TN {p['tn']}")
    for k in ("recall", "precision", "specificity", "accuracy"):
        print(f"  {k:12s} {p[k]:.3f}  [{p[k + '_ci'][0]:.3f}, {p[k + '_ci'][1]:.3f}]")
    print(f"  kappa {p['kappa']:.3f}; human rate {p['human_rate']:.3f}, classifier rate {p['classifier_rate']:.3f}")
    for k in ("unsure_as_yes", "unsure_as_no"):
        q = res[k]
        print(f"{k}: recall {q['recall']:.3f} precision {q['precision']:.3f} kappa {q['kappa']:.3f}")
    print(f"unsure items flagged by the classifier: {res['unsure_flagged']} of {res['labels'].get('unsure', 0)}")
    for c, v in res["by_cell"].items():
        print(f"  {c:45s} {v}")
    return d


if __name__ == "__main__":
    main(sys.argv[1])
