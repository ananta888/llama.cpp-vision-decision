#!/usr/bin/env python3
# Accuracy and calibration of /v1/decision on a labelled image set (see shapes.py). A schema-valid answer is
# not a correct one: this measures how often it is right, and turns the probabilities into a usable confidence.
#
#   evaluate.py DIR --url http://127.0.0.1:8080                  accuracy, NLL, ECE, fitted temperature per field
#   evaluate.py DIR --target-accuracy 0.98 --schema-out cal.json  + per-field threshold (cross-validated) and a
#                                                                  schema with temperature and abstain rules
#   evaluate.py DIR --schema cal.json                             check a calibrated schema on new data: accuracy
#                                                                  and coverage of the answers the server accepts
#   --dump rows.json / --from rows.json                           keep the model outputs, re-run the analysis only
import argparse
import base64
import json
import math
import os
import random
import time
import urllib.request

T_GRID = [math.exp(math.log(0.25) + i * (math.log(10) - math.log(0.25)) / 199) for i in range(200)]


def post(url, body):
    req = urllib.request.Request(url + "/v1/decision", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def context(path, text):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}, {"type": "text", "text": text}]


def tempered(probs, t):
    p = [max(q, 1e-30) ** (1.0 / t) for q in probs]
    z = sum(p)
    return [q / z for q in p]


def nll(rows, t):
    return sum(-math.log(max(tempered(r["probs"], t)[r["y"]], 1e-30)) for r in rows) / len(rows)


def fit_temperature(rows):
    return min(T_GRID, key=lambda t: nll(rows, t))


def confidence(r, t):
    p = tempered(r["probs"], t)
    best = max(range(len(p)), key=p.__getitem__)
    return p[best], int(best == r["y"])


def ece(rows, t, bins=10):
    cs = [confidence(r, t) for r in rows]
    err = 0.0
    for b in range(bins):
        sel = [(c, ok) for c, ok in cs if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if sel:
            err += len(sel) / len(cs) * abs(sum(ok for _, ok in sel) / len(sel) - sum(c for c, _ in sel) / len(sel))
    return err


def choose_threshold(rows, t, target, min_accept):
    # lowest confidence at which the answers at or above it reach the target accuracy (None: never)
    scored = sorted((confidence(r, t) for r in rows), reverse=True)
    best, correct = None, 0
    for k, (c, ok) in enumerate(scored, 1):
        correct += ok
        # only cut between different confidences: a threshold accepts all answers that tie at it
        if k < len(scored) and scored[k][0] == c:
            continue
        if k >= min_accept and correct / k >= target:
            best = c
    return best


def accepted(rows, t, thr):
    if thr is None:
        return 0, 0
    sel = [ok for c, ok in (confidence(r, t) for r in rows) if c >= thr]
    return len(sel), sum(sel)


def calibrate(rows, target, folds, min_accept):
    idx = list(range(len(rows)))
    random.Random(0).shuffle(idx)
    n_acc = n_ok = 0
    for f in range(folds):
        test = set(idx[f::folds])
        train = [rows[i] for i in idx if i not in test]
        t = fit_temperature(train)
        thr = choose_threshold(train, t, target, min_accept)
        a, ok = accepted([rows[i] for i in test], t, thr)
        n_acc, n_ok = n_acc + a, n_ok + ok
    t = fit_temperature(rows)
    thr = choose_threshold(rows, t, target, min_accept)
    a, ok = accepted(rows, t, thr)
    return {
        "temperature": round(t, 3),
        "threshold": None if thr is None else math.floor(thr * 1e4) / 1e4,
        "cv_accepted_accuracy": round(n_ok / n_acc, 4) if n_acc else None,
        "cv_coverage": round(n_acc / len(rows), 4),
        "accepted_accuracy": round(ok / a, 4) if a else None,
        "coverage": round(a / len(rows), 4),
        "usable": thr is not None,
    }


def run_model(args, schema, items):
    rows = {name: [] for name in schema}
    served = {name: [] for name in schema}  # (abstain flag, correct) when the schema has abstain rules
    ms = []
    for k in range(0, len(items), args.batch):
        chunk = items[k:k + args.batch]
        res = post(args.url, {"schema": schema, "contexts": [context(os.path.join(args.dir, it["image"]), args.text) for it in chunk],
                              "return_probs": True, "mode": "tree"})
        ms.append(res["timings"]["per_decision_ms"])
        for it, r in zip(chunk, res["results"]):
            for name, f in r["fields"].items():
                values = [c["value"] for c in f["probs"]]
                rows[name].append({"probs": [c["probability"] for c in f["probs"]], "y": values.index(it["labels"][name])})
                if "abstain" in f:
                    served[name].append((f["abstain"], int(r["decision"][name] == it["labels"][name])))
    return rows, served, sum(ms) / len(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--schema", help="schema to send (default: DIR/schema.json)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--text", default="Answer the questions about this picture.")
    ap.add_argument("--target-accuracy", type=float)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-accept", type=int, default=5, help="fewest accepted answers a threshold must rest on")
    ap.add_argument("--schema-out")
    ap.add_argument("--dump")
    ap.add_argument("--from", dest="src")
    ap.add_argument("--out")
    args = ap.parse_args()

    schema = json.load(open(args.schema or os.path.join(args.dir, "schema.json")))
    items = [json.loads(l) for l in open(os.path.join(args.dir, "labels.jsonl"))]
    t0 = time.time()
    if args.src:
        saved = json.load(open(args.src))
        rows, served, mean_ms = saved["rows"], saved["served"], saved["mean_per_decision_ms"]
    else:
        rows, served, mean_ms = run_model(args, schema, items)
        if args.dump:
            json.dump({"rows": rows, "served": served, "mean_per_decision_ms": mean_ms}, open(args.dump, "w"))

    report = {"n": len(items), "wall_s": round(time.time() - t0, 2), "mean_per_decision_ms": round(mean_ms, 1), "fields": {}}
    for name, rs in rows.items():
        t = fit_temperature(rs)
        field = {
            "accuracy": round(sum(confidence(r, 1.0)[1] for r in rs) / len(rs), 4),
            "mean_confidence": round(sum(confidence(r, 1.0)[0] for r in rs) / len(rs), 4),
            "nll": round(nll(rs, 1.0), 4), "ece": round(ece(rs, 1.0), 4),
            "fitted_temperature": round(t, 3), "nll_fitted": round(nll(rs, t), 4), "ece_fitted": round(ece(rs, t), 4),
        }
        if served[name]:
            kept = [ok for ab, ok in served[name] if not ab]
            field["served_accepted_accuracy"] = round(sum(kept) / len(kept), 4) if kept else None
            field["served_coverage"] = round(len(kept) / len(served[name]), 4)
        if args.target_accuracy:
            field["calibration"] = calibrate(rs, args.target_accuracy, args.folds, args.min_accept)
        report["fields"][name] = field

    if args.target_accuracy and args.schema_out:
        out = json.loads(json.dumps(schema))
        for name, f in report["fields"].items():
            c = f["calibration"]
            out[name]["temperature"] = c["temperature"]
            # an unusable field abstains always: no threshold reached the target
            out[name]["abstain"] = {"min_probability": c["threshold"] if c["usable"] else 1.0}
        json.dump(out, open(args.schema_out, "w"), indent=1)
        report["schema_out"] = args.schema_out
    print(json.dumps(report, indent=1))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
