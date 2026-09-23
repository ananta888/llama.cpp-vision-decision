#!/usr/bin/env python3
# Accuracy and calibration of /v1/decision on a labelled image set (see shapes.py), and a per-field
# temperature fit. A schema-valid answer is not a correct one: this measures how often it is right.
#   evaluate.py --url http://127.0.0.1:8080 DIR [--batch 4] [--out result.json]
import argparse
import base64
import json
import math
import os
import time
import urllib.request


def post(url, body):
    req = urllib.request.Request(url + "/v1/decision", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def context(path, text):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}, {"type": "text", "text": text}]


def ece(conf, correct, bins=10):
    total, err = len(conf), 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(conf) if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if idx:
            err += len(idx) / total * abs(sum(correct[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return err


def tempered(probs, t):
    p = [max(q, 1e-30) ** (1.0 / t) for q in probs]
    z = sum(p)
    return [q / z for q in p]


def nll(rows, t):
    return sum(-math.log(max(tempered(p, t)[y], 1e-30)) for p, y in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--text", default="Answer the questions about this picture.")
    ap.add_argument("--out")
    args = ap.parse_args()

    schema = json.load(open(os.path.join(args.dir, "schema.json")))
    items = [json.loads(l) for l in open(os.path.join(args.dir, "labels.jsonl"))]
    rows = {name: [] for name in schema}  # (probs, index of the true value)
    hits = {name: 0 for name in schema}
    t0, ms = time.time(), []
    for k in range(0, len(items), args.batch):
        chunk = items[k:k + args.batch]
        res = post(args.url, {"schema": schema, "contexts": [context(os.path.join(args.dir, it["image"]), args.text) for it in chunk],
                              "return_probs": True, "mode": "tree"})
        ms.append(res["timings"]["per_decision_ms"])
        for it, r in zip(chunk, res["results"]):
            for name, f in r["fields"].items():
                values = [c["value"] for c in f["probs"]]
                y = values.index(it["labels"][name])
                rows[name].append(([c["probability"] for c in f["probs"]], y))
                hits[name] += r["decision"][name] == it["labels"][name]

    report = {"n": len(items), "wall_s": round(time.time() - t0, 2), "mean_per_decision_ms": round(sum(ms) / len(ms), 1), "fields": {}}
    for name, rs in rows.items():
        grid = [math.exp(math.log(0.05) + i * (math.log(20) - math.log(0.05)) / 199) for i in range(200)]
        t_best = min(grid, key=lambda t: nll(rs, t))
        conf1 = [max(p) for p, _ in rs]
        confT = [max(tempered(p, t_best)) for p, _ in rs]
        correct = [int(p.index(max(p)) == y) for p, y in rs]
        report["fields"][name] = {
            "accuracy": round(hits[name] / len(items), 4),
            "mean_confidence": round(sum(conf1) / len(conf1), 4),
            "nll": round(nll(rs, 1.0), 4), "ece": round(ece(conf1, correct), 4),
            "fitted_temperature": round(t_best, 3),
            "nll_fitted": round(nll(rs, t_best), 4), "ece_fitted": round(ece(confT, correct), 4),
        }
    print(json.dumps(report, indent=1))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
