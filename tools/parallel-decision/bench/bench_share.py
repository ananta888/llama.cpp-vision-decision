#!/usr/bin/env python3
# Compare the ways of answering a schema on the images of shapes.py, image by image and interleaved:
# /v1/decision with shared branch tokens (default), /v1/decision with share_tokens false, and a chat completion
# with a json_schema response. Two schemas: the 4 shape questions, and numeric questions (pixels, nullable)
# whose values share leading digits.
#   bench_share.py DIR --url http://127.0.0.1:8080 [--images 8] [--out bench-share.json]
import argparse
import json
import os
import statistics

from bench import chat, ctx, post

NUMERIC = {
    "center_x": {"type": "integer", "minimum": 0, "maximum": 249, "nullable": True, "description": "x pixel of the center of the largest shape; null if there is none."},
    "center_y": {"type": "integer", "minimum": 0, "maximum": 249, "nullable": True, "description": "y pixel of the center of the largest shape; null if there is none."},
    "width": {"type": "integer", "minimum": 0, "maximum": 249, "nullable": True, "description": "Width of the largest shape in pixels; null if there is none."},
    "height": {"type": "integer", "minimum": 0, "maximum": 249, "nullable": True, "description": "Height of the largest shape in pixels; null if there is none."},
    "covered_percent": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Percent of the image covered by shapes."},
}


def chat_schema(schema):
    # bench.chat only knows enum / integer / boolean; a nullable integer becomes an integer or null
    out = {}
    for n, f in schema.items():
        out[n] = dict(f, type="integer") if f.get("nullable") else f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--mode", default="tree", help="tree (exact probabilities) or auto (greedy above 128 values)")
    ap.add_argument("--out")
    args = ap.parse_args()
    base = json.load(open(os.path.join(args.dir, "schema.json")))
    items = [json.loads(l) for l in open(os.path.join(args.dir, "labels.jsonl"))][:args.images]
    paths = [os.path.join(args.dir, it["image"]) for it in items]
    report = {}
    for name, schema in (("shapes", base), ("numeric", NUMERIC)):
        ms = {"shared": [], "single": [], "chat": []}
        scoring = {"shared": [], "single": []}
        rows = {}
        same, n, maxdp = 0, 0, 0.0
        for share in (True, False):  # warm the cached instructions
            post(args.url, "/v1/decision", {"schema": schema, "contexts": [ctx(paths[0])], "share_tokens": share})
        for p in paths:
            out = {}
            for key, share in (("shared", True), ("single", False)):
                d, t = post(args.url, "/v1/decision", {"schema": schema, "contexts": [ctx(p)], "share_tokens": share, "mode": args.mode, "return_probs": True})
                ms[key].append(t)
                scoring[key].append(d["timings"]["scoring_ms"])
                rows[key] = (d["usage"]["scored_rows"], d["usage"].get("decoded_rows"))
                out[key] = d["results"][0]
            _, t = chat(args.url, chat_schema(schema), p)
            ms["chat"].append(t)
            for f in schema:
                n += 1
                same += out["shared"]["decision"][f] == out["single"]["decision"][f]
                pa = [c["probability"] for c in out["shared"]["fields"][f].get("probs", [])]
                pb = [c["probability"] for c in out["single"]["fields"][f].get("probs", [])]
                maxdp = max([maxdp] + [abs(a - b) for a, b in zip(pa, pb)])
        report[name] = {
            "total_ms_median": {k: round(statistics.median(v), 1) for k, v in ms.items()},
            "scoring_ms_median": {k: round(statistics.median(v), 1) for k, v in scoring.items()},
            "rows_scored_decoded": rows,
            "same_decision_shared_vs_single": round(same / n, 3),
            "max_abs_probability_diff": round(maxdp, 4),
        }
    print(json.dumps(report, indent=1))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
