#!/usr/bin/env python3
# Latency benchmark for vision decisions on the images of shapes.py.
#   bench.py --url http://127.0.0.1:8080 DIR [--images 8] [--out bench.json]
# 1. decision vs chat completion with a json_schema response format (same image, same fields)
# 2. number of fields on one image (1, 2, 4, 8, 16)
# 3. N images: one request with N contexts vs N requests
# 4. cold vs warm cached instructions
import argparse
import base64
import json
import os
import statistics
import time
import urllib.request

EXTRA = {  # more yes/no questions, so the schema can grow to 16 fields
    f"has_{c}": {"type": "boolean", "description": f"Is any shape {c}?"} for c in ["red", "green", "blue", "yellow"]
}
EXTRA.update({f"has_{s}": {"type": "boolean", "description": f"Is any shape a {s}?"} for s in ["circle", "square", "triangle"]})
EXTRA.update({
    "top_left": {"type": "boolean", "description": "Is there a shape in the top left quarter?"},
    "top_right": {"type": "boolean", "description": "Is there a shape in the top right quarter?"},
    "bottom_left": {"type": "boolean", "description": "Is there a shape in the bottom left quarter?"},
    "bottom_right": {"type": "boolean", "description": "Is there a shape in the bottom right quarter?"},
    "more_than_two": {"type": "boolean", "description": "Are there more than two shapes?"},
})


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t) * 1000


def image_url(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def ctx(path):
    return [{"type": "image_url", "image_url": {"url": image_url(path)}}, {"type": "text", "text": "Answer the questions about this picture."}]


def json_schema(schema):
    props = {}
    for name, f in schema.items():
        if f["type"] == "enum":
            props[name] = {"type": "string", "enum": f["choices"]}
        elif f["type"] == "integer":
            props[name] = {"type": "integer", "minimum": f["minimum"], "maximum": f["maximum"]}
        else:
            props[name] = {"type": "boolean"}
    return {"type": "object", "properties": props, "required": list(schema), "additionalProperties": False}


def chat(url, schema, path):
    questions = "\n".join(f"- {n}: {f['description']}" for n, f in schema.items())
    body = {"temperature": 0, "max_tokens": 256,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url(path)}},
                                                      {"type": "text", "text": "Answer as JSON:\n" + questions}]}],
            "response_format": {"type": "json_schema", "json_schema": {"schema": json_schema(schema)}}}
    out, ms = post(url, "/v1/chat/completions", body)
    return json.loads(out["choices"][0]["message"]["content"]), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--out")
    args = ap.parse_args()
    base = json.load(open(os.path.join(args.dir, "schema.json")))
    items = [json.loads(l) for l in open(os.path.join(args.dir, "labels.jsonl"))][:args.images]
    paths = [os.path.join(args.dir, it["image"]) for it in items]
    report = {}

    # 1. decision vs chat completion
    dec_ms, chat_ms, agree, dec_ok, chat_ok, n = [], [], 0, 0, 0, 0
    post(args.url, "/v1/decision", {"schema": base, "contexts": [ctx(paths[0])]})  # warm the cached instructions
    for it, p in zip(items, paths):
        d, ms = post(args.url, "/v1/decision", {"schema": base, "contexts": [ctx(p)]})
        dec_ms.append(ms)
        c, ms = chat(args.url, base, p)
        chat_ms.append(ms)
        for k in base:
            n += 1
            agree += d["results"][0]["decision"][k] == c.get(k)
            dec_ok += d["results"][0]["decision"][k] == it["labels"][k]
            chat_ok += c.get(k) == it["labels"][k]
    report["decision_vs_chat"] = {"decision_ms_median": round(statistics.median(dec_ms), 1), "chat_json_schema_ms_median": round(statistics.median(chat_ms), 1),
                                  "field_agreement": round(agree / n, 3), "decision_accuracy": round(dec_ok / n, 3), "chat_accuracy": round(chat_ok / n, 3)}

    # 2. fields on one image
    full = dict(base, **EXTRA)
    names = list(full)
    report["fields_scaling"] = []
    for k in (1, 2, 4, 8, 16):
        schema = {n: full[n] for n in names[:k]}
        post(args.url, "/v1/decision", {"schema": schema, "contexts": ["warm"]})
        d, ms = post(args.url, "/v1/decision", {"schema": schema, "contexts": [ctx(paths[0])]})
        report["fields_scaling"].append({"fields": k, "total_ms": round(ms, 1), "prefill_ms": round(d["timings"]["prefill_ms"], 1),
                                         "scoring_ms": round(d["timings"]["scoring_ms"], 1), "scored_rows": d["usage"]["scored_rows"]})

    # 3. N images in one request vs N requests
    report["multi_context"] = []
    for k in sorted({k for k in (1, 2, 4, len(paths)) if k <= len(paths)}):
        _, one = post(args.url, "/v1/decision", {"schema": base, "contexts": [ctx(p) for p in paths[:k]]})
        sep = sum(post(args.url, "/v1/decision", {"schema": base, "contexts": [ctx(p)]})[1] for p in paths[:k])
        report["multi_context"].append({"images": k, "one_request_ms": round(one, 1), "separate_requests_ms": round(sep, 1)})

    # 4. cold vs warm instructions cache
    cold, cms = post(args.url, "/v1/decision", {"schema": base, "contexts": [ctx(paths[0])], "cache_prompt": False})
    warm, wms = post(args.url, "/v1/decision", {"schema": base, "contexts": [ctx(paths[0])]})
    report["cache"] = {"cold_ms": round(cms, 1), "warm_ms": round(wms, 1), "cached_tokens_warm": warm["usage"]["cached_tokens"],
                       "media_encode_ms": round(warm["timings"].get("media_encode_ms", 0), 1)}
    print(json.dumps(report, indent=1))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
