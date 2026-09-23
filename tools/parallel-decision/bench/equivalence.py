#!/usr/bin/env python3
# Check that /v1/decision scores an image context like the slot path does: the tree probabilities of an
# enum field must match the next-token distribution of /completion on the same prompt, renormalised over
# the candidates. Catches wrong positions (M-RoPE) and wrong KV contents. The server must run with
# LLAMA_MEDIA_MARKER="<__media__>".
#   equivalence.py --url http://127.0.0.1:8080 --image a.png [--image b.png] --choices red,green,blue
import argparse
import base64
import json
import math
import sys
import urllib.request

MARKER = "<__media__>"


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--image", action="append", required=True)
    ap.add_argument("--choices", required=True)
    ap.add_argument("--description", default="What is shown?")
    ap.add_argument("--before", default="Look at this picture:")
    ap.add_argument("--after", default="Answer the questions about it.")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    url, choices = args.url, args.choices.split(",")
    b64 = [base64.b64encode(open(p, "rb").read()).decode() for p in args.image]

    tok = lambda s: post(url, "/tokenize", {"content": s, "add_special": False, "parse_special": True})["tokens"]
    system = ("Select the requested field value from its allowed values, based on the context. Respond with the JSON value only.\n\n"
              "Fields:\n" + '"label": ' + args.description + "\nAllowed values: " + ", ".join(json.dumps(c) for c in choices) + "\n")
    text = args.before + MARKER * len(b64) + args.after  # parts join without newlines around media
    prompt = post(url, "/apply-template", {"messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]})["prompt"] + "{\n"

    paths = [tok('  "label": ' + json.dumps(c) + "\n") for c in choices]
    n = 0
    while all(len(p) > n + 1 and p[n] == paths[0][n] for p in paths):
        n += 1
    first = [p[n] for p in paths]
    if len(set(first)) != len(first):
        sys.exit("candidates must differ in their first scored token")
    suffix = post(url, "/detokenize", {"tokens": paths[0][:n]})["content"]
    tail = prompt.split(MARKER)[-1]
    if tok(tail + suffix) != tok(tail) + paths[0][:n]:
        sys.exit("tokenization seam differs from the engine's")

    comp = post(url, "/completion", {"prompt": {"prompt_string": prompt + suffix, "multimodal_data": b64},
                                    "n_predict": 1, "n_probs": 400000, "temperature": 0, "cache_prompt": False})
    top = {t["id"]: math.exp(t["logprob"]) for t in comp["completion_probabilities"][0]["top_logprobs"]}
    z = sum(top[t] for t in first)
    ref = {c: top[t] / z for c, t in zip(choices, first)}

    parts = [{"type": "text", "text": args.before}] if args.before else []
    parts += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}} for b in b64]
    parts += [{"type": "text", "text": args.after}]
    dec = post(url, "/v1/decision", {"schema": {"label": {"type": "enum", "choices": choices, "description": args.description}},
                                     "contexts": [parts], "mode": "tree", "cache_prompt": False, "return_probs": True})
    got = {c["value"]: c["probability"] for c in dec["results"][0]["fields"]["label"]["probs"]}
    diff = max(abs(got[c] - ref[c]) for c in choices)
    print(json.dumps({"images": args.image, "decision": got, "completion": ref, "max_abs_diff": diff,
                      "media_tokens": dec["usage"].get("media_tokens"), "context_tokens": dec["usage"]["context_tokens"],
                      "timings": dec["timings"]}, indent=1))
    sys.exit(0 if diff < args.tol else 1)


if __name__ == "__main__":
    main()
