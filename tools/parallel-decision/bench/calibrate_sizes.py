#!/usr/bin/env python3
# Calibrate numeric (e.g. body size) fields of /v1/decision on photos with measured values. Same method as the
# playground page: per field an offset (from 2 samples) or a straight line (from 5), kept only if leave-one-out
# beats the raw error, and an 80% range from the leave-one-out residuals (split conformal). The output file can
# be imported in the playground ("import"), and it keeps the samples, never the photos.
#
#   calibrate_sizes.py DIR --schema body-schema.json --url http://127.0.0.1:8080 --out size-calibration.json
#   DIR/labels.jsonl: {"image": "p01.jpg", "sizes": {"hip_cm": 96, "inseam_cm": null, ...}}   (null: not visible)
import argparse
import base64
import datetime
import json
import math
import os
import urllib.request


def post(url, body):
    req = urllib.request.Request(url + "/v1/decision", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def fit(pairs, linear):
    me = sum(p[0] for p in pairs) / len(pairs)
    mt = sum(p[1] for p in pairs) / len(pairs)
    sxx = sum((e - me) ** 2 for e, _ in pairs)
    sxy = sum((e - me) * (t - mt) for e, t in pairs)
    if not linear or sxx == 0:
        return 1.0, mt - me
    a = sxy / sxx
    return a, mt - a * me


def fit_field(pairs):
    n = len(pairs)
    if n < 2:
        return None

    def loo(linear):
        out = []
        for i, (e, t) in enumerate(pairs):
            a, b = fit(pairs[:i] + pairs[i + 1:], linear)
            out.append(t - (a * e + b))
        return out

    mae = lambda rs: sum(abs(r) for r in rs) / len(rs)
    raw = mae([t - e for e, t in pairs])
    r_off = loo(False)
    r_lin = loo(True) if n >= 5 else None
    linear = r_lin is not None and mae(r_lin) < mae(r_off)
    res = r_lin if linear else r_off
    a, b = fit(pairs, linear)
    absr = sorted(abs(r) for r in res)
    k = math.ceil((n + 1) * 0.8) - 1
    return {"n": n, "kind": "line" if linear else "offset", "a": a, "b": b, "mae_raw": raw, "mae_cal": mae(res),
            "half_width": absr[min(k, n - 1)], "range_ok": k < n, "apply": mae(res) < raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--text", default="Answer the questions about this picture.")
    ap.add_argument("--out", default="size-calibration.json")
    args = ap.parse_args()
    schema = json.load(open(args.schema))
    items = [json.loads(l) for l in open(os.path.join(args.dir, "labels.jsonl"))]
    samples = []
    for it in items:
        with open(os.path.join(args.dir, it["image"]), "rb") as f:
            url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
        r = post(args.url, {"schema": schema, "contexts": [[{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": args.text}]],
                            "return_probs": True})["results"][0]
        est = {}
        for name in it["sizes"]:
            f = r["fields"].get(name)
            if f is None:
                continue
            p_null = next((c["probability"] for c in f.get("probs", []) if c["value"] is None), None)
            iv = f.get("interval_p10_p90")
            est[name] = {"value": f["value"], "lo": iv[0] if iv else None, "hi": iv[1] if iv else None, "p_null": p_null}
        samples.append({"at": it["image"], "est": est, "truth": it["sizes"]})

    fits = {}
    for name in sorted({n for s in samples for n in s["truth"]}):
        pairs = [(s["est"][name]["value"], s["truth"][name]) for s in samples
                 if s["truth"].get(name) is not None and name in s["est"] and s["est"][name]["value"] is not None]
        known = [s for s in samples if name in s["truth"] and name in s["est"]]
        fits[name] = {"fit": fit_field(pairs), "none_right": sum((s["truth"][name] is None) == (s["est"][name]["value"] is None) for s in known),
                      "none_n": len(known)}
        f = fits[name]["fit"]
        if f:
            print(f"{name:20s} n {f['n']:3d}  {f['kind']:6s} a {f['a']:.3f} b {f['b']:+.2f}  error raw {f['mae_raw']:.2f} -> calibrated {f['mae_cal']:.2f}"
                  f"  80% range +-{f['half_width']:.2f}{'' if f['range_ok'] else ' (few samples)'}  {'used' if f['apply'] else 'not better'}"
                  f"  none right {fits[name]['none_right']}/{fits[name]['none_n']}")
        else:
            print(f"{name:20s} fewer than 2 measured samples")
    json.dump({"kind": "decision-size-calibration", "version": 1, "exported": datetime.datetime.now().isoformat(),
               "samples": samples, "fits": fits}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
