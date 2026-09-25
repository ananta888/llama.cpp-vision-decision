import base64
import json
import math
import time

import pytest
import requests
from utils import *

server: ServerProcess

IMG_URLS = [
    "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png",
    "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png",
]
_images: dict[str, str] = {}

LABELS = ["cat", "frog", "truck", "ship"]
SCHEMA = {
    "label":  {"type": "enum", "choices": LABELS, "description": "What is shown?"},
    "animal": {"type": "boolean", "description": "Is it an animal?"},
    "count":  {"type": "integer", "minimum": 0, "maximum": 3, "description": "How many objects?"},
}


def image_b64(i: int) -> str:
    url = IMG_URLS[i]
    for attempt in range(4):
        if url in _images:
            break
        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            _images[url] = base64.b64encode(res.content).decode("utf-8")
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2)  # a flaky network (DNS) should not fail the test
    return _images[url]


def image_context(images: list[int], text: str = "Look at this picture:") -> list[dict]:
    parts: list[dict] = [{"type": "text", "text": text}]
    for i in images:
        parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64(i)}})
    parts.append({"type": "text", "text": "Answer the questions about it."})
    return parts


@pytest.fixture(autouse=True)
def create_server():
    global server
    os.environ["LLAMA_MEDIA_MARKER"] = "<__media__>"
    server = ServerPreset.tinygemma3()
    server.n_ctx = 4096
    server.n_slots = 1
    server.decision_seqs = 12


def decide(data: dict) -> dict:
    res = server.make_request("POST", "/v1/decision", data=data)
    assert res.status_code == 200, res.body
    return res.body


def test_decision_text():
    global server
    server.start()
    data = {"schema": SCHEMA, "contexts": ["A photo of a small grey cat.", "A red truck on a road."]}
    first = decide(data)
    second = decide(data)
    assert len(first["results"]) == 2
    for r in first["results"]:
        assert r["decision"]["label"] in LABELS
        assert isinstance(r["decision"]["animal"], bool)
        assert r["decision"]["count"] in range(4)
        for f in r["fields"].values():
            assert 0.0 < f["probability"] <= 1.0
    assert "media_tokens" not in first["usage"]
    assert first["usage"]["cached_tokens"] == 0
    assert second["usage"]["cached_tokens"] > 0
    assert second["results"] == first["results"]


def reference_probs(parts: list[dict], images: list[int]) -> dict[str, float]:
    # next-token distribution of the slot path (/completion) on the same prompt, over the label candidates
    marker = "<__media__>"
    system = ("Select the requested field value from its allowed values, based on the context. Respond with the JSON value only.\n\nFields:\n"
              + '"label": What is shown?\nAllowed values: ' + ", ".join(json.dumps(c) for c in LABELS) + "\n")
    text, last_media = "", False
    for p in parts:
        if p["type"] == "text":
            text += ("\n" if text and not last_media else "") + p["text"]
            last_media = False
        else:
            text += marker
            last_media = True
    res = server.make_request("POST", "/apply-template", data={"messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]})
    assert res.status_code == 200
    prompt = res.body["prompt"] + "{\n"

    def tokens(s: str) -> list[int]:
        return server.make_request("POST", "/tokenize", data={"content": s, "add_special": False, "parse_special": True}).body["tokens"]

    paths = [tokens('  "label": ' + json.dumps(c) + "\n") for c in LABELS]
    n = 0
    while all(len(p) > n + 1 and p[n] == paths[0][n] for p in paths):
        n += 1
    first = [p[n] for p in paths]
    assert len(set(first)) == len(first), "candidates must differ in their first scored token"
    suffix = server.make_request("POST", "/detokenize", data={"tokens": paths[0][:n]}).body["content"]
    tail = prompt.split(marker)[-1]
    assert tokens(tail + suffix) == tokens(tail) + paths[0][:n], "tokenization seam differs from the engine's"

    res = server.make_request("POST", "/completion", data={
        "prompt": {"prompt_string": prompt + suffix, "multimodal_data": [image_b64(i) for i in images]},
        # the tiny test model ranks the labels low: ask for the whole vocabulary
        "n_predict": 1, "n_probs": 300000, "temperature": 0, "cache_prompt": False,
    })
    assert res.status_code == 200
    top = {t["id"]: math.exp(t["logprob"]) for t in res.body["completion_probabilities"][0]["top_logprobs"]}
    assert all(t in top for t in first), "a candidate is outside the returned top tokens"
    z = sum(top[t] for t in first)
    return {c: top[t] / z for c, t in zip(LABELS, first)}


@pytest.mark.parametrize("images", [[0], [1], [0, 1]])
def test_decision_vision_matches_completion(images):
    global server
    server.start()
    parts = image_context(images)
    ref = reference_probs(parts, images)
    body = decide({"schema": {"label": SCHEMA["label"]}, "contexts": [parts], "mode": "tree", "cache_prompt": False})
    field = body["results"][0]["fields"]["label"]
    assert field["scored_nodes"] == 1
    assert field["value"] == max(ref, key=lambda key: ref[key])
    # the tiny model amplifies kernel differences between the two paths (text-only contexts differ by ~6e-3 too)
    assert abs(field["probability"] - ref[field["value"]]) < 5e-2
    assert body["usage"]["media_chunks"] == len(images)
    assert body["timings"]["media_encode_ms"] >= 0


def test_decision_vision_batch_and_modes():
    global server
    server.start()
    contexts = [image_context([0]), image_context([1]), "A red truck on a road."]
    batch = decide({"schema": SCHEMA, "contexts": contexts})
    for i, c in enumerate(contexts):
        single = decide({"schema": SCHEMA, "contexts": [c]})["results"][0]
        for name, f in single["fields"].items():
            assert batch["results"][i]["fields"][name]["value"] == f["value"]
            # bit-identical on CPU; GPU kernels depend on the batch size and the tiny model amplifies that
            # (a mixed up context moves it by more than 0.1)
            assert abs(batch["results"][i]["fields"][name]["probability"] - f["probability"]) < 3e-2
    tree = decide({"schema": SCHEMA, "contexts": contexts[:1], "mode": "tree"})["results"][0]["decision"]
    greedy = decide({"schema": SCHEMA, "contexts": contexts[:1], "mode": "greedy"})["results"][0]["decision"]
    assert tree["animal"] == greedy["animal"]


@pytest.mark.parametrize(
    "context, message",
    [
        ([{"type": "input_audio", "input_audio": {"data": "AAAA"}}], "text\" or \"image_url"),
        ([{"type": "image_url", "image_url": {"url": "malformed"}}], ""),
        ([{"type": "image_url", "image_url": {"url": "data:text/html;base64,aGVsbG8="}}], ""),
        ([], "non-empty"),
        (42, "non-empty"),
    ]
)
def test_decision_vision_errors(context, message):
    global server
    server.start()
    res = server.make_request("POST", "/v1/decision", data={"schema": SCHEMA, "contexts": [context]})
    assert res.status_code == 400
    assert message in res.body["error"]["message"]
    # the failed request must not leave sequences behind
    decide({"schema": SCHEMA, "contexts": [image_context([0])]})


def test_decision_vision_without_mmproj():
    global server
    server.no_mmproj = True
    server.start()
    res = server.make_request("POST", "/v1/decision", data={"schema": SCHEMA, "contexts": [image_context([0])]})
    assert res.status_code == 400
    assert "mmproj" in res.body["error"]["message"]
    decide({"schema": SCHEMA, "contexts": ["A red truck on a road."]})


def test_decision_disabled():
    global server
    server.decision_seqs = None
    server.start()
    res = server.make_request("POST", "/v1/decision", data={"schema": SCHEMA, "contexts": ["x"]})
    assert res.status_code == 400
    assert "--decision-seqs" in res.body["error"]["message"]


def test_decision_kv_budget():
    global server
    server.n_ctx = 1024
    server.decision_seqs = 24
    server.start()
    long_text = "A long description of a sunny afternoon at the harbour with boats and people. " * 3
    single = decide({"schema": SCHEMA, "contexts": [long_text]})["results"][0]
    batch = decide({"schema": SCHEMA, "contexts": [long_text] * 20})
    assert batch["usage"]["context_tokens"] > 1024
    assert all(r["decision"] == single["decision"] for r in batch["results"])
    res = server.make_request("POST", "/v1/decision", data={"schema": SCHEMA, "contexts": [image_context([0, 1, 0, 1])]})
    assert res.status_code == 400
    assert "does not fit the KV cache" in res.body["error"]["message"]
    decide({"schema": SCHEMA, "contexts": [image_context([0])]})


def test_decision_calibration_and_abstain():
    global server
    server.start()
    data = {"schema": SCHEMA, "contexts": [image_context([1])], "return_probs": True}
    base = decide(data)["results"][0]["fields"]
    for f in base.values():
        probs = [c["probability"] for c in f["probs"]]
        assert abs(sum(probs) - 1.0) < 1e-5
        top2 = sorted(probs, reverse=True)[:2]
        assert abs(f["margin"] - (top2[0] - top2[1])) < 1e-6
        assert abs(f["entropy"] + sum(p * math.log(p) for p in probs if p > 0)) < 1e-5
        assert "abstain" not in f
    # temperature scales the whole-value distribution: p^(1/T), renormalised
    hot = decide(dict(data, temperature=2.0, schema=dict(SCHEMA, label=dict(SCHEMA["label"], temperature=0.5))))["results"][0]["fields"]
    for name, t in (("label", 0.5), ("animal", 2.0), ("count", 2.0)):
        p = [c["probability"] ** (1.0 / t) for c in base[name]["probs"]]
        z = sum(p)
        for c, q in zip(hot[name]["probs"], p):
            assert abs(c["probability"] - q / z) < 1e-5
    # abstain marks uncertain fields; the decision keeps a schema-valid value for every field
    body = decide({"schema": SCHEMA, "contexts": [image_context([1])], "abstain": {"min_probability": 0.99}})
    r = body["results"][0]
    assert set(r["decision"]) == set(SCHEMA)
    assert r["abstained"] == [n for n, f in r["fields"].items() if f["probability"] < 0.99]
    assert all(f["abstain"] == (f["probability"] < 0.99) for f in r["fields"].values())
    for bad in ({"temperature": 0}, {"temperature": "hot"}, {"abstain": {"min_probability": 2}}):
        res = server.make_request("POST", "/v1/decision", data=dict({"schema": SCHEMA, "contexts": ["x"]}, **bad))
        assert res.status_code == 400


@pytest.mark.parametrize("cache_mib", [256, 0])
def test_decision_media_cache_and_limit(cache_mib):
    global server
    server.decision_media_cache = cache_mib
    server.decision_max_media = 3
    server.start()
    first = decide({"schema": SCHEMA, "contexts": [image_context([0])]})
    again = decide({"schema": SCHEMA, "contexts": [image_context([0]), image_context([1]), image_context([1])]})
    assert first["usage"]["media_cached"] == 0
    # image 0 comes from the previous request, the second copy of image 1 from the first one
    assert again["usage"]["media_cached"] == (2 if cache_mib else 0)
    assert again["results"][0] == first["results"][0]
    assert again["results"][1] == again["results"][2]
    res = server.make_request("POST", "/v1/decision", data={"schema": SCHEMA, "contexts": [image_context([0, 1]), image_context([0, 1])]})
    assert res.status_code == 400
    assert "--decision-max-media" in res.body["error"]["message"]


def test_decision_playground_page():
    global server
    server.api_key = "secret"
    server.start()
    res = requests.get(f"http://{server.server_host}:{server.server_port}/decision-playground")
    assert res.status_code == 200
    assert res.headers["Content-Type"].startswith("text/html")
    assert "/v1/decision" in res.text
    res = server.make_request("POST", "/v1/decision", data={"schema": SCHEMA, "contexts": ["x"]})
    assert res.status_code == 401
    res = server.make_request("POST", "/v1/decision", data={"schema": SCHEMA, "contexts": ["x"]}, headers={"Authorization": "Bearer secret"})
    assert res.status_code == 200


def test_decision_trace():
    global server
    server.start()
    data = {"schema": SCHEMA, "contexts": [image_context([0]), "A red truck on a road."]}
    plain = decide(data)
    assert "trace" not in plain and all("trace" not in r for r in plain["results"])
    body = decide(dict(data, trace=True))
    assert body["results"][0]["decision"] == plain["results"][0]["decision"]
    tr = body["trace"]
    assert tr["prefix_tokens"] == body["usage"]["prompt_tokens"] - body["usage"]["context_tokens"]
    assert [f["name"] for f in tr["fields"]] == list(SCHEMA)
    for r in body["results"]:
        t = r["trace"]
        positions = sum(c.get("positions", c["tokens"]) for c in t["chunks"])
        assert t["position_start"] == tr["prefix_tokens"]
        assert t["position_fields"] == t["position_start"] + positions
        assert sum(c["tokens"] for c in t["chunks"]) == r["usage"]["context_tokens"]
    image = body["results"][0]["trace"]
    assert [c["type"] for c in image["chunks"]].count("image") == 1
    assert image["prompt"].count("<image>") == 1
    assert "<__media" not in json.dumps(body)


def test_decision_nullable_fields():
    global server
    server.start()
    schema = {
        "hip_cm": {"type": "integer", "minimum": 70, "maximum": 90, "nullable": True, "aggregate": "median", "description": "Hip size, null if not visible."},
        "view": {"type": "enum", "choices": ["front", "side"], "nullable": True, "description": "View, null if no person."},
    }
    for body in (decide({"schema": schema, "contexts": ["A face seen from very close.", image_context([0])], "return_probs": True}),):
        for r in body["results"]:
            for name, f in r["fields"].items():
                values = [c["value"] for c in f["probs"]]
                assert values[-1] is None
                p_null = f["probs"][-1]["probability"]
                if f["value"] is None:
                    assert p_null >= 0.5 and r["decision"][name] is None
                elif name == "hip_cm":
                    assert 70 <= f["value"] <= 90 and "interval_p10_p90" in f
    # the same field written as JSON Schema
    js = {"properties": {"hip_cm": {"type": ["integer", "null"], "minimum": 70, "maximum": 90}, "view": {"enum": ["front", "side", None]}}}
    r = decide({"schema": js, "contexts": ["A face seen from very close."], "return_probs": True})["results"][0]
    assert [c["value"] for c in r["fields"]["view"]["probs"]] == ["front", "side", None]
    assert r["fields"]["hip_cm"]["probs"][-1]["value"] is None


def test_decision_share_tokens():
    global server
    server.start()
    schema = {
        "size": {"type": "integer", "minimum": 100, "maximum": 199, "nullable": True, "description": "Size, null if not visible."},
        "label": SCHEMA["label"],
    }
    req = {"schema": schema, "contexts": ["A small cat.", image_context([1])], "mode": "tree", "return_probs": True}
    shared = decide(req)
    single = decide({**req, "share_tokens": False})
    assert shared["usage"]["scored_rows"] == single["usage"]["scored_rows"]
    assert shared["usage"]["decoded_rows"] < shared["usage"]["scored_rows"]
    assert single["usage"]["decoded_rows"] == single["usage"]["scored_rows"]
    # same branches, only the batch shape differs: probabilities agree up to backend numerics
    for a, b in zip(shared["results"], single["results"]):
        for name in schema:
            pa = [c["probability"] for c in a["fields"][name]["probs"]]
            pb = [c["probability"] for c in b["fields"][name]["probs"]]
            assert max(abs(x - y) for x, y in zip(pa, pb)) < 5e-2


def test_decision_cpu_options():
    global server
    server.start()
    size = {"type": "integer", "minimum": 100, "maximum": 199, "nullable": True, "description": "Size, null if not visible."}
    req = {"schema": {"size": size}, "contexts": ["A small cat.", image_context([1])], "mode": "tree", "return_probs": True}
    exact = decide(req)
    # pruning opens only likely subtrees, in more rounds; probabilities still sum to 1
    pruned = decide({**req, "tree_prune": 0.2})
    assert pruned["usage"]["decoded_rows"] < exact["usage"]["decoded_rows"]
    assert pruned["timings"]["rounds"] > exact["timings"]["rounds"]
    for r in pruned["results"]:
        assert abs(sum(c["probability"] for c in r["fields"]["size"]["probs"]) - 1) < 1e-4
    assert decide({**req, "tree_prune": 0.0})["usage"]["decoded_rows"] == exact["usage"]["decoded_rows"]
    # greedy rounds continue the paths kept from the round before
    greedy = decide({**req, "mode": "greedy"})
    single = decide({**req, "mode": "greedy", "share_tokens": False})
    assert greedy["usage"]["decoded_rows"] < single["usage"]["decoded_rows"]
    assert [r["decision"] for r in greedy["results"]] == [r["decision"] for r in single["results"]]
    # a coarser grid and a range in the catalogue instead of every value
    step = decide({**req, "schema": {"size": {**size, "step": 10}}, "compact_ranges": True, "trace": True})
    assert [c["value"] for c in step["results"][0]["fields"]["size"]["probs"]] == list(range(100, 200, 10)) + [None]
    assert "Allowed values: integers from 100 to 190 in steps of 10, or null" in step["trace"]["instructions"]
    assert step["usage"]["prompt_tokens"] < exact["usage"]["prompt_tokens"]
    for bad in ({"tree_prune": 1.0}, {"compact_ranges": "yes"}, {"schema": {"size": {**size, "step": 0}}}):
        res = server.make_request("POST", "/v1/decision", data={**req, **bad})
        assert res.status_code == 400


def test_decision_context_cache():
    global server
    server.decision_ctx_cache = 2
    server.start()
    req = {"schema": SCHEMA, "contexts": ["A photo of a small cat on a sofa.", image_context([0])], "mode": "tree", "return_probs": True}
    first = decide(req)
    assert first["usage"]["contexts_cached"] == 0
    again = decide(req)
    # both contexts come from the cache: nothing decoded, no image encoded, the same numbers
    assert again["usage"]["contexts_cached"] == 2
    assert [r["usage"]["context_cached"] for r in again["results"]] == [True, True]
    assert again["usage"].get("media_cached", 0) == 0 and again["timings"]["media_encode_ms"] == 0
    for a, b in zip(first["results"], again["results"]):
        for name in SCHEMA:
            pa = [c["probability"] for c in a["fields"][name]["probs"]]
            pb = [c["probability"] for c in b["fields"][name]["probs"]]
            assert max(abs(x - y) for x, y in zip(pa, pb)) < 1e-4
    # other instructions, or cache_context off: decoded again
    assert decide({**req, "instructions": "Be brief."})["usage"]["contexts_cached"] == 0
    assert decide({**req, "cache_context": False})["usage"]["contexts_cached"] == 0
    # a third context evicts the oldest of the two slots
    decide({**req, "contexts": ["A frog in a pond."]})
    assert decide({**req, "contexts": ["A frog in a pond."]})["usage"]["contexts_cached"] == 1


def test_decision_context_first_layout():
    global server
    server.decision_ctx_cache = 2
    server.start()
    ctx = ["A photo of a small cat on a sofa.", image_context([0])]
    first = decide({"schema": SCHEMA, "contexts": ctx, "layout": "context_first", "trace": True})
    assert first["usage"]["contexts_cached"] == 0
    assert first["trace"]["layout"] == "context_first" and "Fields:" in first["trace"]["context_tail"]
    assert "Fields:" not in first["trace"]["instructions"]
    # other questions about the same contexts: the contexts come from the cache
    other = {"indoor": {"type": "boolean", "description": "Is the scene indoors?"},
             "size": {"type": "enum", "choices": ["small", "medium", "large"], "description": "How big is the main object?"}}
    second = decide({"schema": other, "contexts": ctx, "layout": "context_first", "instructions": "Look closely."})
    assert second["usage"]["contexts_cached"] == 2
    assert set(second["results"][0]["decision"]) == {"indoor", "size"}
    # the same answers as without the cache
    fresh = decide({"schema": other, "contexts": ctx, "layout": "context_first", "instructions": "Look closely.", "cache_context": False})
    assert [r["decision"] for r in fresh["results"]] == [r["decision"] for r in second["results"]]
    res = server.make_request("POST", "/v1/decision", data={"schema": SCHEMA, "contexts": ctx, "layout": "sideways"})
    assert res.status_code == 400
