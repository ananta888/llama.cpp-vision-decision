import base64
import json
import math

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
    if url not in _images:
        res = requests.get(url)
        res.raise_for_status()
        _images[url] = base64.b64encode(res.content).decode("utf-8")
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
    assert field["value"] == max(ref, key=ref.get)
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
            assert abs(batch["results"][i]["fields"][name]["probability"] - f["probability"]) < 1e-5
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
