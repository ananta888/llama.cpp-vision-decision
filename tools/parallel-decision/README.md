# parallel-decision

Answer a finite JSON schema in one batched forward pass, instead of generating the JSON token by token.

Every field of a schema has a fixed set of allowed values (enums, booleans, bounded integers, numbers on a grid).
After the context, each field's allowed values are scored as token paths that fork from the same KV cache, so all
fields are answered in one `llama_decode` and cannot see each other. Each answer comes back with a probability, and
the JSON object is assembled by code, so it always matches the schema.

This directory holds the engine (`decision-engine.*`), a CLI (`llama-parallel-decision`), and the engine is also
served by `llama-server` as `POST /v1/decision`.

## Build

Same as llama.cpp:

```bash
cmake -B build -DGGML_CUDA=ON        # or plain `cmake -B build` for CPU / Metal
cmake --build build --config Release -j
```

## Run the server

`--decision-seqs N` reserves the sequence slots the decisions need: one holds the cached instructions, one per context
in flight, the rest are the parallel questions. It also switches the KV cache to unified, which is what lets the
branches share the context's cells.

```bash
./build/bin/llama-server -m model.gguf -ngl 99 -fa on -c 32768 --decision-seqs 24 --port 8096
```

With a presets file, one loaded model serves chat and decisions:

```ini
[*]
ngl = 99
fa = on
jinja = 1
parallel = 1
cache-type-k = q8_0
cache-type-v = q8_0
decision-seqs = 24

[gemma-4-12b]
model = ./models/gemma-4-12b-it-UD-Q4_K_XL.gguf
ctx-size = 32768
ubatch-size = 512
decision-seqs = 12
```

```bash
./build/bin/llama-server --models-preset models.ini --models-max 1 --port 8096
```

How many sequences a model affords depends on its attention. A plain-attention model shares the context's cells, so
128 sequences cost almost nothing. A sliding-window model (Gemma) allocates its window per sequence, so keep it low
(12 on a 12 GB card). Hybrid models with recurrent layers work, but llama.cpp splits their batches per sequence
length, so branches run in several passes instead of one.

## POST /v1/decision

`contexts` is a list of 1-256 strings. They share one schema, one set of instructions, and one cached prefix; results
come back in the same order.

```bash
curl http://localhost:8096/v1/decision -H "Content-Type: application/json" -d '{
  "model": "gemma-4-12b",
  "instructions": "Answer each question about this support request from its state.",
  "schema": {
    "category": {"type": "enum", "choices": ["billing","technical","cancellation","other"],
                 "description": "What type of support request is this?"},
    "urgent":   {"type": "boolean", "description": "Does this need urgent handling?"},
    "priority": {"type": "enum", "choices": ["low","medium","high","critical"],
                 "description": "Rate support priority."}
  },
  "contexts": ["I was charged twice and need this fixed today."]
}'
```

```json
{
  "object": "decision",
  "results": [
    {
      "decision": {"category": "billing", "urgent": true, "priority": "high"},
      "fields": {
        "category": {"value": "billing",  "probability": 1.0,  "scored_nodes": 1, "tree": true},
        "urgent":   {"value": true,       "probability": 1.0,  "scored_nodes": 1, "tree": true},
        "priority": {"value": "high",     "probability": 0.74, "scored_nodes": 1, "tree": true}
      },
      "usage": {"context_tokens": 21, "scored_rows": 14}
    }
  ],
  "usage": {"prompt_tokens": 137, "cached_tokens": 116, "context_tokens": 21, "scored_rows": 14},
  "timings": {"prefill_ms": 50.7, "scoring_ms": 50.0, "total_ms": 100.7, "rounds": 1, "per_decision_ms": 100.7}
}
```

(That response is a real one: Gemma 4 12B on an RTX 3060, warm cache.)

### Schema

Compact fields, or a JSON Schema object with `properties`:

| type | keys | notes |
|---|---|---|
| `enum` | `choices` (or `enum`) | 1-255 values |
| `boolean` | - | true / false |
| `integer` | `minimum`, `maximum` | 1-255 values |
| `number` | `minimum`, `maximum`, `step` (`multipleOf` in JSON Schema) | fixed-width decimals |

Numeric fields take `aggregate`: `mode` (default), `median` or `mean`.

Every field also takes `temperature` and `abstain` (`x-temperature` / `x-abstain` in JSON Schema), see
[Calibration and abstain](#calibration-and-abstain).

### Options

| field | default | meaning |
|---|---|---|
| `instructions` | `""` | prepended to the generated field catalogue; cached with it |
| `mode` | `auto` | `tree` scores every divergence node and returns exact probabilities; `greedy` walks the trie; `auto` picks tree up to `tree_max` values |
| `tree_max` | 128 | per-field switch between tree and greedy |
| `cache_prompt` | true | reuse the cached instructions + schema prefix |
| `temperature` | 1.0 | default temperature of every field |
| `abstain` | none | default `{"min_probability": p, "min_margin": m}` of every field |
| `return_probs` | false | list every allowed value of a tree field with its probability |

Tree fields also return `margin` (top-1 minus top-2 probability) and `entropy` (nats) of their value distribution.

## Images

With a vision model (`--mmproj`), an entry of `contexts` can be an array of OpenAI-style content parts instead of a
string: `{"type": "text", "text": ...}` and `{"type": "image_url", "image_url": {"url": ...}}`. The image goes through
the same libmtmd path as chat images and is prefilled into the context's KV sequence; every field is then scored from
that one multimodal context. No caption is generated in between.

```bash
IMG=$(base64 -w0 photo.png)
curl http://localhost:8096/v1/decision -H "Content-Type: application/json" -d '{
  "schema": {
    "shape": {"type": "enum", "choices": ["circle","square","triangle"], "description": "Which shape is drawn?"},
    "count": {"type": "integer", "minimum": 1, "maximum": 4, "description": "How many shapes are there?"},
    "dark_background": {"type": "boolean", "description": "Is the background dark?"}
  },
  "contexts": [[
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'$IMG'"}},
    {"type": "text", "text": "Answer the questions about this picture."}
  ]]
}'
```

- Image urls are loaded like chat images: `data:` URIs, raw base64, `http(s)://` (10 MB, 10 s) and `file://` only
  below `--media-path`.
- Several images per context and several image contexts per request work. Text and image contexts can be mixed.
- `usage` adds `media_chunks`, `media_tokens` and `media_cached`; `timings` adds `media_encode_ms` (part of
  `prefill_ms`).
- Positions follow the model: Qwen-VL style models (M-RoPE) give an image fewer positions than tokens, and the fields
  are scored after the image's real position.
- Images are encoded in the request's own mtmd batch, several at a time when the projector supports it.
- Models that attend to image tokens non-causally (Gemma 3, larger Gemma 4, DeepSeek 4 V) need the whole image in one
  ubatch (`-ub`).
- Each group of contexts in flight holds at most one image context, decoded right after the cached prefix: image
  embeddings decoded behind other sequences' KV cells give slightly different numbers (reproducible on the plain slot
  path), while text does not. So a decision does not depend on the other contexts of its request; text contexts are
  still batched around it.
- The runner only chooses among the finite allowed values. It does not read free text out of an image (OCR); use a
  chat completion for that.

Server options for images:

| option | default | meaning |
|---|---|---|
| `--decision-max-media N` | 16 | most images in one request; checked before any image is loaded |
| `--decision-media-cache N` | 256 | MiB of encoded images kept across requests (key: image hash and slice); 0 = off |

A context that cannot fit the KV cache next to the cached instructions is rejected with HTTP 400 before any decode.
Contexts in flight are grouped so that they fit together.

## Calibration and abstain

A schema-valid answer is not a correct answer. Every value comes with a probability, and two knobs help to act on it:

- `temperature` rescales a tree field's value distribution as `p(value)^(1/T)`, renormalised (greedy fields scale each
  step). Fit it on labelled data with `bench/evaluate.py`, which reports accuracy, NLL and ECE per field and the
  temperature that minimises NLL.
- `abstain` marks a field as uncertain when its probability is below `min_probability` or its margin below
  `min_margin`. The field keeps its most likely value, gets `"abstain": true`, and the result lists it in `abstained`,
  so the caller can fall back (ask a larger model, a human, or a chat completion).

```json
"abstain": {"min_probability": 0.8, "min_margin": 0.3}
```

## Checking a model

`bench/equivalence.py` compares the probabilities of an image decision with the next-token distribution of
`/completion` on the same prompt (the slot path). A mismatch points at wrong positions or KV contents. Run the server
with `LLAMA_MEDIA_MARKER="<__media__>"`:

```bash
python3 bench/shapes.py /tmp/shapes -n 64
python3 bench/equivalence.py --url http://127.0.0.1:8096 --image /tmp/shapes/0003.png \
    --choices red,green,blue,yellow --description "What colour are the shapes?"
```

`bench/evaluate.py DIR` measures accuracy and calibration on the images of `shapes.py`, and `bench/bench.py DIR`
measures latency.

## CLI

`llama-parallel-decision` runs the same engine from a worker process (stdin/stdout protocol, one JSON request per
line). Environment: `DECIDE_TREE`, `DECIDE_TREE_MAX`, `DECIDE_NSEQ`, `DECIDE_SPLIT_BOUNDARY`.

## A UI for it

[decision-playground](https://github.com/thecodacus/decision-playground) is a browser-only playground: it talks
straight to your llama-server, runs a decision and the same question as a chat completion side by side with live
timers, and has a small game whose agents decide through the endpoint.

`playground/index.html` is a single static page for image decisions: pick images, edit the schema, and see every
value's probability, margin and the timings. Open it in a browser and point it at the server (CORS is open by default).
