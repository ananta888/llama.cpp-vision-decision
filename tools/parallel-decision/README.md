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
| `integer` | `minimum`, `maximum`, optional `step` (`multipleOf` in JSON Schema) | 1-255 values; `step` counts from the minimum |
| `number` | `minimum`, `maximum`, `step` (`multipleOf` in JSON Schema) | fixed-width decimals |

Numeric fields take `aggregate`: `mode` (default), `median` or `mean`.

Any field can allow `null` as one more value, e.g. for a size the image does not show: `"nullable": true`, a
`"type": ["integer", "null"]` list, or `null` in an `enum` (JSON Schema). `null` is scored like any other value. For
numeric fields the median, mean and `interval_p10_p90` use the numbers only, and the field is `null` when `p(null)` is
0.5 or more.

Every field also takes `temperature` and `abstain` (`x-temperature` / `x-abstain` in JSON Schema), see
[Calibration and abstain](#calibration-and-abstain).

### Options

| field | default | meaning |
|---|---|---|
| `instructions` | `""` | prepended to the generated field catalogue; cached with it |
| `mode` | `auto` | `tree` scores every divergence node and returns exact probabilities; `greedy` walks the trie; `auto` picks tree up to `tree_max` values |
| `tree_max` | 128 | per-field switch between tree and greedy |
| `cache_prompt` | true | reuse the cached instructions + schema prefix |
| `share_tokens` | true | decode tokens that branches have in common (field suffix, leading digits) once; `false` gives every branch its own copy, as before |
| `tree_prune` | 0 | tree fields: open a trie node only when it is reached with at least this probability; a subtree left closed spreads its probability evenly over its values. More rounds, fewer rows |
| `compact_ranges` | false | describe numeric fields in the prompt as a range (`integers from 0 to 249, or null`) instead of listing every value |
| `temperature` | 1.0 | default temperature of every field |
| `abstain` | none | default `{"min_probability": p, "min_margin": m}` of every field |
| `return_probs` | false | list every allowed value of a tree field with its probability |
| `trace` | false | add a `trace`: the rendered prompt, the fields as scored, and per context its text and image chunks (tokens, positions, cache hits), the position the fields start at and its group |

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

A schema-valid answer is not a correct answer, and a raw `probability` is not the chance of being right (SmolVLM
answers `dark_background` at chance with 0.89 mean confidence). Two knobs turn it into a usable confidence:

- `temperature` rescales a tree field's value distribution as `p(value)^(1/T)`, renormalised (greedy fields scale each
  step).
- `abstain` marks a field as uncertain when its probability is below `min_probability` or its margin below
  `min_margin`. The field keeps its most likely value, gets `"abstain": true`, and the result lists it in `abstained`,
  so the caller can fall back (a larger model, a human, a chat completion).

Both are per field and per model, and they should come from labelled data of the real use case, not be guessed.
`bench/evaluate.py` does that:

```bash
# 1. score a labelled set once and keep the outputs
python3 bench/evaluate.py train/ --url http://127.0.0.1:8096 --dump train-rows.json
# 2. per field: fit T, pick the lowest threshold whose accepted answers reach the target accuracy, with a 95% lower
#    bound (Wilson); cross-validated. Groups of another field's predicted value that miss the target become
#    escalation rules. Writes a ready schema and the rules.
python3 bench/evaluate.py train/ --from train-rows.json --target-accuracy 0.95 --confidence-level 0.95 \
    --group-by shape,color,count --schema-out calibrated.json --rules-out rules.json
# 3. check on data the calibration has not seen
python3 bench/evaluate.py holdout/ --url http://127.0.0.1:8096 --dump holdout-rows.json
python3 bench/evaluate.py holdout/ --from holdout-rows.json --check calibrated.json --rules rules.json --target-accuracy 0.95
```

Send `calibrated.json` as the schema and escalate every field in `abstained` and every field a rule in `rules.json`
matches (a rule tests the model's answer for another field, e.g. escalate `count` when `shape` is `square`).

Result on `shapes.py` images (160 to calibrate, 160 new ones to check, 20% empty images whose answer is `unknown`,
target 95% at a 95% lower bound):

| model | field | threshold | held-out accuracy of accepted answers | held-out coverage |
|---|---|---|---|---|
| Qwen3-VL-2B Q8_0 | shape | 0.9999 | 1.000 | 1.00 |
| Qwen3-VL-2B Q8_0 | color | 0.9999 | 1.000 | 1.00 |
| Qwen3-VL-2B Q8_0 | count, no rule | 0.9212 | 0.931 | 0.82 |
| Qwen3-VL-2B Q8_0 | count, rule "shape = square" | 0.9212 | 1.000 | 0.62 |
| Qwen3-VL-2B Q8_0 | dark_background | 0.9316 | 0.991 | 0.72 |
| SmolVLM-500M Q8_0 | shape | 0.837 | 0.975 | 0.51 |
| SmolVLM-500M Q8_0 | color | 0.7733 | 1.000 | 1.00 |
| SmolVLM-500M Q8_0 | count, dark_background | none reaches the target | escalated | 0.00 |

What this shows:

- Thresholds hold on new data when the errors are uncertain ones.
- Calibration cannot catch errors the model makes with high confidence: Qwen3-VL counts three squares as two with
  0.95-0.996. Only the group report finds them (`shape=square` below the target already on the calibration set), and
  the rule fixes it at the cost of coverage.
- Small sets cannot certify high targets: 160 perfect answers give a 95% lower bound of 0.977, so a 98% target needs
  about 200 or more labelled examples per field. `evaluate.py` then refuses a threshold instead of promising too much.
- Offer an "unknown" value in enums (and `0` in counts) so the model can say that the image does not answer the
  question; it must occur in the labelled data too.

Raw reports, schemas and rules: `bench/results/calibration/`.

## Supported vision models

Checked with `bench/equivalence.py` (image decision vs `/completion` on the same prompt), `bench/evaluate.py` and the
tests. CPU build, 2026-09.

| model | positions | checked | result |
|---|---|---|---|
| Qwen3-VL-2B-Instruct Q8_0 | M-RoPE (IMROPE) | equivalence, 1 and 2 images | max abs diff 5e-10 .. 3e-7 |
| Qwen2.5-VL-3B-Instruct Q4_K_M | M-RoPE | equivalence, 1 and 2 images, negative control | 2e-5 .. 8e-4 on one image; a wrong post-image position gives 0.21 .. 0.27 |
| SmolVLM-500M-Instruct Q8_0 | normal, image slices | accuracy, batch == single | bit-identical batch vs single; equivalence tool not usable (its tokenizer merges `{\n` with the next line) |
| tinygemma3 (test model) | normal, non-causal image | tests (CI) | mechanics and errors; too small for tight numbers |

Q4_K_M on CPU is sensitive to how a prompt is split into decode calls: the plain slot path already moves candidate
probabilities by up to 0.027 when the same text prompt is decoded in two passes, so its equivalence numbers are noise,
not errors. Split-stable models (Q8_0 above) match to 1e-7. Audio and video inputs are rejected.

## Benchmarks

CPU only (Ryzen 9 7940HS, 6 threads, no GPU backend), 256x256 images from `bench/shapes.py`, 4 fields
(shape, color, count, dark background), `--decision-media-cache 0`. The machine was shared with other jobs (load
average 20-40), so absolute times are noisy; the ratios are what matters. Raw data: `bench/results/`.

| model | decision (median) | chat + json_schema (median) | field agreement | decision accuracy | chat accuracy |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B Q4_K_M | 27.9 s | 128.7 s | 0.94 | 0.94 | 1.00 |
| Qwen3-VL-2B Q8_0 | 4.4 s | 18.1 s | 1.00 | 1.00 | 1.00 |
| SmolVLM-500M Q8_0 | 2.8 s | 3.4 s | 0.78 | 0.84 | 0.94 |

Fields on one image (total / prefill / scoring):

| model | 1 fields | 2 fields | 4 fields | 8 fields | 16 fields |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B Q4_K_M | 21.5 / 18.0 / 3.5 s | 12.5 / 11.0 / 1.4 s | 12.3 / 11.5 / 0.8 s | 4.9 / 4.1 / 0.8 s | 6.4 / 4.7 / 1.7 s |
| Qwen3-VL-2B Q8_0 | 2.5 / 2.2 / 0.3 s | 2.7 / 2.4 / 0.3 s | 2.6 / 2.2 / 0.4 s | 2.9 / 2.1 / 0.9 s | 3.9 / 2.3 / 1.6 s |
| SmolVLM-500M Q8_0 | 2.2 / 2.1 / 0.0 s | 2.2 / 2.1 / 0.0 s | 2.2 / 2.1 / 0.1 s | 2.2 / 2.1 / 0.2 s | 2.7 / 2.4 / 0.3 s |

N images: one request with N contexts vs N requests:

| model | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Qwen2.5-VL-3B Q4_K_M | 6.6 vs 5.0 | 9.0 vs 8.8 | 19.3 vs 18.3 | 88.3 vs 120.0 |
| Qwen3-VL-2B Q8_0 | 4.4 vs 3.2 | 5.5 vs 26.0 | 103.4 vs 116.9 | 232.2 vs 206.4 |
| SmolVLM-500M Q8_0 | 2.5 vs 2.3 | 4.4 vs 5.4 | 12.1 vs 11.8 | 21.6 vs 23.3 |

| model | field | accuracy | mean confidence | ECE | fitted T | ECE at T |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B Q4_K_M | shape | 1.00 | 1.00 | 0.002 | 0.05 | 0.000 |
| Qwen2.5-VL-3B Q4_K_M | color | 1.00 | 0.99 | 0.007 | 0.05 | 0.000 |
| Qwen2.5-VL-3B Q4_K_M | count | 1.00 | 0.99 | 0.009 | 0.05 | 0.000 |
| Qwen2.5-VL-3B Q4_K_M | dark_background | 0.61 | 0.80 | 0.253 | 1.457 | 0.277 |
| Qwen3-VL-2B Q8_0 | shape | 1.00 | 1.00 | 0.000 | 0.05 | 0.000 |
| Qwen3-VL-2B Q8_0 | color | 1.00 | 1.00 | 0.000 | 0.05 | 0.000 |
| Qwen3-VL-2B Q8_0 | count | 0.94 | 0.99 | 0.055 | 2.029 | 0.058 |
| Qwen3-VL-2B Q8_0 | dark_background | 0.94 | 0.99 | 0.049 | 1.745 | 0.040 |
| SmolVLM-500M Q8_0 | shape | 0.86 | 0.87 | 0.082 | 1.078 | 0.060 |
| SmolVLM-500M Q8_0 | color | 1.00 | 0.99 | 0.005 | 0.05 | 0.000 |
| SmolVLM-500M Q8_0 | count | 0.94 | 0.96 | 0.033 | 1.078 | 0.040 |
| SmolVLM-500M Q8_0 | dark_background | 0.47 | 0.89 | 0.416 | 20.0 | 0.059 |

GPU (RTX 5060 Ti 16 GB via WSL2, CUDA 12.9, `-ngl 99 -fa on`), Qwen3-VL-2B Q8_0, same images and fields: decision
95 ms vs chat + `json_schema` 357 ms (median per image), all 4 fields right in both; 1 / 16 fields on one image
99 / 112 ms; 8 images in one request 0.57 s; a 4000x3000 photo capped at 1024 image tokens 0.74 s cold, 0.39 s with its
embedding cached. Raw data: `bench/results/bench-qwen3-gpu.json`. On a GPU, batched and single requests can differ in the
last digits (kernels depend on the batch size): up to 4e-6 with Qwen3-VL; decisions are the same.

- A decision is 4x faster than a chat completion with a `json_schema` response on the Qwen models, and extra fields
  are nearly free after the image prefill (Qwen3-VL: 1 field 2.5 s, 16 fields 3.9 s).
- Schema-valid is not correct: SmolVLM answers `dark_background` at chance with 0.89 mean confidence. `evaluate.py`
  shows it (ECE 0.42), and a fitted temperature or an abstain rule catches it.
- A warm prefix and image cache halve a repeated request (Qwen3-VL, 2 images + 1 text: 6.8 s cold, 3.3 s warm).

Branches share their common tokens (`share_tokens`, on by default): all branches of a field start with the field
suffix, and the branches of a number share its leading digits. The branches of one batch form a trie, each trie token
is decoded once and belongs to the sequences of all branches below it. `usage.scored_rows` counts the rows of the
branches, `usage.decoded_rows` the rows decoded.

`bench/bench_share.py`, Qwen3-VL-2B Q8_0, 8 images of `bench/shapes.py`, `--decision-media-cache 0`, median per image
(total / scoring). "numeric" asks 4 nullable pixel values (0-249) and a percentage, in tree mode:

| backend | schema | shared tokens | own copy per branch | chat + json_schema | rows scored / decoded |
|---|---|---|---|---|---|
| CPU, 128 seqs | shapes (4 fields) | 1.14 s / 0.13 s | 1.19 s / 0.16 s | 2.76 s | 20 / 14 |
| CPU, 128 seqs | numeric (5 fields) | 4.27 s / 1.98 s | 12.41 s / 10.14 s | 3.33 s | 801 / 125 |
| GPU, 16 seqs | numeric (5 fields) | 302 ms / 232 ms | 347 ms / 275 ms | 482 ms | 801 / 173 |
| GPU, 128 seqs | shapes (4 fields) | 77 ms / 19 ms | 80 ms / 20 ms | 313 ms | 20 / 14 |
| GPU, 128 seqs | numeric (5 fields) | 165 ms / 92 ms | 240 ms / 168 ms | 490 ms | 801 / 125 |

- Sharing pays off for numbers: scoring is 5x faster on the CPU and 1.8x on the GPU.
- In tree mode every trie node of a number is one output, and one decode takes as many outputs as there are decision
  sequences. Wide numeric fields want many sequences (128 costs little on a plain-attention model); with 16 the numeric
  schema takes several rounds.
- Shared and own-copy scoring decode the same tokens in a different batch shape. On the CPU backend the shared logits
  equal decoding every branch on its own sequence exactly; on CUDA both layouts differ from that by up to about 0.5 in
  the logits (kernels depend on the batch shape), probabilities by up to 0.04, and near-ties can flip (95-100% of the
  decisions were the same).

### Numbers on a CPU

On a CPU every decoded row costs, and every output row most (the model computes logits over the whole vocabulary),
while a GPU decodes a few hundred rows about as fast as one. Wide numeric fields therefore have their own switches:

- `compact_ranges`: by default the prompt lists every allowed value, 250 per field for 0-249. Five such fields make a
  5000 token prefix that every branch row attends to. As a range the prefix has 274 tokens.
- `step` on integer fields: fewer values, fewer trie nodes with a choice, fewer outputs.
- `tree_prune` and `mode: greedy`: fewer rows in more rounds. A round continues the paths kept from the round before,
  so it decodes only the new digits. Worth it once the prefix is short; with a 5000 token prefix every round is slow.

Qwen3-VL-2B Q8_0 on the CPU, 12 images with one rectangle of known position and size, 4 nullable pixel fields
(0-249), median per image, mean absolute error over the 48 values:

| variant | time | error |
|---|---|---|
| every value listed, tree (exact) | 4.28 s | 48.8 px |
| `compact_ranges`, tree (exact) | 1.58 s | 46.4 px |
| `compact_ranges`, `tree_prune` 0.05 | 1.01 s | 46.4 px |
| `compact_ranges`, greedy | 0.99 s | 46.4 px |
| `compact_ranges`, tree, `aggregate: median` | 1.65 s | 43.8 px |
| chat with `json_schema` | 6.52 s | 60.6 px |

A cold request (prefix not cached) drops from 43.5 s to 2.9 s with `compact_ranges`. The 2B model is not good at
pixel geometry, but the shorter prompt does not make it worse. The playground has all of them ("Prune below p",
"Number step", "compact ranges").

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

`llama-server` serves a web page for image decisions at `/decision-playground` (source: `playground/index.html`,
built into the server). Drop, paste or pick images (scaled down in the browser, 1024 px by default: a 12 MP photo
would cost thousands of image tokens and minutes on a CPU), choose or edit a schema, run it as a decision, as a chat completion with a `json_schema` response, or both side by side (per-field agreement and time; the chat trace lists every generated token with its probability and alternatives), and get the decision with every value's
probability, margin and abstain marks, a time bar (image encode, prefill, field scoring), and the trace: what happened
step by step, the prompt as the model sees it, the chunks and positions, the fields as scored, and the raw request and
response. The "Body photo check" mode fixes a schema that decides whether a photo is fit for measuring body sizes (one person, head
to feet, front or side view, standing straight, sharp, not distorted) and shows one verdict (yes / no / unsure) with a
checklist, plus size estimates in cm (height, shoulder width, chest, waist, hip, inseam, arm length) as nullable fields
with their p10-p90 range, "none" where the photo does not show them. These are model estimates, not measurements (one
photo has no scale); calibrate them on measured people before relying on them. The page offers two corrections: scale all
sizes by a known body height, or a calibration from your own measurements (enter the tape-measured sizes of a result and
add it as a sample; per size an offset from 2 samples or a straight line from 5, used only if a leave-one-out check beats
the raw estimate, with an 80% range from the leave-one-out errors). Samples keep the model output and your numbers,
never the photo, and export / import as JSON. `bench/calibrate_sizes.py` does the same for a folder of photos with a
`labels.jsonl` of measured sizes and writes a file the page imports. With the webcam as source, the
page takes a frame every 1-30 s (or on demand), scales it like an upload and decides it; frames that would overlap a
running request are skipped and counted, the tab pauses when hidden, and live frames are logged but not kept in the
history. Browsers allow the camera only on localhost / 127.0.0.1 or over https. A trace
downloads as JSON; the last runs stay in the browser's history. The page is public like the web UI;
with `--api-key`, enter the key in the page (it is not stored), since `/v1/decision` still checks it.
