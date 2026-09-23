# AGENT.md - vision-decision fork

Working notes for agents and humans on `ananta888/llama.cpp-vision-decision`. Read [AGENTS.md](AGENTS.md) first: its rules apply here too (ASCII only, short comments, no push / PR / GitHub comments by agents, `Assisted-by:` in commits).

The binding task list is [todos/todo.vision-decision.json](todos/todo.vision-decision.json) (Ananta task-track format). Keep it current: a task is `done` only with implementation, tests and reproducible evidence.

## Goal

Extend the `/v1/decision` runner so a vision-language model answers several typed schema fields about an image directly, from one multimodal KV context, without generating an image description first.

```
image/text -> libmtmd (mmproj encoder, existing) -> multimodal prefill into a trunk sequence
           -> parallel decision engine (existing) -> typed schema decision + probability + timings
```

Rules for the design:

- The decision engine gets no vision code. It learns to score on a trunk that someone else prefilled (the server, with libmtmd).
- No image caption as an intermediate step.
- Reuse existing llama.cpp paths: `handle_media`, `mtmd_tokenize`, `mtmd_batch_*`, `mtmd_helper_decode_image_chunk`, `llama_memory_seq_cp`.
- Text-only `/v1/decision` and normal multimodal chat in `llama-server` must not regress.
- The runner scores finite typed candidate sets only (enum, boolean, bounded integer, number grid). Free OCR / string extraction is out of scope.
- A schema-valid answer is not a correct answer. Probabilities and abstain signals exist so callers can tell the difference.

## Upstream relationship

```
ggml-org/llama.cpp (master)           remote: upstream-main       (fetch only)
  -> thecodacus/llama.cpp (parallel-decision)   remote: upstream-decision (fetch only)
    -> ananta888/llama.cpp-vision-decision (vision-decision)   remote: origin
      -> ananta888/ananta vendor/llama.cpp-vision-decision (git submodule)
```

- Pull: `git fetch upstream-decision && git merge upstream-decision/parallel-decision` (and `upstream-main/master` when needed).
- Give back: only through a human-written PR from `vision-decision`. Upstream llama.cpp requires an issue discussion first for large changes; agents never open PRs or write PR text.
- Keep the diff against `parallel-decision` small and local: new code lives in `tools/parallel-decision/` and in the decision parts of `tools/server/`.

## Verified facts (checked in this tree, not assumed)

| topic | fact | where |
|---|---|---|
| engine sequences | `seq_snap` holds the cached prefix, one trunk per context in flight, the rest are branches forked with `llama_memory_seq_cp` | `tools/parallel-decision/decision-engine.cpp` |
| engine positions (before this work) | branch `pos0 = shared.size() + prefix.size()`: positions equal token counts | `decision-engine.cpp`, `decide_batch` |
| server thread | `handle_decision` runs synchronously on the server loop with `ctx_tgt`; `mctx` is available there | `tools/server/server-context.cpp` |
| unified KV | `seq_cp` inside one stream only adds the seq id to the cells: no copy, M-RoPE extra positions are shared | `src/llama-kv-cache.cpp`, `seq_cp` |
| M-RoPE text | text tokens with a 1D `pos` are broadcast to all rope sections | `src/llama-batch.cpp`, `ubatch_add` |
| M-RoPE check | text tokens of a sequence must start at a position greater than the sequence's max position | `src/llama-batch.cpp`, consistency checks |
| Qwen2/2.5/3-VL image positions | `t = pos_0`, `x = pos_0 + col`, `y = pos_0 + row`; the image uses `max(nx, ny)` positions, not `n_tokens` | `tools/mtmd/mtmd.cpp`, `mtmd_image_tokens_get_decoder_pos`, `_get_n_pos` |
| position type | chosen from the text model's rope type: NORM/NEOX -> normal, MROPE/IMROPE -> M-RoPE, HunyuanVL own layout | `tools/mtmd/mtmd.cpp` |
| non-causal images | Gemma 3, Gemma 4 (larger), DeepSeek4V decode image tokens non-causally; the helper toggles `llama_set_causal_attn` | `mtmd_decode_use_non_causal`, `mtmd-helper.cpp` |
| image wrap tokens | `mtmd_tokenize` adds model-specific begin/end tokens (e.g. `<|vision_start|>`/`<|vision_end|>`); the chat template only sees the media marker | `tools/mtmd/mtmd.cpp` |
| media marker | random per process unless `LLAMA_MEDIA_MARKER` is set, so user text cannot inject media | `tools/server/server-common.cpp`, `get_media_marker` |
| media loading | `handle_media`: http(s) (10 MB, 10 s), `file://` only under `--media-path`, `data:` URIs, raw base64 | `server-common.cpp` |
| server image batching | `mtmd_batch_add_chunk` / `mtmd_batch_encode` encode several images at once, then `mtmd_helper_decode_image_chunk` decodes each | `server-context.cpp`, `process_mtmd_chunk` |

## Architecture of the change

1. Engine (`decision-engine.*`): a context is either text (tokenized and packed by the engine, as before) or a caller prefill that fills a given trunk sequence from a given start position and returns the next position. The engine keeps `pos_next` per trunk and forks branches from there. No libmtmd include in the engine.
2. Server (`handle_decision`): `contexts[i]` may be a string (unchanged) or an OpenAI-style content array with `text` and `image_url` parts. Media go through `handle_media`, the rendered tail through `mtmd_tokenize`, and the prefill uses libmtmd to encode and decode media chunks on the trunk sequence.
3. Everything else (schema compiler, tree / greedy scoring, prefix cache, assembly) is shared by text and vision.

## Development rules

- Read the code before changing it; write facts into the table above with a file reference.
- Small commits, one topic each, subject in llama.cpp style (`decision : ...`), body short, `Assisted-by: Claude Opus 5.5`.
- No new files in the top-level `tests/` directory. Server tests go to `tools/server/tests/unit/`.
- Code: ASCII only, comments 1-2 lines, blend in with surrounding code.
- Blockers are written down with the concrete code / runtime / model reason.

## Tests

- Build: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-server llama-parallel-decision`
- Server tests: `cd tools/server/tests && LLAMA_SERVER_BIN_PATH=../../../build/bin/llama-server python3 -m pytest unit/test_decision.py -v`
- Regression: `unit/test_vision_api.py`, `unit/test_basic.py`, and the text-only decision tests must stay green.
- Equivalence check for positions and KV: for a field whose candidates differ at the first token, the tree probability must match the next-token distribution of `/completion` on the same multimodal prompt (renormalized over the candidates).

## Benchmarks

`tools/parallel-decision/bench/` holds the scripts; results with hardware, model, quant and command go to `tools/parallel-decision/README.md`. Measure at least:

- vision decision vs. chat completion with `json_schema` on the same image and schema (latency, agreement);
- scaling with the number of fields (1 .. 16) on one image;
- multi-context batches (N images, one schema);
- warm vs. cold prefix cache.

## Definition of Done

A supported open-weight VLM served by `llama-server` takes an image directly, and `/v1/decision` scores several independent typed fields from the same multimodal KV context, with probabilities and timings, clean batching and caching, regression tests, benchmarks and full documentation.

## Environment notes

- Dev box: RTX 5060 Ti 16 GB, but no CUDA toolkit (`nvcc`) and no Vulkan SDK, and no passwordless sudo. Builds are CPU-only here; GPU numbers need a CUDA toolkit install.
