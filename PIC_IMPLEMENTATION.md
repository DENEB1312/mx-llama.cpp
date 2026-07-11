# Position-Independent Context Caching (PIC) — Implementation Status

This documents the in-progress port of **EPIC** (Efficient Position-Independent Context
Caching, arXiv:2410.15332) into llama.cpp. It summarizes what is implemented, how it was
verified, and the precise next steps.

## Concept recap

Prefix-based context caching requires an exact token-prefix match. **PIC** instead lets you
precompute KV caches for *chunks* (documents, system prompts, few-shot examples) once and
reuse them in **any order/position** inside later prompts. The accuracy cost comes from the
*attention sink*: each independently-prefilled chunk's first token hogs attention. **LegoLink**
fixes this by recomputing only the first `k` (≤32) tokens of every non-first chunk — reducing
linking cost from O(N²) (CacheBlend) to O(kN). **LegoLink-0** avoids even that by prepending
dummy BOS tokens during chunk prefill (zero linking overhead, approximate).

## What is implemented

### 1. Relocatable KV primitive (`src/llama-kv-cache.cpp` / `.h`)
The enabler for all PIC variants: extract/insert raw KV cache data for an arbitrary cell range.
- `kv_data_size(n_cells)`, `kv_data_get(strm, i0, i1, data)`, `kv_data_set(strm, i0, i1, data)`,
  `set_cell_range(strm, i0, i1, pos, seq)`.
- Blob layout mirrors `state_write_data`/`state_read_data` (per-layer K then V, self-describing
  `llama_pic_v_hdr` for the V tensor so transposed/non-transposed layouts are handled uniformly).

### 2. Public C API (`include/llama.h`)
- `llama_pic_kv_data_size`, `llama_pic_kv_data_get`, `llama_pic_kv_data_set`.

### 3. PIC engine (`src/llama-pic.cpp` / `.h`)
- `llama_pic_encode_chunk(ctx, tokens, dummy_prepend)` — KVGen: prefill + extract a chunk's KV
  (with optional LegoLink-0 dummy prefix).
- `llama_pic_assemble(ctx, chunks, query, k)` — KVLink: inject chunk 0 as-is; for later chunks
  recompute the first `k` link tokens then inject the warm remainder; decode the query last.
  `k == 0` selects LegoLink-0.
- `llama_pic_encode_chunk_state(ctx, tokens)` / `llama_pic_assemble_state(ctx, chunk, query, seq)`
  — **prefix-cache mode** for hybrid/recurrent models: snapshot/restore the full per-sequence
  state (attention KV + recurrent state) via `llama_state_seq_get_data`/`set_data`. The chunk
  carries `seq_state = true`; it is reusable only as a position-0 prefix.
- `llama_pic_assemble_state_chain(ctx, store, chain, query, seq)` — **chained multi-prefix** for
  hybrid models: composes an ordered list of cached prefixes by restoring the longest cached
  combined snapshot, decoding the novel suffix, and caching new combined snapshots in `store`.
- `llama_kv_store` — content-addressed KV store: `cache_id = FNV-1a(model_id + tokens)`,
  `put`/`put_state`/`get`/`has`, optional disk persistence (`*.pic` files carry a `seq_state`
  flag byte so both chunk kinds survive restart).

### 4. Server endpoint (`tools/server/`)
- `POST /v1/context_cache` — accepts `{"chunks": [...], "add_bos": bool}`, encodes each chunk
  via the store, returns `{"cache_ids": [...]}`. Registered in `server.cpp`; handler in
  `server-context.cpp`.
- **Request-time reuse (Next step 1 — done).** The completion and chat request schemas accept
  `cache_ids` (array of strings) and `pic_k` (LegoLink link-token count; `0` = LegoLink-0).
  In `update_slots`, a slot whose task carries `cache_ids` resolves the chunks from the shared,
  per-model `llama_kv_store`, calls `llama_pic_assemble(ctx, chunks, query, k, slot.id)` to inject
  the precomputed KV directly into the slot's sequence, then jumps to `SLOT_STATE_DONE_PROMPT` so
  the normal prompt prefill is skipped and generation continues from the assembled state. Unknown
  `cache_id`s are rejected with an error. The store is now a single per-model map shared between the
  encode endpoint and the generation path (previously a local static inside the endpoint).
- Integration test: `tools/server/tests/unit/test_pic.py`.

### Architecture support: two modes
PIC now supports two families of models via two different mechanisms, selected automatically:

1. **Relocatable KV (full EPIC/LegoLink)** — standard transformer models whose memory is a
   `llama_kv_cache` (`dynamic_cast` in `src/llama-context.cpp:4001`), including `--kv-unified`.
   Chunks are position-independent: any number can be composed in any order at assemble time.
   Detected by `llama_pic_supported(ctx) == true`.

2. **Prefix-cache / chained multi-prefix (seq-state snapshot)** — hybrid/recurrent models
   (e.g. **Qwen3.6**, `qwen35`/`qwen35moe`/`qwen3next` → `llama_memory_hybrid`). The KV primitive
   cannot relocate per-token state because recurrent layers (gated delta net, `ggml_gated_delta_net`)
   hold a single *cumulative, order-dependent* state. Instead the whole per-sequence state
   (attention KV **and** recurrent state) is snapshotted with `llama_state_seq_get_data`/`set_data`.
   Multiple `cache_ids` are supported as an **ordered chain**: `llama_pic_assemble_state_chain`
   reuses the longest already-cached *combined* prefix snapshot, decodes only the novel suffix
   chunks on top, and caches the new combined snapshots for future reuse (RadixAttention-style).
   Constraints: **fixed order** (recurrent state is non-commutative — no reordering, no independent
   composition) and a **non-empty** dynamic prompt after the prefix. The first use of a novel
   combination pays a one-time suffix-decode cost; identical combinations thereafter are a pure
   state restore.

   True *position-independence* (reorder chunks → same answer) is **mathematically impossible** for
   gated-delta-net layers, so it is intentionally not attempted.

DeepSeek-V3 (`llama_kv_cache_dsa`) uses neither path and is still unsupported (graceful 501/400).

### Error-handling hardening ("error-free" pass)
PIC must never crash the server on unsupported input. The following guards were added and
verified live (see "Negative-path verification"):
- **Unsupported architecture.** New public API `llama_pic_supported(ctx)`
  (`include/llama.h`, `src/llama-context.cpp`) returns whether the context's memory backend
  is a relocatable `llama_kv_cache` (nullptr-safe). Both the encode endpoint
  (`post_context_cache`) and the request-time assemble path in `update_slots` check it and
  return `ERROR_TYPE_NOT_SUPPORTED` (HTTP 501) instead of asserting. Unified KV
  (`--kv-unified`) is still a `llama_kv_cache` subclass, so PIC works there normally.
- **Oversized chunk.** `post_context_cache` rejects any chunk whose token count exceeds the
  encode context size (`ERROR_TYPE_INVALID_REQUEST`, HTTP 400) rather than tripping the
  `n_tokens_all <= n_batch` assert.
- **Thread-safety.** `llama_kv_store` gained an internal `std::mutex` guarding `put`/`has`/
  `get`; the shared per-model store map is guarded by `pic_stores_mutex()` in
  `server-context.cpp`. Concurrent encode + completion requests no longer race.
- **Slot reuse.** `slot.release()` → `reset()` clears the `pic_assembled` flag, so a slot
  reused for a normal (non-PIC) request prefills correctly.

### 5. Tests
- `tests/test-pic.cpp` — primitive round-trip (bit-exact), inject+decode control, LegoLink (k=4).
- `tests/test-kv-store.cpp` — store encode/assemble end-to-end: LegoLink (k=4) and LegoLink-0
  (dummy=4).

## Verification

- All tests build (`make test-pic test-kv-store`) and pass on TinyLlama-1.1B-Q4_K_M.
  - Primitive re-inject: **cosine 1.000000** (bit-exact).
  - Store + LegoLink k=4: cosine ≈ 0.83 (expected PIC approximation; chunk-2 warm tokens keep
    chunk-local RoPE).
  - Store + LegoLink-0 (dummy=4): cosine ≈ 0.83.
- `llama-server` builds and the `/v1/context_cache` endpoint was smoke-tested live:
  `{"cache_ids":["a7d56168b5d385f1","1d0ee5eb46ce7b4e","714066797003c72b"]}`.

### End-to-end request-time reuse (verified on gfx906, 2× MI60)
Ran `llama-server` with `Qwen3-4B-Instruct-2507-Q4_0.gguf` (2 GPUs, tensor split, flash-attn,
f16 KV). Confirmed:
- `POST /v1/context_cache` encodes chunks and returns `cache_ids`.
- `/completion` and `/v1/chat/completions` with `cache_ids` reuse the chunk KV at request time;
  the model conditions correctly on the injected chunks (incl. reordered assembly).
- `pic_k = 0` (LegoLink-0) and `pic_k = 4` (LegoLink) both work; unknown `cache_id` → HTTP 400.

### Negative-path verification (gfx906, crash-free)
Confirmed the hardening keeps the server alive (health `ok`) after each error:
- **Oversized chunk** (~23k tokens, `n_ctx = 9000`): → HTTP 400 `invalid_request_error`,
  server survives.
- **`--kv-unified`** (`Qwen3-4B-Instruct-2507-Q4_0`): encode + `/completion` with `cache_ids`
  → HTTP 200 with a coherent completion (unified KV is supported).

### Hybrid model (Qwen3.6) prefix-cache verification (gfx906, 2× MI60)
`Qwen3.6-27B-Q8_0` (`llama_memory_hybrid`):
- `POST /v1/context_cache` with a ~70-token system/context prefix → HTTP 200, returns a
  `cache_id` (prefix-cache/seq-state mode chosen automatically).
- `/completion` with that `cache_id` + a dynamic question → HTTP 200, and the answer is
  **byte-identical** to the full-prefill baseline (no cache): both produce
  *"The Strait of Hormuz connects the **Persian Gulf** and the **Gulf of Oman**."* — proving
  the restored attention KV + recurrent state exactly reproduces a fresh prefill.
- Negative path (server stays alive): empty prompt + `cache_id` → HTTP 400 (prefix-cache needs a
  non-empty prompt).
- Regression: standard `Qwen3-4B-Instruct-2507-Q4_0` still does full multi-chunk relocatable
  assembly (2 chunks, reordered) correctly.

### Chained multi-prefix verification (Qwen3.6-27B, 3 chunks A/B/C, ~1123-token prefix)
- Chain output is **byte-identical** to the full-prefill baseline.
- **1st use** of a novel `[A,B,C]` combo: ~1× baseline (reuses only `[A]`, decodes the novel `B+C`
  suffix — expected one-time cost) and caches the `[A,B]` and `[A,B,C]` combined snapshots.
- **2nd use** of the same `[A,B,C]`: **9.5× prefill** (4263→447 ms), pure state restore, no decode.
- **Partial reuse** `[A,B]`: 356 ms — hits the `[A,B]` intermediate snapshot auto-cached during the
  first `[A,B,C]` call (RadixAttention-style chaining).

### Performance
- `SCRIPT_llama_bench.sh` (Qwen3-4B Q4_0, 2×gfx906, `-b 2048 -ub 2048`): **pp2048 = 3191.8 t/s**,
  **tg128 = 146.0 t/s**.
- PIC request-time reuse micro-benchmark on the same model, ~4615-token prompt (one reused chunk
  + short query):
  - full-prefill baseline: **2203 ms** prefill (~2095 t/s)
  - PIC (`cache_ids`): **411 ms** prefill → **5.36× faster** wall-clock prefill.
  - Note: the server reports `timings.prompt_n` = full assembled length, so its derived pp t/s
    (≈11k) is artificially high — the honest metric is wall-clock prefill time (5.36×). The
    request-time cost is a cheap KV blob injection (memcpy) + query decode, vs O(N) attention
    prefill for the baseline; speedup grows with prompt length.

### Is the speedup "real"? Catches (measured, Qwen3-4B Q4_0, ~4615-token prompt)
- The 5.36× was the **internal** `timings.prompt_ms` ratio (411 ms vs 2203 ms). Wall-clock
  (HTTP + 12 generated tokens) is **2.73×** (852 ms vs 2322 ms). Both are real; they measure
  different scopes.
- **Encode is a one-time cost excluded from the ratio.** Measured encode = **4085 ms** (prefill
  + serialize KV blob to a host buffer) — *more* than a plain baseline prefill (2322 ms), because
  the blob goes through a CPU host round-trip (`kv_data_get`/`kv_data_set`). So for a **single use**
  PIC is a net loss: encode + 1 request = 4937 ms vs 2322 ms baseline.
- **Break-even ≈ 2–3 reuses** of the same chunk. N=2 still favors baseline (~1.1 s); N=3 PIC wins
  by ~0.3 s; N=5 by ~3.3 s; N=10 by ~10.6 s. PIC only pays off when a chunk is reused repeatedly
  (its intended use case: static docs / system prompts / few-shot examples across many requests).
- **Accuracy is approximate** (unit-test cosine ≈ 0.83 vs exact prefill). LegoLink-0 can diverge
  from the true continuation; LegoLink k=4 stays closer. Trade quality for speed.
- **Only TTFT/prefill benefits**; generation throughput (tg128 ≈ 146 t/s) is unchanged.
- **Encode duplicates model weights** in VRAM (separate scratch context) and does a host↔device
  blob copy — both are the "next step 3" inefficiency; keeping the blob on-device would cut the
  encode cost and lower the break-even further.
- Hybrid/recurrent models (Qwen3.6, DeepSeek-V3) are unsupported by the KV primitive.

### Bugs found and fixed during verification
- **Position collision on reuse:** `llama_pic_assemble` already decodes the whole query (its KV
  occupies the final position), so re-adding that token to the batch collided
  (`"inconsistent sequence positions"` → HTTP 500). Fixed by dropping just that one position and
  re-decoding it (identical KV, correct first-token logits).
- **Encode context batch too small:** the scratch encode context used the default `n_batch = 2048`
  but decoded the entire chunk in one `pic_decode`, so any chunk > 2048 tokens asserted
  (`n_tokens_all <= n_batch`). Fixed by sizing the encode context's `n_batch`/`n_ubatch` to
  `n_ctx` in `post_context_cache`.
- **Architecture limitation:** the relocatable KV primitive only supports the standard
  `llama_kv_cache` (`dynamic_cast` in `src/llama-context.cpp:4001`); hybrid/recurrent models
  (Qwen3.6 → `llama_memory_hybrid`, DeepSeek-V3 → `llama_kv_cache_dsa`) abort on encode. Use a
  standard dense model for PIC.


## Precise next steps

### Next step 1 (core value, medium effort) — request-time reuse  — DONE
Wire `cache_ids` into the completion / chat generation path so a request actually reuses stored
KV at request time:
- Extend the `/v1/chat/completions` and `/completion` request schema with a `cache_ids` field
  (array of strings) and/or a `pic_k` field (LegoLink `k`).  **DONE** — parsed in
  `server_task::params_from_json_cmpl`.
- In the slot/prompt processing of `server-context.cpp`, when `cache_ids` are present:  **DONE**
  1. resolve them via the per-model `llama_kv_store`,
  2. call `llama_pic_assemble(slot.ctx, chunks, dynamic_tokens, k)` **before** the normal prefill,
  3. continue generation from the assembled state.
- Constraint: `llama_pic_assemble` requires positions to be populated consecutively, so inject
  chunks in order, decode link tokens first, then inject warm remainder (already handled in the
  engine). The serving slot must expose a `llama_context` whose cells can be written this way.
- Add a server integration test (Python, under `tools/server/tests/`) that generates a cache,
  then sends a chat request referencing the cache ids and checks the answer matches a non-PIC run.
  **DONE** — `tools/server/tests/unit/test_pic.py`.

### Next step 2 (accuracy) — fix warm-token RoPE
The current approximation leaves chunk-2 warm tokens with chunk-local RoPE, which is the dominant
remaining accuracy cost (cosine ≈ 0.83 in the small-model test). Options:
- Re-rotate warm KV to assembled positions on load (adds a RoPE re-application cost, but only on
  the stored blob, not the full attention).
- Or adopt the paper's note that LegoLink-0's dummy prefix + link recompute is usually sufficient;
  validate accuracy on the paper's LongBench tasks before optimizing.

### Next step 3 (performance) — avoid duplicating weights for encode
`post_context_cache` currently creates a scratch `llama_context` per request (duplicates model
weights in VRAM). Replace with a dedicated encode context or reuse the server's own context
outside serving slots.

### Next step 4 (robustness)
- Honor `persist_dir` for a durable cross-restart cache; add TTL/eviction.
- Namespace the store per loaded model alias, not just `llama_model_desc`.

## Files touched
- `src/llama-kv-cache.cpp`, `src/llama-kv-cache.h` — KV primitive.
- `src/llama-pic.cpp`, `src/llama-pic.h` — engine + store (new); store gained a `std::mutex`.
- `include/llama.h` — public PIC API, incl. `llama_pic_supported`.
- `src/llama-context.cpp` — `llama_pic_*` C API entry points, `llama_pic_supported`.
- `tools/server/server-context.cpp`, `tools/server/server-context.h`, `tools/server/server.cpp`,
  `tools/server/CMakeLists.txt` — `/v1/context_cache` endpoint, request-time reuse, error guards.
- `tools/server/server-task.cpp`, `tools/server/server-task.h` — `cache_ids` / `pic_k` params.
- `tests/test-pic.cpp`, `tests/test-kv-store.cpp`, `tests/CMakeLists.txt` — tests.
- `tools/server/tests/unit/test_pic.py` — server integration test.
