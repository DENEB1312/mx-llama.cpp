#pragma once

#include "llama.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Self-describing V header that prefixes the V data of every KV blob produced by
// llama_kv_cache::kv_data_get, so a blob can be sliced/reused regardless of the V layout.
#pragma pack(push, 1)
struct llama_pic_v_hdr {
    int32_t  type;       // ggml type of the V tensor
    uint64_t row_size;   // bytes per cell (n_embd_v_gqa * el_size for transposed V)
    uint32_t n_embd;     // n_embd_v_gqa (needed for transposed per-dim copy)
    uint8_t  transposed; // 1 if the V cache is stored transposed
};
#pragma pack(pop)

// Position-Independent Caching (PIC) engine.
//
// These helpers implement the "KVGen" / "KVStore" / "KVLink" stages of EPIC
// (arXiv:2410.15332) on top of the relocatable KV primitive in llama-kv-cache:
//
//   * llama_pic_encode_chunk  -> KVGen:  prefill a static chunk once, extract its KV blob
//   * llama_kv_store          -> KVStore: content-addressed, reusable cache of chunk KV blobs
//   * llama_pic_assemble      -> KVLink:  concatenate chunks in any order + LegoLink recompute
//
// A "chunk" carries both its original tokens (needed to recompute the link tokens) and the
// precomputed KV blob.

struct llama_pic_chunk {
    std::vector<llama_token> tokens;  // original chunk tokens (for link-token recompute)
    std::vector<uint8_t>     kv;      // relocatable KV blob, OR a full per-sequence state blob when seq_state
    // When true, `kv` is a full llama_state_seq snapshot (attention KV + recurrent state) captured at
    // positions [0, n). This is the "prefix-cache" mode used for hybrid/recurrent models whose memory
    // cannot be position-independently relocated: it is reusable ONLY as a position-0 prefix, in exact
    // order, and cannot be composed with other chunks (recurrent state is cumulative & order-dependent).
    bool                     seq_state = false;
    uint32_t n_tokens() const { return (uint32_t) tokens.size(); }
};

// Prefill `tokens` in `ctx` and extract its KV blob (a "precompiled", position-independent chunk).
//
// `dummy_prepend` > 0 enables the LegoLink-0 variant: that many BOS tokens are prepilled before
// the chunk and their KV is dropped on extract, weakening the chunk-boundary attention sink.
// The context is reset (sequence 0 removed) before prefilling so the function is reusable.
llama_pic_chunk llama_pic_encode_chunk(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t dummy_prepend = 0);

// Assemble `chunks` (in the given order) plus a dynamic `query` into `ctx` using LegoLink:
//   - chunk 0 is injected as-is
//   - for each later chunk, the first `k` tokens are recomputed (link tokens) and the rest injected
//   - the query is decoded last (prefill of the whole assembled prompt)
// `k == 0` selects the zero-overhead LegoLink-0 variant (every chunk injected as-is).
// `seq` is the sequence id the assembled KV is attached to (defaults to 0).
// Returns the absolute position just past the assembled prompt (where generation continues).
uint32_t llama_pic_assemble(llama_context * ctx, const std::vector<llama_pic_chunk> & chunks,
                           const std::vector<llama_token> & query, uint32_t k = 4,
                           llama_seq_id seq = 0);

// Prefix-cache mode (hybrid / recurrent models). Instead of a relocatable per-token KV blob, snapshot
// the ENTIRE per-sequence memory state (attention KV + recurrent state) after prefilling `tokens` at
// positions [0, n). Works for any memory backend via the llama_state_seq_* API. The resulting chunk is
// reusable only as a position-0 prefix (see llama_pic_chunk::seq_state).
llama_pic_chunk llama_pic_encode_chunk_state(llama_context * ctx, const std::vector<llama_token> & tokens);

// Restore a seq-state prefix chunk into `seq` at positions [0, n), then decode `query` after it.
// Returns the absolute position just past the assembled prompt (where generation continues).
uint32_t llama_pic_assemble_state(llama_context * ctx, const llama_pic_chunk & chunk,
                                  const std::vector<llama_token> & query, llama_seq_id seq = 0);

// forward decl (defined below)
class llama_kv_store;

// Chained multi-prefix assembly for hybrid/recurrent (seq_state) models.
//
// `chain` is an ordered list of cache ids whose tokens are concatenated (in order) to form the
// assembled prefix. Because recurrent state is order-dependent, chunks cannot be composed
// independently; instead this reuses the longest already-cached *combined* prefix snapshot, decodes
// only the remaining (novel) chunk tokens on top of it, and caches the new combined snapshots in
// `store` for future reuse (RadixAttention-style). The `query` (dynamic tokens) is decoded last.
//
// - The first chain element is always available (its own snapshot), so the search always terminates.
// - When the exact combined chain was seen before, no decoding happens at all (pure state restore).
// Returns the absolute position just past the assembled prefix + query.
uint32_t llama_pic_assemble_state_chain(llama_context * ctx, llama_kv_store & store,
                                        const std::vector<std::string> & chain,
                                        const std::vector<llama_token> & query, llama_seq_id seq = 0);

//
// Content-addressed KV store. Maps a hash of (model id + chunk tokens) to a precomputed chunk so
// that the same static content can be reused across requests at any position. Optionally persists
// to disk so the cache survives process restarts.
//
class llama_kv_store {
public:
    // `model_id` namespaces the cache (different models have incompatible KV layouts).
    // `persist_dir` != "" enables disk-backed storage under that directory.
    llama_kv_store(const std::string & model_id, const std::string & persist_dir = "");
    llama_kv_store() : model_id_(""), persist_dir_("") {}

    // Deterministic cache id for a chunk (without storing it).
    std::string cache_id(const std::vector<llama_token> & tokens) const;

    // Encode `tokens` in `ctx` and store the resulting chunk; returns its cache id.
    std::string put(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t dummy_prepend = 0);

    // Prefix-cache variant (hybrid/recurrent models): snapshot the full per-sequence state.
    std::string put_state(llama_context * ctx, const std::vector<llama_token> & tokens);

    // Store an already-encoded chunk; returns its cache id.
    std::string put(const llama_pic_chunk & chunk);

    bool has(const std::string & id) const;
    const llama_pic_chunk * get(const std::string & id) const;

    size_t size() const { return chunks_.size(); }

private:
    std::string model_id_;
    std::string persist_dir_;
    mutable std::mutex mu_;
    std::map<std::string, llama_pic_chunk> chunks_;

    void load_from_disk();
    void save_to_disk(const std::string & id, const llama_pic_chunk & chunk) const;
};
