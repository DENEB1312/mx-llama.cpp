#include "llama-pic.h"

#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

//
// internal helpers
//

static void pic_decode(llama_context * ctx, const std::vector<llama_token> & toks,
                       const std::vector<llama_pos> & pos, llama_seq_id seq) {
    llama_batch batch = llama_batch_init(toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = pos[i];
        batch.seq_id[i][0] = seq;
        batch.n_seq_id[i]  = 1;
        batch.logits[i]    = (i + 1 == toks.size()) ? 1 : 0;
    }
    batch.n_tokens = toks.size();
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    GGML_ASSERT(rc == 0 && "pic_decode failed");
}

// Slice a full chunk KV blob (covering `n` cells) down to cells [a, b).
// Parses the self-describing per-layer format written by llama_kv_cache::kv_data_get.
static std::vector<uint8_t> pic_slice_blob(const std::vector<uint8_t> & src, uint32_t n, uint32_t a, uint32_t b) {
    GGML_ASSERT(b > a && b <= n);
    const uint8_t * p = src.data();
    const uint8_t * end = src.data() + src.size();

    std::vector<uint8_t> dst;
    dst.reserve(src.size());

    while (p < end) {
        // ---- K ----
        int32_t k_type; uint64_t k_row;
        memcpy(&k_type, p, sizeof(k_type)); p += sizeof(k_type);
        memcpy(&k_row,  p, sizeof(k_row));  p += sizeof(k_row);
        dst.insert(dst.end(), (uint8_t *) &k_type, (uint8_t *) &k_type + sizeof(k_type));
        dst.insert(dst.end(), (uint8_t *) &k_row,  (uint8_t *) &k_row  + sizeof(k_row));
        const size_t k_copy = (size_t) (b - a) * k_row;
        dst.insert(dst.end(), p + (size_t) a * k_row, p + (size_t) a * k_row + k_copy);
        p += (size_t) n * k_row;

        // ---- V ----
        llama_pic_v_hdr vh;
        memcpy(&vh, p, sizeof(vh)); p += sizeof(vh);
        dst.insert(dst.end(), (uint8_t *) &vh, (uint8_t *) &vh + sizeof(vh));
        const size_t v_copy = (size_t) (b - a) * vh.row_size;
        if (!vh.transposed) {
            dst.insert(dst.end(), p + (size_t) a * vh.row_size, p + (size_t) a * vh.row_size + v_copy);
            p += (size_t) n * vh.row_size;
        } else {
            const size_t v_el = vh.row_size / vh.n_embd; // row_size == n_embd * el_size
            for (uint32_t j = 0; j < vh.n_embd; ++j) {
                const size_t off = ((size_t) a + (size_t) j * n) * v_el;
                dst.insert(dst.end(), p + off, p + off + (size_t) (b - a) * v_el);
            }
            p += (size_t) n * vh.n_embd * v_el;
        }
    }

    return dst;
}

//
// public engine
//

llama_pic_chunk llama_pic_encode_chunk(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t dummy_prepend) {
    const uint32_t n = tokens.size();
    GGML_ASSERT(n > 0);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token bos = llama_vocab_bos(vocab);

    // reset sequence 0 so the function is reusable on the same context
    llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);

    // prefill: optional dummy BOS prefix (LegoLink-0) + the real chunk tokens
    std::vector<llama_token> seq;
    std::vector<llama_pos>   pos;
    seq.reserve(dummy_prepend + n);
    pos.reserve(dummy_prepend + n);
    for (uint32_t i = 0; i < dummy_prepend; ++i) { seq.push_back(bos); pos.push_back(i); }
    for (uint32_t i = 0; i < n; ++i) { seq.push_back(tokens[i]); pos.push_back(dummy_prepend + i); }

    pic_decode(ctx, seq, pos, 0);

    // extract only the real chunk's KV (drop the dummy prefix)
    std::vector<uint8_t> kv(llama_pic_kv_data_size(ctx, n));
    llama_pic_kv_data_get(ctx, 0, dummy_prepend, n, kv.data());

    llama_pic_chunk chunk;
    chunk.tokens = tokens;
    chunk.kv     = std::move(kv);
    return chunk;
}

uint32_t llama_pic_assemble(llama_context * ctx, const std::vector<llama_pic_chunk> & chunks,
                             const std::vector<llama_token> & query, uint32_t k, llama_seq_id seq) {
    GGML_ASSERT(!chunks.empty());

    uint32_t pos = 0;

    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        const auto & chunk = chunks[ci];
        const uint32_t n = chunk.n_tokens();
        GGML_ASSERT(n > 0);

        if (ci == 0 || k == 0) {
            // inject the whole chunk (first chunk, or LegoLink-0: no recompute)
            std::vector<llama_pos> p(n);
            for (uint32_t i = 0; i < n; ++i) p[i] = pos + i;
            llama_pic_kv_data_set(ctx, seq, pos, n, chunk.kv.data(), p.data());
        } else {
            // LegoLink: recompute the first k link tokens, then inject the warm remainder.
            // Order matters: decode link tokens first (keeps positions consecutive), then inject warm.
            const uint32_t kk = std::min(k, n);
            std::vector<llama_token> link(chunk.tokens.begin(), chunk.tokens.begin() + kk);
            std::vector<llama_pos>   pl(kk);
            for (uint32_t i = 0; i < kk; ++i) pl[i] = pos + i;
            pic_decode(ctx, link, pl, seq);

            if (kk < n) {
                auto warm = pic_slice_blob(chunk.kv, n, kk, n);
                const uint32_t nw = n - kk;
                std::vector<llama_pos> pw(nw);
                for (uint32_t i = 0; i < nw; ++i) pw[i] = pos + kk + i;
                llama_pic_kv_data_set(ctx, seq, pos + kk, nw, warm.data(), pw.data());
            }
        }

        pos += n;
    }

    // decode the dynamic query (prefill of the assembled prompt)
    if (!query.empty()) {
        std::vector<llama_pos> pq(query.size());
        for (size_t i = 0; i < query.size(); ++i) pq[i] = pos + (uint32_t) i;
        pic_decode(ctx, query, pq, seq);
        pos += (uint32_t) query.size();
    }

    return pos;
}

//
// prefix-cache mode (hybrid / recurrent models)
//

llama_pic_chunk llama_pic_encode_chunk_state(llama_context * ctx, const std::vector<llama_token> & tokens) {
    const uint32_t n = tokens.size();
    GGML_ASSERT(n > 0);

    // reset sequence 0 so the function is reusable on the same context
    llama_memory_seq_rm(llama_get_memory(ctx), 0, 0, -1);

    std::vector<llama_pos> pos(n);
    for (uint32_t i = 0; i < n; ++i) pos[i] = (llama_pos) i;
    pic_decode(ctx, tokens, pos, 0);

    // snapshot the whole per-sequence memory state (attention KV + recurrent state)
    const size_t sz = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> blob(sz);
    const size_t got = llama_state_seq_get_data(ctx, blob.data(), sz, 0);
    GGML_ASSERT(got > 0 && got <= sz && "failed to snapshot sequence state");
    blob.resize(got);

    llama_pic_chunk chunk;
    chunk.tokens    = tokens;
    chunk.kv        = std::move(blob);
    chunk.seq_state = true;
    return chunk;
}

uint32_t llama_pic_assemble_state(llama_context * ctx, const llama_pic_chunk & chunk,
                                  const std::vector<llama_token> & query, llama_seq_id seq) {
    GGML_ASSERT(chunk.seq_state && "llama_pic_assemble_state requires a seq_state chunk");

    // clear the target sequence, then restore the snapshot into it at positions [0, n)
    llama_memory_seq_rm(llama_get_memory(ctx), seq, 0, -1);
    const size_t used = llama_state_seq_set_data(ctx, chunk.kv.data(), chunk.kv.size(), seq);
    GGML_ASSERT(used > 0 && "failed to restore sequence state");

    uint32_t pos = chunk.n_tokens();

    if (!query.empty()) {
        std::vector<llama_pos> pq(query.size());
        for (size_t i = 0; i < query.size(); ++i) pq[i] = pos + (uint32_t) i;
        pic_decode(ctx, query, pq, seq);
        pos += (uint32_t) query.size();
    }

    return pos;
}

// snapshot the current memory state of `seq` (positions [0, tokens.size())) into a chunk
static llama_pic_chunk pic_snapshot(llama_context * ctx, llama_seq_id seq, std::vector<llama_token> tokens) {
    const size_t sz = llama_state_seq_get_size(ctx, seq);
    std::vector<uint8_t> blob(sz);
    const size_t got = llama_state_seq_get_data(ctx, blob.data(), sz, seq);
    GGML_ASSERT(got > 0 && got <= sz && "failed to snapshot sequence state");
    blob.resize(got);

    llama_pic_chunk chunk;
    chunk.tokens    = std::move(tokens);
    chunk.kv        = std::move(blob);
    chunk.seq_state = true;
    return chunk;
}

uint32_t llama_pic_assemble_state_chain(llama_context * ctx, llama_kv_store & store,
                                        const std::vector<std::string> & chain,
                                        const std::vector<llama_token> & query, llama_seq_id seq) {
    GGML_ASSERT(!chain.empty());

    // resolve each chunk (must be seq_state) and build cumulative token prefixes + their combined ids
    const size_t m = chain.size();
    std::vector<std::vector<llama_token>> pref(m);
    std::vector<std::string>              comb_id(m);
    {
        std::vector<llama_token> acc;
        for (size_t j = 0; j < m; ++j) {
            const llama_pic_chunk * c = store.get(chain[j]);
            GGML_ASSERT(c != nullptr && c->seq_state && "chain element missing or not a seq_state chunk");
            acc.insert(acc.end(), c->tokens.begin(), c->tokens.end());
            pref[j]    = acc;
            comb_id[j] = store.cache_id(acc);
        }
    }

    // clear the target sequence, then restore the longest already-cached combined prefix
    llama_memory_seq_rm(llama_get_memory(ctx), seq, 0, -1);

    int start = -1;
    for (int j = (int) m - 1; j >= 0; --j) {
        const llama_pic_chunk * cc = store.get(comb_id[j]);
        if (cc != nullptr && cc->seq_state) {
            const size_t used = llama_state_seq_set_data(ctx, cc->kv.data(), cc->kv.size(), seq);
            GGML_ASSERT(used > 0 && "failed to restore combined prefix state");
            start = j;
            break;
        }
    }
    GGML_ASSERT(start >= 0 && "chain[0] snapshot must exist");

    uint32_t pos = (uint32_t) pref[start].size();

    // decode the remaining (novel) chunk tokens, caching each new combined snapshot for reuse
    for (size_t j = (size_t) start + 1; j < m; ++j) {
        const std::vector<llama_token> & tk = store.get(chain[j])->tokens;
        std::vector<llama_pos> p(tk.size());
        for (size_t i = 0; i < tk.size(); ++i) p[i] = pos + (llama_pos) i;
        pic_decode(ctx, tk, p, seq);
        pos += (uint32_t) tk.size();

        store.put(pic_snapshot(ctx, seq, pref[j]));
    }

    // decode the dynamic query last
    if (!query.empty()) {
        std::vector<llama_pos> pq(query.size());
        for (size_t i = 0; i < query.size(); ++i) pq[i] = pos + (llama_pos) i;
        pic_decode(ctx, query, pq, seq);
        pos += (uint32_t) query.size();
    }

    return pos;
}

//
// content-addressed KV store
//

namespace {
    // FNV-1a 64-bit
    static uint64_t fnv1a(const uint8_t * data, size_t len, uint64_t h = 1469598103934665603ull) {
        for (size_t i = 0; i < len; ++i) {
            h ^= data[i];
            h *= 1099511628211ull;
        }
        return h;
    }

    static std::string to_hex(uint64_t v) {
        std::ostringstream os;
        os << std::hex << v;
        return os.str();
    }
}

llama_kv_store::llama_kv_store(const std::string & model_id, const std::string & persist_dir)
    : model_id_(model_id), persist_dir_(persist_dir) {
    if (!persist_dir_.empty()) {
        load_from_disk();
    }
}

std::string llama_kv_store::cache_id(const std::vector<llama_token> & tokens) const {
    uint64_t h = fnv1a((const uint8_t *) model_id_.data(), model_id_.size());
    for (const auto & t : tokens) {
        uint32_t v = (uint32_t) t;
        h = fnv1a((const uint8_t *) &v, sizeof(v), h);
    }
    return to_hex(h);
}

std::string llama_kv_store::put(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t dummy_prepend) {
    auto chunk = llama_pic_encode_chunk(ctx, tokens, dummy_prepend);
    return put(chunk);
}

std::string llama_kv_store::put_state(llama_context * ctx, const std::vector<llama_token> & tokens) {
    auto chunk = llama_pic_encode_chunk_state(ctx, tokens);
    return put(chunk);
}

std::string llama_kv_store::put(const llama_pic_chunk & chunk) {
    const std::string id = cache_id(chunk.tokens);
    {
        std::lock_guard<std::mutex> lock(mu_);
        chunks_[id] = chunk;
    }
    if (!persist_dir_.empty()) {
        save_to_disk(id, chunk);
    }
    return id;
}

bool llama_kv_store::has(const std::string & id) const {
    std::lock_guard<std::mutex> lock(mu_);
    return chunks_.find(id) != chunks_.end();
}

const llama_pic_chunk * llama_kv_store::get(const std::string & id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = chunks_.find(id);
    return it == chunks_.end() ? nullptr : &it->second;
}

void llama_kv_store::load_from_disk() {
    // *.pic files: [uint32 n_tokens][n_tokens * int32 tokens][uint8 seq_state][uint64 kv_size][kv bytes]
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::is_directory(persist_dir_, ec)) {
        return;
    }
    for (const auto & ent : fs::directory_iterator(persist_dir_, ec)) {
        if (ent.path().extension() != ".pic") continue;
        std::ifstream f(ent.path(), std::ios::binary);
        if (!f) continue;
        uint32_t n_tokens = 0;
        f.read((char *) &n_tokens, sizeof(n_tokens));
        std::vector<llama_token> tokens(n_tokens);
        f.read((char *) tokens.data(), (std::streamsize) (n_tokens * sizeof(llama_token)));
        uint8_t seq_state = 0;
        f.read((char *) &seq_state, sizeof(seq_state));
        uint64_t kv_size = 0;
        f.read((char *) &kv_size, sizeof(kv_size));
        std::vector<uint8_t> kv(kv_size);
        f.read((char *) kv.data(), (std::streamsize) kv_size);
        if (!f) continue;
        llama_pic_chunk chunk;
        chunk.tokens    = std::move(tokens);
        chunk.kv        = std::move(kv);
        chunk.seq_state = seq_state != 0;
        chunks_[ent.path().stem().string()] = std::move(chunk);
    }
}

void llama_kv_store::save_to_disk(const std::string & id, const llama_pic_chunk & chunk) const {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(persist_dir_, ec);
    const std::string path = persist_dir_ + "/" + id + ".pic";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return;
    const uint32_t n_tokens = chunk.n_tokens();
    const uint64_t kv_size  = chunk.kv.size();
    const uint8_t  seq_state = chunk.seq_state ? 1 : 0;
    f.write((const char *) &n_tokens, sizeof(n_tokens));
    f.write((const char *) chunk.tokens.data(), (std::streamsize) (n_tokens * sizeof(llama_token)));
    f.write((const char *) &seq_state, sizeof(seq_state));
    f.write((const char *) &kv_size, sizeof(kv_size));
    f.write((const char *) chunk.kv.data(), (std::streamsize) kv_size);
}
