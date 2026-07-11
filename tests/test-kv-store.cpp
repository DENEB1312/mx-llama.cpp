#include "llama.h"
#include "llama-pic.h"
#include "testing.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

static std::vector<llama_token> tokenize(llama_context * ctx, const std::string & text, bool add_bos) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::vector<llama_token> tokens(1 + text.size());
    const int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, false);
    tokens.resize(n);
    return tokens;
}

static void decode(llama_context * ctx, const std::vector<llama_token> & toks,
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
    GGML_ASSERT(rc == 0 && "decode failed");
}

static std::vector<float> logits(llama_context * ctx) {
    const llama_model * model = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * l = llama_get_logits(ctx);
    return std::vector<float>(l, l + n_vocab);
}

static double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        na  += (double) a[i] * a[i];
        nb  += (double) b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

static int argmax(const std::vector<float> & a) {
    int best = 0;
    for (size_t i = 1; i < a.size(); ++i) if (a[i] > a[best]) best = (int) i;
    return best;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i < argc; ++i) if (argv[i] && argv[i][0] != '\0') model_path = argv[i];
    if (model_path == nullptr) model_path = getenv("LLAMA_PIC_MODEL");
    if (model_path == nullptr) model_path = "/tmp/opencode/tinyllama.gguf";

    llama_backend_init();

    auto mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) { fprintf(stderr, "failed to load model '%s'\n", model_path); return 1; }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 2048;

    const std::string chunk1_text = "The company Thenum was founded in 1998. It is located in the northern district. "
                                     "Derek is a single man living in the Thenum district.";
    const std::string chunk2_text = "He works at the Chrysan Company. The Chrysan Company manufactures optical sensors "
                                     "and employs over two thousand people.";
    const std::string query_text  = "Which company does Derek work for? Answer in five words.";

    // ---- Reference: full prompt prefill, then one more token ----
    llama_context * ctx_ref = llama_init_from_model(model, cparams);
    auto t1 = tokenize(ctx_ref, chunk1_text, true);
    auto t2 = tokenize(ctx_ref, chunk2_text, false);
    auto tq = tokenize(ctx_ref, query_text,  false);
    std::vector<llama_token> full = t1; full.insert(full.end(), t2.begin(), t2.end()); full.insert(full.end(), tq.begin(), tq.end());

    std::vector<llama_pos> pos_all(full.size());
    for (size_t i = 0; i < full.size(); ++i) pos_all[i] = (llama_pos) i;
    decode(ctx_ref, full, pos_all, 0);
    const auto L_ref = logits(ctx_ref);

    std::vector<llama_token> next_tok = { llama_vocab_bos(llama_model_get_vocab(model)) };
    std::vector<llama_pos>  next_pos  = { (llama_pos) full.size() };
    decode(ctx_ref, next_tok, next_pos, 0);
    const auto L_ref_next = logits(ctx_ref);
    (void) L_ref;

    // ---- KV store: encode chunks, then assemble via LegoLink ----
    char model_desc[256];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    auto store = llama_kv_store(model_desc);

    llama_context * ctx_enc = llama_init_from_model(model, cparams); // reused for each encode
    const std::string id1 = store.put(ctx_enc, t1);
    const std::string id2 = store.put(ctx_enc, t2);
    llama_free(ctx_enc);

    GGML_ASSERT(store.has(id1) && store.has(id2));
    GGML_ASSERT(store.cache_id(t1) == id1);
    GGML_ASSERT(store.size() == 2);

    // LegoLink (recompute first k tokens of chunk 2)
    {
        std::vector<llama_pic_chunk> chunks = { *store.get(id1), *store.get(id2) };
        llama_context * ctx_pic = llama_init_from_model(model, cparams);
        llama_pic_assemble(ctx_pic, chunks, tq, 4);
        const auto L = logits(ctx_pic);
        const double cos = cosine(L, L_ref_next);
        const bool top1 = argmax(L) == argmax(L_ref_next);
        printf("Store+LegoLink k=4: cosine=%.6f top1_match=%d\n", cos, (int) top1);
        GGML_ASSERT(std::isfinite(cos) && cos > 0.7);
        llama_free(ctx_pic);
    }

    // LegoLink-0 (zero-overhead: dummy-prepend during encode, no recompute during assemble)
    {
        llama_context * ctx_e0 = llama_init_from_model(model, cparams);
        const std::string id1z = store.put(ctx_e0, t1, 4);
        const std::string id2z = store.put(ctx_e0, t2, 4);
        llama_free(ctx_e0);

        std::vector<llama_pic_chunk> chunks = { *store.get(id1z), *store.get(id2z) };
        llama_context * ctx_pic = llama_init_from_model(model, cparams);
        llama_pic_assemble(ctx_pic, chunks, tq, 0); // k == 0 -> inject everything
        const auto L = logits(ctx_pic);
        const double cos = cosine(L, L_ref_next);
        const bool top1 = argmax(L) == argmax(L_ref_next);
        printf("Store+LegoLink-0 (dummy=4): cosine=%.6f top1_match=%d\n", cos, (int) top1);
        GGML_ASSERT(std::isfinite(cos) && cos > 0.7);
        llama_free(ctx_pic);
    }

    llama_free(ctx_ref);
    llama_model_free(model);
    llama_backend_free();

    printf("PIC KV-store tests passed.\n");
    return 0;
}
