#include "llama.h"
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

// decode a batch of tokens at the given absolute positions; logits are produced only for the last token
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
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed with rc=%d\n", rc);
        exit(1);
    }
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

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs((double) a[i] - (double) b[i]));
    }
    return m;
}

static int argmax(const std::vector<float> & a) {
    int best = 0;
    for (size_t i = 1; i < a.size(); ++i) {
        if (a[i] > a[best]) best = (int) i;
    }
    return best;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (argv[i] && argv[i][0] != '\0') {
            model_path = argv[i];
        }
    }
    if (model_path == nullptr) {
        model_path = getenv("LLAMA_PIC_MODEL");
    }
    if (model_path == nullptr) {
        model_path = "/home/iacopo/.cache/qmd/models/hf_ggml-org_embeddinggemma-300M-Q8_0.gguf";
    }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load model '%s'\n", model_path);
        return 1;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    llama_context * ctx_ref = llama_init_from_model(model, cparams);
    if (ctx_ref == nullptr) {
        fprintf(stderr, "failed to init context\n");
        return 1;
    }

    const std::string prompt =
        "The company Thenum was founded in 1998. It is located in the northern district. "
        "Derek is a single man living in the Thenum district. He works at the Chrysan Company. "
        "The Chrysan Company manufactures optical sensors and employs over two thousand people. "
        "Which company does Derek work for? Answer in five words.";

    const auto toks = tokenize(ctx_ref, prompt, true);
    const int N = (int) toks.size();
    const int N1 = N / 2;          // chunk 1 length
    const int N2 = N - N1;         // chunk 2 length
    const int k  = std::min(4, N2); // LegoLink: recompute first k tokens of chunk 2

    std::vector<llama_pos> pos_all(N);
    for (int i = 0; i < N; ++i) pos_all[i] = i;

    // ---- Reference: full prompt prefill, then one more token ----
    decode(ctx_ref, toks, pos_all, 0);

    // extract reference KV blobs before decoding the extra token (keeps cells [0, N) untouched)
    auto extract = [&](llama_context * c, uint32_t i0, uint32_t n) {
        std::vector<uint8_t> blob(llama_pic_kv_data_size(c, n));
        llama_pic_kv_data_get(c, 0, i0, n, blob.data());
        return blob;
    };
    auto blob_full = extract(ctx_ref, 0, N);
    auto blob1_ref = extract(ctx_ref, 0, N1);

    std::vector<llama_token> next_tok = { llama_vocab_bos(llama_model_get_vocab(model)) };
    std::vector<llama_pos>  next_pos  = { N };
    decode(ctx_ref, next_tok, next_pos, 0);
    const auto L_ref_next = logits(ctx_ref);

    // ===== Test A: full-prompt KV extracted then re-injected must reproduce reference exactly =====
    {
        llama_context * ctx_a = llama_init_from_model(model, cparams);
        llama_pic_kv_data_set(ctx_a, 0, 0, N, blob_full.data(), pos_all.data());
        decode(ctx_a, next_tok, next_pos, 0);
        const auto L_a = logits(ctx_a);

        const double cos = cosine(L_a, L_ref_next);
        const double mad = max_abs_diff(L_a, L_ref_next);
        printf("Test A (full re-inject): cosine=%.6f max_abs_diff=%.6e\n", cos, mad);
        GGML_ASSERT(cos > 0.9999 && mad < 1e-3);

        llama_free(ctx_a);
    }

    // ===== Test B (control): inject chunk1 from reference, then decode chunk2 fully =====
    // chunk2 is recomputed at correct positions, so this must match the reference very closely.
    // (A tiny residual FP difference vs. the single combined forward is expected and harmless.)
    {
        llama_context * ctx_b = llama_init_from_model(model, cparams);
        std::vector<llama_pos> pos1(N1);
        for (int i = 0; i < N1; ++i) pos1[i] = i;
        llama_pic_kv_data_set(ctx_b, 0, 0, N1, blob1_ref.data(), pos1.data());

        std::vector<llama_token> chunk2(toks.begin() + N1, toks.end());
        std::vector<llama_pos>  pos2(N2);
        for (int i = 0; i < N2; ++i) pos2[i] = N1 + i;
        decode(ctx_b, chunk2, pos2, 0);          // full correct recompute of chunk 2
        decode(ctx_b, next_tok, next_pos, 0);
        const auto L_b = logits(ctx_b);

        const double cos = cosine(L_b, L_ref_next);
        const double mad = max_abs_diff(L_b, L_ref_next);
        printf("Test B (inject chunk1 + decode chunk2): cosine=%.6f max_abs_diff=%.6e\n", cos, mad);
        GGML_ASSERT(cos > 0.99);

        llama_free(ctx_b);
    }

    // ===== Test C (LegoLink): inject chunk1 + chunk2-warm, recompute only the first k link tokens =====
    {
        llama_context * ctx_c2 = llama_init_from_model(model, cparams);
        decode(ctx_c2, std::vector<llama_token>(toks.begin() + N1, toks.end()),
               std::vector<llama_pos>(N2, 0), 0); // chunk2 prefilled at local positions 0..N2-1
        auto blob2_warm = extract(ctx_c2, k, N2 - k);   // tokens [k, N2)
        llama_free(ctx_c2);

        // Extract chunk1 KV from a context that prefilled chunk1 at its own (correct) positions.
        std::vector<llama_pos> pos1(N1);
        for (int i = 0; i < N1; ++i) pos1[i] = i;
        llama_context * ctx_c1b = llama_init_from_model(model, cparams);
        decode(ctx_c1b, std::vector<llama_token>(toks.begin(), toks.begin() + N1),
               std::vector<llama_pos>(pos1.begin(), pos1.end()), 0);
        auto blob1 = extract(ctx_c1b, 0, N1);
        llama_free(ctx_c1b);

        llama_context * ctx_c = llama_init_from_model(model, cparams);

        // Populate positions strictly in order so the scheduler's consecutive-position rule holds:
        //   1) inject chunk1 (positions 0 .. N1-1)
        //   2) decode the k link tokens of chunk2 (positions N1 .. N1+k-1) -> recomputed by the model
        //   3) inject chunk2 warm tokens (positions N1+k .. N1+N2-1)
        //   4) decode the final token
        // This is exactly the LegoLink linking step: only the first k tokens of each non-first
        // chunk are recomputed; the rest reuse the precomputed (position-independent) chunk KV.

        llama_pic_kv_data_set(ctx_c, 0, 0, N1, blob1.data(), pos1.data());

        std::vector<llama_token> link(toks.begin() + N1, toks.begin() + N1 + k);
        std::vector<llama_pos>  pos_link(k);
        for (int i = 0; i < k; ++i) pos_link[i] = N1 + i;
        decode(ctx_c, link, pos_link, 0);

        std::vector<llama_pos> pos_warm(N2 - k);
        for (int i = 0; i < N2 - k; ++i) pos_warm[i] = N1 + k + i;
        llama_pic_kv_data_set(ctx_c, 0, N1 + k, N2 - k, blob2_warm.data(), pos_warm.data());

        // final token
        decode(ctx_c, next_tok, next_pos, 0);
        const auto L_c = logits(ctx_c);

        const double cos = cosine(L_c, L_ref_next);
        const double mad = max_abs_diff(L_c, L_ref_next);
        const bool top1 = argmax(L_c) == argmax(L_ref_next);
        printf("Test C (LegoLink k=%d): cosine=%.6f max_abs_diff=%.6e top1_match=%d\n",
               k, cos, mad, (int) top1);
        // LegoLink is an approximation: chunk2 warm tokens keep their chunk-local RoPE, which
        // introduces a small attention error (this is the fundamental PIC accuracy cost studied
        // in EPIC, arXiv:2410.15332). We therefore require the pipeline to run and the predicted
        // next token to be preserved, rather than bit-exact logits.
        GGML_ASSERT(std::isfinite(cos) && cos > 0.8 && top1);

        llama_free(ctx_c);
    }

    llama_free(ctx_ref);
    llama_model_free(model);
    llama_backend_free();

    printf("PIC tests passed.\n");
    return 0;
}
