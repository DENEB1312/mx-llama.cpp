// Minimal test for Gated DeltaNet kernel
// Tests that the kernel runs without crashing and produces valid output

#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

int main(int argc, char * argv[]) {
    (void)argc; (void)argv;
    printf("=== Minimal Gated DeltaNet Test ===\n\n");

    // Test configs: S_v, H, n_tokens, KDA
    struct TestCase {
        int64_t S_v;
        int64_t H;
        int64_t n_tokens;
        bool kda;
        const char * name;
    };

    TestCase tests[] = {
        {16, 4, 64, false, "S_v=16, CS=64, single chunk"},
        {32, 4, 128, false, "S_v=32, CS=64, two chunks"},
        {64, 4, 100, false, "S_v=64, padding (100%64=36)"},
        {64, 4, 16, true, "S_v=64, CS=16, KDA, single chunk"},
        {64, 4, 32, true, "S_v=64, CS=16, KDA, two chunks"},
        {128, 4, 200, false, "S_v=128, large state"},
    };

    // Initialize CPU backend for tensor operations
    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);

    int passed = 0;
    int failed = 0;

    for (auto & tc : tests) {
        printf("Test: %s\n", tc.name);
        printf("  S_v=%lld, H=%lld, n_tokens=%lld, KDA=%d\n",
               (long long)tc.S_v, (long long)tc.H, (long long)tc.n_tokens, (int)tc.kda);

        // Create CPU context
        ggml_init_params params = {.mem_size = 64 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = false};
        ggml_context * ctx = ggml_init(params);

        // Build graph
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.S_v, tc.H, tc.n_tokens, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.S_v, tc.H, tc.n_tokens, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.S_v, tc.H, tc.n_tokens, 1);

        const int64_t g_ne0 = tc.kda ? tc.S_v : 1;
        ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g_ne0, tc.H, tc.n_tokens, 1);
        ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, tc.H, tc.n_tokens, 1);
        ggml_tensor * state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tc.S_v * tc.S_v * tc.H, 1, 1);

        ggml_set_name(q, "q");
        ggml_set_name(k, "k");
        ggml_set_name(v, "v");
        ggml_set_name(g, "g");
        ggml_set_name(beta, "beta");
        ggml_set_name(state, "state");

        // L2 normalize
        q = ggml_l2_norm(ctx, q, 1e-6f);
        k = ggml_l2_norm(ctx, k, 1e-6f);

        ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state);

        // Initialize with deterministic data using ggml_tensor_set_data
        std::mt19937 rng(42);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
                t->op == GGML_OP_REPEAT || t->op == GGML_OP_GET_ROWS || t->op == GGML_OP_DIAG ||
                t->op == GGML_OP_FILL || t->op == GGML_OP_CPY || t->op == GGML_OP_CONCAT ||
                t->op == GGML_OP_PAD) continue;

            const size_t n = ggml_nelements(t);
            std::vector<float> data(n);
            if (strcmp(t->name, "g") == 0) {
                std::uniform_real_distribution<float> dist(-20.0f, -1e-4f);
                for (size_t i = 0; i < n; i++) data[i] = dist(rng);
            } else if (strcmp(t->name, "beta") == 0) {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                for (size_t i = 0; i < n; i++) data[i] = dist(rng);
            } else if (strcmp(t->name, "v") == 0) {
                std::uniform_real_distribution<float> dist(-0.3f, 5.0f);
                for (size_t i = 0; i < n; i++) data[i] = dist(rng);
            } else {
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                for (size_t i = 0; i < n; i++) data[i] = dist(rng);
            }
            ggml_backend_tensor_set(t, data.data(), 0, n * sizeof(float));
        }

        // Compute on CPU
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        // Get output
        const size_t out_size = ggml_nelements(out) * sizeof(float);
        std::vector<float> cpu_out(out_size / sizeof(float));
        ggml_backend_tensor_get(out, cpu_out.data(), 0, out_size);

        // Get final state
        const size_t state_size = ggml_nelements(state) * sizeof(float);
        std::vector<float> cpu_state(state_size / sizeof(float));
        ggml_backend_tensor_get(state, cpu_state.data(), 0, state_size);

        float out_max = *std::max_element(cpu_out.begin(), cpu_out.end());
        float out_min = *std::min_element(cpu_out.begin(), cpu_out.end());
        float out_mean = std::accumulate(cpu_out.begin(), cpu_out.end(), 0.0f) / cpu_out.size();

        float state_max = *std::max_element(cpu_state.begin(), cpu_state.end());
        float state_min = *std::min_element(cpu_state.begin(), cpu_state.end());
        float state_mean = std::accumulate(cpu_state.begin(), cpu_state.end(), 0.0f) / cpu_state.size();

        printf("  CPU output: max=%.4f, min=%.4f, mean=%.4f\n", out_max, out_min, out_mean);
        printf("  CPU state:  max=%.4f, min=%.4f, mean=%.4f\n", state_max, state_min, state_mean);

        // Check for NaN/Inf
        bool has_nan = false;
        for (float v : cpu_out) {
            if (std::isnan(v) || std::isinf(v)) {
                has_nan = true;
                break;
            }
        }
        for (float v : cpu_state) {
            if (std::isnan(v) || std::isinf(v)) {
                has_nan = true;
                break;
            }
        }

        if (has_nan) {
            printf("  FAIL: NaN/Inf detected\n");
            failed++;
        } else {
            printf("  PASS: No NaN/Inf\n");
            passed++;
        }

        ggml_free(ctx);
        printf("\n");
    }

    printf("=== Summary ===\n");
    printf("Passed: %d, Failed: %d\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
