#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int64_t K = 4;
constexpr int64_t M = 4;
constexpr int64_t N = 2;
constexpr int64_t N_EXPERTS = 2;
constexpr int N_CHAIN = 6000;

struct split_data {
    const ggml_tensor * weight = nullptr;
    const ggml_tensor * input  = nullptr;
    const ggml_tensor * expert_weight = nullptr;
    int callback_calls = 0;
};

ggml_backend_meta_split_state get_split_state(const ggml_tensor * tensor, void * userdata) {
    split_data * data = static_cast<split_data *>(userdata);
    ++data->callback_calls;

    ggml_backend_meta_split_state state = {};
    state.nr[0] = 1;
    state.n_segments = 1;
    if (tensor == data->expert_weight) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_2;
        state.ne[0] = N_EXPERTS / 2;
        state.ne[1] = N_EXPERTS - state.ne[0];
    } else if (tensor == data->weight || tensor == data->input) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_0;
        state.ne[0] = K / 2;
        state.ne[1] = K - state.ne[0];
    } else {
        state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    }
    return state;
}

ggml_tensor * append_neg_chain(ggml_context * ctx, ggml_tensor * tensor) {
    for (int i = 0; i < N_CHAIN; ++i) {
        tensor = ggml_neg(ctx, tensor);
    }
    return tensor;
}

bool check_tensor(ggml_tensor * tensor, const std::vector<float> & expected, const char * graph_name) {
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 1e-6f * std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance) {
            std::fprintf(stderr, "%s mismatch[%zu]: got %.9g expected %.9g\n",
                    graph_name, i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        std::fprintf(stderr, "CPU backend unavailable\n");
        return 1;
    }

    split_data split;
    ggml_backend_dev_t devices[2] = { cpu, cpu };
    ggml_backend_dev_t meta_device = ggml_backend_meta_device(devices, 2, get_split_state, &split);
    ggml_backend_t meta = ggml_backend_dev_init(meta_device, nullptr);
    if (!meta) {
        std::fprintf(stderr, "failed to initialize Meta backend\n");
        return 1;
    }
    ggml_backend_buffer_type_t meta_buft = ggml_backend_dev_buffer_type(meta_device);

    ggml_context * static_ctx = ggml_init({ 8 * ggml_tensor_overhead(), nullptr, true });
    if (!static_ctx) {
        ggml_backend_free(meta);
        return 1;
    }
    ggml_tensor * weight = ggml_new_tensor_2d(static_ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * input  = ggml_new_tensor_2d(static_ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * seed   = ggml_new_tensor_2d(static_ctx, GGML_TYPE_F32, M, N);
    ggml_tensor * expert_weight = ggml_new_tensor_3d(static_ctx, GGML_TYPE_F32, K, M, N_EXPERTS);
    ggml_tensor * expert_input  = ggml_new_tensor_3d(static_ctx, GGML_TYPE_F32, K, 1, N);
    ggml_tensor * route_a = ggml_new_tensor_2d(static_ctx, GGML_TYPE_I32, 1, N);
    ggml_tensor * route_b = ggml_new_tensor_2d(static_ctx, GGML_TYPE_I32, 1, N);
    ggml_set_name(weight, "cache_static_weight");
    ggml_set_name(input,  "cache_static_input");
    ggml_set_name(seed,   "cache_static_seed");
    ggml_set_name(expert_weight, "cache_static_expert_weight");
    ggml_set_name(expert_input,  "cache_static_expert_input");
    ggml_set_name(route_a, "cache_static_route_a");
    ggml_set_name(route_b, "cache_static_route_b");
    split.weight = weight;
    split.input  = input;
    split.expert_weight = expert_weight;

    ggml_backend_buffer_t static_buffer = ggml_backend_alloc_ctx_tensors_from_buft(static_ctx, meta_buft);
    if (!static_buffer) {
        std::fprintf(stderr, "failed to allocate static Meta tensors\n");
        ggml_free(static_ctx);
        ggml_backend_free(meta);
        return 1;
    }

    std::vector<float> weight_data(ggml_nelements(weight));
    std::vector<float> input_data(ggml_nelements(input));
    std::vector<float> seed_data(ggml_nelements(seed));
    std::vector<float> expert_weight_data(ggml_nelements(expert_weight));
    std::vector<float> expert_input_data(ggml_nelements(expert_input));
    const std::vector<int32_t> route_a_data = { 0, 1 };
    const std::vector<int32_t> route_b_data = { 1, 0 };
    for (int64_t row = 0; row < M; ++row) {
        for (int64_t col = 0; col < K; ++col) {
            weight_data[col + K*row] = 0.25f * float(1 + col + 3*row);
        }
    }
    for (int64_t row = 0; row < N; ++row) {
        for (int64_t col = 0; col < K; ++col) {
            input_data[col + K*row] = 0.5f + float(col + 2*row);
        }
    }
    for (size_t i = 0; i < seed_data.size(); ++i) {
        seed_data[i] = 10.0f + float(i);
    }
    for (int64_t expert = 0; expert < N_EXPERTS; ++expert) {
        for (int64_t row = 0; row < M; ++row) {
            for (int64_t col = 0; col < K; ++col) {
                expert_weight_data[col + K*(row + M*expert)] =
                    0.125f * float(1 + col + 2*row + 5*expert);
            }
        }
    }
    for (int64_t token = 0; token < N; ++token) {
        for (int64_t col = 0; col < K; ++col) {
            expert_input_data[col + K*token] = 0.25f * float(1 + col + 3*token);
        }
    }
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(input,  input_data.data(),  0, input_data.size()  * sizeof(float));
    ggml_backend_tensor_set(seed,   seed_data.data(),   0, seed_data.size()   * sizeof(float));
    ggml_backend_tensor_set(expert_weight, expert_weight_data.data(), 0, expert_weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(expert_input,  expert_input_data.data(),  0, expert_input_data.size()  * sizeof(float));
    ggml_backend_tensor_set(route_a, route_a_data.data(), 0, route_a_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(route_b, route_b_data.data(), 0, route_b_data.size() * sizeof(int32_t));

    const int graph_capacity = N_CHAIN + 8;
    const size_t graph_mem_size =
        size_t(graph_capacity) * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(graph_capacity, false) + 4096;
    std::vector<uint8_t> graph_memory(graph_mem_size);
    ggml_context * graph_ctx = ggml_init({ graph_memory.size(), graph_memory.data(), true });
    ggml_gallocr_t graph_alloc = ggml_gallocr_new(meta_buft);
    if (!graph_ctx || !graph_alloc) {
        std::fprintf(stderr, "failed to create reusable graph arena\n");
        if (graph_alloc) ggml_gallocr_free(graph_alloc);
        if (graph_ctx) ggml_free(graph_ctx);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(meta);
        return 1;
    }

    ggml_tensor * first_a = ggml_mul_mat(graph_ctx, weight, input);
    ggml_tensor * zero_a = ggml_scale(graph_ctx, seed, 0.0f);
    ggml_tensor * output_a = append_neg_chain(graph_ctx, ggml_add(graph_ctx, first_a, zero_a));
    ggml_cgraph * graph_a = ggml_new_graph_custom(graph_ctx, graph_capacity, false);
    ggml_build_forward_expand(graph_a, output_a);
    if (!ggml_gallocr_alloc_graph(graph_alloc, graph_a) ||
            ggml_backend_graph_compute(meta, graph_a) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "first Meta graph failed\n");
        return 1;
    }

    std::vector<float> expected_a(M * N);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += weight_data[k + K*m] * input_data[k + K*n];
            }
            expected_a[m + M*n] = sum; // N_CHAIN is even.
        }
    }
    if (!check_tensor(output_a, expected_a, "graph A")) {
        return 1;
    }
    const int static_calls_after_a = split.callback_calls;

    // Rebuild in the exact same arena. The first and all subsequent compute
    // tensor addresses are deliberately recycled, but the first operation and
    // split state change from PARTIAL MUL_MAT to MIRRORED DUP.
    ggml_reset(graph_ctx);
    ggml_tensor * first_b = ggml_dup(graph_ctx, seed);
    ggml_tensor * zero_b = ggml_scale(graph_ctx, seed, 0.0f);
    ggml_tensor * output_b = append_neg_chain(graph_ctx, ggml_add(graph_ctx, first_b, zero_b));
    ggml_cgraph * graph_b = ggml_new_graph_custom(graph_ctx, graph_capacity, false);
    ggml_build_forward_expand(graph_b, output_b);
    if (first_b != first_a) {
        std::fprintf(stderr, "test setup failed to recycle first tensor address\n");
        return 1;
    }
    if (!ggml_gallocr_alloc_graph(graph_alloc, graph_b) ||
            ggml_backend_graph_compute(meta, graph_b) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "second Meta graph failed\n");
        return 1;
    }
    if (!check_tensor(output_b, seed_data, "graph B")) {
        return 1;
    }
    if (split.callback_calls != static_calls_after_a) {
        std::fprintf(stderr, "static split callback was recomputed across graph rebuild: %d -> %d\n",
                static_calls_after_a, split.callback_calls);
        return 1;
    }

    auto expected_expert_chain = [&](const std::vector<int32_t> & routes) {
        std::vector<float> expected(M * N);
        for (int64_t token = 0; token < N; ++token) {
            const int64_t expert = routes[token];
            std::vector<float> hidden(M);
            for (int64_t row = 0; row < M; ++row) {
                for (int64_t col = 0; col < K; ++col) {
                    hidden[row] += expert_weight_data[col + K*(row + M*expert)] *
                        expert_input_data[col + K*token];
                }
            }
            for (int64_t row = 0; row < M; ++row) {
                for (int64_t col = 0; col < K; ++col) {
                    expected[row + M*token] += expert_weight_data[col + K*(row + M*expert)] * hidden[col];
                }
            }
        }
        return expected;
    };

    ggml_reset(graph_ctx);
    ggml_tensor * expert_first_a = ggml_mul_mat_id(graph_ctx, expert_weight, expert_input, route_a);
    ggml_tensor * expert_output_a = ggml_mul_mat_id(graph_ctx, expert_weight, expert_first_a, route_a);
    ggml_cgraph * expert_graph_a = ggml_new_graph_custom(graph_ctx, graph_capacity, false);
    ggml_build_forward_expand(expert_graph_a, expert_output_a);
    if (!ggml_gallocr_alloc_graph(graph_alloc, expert_graph_a) ||
            ggml_backend_graph_compute(meta, expert_graph_a) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "first expert Meta graph failed\n");
        return 1;
    }
    if (!check_tensor(expert_output_a, expected_expert_chain(route_a_data), "expert graph A")) {
        return 1;
    }
    const int static_calls_after_expert_a = split.callback_calls;

    ggml_reset(graph_ctx);
    ggml_tensor * expert_first_b = ggml_mul_mat_id(graph_ctx, expert_weight, expert_input, route_b);
    ggml_tensor * expert_output_b = ggml_mul_mat_id(graph_ctx, expert_weight, expert_first_b, route_b);
    ggml_cgraph * expert_graph_b = ggml_new_graph_custom(graph_ctx, graph_capacity, false);
    ggml_build_forward_expand(expert_graph_b, expert_output_b);
    if (expert_first_b != expert_first_a || expert_output_b != expert_output_a) {
        std::fprintf(stderr, "test setup failed to recycle expert tensor addresses\n");
        return 1;
    }
    if (!ggml_gallocr_alloc_graph(graph_alloc, expert_graph_b) ||
            ggml_backend_graph_compute(meta, expert_graph_b) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "second expert Meta graph failed\n");
        return 1;
    }
    if (!check_tensor(expert_output_b, expected_expert_chain(route_b_data), "expert graph B")) {
        return 1;
    }
    // Only the new route_b source needs split classification.
    if (split.callback_calls != static_calls_after_expert_a + 1) {
        std::fprintf(stderr, "unexpected expert split callback count across graph rebuild: %d -> %d\n",
                static_calls_after_expert_a, split.callback_calls);
        return 1;
    }

    ggml_gallocr_free(graph_alloc);
    ggml_free(graph_ctx);
    ggml_backend_buffer_free(static_buffer);
    ggml_free(static_ctx);
    ggml_backend_free(meta);
    std::printf("PASS Meta split cache rebuild: %d recycled nodes; static callback count %d\n",
            N_CHAIN + 5, split.callback_calls);
    return 0;
}
