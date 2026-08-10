#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int64_t N_EXPERTS     = 6;
constexpr int64_t N_EXPERT_USED = 6;
constexpr int64_t N_INPUT       = 4;
constexpr int64_t N_HIDDEN      = 4;
constexpr int64_t N_OUTPUT      = 2;
constexpr float   SWIGLU_LIMIT  = 0.35f;

struct split_data {
    std::array<const ggml_tensor *, 3> weights = {};
};

ggml_backend_meta_split_state get_split_state(const ggml_tensor * tensor, void * userdata) {
    const split_data * data = static_cast<const split_data *>(userdata);
    ggml_backend_meta_split_state state = {};
    state.nr[0] = 1;
    state.n_segments = 1;
    if (std::find(data->weights.begin(), data->weights.end(), tensor) != data->weights.end()) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_2;
        state.ne[0] = N_EXPERTS / 3;
        state.ne[1] = N_EXPERTS / 3;
        state.ne[2] = N_EXPERTS - state.ne[0] - state.ne[1];
    } else {
        state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    }
    return state;
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

bool run_case(ggml_backend_t backend, ggml_backend_buffer_type_t buft, split_data & split,
        int64_t n_tokens, int64_t first_run, int64_t n_runs) {
    const size_t data_size = 8 * ggml_tensor_overhead();
    const size_t graph_size = 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false);
    ggml_context * data_ctx = ggml_init({ data_size, nullptr, true });
    ggml_context * graph_ctx = ggml_init({ graph_size, nullptr, true });
    if (!data_ctx || !graph_ctx) {
        std::fprintf(stderr, "N=%lld: failed to create contexts\n", (long long) n_tokens);
        if (data_ctx)  ggml_free(data_ctx);
        if (graph_ctx) ggml_free(graph_ctx);
        return false;
    }

    ggml_tensor * gate_w = ggml_new_tensor_3d(data_ctx, GGML_TYPE_F32, N_INPUT,  N_HIDDEN, N_EXPERTS);
    ggml_tensor * up_w   = ggml_new_tensor_3d(data_ctx, GGML_TYPE_F32, N_INPUT,  N_HIDDEN, N_EXPERTS);
    ggml_tensor * down_w = ggml_new_tensor_3d(data_ctx, GGML_TYPE_F32, N_HIDDEN, N_OUTPUT, N_EXPERTS);
    ggml_tensor * input  = ggml_new_tensor_3d(data_ctx, GGML_TYPE_F32, N_INPUT, 1, n_tokens);
    ggml_tensor * ids    = ggml_new_tensor_2d(data_ctx, GGML_TYPE_I32, N_EXPERT_USED, n_tokens);
    ggml_tensor * route  = ggml_new_tensor_3d(data_ctx, GGML_TYPE_F32, 1, N_EXPERT_USED, n_tokens);
    ggml_tensor * shared = ggml_new_tensor_2d(data_ctx, GGML_TYPE_F32, N_OUTPUT, n_tokens);

    ggml_set_name(gate_w, "chain_gate_exps");
    ggml_set_name(up_w,   "chain_up_exps");
    ggml_set_name(down_w, "chain_down_exps");
    ggml_set_name(input,  "chain_input");
    ggml_set_name(ids,    "chain_ids");
    ggml_set_name(route,  "chain_route_weights");
    ggml_set_name(shared, "chain_shared_expert");
    split.weights = { gate_w, up_w, down_w };

    ggml_backend_buffer_t data_buffer = ggml_backend_alloc_ctx_tensors_from_buft(data_ctx, buft);
    if (!data_buffer) {
        std::fprintf(stderr, "N=%lld: failed to allocate Meta leaf tensors\n", (long long) n_tokens);
        ggml_free(graph_ctx);
        ggml_free(data_ctx);
        split.weights = {};
        return false;
    }

    ggml_tensor * gate = ggml_mul_mat_id(graph_ctx, gate_w, input, ids);
    ggml_tensor * up = ggml_mul_mat_id(graph_ctx, up_w, input, ids);
    ggml_tensor * gate_clamped = ggml_clamp(graph_ctx, gate, -INFINITY, SWIGLU_LIMIT);
    ggml_tensor * up_clamped = ggml_clamp(graph_ctx, up, -SWIGLU_LIMIT, SWIGLU_LIMIT);
    ggml_tensor * hidden = ggml_swiglu_split(graph_ctx, gate_clamped, up_clamped);
    ggml_tensor * down = ggml_mul_mat_id(graph_ctx, down_w, hidden, ids);
    ggml_tensor * route_repeat = ggml_repeat(graph_ctx, route, down);
    ggml_tensor * weighted = ggml_mul(graph_ctx, down, route_repeat);

    std::array<ggml_tensor *, N_EXPERT_USED> slots;
    for (int64_t slot = 0; slot < N_EXPERT_USED; ++slot) {
        slots[slot] = ggml_view_2d(graph_ctx, weighted, N_OUTPUT, n_tokens, weighted->nb[2], slot * weighted->nb[1]);
    }
    ggml_tensor * sum = ggml_add(graph_ctx, slots[0], slots[1]);
    for (int64_t slot = 2; slot < N_EXPERT_USED; ++slot) {
        sum = ggml_add(graph_ctx, sum, slots[slot]);
    }
    ggml_tensor * materialized = ggml_add(graph_ctx, sum, shared);
    ggml_tensor * output = ggml_dup(graph_ctx, materialized);

    ggml_set_name(gate,         "chain_gate");
    ggml_set_name(up,           "chain_up");
    ggml_set_name(gate_clamped, "chain_gate_clamped");
    ggml_set_name(up_clamped,   "chain_up_clamped");
    ggml_set_name(hidden,       "chain_hidden");
    ggml_set_name(down,         "chain_down");
    ggml_set_name(weighted,     "chain_weighted");
    ggml_set_name(materialized, "chain_shared_frontier");
    ggml_set_name(output,       "chain_output");

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, 64, false);
    for (ggml_tensor * slot : slots) {
        ggml_build_forward_expand(graph, slot);
    }
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t alloc = ggml_gallocr_new(buft);
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
        std::fprintf(stderr, "N=%lld: failed to allocate Meta graph\n", (long long) n_tokens);
        if (alloc) ggml_gallocr_free(alloc);
        ggml_backend_buffer_free(data_buffer);
        ggml_free(graph_ctx);
        ggml_free(data_ctx);
        split.weights = {};
        return false;
    }

    std::vector<float> gate_data(ggml_nelements(gate_w));
    std::vector<float> up_data(ggml_nelements(up_w));
    std::vector<float> down_data(ggml_nelements(down_w));
    for (int64_t expert = 0; expert < N_EXPERTS; ++expert) {
        for (int64_t row = 0; row < N_HIDDEN; ++row) {
            for (int64_t col = 0; col < N_INPUT; ++col) {
                const size_t i = col + N_INPUT * (row + N_HIDDEN * expert);
                gate_data[i] = 0.01f * float(1 + col + 2*row + 3*expert);
                up_data[i] = 0.015f * float(1 + 2*col + row + 2*expert);
            }
        }
        for (int64_t row = 0; row < N_OUTPUT; ++row) {
            for (int64_t col = 0; col < N_HIDDEN; ++col) {
                const size_t i = col + N_HIDDEN * (row + N_OUTPUT * expert);
                down_data[i] = 0.02f * float(1 + col + 3*row + 2*expert);
            }
        }
    }

    ggml_backend_tensor_set(gate_w, gate_data.data(), 0, gate_data.size()*sizeof(float));
    ggml_backend_tensor_set(up_w, up_data.data(), 0, up_data.size()*sizeof(float));
    ggml_backend_tensor_set(down_w, down_data.data(), 0, down_data.size()*sizeof(float));

    bool ok = true;
    constexpr int32_t id_pattern[N_EXPERT_USED] = { 0, 2, 4, 1, 3, 5 };
    for (int64_t local_run = 0; ok && local_run < n_runs; ++local_run) {
        const int64_t run = first_run + local_run;
        std::vector<float> input_data(ggml_nelements(input));
        std::vector<float> route_data(ggml_nelements(route));
        std::vector<float> shared_data(ggml_nelements(shared));
        std::vector<int32_t> ids_data(ggml_nelements(ids));
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t row = 0; row < N_OUTPUT; ++row) {
                shared_data[row + N_OUTPUT*token] = 0.01f * float(1 + row + 3*token + run % 4);
            }
            for (int64_t slot = 0; slot < N_EXPERT_USED; ++slot) {
                ids_data[slot + N_EXPERT_USED*token] = id_pattern[(slot + token + run) % N_EXPERT_USED];
                route_data[slot + N_EXPERT_USED*token] =
                    0.05f * float(1 + (slot + 2*token + run) % 7);
            }
            for (int64_t col = 0; col < N_INPUT; ++col) {
                input_data[col + N_INPUT*token] =
                    0.025f * float(1 + col + 3*token + run % 5);
            }
        }

        std::vector<float> expected = shared_data;
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t slot = 0; slot < N_EXPERT_USED; ++slot) {
                const int64_t expert = ids_data[slot + N_EXPERT_USED*token];
                float hidden_ref[N_HIDDEN];
                for (int64_t row = 0; row < N_HIDDEN; ++row) {
                    float gate_ref = 0.0f;
                    float up_ref = 0.0f;
                    for (int64_t col = 0; col < N_INPUT; ++col) {
                        const size_t wi = col + N_INPUT*(row + N_HIDDEN*expert);
                        const float x = input_data[col + N_INPUT*token];
                        gate_ref += gate_data[wi] * x;
                        up_ref += up_data[wi] * x;
                    }
                    gate_ref = std::min(gate_ref, SWIGLU_LIMIT);
                    up_ref = std::clamp(up_ref, -SWIGLU_LIMIT, SWIGLU_LIMIT);
                    hidden_ref[row] = silu(gate_ref) * up_ref;
                }
                for (int64_t row = 0; row < N_OUTPUT; ++row) {
                    float value = 0.0f;
                    for (int64_t col = 0; col < N_HIDDEN; ++col) {
                        value += down_data[col + N_HIDDEN*(row + N_OUTPUT*expert)] * hidden_ref[col];
                    }
                    expected[row + N_OUTPUT*token] += route_data[slot + N_EXPERT_USED*token] * value;
                }
            }
        }

        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size()*sizeof(float));
        ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size()*sizeof(int32_t));
        ggml_backend_tensor_set(route, route_data.data(), 0, route_data.size()*sizeof(float));
        ggml_backend_tensor_set(shared, shared_data.data(), 0, shared_data.size()*sizeof(float));

        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        std::vector<float> actual(expected.size());
        if (status == GGML_STATUS_SUCCESS) {
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size()*sizeof(float));
        } else {
            std::fprintf(stderr, "run=%lld N=%lld Meta MoE chain failed: %s\n",
                    (long long) run, (long long) n_tokens, ggml_status_to_string(status));
            ok = false;
        }

        for (size_t i = 0; ok && i < actual.size(); ++i) {
            const float tolerance = 2e-5f * std::max(1.0f, std::fabs(expected[i]));
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance) {
                std::fprintf(stderr, "run=%lld N=%lld mismatch[%zu]: got %.9g expected %.9g tol %.9g\n",
                        (long long) run, (long long) n_tokens, i, actual[i], expected[i], tolerance);
                ok = false;
            }
        }
    }

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(data_buffer);
    ggml_free(graph_ctx);
    ggml_free(data_ctx);
    split.weights = {};
    return ok;
}

} // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t cuda = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_dev_t rocm = ggml_backend_dev_by_name("ROCm0");
    if (!rocm) {
        rocm = ggml_backend_dev_by_name("HIP0");
    }
    ggml_backend_dev_t cpu = ggml_backend_dev_by_name("CPU");
    if (!cuda || !rocm || !cpu) {
        std::printf("SKIP: requires CUDA0, ROCm0, and CPU\n");
        return 77;
    }

    split_data split;
    ggml_backend_dev_t devices[3] = { cuda, rocm, cpu };
    ggml_backend_dev_t meta_device = ggml_backend_meta_device(devices, 3, get_split_state, &split);
    ggml_backend_t meta = ggml_backend_dev_init(meta_device, nullptr);
    if (!meta) {
        std::fprintf(stderr, "failed to initialize mixed Meta backend\n");
        return 1;
    }

    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(meta_device);
    constexpr int64_t N_GRAPH_BUILDS = 10;
    constexpr int64_t N_RUNS_PER_GRAPH = 2;
    bool all_ok = true;
    for (int64_t rebuild = 0; rebuild < N_GRAPH_BUILDS; ++rebuild) {
        const int64_t n_tokens = rebuild % 2 == 0 ? 1 : 3;
        all_ok = run_case(meta, buft, split, n_tokens, rebuild*N_RUNS_PER_GRAPH, N_RUNS_PER_GRAPH) && all_ok;
    }
    ggml_backend_free(meta);
    if (all_ok) {
        std::printf("PASS mixed CUDA0+ROCm0+CPU Meta MiMo-like shared-input MoE chain: 20 F32 runs, alternating N=1,3 rebuilds\n");
    }
    return all_ok ? 0 : 1;
}
