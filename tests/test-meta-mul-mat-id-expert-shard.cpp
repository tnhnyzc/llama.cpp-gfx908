#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int64_t N_EXPERTS     = 4;
constexpr int64_t N_EXPERT_USED = 4;
constexpr int64_t N_OUTPUT      = 2;
constexpr int64_t F32_K         = 4;
constexpr int64_t IQ2_XXS_K     = 1024;

struct split_data {
    const ggml_tensor * weights = nullptr;
};

ggml_backend_meta_split_state get_split_state(const ggml_tensor * tensor, void * userdata) {
    const split_data * data = static_cast<const split_data *>(userdata);
    ggml_backend_meta_split_state state = {};
    state.nr[0] = 1;
    state.n_segments = 1;
    if (tensor == data->weights) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_2;
        state.ne[0] = N_EXPERTS / 2;
        state.ne[1] = N_EXPERTS - state.ne[0];
    } else {
        state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    }
    return state;
}

bool run_case(ggml_backend_t backend, ggml_backend_buffer_type_t buft, split_data & split,
        ggml_type type, int64_t n_tokens, bool production_input) {
    const int64_t k = type == GGML_TYPE_IQ2_XXS ? IQ2_XXS_K : F32_K;
    const int64_t n_input_channels = production_input ? 1 : N_EXPERT_USED;
    ggml_context * data_ctx = ggml_init({ 4 * ggml_tensor_overhead(), nullptr, true });
    ggml_context * graph_ctx = ggml_init({
        8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true });
    if (!data_ctx || !graph_ctx) {
        std::fprintf(stderr, "failed to create contexts\n");
        if (data_ctx)  ggml_free(data_ctx);
        if (graph_ctx) ggml_free(graph_ctx);
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_3d(data_ctx, type, k, N_OUTPUT, N_EXPERTS);
    ggml_tensor * input   = ggml_new_tensor_3d(data_ctx, GGML_TYPE_F32, k, n_input_channels, n_tokens);
    ggml_tensor * ids     = ggml_new_tensor_2d(data_ctx, GGML_TYPE_I32, N_EXPERT_USED, n_tokens);
    ggml_set_name(weights, "meta_expert_weights");
    ggml_set_name(input,   "meta_expert_input");
    ggml_set_name(ids,     "meta_global_ids");
    split.weights = weights;

    ggml_backend_buffer_t data_buffer = ggml_backend_alloc_ctx_tensors_from_buft(data_ctx, buft);
    if (!data_buffer) {
        std::fprintf(stderr, "%s N=%lld: failed to allocate Meta leaf tensors\n",
                ggml_type_name(type), (long long) n_tokens);
        ggml_free(graph_ctx);
        ggml_free(data_ctx);
        return false;
    }

    ggml_tensor * partial = ggml_mul_mat_id(graph_ctx, weights, input, ids);
    ggml_set_name(partial, "meta_partial");
    // A following mirrored node makes the PARTIAL MMID a materialization
    // boundary in the current public Meta scheduling contract.
    ggml_tensor * output = ggml_dup(graph_ctx, partial);
    ggml_set_name(output, "meta_reconstructed");
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, 8, false);
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t alloc = ggml_gallocr_new(buft);
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
        std::fprintf(stderr, "%s N=%lld: failed to allocate Meta graph\n",
                ggml_type_name(type), (long long) n_tokens);
        if (alloc) ggml_gallocr_free(alloc);
        ggml_backend_buffer_free(data_buffer);
        ggml_free(graph_ctx);
        ggml_free(data_ctx);
        return false;
    }

    std::vector<float> weights_f32(ggml_nelements(weights));
    for (int64_t expert = 0; expert < N_EXPERTS; ++expert) {
        for (int64_t row = 0; row < N_OUTPUT; ++row) {
            for (int64_t col = 0; col < k; ++col) {
                const size_t i = col + k * (row + N_OUTPUT * expert);
                const int centered = int((37*col + 53*row + 71*expert) % 251) - 125;
                weights_f32[i] = type == GGML_TYPE_IQ2_XXS
                        ? float(centered) / 64.0f + 0.03125f * float(1 + row + 2*expert)
                        : 0.125f * float(1 + col + 7*row + 19*expert);
            }
        }
    }

    std::vector<uint8_t> weights_q;
    std::vector<float> weights_ref = weights_f32;
    const void * weights_storage = weights_f32.data();
    size_t weights_size = weights_f32.size() * sizeof(float);
    if (type == GGML_TYPE_IQ2_XXS) {
        weights_q.resize(ggml_nbytes(weights));
        std::vector<float> importance(k);
        for (int64_t col = 0; col < k; ++col) {
            importance[col] = 1.0f + float((7*col) % 17) / 16.0f;
        }
        const size_t written = ggml_quantize_chunk(type, weights_f32.data(), weights_q.data(),
                0, N_OUTPUT * N_EXPERTS, k, importance.data());
        const ggml_type_traits * traits = ggml_get_type_traits(type);
        if (written != weights_q.size() || !traits || !traits->to_float) {
            std::fprintf(stderr, "IQ2_XXS quantization/dequantization setup failed\n");
            ggml_gallocr_free(alloc);
            ggml_backend_buffer_free(data_buffer);
            ggml_free(graph_ctx);
            ggml_free(data_ctx);
            return false;
        }
        const size_t row_size = ggml_row_size(type, k);
        for (int64_t row = 0; row < N_OUTPUT * N_EXPERTS; ++row) {
            traits->to_float(weights_q.data() + row * row_size, weights_ref.data() + row * k, k);
        }
        weights_storage = weights_q.data();
        weights_size = weights_q.size();
    }

    std::vector<float> input_data(ggml_nelements(input));
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t input_channel = 0; input_channel < n_input_channels; ++input_channel) {
            for (int64_t col = 0; col < k; ++col) {
                const size_t i = col + k * (input_channel + n_input_channels * token);
                if (type == GGML_TYPE_IQ2_XXS) {
                    const int q = col % 32 == 31 ? 127 : int((11*col + 17*input_channel + 23*token) % 127) - 63;
                    input_data[i] = float(q) * (float(1 + input_channel + 2*token) / 128.0f);
                } else {
                    input_data[i] = 0.25f + float(col) + 4.0f*float(input_channel) + 20.0f*float(token);
                }
            }
        }
    }

    const int32_t pattern[N_EXPERT_USED] = { 0, 2, 1, 3 };
    std::vector<int32_t> ids_data(ggml_nelements(ids));
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t slot = 0; slot < N_EXPERT_USED; ++slot) {
            ids_data[slot + N_EXPERT_USED*token] = pattern[(slot + token) % N_EXPERT_USED];
        }
    }

    std::vector<float> expected(ggml_nelements(output));
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t slot = 0; slot < N_EXPERT_USED; ++slot) {
            const int64_t expert = ids_data[slot + N_EXPERT_USED*token];
            for (int64_t row = 0; row < N_OUTPUT; ++row) {
                float sum = 0.0f;
                for (int64_t col = 0; col < k; ++col) {
                    const size_t wi = col + k * (row + N_OUTPUT*expert);
                    const int64_t input_channel = production_input ? 0 : slot;
                    const size_t xi = col + k * (input_channel + n_input_channels*token);
                    sum += weights_ref[wi] * input_data[xi];
                }
                expected[row + N_OUTPUT*(slot + N_EXPERT_USED*token)] = sum;
            }
        }
    }

    ggml_backend_tensor_set(weights, weights_storage, 0, weights_size);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size()*sizeof(float));
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size()*sizeof(int32_t));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    std::vector<float> actual(expected.size());
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size()*sizeof(float));
    }

    bool ok = status == GGML_STATUS_SUCCESS;
    const float rel_tol = type == GGML_TYPE_IQ2_XXS ? 2e-3f : 1e-6f;
    for (size_t i = 0; ok && i < actual.size(); ++i) {
        const float tol = rel_tol * std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tol) {
            std::fprintf(stderr, "%s N=%lld mismatch[%zu]: got %.9g expected %.9g tol %.9g\n",
                    ggml_type_name(type), (long long) n_tokens, i, actual[i], expected[i], tol);
            ok = false;
        }
    }
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s N=%lld Meta graph failed: %s\n", ggml_type_name(type),
                (long long) n_tokens, ggml_status_to_string(status));
    }

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(data_buffer);
    ggml_free(graph_ctx);
    ggml_free(data_ctx);
    split.weights = nullptr;
    return ok;
}

} // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t cuda = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_dev_t rocm = ggml_backend_dev_by_name("ROCm0");
    if (!rocm) rocm = ggml_backend_dev_by_name("HIP0");
    if (!cuda || !rocm) {
        std::printf("SKIP: requires both CUDA0 and ROCm0\n");
        return 77;
    }

    split_data split;
    ggml_backend_dev_t devices[2] = { cuda, rocm };
    ggml_backend_dev_t meta_device = ggml_backend_meta_device(devices, 2, get_split_state, &split);
    ggml_backend_t meta = ggml_backend_dev_init(meta_device, nullptr);
    if (!meta) {
        std::fprintf(stderr, "failed to initialize mixed Meta backend\n");
        return 1;
    }

    bool all_ok = true;
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(meta_device);
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_IQ2_XXS }) {
        for (bool production_input : { false, true }) {
            for (int64_t n_tokens : { INT64_C(1), INT64_C(3) }) {
                all_ok = run_case(meta, buft, split, type, n_tokens, production_input) && all_ok;
            }
        }
    }
    ggml_backend_free(meta);
    ggml_quantize_free();
    if (all_ok) {
        std::printf("PASS mixed CUDA0+ROCm0 Meta expert reconstruction: shared+slot input F32/IQ2_XXS N=1,3\n");
    }
    return all_ok ? 0 : 1;
}
