#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int64_t N_EXPERTS = 4;
constexpr int64_t N_USED    = 4;
constexpr int64_t K         = 4;
constexpr int64_t M         = 3;

struct split_data {
    const ggml_tensor * weights = nullptr;
};

ggml_backend_meta_split_state get_split_state(const ggml_tensor * tensor, void * userdata) {
    const split_data * split = static_cast<const split_data *>(userdata);
    ggml_backend_meta_split_state state = {};
    state.nr[0] = 1;
    state.n_segments = 1;
    if (tensor == split->weights) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_2;
        state.ne[0] = N_EXPERTS / 2;
        state.ne[1] = N_EXPERTS - state.ne[0];
    } else {
        state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    }
    return state;
}

bool run_case(ggml_backend_t backend, ggml_backend_buffer_type_t buft, split_data & split, bool mixed_fanout) {
    ggml_context * data_ctx = ggml_init({ 4 * ggml_tensor_overhead(), nullptr, true });
    ggml_context * graph_ctx = ggml_init({
        12 * ggml_tensor_overhead() + ggml_graph_overhead_custom(12, false), nullptr, true });
    if (!data_ctx || !graph_ctx) {
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_3d(data_ctx, GGML_TYPE_F32, K, M, N_EXPERTS);
    ggml_tensor * input   = ggml_new_tensor_3d(data_ctx, GGML_TYPE_F32, K, N_USED, 1);
    ggml_tensor * ids     = ggml_new_tensor_2d(data_ctx, GGML_TYPE_I32, N_USED, 1);
    split.weights = weights;

    ggml_backend_buffer_t data_buffer = ggml_backend_alloc_ctx_tensors_from_buft(data_ctx, buft);
    ggml_tensor * partial = ggml_mul_mat_id(graph_ctx, weights, input, ids);
    ggml_tensor * output = ggml_soft_max(graph_ctx, partial);
    ggml_tensor * expert_branch = mixed_fanout ? ggml_neg(graph_ctx, partial) : nullptr;
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, 12, false);
    if (expert_branch) {
        ggml_build_forward_expand(graph, expert_branch);
    }
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t alloc = ggml_gallocr_new(buft);
    if (!data_buffer || !alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
        if (alloc) ggml_gallocr_free(alloc);
        if (data_buffer) ggml_backend_buffer_free(data_buffer);
        ggml_free(graph_ctx);
        ggml_free(data_ctx);
        split.weights = nullptr;
        return false;
    }

    std::vector<float> weights_data(ggml_nelements(weights));
    std::vector<float> input_data(ggml_nelements(input));
    std::vector<int32_t> ids_data = { 0, 2, 1, 3 };
    for (size_t i = 0; i < weights_data.size(); ++i) {
        weights_data[i] = 0.03125f * float(1 + i % 29);
    }
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = 0.0625f * float(1 + i % 17);
    }
    ggml_backend_tensor_set(weights, weights_data.data(), 0, weights_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size() * sizeof(int32_t));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    bool ok = mixed_fanout ? status == GGML_STATUS_FAILED : status == GGML_STATUS_SUCCESS;
    if (!mixed_fanout && ok) {
        std::vector<float> actual(ggml_nelements(output));
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        for (int64_t slot = 0; slot < N_USED && ok; ++slot) {
            float logits[M];
            float max_logit = -INFINITY;
            for (int64_t row = 0; row < M; ++row) {
                logits[row] = 0.0f;
                const int64_t expert = ids_data[slot];
                for (int64_t col = 0; col < K; ++col) {
                    logits[row] += weights_data[col + K * (row + M * expert)] * input_data[col + K * slot];
                }
                max_logit = std::max(max_logit, logits[row]);
            }
            float denom = 0.0f;
            for (float logit : logits) {
                denom += std::exp(logit - max_logit);
            }
            for (int64_t row = 0; row < M; ++row) {
                const float expected = std::exp(logits[row] - max_logit) / denom;
                const float got = actual[row + M * slot];
                ok = std::fabs(got - expected) <= 1e-6f;
            }
        }
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
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        return 1;
    }

    split_data split;
    ggml_backend_dev_t devices[2] = { cpu, cpu };
    ggml_backend_dev_t meta_device = ggml_backend_meta_device(devices, 2, get_split_state, &split);
    ggml_backend_t meta = ggml_backend_dev_init(meta_device, nullptr);
    const ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(meta_device);
    const bool ok = meta && run_case(meta, buft, split, false) && run_case(meta, buft, split, true);
    if (meta) {
        ggml_backend_free(meta);
    }
    if (ok) {
        std::printf("PASS Meta expert-partial materialization and mixed-fanout rejection\n");
    }
    return ok ? 0 : 1;
}
