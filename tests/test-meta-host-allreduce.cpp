#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int64_t K = 6;
constexpr int64_t M = 2;

struct split_data {
    const ggml_tensor * weight = nullptr;
    const ggml_tensor * input  = nullptr;
};

ggml_backend_meta_split_state get_split_state(const ggml_tensor * tensor, void * userdata) {
    const split_data * split = static_cast<const split_data *>(userdata);
    ggml_backend_meta_split_state state = {};
    state.nr[0] = 1;
    state.n_segments = 1;
    if (tensor == split->weight || tensor == split->input) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_0;
        state.ne[0] = 2;
        state.ne[1] = 2;
        state.ne[2] = 2;
    } else {
        state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    }
    return state;
}

} // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_name("CPU");
    if (!cpu) {
        std::fprintf(stderr, "CPU backend not found\n");
        return 1;
    }

    split_data split;
    std::array<ggml_backend_dev_t, 3> devices = { cpu, cpu, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devices.data(), devices.size(), get_split_state, &split);
    ggml_backend_t meta = ggml_backend_dev_init(meta_dev, nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(meta_dev);
    if (!meta || !buft) {
        std::fprintf(stderr, "failed to initialize Meta backend\n");
        return 1;
    }

    ggml_context * data_ctx = ggml_init({ 4 * ggml_tensor_overhead(), nullptr, true });
    ggml_context * graph_ctx = ggml_init({ 8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true });
    ggml_tensor * weight = ggml_new_tensor_2d(data_ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * input  = ggml_new_tensor_2d(data_ctx, GGML_TYPE_F32, K, 1);
    split.weight = weight;
    split.input = input;

    ggml_backend_buffer_t data_buffer = ggml_backend_alloc_ctx_tensors_from_buft(data_ctx, buft);
    ggml_tensor * output = ggml_mul_mat(graph_ctx, weight, input);
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, 8, false);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t alloc = ggml_gallocr_new(buft);
    if (!data_buffer || !alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
        std::fprintf(stderr, "failed to allocate Meta tensors\n");
        return 1;
    }

    bool ok = true;
    for (int run = 0; run < 4; ++run) {
        std::vector<float> weight_data(K * M);
        std::vector<float> input_data(K);
        for (int64_t row = 0; row < M; ++row) {
            for (int64_t col = 0; col < K; ++col) {
                weight_data[col + K * row] = 0.1f * float(1 + col + 2 * row + run);
            }
        }
        for (int64_t col = 0; col < K; ++col) {
            input_data[col] = 0.05f * float(1 + 2 * col + run);
        }

        ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
        if (ggml_backend_graph_compute(meta, graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "Meta compute failed\n");
            ok = false;
            break;
        }

        std::array<float, M> actual = {};
        ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
        for (int64_t row = 0; row < M; ++row) {
            float expected = 0.0f;
            for (int64_t col = 0; col < K; ++col) {
                expected += weight_data[col + K * row] * input_data[col];
            }
            if (std::fabs(actual[row] - expected) > 1e-5f) {
                std::fprintf(stderr, "run=%d row=%lld got %.9g expected %.9g\n",
                    run, (long long) row, actual[row], expected);
                ok = false;
            }
        }
    }

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(data_buffer);
    ggml_free(graph_ctx);
    ggml_free(data_ctx);
    ggml_backend_free(meta);
    return ok ? 0 : 1;
}
