#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int64_t N_EXPERTS     = 6;
constexpr int64_t N_EXPERT_USED = 6;
constexpr int64_t N_INPUT       = 4;
constexpr int64_t N_OUTPUT      = 2;

struct split_data {
    const ggml_tensor * weight = nullptr;
};

ggml_backend_meta_split_state get_split_state(const ggml_tensor * tensor, void * userdata) {
    const split_data * data = static_cast<const split_data *>(userdata);
    ggml_backend_meta_split_state state = {};
    state.nr[0] = 1;
    state.n_segments = 1;
    if (tensor == data->weight) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_2;
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
    setenv("GGML_META_HOST_ALLREDUCE", "1", 1);
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
    ggml_backend_dev_t expert_devs[3] = { cuda, rocm, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(expert_devs, 3, get_split_state, &split);
    ggml_backend_t cuda_backend = ggml_backend_dev_init(cuda, nullptr);
    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_dev_init(cpu, nullptr);
    if (!cuda_backend || !meta_backend || !cpu_backend) {
        std::fprintf(stderr, "failed to initialize direct CUDA, expert Meta, or scheduler CPU backend\n");
        if (cpu_backend) ggml_backend_free(cpu_backend);
        if (meta_backend) ggml_backend_free(meta_backend);
        if (cuda_backend) ggml_backend_free(cuda_backend);
        return 1;
    }

    const size_t static_size = 8 * ggml_tensor_overhead();
    const size_t graph_size = 48 * ggml_tensor_overhead() + ggml_graph_overhead_custom(48, false);
    ggml_context * meta_ctx = ggml_init({ static_size, nullptr, true });
    ggml_context * cuda_ctx = ggml_init({ static_size, nullptr, true });
    ggml_context * graph_ctx = ggml_init({ graph_size, nullptr, true });
    if (!meta_ctx || !cuda_ctx || !graph_ctx) {
        std::fprintf(stderr, "failed to create test contexts\n");
        return 1;
    }

    ggml_tensor * weight = ggml_new_tensor_3d(meta_ctx, GGML_TYPE_F32, N_INPUT, N_OUTPUT, N_EXPERTS);
    ggml_set_name(weight, "scheduler_ffn_up_exps.weight");
    split.weight = weight;

    ggml_tensor * input = ggml_new_tensor_3d(cuda_ctx, GGML_TYPE_F32, N_INPUT, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_I32, N_EXPERT_USED, 1);
    ggml_tensor * shared = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, N_OUTPUT, 1);
    ggml_set_name(input, "scheduler_cuda_input");
    ggml_set_name(ids, "scheduler_cuda_ids");
    ggml_set_name(shared, "scheduler_cuda_shared");

    ggml_backend_buffer_t meta_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        meta_ctx, ggml_backend_dev_buffer_type(meta_dev));
    ggml_backend_buffer_t cuda_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        cuda_ctx, ggml_backend_dev_buffer_type(cuda));
    if (!meta_buffer || !cuda_buffer) {
        std::fprintf(stderr, "failed to allocate static test buffers\n");
        return 1;
    }
    ggml_backend_buffer_set_usage(meta_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_tensor * projected = ggml_mul_mat_id(graph_ctx, weight, input, ids);
    std::array<ggml_tensor *, N_EXPERT_USED> slots;
    for (int64_t slot = 0; slot < N_EXPERT_USED; ++slot) {
        slots[slot] = ggml_view_2d(graph_ctx, projected, N_OUTPUT, 1, projected->nb[2], slot * projected->nb[1]);
    }
    ggml_tensor * sum = ggml_add(graph_ctx, slots[0], slots[1]);
    for (int64_t slot = 2; slot < N_EXPERT_USED; ++slot) {
        sum = ggml_add(graph_ctx, sum, slots[slot]);
    }
    ggml_tensor * materialized = ggml_add(graph_ctx, sum, shared);
    ggml_tensor * output = ggml_dup(graph_ctx, materialized);
    ggml_set_name(projected, "scheduler_expert_projected");
    ggml_set_name(sum, "scheduler_expert_sum");
    ggml_set_name(materialized, "scheduler_materialization_frontier");
    ggml_set_name(output, "scheduler_cuda_output");

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, 48, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backends[3] = { cuda_backend, meta_backend, cpu_backend };
    ggml_backend_buffer_type_t bufts[3] = {
        ggml_backend_dev_buffer_type(cuda), ggml_backend_dev_buffer_type(meta_dev),
        ggml_backend_dev_buffer_type(cpu)
    };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 3, 64, false, true);
    ggml_backend_sched_set_tensor_backend(sched, output, cuda_backend);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        std::fprintf(stderr, "scheduler failed to allocate mixed expert graph\n");
        return 1;
    }

    std::vector<float> weight_data(ggml_nelements(weight));
    for (int64_t expert = 0; expert < N_EXPERTS; ++expert) {
        for (int64_t row = 0; row < N_OUTPUT; ++row) {
            for (int64_t col = 0; col < N_INPUT; ++col) {
                weight_data[col + N_INPUT * (row + N_OUTPUT * expert)] =
                    0.01f * float(1 + col + 3 * row + 5 * expert);
            }
        }
    }
    const std::array<float, N_INPUT> input_data = { 0.25f, -0.5f, 0.75f, 1.0f };
    const std::array<int32_t, N_EXPERT_USED> ids_data = { 0, 2, 4, 1, 3, 5 };
    const std::array<float, N_OUTPUT> shared_data = { 0.125f, -0.25f };
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, sizeof(input_data));
    ggml_backend_tensor_set(ids, ids_data.data(), 0, sizeof(ids_data));
    ggml_backend_tensor_set(shared, shared_data.data(), 0, sizeof(shared_data));

    std::array<float, N_OUTPUT> expected = shared_data;
    for (int32_t expert : ids_data) {
        for (int64_t row = 0; row < N_OUTPUT; ++row) {
            for (int64_t col = 0; col < N_INPUT; ++col) {
                expected[row] += weight_data[col + N_INPUT * (row + N_OUTPUT * expert)] * input_data[col];
            }
        }
    }

    const ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
    std::array<float, N_OUTPUT> actual = {};
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
    }

    const bool routed_to_meta = ggml_backend_sched_get_tensor_backend(sched, projected) == meta_backend;
    const bool returned_to_cuda = ggml_backend_sched_get_tensor_backend(sched, output) == cuda_backend;
    bool ok = status == GGML_STATUS_SUCCESS && routed_to_meta && returned_to_cuda;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 2e-5f * std::max(1.0f, std::fabs(expected[i]));
        ok = ok && std::isfinite(actual[i]) && std::fabs(actual[i] - expected[i]) <= tolerance;
        if (!ok) {
            std::fprintf(stderr, "mismatch[%zu]: got %.9g expected %.9g\n", i, actual[i], expected[i]);
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(cuda_buffer);
    ggml_backend_buffer_free(meta_buffer);
    ggml_free(graph_ctx);
    ggml_free(cuda_ctx);
    ggml_free(meta_ctx);
    ggml_backend_free(cpu_backend);
    ggml_backend_free(meta_backend);
    ggml_backend_free(cuda_backend);

    if (ok) {
        std::printf("PASS direct CUDA -> CUDA+ROCm+CPU expert Meta -> direct CUDA scheduler path\n");
    }
    return ok ? 0 : 1;
}
