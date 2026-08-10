#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int64_t N_INPUT       = 4096;
constexpr int64_t N_HIDDEN      = 2048;
constexpr int64_t N_OUTPUT      = 4096;
constexpr int64_t N_EXPERTS     = 256;
constexpr int64_t N_EXPERT_USED = 8;
constexpr int64_t N_TOKENS      = 1;
constexpr float   SWIGLU_LIMIT  = 7.0f;

constexpr std::array<int64_t, 3> EXPERT_SPLIT = { 43, 101, 112 };
static_assert(EXPERT_SPLIT[0] + EXPERT_SPLIT[1] + EXPERT_SPLIT[2] == N_EXPERTS);

struct split_data {
    std::array<const ggml_tensor *, 3> weights = {};
};

struct benchmark_result {
    bool ok = false;
    double compute_mean_us = 0.0;
    double compute_p50_us  = 0.0;
    double compute_p90_us  = 0.0;
    double ids_mean_us     = 0.0;
    double total_mean_us   = 0.0;
};

ggml_backend_meta_split_state get_split_state(const ggml_tensor * tensor, void * userdata) {
    const split_data * data = static_cast<const split_data *>(userdata);
    ggml_backend_meta_split_state state = {};
    state.nr[0] = 1;
    state.n_segments = 1;
    if (std::find(data->weights.begin(), data->weights.end(), tensor) != data->weights.end()) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_2;
        for (size_t i = 0; i < EXPERT_SPLIT.size(); ++i) {
            state.ne[i] = EXPERT_SPLIT[i];
        }
    } else {
        state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    }
    return state;
}

double mean(const std::vector<double> & values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const size_t i = std::min(values.size() - 1,
            static_cast<size_t>(std::llround(q * double(values.size() - 1))));
    return values[i];
}

void set_cpu_threads(ggml_backend_t backend, int n_threads) {
    if (backend) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto set_n_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (set_n_threads) set_n_threads(backend, n_threads);
    }
}

std::array<int32_t, N_EXPERT_USED> ids_for_run(int run) {
    // 37 is coprime with 256. Rotating this nonuniform set eventually touches
    // every expert and prevents the 49.5 MiB active working set from becoming
    // an artificial warm-cache CPU benchmark.
    constexpr std::array<int32_t, N_EXPERT_USED> offsets = { 0, 17, 51, 83, 119, 151, 201, 239 };
    std::array<int32_t, N_EXPERT_USED> ids = {};
    const int32_t base = (37 * run) & 255;
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = (base + offsets[i]) & 255;
    }
    return ids;
}

benchmark_result run_benchmark(
        ggml_backend_t backend, ggml_backend_buffer_type_t buft,
        split_data * split, const char * label, int n_warmup, int n_runs) {
    benchmark_result result;
    const size_t data_ctx_size = 16 * ggml_tensor_overhead();
    const size_t graph_ctx_size = 96 * ggml_tensor_overhead() + ggml_graph_overhead_custom(96, false);
    ggml_context * data_ctx = ggml_init({ data_ctx_size, nullptr, true });
    ggml_context * graph_ctx = ggml_init({ graph_ctx_size, nullptr, true });
    if (!data_ctx || !graph_ctx) {
        std::fprintf(stderr, "%s: failed to create contexts\n", label);
        if (data_ctx)  ggml_free(data_ctx);
        if (graph_ctx) ggml_free(graph_ctx);
        return result;
    }

    ggml_tensor * gate_w = ggml_new_tensor_3d(data_ctx, GGML_TYPE_IQ2_XXS,
            N_INPUT, N_HIDDEN, N_EXPERTS);
    ggml_tensor * up_w = ggml_new_tensor_3d(data_ctx, GGML_TYPE_IQ2_XXS,
            N_INPUT, N_HIDDEN, N_EXPERTS);
    ggml_tensor * down_w = ggml_new_tensor_3d(data_ctx, GGML_TYPE_IQ2_XXS,
            N_HIDDEN, N_OUTPUT, N_EXPERTS);
    ggml_tensor * input = ggml_new_tensor_3d(data_ctx, GGML_TYPE_F32,
            N_INPUT, 1, N_TOKENS);
    ggml_tensor * ids = ggml_new_tensor_2d(data_ctx, GGML_TYPE_I32,
            N_EXPERT_USED, N_TOKENS);
    ggml_tensor * route = ggml_new_tensor_3d(data_ctx, GGML_TYPE_F32,
            1, N_EXPERT_USED, N_TOKENS);

    ggml_set_name(gate_w, "mimo_gate_exps");
    ggml_set_name(up_w,   "mimo_up_exps");
    ggml_set_name(down_w, "mimo_down_exps");
    ggml_set_name(input,  "mimo_shared_input");
    ggml_set_name(ids,    "mimo_top8_ids");
    ggml_set_name(route,  "mimo_route_weights");
    if (split) {
        split->weights = { gate_w, up_w, down_w };
    }

    ggml_backend_buffer_t data_buffer = ggml_backend_alloc_ctx_tensors_from_buft(data_ctx, buft);
    if (!data_buffer) {
        std::fprintf(stderr, "%s: failed to allocate %.1f MiB of logical expert tensors\n",
                label, double(ggml_nbytes(gate_w) + ggml_nbytes(up_w) + ggml_nbytes(down_w))/(1024.0*1024.0));
        ggml_free(graph_ctx);
        ggml_free(data_ctx);
        if (split) split->weights = {};
        return result;
    }

    ggml_tensor * gate = ggml_mul_mat_id(graph_ctx, gate_w, input, ids);
    ggml_tensor * up = ggml_mul_mat_id(graph_ctx, up_w, input, ids);
    ggml_tensor * gate_clamped = ggml_clamp(graph_ctx, gate, -INFINITY, SWIGLU_LIMIT);
    ggml_tensor * up_clamped = ggml_clamp(graph_ctx, up, -SWIGLU_LIMIT, SWIGLU_LIMIT);
    ggml_tensor * hidden = ggml_swiglu_split(graph_ctx, gate_clamped, up_clamped);
    ggml_tensor * down = ggml_mul_mat_id(graph_ctx, down_w, hidden, ids);
    ggml_tensor * weighted = ggml_mul(graph_ctx, down, ggml_repeat(graph_ctx, route, down));

    std::array<ggml_tensor *, N_EXPERT_USED> slots = {};
    for (int64_t slot = 0; slot < N_EXPERT_USED; ++slot) {
        slots[slot] = ggml_view_2d(graph_ctx, weighted, N_OUTPUT, N_TOKENS,
                weighted->nb[2], slot * weighted->nb[1]);
    }
    ggml_tensor * sum = ggml_add(graph_ctx, slots[0], slots[1]);
    for (int64_t slot = 2; slot < N_EXPERT_USED; ++slot) {
        sum = ggml_add(graph_ctx, sum, slots[slot]);
    }
    ggml_tensor * output = sum;
    ggml_set_name(output, "mimo_routed_ffn_output");

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, 96, false);
    for (ggml_tensor * slot : slots) {
        ggml_build_forward_expand(graph, slot);
    }
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t alloc = ggml_gallocr_new(buft);
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph)) {
        std::fprintf(stderr, "%s: failed to allocate graph\n", label);
        if (alloc) ggml_gallocr_free(alloc);
        ggml_backend_buffer_free(data_buffer);
        ggml_free(graph_ctx);
        ggml_free(data_ctx);
        if (split) split->weights = {};
        return result;
    }

    // A zero IQ2_XXS buffer has zero scales and is a valid deterministic
    // weight tensor. It preserves all weight loads/dispatches while making the
    // expected output exact zero and avoids a second 6+ GiB host allocation.
    const size_t one_weight_size = std::max({ ggml_nbytes(gate_w), ggml_nbytes(up_w), ggml_nbytes(down_w) });
    std::vector<uint8_t> weight_data(one_weight_size, 0);
    std::fprintf(stderr, "%s: uploading 3 x %.1f MiB logical expert tensors\n",
            label, double(one_weight_size)/(1024.0*1024.0));
    ggml_backend_tensor_set(gate_w, weight_data.data(), 0, ggml_nbytes(gate_w));
    ggml_backend_tensor_set(up_w,   weight_data.data(), 0, ggml_nbytes(up_w));
    ggml_backend_tensor_set(down_w, weight_data.data(), 0, ggml_nbytes(down_w));
    std::vector<uint8_t>().swap(weight_data);

    std::vector<float> input_data(ggml_nelements(input));
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = float(int(i % 127) - 63) / 128.0f;
    }
    std::array<float, N_EXPERT_USED> route_data = {};
    route_data.fill(1.0f / float(N_EXPERT_USED));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size()*sizeof(float));
    ggml_backend_tensor_set(route, route_data.data(), 0, route_data.size()*sizeof(float));

    std::vector<double> compute_us;
    std::vector<double> ids_us;
    std::vector<double> total_us;
    compute_us.reserve(n_runs);
    ids_us.reserve(n_runs);
    total_us.reserve(n_runs);

    using clock = std::chrono::steady_clock;
    bool ok = true;
    for (int run = -n_warmup; ok && run < n_runs; ++run) {
        const auto run_ids = ids_for_run(run + n_warmup);
        const auto t0 = clock::now();
        ggml_backend_tensor_set(ids, run_ids.data(), 0, sizeof(run_ids));
        const auto t1 = clock::now();
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const auto t2 = clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "%s: graph failed at run %d: %s\n",
                    label, run, ggml_status_to_string(status));
            ok = false;
            break;
        }
        if (run >= 0) {
            ids_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            compute_us.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
            total_us.push_back(std::chrono::duration<double, std::micro>(t2 - t0).count());
        }
    }

    std::array<float, N_OUTPUT> output_data = {};
    if (ok) {
        ggml_backend_tensor_get(output, output_data.data(), 0, sizeof(output_data));
        for (size_t i = 0; i < output_data.size(); ++i) {
            if (output_data[i] != 0.0f || std::signbit(output_data[i])) {
                std::fprintf(stderr, "%s: nonzero output[%zu] = %.9g\n", label, i, output_data[i]);
                ok = false;
                break;
            }
        }
    }

    if (ok) {
        result.ok = true;
        result.compute_mean_us = mean(compute_us);
        result.compute_p50_us = percentile(compute_us, 0.50);
        result.compute_p90_us = percentile(compute_us, 0.90);
        result.ids_mean_us = mean(ids_us);
        result.total_mean_us = mean(total_us);
        std::printf("RESULT %-17s compute_mean_us=%9.2f p50_us=%9.2f p90_us=%9.2f ids_mean_us=%7.2f total_mean_us=%9.2f runs=%d\n",
                label, result.compute_mean_us, result.compute_p50_us, result.compute_p90_us,
                result.ids_mean_us, result.total_mean_us, n_runs);
    }

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(data_buffer);
    ggml_free(graph_ctx);
    ggml_free(data_ctx);
    if (split) split->weights = {};
    return result;
}

benchmark_result run_direct(ggml_backend_dev_t device, const char * label,
        int n_warmup, int n_runs, int cpu_threads) {
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) {
        std::fprintf(stderr, "%s: backend initialization failed\n", label);
        return {};
    }
    set_cpu_threads(backend, cpu_threads);
    benchmark_result result = run_benchmark(backend,
            ggml_backend_get_default_buffer_type(backend), nullptr, label, n_warmup, n_runs);
    ggml_backend_free(backend);
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    const int n_runs = argc > 1 ? std::max(1, std::atoi(argv[1])) : 64;
    const int n_warmup = argc > 2 ? std::max(1, std::atoi(argv[2])) : 4;
    const int cpu_threads = argc > 3 ? std::max(1, std::atoi(argv[3])) : 10;
    const bool meta_only = argc > 4 && std::string(argv[4]) == "meta-only";

    ggml_backend_load_all();
    ggml_backend_dev_t cuda = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_dev_t rocm = ggml_backend_dev_by_name("ROCm0");
    if (!rocm) rocm = ggml_backend_dev_by_name("HIP0");
    ggml_backend_dev_t cpu = ggml_backend_dev_by_name("CPU");
    if (!cuda || !rocm || !cpu) {
        std::fprintf(stderr, "requires CUDA0, ROCm0, and CPU\n");
        return 77;
    }

    benchmark_result cuda_result;
    benchmark_result rocm_result;
    benchmark_result cpu_result;
    if (!meta_only) {
        cuda_result = run_direct(cuda, "CUDA0-full", n_warmup, n_runs, cpu_threads);
        rocm_result = run_direct(rocm, "ROCm0-full", n_warmup, n_runs, cpu_threads);
        cpu_result = run_direct(cpu, "CPU10-full", n_warmup, n_runs, cpu_threads);
    }

    split_data split;
    ggml_backend_dev_t devices[3] = { cuda, rocm, cpu };
    ggml_backend_dev_t meta_device = ggml_backend_meta_device(devices, 3, get_split_state, &split);
    ggml_backend_t meta = ggml_backend_dev_init(meta_device, nullptr);
    benchmark_result meta_result;
    if (meta) {
        set_cpu_threads(ggml_backend_meta_simple_backend(meta, 2), cpu_threads);
        const bool primary = std::getenv("GGML_META_HOST_REDUCE_PRIMARY") != nullptr;
        const bool skip_reduce = std::getenv("GGML_META_SKIP_TERMINAL_REDUCE") != nullptr;
        meta_result = run_benchmark(meta, ggml_backend_dev_buffer_type(meta_device), &split,
                skip_reduce ? "Meta-no-reduce" : (primary ? "Meta-primary" : "Meta-43-101-112"),
                n_warmup, n_runs);
        ggml_backend_free(meta);
    }

    const bool ok = meta_result.ok && (meta_only || (cuda_result.ok && rocm_result.ok && cpu_result.ok));
    if (ok && !meta_only) {
        const double serial_48_us = 8.0*cuda_result.compute_mean_us
                + 19.0*rocm_result.compute_mean_us + 21.0*cpu_result.compute_mean_us;
        const double parallel_48_us = 48.0*meta_result.compute_mean_us;
        std::printf("PROJECTION serial_8_19_21_ms=%.3f parallel_48_ms=%.3f routed_ffn_speedup=%.3fx saved_ms=%.3f\n",
                serial_48_us/1000.0, parallel_48_us/1000.0,
                serial_48_us/parallel_48_us, (serial_48_us - parallel_48_us)/1000.0);
    }
    ggml_quantize_free();
    return ok ? 0 : 1;
}
