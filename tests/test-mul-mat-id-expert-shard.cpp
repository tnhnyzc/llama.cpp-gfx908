#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

// Contract under test:
//
// A sharded MUL_MAT_ID stores a contiguous local expert range in src0 while
// src2 continues to contain global router IDs.  The offset maps global ID
// `expert_offset + local_id` to src0[:, :, local_id].  IDs outside the local
// range must not read src0 and must overwrite the corresponding dst slot with
// exact zero.
//
namespace {

// F32 stays deliberately tiny; IQ2_XXS requires a full 256-element block.
constexpr int64_t F32_K           = 4;
constexpr int64_t IQ2_XXS_K       = 1024;
constexpr int64_t M               = 2;
constexpr int64_t N_EXPERT_USED   = 4;
constexpr int64_t N_LOCAL_EXPERTS = 2;
constexpr int32_t EXPERT_OFFSET   = 2;
constexpr float   POISON          = 123.0f;

bool run_case(
        ggml_backend_t backend, const char * backend_name, ggml_type weight_type, int64_t n_tokens,
        bool expert_sharded = true, int64_t iq2_k = IQ2_XXS_K, bool production_input = false) {
    const int64_t k = weight_type == GGML_TYPE_IQ2_XXS ? iq2_k : F32_K;
    const int64_t n_input_channels = production_input ? 1 : N_EXPERT_USED;
    const char * type_name = ggml_type_name(weight_type);
    const char * layout_name = expert_sharded ? "sharded" : "legacy";
    const char * input_name = production_input ? "shared-input" : "slot-input";
    const size_t ctx_size = 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false);
    ggml_init_params params = {
        /* .mem_size   = */ ctx_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create ggml context\n");
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_3d(ctx, weight_type, k, M, N_LOCAL_EXPERTS);
    ggml_tensor * input   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_input_channels, n_tokens);
    ggml_tensor * ids     = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_EXPERT_USED, n_tokens);
    ggml_tensor * output  = ggml_mul_mat_id(ctx, weights, input, ids);

    ggml_set_name(weights, "local_expert_weights");
    ggml_set_name(input,   "expert_inputs");
    ggml_set_name(ids,     "global_expert_ids");
    ggml_set_name(output,  "shard_output");

    if (expert_sharded) {
        ggml_mul_mat_id_set_expert_offset(output, EXPERT_OFFSET);
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        std::fprintf(stderr, "failed to allocate test tensors\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<float> weights_data(ggml_nelements(weights));
    for (int64_t local_expert = 0; local_expert < N_LOCAL_EXPERTS; ++local_expert) {
        for (int64_t row = 0; row < M; ++row) {
            for (int64_t col = 0; col < k; ++col) {
                const size_t index = col + k * (row + M * local_expert);
                if (weight_type == GGML_TYPE_IQ2_XXS) {
                    const int centered = int((37*col + 53*row + 71*local_expert) % 251) - 125;
                    weights_data[index] = float(centered) / 64.0f + 0.03125f * float(1 + row + 2*local_expert);
                } else {
                    weights_data[index] = 1.0f + float(col) + 10.0f * float(row) + 100.0f * float(local_expert);
                }
            }
        }
    }

    std::vector<uint8_t> weights_quantized;
    std::vector<float> weights_reference = weights_data;
    const void * weights_storage = weights_data.data();
    size_t weights_storage_size = weights_data.size() * sizeof(float);
    if (weight_type == GGML_TYPE_IQ2_XXS) {
        weights_quantized.resize(ggml_nbytes(weights));
        std::vector<float> importance(k);
        for (int64_t col = 0; col < k; ++col) {
            importance[col] = 1.0f + float((7*col) % 17) / 16.0f;
        }

        const size_t written = ggml_quantize_chunk(
                weight_type, weights_data.data(), weights_quantized.data(), 0,
                M * N_LOCAL_EXPERTS, k, importance.data());
        if (written != weights_quantized.size()) {
            std::fprintf(stderr, "%s: quantization wrote %zu bytes, expected %zu\n",
                    backend_name, written, weights_quantized.size());
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            return false;
        }

        const ggml_type_traits * traits = ggml_get_type_traits(weight_type);
        if (traits == nullptr || traits->to_float == nullptr) {
            std::fprintf(stderr, "%s: %s has no public dequantizer\n", backend_name, type_name);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            return false;
        }

        const size_t row_size = ggml_row_size(weight_type, k);
        for (int64_t row = 0; row < M * N_LOCAL_EXPERTS; ++row) {
            traits->to_float(weights_quantized.data() + row * row_size,
                    weights_reference.data() + row * k, k);
        }
        weights_storage = weights_quantized.data();
        weights_storage_size = weights_quantized.size();
    }

    std::vector<float> input_data(ggml_nelements(input));
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t input_channel = 0; input_channel < n_input_channels; ++input_channel) {
            for (int64_t col = 0; col < k; ++col) {
                const size_t index = col + k * (input_channel + n_input_channels * token);
                if (weight_type == GGML_TYPE_IQ2_XXS) {
                    // Each 32-element block contains an exact q8 scale anchor.
                    // This keeps activation quantization error small and makes
                    // the dequantized-weight reference independently useful.
                    const int q = col % 32 == 31
                            ? 127
                            : int((11*col + 17*input_channel + 23*token) % 127) - 63;
                    const float scale = float(1 + input_channel + 2*token) / 128.0f;
                    input_data[index] = float(q) * scale;
                } else {
                    input_data[index] = 0.25f + float(col) + 4.0f * float(input_channel) + 20.0f * float(token);
                }
            }
        }
    }

    // Every token contains IDs below, inside, and above this shard's [2, 4)
    // range.  Rotate them for N > 1 to exercise all output slots.
    const int32_t sharded_ids_pattern[] = { 0, 2, 3, 5 };
    // Owned slots 1 and 2 map identically to the sharded pattern after
    // subtracting EXPERT_OFFSET. Other slots are valid filler for the legacy
    // graph, whose kernels intentionally do not accept foreign IDs.
    const int32_t legacy_ids_pattern[] = { 0, 0, 1, 1 };
    const int32_t * ids_pattern = expert_sharded ? sharded_ids_pattern : legacy_ids_pattern;
    std::vector<int32_t> ids_data(ggml_nelements(ids));
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t slot = 0; slot < N_EXPERT_USED; ++slot) {
            ids_data[slot + N_EXPERT_USED * token] = ids_pattern[(slot + token) % N_EXPERT_USED];
        }
    }

    std::vector<float> output_data(ggml_nelements(output), POISON);
    std::vector<float> expected(output_data.size(), 0.0f);
    std::vector<uint8_t> owned(output_data.size(), 0);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t slot = 0; slot < N_EXPERT_USED; ++slot) {
            const int32_t global_expert = ids_data[slot + N_EXPERT_USED * token];
            const int32_t expert_base = expert_sharded ? EXPERT_OFFSET : 0;
            if (global_expert < expert_base || global_expert >= expert_base + N_LOCAL_EXPERTS) {
                continue;
            }

            const int64_t local_expert = global_expert - expert_base;
            for (int64_t row = 0; row < M; ++row) {
                float sum = 0.0f;
                for (int64_t col = 0; col < k; ++col) {
                    const size_t wi = col + k * (row + M * local_expert);
                    const int64_t input_channel = production_input ? 0 : slot;
                    const size_t xi = col + k * (input_channel + n_input_channels * token);
                    sum += weights_reference[wi] * input_data[xi];
                }
                const size_t oi = row + M * (slot + N_EXPERT_USED * token);
                expected[oi] = sum;
                owned[oi] = 1;
            }
        }
    }

    ggml_backend_tensor_set(weights, weights_storage,     0, weights_storage_size);
    ggml_backend_tensor_set(input,   input_data.data(),   0, input_data.size()   * sizeof(float));
    ggml_backend_tensor_set(ids,     ids_data.data(),     0, ids_data.size()     * sizeof(int32_t));
    ggml_backend_tensor_set(output,  output_data.data(),  0, output_data.size()  * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: %s %s %s K=%lld MUL_MAT_ID graph failed for N=%lld: %s\n",
                backend_name, type_name, layout_name, input_name, (long long) k,
                (long long) n_tokens, ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

    bool ok = true;
    size_t n_mismatches = 0;
    for (size_t i = 0; i < output_data.size(); ++i) {
        const bool foreign_slot = owned[i] == 0;
        const float relative_tolerance = weight_type == GGML_TYPE_IQ2_XXS ? 2e-3f : 1e-6f;
        const float tolerance = foreign_slot ? 0.0f
                : relative_tolerance * std::max(1.0f, std::fabs(expected[i]));
        const bool mismatch = foreign_slot
                ? output_data[i] != 0.0f || std::signbit(output_data[i])
                : !std::isfinite(output_data[i]) || std::fabs(output_data[i] - expected[i]) > tolerance;
        if (mismatch) {
            std::fprintf(stderr,
                    "%s: %s %s %s K=%lld N=%lld mismatch at output[%zu]: got %.9g, expected %.9g (tolerance %.9g)\n",
                    backend_name, type_name, layout_name, input_name, (long long) k,
                    (long long) n_tokens, i, output_data[i], expected[i], tolerance);
            if (!foreign_slot) {
                const int64_t row = i % M;
                const int64_t slot_token = i / M;
                const int64_t slot = slot_token % N_EXPERT_USED;
                const int64_t token = slot_token / N_EXPERT_USED;
                std::fprintf(stderr, "  references by local expert:");
                for (int64_t local_expert = 0; local_expert < N_LOCAL_EXPERTS; ++local_expert) {
                    float candidate = 0.0f;
                    for (int64_t col = 0; col < k; ++col) {
                        const size_t wi = col + k * (row + M * local_expert);
                        const int64_t input_channel = production_input ? 0 : slot;
                        const size_t xi = col + k * (input_channel + n_input_channels * token);
                        candidate += weights_reference[wi] * input_data[xi];
                    }
                    std::fprintf(stderr, " local[%lld]=%.9g", (long long) local_expert, candidate);
                }
                std::fprintf(stderr, "\n");
            }
            ok = false;
            ++n_mismatches;
        }
    }
    if (!ok) {
        std::fprintf(stderr, "%s: %s %s %s K=%lld N=%lld had %zu/%zu mismatched values\n",
                backend_name, type_name, layout_name, input_name, (long long) k,
                (long long) n_tokens, n_mismatches, output_data.size());
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    struct backend_target {
        const char * label;
        std::array<const char *, 2> device_names;
    };

    const backend_target targets[] = {
        { "CPU",   { "CPU",   nullptr } },
        { "CUDA0", { "CUDA0", nullptr } },
        { "ROCm0", { "ROCm0", "HIP0"  } },
    };

    const char * backend_filter = argc > 1 ? argv[1] : nullptr;
    const int64_t n_filter = argc > 2 ? std::strtoll(argv[2], nullptr, 10) : 0;
    const char * layout_filter = argc > 3 ? argv[3] : "sharded";
    const bool expert_sharded = std::strcmp(layout_filter, "sharded") == 0;
    const int64_t iq2_k = argc > 4 ? std::strtoll(argv[4], nullptr, 10) : IQ2_XXS_K;
    if (n_filter < 0 || n_filter > 3 ||
            (!expert_sharded && std::strcmp(layout_filter, "legacy") != 0) ||
            iq2_k <= 0 || iq2_k % 256 != 0) {
        std::fprintf(stderr,
                "usage: %s [CPU|CUDA0|ROCm0] [1|2|3] [sharded|legacy] [IQ2_K divisible by 256]\n",
                argv[0]);
        return 2;
    }

    bool all_ok = true;
    int n_available = 0;
    int n_tested = 0;
    for (const backend_target & target : targets) {
        if (backend_filter != nullptr && std::strcmp(backend_filter, target.label) != 0) {
            continue;
        }

        ggml_backend_dev_t device = nullptr;
        for (const char * device_name : target.device_names) {
            if (device_name != nullptr && (device = ggml_backend_dev_by_name(device_name)) != nullptr) {
                break;
            }
        }

        if (device == nullptr) {
            std::printf("SKIP %s: backend is not available\n", target.label);
            continue;
        }
        ++n_available;

        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (backend == nullptr) {
            std::fprintf(stderr, "FAIL %s: backend initialization failed\n", target.label);
            all_ok = false;
            continue;
        }

        const char * device_name = ggml_backend_dev_name(device);
        bool backend_ok = true;
        for (bool production_input : { false, true }) {
            if (n_filter == 0 || n_filter == 1) {
                backend_ok = run_case(backend, device_name, GGML_TYPE_F32, 1,
                        expert_sharded, IQ2_XXS_K, production_input) && backend_ok;
                backend_ok = run_case(backend, device_name, GGML_TYPE_IQ2_XXS, 1,
                        expert_sharded, iq2_k, production_input) && backend_ok;
            }
            if (n_filter == 0 || n_filter == 2) {
                backend_ok = run_case(backend, device_name, GGML_TYPE_IQ2_XXS, 2,
                        expert_sharded, iq2_k, production_input) && backend_ok;
            }
            if (n_filter == 0 || n_filter == 3) {
                backend_ok = run_case(backend, device_name, GGML_TYPE_F32, 3,
                        expert_sharded, IQ2_XXS_K, production_input) && backend_ok;
                backend_ok = run_case(backend, device_name, GGML_TYPE_IQ2_XXS, 3,
                        expert_sharded, iq2_k, production_input) && backend_ok;
            }
        }
        ggml_backend_free(backend);

        ++n_tested;
        if (backend_ok) {
            if (n_filter == 0) {
                std::printf("PASS %s: %s shared+slot input F32 N=1,N=3; IQ2_XXS K=%lld N=1,N=2,N=3\n",
                        device_name, layout_filter, (long long) iq2_k);
            } else if (n_filter == 2) {
                std::printf("PASS %s: %s shared+slot input IQ2_XXS K=%lld N=2\n",
                        device_name, layout_filter, (long long) iq2_k);
            } else {
                std::printf("PASS %s: %s shared+slot input F32 and IQ2_XXS K=%lld N=%lld\n",
                        device_name, layout_filter, (long long) iq2_k, (long long) n_filter);
            }
        } else {
            std::fprintf(stderr, "FAIL %s: expert-shard contract\n", device_name);
            all_ok = false;
        }
    }

    std::printf("MUL_MAT_ID expert-shard summary: %d backend(s) tested\n", n_tested);
    ggml_quantize_free();
    if (backend_filter != nullptr && n_available == 0) {
        return 77;
    }
    return all_ok && n_tested > 0 ? 0 : 1;
}
