#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#if defined(GGML_USE_HIP)

// Experimental gfx908/Qwen GDN prefill bridge.  The five kernels are fixed-shape
// FP16 Triton/FLA kernels compiled for gfx908 and are loaded only when explicitly
// enabled.  The ordinary recurrent kernel remains the default and handles every
// non-matching shape.
struct gdn_chunk_gfx908_modules {
    hipModule_t   modules[5]   = {};
    hipFunction_t functions[5] = {};
};

enum gdn_chunk_gfx908_kernel {
    GDN_CHUNK_CUMSUM,
    GDN_CHUNK_KKT_SOLVE,
    GDN_CHUNK_WU,
    GDN_CHUNK_H,
    GDN_CHUNK_O,
};

static gdn_chunk_gfx908_modules & get_gdn_chunk_gfx908_modules() {
    static gdn_chunk_gfx908_modules result;
    static std::once_flag           once;

    std::call_once(once, [] {
        const char * env_dir = std::getenv("GGML_HIP_GDN_CHUNK_GFX908_DIR");
        const std::string dir = env_dir != nullptr
            ? env_dir
            : "/home/llm/mi100/triton-gdn-gfx908-f16-vfirst-modulo";

        const char * files[5] = {
            "chunk_local_cumsum_scalar_kernel.hsaco",
            "chunk_gated_delta_rule_fwd_kkt_solve_kernel.hsaco",
            "recompute_w_u_fwd_kernel.hsaco",
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64.hsaco",
            "chunk_fwd_kernel_o.hsaco",
        };
        const char * names[5] = {
            "chunk_local_cumsum_scalar_kernel",
            "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
            "recompute_w_u_fwd_kernel",
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            "chunk_fwd_kernel_o",
        };

        for (int i = 0; i < 5; ++i) {
            const std::string path = dir + "/" + files[i];
            if (std::getenv("GGML_HIP_GDN_CHUNK_GFX908_DEBUG") != nullptr) {
                std::fprintf(stderr, "gdn-chunk loading %s\n", path.c_str());
            }
            CUDA_CHECK(hipModuleLoad(&result.modules[i], path.c_str()));
            CUDA_CHECK(hipModuleGetFunction(&result.functions[i], result.modules[i], names[i]));
            if (std::getenv("GGML_HIP_GDN_CHUNK_GFX908_DEBUG") != nullptr) {
                std::fprintf(stderr, "gdn-chunk loaded %s\n", names[i]);
            }
        }
    });

    return result;
}

template <typename dst_t>
static __global__ void gdn_chunk_convert_f32(const float * src, dst_t * dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (dst_t) src[i];
    }
}

static __global__ void gdn_chunk_convert_f16(const half * src, float * dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (float) src[i];
    }
}

static __global__ void gdn_chunk_pack_v_f16(
        const float * src, half * dst, size_t n, int64_t sv1, int64_t sv2) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const size_t col = i % 128;
        const size_t h   = (i / 128) % 48;
        const size_t t   = i / (128 * 48);
        dst[i] = (half) src[t * sv2 + h * sv1 + col];
    }
}

static void gdn_chunk_launch_module(
        const char * name,
        hipFunction_t function,
        uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
        uint32_t block_x, uint32_t shared_bytes,
        hipStream_t stream, void ** args) {
    const bool debug = std::getenv("GGML_HIP_GDN_CHUNK_GFX908_DEBUG") != nullptr;
    if (debug) {
        std::fprintf(stderr, "gdn-chunk launch %s grid=(%u,%u,%u) block=%u shared=%u\n",
            name, grid_x, grid_y, grid_z, block_x, shared_bytes);
    }
    CUDA_CHECK(hipModuleLaunchKernel(
        function,
        grid_x, grid_y, grid_z,
        block_x, 1, 1,
        shared_bytes, stream,
        args, nullptr));
    if (debug) {
        CUDA_CHECK(hipStreamSynchronize(stream));
        std::fprintf(stderr, "gdn-chunk complete %s\n", name);
    }
}

static bool try_launch_gdn_chunk_gfx908(
        ggml_backend_cuda_context & ctx,
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * initial_state,
        float * dst, float * final_state,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t H_k, int64_t rq3,
        bool kda, bool keep_rs, float scale, cudaStream_t stream) {
    const char * enabled = std::getenv("GGML_HIP_GDN_CHUNK_GFX908");
    if (enabled == nullptr || std::atoi(enabled) == 0) {
        return false;
    }

    const int id = ggml_cuda_get_device();
    if (std::getenv("GGML_HIP_GDN_CHUNK_GFX908_DEBUG") != nullptr) {
        static bool printed = false;
        if (!printed) {
            printed = true;
            std::fprintf(stderr,
                "gdn-chunk guard: cc=0x%x S=%lld H=%lld Hk=%lld T=%lld B=%lld rq3=%lld "
                "kda=%d keep=%d q=(%lld,%lld,%lld) v=(%lld,%lld,%lld) b=(%lld,%lld,%lld)\n",
                ggml_cuda_info().devices[id].cc,
                (long long) S_v, (long long) H, (long long) H_k,
                (long long) n_tokens, (long long) n_seqs, (long long) rq3,
                (int) kda, (int) keep_rs,
                (long long) sq1, (long long) sq2, (long long) sq3,
                (long long) sv1, (long long) sv2, (long long) sv3,
                (long long) sb1, (long long) sb2, (long long) sb3);
        }
    }
    if (ggml_cuda_info().devices[id].cc != GGML_CUDA_CC_CDNA1 ||
        S_v != 128 || H != 48 || H_k != 16 ||
        n_seqs != 1 || rq3 != 1 || n_tokens < 64 ||
        kda || keep_rs ||
        sq1 != 128 || sq2 != 16*128 || sq3 != n_tokens*16*128 ||
        sv1 != 128 || sv2 < 48*128 || sv3 != n_tokens*sv2 ||
        sb1 != 1 || sb2 != 48 || sb3 != n_tokens*48) {
        return false;
    }

    GGML_ASSERT(n_tokens <= std::numeric_limits<int32_t>::max());
    const int32_t T  = (int32_t) n_tokens;
    const uint32_t NT = (uint32_t) ((n_tokens + 63) / 64);

    const size_t qk_count    = (size_t) n_tokens * 16 * 128;
    const size_t vh_count    = (size_t) n_tokens * 48 * 128;
    const size_t gate_count  = (size_t) n_tokens * 48;
    const size_t a_count     = (size_t) n_tokens * 48 * 64;
    const size_t h_count     = (size_t) NT * 48 * 128 * 128;

    ggml_cuda_pool_alloc<half>  q_h(ctx.pool(), qk_count);
    ggml_cuda_pool_alloc<half>  k_h(ctx.pool(), qk_count);
    ggml_cuda_pool_alloc<half>  v_h(ctx.pool(), vh_count);
    ggml_cuda_pool_alloc<float> g_cum(ctx.pool(), gate_count);
    ggml_cuda_pool_alloc<half>  a_inv(ctx.pool(), a_count);
    ggml_cuda_pool_alloc<half>  w_h(ctx.pool(), vh_count);
    ggml_cuda_pool_alloc<half>  u_h(ctx.pool(), vh_count);
    ggml_cuda_pool_alloc<half>  h_h(ctx.pool(), h_count);
    ggml_cuda_pool_alloc<half>  v_new_h(ctx.pool(), vh_count);
    ggml_cuda_pool_alloc<half>  out_h(ctx.pool(), vh_count);

    constexpr int convert_threads = 256;
    gdn_chunk_convert_f32<<<(qk_count + convert_threads - 1) / convert_threads, convert_threads, 0, stream>>>(
        q, q_h.get(), qk_count);
    gdn_chunk_convert_f32<<<(qk_count + convert_threads - 1) / convert_threads, convert_threads, 0, stream>>>(
        k, k_h.get(), qk_count);
    gdn_chunk_pack_v_f16<<<(vh_count + convert_threads - 1) / convert_threads, convert_threads, 0, stream>>>(
        v, v_h.get(), vh_count, sv1, sv2);
    CUDA_CHECK(cudaGetLastError());

    auto & modules = get_gdn_chunk_gfx908_modules();
    // Triton's AMD launcher appends these two implicit kernargs to every
    // generated kernel.  This exact-shape prototype uses neither scratch
    // region, matching Triton's zero-initialized launch path.
    hipDeviceptr_t global_scratch  = 0;
    hipDeviceptr_t profile_scratch = 0;

    {
        const float * g_arg = g;
        float * gc_arg = g_cum.get();
        const float cumsum_scale = 1.4426950408889634f;
        void * args[] = {
            &g_arg, &gc_arg, (void *) &cumsum_scale, (void *) &T,
            &global_scratch, &profile_scratch
        };
        gdn_chunk_launch_module("cumsum", modules.functions[GDN_CHUNK_CUMSUM],
            NT, 48, 1, 64, 0, stream, args);
    }
    CUDA_CHECK(cudaMemsetAsync(a_inv.get(), 0, a_count * sizeof(half), stream));
    {
        half * k_arg = k_h.get();
        float * gc_arg = g_cum.get();
        const float * beta_arg = beta;
        half * a_arg = a_inv.get();
        void * args[] = {
            &k_arg, &gc_arg, &beta_arg, &a_arg, (void *) &T,
            &global_scratch, &profile_scratch
        };
        gdn_chunk_launch_module("kkt-solve", modules.functions[GDN_CHUNK_KKT_SOLVE],
            NT, 48, 1, 64, 6144, stream, args);
    }
    {
        half * k_arg = k_h.get();
        half * v_arg = v_h.get();
        const float * beta_arg = beta;
        half * w_arg = w_h.get();
        half * u_arg = u_h.get();
        half * ai_arg = a_inv.get();
        float * gc_arg = g_cum.get();
        void * args[] = {
            &k_arg, &v_arg, &beta_arg, &w_arg, &u_arg, &ai_arg, &gc_arg, (void *) &T,
            &global_scratch, &profile_scratch
        };
        gdn_chunk_launch_module("wu", modules.functions[GDN_CHUNK_WU],
            NT, 48, 1, 256, 16384, stream, args);
    }
    {
        half * k_arg = k_h.get();
        half * u_arg = u_h.get();
        half * w_arg = w_h.get();
        half * vn_arg = v_new_h.get();
        float * gc_arg = g_cum.get();
        half * h_arg = h_h.get();
        const float * h0_arg = initial_state;
        float * ht_arg = final_state;
        void * args[] = {
            &k_arg, &u_arg, &w_arg, &vn_arg, &gc_arg, &h_arg, &h0_arg, &ht_arg, (void *) &T,
            &global_scratch, &profile_scratch
        };
        gdn_chunk_launch_module("h", modules.functions[GDN_CHUNK_H],
            4, 48, 1, 128, 8192, stream, args);
    }
    {
        half * q_arg = q_h.get();
        half * k_arg = k_h.get();
        half * vn_arg = v_new_h.get();
        half * h_arg = h_h.get();
        float * gc_arg = g_cum.get();
        half * out_arg = out_h.get();
        void * args[] = {
            &q_arg, &k_arg, &vn_arg, &h_arg, &gc_arg, &out_arg, &scale, (void *) &T,
            &global_scratch, &profile_scratch
        };
        gdn_chunk_launch_module("o", modules.functions[GDN_CHUNK_O],
            1, NT, 48, 512, 32768, stream, args);
    }

    gdn_chunk_convert_f16<<<(vh_count + convert_threads - 1) / convert_threads, convert_threads, 0, stream>>>(
        out_h.get(), dst, vh_count);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#endif // defined(GGML_USE_HIP)

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        // beta and v[col] are wave-uniform.  Loading them in every lane
        // creates 64 redundant global loads on gfx908.
        float beta_val = lane == 0 ? *beta_t : 0.0f;
        beta_val = __shfl_sync(0xffffffff, beta_val, 0, warp_size);
        float v_col = lane == 0 ? v_t[col] : 0.0f;
        v_col = __shfl_sync(0xffffffff, v_col, 0, warp_size);

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            // g is scalar for the whole head/token in the non-KDA path.
            // Compute exp only once per wave instead of once per lane.
            float g_val = lane == 0 ? expf(*g_t) : 0.0f;
            g_val = __shfl_sync(0xffffffff, g_val, 0, warp_size);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_col - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_col - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

#if defined(GGML_USE_HIP)
    // Retained-state mode (used by speculative decoding) needs K rollback
    // snapshots, while the chunk kernel returns only its final state.  Process
    // the large prefix with the chunk kernel, then run a recurrent correction
    // tail to populate every rollback slot.  A tail larger than K can be used
    // to damp FP16 chunk-state drift before generation begins.
    if (keep_rs && !kda && n_seqs == 1 && n_tokens > K) {
        int64_t tail_tokens = K;
        if (const char * tail_env = std::getenv("GGML_HIP_GDN_CHUNK_GFX908_RS_TAIL")) {
            const int64_t requested_tail = std::atoll(tail_env);
            if (requested_tail > tail_tokens) {
                tail_tokens = requested_tail;
            }
        }
        if (tail_tokens > n_tokens) {
            tail_tokens = n_tokens;
        }
        const int64_t prefix_tokens = n_tokens - tail_tokens;

        if (prefix_tokens >= 64) {
            ggml_cuda_pool_alloc<float> prefix_state(
                ctx.pool(), (size_t) S_v * S_v * H * n_seqs);

            if (try_launch_gdn_chunk_gfx908(
                    ctx, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, prefix_state.get(),
                    S_v, H, prefix_tokens, n_seqs,
                    sq1, sq2, prefix_tokens * sq2,
                    sv1, sv2, prefix_tokens * sv2,
                    sb1, sb2, prefix_tokens * sb2,
                    neqk1, rq3, kda, false, scale, stream)) {
                const float * q_tail = q_d + prefix_tokens * sq2;
                const float * k_tail = k_d + prefix_tokens * sq2;
                const float * v_tail = v_d + prefix_tokens * sv2;
                const float * g_tail = g_d + prefix_tokens * sb2;
                const float * b_tail = b_d + prefix_tokens * sb2;
                float * dst_tail = dst_d + prefix_tokens * S_v * H;

                if (std::getenv("GGML_HIP_GDN_CHUNK_GFX908_DEBUG") != nullptr) {
                    std::fprintf(stderr,
                        "gdn-chunk retained-state split: prefix=%lld tail=%lld K=%d\n",
                        (long long) prefix_tokens, (long long) tail_tokens, K);
                }

                launch_gated_delta_net<false, true>(
                    q_tail, k_tail, v_tail, g_tail, b_tail, prefix_state.get(),
                    dst_tail, state_d,
                    S_v, H, tail_tokens, n_seqs,
                    sq1, sq2, tail_tokens * sq2,
                    sv1, sv2, tail_tokens * sv2,
                    sb1, sb2, tail_tokens * sb2,
                    neqk1, rq3, scale, state_slot_stride, K, stream);
                return;
            }
        }
    }

    if (try_launch_gdn_chunk_gfx908(
            ctx, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
            neqk1, rq3, kda, keep_rs, scale, stream)) {
        return;
    }
#endif

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
