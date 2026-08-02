#pragma once

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <cuda.h>

struct gdn_chunk_sm86_modules {
    CUmodule   modules[5]   = {};
    CUfunction functions[5] = {};
};

enum gdn_chunk_sm86_kernel {
    GDN_SM86_CUMSUM,
    GDN_SM86_KKT_SOLVE,
    GDN_SM86_WU,
    GDN_SM86_H,
    GDN_SM86_O,
};

static void gdn_chunk_sm86_cu_check(CUresult result, const char * operation) {
    if (result == CUDA_SUCCESS) {
        return;
    }

    const char * name = "unknown";
    const char * description = "unknown";
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    GGML_ABORT("gdn chunk SM86: %s failed: %s (%s)", operation, name, description);
}

static gdn_chunk_sm86_modules & get_gdn_chunk_sm86_modules() {
    static gdn_chunk_sm86_modules result;
    static std::once_flag         once;

    std::call_once(once, [] {
        const char * env_dir = std::getenv("GGML_CUDA_GDN_CHUNK_SM86_DIR");
        const std::string dir = env_dir != nullptr
            ? env_dir
            : "/home/llm/mi100/triton-gdn-sm86";

        const char * files[5] = {
            "chunk_local_cumsum_scalar_kernel.cubin",
            "chunk_gated_delta_rule_fwd_kkt_solve_kernel.cubin",
            "recompute_w_u_fwd_kernel.cubin",
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64.cubin",
            "chunk_fwd_kernel_o.cubin",
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
            gdn_chunk_sm86_cu_check(cuModuleLoad(&result.modules[i], path.c_str()), "cuModuleLoad");
            gdn_chunk_sm86_cu_check(
                cuModuleGetFunction(&result.functions[i], result.modules[i], names[i]),
                "cuModuleGetFunction");
        }
    });

    return result;
}

template <typename dst_t>
static __global__ void gdn_chunk_sm86_convert_f32(const float * src, dst_t * dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (dst_t) src[i];
    }
}

static __global__ void gdn_chunk_sm86_convert_f16(const half * src, float * dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (float) src[i];
    }
}

static __global__ void gdn_chunk_sm86_pack_v_f32(
        const float * src, half * dst, size_t n, int64_t sv1, int64_t sv2) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const size_t col = i % 128;
        const size_t h   = (i / 128) % 48;
        const size_t t   = i / (128 * 48);
        dst[i] = (half) src[t * sv2 + h * sv1 + col];
    }
}

static void gdn_chunk_sm86_launch(
        CUfunction function,
        uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
        uint32_t block_x, uint32_t shared_bytes,
        cudaStream_t stream, void ** args) {
    gdn_chunk_sm86_cu_check(
        cuLaunchKernel(
            function,
            grid_x, grid_y, grid_z,
            block_x, 1, 1,
            shared_bytes, (CUstream) stream,
            args, nullptr),
        "cuLaunchKernel");
}

static bool try_launch_gdn_chunk_sm86(
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
    const char * enabled = std::getenv("GGML_CUDA_GDN_CHUNK_SM86");
    if (enabled == nullptr || std::atoi(enabled) == 0) {
        return false;
    }

    const int id = ggml_cuda_get_device();
    if (ggml_cuda_info().devices[id].cc != 860 ||
        S_v != 128 || H != 48 || H_k != 16 ||
        n_seqs != 1 || rq3 != 1 || n_tokens < 64 ||
        kda || keep_rs ||
        sq1 != 128 || sq2 != 16*128 || sq3 != n_tokens*16*128 ||
        sv1 != 128 || sv2 < 48*128 || sv3 != n_tokens*sv2 ||
        sb1 != 1 || sb2 != 48 || sb3 != n_tokens*48) {
        return false;
    }

    GGML_ASSERT(n_tokens <= std::numeric_limits<int32_t>::max());
    const uint32_t T  = (uint32_t) n_tokens;
    const uint32_t NT = (T + 63) / 64;

    const size_t qk_count   = (size_t) n_tokens * 16 * 128;
    const size_t vh_count   = (size_t) n_tokens * 48 * 128;
    const size_t gate_count = (size_t) n_tokens * 48;
    const size_t a_count    = (size_t) n_tokens * 48 * 64;
    const size_t h_count    = (size_t) NT * 48 * 128 * 128;

    ggml_cuda_pool_alloc<half>  q_h(ctx.pool(), qk_count);
    ggml_cuda_pool_alloc<half>  k_h(ctx.pool(), qk_count);
    ggml_cuda_pool_alloc<half>  v_h(ctx.pool(), vh_count);
    ggml_cuda_pool_alloc<float> g_cum(ctx.pool(), gate_count);
    ggml_cuda_pool_alloc<half>  a_h(ctx.pool(), a_count);
    ggml_cuda_pool_alloc<half>  w_h(ctx.pool(), vh_count);
    ggml_cuda_pool_alloc<half>  u_h(ctx.pool(), vh_count);
    ggml_cuda_pool_alloc<half>  h_h(ctx.pool(), h_count);
    ggml_cuda_pool_alloc<half>  v_new_h(ctx.pool(), vh_count);
    ggml_cuda_pool_alloc<half>  out_h(ctx.pool(), vh_count);

    constexpr int convert_threads = 256;
    gdn_chunk_sm86_convert_f32<<<
        (qk_count + convert_threads - 1) / convert_threads, convert_threads, 0, stream>>>(
        q, q_h.get(), qk_count);
    gdn_chunk_sm86_convert_f32<<<
        (qk_count + convert_threads - 1) / convert_threads, convert_threads, 0, stream>>>(
        k, k_h.get(), qk_count);
    gdn_chunk_sm86_pack_v_f32<<<
        (vh_count + convert_threads - 1) / convert_threads, convert_threads, 0, stream>>>(
        v, v_h.get(), vh_count, sv1, sv2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(a_h.get(), 0, a_count * sizeof(half), stream));

    auto & modules = get_gdn_chunk_sm86_modules();
    CUdeviceptr global_scratch  = 0;
    CUdeviceptr profile_scratch = 0;

    {
        const float * g_arg = g;
        float * gc_arg = g_cum.get();
        constexpr float rcp_ln2 = 1.4426950408889634f;
        void * args[] = {
            &g_arg, &gc_arg, (void *) &rcp_ln2, (void *) &T,
            &global_scratch, &profile_scratch
        };
        gdn_chunk_sm86_launch(
            modules.functions[GDN_SM86_CUMSUM],
            NT, 48, 1, 128, 8, stream, args);
    }
    {
        half * k_arg = k_h.get();
        float * gc_arg = g_cum.get();
        const float * beta_arg = beta;
        half * a_arg = a_h.get();
        void * args[] = {
            &k_arg, &gc_arg, &beta_arg, &a_arg, (void *) &T,
            &global_scratch, &profile_scratch
        };
        gdn_chunk_sm86_launch(
            modules.functions[GDN_SM86_KKT_SOLVE],
            NT, 48, 1, 64, 7168, stream, args);
    }
    {
        half * k_arg = k_h.get();
        half * v_arg = v_h.get();
        const float * beta_arg = beta;
        half * w_arg = w_h.get();
        half * u_arg = u_h.get();
        half * a_arg = a_h.get();
        float * gc_arg = g_cum.get();
        void * args[] = {
            &k_arg, &v_arg, &beta_arg, &w_arg, &u_arg, &a_arg, &gc_arg, (void *) &T,
            &global_scratch, &profile_scratch
        };
        gdn_chunk_sm86_launch(
            modules.functions[GDN_SM86_WU],
            NT, 48, 1, 128, 32768, stream, args);
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
        gdn_chunk_sm86_launch(
            modules.functions[GDN_SM86_H],
            4, 48, 1, 128, 12288, stream, args);
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
        gdn_chunk_sm86_launch(
            modules.functions[GDN_SM86_O],
            4, NT, 48, 64, 20480, stream, args);
    }

    gdn_chunk_sm86_convert_f16<<<
        (vh_count + convert_threads - 1) / convert_threads, convert_threads, 0, stream>>>(
        out_h.get(), dst, vh_count);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#endif
