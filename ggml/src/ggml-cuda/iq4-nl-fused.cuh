#pragma once

#include "common.cuh"

// Experimental exact-shape IQ4_NL x FP16 MFMA path for CDNA1/gfx908.
// This is deliberately narrow while the kernel is correctness/performance
// qualified in full llama.cpp workloads.

#if defined(GGML_USE_HIP)

using iq4_fused_halfx4_t  = __attribute__((ext_vector_type(4)))  _Float16;
using iq4_fused_halfx8_t  = __attribute__((ext_vector_type(8)))  _Float16;
using iq4_fused_floatx16_t = __attribute__((ext_vector_type(16))) float;
using iq4_fused_uintx4_t  = __attribute__((ext_vector_type(4)))  uint32_t;

static __device__ __forceinline__ uint32_t iq4_fused_lookup4_bias128(
        const uint32_t packed_q, const bool high_nibble) {
    // IQ4_NL's signed values, biased by 128, packed four per register.
    constexpr uint32_t lut0 = 0x3f2d1801;
    constexpr uint32_t lut1 = 0x766a5d4f;
    constexpr uint32_t lut2 = 0xa6998d81;
    constexpr uint32_t lut3 = 0xf1d9c5b5;

    const uint32_t indices = (high_nibble ? packed_q >> 4 : packed_q) & 0x0f0f0f0f;
    const uint32_t lower = __builtin_amdgcn_perm(lut1, lut0, indices & 0x07070707);
    const uint32_t upper = __builtin_amdgcn_perm(lut3, lut2, indices & 0x07070707);
    const uint32_t select = 0x03020100 | ((indices & 0x08080808) >> 1);
    return __builtin_amdgcn_perm(upper, lower, select);
}

static __device__ __forceinline__ iq4_fused_halfx4_t iq4_fused_u8x4_to_scaled_half(
        const uint32_t packed, const _Float16 d) {
    constexpr uint32_t fp16_adder = 0x64646464;
    constexpr uint32_t select01 = 0x05010500;
    constexpr uint32_t select23 = 0x05030502;
    constexpr uint32_t magic = 0x64806480;

    uint32_t h01 = __builtin_amdgcn_perm(fp16_adder, packed, select01);
    uint32_t h23 = __builtin_amdgcn_perm(fp16_adder, packed, select23);
    asm volatile(
        "v_pk_add_f16 %0, %1, %2 neg_lo:[0,1] neg_hi:[0,1]"
        : "=v"(h01) : "v"(h01), "s"(magic));
    asm volatile(
        "v_pk_add_f16 %0, %1, %2 neg_lo:[0,1] neg_hi:[0,1]"
        : "=v"(h23) : "v"(h23), "s"(magic));

    iq4_fused_halfx4_t result;
    reinterpret_cast<uint32_t *>(&result)[0] = h01;
    reinterpret_cast<uint32_t *>(&result)[1] = h23;
    result *= d;
    return result;
}

template<bool tail>
__launch_bounds__(256, 1)
static __global__ void iq4_nl_fp16_mfma_gfx908_exact(
        const block_iq4_nl * __restrict__ x,
        const half2 * __restrict__ yh,
        float * __restrict__ dst,
        const int k, const int nrows, const int m) {
    constexpr int block_rows = 128;
    constexpr int block_cols = 128;
    constexpr int k_tile = 64;
    constexpr int smem_stride = 72;

    __shared__ _Float16 sh_ab[
        block_rows * smem_stride + block_cols * smem_stride];
    _Float16 * const sh_a = sh_ab;
    _Float16 * const sh_b = sh_ab + block_rows * smem_stride;

    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int tid = wave * 64 + lane;
    const int row0 = blockIdx.x * block_rows;
    const int col0 = tail ? (m / block_cols) * block_cols
                          : blockIdx.y * block_cols;
    const int wave_row = (wave & 1) * 32;
    const int wave_col = (wave >> 1) * 32;
    const int blocks_per_row = k / 32;

    iq4_fused_floatx16_t C[2][2] = {};

    auto load_raw_tile = [&](const int load_kt,
                             iq4_fused_halfx8_t * gb,
                             _Float16 & qd,
                             iq4_fused_uintx4_t & qraw) {
        const int r = tid >> 1;
        const int qb = tid & 1;
        const block_iq4_nl & b =
            x[(int64_t)(row0 + r) * blocks_per_row + load_kt / 32 + qb];
        qd = *reinterpret_cast<const _Float16 *>(&b.d);
        qraw = *reinterpret_cast<const iq4_fused_uintx4_t *>(b.qs);

        const _Float16 * yhh = reinterpret_cast<const _Float16 *>(yh);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int c = wave * 32 + lane / 8 + q * 8;
            const int kv = lane % 8;
            if constexpr (tail) {
                gb[q] = col0 + c < m
                    ? *reinterpret_cast<const iq4_fused_halfx8_t *>(
                        yhh + (int64_t)(col0 + c) * k + load_kt + kv * 8)
                    : iq4_fused_halfx8_t{};
            } else {
                gb[q] = *reinterpret_cast<const iq4_fused_halfx8_t *>(
                    yhh + (int64_t)(col0 + c) * k + load_kt + kv * 8);
            }
        }
    };

    auto decode_raw_group = [&](const _Float16 qd,
                                const iq4_fused_uintx4_t qraw,
                                iq4_fused_halfx8_t * ga,
                                const int group) {
        const int qword = (group & 1) * 2;
        const bool high = group >= 2;
        const uint32_t values0 = iq4_fused_lookup4_bias128(qraw[qword + 0], high);
        const uint32_t values1 = iq4_fused_lookup4_bias128(qraw[qword + 1], high);
        reinterpret_cast<iq4_fused_halfx4_t *>(&ga[group])[0] =
            iq4_fused_u8x4_to_scaled_half(values0, qd);
        reinterpret_cast<iq4_fused_halfx4_t *>(&ga[group])[1] =
            iq4_fused_u8x4_to_scaled_half(values1, qd);
    };

    iq4_fused_halfx8_t ga[4];
    iq4_fused_halfx8_t gb[4];
    _Float16 qd = {};
    iq4_fused_uintx4_t qraw = {};
    load_raw_tile(0, gb, qd, qraw);
#pragma unroll
    for (int group = 0; group < 4; ++group) {
        decode_raw_group(qd, qraw, ga, group);
    }

    const int r = tid >> 1;
    const int qb = tid & 1;
#pragma unroll
    for (int group = 0; group < 4; ++group) {
        *reinterpret_cast<iq4_fused_halfx8_t *>(
            &sh_a[r * smem_stride + qb * 32 + group * 8]) = ga[group];
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int c = wave * 32 + lane / 8 + q * 8;
        const int kv = lane % 8;
        *reinterpret_cast<iq4_fused_halfx8_t *>(
            &sh_b[c * smem_stride + kv * 8]) = gb[q];
    }
    __syncthreads();

    for (int kt = 0; kt < k; kt += k_tile) {
        iq4_fused_halfx8_t next_ga[4];
        iq4_fused_halfx8_t next_gb[4];
        _Float16 next_qd = {};
        iq4_fused_uintx4_t next_qraw = {};
        if (kt + k_tile < k) {
            load_raw_tile(kt + k_tile, next_gb, next_qd, next_qraw);
        }

        iq4_fused_halfx8_t A8[4][2];
        iq4_fused_halfx8_t B8[4][2];
        const int matrix_lane = lane & 31;
        const int k_lane = (lane >> 5) * 8;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int kk = ks * 16;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                A8[ks][i] = *reinterpret_cast<const iq4_fused_halfx8_t *>(
                    &sh_a[(wave_row + i * 64 + matrix_lane) * smem_stride + kk + k_lane]);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                B8[ks][j] = *reinterpret_cast<const iq4_fused_halfx8_t *>(
                    &sh_b[(wave_col + j * 64 + matrix_lane) * smem_stride + kk + k_lane]);
            }
        }

#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
            for (int phase = 0; phase < 2; ++phase) {
                iq4_fused_halfx4_t A[2];
                iq4_fused_halfx4_t B[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    A[i] = phase == 0
                        ? __builtin_shufflevector(A8[ks][i], A8[ks][i], 0, 1, 2, 3)
                        : __builtin_shufflevector(A8[ks][i], A8[ks][i], 4, 5, 6, 7);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    B[j] = phase == 0
                        ? __builtin_shufflevector(B8[ks][j], B8[ks][j], 0, 1, 2, 3)
                        : __builtin_shufflevector(B8[ks][j], B8[ks][j], 4, 5, 6, 7);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        C[i][j] = __builtin_amdgcn_mfma_f32_32x32x8f16(
                            A[i], B[j], C[i][j], 0, 0, 0);
                    }
                }

                if (kt + k_tile < k) {
                    decode_raw_group(next_qd, next_qraw, next_ga, ks * 2 + phase);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int ks = 2; ks < 4; ++ks) {
#pragma unroll
            for (int phase = 0; phase < 2; ++phase) {
                iq4_fused_halfx4_t A[2];
                iq4_fused_halfx4_t B[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    A[i] = phase == 0
                        ? __builtin_shufflevector(A8[ks][i], A8[ks][i], 0, 1, 2, 3)
                        : __builtin_shufflevector(A8[ks][i], A8[ks][i], 4, 5, 6, 7);
                }
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    B[j] = phase == 0
                        ? __builtin_shufflevector(B8[ks][j], B8[ks][j], 0, 1, 2, 3)
                        : __builtin_shufflevector(B8[ks][j], B8[ks][j], 4, 5, 6, 7);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        C[i][j] = __builtin_amdgcn_mfma_f32_32x32x8f16(
                            A[i], B[j], C[i][j], 0, 0, 0);
                    }
                }

                if (kt + k_tile < k) {
                    const int store_group = (ks - 2) * 2 + phase;
                    *reinterpret_cast<iq4_fused_halfx8_t *>(
                        &sh_a[r * smem_stride + qb * 32 + store_group * 8]) =
                            next_ga[store_group];

                    const int c = wave * 32 + lane / 8 + store_group * 8;
                    const int kv = lane % 8;
                    *reinterpret_cast<iq4_fused_halfx8_t *>(
                        &sh_b[c * smem_stride + kv * 8]) =
                            next_gb[store_group];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int l = 0; l < 16; ++l) {
                const int local_m = (l / 4) * 8 + (lane / 32) * 4 + l % 4;
                const int local_n = lane % 32;
                const int row = row0 + wave_row + i * 64 + local_m;
                const int col = col0 + wave_col + j * 64 + local_n;
                if constexpr (tail) {
                    if (col < m) {
                        dst[(int64_t)col * nrows + row] = C[i][j][l];
                    }
                } else {
                    dst[(int64_t)col * nrows + row] = C[i][j][l];
                }
            }
        }
    }
}

#endif // GGML_USE_HIP

static inline void ggml_cuda_iq4_nl_fused_gfx908(
        const void * x, const half * y, float * dst,
        const int k, const int nrows, const int m, cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    GGML_ASSERT(k % 64 == 0);
    GGML_ASSERT(nrows % 128 == 0);
    const int full_tiles = m / 128;
    if (full_tiles > 0) {
        iq4_nl_fp16_mfma_gfx908_exact<false><<<
            dim3(nrows / 128, full_tiles), dim3(64, 4), 0, stream>>>(
                static_cast<const block_iq4_nl *>(x),
                reinterpret_cast<const half2 *>(y),
                dst, k, nrows, m);
    }
    if (m % 128 != 0) {
        iq4_nl_fp16_mfma_gfx908_exact<true><<<
            dim3(nrows / 128, 1), dim3(64, 4), 0, stream>>>(
                static_cast<const block_iq4_nl *>(x),
                reinterpret_cast<const half2 *>(y),
                dst, k, nrows, m);
    }
#else
    GGML_UNUSED_VARS(x, y, dst, k, nrows, m, stream);
    GGML_ABORT("gfx908 IQ4_NL fused kernel was not compiled");
#endif
}
