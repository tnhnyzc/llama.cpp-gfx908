#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

void ggml_cuda_op_ssm_conv_l2_fused(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst,
    ggml_tensor * q_l2_dst, ggml_tensor * k_l2_dst);
