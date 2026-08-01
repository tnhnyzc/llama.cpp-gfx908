#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root_dir}/build-mixed}"
rocm_root="${ROCM_ROOT:-/opt/rocm}"
cuda_root="${CUDA_ROOT:-/usr/local/cuda}"
cuda_archs="${CUDA_ARCHS:-86}"
hip_archs="${HIP_ARCHS:-gfx908}"
jobs="${JOBS:-$(nproc)}"

if [[ ! -x "${cuda_root}/bin/nvcc" ]]; then
    echo "CUDA compiler not found at ${cuda_root}/bin/nvcc" >&2
    exit 1
fi

hip_compiler="${rocm_root}/lib/llvm/bin/clang++"
if [[ ! -x "${hip_compiler}" ]]; then
    hip_compiler="${rocm_root}/bin/clang++"
fi
if [[ ! -x "${hip_compiler}" ]]; then
    echo "HIP compiler not found under ${rocm_root}" >&2
    exit 1
fi

rocm_rpath="${rocm_root}/lib;${rocm_root}/lib64"

cmake -S "${root_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU=ON \
    -DGGML_CUDA=ON \
    -DGGML_HIP=ON \
    -DGGML_RPC=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_CUDA_NCCL=OFF \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_MMQ_MFMA=ON \
    -DGGML_HIP_NO_VMM=ON \
    -DGGML_HIP_RCCL=OFF \
    -DGGML_HIP_ROCWMMA_FATTN=OFF \
    -DGGML_RPC_RDMA=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DCMAKE_CUDA_COMPILER="${cuda_root}/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_archs}" \
    -DCMAKE_HIP_COMPILER="${hip_compiler}" \
    -DCMAKE_HIP_ARCHITECTURES="${hip_archs}" \
    -DCMAKE_PREFIX_PATH="${rocm_root}" \
    -DCMAKE_BUILD_RPATH="${rocm_rpath}" \
    -DCMAKE_INSTALL_RPATH="${rocm_rpath}"

cmake --build "${build_dir}" --parallel "${jobs}"

echo
echo "Mixed backend build complete: ${build_dir}/bin"
"${build_dir}/bin/llama-cli" --list-devices
