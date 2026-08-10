#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir=${GFX908_BUILD_DIR:-"${repo_dir}/build-prod"}
rocm_dir=${GFX908_ROCM_DIR:-/home/llm/mi100/rocm-gfx908-7.15.0a20260720}
gdn_source_dir=${GFX908_GDN_SOURCE_DIR:-/home/llm/mi100/triton-gdn-gfx908-f16-vfirst-modulo}
gdn_runtime_dir="${build_dir}/runtime/gdn"
jobs=${GFX908_BUILD_JOBS:-"$(nproc)"}

test -x "${rocm_dir}/lib/llvm/bin/clang++"
test -d "${gdn_source_dir}"

mkdir -p "${gdn_runtime_dir}"
(
    cd "${gdn_source_dir}"
    sha256sum -c "${repo_dir}/scripts/gfx908/gdn-sha256.txt"
)

while read -r _ file; do
    install -m 0644 "${gdn_source_dir}/${file}" "${gdn_runtime_dir}/${file}"
done < "${repo_dir}/scripts/gfx908/gdn-sha256.txt"

export PATH="${rocm_dir}/bin:${rocm_dir}/lib/llvm/bin:${PATH}"
export CMAKE_PREFIX_PATH="${rocm_dir}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

cmake --fresh -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${rocm_dir}" \
    -DGGML_HIP=ON \
    -DGGML_CUDA=OFF \
    -DCMAKE_HIP_ARCHITECTURES=gfx908 \
    -DGGML_NATIVE=ON \
    -DGGML_OPENMP=ON \
    -DGGML_BLAS=OFF

cmake --build "${build_dir}" --config Release -j "${jobs}" --target \
    llama-server llama-cli llama-bench test-backend-ops

(
    cd "${gdn_runtime_dir}"
    sha256sum -c "${repo_dir}/scripts/gfx908/gdn-sha256.txt"
)

{
    printf 'source_commit=%s\n' "$(git -C "${repo_dir}" rev-parse HEAD)"
    printf 'source_branch=%s\n' "$(git -C "${repo_dir}" branch --show-current)"
    printf 'built_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'rocm_dir=%s\n' "${rocm_dir}"
    printf 'hip_arch=gfx908\n'
    printf 'gdn_runtime_dir=%s\n' "${gdn_runtime_dir}"
    sha256sum "${build_dir}/bin/llama-server" "${build_dir}/bin/libggml-hip.so.0"
} > "${build_dir}/BUILD-MANIFEST.txt"

echo "gfx908 release ready: ${build_dir}"
