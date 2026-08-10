#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir=${GFX908_BUILD_DIR:-"${repo_dir}/build-prod"}
rocm_dir=${GFX908_ROCM_DIR:-/home/llm/mi100/rocm-gfx908-7.15.0a20260720}
gdn_runtime_dir="${build_dir}/runtime/gdn"

test -x "${build_dir}/bin/llama-server"
test -x "${build_dir}/bin/test-backend-ops"
test -s "${build_dir}/BUILD-MANIFEST.txt"

(
    cd "${gdn_runtime_dir}"
    sha256sum -c "${repo_dir}/scripts/gfx908/gdn-sha256.txt"
)

manifest_commit=$(sed -n 's/^source_commit=//p' "${build_dir}/BUILD-MANIFEST.txt")
git -C "${repo_dir}" merge-base --is-ancestor "${manifest_commit}" HEAD
git -C "${repo_dir}" diff --quiet "${manifest_commit}"..HEAD -- . \
    ':(exclude)docs/**' \
    ':(exclude)scripts/gfx908/validate-release.sh'

export LD_LIBRARY_PATH="${build_dir}/bin:${rocm_dir}/lib:${rocm_dir}/lib64:${rocm_dir}/llvm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${build_dir}/bin/llama-server" --version

if strings "${build_dir}/bin/libggml-hip.so.0" | grep -q '/llama.cpp-.*-2026.*/.*hsaco'; then
    echo "warning: HIP library contains a dated absolute HSACO path" >&2
    exit 1
fi

echo "gfx908 release validation passed: ${build_dir}"
