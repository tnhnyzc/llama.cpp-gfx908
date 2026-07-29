# Building for gfx908

## Tested configuration

The daily-use build used Ubuntu in a Proxmox VM with the MI100 passed through,
a custom ROCm development stack, and:

```sh
cmake -S . -B build-gfx908 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx908 \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON

cmake --build build-gfx908 --config Release -j
```

When ROCm is not installed in a system path, prepend its `bin` directory to
`PATH` and pass its prefix through `CMAKE_PREFIX_PATH`. At runtime, ensure its
`lib`, `lib64`, LLVM library directory, and the build's `bin` directory are in
`LD_LIBRARY_PATH`.

Verify the target in startup output. The build should identify the AMD
device as CDNA1/gfx908 and report HIP flash attention and MMQ MFMA support.

## Optional recurrent prefill kernels

The chunked GDN path dynamically loads five precompiled gfx908 HSACO files. The
binaries are not yet shipped in this repository because their source and
license/provenance need to be packaged cleanly first.

Set both variables when using that path:

```sh
export GGML_HIP_GDN_CHUNK_GFX908=1
export GGML_HIP_GDN_CHUNK_GFX908_DIR=/absolute/path/to/gfx908-gdn-hsacos
```

Without `GGML_HIP_GDN_CHUNK_GFX908=1`, the normal llama.cpp recurrence remains
active. The route is guarded to the exact tested Qwen GDN dimensions and
falls back for other shapes.

## Checks before daily use

1. Build without warnings promoted to errors.
2. Run `test-backend-ops` for the complete HIP matmul and flash-attention sets.
3. Run the production-shaped cases added by this branch.
4. Compare deterministic logits/tokens against the upstream control.
5. Run perplexity on the same corpus and invocation for both builds.
6. Run reversed-order warm A/B performance tests.
7. Deploy through a new build directory and retain the previous binary/config
   as the rollback target.

The clean public history builds successfully for gfx908. It has not yet been
rerun through the full correctness and performance matrix on the MI100. That
rerun should happen before it replaces the existing daily build.
