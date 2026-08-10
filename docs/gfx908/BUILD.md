# Building for gfx908

## Canonical release build

The daily MI100 deployment is built from this repository with one command:

```sh
scripts/gfx908/build-release.sh
```

The canonical output is `build-prod`. It contains the server, CLI, benchmark
and backend-test binaries, the exact GDN HSACO files used at runtime, and a
`BUILD-MANIFEST.txt` recording the source commit, ROCm prefix and binary
hashes. `scripts/gfx908/validate-release.sh` rejects an incomplete or stale
release before deployment.

llama-swap should reference only the stable `llama.cpp-gfx908-current` symlink
and `build-prod`; experimental worktrees and build directories are never
production dependencies.

## Tested configuration

The qualified build used Ubuntu in a Proxmox VM with the MI100 passed through,
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

Verify the target in startup output. A qualified build should identify the AMD
device as CDNA1/gfx908 and report HIP flash attention and MMQ MFMA support.

## Optional recurrent prefill kernels

The chunked GDN path dynamically loads five precompiled gfx908 HSACO files. The
public repository records their qualified hashes but does not publish the
binaries while their source and license/provenance are being packaged. The
release script verifies the local qualified set and copies it into
`build-prod/runtime/gdn`, making the deployed build self-contained.

Set both variables when using that path:

```sh
export GGML_HIP_GDN_CHUNK_GFX908=1
export GGML_HIP_GDN_CHUNK_GFX908_DIR=/path/to/llama.cpp-gfx908-current/build-prod/runtime/gdn
```

Without `GGML_HIP_GDN_CHUNK_GFX908=1`, the normal llama.cpp recurrence remains
active. The route is guarded to the exact qualified Qwen GDN dimensions and
falls back for other shapes.

## Minimum validation before daily use

1. Build without warnings promoted to errors.
2. Run `test-backend-ops` for the complete HIP matmul and flash-attention sets.
3. Run the production-shaped cases added by this branch.
4. Compare deterministic logits/tokens against the upstream control.
5. Run perplexity on the same corpus and invocation for both builds.
6. Run reversed-order warm A/B performance tests.
7. Deploy through a new build directory and retain the previous binary/config
   as the rollback target.

The clean public history has been rebuilt successfully for the gfx908 target.
It has not yet completed the hardware runtime and performance oracle on an
MI100. The next MI100 session should perform that final reproducibility fence
before replacing the existing daily build.
