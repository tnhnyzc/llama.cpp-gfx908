# Building for gfx908

## Release configuration

Use a fresh build directory and the qualified ROCm prefix:

```sh
ROCM_DIR=/path/to/rocm-gfx908

cmake -S . -B build-gfx908 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ROCM_DIR" \
  -DCMAKE_HIP_COMPILER="$ROCM_DIR/lib/llvm/bin/clang++" \
  -DCMAKE_HIP_ARCHITECTURES=gfx908 \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_NATIVE=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=ON

cmake --build build-gfx908 -j
```

At runtime, put the build's `bin` directory and the ROCm `lib`, `lib64`, and
LLVM library directories in `LD_LIBRARY_PATH`. The server should identify the
device as gfx908/CDNA1 and report the expected HIP graph/MMQ capabilities.

## External recurrent runtime

The chunked GDN route dynamically loads five qualified gfx908 HSACO files.
Package them under the release build, for example `runtime/gdn`, then set:

```sh
export GGML_HIP_GDN_CHUNK_GFX908=1
export GGML_HIP_GDN_CHUNK_GFX908_DIR=/path/to/build-gfx908/runtime/gdn
```

The normal llama.cpp recurrent path remains active when the feature is off or
the exact shape is ineligible. Record the HSACO and binary hashes in a build
manifest; do not make a production profile depend on an experiment directory.

## Minimum release gate

Before changing a serving profile:

1. record source commit, compiler/ROCm prefix, CMake options, and binary hashes;
2. run the covered ROCm backend tests, including production-shaped cases;
3. verify deterministic output and the expected runtime route census;
4. run matched, thermally controlled performance A/Bs where source changed;
5. smoke the affected real model and speculative route; and
6. deploy to a new dated directory while retaining the previous build/config.

The current qualified build and results are recorded in [STATUS.md](STATUS.md)
and [README.md](README.md). The dated
[QUALIFIED-B1-20260810.md](QUALIFIED-B1-20260810.md) page remains an earlier
same-parent checkpoint.
