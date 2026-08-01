# Mixed CUDA and HIP operation

This branch adds a reproducible build mode for using an NVIDIA GPU and an AMD
MI100 in one llama.cpp process. The immediate goal is capacity: keep more model
weights and KV cache in VRAM for high-bit dense models and large MoEs, with
whole layers assigned to one backend or the other.

The implementation uses llama.cpp's dynamic backend loader. CUDA and HIP are
compiled as independent shared modules and loaded into the same process. No
NCCL/RCCL bridge is required. The gfx908 kernel changes remain in the HIP
module; the same source tree also produces the CUDA module.

## Build

The helper accepts the ROCm/CUDA locations and target architectures through the
environment. The homelab's initial mixed target is an RTX 3090 (`sm_86`) plus
an MI100 (`gfx908`):

```sh
ROCM_ROOT=/home/llm/mi100/rocm-gfx908-7.15.0a20260720 \
CUDA_ROOT=/usr/local/cuda \
CUDA_ARCHS=86 \
HIP_ARCHS=gfx908 \
BUILD_DIR="$PWD/build-mixed" \
JOBS=8 \
./scripts/build-mixed-cuda-hip.sh
```

Use `CUDA_ARCHS='61;86'` only when the P40 must also be supported. Restricting
the production build to the installed architectures keeps compile time and
module size down.

The ROCm runtime path is embedded in the resulting modules. This is deliberate:
the custom gfx908 stack must load correctly under systemd and llama-swap without
depending on an interactive shell's `LD_LIBRARY_PATH`.

## Initial safe operating mode

Start with whole-layer placement:

```sh
build-mixed/bin/llama-server \
  --model /path/to/model.gguf \
  --device CUDA0,HIP0 \
  --split-mode layer \
  --gpu-layers all \
  --fit on \
  --fit-target 1024,1024
```

`--tensor-split` can be added once the desired placement is known. Its values
are proportions, not MiB. Capacity-proportional placement (`24,32`) is a useful
first probe, but it is not automatically the fastest arrangement. For models
that fit across the two GPUs, prefer filling the faster backend while retaining
enough headroom for KV cache and compute buffers, then place the remaining
contiguous layers on the other card.

Do not begin with `row` or experimental `tensor` split. Those modes assume
tighter cooperation and are not qualified across CUDA and HIP. Layer split
requires only small activation transfers at backend boundaries and keeps each
layer's weights and KV state local to its device.

## First-install qualification

Before adding a llama-swap profile:

1. Confirm `--list-devices` exposes both `CUDA0` and `HIP0` in one process.
2. Confirm `ldd libggml-cuda.so` resolves only the expected CUDA stack and
   `ldd libggml-hip.so` resolves the selected ROCm stack.
3. Run `test-backend-ops` for `ADD`, `MUL_MAT`, flash attention and the
   production quant types independently on CUDA0 and HIP0.
4. Run a model that fits on each card separately and compare deterministic
   output against the mixed layer-split run.
5. Run a model that cannot fit on either card alone to prove mixed capacity.
6. Capture startup tensor placement, per-device VRAM, PCIe traffic, PP/TG,
   power and temperatures.
7. Repeat with reversed run order and at least three placement ratios. Keep the
   same model, context, ubatch, prompt and generation length.

The first placement sweep should include:

- automatic fit with a fixed 1 GiB margin per GPU;
- capacity-proportional `24,32`;
- 3090-heavy placement; and
- MI100-heavy placement.

For MoEs, record actual expert tensor placement rather than inferring it from
the GGUF file size. Expert overrides may eventually outperform uniform layer
placement, but should follow the clean layer-split baseline.

## Current qualification state

The mixed build was compiled on 2026-08-01 with CUDA 12.8 (`sm_86`) and the
custom ROCm 7.15 development stack (`gfx908`). With the MI100 absent, the same
process loaded both backend modules, exposed the RTX 3090 and P40 through CUDA,
and handled the HIP module's no-device result without affecting CUDA. CUDA0
passed all 1,164 `MUL_MAT` correctness cases in `test-backend-ops`.

Physical CUDA+HIP execution remains intentionally unqualified until both cards
are installed. The existing HIP production build and llama-swap configuration
remain the rollback path.

## Likely follow-up work

The existing loader and layer scheduler may be sufficient for the capacity
target. If measurements show avoidable losses, the next useful work is narrow:

- architecture-aware automatic layer placement based on usable VRAM and
  measured per-layer latency;
- minimizing host-staged transfers at CUDA/HIP boundaries;
- explicit expert placement for partially offloaded MoEs; and
- separate target/draft placement when speculative models benefit from it.

A new cross-vendor collective backend is not justified unless layer placement
proves inadequate. The first hardware oracle should decide that question.
