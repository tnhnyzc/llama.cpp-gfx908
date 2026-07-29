# Engineering record

## What the investigation established

The MI100 was healthy before source tuning: sustained memory tests reached
roughly 913-992 GB/s against 1.228 TB/s theoretical HBM bandwidth, power and
thermal behavior were stable, and ECC remained clean. Weak application results
therefore could not be attributed to a broken card or obviously unhealthy ROCm
installation.

Prompt processing and token generation required different work:

- Prompt processing exposed routing, recurrent-kernel, dequantization, GEMM
  selection and flash-attention gaps. Several shared or model-family changes
  produced substantial cumulative gains.
- Decode remained dominated by quantized matrix-vector kernels, with attention
  becoming increasingly important at long context. Aggregate bandwidth alone
  did not describe the limit: request width, wave64 dependencies, occupancy,
  memory-level parallelism, launch overhead and serial latency all mattered.

## Corrections that shaped the method

- A fast T256/I128 MMQ candidate failed complete output coverage. Its timings
  were discarded, and exact multi-tile CPU-reference cases became mandatory.
- High VGPR usage was initially discussed as spilling without scratch evidence.
  Later work separated occupancy pressure from actual spills.
- Whole-model MoE measurements were initially ranked by bytes in the GGUF,
  overlooking CPU-offloaded experts. GPU-kernel conclusions now require direct
  backend tests or verified tensor placement.
- Several attractive whole-model gains exceeded the measured kernel-share
  bound because controls came from different build lineages. Same-day controls,
  binary hashes and reversed ordering are now required.
- Profiler per-dispatch overhead materially inflated short-kernel durations.
  Corrected kernel time is kept separate from wall time and launch/sync gaps.

These corrections explain why the current branch includes focused regression
tests and conservative guards.

## Current bottleneck map

### Prompt processing

- Qwen recurrent GDN: addressed for the exact tested shape with chunking.
- Shared quantized FFN path: improved through wave64 dequantization and exact
  rocBLAS selection; further progress likely requires a Tensile-class fused
  quant-to-FP16 MFMA schedule.
- Flash attention: materially improved, especially at long context and head
  dimension 512.
- Data movement/concat: smaller but measurable remaining surface.

### Decode

- MMVQ is the dominant kernel family and approximately 66-74% of representative
  token time, depending on model and context.
- Q4_K/Q5_K metadata work had a real CDNA1 branch-divergence opportunity and is
  addressed here.
- IQ4_NL is much closer to its practical streaming limit; broad arithmetic
  deletion probes bounded remaining inner-loop arithmetic gains.
- Roughly 1,780 dispatches per representative token make launch and graph gaps
  relevant, but removing them is mostly an upstream graph/fusion problem rather
  than a simple gfx908 kernel switch.
- Long-context quantized attention had a separate wave64 subgroup gap and now
  gains approximately 2-4% whole-model TG at the measured depths.

## Highest-value future work

1. Rebase the clean stack onto current upstream and rerun the test matrix.
2. Package the chunked GDN source/assets reproducibly.
3. Compare CUDA and HIP with the same model, depth, speculation and profiler
   correction to isolate launch latency and memory-level parallelism.
4. Build a minimal persistent/shared-activation experiment only if dispatch
   traces confirm repeated identical activation quantization can be reused.
5. Continue the fused FP16-MFMA path by transplanting the block-owned quant
   decoder into a selected exact-shape Tensile-class schedule.
6. Treat Vulkan as a later independent backend port using HIP as the reference,
   rather than assuming HIP tuning transfers automatically.

## How results are labeled

- **Qualified:** correctness, same-lineage A/B, reversed order where needed,
  and whole-model validation completed.
- **Staged:** direct-kernel correctness/performance is established but production
  breadth or quality validation is incomplete.
- **Hypothesis:** supported by static analysis or profiling but not yet tested by
  a discriminating intervention.
- **Rejected:** failed correctness, regressed, was neutral, or was confounded.

Future notes should use one of these labels and link the exact commit and
artifact. This repository is intended to make claims auditable rather than to
maximize the number of patches carried.
