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

## Measurement practice

- Kernel timings are accepted only after complete output-coverage tests against
  a CPU reference. This excludes fast but incomplete tile geometries.
- VGPR pressure and register spilling are treated separately; spill claims
  require compiler metadata or observed scratch traffic.
- MoE results include verified GPU/CPU tensor placement rather than inferring
  execution surface from bytes in the model file.
- Whole-model gains are checked against kernel-share bounds and same-lineage
  controls, with reversed run order where variance matters.
- Profiler-adjusted kernel time is reported separately from wall time,
  dispatch overhead and synchronization gaps.

These rules are reflected in the regression tests and architecture guards.

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

### Decode memory-system result

A counter profile of the dominant IQ4_NL FFN MMVQ dispatch at
`m=17408, k=5120` read 50.1 MB from DRAM, matching the tensor's weight size,
in 63.8 us. That is 786 GB/s with a 14% L2 hit rate. The same profile recorded:

- zero `TCC_EA0_RDREQ_DRAM_CREDIT_STALL`;
- approximately 375 cycles of TCP-to-TCC read latency; and
- `TCP_PENDING_STALL_CYCLES` during 76.6% of measured TCC cycles.

For this kernel, the limiting behavior was outstanding-request capacity and
latency hiding rather than a saturated HBM controller. This makes the earlier
913-992 GB/s result a useful observed streaming range, not a hard MMVQ
bandwidth ceiling. It does not establish that every decode kernel, model or
backend has the same limit.

Static source and ISA analysis provides a plausible mechanism. Several compact
quant blocks have 17-, 18- or 34-byte strides and use load helpers that remain
safe at two-byte alignment; the measured gfx908 loop consequently contains
narrow weight loads. Wider IQ4_NL loads recovered approximately 4.8-6.0% in
the tested path, but a repacked or software-pipelined layout remains a
hypothesis rather than a validated general solution. The underlying block
layout is shared source; the measured performance impact is gfx908-specific.

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

- **Validated:** correctness, same-lineage A/B, reversed order where needed,
  and whole-model validation completed.
- **Staged:** direct-kernel correctness/performance is established but production
  breadth or quality validation is incomplete.
- **Hypothesis:** supported by static analysis or profiling but not yet tested by
  a discriminating intervention.
- **Rejected:** failed correctness, regressed, was neutral, or was confounded.

Future notes should use one of these labels and link the exact commit and
artifact. This repository is intended to make claims auditable rather than to
maximize the number of patches carried.
