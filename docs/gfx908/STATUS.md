# Optimization status

This file is the short source of truth for what is enabled, what must be opted
into, and what should not be mistaken for a production result.

## Qualified defaults

- CDNA1 wave64 dequantization for the covered quant types. Set
  `GGML_HIP_DEQUANT_WAVE64=0` to restore the upstream launch geometry.
- CDNA1 N=1 MMVQ rows-per-block selection: two rows for supported types and one
  row for Q5_K. Override with `GGML_HIP_MMVQ_ROWS_GFX908` for regression tests.
- CDNA1 DPP reductions scoped to MMVQ.
- IQ4_NL 16-byte N=1 load path.
- Q4_K/Q5_K branchless scale/min reconstruction and q8_1 block-sum reuse.
- CDNA1 flash-attention f32 MFMA accumulation for prefill-width tiles while
  retaining the smaller accumulator for latency-sensitive decode.
- Head-size 512 MFMA routing. Set `GGML_HIP_FATTN_MMA_HS512=0` to restore the
  upstream head-size bound.
- Long-context 32x2 FA tile at KV >= 8192. Set
  `GGML_HIP_FATTN_LONGCTX_GFX908=0` to disable it.
- CDNA1 quantized FA KQ subgroup width and the measured vector/MMA crossover.
  `GGML_HIP_FATTN_MMA_DECODE_THRESH` remains available for crossover sweeps.
- Q4_0/Q4_1/Q5_0/Q5_1 CDNA1 MMQ batch routing.

## Opt-in production routes

- `GGML_HIP_GEMM_AUTOTUNE_GFX908=1`: exact-shape rocBLAS solution tuning.
  Persist results with `GGML_HIP_GEMM_AUTOTUNE_CACHE=/path/to/cache.tsv`.
- `GGML_HIP_IQ4_NL_FUSED_GFX908=1`: exact Qwen FFN IQ4_NL prefill route for
  qualified M values. This is shape-specific, not a universal IQ4 kernel.
- `GGML_HIP_GDN_CHUNK_GFX908=1`: exact-shape chunked Qwen GDN prefill. Requires
  the external HSACO directory described in BUILD.md.
- `GGML_HIP_Q5_K_MMQ_N3_GFX908=1`: exact Q5_K speculative-width route.
- `GGML_HIP_DISABLE_GFX908_M2_GATE_FUSION=1`: disables the IQ4_NL M=2 gate/up
  fusion when the selected workload performs better without it.

## Measurement controls

- `GGML_HIP_FATTN_NCOLS_256` and `GGML_HIP_FATTN_NCOLS_512` override FA tile
  geometry for controlled sweeps; defaults are the qualified selectors.
- `GGML_HIP_GEMM_AUTOTUNE_DEBUG=1` and
  `GGML_HIP_GDN_CHUNK_GFX908_DEBUG=1` enable diagnostic logging.
- `GGML_HIP_CONCAT_TRANSPOSE_GFX908` controls the transposed concat path.

## Not production claims

- A standalone fused IQ4_NL FP16-MFMA feasibility kernel substantially reduced
  decode overhead but did not yet beat the selected Tensile schedule. It is not
  the same as a complete production fused kernel.
- Several invalid or incomplete MMQ geometries produced attractive numbers
  before failing full output-coverage tests. They are intentionally absent.
- Q4_0 wider-load experiments did not reproduce the IQ4_NL gain in the full
  model and are absent.
- Q6_K metadata deletion bounded that path at about 0.61% of its tested kernel;
  no corresponding rewrite is retained.
- MXFP4 and IQ2_S full-server probes on offloaded MoE models cannot isolate GPU
  MMVQ performance and are not treated as kernel conclusions.
- Broad compiler flags, XNACK targeting, larger MMVQ wave counts, and the tested
  manual software-prefetch variants were neutral or negative.

## Deployment layout

- `/home/llm/mi100/llama.cpp-gfx908` is the only production source tree.
- `/home/llm/mi100/llama.cpp-gfx908/build-prod` is the only production HIP
  build.
- `/home/llm/mi100/llama.cpp-gfx908-current` is the atomic deployment symlink.
- GDN HSACO files live inside `build-prod/runtime/gdn` and are verified against
  `scripts/gfx908/gdn-sha256.txt`.
- Experimental branches remain in Git; they do not require persistent
  worktrees or build products.

## Open portability work

- Package the chunked GDN kernel source and reproducible HSACO build process.
- Add CI that at least compiles the HIP/gfx908 target; hardware performance CI
  is not currently available.
- Rebase onto current upstream and repeat the complete oracle before changing
  the production branch.
- Re-check generic paths touched by the old production tree and add explicit
  CDNA1 guards wherever the optimization is not intended for CUDA/RDNA.
