# llama.cpp for gfx908 / AMD Instinct MI100

This repository tracks a measured HIP optimization branch for the AMD Instinct
MI100 (`gfx908`, CDNA1). It serves two purposes:

1. a reproducible source tree for the build used in a single-user homelab; and
2. a public engineering record of the profiling, correctness checks, accepted
   changes, rejected experiments, and remaining bottlenecks.

The project is independent of AMD and upstream llama.cpp. Changes are narrowly
guarded where practical, but the branch should be treated as experimental until
it has been reproduced on more than one MI100 and ROCm stack.

## Repository state

- Upstream source base: `e8e6c7af2456fd50bb62f7a2bbd642e6fb14ae77`
- Qualified production source: the `gfx908-production` branch
- Production snapshot reconstructed from: `57530d9220648a8e96a10b0f1dfdc336faa9773b`
- Primary hardware: AMD Instinct MI100 32 GB, `gfx908:sramecc+:xnack-`
- Qualification stack: custom ROCm 7.15 development build, clang 23
- Last qualification date: 2026-07-27

The public history was reconstructed from the deployed tree into five logical
commits. All 15 modified source and test files were verified byte-for-byte
against the qualified snapshot. Unrelated files that had disappeared from an
old copied worktree were restored from upstream and are not part of this fork.

## Change groups

| Group | Main purpose | Default state |
|---|---|---|
| Recurrent prefill | 64-token chunked gated-delta-rule path for the tested Qwen shape | Opt-in; external gfx908 HSACOs required |
| Quantized prefill/GEMM | wave64 dequantization, exact-shape rocBLAS selection/autotuning, concat and MMQ routing | Mixed: safe defaults plus opt-in exact-shape routes |
| Quantized decode | CDNA1 MMVQ geometry, DPP reductions, wider IQ4 loads, branchless Q4_K/Q5_K metadata decode | Enabled for qualified gfx908 shapes |
| Flash attention | native f32 MFMA accumulation, long-context tiles, head-size routing and wave64 KQ subdivision | Enabled with rollback switches |
| Regression oracles | production-shaped quantized matmul and flash-attention cases | Test-only |

See [STATUS.md](STATUS.md) for exact switches and qualification boundaries.

## Measured highlights

These are isolated or matched A/B results, not a promise for every model or
prompt. Full commands and provenance belong in [BENCHMARKS.md](BENCHMARKS.md).

| Change | Qualified result on MI100 |
|---|---|
| Chunked GDN, Qwen recurrent prefill | +12% at pp512 and about +19% at pp1024/2048 for Q6_K; similar transfer to Q4_K_M |
| Runtime rocBLAS GEMM autotuner | IQ4_NL pp4096 1160.4 → 1445.9 tok/s with a warm cache in the same-build global-off/on control; replaces fragile hand-selected solution IDs with measured per-shape selection |
| Combined Qwen IQ4_NL prefill stack | +22% to +27% at pp512-2048 in the original matched oracle |
| CDNA1 long-context flash-attention tile | approximately +5% PP at 32k and +9% at 64k in the isolated Qwen oracle |
| Native f32 FA accumulation | approximately +4.8% on a real 27.9k-token prefill; decode regression removed by width gating |
| Quantized FA KQ subgroup | attention kernel +11.8% at 32k and +12.5% at 64k; approximately +2-4% whole-model long-context TG |
| Q4_K/Q5_K branchless metadata decode stack | Q4_K_M whole-model TG +10.6%; Q5_K direct kernels +10.8-11.3% |
| IQ4_NL/Q6_K N=1 MMVQ geometry | approximately +2.4% IQ4_NL and +3.7% Q6_K in the qualifying shallow decode oracle |

Real llama-swap observations after the full stack reached roughly 1.3-1.4k
tok/s prefill at favorable Qwen3.6-27B prompt shapes with `-ub 4096`, around
50 tok/s for Q6_K MTP at low context, and around 58-61 tok/s for IQ4_NL MTP.
Those service figures combine workload, cache and speculative-decoding effects;
they are retained as deployment observations rather than clean incremental A/Bs.

## Start here

- [BUILD.md](BUILD.md) — build and runtime setup
- [STATUS.md](STATUS.md) — enabled, opt-in, experimental and rejected work
- [BENCHMARKS.md](BENCHMARKS.md) — oracle rules and retained measurements
- [UPSTREAM.md](UPSTREAM.md) — branch and update policy
- [ENGINEERING.md](ENGINEERING.md) — architecture findings and investigation map

The rest of the repository is the upstream llama.cpp source tree. Upstream
documentation remains authoritative for general llama.cpp usage.
