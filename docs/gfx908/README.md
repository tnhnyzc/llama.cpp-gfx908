# llama.cpp for gfx908 / AMD Instinct MI100

This repository is a HIP-optimized llama.cpp branch for the AMD Instinct MI100
(`gfx908`, CDNA1). It is both:

1. a reproducible source tree for the build used in a single-user homelab; and
2. a record of what was profiled, changed, tested, rejected, and still needs
   work.

The project is independent of AMD and upstream llama.cpp. The changes are kept
close to gfx908 where practical. They work on the system described below, but
have not yet been reproduced on another MI100 and ROCm setup.

On the tested Qwen3.6-27B setup, favorable prefill throughput is about
1.3-1.5k tok/s, compared with roughly 700 tok/s on the upstream default
configuration. Decode gains vary more by model, quant and context length.

## Repository state

- Upstream source base: `e8e6c7af2456fd50bb62f7a2bbd642e6fb14ae77`
- Daily-use source: the `gfx908-production` branch
- Daily-use snapshot reference: `57530d9220648a8e96a10b0f1dfdc336faa9773b`
- Primary hardware: AMD Instinct MI100 32 GB, `gfx908:sramecc+:xnack-`
- Tested stack: custom ROCm 7.15 development build, clang 23
- Last full test date: 2026-07-27

The source history is organized into five logical change groups and was checked
against the daily-use snapshot. Details are recorded in
[ENGINEERING.md](ENGINEERING.md).

## Change groups

| Group | Main purpose | Default state |
|---|---|---|
| Recurrent prefill | 64-token chunked gated-delta-rule path for the tested Qwen shape | Opt-in; external gfx908 HSACOs required |
| Quantized prefill/GEMM | wave64 dequantization, exact-shape rocBLAS selection/autotuning, concat and MMQ routing | Mixed: safe defaults plus opt-in exact-shape routes |
| Quantized decode | CDNA1 MMVQ geometry, DPP reductions, wider IQ4 loads, branchless Q4_K/Q5_K metadata decode | Enabled for tested gfx908 shapes |
| Flash attention | native f32 MFMA accumulation, long-context tiles, head-size routing and wave64 KQ subdivision | Enabled with rollback switches |
| Regression tests | production-shaped quantized matmul and flash-attention cases | Test-only |

See [STATUS.md](STATUS.md) for exact switches and tested boundaries.

## Performance highlights

Most rows below are matched A/B tests. Results still vary with model, prompt,
context and speculative acceptance; full commands are in
[BENCHMARKS.md](BENCHMARKS.md).

| Change | Result on MI100 |
|---|---|
| Chunked GDN, Qwen recurrent prefill | +12% at pp512 and about +19% at pp1024/2048 for Q6_K; similar transfer to Q4_K_M |
| Runtime rocBLAS GEMM autotuner | IQ4_NL pp4096 1160.4 → 1445.9 tok/s with a warm cache in the same-build global-off/on control; uses measured per-shape selection instead of fixed solution IDs |
| Combined Qwen IQ4_NL prefill stack | +22% to +27% at pp512-2048 in matched A/B tests |
| CDNA1 long-context flash-attention tile | approximately +5% PP at 32k and +9% at 64k in the isolated Qwen test |
| Native f32 FA accumulation | approximately +4.8% on a real 27.9k-token prefill, with width gating to preserve decode performance |
| Quantized FA KQ subgroup | attention kernel +11.8% at 32k and +12.5% at 64k; approximately +2-4% whole-model long-context TG |
| Q4_K/Q5_K branchless metadata decode stack | Q4_K_M whole-model TG +10.6%; Q5_K direct kernels +10.8-11.3% |
| IQ4_NL/Q6_K N=1 MMVQ geometry | approximately +2.4% IQ4_NL and +3.7% Q6_K in the matched shallow decode test |

With llama-swap, favorable Qwen3.6-27B prefill runs reached roughly 1.3-1.4k
tok/s with `-ub 4096`. Low-context MTP generation reached 50-52 tok/s for Q6_K
and 58-61 tok/s for IQ4_NL, versus upstream service baselines of 40.55 and
51.07 tok/s. That is a practical improvement of roughly +23-28% and +14-19%,
respectively. The MTP figures are service observations rather than a controlled
A/B because prompts, cache state and acceptance rates differed.

Favorable Qwen prefill is roughly twice the upstream default result. Shallow
non-speculative TG improved much less. At long context, the quantized KQ
subdivision reduced the attention kernel by 11.8-12.5% at 32-64k and improved
whole-model no-spec TG by approximately 2-4%. This path does not depend on MTP.

## Model-level anchors

These server measurements show what the combined changes looked like in daily
use. They include backend and configuration changes, so they should not be read
as the effect of one kernel.

| Model | Baseline | Current results | Summary |
|---|---|---|---|
| Qwen3.6-27B Q6_K | PP about 714 tok/s with default ubatch; shallow no-spec TG about 27 tok/s | favorable service PP 1.3-1.4k; low-context MTP 50-52 tok/s | Large PP gain; smaller shallow no-spec TG gain |
| Qwen3.6-27B IQ4_NL | PP about 712 tok/s; shallow no-spec TG about 36.7 tok/s | favorable service PP about 1.4k; low-context MTP 58-61 tok/s | Fastest resident 27B profile; TG varies with context and acceptance |
| Gemma4-31B, Q4_0-based | server PP 459.7 tok/s; pp4096 510.1 tok/s; shallow TG 45.55 tok/s | pp4096 1078.3, pp8192 1047.1 and pp32768 751.4 tok/s; real 20k-token prefill 757.4 tok/s; shallow TG 45-46.5 tok/s | CDNA1 batch routing roughly doubled direct prefill while preserving shallow TG |
| GPT-OSS-120B, CPU-offloaded MXFP4 MoE | 23.2 TG on upstream | 35.7 TG in the matched A/B; typically 36-38 warm | +54% in the matched test; CPU placement and contention remain relevant |
| Step-3.7-Flash, heavily offloaded | 14.7 TG | 16.9 TG in the matched A/B; typically 17-18 warm | About +15%; storage and CPU traffic limit the GPU-side gain |

The Qwen rows compare the upstream default invocation with tuned daily-use
settings. They show practical end-to-end performance, while the matched
code-level comparisons remain in [BENCHMARKS.md](BENCHMARKS.md).

## Start here

- [BUILD.md](BUILD.md) — build and runtime setup
- [STATUS.md](STATUS.md) — enabled, optional, experimental and rejected work
- [BENCHMARKS.md](BENCHMARKS.md) — test method and recorded measurements
- [UPSTREAM.md](UPSTREAM.md) — branch and update policy
- [ENGINEERING.md](ENGINEERING.md) — architecture findings and investigation map

The rest of the repository is the upstream llama.cpp source tree. Upstream
documentation remains authoritative for general llama.cpp usage.
