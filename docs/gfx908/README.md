# llama.cpp for gfx908 / AMD Instinct MI100

This repository is a HIP-optimized llama.cpp branch for the AMD Instinct MI100
(`gfx908`, CDNA1). It is both:

1. a reproducible source tree for the build used in a single-user homelab; and
2. a record of what was profiled, changed, tested, rejected, and still needs
   work.

The project is independent of AMD and upstream llama.cpp. The changes are kept
close to gfx908 where practical. They work on the system described below, but
have not yet been reproduced on another MI100 and ROCm setup.

On the tested Qwen3.6-27B setup, favorable prefill throughput rose from roughly
700 tok/s on the first upstream run to about 1.3-1.5k tok/s after configuration
and code tuning. Decode gains vary more by model, quant and context length; the
tables below show where they came from.

## Repository state

- Upstream source base: `e8e6c7af2456fd50bb62f7a2bbd642e6fb14ae77`
- Daily-use source: the `gfx908-production` branch
- Production snapshot reconstructed from: `57530d9220648a8e96a10b0f1dfdc336faa9773b`
- Primary hardware: AMD Instinct MI100 32 GB, `gfx908:sramecc+:xnack-`
- Tested stack: custom ROCm 7.15 development build, clang 23
- Last full test date: 2026-07-27

The public history was reconstructed from the daily-use tree into five logical
change groups. The source and test changes were checked against that snapshot;
the reconstruction details are recorded in [ENGINEERING.md](ENGINEERING.md).

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
| Runtime rocBLAS GEMM autotuner | IQ4_NL pp4096 1160.4 → 1445.9 tok/s with a warm cache in the same-build global-off/on control; replaces fragile hand-selected solution IDs with measured per-shape selection |
| Combined Qwen IQ4_NL prefill stack | +22% to +27% at pp512-2048 in the original matched test |
| CDNA1 long-context flash-attention tile | approximately +5% PP at 32k and +9% at 64k in the isolated Qwen test |
| Native f32 FA accumulation | approximately +4.8% on a real 27.9k-token prefill; decode regression removed by width gating |
| Quantized FA KQ subgroup | attention kernel +11.8% at 32k and +12.5% at 64k; approximately +2-4% whole-model long-context TG |
| Q4_K/Q5_K branchless metadata decode stack | Q4_K_M whole-model TG +10.6%; Q5_K direct kernels +10.8-11.3% |
| IQ4_NL/Q6_K N=1 MMVQ geometry | approximately +2.4% IQ4_NL and +3.7% Q6_K in the matched shallow decode test |

Real llama-swap observations after the full stack reached roughly 1.3-1.4k
tok/s prefill at favorable Qwen3.6-27B prompt shapes with `-ub 4096`, around
50 tok/s for Q6_K MTP at low context, and around 58-61 tok/s for IQ4_NL MTP.
The first upstream service observations were 40.55 tok/s for Q6_K MTP and
51.07 tok/s for IQ4_NL MTP. Against those starting points, the later ranges are
roughly +23-28% and +14-19%, respectively. This is the practical change seen in
daily use, not a controlled MTP-only A/B: prompts, cache state and acceptance
rates differed between runs.

In practical terms, favorable Qwen prefill roughly doubled from the first stock
experience. Shallow non-speculative TG moved much less. The more transferable TG
gain appears at long context, where attention becomes a larger part of each
token: the quantized KQ subdivision reduced the attention kernel by 11.8-12.5%
at 32-64k and translated to approximately +2-4% whole-model no-spec TG. This is
separate from MTP acceptance and therefore applies to models without speculation.

## Model-level anchors

These server measurements show what the combined changes looked like in daily
use. They include backend and configuration changes, so they should not be read
as the effect of one kernel.

| Model | Earlier state | Improved state | Practical reading |
|---|---|---|---|
| Qwen3.6-27B Q6_K | stock PP around 714 tok/s with default ubatch; shallow no-spec TG around 27 tok/s | favorable service PP around 1.3-1.4k; low-context MTP around 50-52 tok/s | PP changed dramatically; shallow base TG improved much less |
| Qwen3.6-27B IQ4_NL | stock PP around 712 tok/s; shallow no-spec TG around 36.7 tok/s | favorable service PP around 1.4k; low-context MTP around 58-61 tok/s | fastest resident 27B profile; TG remains context- and acceptance-sensitive |
| Gemma4-31B, Q4_0-based | 459.7 PP / 45.55 TG in the matched old-server run; pp4096 510.1 on the old MMQ route | 738.2 PP / 45.68 TG at the first clean server checkpoint; later pp4096 1078.3, pp8192 1047.1 and pp32768 751.4 | Routing roughly doubled the direct prefill benchmark; later real use reached 757.4 PP at a ~20k prompt, about 45-46.5 TG shallow and 37.5 TG after that prefill |
| GPT-OSS-120B, CPU-offloaded MXFP4 MoE | 23.2 TG on plain mainline | about 35.7 TG in the matched migration; typically 36-38 warm | +54% measured TG, but CPU placement and contention remain part of the result |
| Step-3.7-Flash, heavily offloaded | 14.7 TG | 16.9 TG in the matched migration; typically 17-18 warm | about +15%; storage and CPU traffic cap the GPU-side benefit |

The first two rows compare the historical default invocation with later daily
use, so they show the practical journey rather than a code-only ratio. The next
MI100 test pass will add a same-settings upstream-versus-current row for Q6_K
and IQ4_NL.

## Start here

- [BUILD.md](BUILD.md) — build and runtime setup
- [STATUS.md](STATUS.md) — enabled, optional, experimental and rejected work
- [BENCHMARKS.md](BENCHMARKS.md) — test method and recorded measurements
- [UPSTREAM.md](UPSTREAM.md) — branch and update policy
- [ENGINEERING.md](ENGINEERING.md) — architecture findings and investigation map

The rest of the repository is the upstream llama.cpp source tree. Upstream
documentation remains authoritative for general llama.cpp usage.
