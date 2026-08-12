# llama.cpp for gfx908 / AMD Instinct MI100

This is an independently maintained llama.cpp optimization fork for AMD
Instinct MI100 (`gfx908`, CDNA1). It contains the source used by one measured
homelab deployment, with hardware-specific eligibility guards and rollback
controls. It is not affiliated with AMD, Lemonade, Unsloth, or upstream
llama.cpp.

## Current qualified release

- Production branch: `gfx908-production`
- Qualified code head: `711c7bccf5580e2b68a2342e70a73808c4b5534e`
- Latest qualified gfx908 feature: `10321d50fd4447b629627cf75b2ccc90d0b7c4b5`
- Upstream merged through: `0b1bad14ff204627636aeb1de22ddcd5acb859d4`
- Build: llama.cpp `10430 (711c7bccf)`, Release, HIP `gfx908`
- Qualification date: 2026-08-12

The current Qwen3.6-27B IQ4_NL production profile uses the validated build and
enables the qualified SSM-to-Q/K L2 execution island. See
[STATUS.md](STATUS.md) for the exact deployment and runtime controls.

## Current stock comparison

This comparison uses the same MI100, host, and model files. The stock arm is
[Lemonade b1310](https://github.com/lemonade-sdk/llamacpp-rocm/releases/tag/b1310),
an independently packaged current gfx908 build at llama.cpp commit `2468576f`.
It is a practical stock reference, not a compiler-matched per-feature ablation.

The model files come from
[Unsloth Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF).
They include MTP layers, but every number below is **non-speculative** main-model
throughput.

<!-- markdownlint-disable MD013 -->

| Quant | Exact file size | SHA-256 (short) |
| --- | ---: | --- |
| IQ4_NL | 16,337,626,240 bytes (16.34 GB) | `dcf1b90d…47b9` |
| Q6_K | 22,884,406,400 bytes (22.88 GB) | `773f1bf0…02e` |

Protocol: three fresh processes per cell, using:

```text
llama-bench -p 8192 -n 1024 -b 8192 -ub {1024,2048,4096} -t 20 -r 1 -ngl 999 -o json
```

The runs use default f16 KV, flash-attention auto, and no speculative decoder.
Values are means in tokens/s; TG is shallow batch one.

| Quant | `-ub` | Stock pp8192 | gfx908 pp8192 | Change | Stock TG1024 | gfx908 TG1024 | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IQ4_NL | 1024 | 835.029 | **1226.406** | **+46.87%** | 36.064 | **42.783** | **+18.63%** |
| IQ4_NL | 2048 | 897.903 | **1444.384** | **+60.86%** | 36.380 | **42.982** | **+18.15%** |
| IQ4_NL | 4096 | 1070.240 | **1548.267** | **+44.67%** | 36.297 | **43.132** | **+18.83%** |
| Q6_K | 1024 | 837.751 | **1189.095** | **+41.94%** | 26.403 | **29.458** | **+11.57%** |
| Q6_K | 2048 | 899.264 | **1416.460** | **+57.51%** | 26.001 | **29.778** | **+14.53%** |
| Q6_K | 4096 | 1071.217 | **1531.563** | **+42.97%** | 26.332 | **29.868** | **+13.43%** |

A separate three-process IQ4_NL TG-only check at 1024 generated tokens
averaged `43.417 t/s`. The latest same-parent SSM-to-Q/K L2 A/B recovered
`0.3944 ms/token`, moving `42.3867` to `43.1067 t/s` with a 95% interval of
`0.2910–0.4978 ms/token`.

## Correctness result

Within the tested Qwen3.6-27B IQ4_NL and Q6_K production paths, no correctness
regression was detected. The affected same-parent optimization boundaries remain
bitwise equal; independent stock and CPU comparisons show only small numerical
drift between backend/toolchain builds.

Cross-build floating-point output is not expected to be bit-identical, so the
external check replays the same canonical token context at every step instead
of allowing an early near-tie to cascade through the rest of a generation. The
GPU arms use q8_0 K/V and flash-attention on, matching the production-style
server path.

| Quant | Fixed contexts | Stock vs gfx908 top-1 differences | Stock token absent from gfx908 top-20 | High-margin flips¹ |
| --- | ---: | ---: | ---: | ---: |
| IQ4_NL | 768 | 6 (0.78%) | 0 | 0 |
| Q6_K | 768 | 2 (0.26%) | 0 | 0 |

<!-- markdownlint-enable MD013 -->

¹ A high-margin flip requires both implementations to separate their first and
second choices by at least `0.25` log-probability.

A cache-matched CPU replay adjudicated every IQ4_NL context. It differed from
stock GPU in 8/768 cases and from the production GPU in 9/768. Across the eight
contexts where the GPU arms disagreed, it matched stock five times, production
four times, and neither once; matches can overlap where stock and production
agree. This is consistent with backend/toolchain numerical drift, not a
systematic production-path error.

The promoted build also passes 1216/1216 selected ROCm backend cases, retains
exactly 48 recurrent groups, reproduces deterministic 32-token runs, and keeps
the seven graph-visible boundaries affected by the latest island bitwise equal
to its same-parent control.

## What this branch adds

- Exact-shape gfx908 rocBLAS selection and a persistent autotune cache for the
  large GEMMs that dominate prompt processing.
- Chunked recurrent GDN and CDNA1-aware flash-attention paths for prefill.
- Guarded IQ4_NL and K-quant decode routes, including a 256-thread MMVF
  selector for measured shallow-B1 shapes.
- Owner-preserving recurrent execution islands for paired MMVF, norm/scale,
  epilogue work, and the exact SSM-to-Q/K L2 producer-consumer boundary.
- Production-shaped correctness tests, route census checks, feature toggles,
  and explicit rollback paths.

Closed or below-threshold experiments are not stacked into production. Detailed
negative results and raw lab artifacts remain in the separate GPU Lab notebook
rather than being presented as supported public features.

## Documentation

- [STATUS.md](STATUS.md) — current code, deployment, validation, and controls
- [BUILD.md](BUILD.md) — reproducible gfx908 build and minimum release gate
- [UPSTREAM.md](UPSTREAM.md) — upstream integration and promotion policy
- [BENCHMARKS.md](BENCHMARKS.md) — retained historical measurement record
- [QUALIFIED-B1-20260810.md](QUALIFIED-B1-20260810.md) — earlier same-parent
  B1 checkpoint
- [ENGINEERING.md](ENGINEERING.md) — architecture findings and corrections
- [CONSOLIDATION.md](CONSOLIDATION.md) — dated deployment-consolidation record

The rest of the repository follows upstream llama.cpp. Upstream documentation
remains authoritative for general use.

## License and credit

This fork remains under llama.cpp's MIT license. If the gfx908 optimization
work is useful in research or downstream software, credit to this repository
and its maintainer, [Tunahan (`@tnhnyzc`)](https://github.com/tnhnyzc), is
appreciated. Citation metadata is provided in the repository's
[`CITATION.cff`](../../CITATION.cff).
