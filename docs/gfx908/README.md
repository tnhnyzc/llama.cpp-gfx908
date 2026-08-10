# llama.cpp for gfx908 / AMD Instinct MI100

This is an independently maintained llama.cpp optimization fork for AMD
Instinct MI100 (`gfx908`, CDNA1). It contains the source used by one measured
homelab deployment and retains hardware-specific guards and rollback controls.
It is not affiliated with AMD or upstream llama.cpp.

## Current release

- Production branch: `gfx908-production`
- Qualified binary source: `bef57196449ac04d8dd61412faabb69a917bb3af`
- Binary-source tag: `gfx908-production-bailingmoe3-20260810`
- Qualified upstream merge: `e9bc41ced277e432bb7b428d11baddee16fd9f1a`
- Hardware/toolchain: MI100 32 GB, `gfx908:sramecc+:xnack-`, clang 23
- Qualification date: 2026-08-10

Commits after the binary-source tag are documentation-only. The exact dated
build and rollback information are recorded in [STATUS.md](STATUS.md).

## Qualified shallow-B1 result

Qwen3.6-27B IQ4_NL shallow batch-one decode was measured with six balanced,
collector-controlled pairs:

| State | Throughput | Latency | Energy/token |
| --- | ---: | ---: | ---: |
| same-parent control | 39.8804 t/s | 25.0754 ms | 7.2343 J |
| consolidated gfx908 | **41.7821 t/s** | **23.9339 ms** | **6.9176 J** |

The retained recovery is `1.1415 ms/token` (95% CI `0.9535–1.3295`), or
`+4.77%` throughput and `-4.38%` energy/token. The later upstream merge was
performance-equivalent, so `41.7817 t/s / 23.933923 ms/token` remains the
canonical calibration. See [QUALIFIED-B1-20260810.md](QUALIFIED-B1-20260810.md).

## What this branch adds

- CDNA1-aware quantized decode, flash-attention, GEMM selection, and recurrent
  execution paths retained from the earlier qualified fork.
- A 256-thread MMVF selector for the measured gfx908 shapes.
- A 48-group recurrent paired-MMVF and norm-scale execution island.
- A disjoint direct norm-to-q8 route for exactly eligible producer fanout.
- BailingMoE3 support from upstream PR 26608, validated by loading and
  generating with Ling 3.0 Flash.
- Production-shaped correctness cases and explicit environment rollbacks.

Closed or below-threshold experiments are not stacked into production. Their
results live in the separate GPU Lab notebook rather than on public branches.

## Documentation

- [STATUS.md](STATUS.md) — current code, build, runtime gates, and exclusions
- [BUILD.md](BUILD.md) — reproducible gfx908 build and minimum validation
- [QUALIFIED-B1-20260810.md](QUALIFIED-B1-20260810.md) — frozen B1 evidence
- [UPSTREAM.md](UPSTREAM.md) — two-branch update and promotion policy
- [BENCHMARKS.md](BENCHMARKS.md) — retained historical measurement record
- [ENGINEERING.md](ENGINEERING.md) — architecture findings and corrections
- [CONSOLIDATION.md](CONSOLIDATION.md) — earlier 2026-08-02 deployment history

The rest of the repository follows upstream llama.cpp. Upstream documentation
remains authoritative for general usage.
