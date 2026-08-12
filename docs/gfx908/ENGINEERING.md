# Engineering record

> This file preserves architecture findings and corrected hypotheses from the
> longer investigation. Current production scope is summarized in
> [STATUS.md](STATUS.md).

## What the investigation established

The MI100 was healthy before source tuning: the original sustained memory tests
reached roughly 913-992 GB/s against 1.228 TB/s theoretical HBM bandwidth,
power and thermal behavior were stable, and ECC remained clean. Those original
figures are not the read-only decode ceiling. Later 2 GB read-only streaming
tests reached 1,128-1,146 GB/s, while an MMVQ-shaped row/stride access test
reached 1,017-1,026 GB/s across 64-, 128- and 256-thread workgroups. The older
~926 GB/s figure came from a different, effectively read+write/insufficient-MLP
workload and must not be used to conclude that decode MMVQ is bandwidth-complete.
Weak application results therefore could not be attributed to a broken card or
obviously unhealthy ROCm installation.

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
- The earlier ~926 GB/s "practical HBM ceiling" was incorrectly generalized
  from a different streaming workload. Read-only and MMVQ-shaped measurements
  now define separate ceilings. Q6_K decode MMVQ sustains about 837 GB/s against
  roughly 1,020 GB/s demonstrated for its access pattern, but exact fused-kernel
  counters show that this gap is not primarily wasted DRAM transactions: a
  146.23 MB production gate/up dispatch requested 148.23 MB from DRAM, only
  1.37% overhead. Its larger exposed limit was Q6 unpack arithmetic and waits.

These corrections are part of the result, not historical clutter: they explain
why the current branch contains regression oracles and conservative guards.

## Retained architecture findings

The findings below retain their measured parent and workload. They explain the
current implementation, but their historical timings are not estimates of
unclaimed headroom in the latest release.

### Prompt processing

- Qwen recurrent GDN: addressed for the exact tested shape with chunking.
- Shared quantized FFN path: improved through wave64 dequantization and exact
  rocBLAS selection; further progress likely requires a Tensile-class fused
  quant-to-FP16 MFMA schedule.
- Flash attention: materially improved, especially at long context and head
  dimension 512.
- Data movement/concat: smaller but measurable remaining surface.

### Decode

- MMVQ was the dominant kernel family in the retained representative profiles,
  at approximately 66-74% of token time depending on model and context.
- Q4_K/Q5_K metadata work had a real CDNA1 branch-divergence opportunity and is
  addressed here.
- Arithmetic deletion probes show that IQ4_NL is not primarily ALU-bound, but
  the corrected read-only ceiling means this does not imply that MMVQ is close
  to the card's streaming limit. The remaining problem is issuing and retaining
  enough independent memory work without crossing a VGPR occupancy boundary.
- A full-stage Q6_K rows=2 software pipeline is rejected. Explicit next-stage
  loads regressed direct kernels 24-27% whether the current stage lived in
  AGPRs or VGPRs; the VGPR form also reduced Qwen3.6-27B Q6_K d0 TG from a
  28.168 tok/s control bracket to 24.520 tok/s (-12.95%). Coarse full-stage
  waits destroy useful fine-grained scheduling in the existing compiler path.
- Q6_K zero-point arithmetic is materially improved on CDNA1. The baseline
  converts each packed unsigned 6-bit group to signed bytes with a long SDWA
  saturating-subtract expansion. The retained candidate instead evaluates
  `dot(q_unsigned, q8) - 32*dot(1, q8)`, adding one `v_dot4` while deleting the
  byte-conversion sequence. All 20 Q6_K MUL_MAT cases and both exact production
  fused gate/up cases pass. The hot fused kernel falls from 842 to 768 static
  instructions and 44 to 41 VGPR, with no spills. Counters show VALU instructions
  falling 19.845M to 13.387M (-32.5%), unchanged VMEM/DRAM traffic, and dispatch
  time falling 186.719 to 164.480 us (-11.9%). A cold shared-library A/B/B/A
  bracket raises Qwen3.6-27B Q6_K TG from 28.258 to 29.799 tok/s (+5.45%).
- Projection-split wave ownership is rejected at -0.91% end to end. Direct
  unaligned dword source loads are neutral (+0.06%) because the baseline compiler
  already emits dword loads. Cooperative metadata shuffles regress approximately
  37.5%; gfx908 shuffle/control cost exceeds any saved replicated requests. A
  CDNA1 two-iteration outer-K-loop unroll is also neutral/slightly negative at
  -0.21% end to end; compiler loop hints do not expose the missing MLP.
- Roughly 1,780 dispatches in an earlier representative token made launch and
  graph gaps relevant, but later exact decomposition showed that much of the
  non-projection tail is semantic work rather than generic replay overhead.
- Standalone adjacent Q8_1 activation reuse is measured and parked. Qwen3.6-27B
  Q6_K has 80 reusable sibling quantizations among 337 MMVQ calls per decode
  evaluation (23.74%). A graph-disabled prototype removes exactly those 80
  launches and cuts `quantize_q8_1` time 23.57%, but this is only 0.95% of
  summed kernel time. End-to-end brackets disagreed (+0.50% short, -0.56%
  sustained under strong clock drift), so no TG gain is claimed. A graph-safe
  cache is not justified unless incorporated into broader persistent/fused
  execution that also removes inter-kernel gaps. Artifact:
  `results/claude-tg-20260802/q8-reuse/FINDINGS.md` on the experiment host.
- Long-context quantized attention had a separate wave64 subgroup gap and now
  gains approximately 2-4% whole-model TG at the measured depths.

## Highest-value future work

The Q6_K zero-point reformulation, portable GDN route, same-source CUDA mirror,
and upstream requalification listed here previously are complete. The current
order is deliberately narrower:

1. Census unclaimed producer-consumer populations that could absorb exact
   downstream work before values retire, while preserving graph-visible f32
   boundaries, fanout semantics, and independently schedulable workgroups.
   Source work requires a coherent class near `0.5 ms/token` and a first valid
   mechanism near `0.2 ms/token`.
2. Revisit projection only with a concrete same-byte execution contract that
   preserves the successful two-row, K-coalesced ownership and CU exposure.
   The fitted fixed/instruction differential is calibration evidence, not a
   directly recoverable budget.
3. Keep long-context flash-attention work separate from shallow batch-one
   decode, with its own context-dependent whole-model gate.
4. Continue dated upstream merges only through the build, correctness, route,
   process-control, and rollback procedure in [UPSTREAM.md](UPSTREAM.md).

Previously rejected projection splitting, metadata shuffles, full-stage
prefetch, fixed-block compression, and standalone activation caching should not
be reopened without a materially different ownership or representation
contract.

## Evidence levels

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
