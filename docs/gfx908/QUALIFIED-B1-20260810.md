# Qualified shallow-B1 consolidation: 2026-08-10

> **Current disposition:** this four-effect state was committed, qualified
> through upstream merge `e9bc41ced`, and released in the production binary
> sourced from `bef571964`. The upstream merge was performance-equivalent and
> does not replace the frozen calibration below.

This record freezes the last fully measured state on upstream parent
`ead92eb55c6b17f45af99fbcf381414e76757f92` before the August 10 upstream
integration.

## Included source

Code commit `cb1375d3b26644d07a5642469d2b151a49b8189a` contains exactly these four
qualified effects:

1. CDNA1 MMVF 256-thread selection;
2. recurrent paired-256 MMVF execution for 48 sibling pairs;
3. recurrent norm-scale/q8/MMVF execution islands for the same 48 groups;
4. the disjoint strict direct norm-to-q8 route for 80 producers and 112 MMVQ
   consumers.

The recurrent route has priority over direct q8, so the 48 mixed-consumer
producers cannot enter the direct-q8 set. Each non-default execution island
retains an explicit environment gate.

The retained source patch hashes are:

- MMVF/pair/norm-scale:
  `cd082fa5d85170755019890ce4cc074bbe48fdde3f9bb3a73e709b25a55c6ad4`;
- direct q8:
  `e337b2243dc263973a5182222ffa36c1ee094dc8342b01cca7ed960b84af8099`.

## Validation

The commit-bound build passed 1328/1328 selected RMS_NORM, MUL, and MUL_MAT
ROCm backend tests. Two control and two candidate 64-token deterministic runs
were identical after removing build and timing metadata. The normalized output
SHA-256 was
`8e83a838da888d89521f49f3183c9b985566cc1b4c9aabc69101153e5c4db92e`.

The runtime census selected exactly:

- 48 recurrent MMVF pairs, zero rejected groups;
- 48 recurrent norm-scale groups;
- 96 q8 and 96 MMVF consumers for those groups;
- 81 remaining width families rejected by the recurrent island.

The direct-q8 route retains its prior exact 80-producer / 112-consumer proof.

## Frozen performance

Qwen3.6-27B IQ4_NL shallow B1 used six balanced collector pairs, four
`tg128` repetitions per observation, 30-second cooldowns, empty-GPU gates, and
matched thermal entry conditions.

| state | mean t/s | mean ms/token | energy/token |
| --- | ---: | ---: | ---: |
| clean same-parent control | 39.880440 | 25.075418 | 7.234279 J |
| consolidated | **41.782118** | **23.933923** | **6.917636 J** |

All six paired effects favored the consolidated state. Mean recovery was
`1.141495 ms/token`, with 95% Student-t interval `[0.953450, 1.329541]`.
Mean throughput improved `4.771%` and energy/token fell `4.38%`. The candidate
ran faster despite a 21.9 MHz lower mean GFX clock; HBM/hotspot maxima were
matched at 83-84 C.

This measurement supersedes the additive `23.801116 ms/token / 42.014836 t/s`
prediction. Use `23.933923 ms/token / 41.7817 t/s` as the golden pre-upstream
reference.

Raw artifacts are retained on the MI100 host at:

- `/home/llm/bench/gfx908-qualified-consolidation-correctness-20260810`;
- `/home/llm/bench/gfx908-qualified-consolidation-ab-20260810`.

## Explicit exclusions

No source from these investigations is included: MTP N3 one-wave, stripped or
one-wave IQ4_NL N1, generic final-wave reduction, Q/K L2 sibling fusion, ADD
chain deletion/fusion, mixed f32+q8, MMVF split columns, recurrent gated-q8,
Q/K L2-to-GDN scale island, generic wave64 packing, destructive MMVQ controls,
or STREAM minimum-node substitutions.

Those results remain preserved as negative, below-threshold, superseded, or
diagnostic evidence in the GPU Lab notebook. Their exact close/reopen
boundaries must not be broadened or erased.

## Freeze boundary

The pre-upstream state is tagged `gfx908-b1-qualified-ead92-20260810`; the
qualified upstream and binary-source checkpoints have separate release tags.
Future upstream or source changes require a new build and oracle and do not
retroactively alter this measurement.
