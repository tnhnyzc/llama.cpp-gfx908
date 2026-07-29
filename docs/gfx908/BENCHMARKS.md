# Benchmark and correctness record

## Test method

Performance numbers are accepted only when the test records:

- exact model and tensor quantization, not the filename alone;
- source commit and build configuration;
- prompt/decode shape, batch and ubatch decomposition;
- cold versus warm samples;
- run order, with the order reversed when the result is close;
- kernel route and absolute latency, not percentage alone;
- correctness against a CPU or unchanged-backend reference;
- GPU power and temperature; and
- production frequency or relevance of the measured shape.

Perplexity comparisons must use the same corpus, tokenization, chunk count,
batch width and command line. Server figures are kept separate from
`llama-bench` and direct-kernel measurements.

## What the historical numbers mean

There are two useful but different views of the project:

1. The first recorded upstream run shows the out-of-box experience before any
   gfx908 work. It used `llama-bench` defaults and is the honest historical
   starting point.
2. Later optimization work used larger explicit batch and ubatch settings. Those
   runs isolate real improvements, but they cannot be arranged into a single
   code-only speedup by comparing their absolute values with the default run.

The final clean branch has been rebuilt for gfx908 but has not yet been rerun on
MI100 with identical settings. Until that run exists, this page does not invent
a single "upstream to current" percentage from unlike tests.

The practical endpoint is nevertheless meaningful: favorable 27B prefill moved
from roughly 712-714 tok/s in the first stock run to roughly 1.3-1.5k tok/s over
the course of the project. That is the real user-visible progression, with both
configuration and code improvements included. A future normalized row will
separate their contributions without replacing this historical record.

## Original upstream baseline

The recorded raw runs use upstream commit
`e8e6c7af2456fd50bb62f7a2bbd642e6fb14ae77`, ROCm
`7.15.0a20260720`, full GPU offload, flash attention and no speculation:

```text
llama-bench -p 128,512,2048,8192 -n 128 -r 5 -ngl 999 -fa auto
```

| Quant | pp128 | pp512 | pp2048 | pp8192 | TG128 |
|---|---:|---:|---:|---:|---:|
| Q6_K | 383.98 ± 46.71 | 709.51 ± 30.69 | 714.00 ± 1.33 | 683.60 ± 2.53 | 26.70 ± 0.54 |
| IQ4_NL | 617.80 ± 86.61 | 708.53 ± 31.85 | 712.30 ± 1.51 | 682.05 ± 2.84 | 36.65 ± 0.09 |

These unexpectedly low PP values are real for that invocation. Later figures
above 1,000 tok/s combine code improvements with deliberate batch/ubatch tuning,
so the difference is not attributable to kernels alone. The archived console
records are under [`benchmarks/gfx908/history`](../../benchmarks/gfx908/history/README.md).

At this commit, the omitted defaults were `-b 2048 -ub 512`, F16 K/V cache and
automatic CPU thread count. A second recorded run kept the source completely
unchanged but used `-b 2048 -ub 1024`, Q8 K/V and 20 threads:

| Quant | pp128 | pp512 | pp1024 | pp2048 |
|---|---:|---:|---:|---:|
| Q6_K | 380.02 | 686.14 | 858.44 | 850.16 |
| IQ4_NL | 603.88 | 702.53 | 867.78 | 856.98 |

That is approximately +19% Q6_K and +20% IQ4_NL at pp2048 over the original
default invocation. Ubatches account for most of the difference; K/V precision
has little effect on zero-depth bulk prefill, and CPU thread count is secondary
once the work is on GPU. Configuration tuning is part of the practical project,
but this row makes its contribution visible rather than attributing it to code.

Early real-server MTP requests on the same upstream build reached 40.55 tok/s
for Q6_K and 51.07 tok/s for IQ4_NL. These were individual service requests with
different prompts and acceptance rates, so they are historical service
observations rather than a controlled no-spec/MTP comparison.

## Short chronology

The project moved through the following major turning points. Rows in the
absolute-result column are recorded measurements, but only arrows within one row
are controlled comparisons. Rows that change ubatch are configuration gains.

| Stage | Representative result | Interpretation |
|---|---|---|
| Upstream `e8e6c7af` | Q6/IQ4 pp2048: 714/712 tok/s | Historical default invocation |
| Upstream, explicit ubatch 1024 | Q6/IQ4 pp2048: 850/857 tok/s | Same unmodified source; approximately +19-20% from the practical configuration |
| Chunked recurrent prefill | Q6 +12.1% pp512, +19.3% pp2048 | Removed the dominant recurrent-prefill bottleneck |
| Larger ubatch | IQ4 pp4096: 1027.5 → 1105.7 | +7.6% from ubatch 1024 → 2048 |
| Exact rocBLAS solution selection | IQ4: 1112.45 → 1346.21; Q6: 1108.05 → 1345.42 | About +21% at pp4096, TG neutral |
| Runtime GEMM autotuning | IQ4 pp4096: 1160.4 → 1445.9 | Same later build, autotuning globally off versus warm cache; overlaps the preceding row |
| Tiled recurrent concat | Q6 pp4096: 1400.26 → 1440.22 | +2.9% isolated data-layout improvement |
| ubatch 4096 | IQ4: 1444.80 → 1518.17; Q6: 1435.94 → 1531.11 | Additional PP at a material VRAM cost |
| Later attention/decode work | See the isolated tables below | Mostly long-context PP/TG and quant-specific gains |
| Current clean branch | Runtime result pending | Source reconstructed and compiled; MI100 rerun still required |

This chronology is intentionally not summed. Controls overlap, several stages
used different ubatches, and later attention/decode work affects different model
shapes and context depths.

## Incremental results

### Recurrent prefill

Chunked GDN, identical build with the route toggled and HIP graphs disabled:

| Quant | pp512 | pp1024 | pp2048 |
|---|---:|---:|---:|
| Q6_K | +12.1% | +17.6% | +19.3% |
| Q4_K_M | +11.4% | +17.1% | +18.9% |

The combined IQ4_NL prefill stack measured +27.3% at pp512, +22.2% at pp1024
and +24.0% at pp2048 in its original matched test.

### Flash attention

| Change | Direct/whole-model result |
|---|---|
| Native f32 VKQ MFMA accumulation | +4.8% on a real 27.9k-token prefill; width gate removed the decode regression |
| Long-context 32x2 tile | +5.3% PP at 32k and +9.0% at 64k; neutral around 4-8k |
| Quantized KQ subgroup | kernel +11.8% at 32k and +12.5% at 64k; approximately +2-4% no-spec whole-model TG |
| Head-size 512 MFMA route | Gemma4 pp8192 +3.4%, pp32768 +22.7%, TG neutral in the matched test |

The quantized-FA deployment passed 681/681 covered cases. A 12-chunk PTB run
was indistinguishable at `11.4265 ± 0.55086` versus
`11.4264 ± 0.55086`.

### Quantized decode

Q4_K/Q5_K branchless metadata reconstruction:

| Shape/result | Control | Candidate | Change |
|---|---:|---:|---:|
| Q5_K 10240 x 5120 direct kernel | 55.53 us | 50.13 us | +10.8% |
| Q5_K 5120 x 6144 direct kernel | 37.08 us | 33.32 us | +11.3% |
| Qwen Q4_K_M whole-model TG mean | 31.695 t/s | 35.05 t/s | +10.59% |

Qualification covered 22/22 Q5_K cases, 43/43 Q4_K cases, and 1164/1164
ROCm `MUL_MAT` cases. The four-chunk Q4_K_M perplexity result was
`14.3074 ± 1.21672` for control and `14.3131 ± 1.21708` for candidate.

The 10.59% whole-model result belongs to the separate
Qwen3.6-27B-Q4_K_M GGUF, where Q4_K and Q5_K account for 72.6% of weight bytes.
That model was not referenced by the daily llama-swap profiles. On the served
IQ4_NL model, Q5_K has a much smaller surface, so this number must not be carried
over as the daily profile's expected gain.

## Deployment observations

These are useful real-world bounds but not clean A/B measurements:

- Qwen3.6-27B Q6_K reached approximately 1.38k tok/s on a favorable initial
  10k prefill with `-ub 4096`; larger recorded contexts declined as expected.
- Qwen3.6-27B IQ4_NL reached approximately 1.42k tok/s in the comparable
  service workload.
- Gemma4-31B moved from 510.1 to 1035.9 tok/s at pp4096 when CDNA1 stopped
  forcing large Q4_0 batches through MMQ. Wave64 dequantization later raised
  pp4096 from 1037.7 to 1078.3 tok/s. The long-context FA route measured 1047.1
  tok/s at pp8192 and 751.4 at pp32768; a real server request at roughly 20k
  prompt tokens reached 757.4 tok/s.
- Low-context MTP generation reached about 50-52 tok/s Q6_K and 58-61 tok/s
  IQ4_NL. Compared with the early upstream service observations of 40.55 and
  51.07 tok/s, those ranges are roughly +23-28% and +14-19%. This is a useful
  daily-use comparison, but not a controlled MTP-only A/B because prompts,
  cache state and acceptance differed. Long-context attention and speculative
  acceptance also reduce those rates.
- Gemma4-31B Q4-based service decode reached about 45-46.5 tok/s at shallow
  context and about 37.5 tok/s after a roughly 20k-token prefill.
- Offloaded GPT-OSS-120B reached roughly 36-40 tok/s depending on warm state and
  CPU contention. Its GPU kernel performance cannot be inferred directly from
  this whole-server result because many experts reside on CPU.

## Artifact policy

Raw profiler databases are too large and environment-specific for the main
source branch. Compact CSV/JSON summaries, commands, hashes and analyzer scripts
should be added under `benchmarks/gfx908/` as the test matrix is rerun. Large raw
captures may be attached to tagged releases or stored externally with hashes.
