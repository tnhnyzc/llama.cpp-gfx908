# Benchmark and correctness record

> This is the retained historical measurement record. For the current frozen
> production result and deployment identity, start with [README.md](README.md)
> and [STATUS.md](STATUS.md). Values here remain attached to their dated
> parents and should not be read as the present stock comparison.

## Oracle rules

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

## Original upstream baseline

Initial matched oracle on Qwen3.6-27B, `-b 1024 -ub 1024`, Q8 K/V cache,
flash attention, no speculation:

<!-- markdownlint-disable MD013 -->

| Quant | pp128 mean / warm | pp512 mean / warm | pp2048 | TG128 |
| --- | ---: | ---: | ---: | ---: |
| Q6_K | 378 / ~403 | 699 / ~718.5 | 868.1 | 27.529 |
| IQ4_NL | 584 / ~640.6 | 694.8 / ~712.7 | 859.9 | 35.564 |

<!-- markdownlint-enable MD013 -->

MTP with one draft token raised the deterministic warm completion from 27.65
to 40.81 tok/s for Q6_K and from 36.23 to 49.89 tok/s for IQ4_NL, with 93.8%
and 96.9% acceptance respectively.

## Retained incremental results

### Recurrent prefill

Chunked GDN, identical build with the route toggled and HIP graphs disabled:

| Quant | pp512 | pp1024 | pp2048 |
| --- | ---: | ---: | ---: |
| Q6_K | +12.1% | +17.6% | +19.3% |
| Q4_K_M | +11.4% | +17.1% | +18.9% |

The combined IQ4_NL prefill stack measured +27.3% at pp512, +22.2% at pp1024
and +24.0% at pp2048 in its original matched oracle.

### Flash attention

<!-- markdownlint-disable MD013 -->

| Change | Direct/whole-model result |
| --- | --- |
| Native f32 VKQ MFMA accumulation | +4.8% on a real 27.9k-token prefill; width gate removed the decode regression |
| Long-context 32x2 tile | +5.3% PP at 32k and +9.0% at 64k; neutral around 4-8k |
| Quantized KQ subgroup | kernel +11.8% at 32k and +12.5% at 64k; approximately +2-4% no-spec whole-model TG |
| Head-size 512 MFMA route | Gemma4 pp8192 +3.4%, pp32768 +22.7%, TG neutral in the qualifying oracle |

<!-- markdownlint-enable MD013 -->

The quantized-FA deployment passed 681/681 covered cases. A 12-chunk PTB run
was indistinguishable at `11.4265 ± 0.55086` versus
`11.4264 ± 0.55086`.

### Quantized decode

Q4_K/Q5_K branchless metadata reconstruction:

| Shape/result | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| Q5_K 10240 x 5120 direct kernel | 55.53 us | 50.13 us | +10.8% |
| Q5_K 5120 x 6144 direct kernel | 37.08 us | 33.32 us | +11.3% |
| Qwen Q4_K_M whole-model TG mean | 31.695 t/s | 35.05 t/s | +10.59% |

Qualification covered 22/22 Q5_K cases, 43/43 Q4_K cases, and 1164/1164
ROCm `MUL_MAT` cases. The four-chunk Q4_K_M perplexity result was
`14.3074 ± 1.21672` for control and `14.3131 ± 1.21708` for candidate.

## Deployment observations

These are useful real-world bounds but not clean A/B measurements:

- Qwen3.6-27B Q6_K reached approximately 1.38k tok/s on a favorable initial
  10k prefill with `-ub 4096`; larger retained contexts declined as expected.
- Qwen3.6-27B IQ4_NL reached approximately 1.42k tok/s in the comparable
  service workload.
- Low-context MTP generation reached about 50-52 tok/s Q6_K and 58-61 tok/s
  IQ4_NL; long-context attention and speculative acceptance reduce those rates.
- Gemma4-31B Q4-based service decode reached about 45-46.5 tok/s at shallow
  context and about 37.5 tok/s after a roughly 20k-token prefill.
- Offloaded GPT-OSS-120B reached roughly 36-40 tok/s depending on warm state and
  CPU contention. Its GPU kernel performance cannot be inferred directly from
  this whole-server result because many experts reside on CPU.

## Artifact policy

Raw profiler databases are too large and environment-specific for the main
source branch. Compact CSV/JSON summaries, commands, hashes and analyzer scripts
should be added under `benchmarks/gfx908/` as the oracle is rerun. Large raw
captures may be attached to tagged releases or stored externally with hashes.
