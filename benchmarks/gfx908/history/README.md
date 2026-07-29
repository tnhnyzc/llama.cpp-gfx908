# Historical gfx908 benchmark records

These are the first retained `llama-bench` runs from the MI100 optimization
project. They establish the out-of-box upstream starting point before any local
gfx908 changes.

## Environment

- GPU: AMD Instinct MI100 32 GB
- Target: `gfx908:sramecc+:xnack-`
- llama.cpp: `e8e6c7af2456fd50bb62f7a2bbd642e6fb14ae77`
- ROCm: `7.15.0a20260720`
- Command: `llama-bench -p 128,512,2048,8192 -n 128 -r 5 -ngl 999 -fa auto`

## Qwen3.6-27B Q6_K

| Test | Throughput |
|---|---:|
| pp128 | 383.98 ± 46.71 tok/s |
| pp512 | 709.51 ± 30.69 tok/s |
| pp2048 | 714.00 ± 1.33 tok/s |
| pp8192 | 683.60 ± 2.53 tok/s |
| tg128 | 26.70 ± 0.54 tok/s |

Model size was 21.30 GiB with 27.32 billion parameters.

## Qwen3.6-27B IQ4_NL

| Test | Throughput |
|---|---:|
| pp128 | 617.80 ± 86.61 tok/s |
| pp512 | 708.53 ± 31.85 tok/s |
| pp2048 | 712.30 ± 1.51 tok/s |
| pp8192 | 682.05 ± 2.84 tok/s |
| tg128 | 36.65 ± 0.09 tok/s |

Model size was 15.21 GiB with 27.32 billion parameters.

## Interpretation

These records preserve the historical default invocation. They are not the
future canonical control because later production testing explicitly tuned
batch and ubatch sizes. A fresh MI100 qualification should run both upstream and
the current branch with identical explicit settings and record each grouped
commit as an optional intermediate checkpoint.
