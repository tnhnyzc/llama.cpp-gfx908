# gfx908 production status

This is the concise source of truth for the qualified MI100 state on
2026-08-12.

## Source and deployment

- Qualified code head: `711c7bccf5580e2b68a2342e70a73808c4b5534e`.
- Qualified SSM-to-Q/K L2 commit: `10321d50fd4447b629627cf75b2ccc90d0b7c4b5`.
- Upstream merged through: `0b1bad14ff204627636aeb1de22ddcd5acb859d4`.
- Validated build:
  `/home/llm/bench/build-gfx908-production-ssm-l2-upstream-20260812`.
- Deployed Qwen3.6-27B IQ4 build:
  `/home/llm/mi100/llama.cpp-gfx908/build-prod-20260812-ssm-l2-upstream`.

The validated and deployed `llama-server` and `libggml-hip.so` hashes match.
The earlier recurrent-epilogue build and pre-deployment configuration remain
available for rollback. Other serving profiles were not migrated merely to
standardize paths.

## Validation

The build reports llama.cpp `10430 (711c7bccf)` and passed:

- 1216/1216 selected ROCm backend cases;
- exactly 48 unique recurrent groups and 192 construction records;
- deterministic 32-token repeat comparison; and
- the expected seven graph boundaries for SSM-to-Q/K L2, bitwise against its
  same-parent control.

An identical-binary process control accepted the fresh-process benchmark
fixture. The subsequent fixed 20-pair clean-build comparison measured
`711c7bccf` minus feature-only `10321d50f` at `-0.0026 ms/token`, with its 95%
interval `[-0.0605,+0.0552]` entirely inside the predeclared
`±0.100 ms/token` equivalence margin.

The current stock performance and fixed-context correctness comparison is in
[README.md](README.md). The latest isolated feature recovered
`0.3944 ms/token` over six collector-controlled pairs; its 95% interval was
`0.2910–0.4978 ms/token`.

## Active IQ4 production controls

- `GGML_HIP_GDN_CHUNK_GFX908=1`
- `GGML_HIP_GEMM_AUTOTUNE_GFX908=1`
- `GGML_HIP_IQ4_NL_FUSED_GFX908=1`
- `GGML_HIP_DISABLE_GFX908_M2_GATE_FUSION=1`
- `GGML_HIP_RECURRENT_MMVF_PAIR_GFX908=1`
- `GGML_HIP_RECURRENT_NORM_SCALE_ISLAND_GFX908=1`
- `GGML_HIP_RECURRENT_EPILOGUE_ISLAND_GFX908=1`
- `GGML_HIP_SSM_L2_ISLAND_GFX908=1`
- `GGML_HIP_SHARED_NORM_Q8_GFX908=1`

Each specialized route is guarded by architecture, shape, graph, and fanout
eligibility. Turning off a control returns to the retained general path; the
GDN runtime additionally requires its packaged HSACO directory. See
[BUILD.md](BUILD.md) before enabling these controls for a different model.

## Scope and exclusions

The current SSM-to-Q/K L2 route preserves the SSM producer's ownership and
absorbs exact downstream normalization before its Q/K tiles retire. It does not
revive the earlier consumer-side Q/K L2-GDN reconstruction experiment, which
remains excluded.

Other failed or below-threshold experiments are likewise not present in the
production composition. Historical measurements remain in
[BENCHMARKS.md](BENCHMARKS.md) and dated checkpoint documents. No experimental
source branch is open at this checkpoint.
