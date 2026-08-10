# gfx908 production status

This is the concise source of truth for the MI100 fork as frozen on 2026-08-10.

## Source and deployment

- Public branches: `gfx908-production` and `upstream` only.
- Qualified binary source: `bef57196449ac04d8dd61412faabb69a917bb3af`.
- Dated build:
  `/home/llm/mi100/llama.cpp-gfx908/build-prod-20260810-bailingmoe3`.
- Previous build and pre-deployment config remain available for rollback.
- `llama-swap` MI100-only llama.cpp profiles use the dated build; mixed and
  vLLM profiles are unchanged.

The final build reports llama.cpp `10398 (bef571964)` and passed:

- 1358/1358 selected ROCm RMS_NORM, MUL, and MUL_MAT cases;
- the architecture registry, including BailingMoE3; and
- 111 chat-parser tests / 524 assertions with zero failures.

## Qualified shallow-B1 composition

The production code contains exactly four newly qualified effects:

1. CDNA1 MMVF 256-thread selection;
2. recurrent paired-256 MMVF execution for 48 sibling pairs;
3. a 48-group recurrent norm-scale/q8/paired-MMVF island; and
4. the disjoint strict direct norm-to-q8 route for 80 producers and 112 MMVQ
   consumers.

Together they measure `41.7817 t/s / 23.933923 ms/token` on the canonical
Qwen3.6-27B IQ4_NL shallow-B1 workload. The recurrent population takes
priority over direct q8, preventing overlap or double counting.

## Runtime controls

- `GGML_HIP_GEMM_AUTOTUNE_GFX908=1` enables exact-shape rocBLAS tuning; use
  `GGML_HIP_GEMM_AUTOTUNE_CACHE` for the persistent cache.
- `GGML_HIP_GDN_CHUNK_GFX908=1` enables the qualified chunked recurrent path;
  `GGML_HIP_GDN_CHUNK_GFX908_DIR` must identify the packaged HSACO directory.
- `GGML_HIP_Q5_K_MMQ_N3_GFX908=1` enables the retained Q5_K N3 route.
- `GGML_HIP_RECURRENT_MMVF_PAIR_GFX908=1` selects the 48-pair projection path.
- `GGML_HIP_RECURRENT_NORM_SCALE_ISLAND_GFX908=1` selects the dependent
  norm-scale island and requires the recurrent pair route.
- `GGML_HIP_SHARED_NORM_Q8_GFX908=1` selects the strict disjoint direct-q8 path.

The older guarded CDNA1 dequantization, MMVQ, flash-attention, and exact-shape
prefill controls remain in source. See `--help`, source guards, and
[BUILD.md](BUILD.md) before enabling a route on a different model or shape.

## Serving notes

- Qwen3.6/Qwen3.8 and Gemma 4 MTP profiles explicitly use
  `--spec-draft-p-min 0.0`; Qwen testing found `0.5` slower despite a higher
  reported acceptance ratio.
- Muse retains its separately established DFlash `p_min=0.6` setting.
- Gemma 4 plus its real MTP model loaded and generated successfully.
- Ling 3.0 Flash loaded and generated through BailingMoE3. Its single cold,
  partly host-offloaded observation is not a qualified performance result.

## Explicit exclusions

Production does not include the failed or below-threshold MTP N3 one-wave,
stripped/one-wave IQ4_NL, mixed f32+q8, generic wave64 packing, Q/K L2-GDN,
MMVF split-column, recurrent gated-q8, destructive MMVQ, or STREAM-floor
experiments. Do not infer production support from their archived source.

No further optimization branch is open at this freeze point.
