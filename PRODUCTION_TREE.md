# This tree is live in production (since 2026-07-24)

`build-at/bin/llama-server` is referenced by
`/home/llm/config/llama-swap/config.yaml` for **Qwen3.6-27B** and
**Qwen3.6-27B-Fast**, together with:

    GGML_HIP_GEMM_AUTOTUNE_GFX908=1
    GGML_HIP_GEMM_AUTOTUNE_CACHE=/home/llm/config/llama-swap/gfx908-gemm.tsv

Contains: rocBLAS GEMM autotuner (A3) and the tiled-transpose concat fast
path (F6, on by default).

Do not edit sources or rebuild here. Branch a new tree for experiments.
The previous production tree is `llama.cpp-claude-rocblas-sol` (hardcoded
solution table) — keep it as the rollback target.
