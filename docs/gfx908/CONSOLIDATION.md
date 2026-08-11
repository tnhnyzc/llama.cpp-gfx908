# gfx908 deployment consolidation

> Historical note: this describes the 2026-08-02 consolidation. It has been
> superseded operationally by the 2026-08-10 production state in
> [STATUS.md](STATUS.md), but remains useful provenance.

On 2026-08-02 the MI100 llama.cpp deployment was reduced to one source tree and
one production build:

- source: `/home/llm/mi100/llama.cpp-gfx908`
- build: `/home/llm/mi100/llama.cpp-gfx908/build-prod`
- stable path: `/home/llm/mi100/llama.cpp-gfx908-current`
- production branch and build commit: `gfx908-production` at `387c97018`
- GDN runtime: `build-prod/runtime/gdn`, verified by
  `scripts/gfx908/gdn-sha256.txt`

The release was built by `scripts/gfx908/build-release.sh` against the pinned
custom ROCm tree and checked by `scripts/gfx908/validate-release.sh`. ROCm
MUL_MAT correctness passed 1,216/1,216 cases. Both Qwen3.6-27B profiles started
from the canonical fingerprint. A 3,799-token native-tool request completed at
about 1,239 prompt tok/s and exercised the chunked GDN route that had previously
failed because its HSACO directory was absent.

All llama.cpp MI100 profiles in `/home/llm/config/llama-swap/config.yaml` now
use the stable path. Laguna no longer has a separate binary. The vLLM MI100
profile remains separate because it is a different inference engine.

## Historical experiments

Before deleting duplicate worktrees, every dirty source delta was committed to
a named local archive branch. Older mainline and copied-tree experiments were
imported under `archive-import/`; experiments already sharing the canonical Git
repository live under `archive/`. There are 66 archive refs in total. Benchmark
results, profiler captures, engineering notes, the custom ROCm tree, and the
qualified GDN artifacts were not removed.

Useful commands:

```sh
git -C /home/llm/mi100/llama.cpp-gfx908 branch --list 'archive/*' 'archive-import/*'
/home/llm/mi100/llama.cpp-gfx908/scripts/gfx908/validate-release.sh
/usr/local/sbin/llama-swap-config-check
```

Historical configuration snapshots still contain their original absolute
paths. They are evidence, not executable rollback targets. Source rollback is
through Git; deployment rollback is a rebuild followed by an atomic update of
`llama.cpp-gfx908-current`.
