# Upstream and branch policy

The public repository intentionally has two branches:

- `gfx908-production` — the qualified MI100 fork and default branch;
- `upstream` — the pristine upstream commit used by the latest qualified merge.

Experiments and historical worktree states are not published as branches.
Qualified checkpoints use annotated tags; the compact history-archive tag
retains the removed branch tips when forensic recovery is needed.

Recommended remotes:

```sh
git remote add origin https://github.com/tnhnyzc/llama.cpp-gfx908.git
git remote add upstream https://github.com/ggml-org/llama.cpp.git
```

## Promotion procedure

1. Fetch upstream and record both source tips.
2. Integrate on an isolated local branch or worktree.
3. Resolve conflicts by mechanism and preserve gfx908 guards.
4. Build into a new dated directory with a manifest.
5. Run correctness, deterministic-output, model-route, and controlled
   performance gates appropriate to the changed code.
6. Promote only the validated state to `gfx908-production`.
7. Move `upstream` to the pristine commit that was actually qualified.
8. Tag the release and retain the prior build/config for rollback.

Microbenchmark improvement alone never earns promotion, and historical archive
refs should not be republished as normal GitHub branches.
