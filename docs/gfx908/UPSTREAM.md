# Upstream and branch policy

This is a standalone repository rather than a GitHub fork so it can coexist
with the account's upstream-contribution fork.

Recommended remotes:

```sh
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git remote add origin https://github.com/tnhnyzc/llama.cpp-gfx908.git
```

## Branches

- `gfx908-production`: exact source of the last qualified daily build. This is
  the default branch until a newer upstream integration passes the full oracle.
- `upstream`: mirror of a known upstream commit, with no local changes.
- `gfx908-next`: integration branch for upstream updates and qualified new work.
- `experiments/<topic>`: disposable or retained investigations. An experiment
  is never merged solely because a microbenchmark improved.

## Update procedure

1. Fetch upstream and record the old/new base commits.
2. Rebase or replay the five logical change groups onto `gfx908-next`.
3. Resolve conflicts by mechanism, not by blindly preferring either side.
4. Build into a new directory.
5. Run the correctness, perplexity, PP, TG, MTP and thermal oracle.
6. Compare against both current production and pristine new upstream.
7. Promote `gfx908-next` only when it is non-regressing on daily models and all
   accepted changes remain attributable.
8. Tag the promoted state and retain the old build/config for rollback.

The upstream branch may move regularly; the production branch should move only
after hardware qualification. That keeps daily use stable without allowing the
fork to become permanently detached from llama.cpp development.
