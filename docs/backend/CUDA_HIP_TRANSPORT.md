# CUDA-HIP scheduler transport

## Scope

This work addresses graph execution across independently loaded CUDA, ROCm and CPU backends. It does not change model placement policy, quantization kernels or model-specific graph construction.

The immediate production symptom is severe prompt-processing loss when a graph repeatedly crosses CUDA and ROCm. Native CUDA and HIP prefill are both healthy on the same host, so the transport boundary must be measured and fixed independently of model routing.

## Current behavior

The scheduler asks the destination backend to perform an asynchronous tensor copy. CUDA-to-CUDA and ROCm-to-ROCm copies can use native device copies and runtime events. CUDA-to-ROCm and ROCm-to-CUDA copies cannot: the backends are separate dynamic libraries with separate runtime contexts, streams and event types.

The generic fallback currently runs once per tensor:

1. synchronize the producer backend;
2. synchronize the consumer backend or its copy-slot event;
3. allocate pageable host memory;
4. copy device to host;
5. copy host to device;
6. free the host allocation.

This is correct, but it serializes a graph boundary at tensor granularity and prevents transfer state from surviving between graph executions.

## Required invariants

- Never pass a device pointer or runtime event from one vendor runtime into the other.
- Preserve the scheduler's existing copy-slot lifetime rules.
- Do not enqueue consumer compute until every required host-to-device copy has been enqueued on the consumer stream.
- Do not reuse staging memory until the consumer event for that copy slot has completed.
- Preserve views, tensor offsets, layouts and user-input ownership semantics.
- Keep native same-runtime peer copies on their existing fast path.
- Retain a correctness-first fallback when host registration or asynchronous access is unavailable.

## Architecture

### 1. Transfer plan

Build a transfer plan for each scheduler split after graph allocation. Classify every split input as one of:

- native asynchronous copy;
- host-resident input or partial-expert upload;
- heterogeneous staged copy.

The plan records producer, consumer, source tensor, destination tensor, byte count, staging offset and copy slot. Classification must be separate from execution; the existing `cpy_tensor_async` interface conflates capability discovery with enqueueing work.

### 2. Boundary transaction

Execute heterogeneous copies as one transaction per split:

1. wait once for the consumer copy slot to become reusable;
2. enqueue device-to-host copies from every producer into its assigned staging range;
3. synchronize each unique producer backend once;
4. enqueue all host-to-device copies on the consumer backend;
5. enqueue the split graph on the same consumer stream;
6. record the existing consumer copy-slot event.

This removes repeated producer waits, repeated consumer waits and per-tensor allocation while retaining explicit, portable ordering.

### 3. Staging pool

Own persistent staging allocations in the scheduler. The pool is indexed by pipeline copy slot and grows to the largest transfer plan seen for that slot.

Initial probing also showed that llama.cpp's legacy `ggml_backend_register_host_buffer` callback cannot be reused for transport. It registers mapped model weights with a read-only flag and is gated by `GGML_CUDA_REGISTER_HOST`; CUDA correctly rejects a D2H copy targeting such a range. Transport therefore needs a separate read/write registration capability with an explicit lifetime.

The conservative design uses distinct runtime-owned ranges:

1. producer D2H into producer-registered staging;
2. host memcpy into consumer-registered staging;
3. consumer H2D from consumer staging.

This adds one host-memory copy but preserves explicit runtime ownership. It still removes per-tensor allocation and repeated synchronization, and the host copy can operate on packed boundary data rather than many small allocations. Sharing one writable range between both runtimes remains a possible later optimization, but it must be validated explicitly rather than inferred from successful registration calls.

Host registration needs a first-class backend capability with explicit register/unregister lifetime. It should not depend on the existing model-mmap registration environment variable, and transport buffers must be writable in both directions.

If dual registration fails, retain the pooled allocation and use the backend's safe synchronous host access path. This still eliminates allocation churn and repeated boundary waits; the diagnostic output must identify the degraded mode.

### 4. Later asynchronous overlap

CUDA events cannot be consumed by ROCm streams and ROCm events cannot be consumed by CUDA streams. Full producer/consumer overlap therefore requires scheduler-level continuation or worker execution, not fabricated event interop.

That is a second phase. First establish the batched transaction baseline. If its remaining wall-time share justifies more complexity, a worker can wait on producer completion and enqueue the consumer transfer and graph while the main scheduler advances independent work.

## Implemented phase-one path

The transaction path is enabled automatically when the scheduler detects a pair of independently loaded accelerator runtimes whose registries both expose writable host-staging support. There is no per-model or environment opt-in. Same-runtime CUDA/CUDA and HIP/HIP transfers keep their native paths, while CPU and unsupported backend pairs keep the existing fallback. Operational rollback is the previous production build.

The implementation currently:

- identifies boundaries between independently loaded accelerator runtimes while leaving same-runtime CUDA and HIP copies on their native path;
- owns separate writable producer and consumer host ranges for every backend pair and pipeline copy slot;
- grows those ranges only when the graph requires more capacity, then reuses them without per-graph allocation;
- waits on the consumer copy-slot event on the host before overwriting staging memory, batches all producer D2H operations, synchronizes each producer once, packs the host data, and enqueues all H2D operations before consumer compute;
- falls back to the existing blocking copy if either backend lacks writable host registration;
- releases scheduler staging explicitly while backend registrations are still alive. This requires the llama context to destroy its scheduler before its backend owners because ordinary C++ member destruction order is the reverse.

The consumer reuse wait must be a host-side event synchronization. A stream-side wait only orders future consumer stream work; it does not stop the CPU from overwriting host staging that an earlier H2D operation may still be reading.

## Measurement contract

`GGML_SCHED_COPY_PROFILE=N` enables aggregate diagnostics every N graph executions without changing copy behavior. Report by backend pair:

- copy count and logical bytes;
- native asynchronous versus fallback copies;
- consumer-slot wait time;
- producer synchronization time;
- consumer synchronization time;
- blocking copy time.

Production conclusions require warm, same-lineage A/B runs and both absolute wall time and transfer share. Profiler-instrumented results are not compared directly with uninstrumented throughput.

## Validation

### Synthetic

- deterministic byte-pattern copies in both CUDA-to-ROCm directions;
- separate producer- and consumer-registered staging ranges; never assume dual registration of one range;
- sizes from tiny activation tensors through representative prefill tensors;
- contiguous tensors and views with offsets;
- repeated reuse of every scheduler copy slot;
- registration failure and pageable fallback;
- CPU-to-GPU partial-expert uploads remain unchanged.

The initial writable-registration probe passed exact-byte validation in both directions. Thirty-iteration means on the RTX 3090 and MI100 were:

| Direction | Bytes | Generic fallback | Persistent staged prototype | Change |
|---|---:|---:|---:|---:|
| CUDA to ROCm | 4 KiB | 25.86 us | 17.91 us | -30.7% |
| CUDA to ROCm | 64 KiB | 45.19 us | 35.96 us | -20.4% |
| CUDA to ROCm | 1 MiB | 446.82 us | 422.34 us | -5.5% |
| CUDA to ROCm | 4 MiB | 1568.02 us | 1572.27 us | neutral |
| ROCm to CUDA | 4 KiB | 29.31 us | 19.06 us | -35.0% |
| ROCm to CUDA | 64 KiB | 46.98 us | 34.77 us | -26.0% |
| ROCm to CUDA | 1 MiB | 436.25 us | 403.88 us | -7.4% |
| ROCm to CUDA | 4 MiB | 1553.95 us | 1546.56 us | neutral |

These are isolated transfer latencies, not model-throughput claims. They establish that reusable writable staging is valid and that its direct benefit is concentrated in small transfers; boundary batching is still required to remove repeated synchronization across a group of tensors.

### Model oracles

An intentionally boundary-heavy Gemma4-31B QAT stress graph placed complete even layers on the RTX 3090 and complete odd layers on the MI100. This creates roughly 60 CUDA-to-HIP or HIP-to-CUDA graph boundaries without splitting operations inside a layer. It is a transport stress oracle, not a recommended model placement.

Warm three-sample `llama-bench` runs, followed by a reversed-order control, measured:

| Test | Control | Staged | Change |
|---|---:|---:|---:|
| PP128 | 179.99 tok/s | 187.85 tok/s | +4.36% |
| TG64 | 19.00 tok/s | 22.21 tok/s | +16.90% |

The original forward control was 180.56 tok/s PP and 19.08 tok/s TG, so the reversed control reproduced it closely. A later single-turn fence after narrowing staging detection to independently loaded runtimes measured 18.9/18.8/19.1 tok/s control and 21.3/21.4/21.6 tok/s staged, with the reversed control returning 17.9/19.0/18.8 tok/s. The exact percentage depends on the frontend and decode length, but the large transport effect is repeatable.

At temperature zero, control and staged execution produced byte-identical generated text. The writable-registration probe also retained exact-byte correctness in both transfer directions.

The same path was neutral on naturally split Qwen3.6-27B prompt processing and improved TG64 from 40.15 to 40.67 tok/s (+1.31%). That graph has few cross-runtime boundaries, which is the expected contrast with the stress oracle.

Step-3.7-Flash was then tested with its production model, explicit CUDA/ROCm/CPU tensor placement, 65K context configuration, Q8 KV cache, MTP plus n-gram speculation and the existing `nmax=2`. Cold decode and the first large-M prompt were discarded independently for each process. The qualified prompt contained 10,204 tokens.

Two deterministic prompts were compared after warm-up and followed by a reversed control:

| Oracle | Control | Staged | Change | Acceptance |
|---|---:|---:|---:|---:|
| C TG128 | 28.96 tok/s | 30.17 tok/s | +4.18% | 107/123 both |
| D TG128 | 33.26 tok/s | 34.01 tok/s | +2.24% | 109/109 both |
| C PP10204 | 193.46 tok/s | 190.49 tok/s | -1.54% | n/a |
| D PP10204 | 191.21 tok/s | 196.17 tok/s | +2.59% | n/a |

Generated output hashes matched for both control/candidate pairs. Averaged against the reversed controls, TG improved by 3.15% and PP by 0.52%; the latter is neutral within run variance. The unusually high absolute TG figures come from the deliberately repetitive oracle and its high MTP acceptance, so the production conclusion is the matched relative gain rather than a new general Step baseline.

- Step-3.7-Flash: retain a later real-chat validation because the deterministic transport oracle intentionally produces unusually high speculation acceptance.
- GPT-OSS-120B: regression fence for the already-fast mixed MoE path.
- No DeepSeek-V4-Flash conclusions until its model support and execution path are stable.

### Acceptance

- output identity or the existing correctness tolerance for every synthetic and model oracle;
- no per-graph host allocation after staging warm-up;
- producer synchronization count bounded by unique producer backends per split, not tensor count;
- no regression on native CUDA-only, HIP-only or same-runtime multi-device execution;
- production deployment only after reversed-order warm A/B runs.
