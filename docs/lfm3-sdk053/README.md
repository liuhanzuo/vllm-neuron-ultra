# LFM3 SDK 0.5.3 source migration

This branch contains an **unqualified source candidate** for the original
`LiquidAI/LFM3-1.7B` revision `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`.
No native import, installation, compilation, device execution, model quality
check, cold-cache trial or performance measurement was performed for this
migration. No speedup from the old Python 3.10 / vLLM 0.11 / NxDI 0.8 package
applies to this new stack.

The target is Python 3.12, vLLM source 0.16.0 (distribution label
`0.16.0+empty`, with real editable source), vLLM-Neuron 0.5.3, Torch
2.9.1+cpu, Torch-NeuronX 2.9.0.2.15.32035+de43f57c, NxDI
0.10.18399+ed62453e, NxD 0.19.28492+435aae2b, NKI
0.5.0+28631259367.ga768afa6, compiler 2.26.6360.0+6f180f47 and Transformers
4.57.6. Artifact admission hashes installed Python/native files and the real
editable vLLM/plugin trees; `+empty` is not treated as missing source.

## Implemented behavior

- A general plugin does nothing unless `NEURON_LFM3=1`. Enabling it registers
  the Transformers config, vLLM architecture and NxDI causal-LM implementation.
  Occupied registrations are rejected before mutations. A process cannot
  switch writer, capacity or context after installation.
- The original model mathematics, TP partitioning, tied weights, FP32 RoPE,
  tokenizer fixtures and official model file hashes are retained. The old
  tokenizer-view schema identifies model inputs only. The independent new
  artifact schema is `vllm-neuron-ultra-lfm3-sdk053-v1`.
- The reference writer is `reference_sdk053_v1`; the new candidate is
  `nki05_typed_decode_v1`. The latter retains the original writer for one row;
  multiple rows use the accepted typed NKI 0.5 sources. Package-relative import
  is the only change to the accepted JIT source. No simulator writeback helper
  is in this path. Native NKI imports are lazy and require loader-owned
  compile/load admission after model/config/software/source identity checks.
- LFM artifact handling runs before the generic loader's config hash, automatic
  path deletion or cache-miss recompilation. Existing missing-manifest,
  incomplete, corrupt or mismatched artifacts fail closed. Fresh compilation
  requires explicit opt-in and unused `.work`/`.cache` sibling directories.
  A completion manifest is published only after both compilation and load.
- The fast CPU sampler is installed on each new `self.cpu_sampler` instance.
  The vLLM 0.16 execute/deferred-sampling protocol remains intact, including
  request-order restoration, grammar masks and then sampling. Explicit request
  generators and unsupported/processed-logprob cases retain the original
  operation. Unseeded samples do not promise identical RNG consumption.
- The separate upstream loader import bug is fixed to import the `Sampler`
  class, rather than the `sampler` module. This is not evidence for the cause
  of the separate Qwen r6 API failure.

## Scope and request state

Only Trn1, original BF16, TP2, B4/B8/B16 and contexts 2048/4096 are candidates.
Every capacity/context combination remains unqualified. The caller must set
`NEURON_LFM3_CAPACITY` and `NEURON_LFM3_CONTEXT` consistently with vLLM's
`max_num_seqs` and `max_model_len`; defaults are 4 and 2048. `block_size` must
explicitly equal context. Prefix caching, chunked prefill and asynchronous
vLLM scheduling must explicitly be disabled before hybrid config validation.
NxDI asynchronous execution, truncation, quantization, KV transfer,
speculation, LoRA and additional parallel modes are rejected. NxDI root
configuration has `max_batch_size == tkg_batch_size == batch_size == B`,
`ctx_batch_size=1`, `apply_seq_ids_mask=True` and inactive slot `-1`.

The scheduler has one contiguous group of 24 attention-shaped proxy specs,
with **B+1 blocks including null**. NxDI owns **B true request slots**. The
proxy byte count is a scheduler budget, not a measurement of HBM use.
Separately, each TP rank retains 13 convolution states and 22 K/V states,
35 total; TP2 has 70. Shapes are `(B,1024,7)` and `(B,2,context,128)`.
The loaded CTE and TKG state dictionaries must expose all 35 named indices per
rank with the expected BF16 shapes. No GPU Mamba cache or fake state-copy
implementation was introduced.

Each scheduler step is validated before freeing/allocating slots or updating
cached requests. This includes all prompts, token IDs, proxy blocks, slot
capacity, decode positions and unsupported scheduler payloads. Finished or
aborted requests can reuse their ID as a new request in the same step.
Intermittently unscheduled requests keep their true slot. Preemption/resume is
rejected because no hybrid state checkpoint/restore implementation exists.
Overlong prompts or exhausted slots fail without slot mutation. A fresh
prefill's actual state clearing and cross-request isolation still require
runtime lifecycle checks.

## Artifact isolation and runtime boundary

`NEURON_COMPILED_ARTIFACTS=/absolute/path/candidate` requires
`BASE_COMPILE_WORK_DIR=/absolute/path/candidate.work` and
`NEURON_COMPILE_CACHE_URL=/absolute/path/candidate.cache`. The parent must be
an ordinary existing directory. Fresh builds require all three paths unused,
`NEURON_LFM3_ALLOW_COMPILE=1`, and `NEURON_LFM3_EXPERIMENTAL_RUNTIME=1`.
The target must be explicit: `NEURON_PLATFORM_TARGET_OVERRIDE=trn1`,
`NEURON_LOGICAL_NC_CONFIG=1`, `NKI_SIMULATOR=0`.
`NEURON_LFM3_NKI_WRITER=0/1` selects reference/typed, and both currently require
`NEURON_CPU_SAMPLER_FAST=1`.

These are plugin configuration requirements, **not authorization to launch**.
The owner still controls serial device admission, resource budgets and the
scientific gates. This source work created no runtime service, scheduling
lock, handoff, compile task or device process. A process-local admission
context only binds writer tracing to the already admitted artifact identity;
it does not replace external queue ownership or device qualification.

## Required qualification, all pending on this new stack

The machine-readable `required-gates.json` preserves the original criteria
and adds 4096 as a supplement. Original controls and old-stack results remain
separately labeled. In particular:

- Preserve the full original primitive matrix (113 graphs including its
  supplements), raw BF16/FP32 bit patterns, input immutability, one-store rule,
  invalid/duplicate/out-of-order slots, int64 normalization and MUST_ALIAS.
- Add LFM-specific `(B,2,4096,128)` standalone and MUST_ALIAS checks for every
  candidate capacity. Another model's four-KV-head 4096 graph cannot substitute.
- Keep B4/B8/B16 numeric frames 32/32/64; cold frames 96/200/400; 1529-step
  decode with 84/168/336 frames; and 96 quality cases per candidate.
- Keep seven fresh cold processes, all 70 states, 140 fill and 140 zero
  roundtrips, five snapshots, all old buckets and request-slot reuse.
- Add crossing 2047/2048, approaching 4095, exact capacity rejection without
  truncation, old-tail clearing and new/old request isolation. Root buckets
  `[4096]` do not imply one graph: default CTE/TKG bucketing still includes
  `[128,256,512,1024,2048,4096]`.
- Keep five workloads, three rounds, shared concurrency 1/4 and separately
  reported capacity concurrency 8/16; 704 measured requests, 189952 output
  tokens and 16 warmups per variant; 384 greedy text pairs per candidate.
  New-stack reference and typed runs need matched conditions. No speedup is
  estimated or inherited.

## Source validation

Run the standalone tests with no installed Torch/Neuron dependency:

```bash
taskset -c 7 nice -n 19 prlimit --as=536870912 --cpu=60 -- \
  python3 -B tests/source/test_lfm3_sdk053.py
```

The tests use stdlib objects and mock dependencies. They execute complete
loader, runner, sampler-wrapper and registration modules where applicable,
including deferred grammar/sampling order; they do not execute tensor math.
They cover artifact load/compile failure behavior, identity/scope rejection,
slot lifecycle, preflight-before-mutation, state ledger and disabled imports.
The 25 tests pass. The first run had five test-harness errors because the
mock package hid the real worker source path; correcting the harness resolved
them without changing implementation behavior. AST/source comparisons only
establish retained code, not model equivalence or SDK compatibility.

`source-origins.json` fixes the original copied bytes. `source-manifest.json`
fixes this delivery's source bytes. The mathematical core, helpers and CPU
sampler are copied byte-for-byte; modeling has only a routing-comment
correction, with unchanged AST. Runtime compatibility of the new source,
compiled state naming, native ABI, compiler cache isolation, all numerical
and alias gates, cold/long/reuse tests, quality and performance remain to be
verified under the owner's serial runtime plan.
