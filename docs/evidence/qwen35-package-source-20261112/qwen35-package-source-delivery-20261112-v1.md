Qwen3.5 standalone package source delivery, 2026-11-12

Implemented in `/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107`: fourteen new Python modules under `vllm_neuron/models/qwen35/`, source-only tests/inventories under `test/qwen35/`, and an eight-line opt-in admission change to the uninstalled checkout's generic loader. No installed, frozen, or measured experiment source was edited. No weights, compiled artifacts, services, locks, catalog, or handoff were changed. No native libraries were imported and no model/compiler/server was run.

The source ports the original Qwen/Qwen3.5-2B text-only BF16 TP2/B4/context4096 corrected split baseline. `primitives.py`, `text_core.py`, and `modeling.py` preserve the original numerical bodies; the sole mathematical-source substitution is the measured `torch.split` to `tensor_split` helper. Original source SHA256s and an AST contract are retained. There is no failed GDN candidate, alternate architecture, NKI cache path, or sampler copy. Existing `worker/cpu_sampler.py` and its `NEURON_CPU_SAMPLER_FAST=1` opt-in are reused by the normal runner.

The model retains all 24 hybrid layers, FP32 recurrent state/A_log/dt_bias/rotary frequencies across BF16 casts, original BF16 checkpoint bytes, all six context and six decode routes, single-token prefill padding, functional writes preserving inactive slot -1, original context masking, and maximum-size KV aliases across decode buckets. `state.clear_states` copies CPU zeros into the same 96 device tensor bindings and checks raw bytes after each copy. It never reinitializes or reloads graphs. It is an explicit cold-diagnostic API; serving request lifecycle is unchanged. `validation.py` retains pure numerical/state and long1541/reuse/partial-batch/resume frame contracts for a future external runner.

`prepare_view.py` creates a fresh text configuration view using the exact original transformation. It verifies pinned Hub Git-blob/LFS identities, leaves checkpoint/tokenizer bytes unchanged, and publishes only its own view manifest. `registration.py` checks all occupied identifiers before registering the process-local Transformers, vLLM, and NxDI classes. Explicit `serve.py` repeats registration in spawned workers; setup.py was not changed and no implicit general plugin entry is claimed. `build.py` provides an explicit serial compile/load entry from a complete NeuronConfig keyword JSON.

Artifact admission follows the LFM pattern before generic path deletion or ValueError fallback. It binds every packaged source, shared loader/runner/sampler source, pinned model/view, complete configuration, pinned runtime versions/SDK sources, Trn1 identity, sampler selection and relevant compiler environment. The SDK's documented quantize_clamp_bound +Infinity is compared without confusing JSON booleans/integers or arbitrary nonfinite values. Existing mismatched/incomplete directories are rejected without deletion/retry. A missing artifact requires `NEURON_QWEN35_ALLOW_COMPILE=1`, a fresh explicit `NEURON_COMPILED_ARTIFACTS` path, and a preexisting parent. Compile/load completion publishes a new manifest only after both succeed. There is no invented qualification receipt/authority or mandatory approval workflow.

New packaged graphs have a new source identity. Frozen baseline artifacts and their provenance do not fit this package and were neither relabeled nor modified. Changing the shared loader also changes source identities for other packages that fingerprint it; their existing manifests may correctly reject this checkout. Do not regenerate previous artifact provenance to hide that difference.

Source validation (run from the checkout):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 test/qwen35/run_source_tests.py
PYTHONDONTWRITEBYTECODE=1 python3 test/qwen35/source_inventory.py
```

The guarded runner passes 24 tests and rejects every actual torch/Neuron/Transformers/vLLM/NumPy import; fake modules provide the tested control flow. It checks original AST body equivalence and mutation detection; split indices; original one-token route; independent/inactive slot writes; two dirty-state clears preserving all 96 bindings; failed copy detection; narrow config transform; registration collision/idempotence behavior; source-bound admission; explicit compile opt-in; manifest publication only after mocked compile/load; partial-build preservation; and propagation of cached load errors. These are source/mock checks, not CPU tensor or device correctness results.

`test/qwen35/dependency_inventory.json` records every direct Python import and exact packaged/shared source hash, the original experimental source inventory, runtime versions, and 20 pinned SDK source hashes. This is a direct static source import closure, not a dynamic native-library dependency audit. No experiments-directory import or absolute experiment runtime path remains.

Future runtime invocation shape, provided for source review only (not executed):

```sh
# In the independently scheduled pinned runtime, prepare a fresh view first:
python -m vllm_neuron.models.qwen35.prepare_view ORIGINAL_PINNED_MODEL FRESH_TEXT_VIEW

# Use explicit experimental serving, CPU sampling, original settings and a fresh artifact path.
NEURON_QWEN35=1 NEURON_ON_DEVICE_SAMPLING_DISABLED=1 \
XLA_DISABLE_FUNCTIONALIZATION=1 NEURON_COMPILED_ARTIFACTS=FRESH_ARTIFACT_PATH \
NEURON_QWEN35_ALLOW_COMPILE=1 \
python -m vllm_neuron.models.qwen35.serve \
  --model FRESH_TEXT_VIEW --served-model-name qwen3.5-2b-text \
  --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 4 \
  --no-enable-prefix-caching --host 127.0.0.1 --port 8010 \
  --additional-config '{"override_neuron_config":{"on_device_sampling_config":null,"apply_seq_ids_mask":true,"context_encoding_buckets":[128,256,512,1024,2048,4096]}}'
```

Use `NEURON_CPU_SAMPLER_FAST=1` to select the existing fast sampler; omission selects the existing standard sampler. The selected value is part of package artifact identity. Future package qualification must explicitly record which was used. Native compiler/runtime environment controls and serial ownership remain the external runner's responsibility.

All real gates remain pending for this package: native imports/registration, exact derived configuration compatibility in the pinned SDK, fresh compilation of 12 graphs and their tensor_split HLO inspection, unchanged tokenizer token IDs, canonical32 logits, cold84 including prefix2052, true-zero device-copy behavior, long1541/87 with reuse/inactive resume, greedy/seeded sampler histories, HTTP quality96, workloads, spawn behavior and performance. The external full-model qualification runner has not been ported or run. Source equivalence does not establish fresh graph identity or numerical correctness. The explicit entry permits experimental execution under the normal external scheduling process; this source delivery makes no device, quality, throughput, or deployment claim.
