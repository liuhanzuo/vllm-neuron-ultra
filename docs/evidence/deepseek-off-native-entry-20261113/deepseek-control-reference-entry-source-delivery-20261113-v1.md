# DeepSeek observer-off native reference loader

The existing `deepseek_cold_control_entry_v1.reference` accepts an already
loaded native model. Source search found no fresh-process native loader for
that arm. The new `deepseek_cold_control_reference_entry_v1.py` supplies this
missing connection, without changing the frozen control leaf or scientific
driver. It does not launch a process, compile, create a queue, acquire a lock,
inspect physical History, delete data or grant runtime admission.

## Callable interface and outer obligations

```python
from deepseek_cold_control_reference_entry_v1 import run

result = run(
    destination=fresh_off_metadata_root,
    reference_directory=fresh_off_metadata_root / 'cold-reference',
    data_root=fresh_off_native_payload_root,
    execution_guard=owning_serial_execution_guard,
)
```

These are placeholders, not a selected current run. The outer worker must
authenticate the complete original model weights, model config, native Python
and library environment, all sources and protocol inputs before invoking this
leaf, including the original long-decode reference metadata read by `build()`.
The external guard must raise if source/weight/config/input identities, serial
ownership or reserved resources cease to be valid. Its return value is ignored;
it is not a boolean permission flag. The leaf does not derive expected weights
or config hashes from files it is about to use, and its native-source check is
not a substitute for this complete outer admission.

Use a fresh process for each of native/off, native/on, device/off and device/on.
The entry refuses any already imported Torch, NumPy, Transformers, Neuron or
vLLM package, including dotted submodules, and refuses a pre-existing Python
trace or disabled assertions. Its module imports only the standard library.
Native imports happen only after the external guard and output-path checks.
The caller selects the original native environment; the leaf changes no
environment variable and uses the original two-thread Torch setting.

Output metadata and payload roots must already exist as ordinary absolute,
disjoint directories, separate also from the fixed official model directory.
Symlink ancestors and the protected `deepseek-cold-kv-diagnostic-v2` namespace
are rejected. `reference_directory` must be exactly `destination/cold-reference`.
That directory, `destination/cache-captures/native` and `data_root/native` must
all be fresh, including absence of dangling symlinks. The original leaf and
Capture create their own subdirectories; this loader does not clean or retry.

## Original model and entire scientific loop

The model path remains `/mnt/nvme/models/DeepSeek-R1-0528-Qwen3-8B`. The loader
uses the original `qwen3_controls_reference.canonical_config` YaRN translation,
then exactly `hf.Qwen3ForCausalLM.from_pretrained` with original config,
`dtype=torch.bfloat16`, `attn_implementation='eager'`, `local_files_only=True`
and `output_loading_info=True`. It rejects incomplete checkpoint loading,
another model class, a parameter count other than 8,190,735,360, non-CPU
parameters, tied output head/embedding, non-BF16 dtype or non-eval mode.
The fixed HF source SHA remains
`64fec98db8cd0edaa740baf026be6adb1d5011e38bfa050ccdf6693b1da9f95d`.

It calls the entire unchanged `deepseek_cold_control_entry_v1.reference` once.
That original leaf in turn calls the entire unchanged
`validate_deepseek_cold_control_v1.reference`, retaining seven prefixes, both
input fixtures, original prefill plus two cached updates, all 42 forward
calls/full-vocabulary logits and all 3,024 full-history KV records. No model
loop, prefix, state, threshold or capture method is copied into the new leaf.

The loader temporarily wraps the model's existing `forward` to invoke the
external guard immediately before and after each complete call, returns its
result untouched, and counts 42 completed calls. This adds an ownership/resource
check boundary; it adds no tensor-capture hook and changes no inputs or model
state. `finally` restores an existing instance attribute or removes the
temporary attribute to restore the original inherited descriptor. This also
runs for load-following admission failures, partial wrapper installation and
driver/capture failures. Captured partial files remain evidence. A native
failure, including the device gate's expected string, is never classified as
successful native completion.

Successful return additionally requires the original exact control receipt,
all 42 completed calls, no residual trace, the complete original native
manifest via `check_manifest`, and complete finite reference metadata with
original prefixes, revision, criteria and source bindings. The original
manifest validator checks the ordered 42 frames and 3,024-record count; it is
not a full raw-content comparison. Root still has to authenticate actual
producer exits and all output hashes, then run the existing complete raw
reproduction/neutrality verifier independently. No tensor is deserialized by
these new completion checks.

## Source and synthetic validation

20 audited source/in-memory tests passed, actual terminal `bee03b`, exit 0.
They execute both the real frozen control leaf and the real entire original
reference loop with fake Torch tensors, model and Capture. This exercises 42
calls and ordered 3,024 stage-record enumeration. The original
`canonical_config` function body executes with a fake config class, and the
original full native manifest validator executes unchanged.

Tests cover loader arguments and YaRN translation; pre-imported native
packages/trace; guard failures before import, config, load, forward and finish;
all checkpoint loading errors; parameter/head/CPU/BF16 checks; inherited and
own forward restoration; partial installation; omitted/reordered/changed final
frames; altered final reference criteria, source or completion state; wrong,
overlapping, existing or symlink output paths; and byte-frozen dependencies.
The first test attempt failed because the test harness bound a `Path` mock as
an already-bound method. Correcting that in-memory mock fixed the tests; no
runtime implementation change was needed for that failure. The first failure
log remains preserved.

The runner rejects real native imports, process/lock/network operations,
`/proc`, `/sys`, `/mnt/nvme` and runtime/payload JSON or binary file access.
All fixture files and writes remain in memory. These tests do not establish
actual model loading, raw KV bytes, numerical reproduction or performance.

The source-only AST closure records hashes and stable identities for 83 local
Python files. It is a conservative static local dependency set, not complete
external SDK/library or dynamic-import admission. The owning runtime must
combine it with the unchanged full original source and input closure.

## Remaining work

Root source review, concrete serial integration, complete current source/input
and resource admission, actual fresh native/on/off and device/on/off tool exits,
and full 126-logit/9,072-KV/144-state reproduction and neutrality remain due.
The two-arm raw-output bound remains 47,150,344,192 bytes, excluding compiler,
model artifacts, weights/sharding and shared memory. No compression or sharing
saving is assumed. Original long decode 2,041 steps/93 positions, request-slot
reuse, partial/inactive recovery, quality96 and matched HTTP gates remain.
This source delivery has no model qualification or new speedup result.
