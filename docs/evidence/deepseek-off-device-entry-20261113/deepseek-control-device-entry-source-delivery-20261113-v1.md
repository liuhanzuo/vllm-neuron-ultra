DeepSeek observer-off DEVICE load leaf

The existing deepseek_cold_control_entry_v1.device already supplies the full
observer-off scientific protocol, but requires a model loaded by its caller.
Source search found no fresh-process off DEVICE loading entry. The new
deepseek_cold_control_device_entry_v1.py supplies that missing connection and
external native-reference authentication; no existing source is modified.

The exact callable is:

```python
from deepseek_cold_control_device_entry_v1 import run

result = run(
    artifact=original_case['artifact'],
    expected_artifact_sha256=root_authenticated_original_artifact_hashes,
    case=original_case,
    destination=fresh_off_metadata_root,
    reference_directory=fresh_off_metadata_root / 'cold-reference',
    data_root=fresh_off_device_payload_root,
    original_cold_metadata=root_authenticated_complete_original_cold_report,
    expected_reference_sha256=root_authenticated_fresh_off_native_hashes,
    execution_guard=owning_serial_execution_guard,
)
```

The artifact hash map must contain exactly the original artifact's model.pt and
neuron_config.json. The latter retains the fixed original SHA
26b0382de78b251aaa45fcd2c70e632140bba9f1120579e3888fed5685860a18 and8378 bytes.
The reference map must contain exactly destination/cold-reference/cold-reference.pt
and cold-reference.json, from the actual completed fresh observer-off native
phase. Expected hashes come from independent root admission, never by deriving
expectations from these files during load. Every artifact/reference file is
stream-hashed before native import; artifact files are rechecked immediately
before model.load and both references again before the frozen driver can call
torch.load. The complete original config is authenticated before parsing and
must roundtrip exactly through the original config class.

The artifact and model-loader paths must equal the original complete case and
the accepted graph module's original path constants. The model is exactly
NeuronDeepSeekQwen3ForCausalLM with the original config and BF16 checkpoint view,
CPU sampling, TP2, B4, context4096 and all12 original graphs. The leaf calls
cls.get_config_cls().load, cls(original_model_path,config=cfg), then
model.load(original_artifact,skip_warmup=True). It never compiles or constructs
the capture model. Original config requires no saved weight shards and no skip
of runtime sharding; an artifact/weights substitute is rejected. The outer
runtime still authenticates every original model/config/weight/SDK file and
reserves runtime sharding, shared memory and load resources. A two-file artifact
map is not a substitute for that complete external model/environment closure.

The leaf calls unchanged validate_deepseek_cold_control_v1.main once. All nested
scientific loops remain mechanically identical to the original cold driver:
seven prefixes, two fixtures, four slots,42 model calls,84 device frames,6048
full-history KV records,288 fill/288 zero controls, actual warmup and144-state
snapshots. No fourteen-failure subset or new thresholds are introduced.
The old control Capture and lifecycle remain unchanged and own their original
full output writing; the leaf adds no extra tensor capture or reset.

Only the exact built-in SystemExit with original failure text can proceed.
Revoked guard, changed success, arbitrary exit/error and incomplete evidence
remain failures. Classification requires the complete84-frame KV manifest,
original metadata comparison and full bridge classification with all14 original
failed indices, plus unchanged state-copy/lifecycle validators and all144
warmup transitions/bindings. Raw equality remains pending for the independent
original-to-fresh-off verifier; model qualification, capture neutrality and
speedup claims remain false.

The process must be fresh before native imports and have no existing trace.
Output roots must be ordinary, mutually disjoint from each other and the
original artifact, outside the protected old diagnostic namespace, and all
device outputs absent, including partial temporary reports. Guard wrappers
surround original wrapper forwards without changing results. Both original
own/inherited _forward attributes are restored even if installation or the
driver fails. No process, queue, lock, History, cleanup or lifecycle admission
is created by this leaf. The external owner supplies seriality and handles
actual exit/drain evidence.

Eighteen audited tests use in-memory fake models/paths/artifact fingerprints and
the actual unchanged full metadata/manifest validators. They cover the complete
42-call/84-frame driver invocation; exact load arguments; hashes before native
import/deserialization; changed artifact/reference/config/case/state layout;
native/fresh-path checks; partial route installation; same-message guard exit;
unexpected success; missing/reordered/short manifest; all84 metrics/all144
states; and byte-frozen original driver/model/control sources. Actual runtime
JSON, model, artifact, native-reference, tensor, proc, lock and native imports
are prohibited by the audit runner. Tests establish wiring and rejection paths,
not actual Neuron execution or numerical neutrality.

The static local import closure contains89 Python files, discovered from source
AST without importing them or reading any existing runtime/closure manifest.
It is a conservative local closure, not complete SDK/dynamic source admission.
Root must combine the new entry pin with its already accepted full source,
weight and runtime closure when it later selects an actual run.

The original two-arm output bound remains47,150,344,192 bytes, excluding compiler
scratch/cache/NEFF/model and weight/shared-memory peaks. No space saving is
claimed. Actual fresh on/off native/device exits, full126 raw logits/9072 KV/
144-state neutrality, original long2041/93 checks, request-slot reuse,
partial/inactive recovery, quality96 and matched HTTP qualification remain
required. This delivery selects no runtime references and grants no launch.
