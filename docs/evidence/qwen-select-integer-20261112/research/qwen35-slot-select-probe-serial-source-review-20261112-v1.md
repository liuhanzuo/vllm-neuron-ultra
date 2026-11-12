# Twelve-call select diagnostic: source/mock delivery

This delivery implements the frozen three-graph proposal as a finite serial
successor. It contains executable source and standard-library mocks only. It has
not compiled a graph, loaded a model, executed Neuron, acquired a lock, started a
service, executed History, selected a current catalog, or created runtime input.
The original failed reference and the successful six-call r2 diagnostic remain
frozen. This successor never calls those existing graphs or repeats r2's six
calls. Root must independently review this delivery and bind the then-current
complete history before any actual launch.

## Fixed work and interpretation

The order is arm, then profile, then mapping: exactly 3 x 2 x 2 = 12 calls.

| Fresh graph | Input/output payload type | Required saved HLO |
| --- | --- | --- |
| `one_bf16_select` | BF16 | One BF16 select, update row 0 and runtime slot 0 |
| `internal_s16_select` | BF16 | Two input BF16-to-S16 BitcastConvert paths, four S16 selects, final S16-to-BF16 BitcastConvert |
| `external_s16_select` | S16 | Four S16 selects, S16 inputs/output, no conversions |

Both profiles are unchanged: `original_raw`, `negative_nan_raw`. Both mappings
are unchanged: `idle=(-1,-1,-1,-1)` and `one_active=(3,-1,-1,-1)`. Shape is
`(4,3072,4)`; each cache/update/output contains all 98,304 bytes. The idle oracle
is the full cache. The active oracle preserves cache rows 0 through 2 and copies
update row 0 into output row 3. External S16 tensors reinterpret copied uint8
storage through `view(torch.int16)`; there is no numeric `.to(int16)` cast.

Compiler arguments remain the original `qwen35_predicated_slot_protocol_v1.FLAGS`,
including `--auto-cast=none`, `-O2`, `--lnc=1`, and the original tensorizer options.
There is no flag sweep or invented fast-math option. Every arm compiles once into
its fresh directory. Each traced object is deleted before loading its saved
artifact, and that loaded object is deleted before tracing the next arm. All
actual calls use core 0. There are no timing calls, fallback, retries, additional
graphs, or automatic successors.

A byte mismatch is recorded and the remaining scheduled calls continue. A trace,
HLO, invocation, runtime, input-integrity or ownership exception aborts the
successor, retaining the incomplete report and completed captures. Diagnostic
completion does not mean that every output matched.

## Whole HLO and capture checks

`qwen35_slot_select_probe_hlo_v1.py` parses the complete saved compiler-input HLO
with bounded stdlib protobuf readers. Descriptor field numbers, primitive enums,
shape fields and slice fields are checked against the pinned SDK Python source
without importing that SDK. The gate resolves the actual entry root, operands,
parameters, tuple projections and called computations. It proves the exact
cache/update/slot origins, update row for each select, signed destination-index
equality, branch order, dtype and complete output shape. Every saved instruction
and computation must participate in that output dataflow.

The gate rejects numeric Convert, arithmetic, custom-call/backend targets,
control-only paths, cyclic/missing operands, unused decoy graphs and wrong input,
slot or update rows. Broadcast validation includes element correspondence:
predicate expansion must use destination axis 0, and whole update rows must
expand on axes [1,2]. A predicate incorrectly broadcast on trailing axis 2 is
rejected even though that axis also has length four. Shape-preserving broadcasts
and reshapes preserve flat element order.

Called computations are evaluated per invocation. A shared BF16-to-S16 callee
used by both inputs therefore contributes two bitcasts; the output bitcast makes
three. The checker accepts both this shared-callee form and three inlined
bitcasts. It does not mistake two distinct bitcast instructions for only two
actual reinterpretation paths. Missing or altered compiler forms fail closed;
the runner does not substitute a graph or weaken the gate.

Whole artifact hashes are checked before/after actual `torch.jit.load` and after
all four calls. Actual loaded forward source must contain the Neuron `forward_v2`
invocation. The graph is explicitly moved to core 0. The report and immutable
artifact sidecar must agree on the complete HLO reference and recomputed proof.
These checks establish the saved frontend compiler-input graph and invocation
path; they do not identify a final neuronx-cc optimization pass.

The report saves four full cache/update input files, two full slot files, all
twelve complete output files, and every differing byte offset as little-endian
u32. Every call also records all three actual input-after hashes. Validation
rereads every complete file, independently recomputes all offsets and expected
bytes, checks exact call order/type/shape, and requires unchanged inputs. Even
if every output byte differs, raw inputs plus outputs plus offsets total
6,291,520 bytes, below the unchanged 8 MiB diagnostic payload cap.

## Preserved lifecycle

The serial entry reuses r2's `History`, `RetainedLock`, `Service`, running ACK,
actual child wait, cleanup and retained-until-drained behavior. It retains
`SDKEnvironment` and the exact r2 leaf guard, including installation of the
source-pinned context repair at the first post-native `before_compile:` boundary.
The complete outer function is AST-identical to r2 after only namespace/schema
normalization. Environment/resource/scan helpers, owner/group checks and child
capture helpers are AST-identical. Original sources are never edited.

The existing lock identity, isolated Python 3.10 interpreter, CPU affinity [6,7],
nice 19, core 0, clean environment, memory/entry/file/storage floors, output caps,
ACK deadlines and drain policy are retained. All three graphs share the original
single diagnostic layout deadline of 900 seconds. The supervisor remains bounded
at 7,200 seconds and the outer at 9,000 seconds. No extra reservation or weakened
resource policy is introduced.

## Root integration

The source inventory is additive. Root must merge its `static_local_source_sha256`
and `sdk_source_sha256` into the complete then-current historical source map,
rejecting any conflicting SHA. It must retain all original historical metadata
identities and source witnesses. The inventory itself is not admission, and this
delivery provides no selected current binding.

The exact spec fields are:

```text
schema = "qwen35_slot_select_probe_serial_spec_v1"
run_name = root-selected fresh bounded name
output_root = "/mnt/nvme/neuron-work/qwen35-slot-select-probe-" + run_name
history_overlay = root-selected complete current overlay hash reference
current_serial_catalog = then-current complete catalog hash reference
catalog_predecessor = independently reviewed predecessor catalog hash reference
source_sha256 = complete historical + delivery-local + SDK source map
input_extension = complete metadata path -> {sha256, identity} map
running_handoff_path = fresh research/continuation-handoff-<name>.json
lfm_r2_scope = original required complete scope hash reference
source_delivery = this new complete source-delivery hash reference
primitive_manifest = contract.MANIFEST, unchanged
workload = diagnostic.workload(), unchanged
limits = contract.LIMITS, unchanged
lock_identity = original LOCK_IDENTITY, unchanged
native_python = contract.PYTHON, unchanged
automatic_successor = false
```

All ordinary hash references above contain exactly `{path, sha256}`. The frozen
primitive manifest additionally contains its original byte count. Both catalog
references must appear in `input_extension` with their exact SHA and actual
metadata identity, as must all three frozen `diagnostic.INPUTS` metadata files:
r2 root actual-exit review, r2 independent raw review, and the mechanism/source
review. These references do not authorize reading or rerunning old model
artifacts. Preserve the complete existing history domain when adding them.

`validate_catalog_successor` requires the original catalog schema, root reviewer,
passed/no blockers and full-inclusion fields. The current run list must retain
the entire predecessor list byte-for-JSON-value, in order, with no deletion,
rewriting, reordering or duplicate terminal path; only appended rows are allowed.
The completed SDK base must be identical. An equal catalog is permitted when
nothing new has completed. The relation has no fixed current row count. This
pure relation is additional to the original History collectors, which still
authenticate every current row, complete scope, all sources/metadata, then-current
idle handoff, and physical ownership before admission. Do not replace those
collectors with this prefix check or pin a historical total as today's count.

Root's review envelope remains exactly
`qwen35_sampler_runtime_source_review_v1`, reviewer `/root`, `passed=true`,
`blockers=[]`, the exact spec reference and source map, and
`launch_authorized=true`. That review is not created by this source delivery.
The future outer command is:

```text
/home/ubuntu/venvs/vllm_neuron/bin/python -I -B \
  -X pycache_prefix=<fresh-output-root>/bytecode-never-created \
  /home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_slot_select_probe_serial_v1.py \
  --mode outer --spec <root-spec-path> --spec-sha256 <root-spec-sha256> \
  --review <root-review-path> --review-sha256 <root-review-sha256>
```

Root must observe the actual tool session and supervisor ownership, then issue
the original running handoff/ACK for that exact admission and external tool
session. The event is `qwen35_slot_select_probe_running_handoff_required`.
Supervisor/layout modes remain internal admitted entries. After the actual
external tool terminal, root must independently establish owner disappearance,
complete service/cgroup drain and original lock release, then reconcile that
actual terminal. The runner deliberately leaves external tool exit pending and
never attests to its own exit. Root must preserve a failed terminal as failure
and append the completed run legally to subsequent full history.

## Source/mock evidence and limits

Final check: pinned Python `-I -B check_qwen35_slot_select_probe_serial_v1.py`,
16 tests passed, session 25215, terminal chunk `e5eb10`, actual exit 0.
Native import blockers cover torch, torch_xla, torch_neuronx, NumPy, ctypes,
neuronxcc, ml_dtypes, libneuronxla, transformers and vLLM. The successful leaf
mock runs the actual new leaf with three synthetic compiler-format HLOs and
explicitly fake artifacts: all 12 calls, first-arm mismatches, later-arm matches,
complete captures, actual validation and peak one live fake compiled object.

Negative controls cover wrong broadcast axes, dtype, numeric conversion,
predicate, slot/update rows, host bypass, detached graphs, duplicate IDs,
missing selects, malformed protobuf, HLO failure before any runtime call, host
forward rejection, runtime exception with no retry or later compilation, full
offset corruption, report/input/core drift, immutable workload/flags/resources,
missing or altered catalog metadata bindings, prefix rewrite/reorder/removal,
duplicate terminals and false root review. Variable-length catalog mocks use
2->2, 2->5 and 3->8 rows. Complete inventory generation and every resulting
source/document hash are checked, including exclusive output creation.
`--help` stays inert. Frozen r2 and mechanism/proposal
source hashes are checked. Original static-closure parsing emits existing
invalid-escape deprecation warnings; no check failed.

No actual compile, runtime fidelity, speedup, model acceptance or complete
primitive qualification is claimed. The original eight-layout qualification
remains incomplete. Even twelve bitwise matches would require a separately
reviewed successor to pass all original eight layouts, both modes, all 625
mappings, raw and finite profiles, persistent-state/old-read/input checks and
the original timing schedule. No such successor is automatically launched.
