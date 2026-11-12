# Complete Qwen slot matrix with an explicitly new integer reference

This delivery contains source and standard-library/mock validation only. It has
not imported native packages, invoked Neuron/compiler/model code, selected
current History, acquired a lock, launched a service, or created a runtime spec,
review, catalog, or handoff. It establishes no new speedup or qualification.

## Scientific change and retained failure

Only the compiled reference writer changes. BF16 cache and update tensors are
bitcast to S16, passed through the original `qwen35_text_port.store_slots`, then
bitcast back to BF16. FP32 uses S32 and returns FP32. All four lowerers call
`value.bitcast`; there is no numeric dtype conversion. The factory checks the
original exact layout, floating dtype, and four signed int64 slots.

The NKI candidate, host oracle, original `self.cache[slots.clamp_min(0)]` old-value
gather, alias declarations, call order, paired timing, CPU materialization,
comparators, profiles and compiler flags are unchanged. Source tests compare
the complete normalized leaf and serial AST against the frozen originals.

The original floating-reference failure remains a failure. Its actual exit is
bound by SHA-256 alongside the actual 12-call select diagnostic (session 22831,
terminal chunk `a07595`, exit 0) and root finalization. In that diagnostic four
single-BF16-select calls differed by 139/136/155/148 bytes, while eight internal
or external S16-select calls were byte-identical. These are supporting probe
results, not qualification of the complete matrix, FP32, persistent state or a
model. No previously failed artifact was changed.

Reports and timing summaries identify `integer_bitcast_store_slots_v1` and set
`original_model_baseline_speedup_claim=false`. Any later ratio is the NKI
candidate relative to this new reference, with the existing host-call timing
scope. It must not be reported as an original baseline or model speedup.

## Complete original matrix and resource policy

The matrix remains conv BF16 `(4,3072,4)`, recurrent FP32 `(4,8,128,128)` and six
KV BF16 layouts `(4,1,L,256)` for L = 128, 256, 512, 1024, 2048, 4096. Each keeps
all 625 correctness mappings, raw and finite profiles, standalone and
persistent MUST_ALIAS modes, and reference and candidate variants.

There are 32 graphs and 46,368 planned calls: 40,000 correctness, 1,024 warmup,
5,120 measured and 224 resets. Each layout has 5,796 calls. Complete updated
output, unchanged input, old-read and actual persistent device-state bytes
remain required, including signed zero and NaN payload negative controls.

The inherited limits remain 900 seconds/layout, 7,200 seconds total, 9,000
seconds outer; memory max 6 GiB/high 5 GiB, zero swap, CPUs 6 and 7, nice 19,
visible core 0; 1 GiB output/layout, 8 GiB total, 128 MiB/model artifact. Root
free floor is 28 GiB; NVMe floor is 30 GiB plus remaining output reservation and
2 GiB headroom, hence 40 GiB initially. The separate LFM requirement is
unchanged. None of these budgets has been measured on this successor.

## Files and integration

- `../qwen35_integer_slot_reference_v1.py`: lazy integer-reference factory.
- `../qwen35_integer_reference_protocol_v1.py`: original science functions plus
  explicit reference identity and SHA-bound prior evidence.
- `../qwen35_integer_reference_device_leaf_v1.py`: complete eight-layout leaf
  using the new writer only in its compiled reference variant.
- `qwen35_integer_reference_serial_contract_v1.py` and
  `qwen35_integer_reference_serial_v1.py`: complete original serial lifecycle,
  resource/output checks and exact report validation with new namespaces.
- `prepare_qwen35_integer_reference_serial_v1.py`: additive source inventory;
  its closure starts from every primitive manifest source, including original
  files referenced as text by the source tests.
- `check_qwen35_integer_reference_source_v1.py` and
  `check_qwen35_integer_reference_serial_v1.py`: standard-library/mock checks.
- `qwen35-integer-reference-primitive-source-manifest-20261112-v1.json`:
  immutable scientific source references and workload.
- `qwen35-integer-reference-serial-source-delivery-20261112-v1.json`: complete
  source inventory with 622 local source pins and 2,822 SDK Python source pins.

Future root integration uses the new contract directly: schema
`qwen35_integer_reference_serial_spec_v1`, `contract.MANIFEST`,
`contract.protocol.workload()`, `contract.LIMITS`, and the source delivery above.
The output namespace is
`/mnt/nvme/neuron-work/qwen35-integer-reference-slot-<run_name>` and service
suffix is `-integer-reference-slot`. Root must supply then-current complete
source/dependency pins and metadata identities, the full predecessor History,
current catalog and the original root review envelope. All three
`protocol.REFERENCE_EVIDENCE` rows must be present in `input_extension` with
their exact hashes and observed identities. The inventory is additive to the
original complete history; it is not a replacement history or launch grant.

The new serial entry reuses the exact original History/Authority,
RetainedLock, Service, root ACK, physical drain, actual subprocess wait and
external-tool terminal reconciliation. SDKEnvironment initialization, scoped
context repair and restoration are unchanged. No runtime binding is included
in this source-only delivery.

## Checks actually performed

The source suite passed all 10 tests (`7c4b28`, exit 0), including all eight
layouts times all 625 mappings through fake integer bitcast/store wiring. Fake
tensors carry representative word rows and declared layout metadata; this is
not a full-size tensor or hardware test.

The selected serial suite covered 24 different tests. The first run (session
77532, terminal `4bde65`, exit 1) passed 23 and exposed one inventory gap:
AST import discovery omitted original leaf/test files referenced only as text.
The preparer now seeds every manifest source. The previously failing real
source-inventory test then passed (session 3717, terminal `442073`, exit 0),
including rejection of missing or forged hashes for each prior evidence file.
No original source was edited to obtain these passes.

Selected serial classes were ContractTests, StorageTests, ScheduleTests,
CaptureTests, LeafCallbackTests, OriginalServiceTests, OuterLifecycleTests and
SDKEnvironmentTests.test_complete_source_pins_and_frozen_leaf_protocol_unchanged.
The other nine inherited SDK tests were not rerun: SDK behavior was unchanged.
Original-source parsing emitted existing invalid-escape DeprecationWarnings;
these did not fail the checks.

Inventory generation completed with session 46114, terminal `2ca8e4`, exit 0.
The original primitive manifest sources, original serial-v3 delivery sources
and all original protocol source pins were rechecked unchanged. These results
are 34 distinct passing source/mock tests, not 34 device tests.

The full native matrix remains pending. In particular source tests do not prove
the emitted FP32/S32 lowering, BF16 payload fidelity in the unchanged old-value
gather, or persistent state behavior. Any such failure must remain an actual
failure with the complete saved report; the workload or byte checks must not
be reduced to obtain a passing result.
