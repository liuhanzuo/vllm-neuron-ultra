# Outer types.py source timing and minimal r2 wiring

The r1 record establishes that `types` was loaded from the expected path but had no `entry.sources` receipt. It does **not** record whether `types` was already present at literal entry. Its timing cannot be reconstructed more precisely from r1.

The v1 child imports `json` at line 66, before the candidate PureLoader is even defined at line 101. The inspected outer Python source contains the unconditional import chain `json.__init__:106 → json.decoder:3 → re.__init__:124 → enum:3 → types`. Thus `types` is present by completion of that first JSON import. The late finder cannot observe its earlier execution: ordinary imports reuse `sys.modules`.

The literal helper and child initially import only `sys`, `posix` and `time`. This does not prove that interpreter initialization left `types` absent. The inspected source chain is a source explanation, not a retroactive timing trace. A future earliest checkpoint must distinguish “already loaded at the checkpoint” from “loaded by the JSON chain after the checkpoint.” Even the former is not, by itself, proof that interpreter startup was the cause.

The new `minicpm5_bootstrap_types_source_v1.py` is an inert, import-free component for the existing exact literal. It contains exactly the already saved outer `types.py` pin: 10,993 bytes, SHA256 `345474ef027a1273f353da9bdc1f7c18f65335e72e681bcc0376774cc51f2405`. It wraps the retained first finder for that one name and delegates all other names unchanged. It uses the existing Entry source reader, no-cache fence, shared deadline and fixed 16 MiB whole-source cap.

If `types` is absent at its first checkpoint, its loader verifies the exact source before compiling it. It records `entry.sources` only after `exec_module` has returned successfully and the same module remains registered with the exact expected file/spec/loader. Calling `get_code` or failing during execution does not create an execution receipt.

If `types` is already present, the component preserves that module object. It checks exact source/spec/ordinary-loader correspondence and verifies the current pinned source through Entry, but returns `executed_through_pinned_loader=false`, `execution_receipt=null` and `original_execution_provenance_gap=true`. This is current file and module-metadata correlation, not proof of the bytes previously executed or the current definitions in memory. It neither unloads nor reloads the module and never inserts a synthetic receipt into `entry.sources`.

No existing child, literal builder, runner, plan or admission is modified. Root's r2 integration still needs these five explicit connections:

1. Add this exact helper source to the reviewed literal and source closure. Execute its definitions in a private helper namespace alongside the existing no-cache helper, before calling child `main`. Do not import a local module before `Entry.plan` exists.
2. In a new immutable child successor, instantiate `TypesSourceObservation(sys)` for `outer312` immediately after argument validation and before the earliest maps snapshot. SDK behavior is unaffected.
3. After the existing original-finder installation and exact parent ACK, call `tracker.install_before_json(entry, external, helpers['SOURCE_TOTAL_CAP'])` immediately before the existing `import json`. Immediately after successful import, call `tracker.after_json()`.
4. Preserve the existing candidate-table authentication and later DiscoveryFinder. Its `original_finder` must be the installed early finder; this preserves the existing retained finder ownership check. Before final observation output, call `tracker.finish(table['outer312']['pure_source_candidates'][PIN['path']])`. The static early pin must exactly equal this later authenticated row. Store the returned record as a separate types source observation.
5. Keep preexisting-module provenance gaps explicit in the output and any future authority gate. The existing boolean “missing content origins” covers native mapping gaps only and must not silently hide this separate source-execution gap. Preserve all original ACK, resource, deadline, failure-evidence and lifecycle obligations when binding the r2 literal/plan; this source delivery is not that binding or runtime admission.

Both source reads (execution and final current-file verification) count toward the original total cap. The component contains no timer, process, service, lock or filesystem operations of its own. Its source reader is a supplied existing Entry port. There is no history framework change.

Twelve mocked tests cover successful first import, honest preexisting correlation, get-code-only and failed-execution receipts, bad provider/metadata/hash, source cap exhaustion, module replacement, invented receipts, pin mismatch and explicit reload refusal. They do not run the outer interpreter or establish actual r2 behavior. The r1 record remains unchanged.
