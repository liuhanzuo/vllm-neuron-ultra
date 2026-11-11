# Native source phase successor: runner v8 and context v2

This delivery integrates the accepted source-phase prototype into new source files only. No actual successor admission, plan, launch, lock acquisition, model execution, or live patch is performed. Current r1/session 88265, runner v7, native context v1 and both frozen seal v1 review artifacts remain unchanged.

## Exact interface and retained behavior

`minicpm5_layer1_operator_native_runner_v8.py` retains the original CLI, bootstrap, coordinator, process topology, admission, descriptor lease, complete history/physical checks, output guards and driver v4. It imports native context v2 and supplies four internal keyword arguments to its constructor: `contract`, `resolved`, `lease`, `ports`. These are the existing admitted objects. There is no new command-line or business interface. A future separately reviewed additive source extension must pin the new runner, context v2 and seal v2 as well as every inherited source; the tests verify that the original `build_contract` accepts the actual new runner path and source hash.

`minicpm5_layer1_operator_native_runtime_context_v2.py` inherits native context v1. The original constructor still authenticates the complete plan and makes its own deep copy. The original classifier/read prefix is retained verbatim. The standalone prototype `phase_partition` helper is **not** used here. Only `sources.verify_text_sources(partition, self.guard)` is replaced by the sealed phase around the same complete per-file authentication.

After successful phase exit, `context.plan` is the original plain context copy again. The entire original admission read, source-authentication receipt read, binding checks, `validate_native_receipt`, nine native identity checks, stored runtime receipt identities and final guard are retained. The integration test compares their AST against v1. `_sources_checked` becomes true only after those checks return a private completion token normally. An early return, skipped/incomplete phase or repeated call cannot grant model permission.

The normal source phase retains every source hash, the original source partition, three Python-byte fixtures, lease, live storage floors, complete descriptor-relative output walk and allocation checks. It moves only static full-plan serialization and catalog reconstruction out of the per-text loop. All bootstrap source passes, historical metadata replay and physical checks still execute.

## Independent failure change

Context v2 also latches **any** classifier, source, phase-exit, admission, receipt, native identity or completion failure across the whole `verify_sources` operation. Both context guard and external guard then raise the same original exception permanently; `_sources_checked` is false. Catching the exception cannot enable retry, `verify_input`, or model work. This is a separately stated behavior change beyond the cost reduction. The original guard and input APIs remain inherited for normal operation.

## Snapshot lifetime and mutation contract

`research/minicpm5_native_source_phase_seal_v2.py` subclasses frozen seal v1. It changes only successful-exit cleanup: after all exit checks restore the original context API, it clears temporary full-tree snapshots and bound-method/closure references. The integration test disables automatic GC and verifies that the phase weak reference is already dead when `verify_sources` returns. No explicit collection is introduced. Failure retains the permanent latch; normal API is not restored on failure.

During authentication the consumed plan, partition and static catalog use private mapping proxies and tuples. Old mutable aliases—including the independently deep-copied context plan—are fully rechecked at exit before any model/weight work. Their mutations are not checked every guard, and an old alias changed then completely restored before exit is not traceable. Such a transient change does not alter the sealed values consumed by this phase. Context/callback/writer rebinding and observed failures still reject immediately. Unknown objects, sets, tuples supplied as plain input, Paths, custom containers, cycles and non-finite values remain rejected rather than normalized or discarded.

The real overlay `resolved` includes `current_scope`, `native_bundle`, `current_serial_view` and historical branches. Source inspection traced the return construction through overlay v2, recovered completion v1, operator runtime completion v3, SDK completion v2, batch-layout completion v3, native-cache completion, saved-audit completion, recovered r12 and its current-history/owner-variant builders. Exposed unit sets are sorted to lists; temporary tuple lookup keys and observed `(reference, value)` pairs are consumed before returning; owner variants return dictionaries/lists; paths retained in references are strings. Metadata leaves and unpacked physical pairs come from JSON readers. No known returned non-JSON member was found in that inspected chain. This is a source-shape assessment, not validation of the actual full object.

The richer mock contains nested `current_scope`, an `_original_done` branch, original source maps and plans, `native_bundle` and `current_serial_view`. It verifies successful strict sealing and rejects a deep unknown type or historical mutation. The actual tree was not read. **Real snapshot peak memory remains unknown**: seal v1 recursively copies the entire resolved tree, including repeated historical branches, and performs full boundary serializations. Cleanup prevents overlap from a lingering successful phase, but does not establish that actual peak memory fits the admitted envelope. No history is deleted or type rule relaxed to hide this limitation.

## Six bounded progress lines

Only native mode emits these lines to its existing stdout/runtime log:

- `bootstrap.begin` and `bootstrap.end`: surround complete `resolve_admitted`, after actual gate and running-handoff authentication.
- `source.begin` and `source.end`: surround context creation, source verification and loaded-source verification.
- `driver.begin` and `driver.end`: surround the unchanged driver plus original post-driver loaded-source/final guard checks.

Each JSON line has only `phase`, monotonic `elapsed_seconds` since runner import, acknowledged `pid`, and `start_ticks`; it is capped at 512 bytes before LF. It does not read an unauthenticated admission or sample a new identity for logging. There are no coordinator progress lines. These observations cannot distinguish loader, saved inputs or first arm within driver and do not prove completion or actual external exit. No driver was copied for finer logging.

## Validation and evidence limits

Command from the repository root:

```bash
python3 -B -s research/check_minicpm5_native_source_phase_integration_v1.py
```

Seven integration methods passed independently twice:

- Author: session 79823, terminal `ffc07f`, exit 0; 6.085 seconds.
- Root: session 16148, terminal `d1fc98`, exit 0; 6.008 seconds, reported by root.

Tests execute the unchanged original context class and the new context class from source, avoiding historical import-time JSON reads. The original receipt and native identity validation functions run against synthetic metadata and mocked observations. The source package's classifier implementation is not reexecuted against real data; its unchanged call path and source-pinned reader are exercised with a synthetic classifier. Tests cover original context deep-copy compatibility, post-phase plain object types, full receipt checks before permission, all failure stages, premature returns, repeated verification, strict rich-history values, source/alias mutation, release without GC, and AST equivalence of all runner functions after removing the six log calls and constructor keywords.

The earlier accepted seal mock measured constant catalog/plan work with unchanged hashes and additional boundary dynamic scans. This integration introduces no new actual performance result. No native framework, original large plan, raw payload, network, lock, service, or model was used. The real workload has not been run with this successor. Root owns the subsequent source review and publication; this delivery itself grants no runtime authorization.
