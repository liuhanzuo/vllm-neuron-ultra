# MiniCPM5 restore-control raw transport and retirement contract

This delivery implements a finite transport, an explicit legacy replay entry,
an independent complete replay comparator, and a guarded retirement transaction.
It does **not** admit production deletion. No real payload has been read by this
work and no actual History, lock, queue, native model or cleanup has executed.

The selection is exactly 14 calls × 168 restore-after BF16 raw files. Each is
4,325,376 bytes. Their saved allocated total is 10,173,284,352 bytes
(9.474609375 GiB). Each maps only to the same call/index control-after file.
All98 serialized outputs, original weights, producer manifests, failures, gates,
and every other payload are retained. The complete immutable transport plan is
`minicpm5-retained-restore-control-transport-plan-20261113-v1.json`, SHA256
`996215f14e08e8b1a2a1e8f0e53e776a0dfe919102a60411a7e316706c5d61da`.
It contains both original full producer rows and both distinct physical
identities for every pair. Public production interfaces require that exact map.

## Callable interfaces

| Interface | Behavior and required inputs |
|---|---|
| `transport.build_production_plan(read_metadata)` | Metadata-only reconstruction from three fixed anchors and all28 authenticated manifests; must reproduce the fixed complete map digest. |
| `transport.verify_pairs(plan, receipt_path, guard)` | Under an external admitted lease, streams all2352 full original/keeper pairs and verifies equality, SHA, sizes and stable singleton descriptor/path identities. Publishes only a complete durable receipt. |
| `transport.Replay(plan, verification_ref, guard).read_state(original_row)` | Accepts the exact full original producer row; returns complete original bytes plus separate historical capture and current keeper identity. Unknown logical paths fail. |
| `Replay.export_state(original_row, destination, receipt_path)` | Restores bytes into a fresh independent regular file. Fsyncs file and parent, fully rereads actual exported bytes and identity, then writes a receipt with the honest new inode. Never restores or invents the original inode. |
| `legacy.Payloads(original_plan_bytes, storage_plan, verification_ref, guard)` | Keeps the unchanged original11859 logical input map; mapped states resolve explicitly, all other input identities remain strict. |
| `legacy.run_replay(payloads, destination, external_guard)` | New standalone CPU-worker entry for the entire old analysis. Uses a fresh output namespace and requires real root-reviewed worker/serial/source admission. No old module or `sys.modules` entry is patched. |
| `legacy.check_saved_terminal(original_protocol, original_queue, payloads)` | Retains the old full queue receipt check while explicitly replacing the old payload-inode verify entry. This is separate from actually rerunning the frozen terminal-auditor executable. |
| `compare.compare_original_replay(plan, verification_ref, replay_result_ref, output_path, guard)` | Post-worker-exit independent full old/new comparison. Requires14 scientific reports and98 layout records byte-identical, complete11859 logical admission, and all HF32/cache/hook/stage/failure results unchanged. |
| `transport.prepare_retirement_intent(...)` | Writes a durable complete selection and prerequisite references while all originals still exist. This is an intent, not permission to delete; reference semantics are validated by the transaction. |
| `transaction.retire(plan, verification_ref, intent_ref, journal, guard)` | Sole public deletion entry. Authenticates the actual executing source closure, original1869 source pins, complete replay equality and current History/lease evidence. Recomputes the full old/new comparison before any unlink. |
| `transport.audit_retired_state(...)` | Fully rereads every keeper and verifies every logical original path is absent. Records postconditions only; does not invent an observed unlink or process exit. |

Private synthetic helpers are for test fixtures under one `/tmp` directory.
They must not be used as production entrypoints. The tests never target `/mnt`
or experiment payload namespaces.

## Exact legacy integration

The five old consumer boundaries are concrete source successors:

1. `verify_protocol_metadata` mechanically retains every old recovery protocol
   metadata/source/provenance check, replacing only its11859 live-original
   identity loop with explicit logical-capture/current-storage verification.
2. `evidence_view` copies original function code objects into a separate global
   namespace. Only the capture IO dependency and mutable accumulators change.
   `STATS` remains the historical original capture provenance; `LIVE_STORAGE`
   records actual keeper identities separately. Earlier matched32 inputs outside
   the11859 reanalysis map retain the original live identity check without aliasing.
3. `Payloads.admit_all` hashes all11859 logical inputs in full. A mapped logical
   restore path consumes its complete keeper bytes. Every unchanged input still
   uses its original physical identity. The full original logical hash map is
   preserved; physical resolution is additional explicit provenance.
4. `Tensors.load_state` decodes the exact mapped bytes with the original BF16,
   shape, full-size, digest and nonfinite constraints. Unmapped reads use the
   unchanged old reader.
5. `replay_analysis` preserves the complete old scientific body. AST tests prove
   that only explicit tensor dependency injection and the final identity-loop
   call differ. The old queue terminal receipt remains usable through the new
   explicit terminal-check entry. The frozen old scripts remain unmodified and
   are not claimed to run directly after originals are removed.

The independent comparator authenticates the old complete result at
SHA256 `83d3329b4c0cce71cb30dfe95b6942a750fad65081cc4d6d93dd7f6aebeb1c11`.
It follows all14 reports and demands complete byte identity, including all9408
cache comparisons,5964 CPU hook pairs and13632 stage prediction vectors. It
also checks all7×32 HF rows,256 effects, all98 layout proofs and every original
failure by full comparison equality. The only normalized scientific references
are the exact14 old/new output paths and their authenticated report references.
New admission allows exactly two additional fields for honest capture/storage
provenance. A successful new run or a structural gate alone cannot qualify.

## Standalone replay-worker admission

The external guard is supplied by a reviewed root caller. It must verify the
current complete History, original serial lock, all predecessor owners/cgroups,
source closure and actual worker identity. It must remain usable as a lightweight
lease check after the worker installs the original no-subprocess/no-network
native audit hook. Do not call real History or service commands from that
lightweight callback after activation.

Its returned lease must include:

- schema `minicpm5_retained_raw_replay_worker_lease_v1`;
- original and storage plan digests, fresh output root;
- exact `worker_unit`, `worker_owner` (PID/start ticks), and actual argv;
- externally verified complete current History/source/lock evidence.

The new worker itself retains the old interpreter, working directory, debug,
no-bytecode, clean environment, fresh pycache/temp directory, CPU7/Nice19,
MemoryMax6GiB/MemoryHigh5GiB/no swap/TasksMax32 checks. It verifies the exact new
service owner and requires it to be the only heavy worker. It preserves the
original source verification,256MiB output budget,32MiB file cap and disk floors.
Only the temporary/output execution namespace changes; the scientific plan
bytes and all original runtime restrictions remain explicit.

A real outer launcher/independent exit observer must still be bound after the
currently running serial work actually terminates. This delivery chooses no
current catalog, handoff, invocation, worker or execution slot. No real replay
has run; the actual complete equality receipt is therefore still absent.

## Deletion prerequisites and durable transaction

The future production admission must bind the exact map, complete verification,
independent source/full-replay review, actual remover and legacy sources,
current complete History compatibility, and an exclusive namespace-write lease.
The review includes the entire unchanged original source map plus all actual
new/local executing sources. The implementation verifies the actual remover,
transport and comparator module paths, exact legacy source path, source hashes,
and every loaded local module. It rejects a different valid `.py` file. Executed
sources are rechecked during the transaction; the full original source identities
are rechecked before completion. It also recomputes the independent complete
old/new replay comparison from its authenticated result references.

For each pair, the transaction performs a fresh complete comparison, writes and
fsyncs an exact per-pair intent, rechecks the actual original descriptor/path,
unlinks only that basename through an opened parent descriptor, fsyncs the
parent, then writes and fsyncs a separate completion receipt. Every partial or
failed journal remains. Keepers remain untouched singleton files. Completion
requires all2352 pair receipts and a fresh full keeper-byte postcondition audit.
The reported retired allocated bytes are the original allocated blocks; no
claim is made that filesystem-wide free-space delta equals this value exactly.

Ordinary Linux unlink is not an atomic unlink-if-inode operation. The root lease
must exclude **all** writers/path replacement, including other task agents and
outside tools, throughout the transaction. Path/descriptor checks and dirfd
anchoring do not claim protection against a writer violating that lease in the
interval between final stat and unlink. Without that guarantee, deletion stays
blocked.

## Interrupted receipts: supported recovery and explicit limits

If execution stops after a complete durable intent but before unlink, retry
fully rechecks the still-original pair before deleting. If it stops after unlink
but before a completion file exists, retry requires that exact durable intent
and full keeper verification, then records `unlink_observed_in_this_attempt=false`
and `recovered_after_unrecorded_unlink=true`. An absent original without its
exact intent is rejected. A replacement original is never removed.

An interruption **inside** a JSON write can leave a partial intent/done file.
The implementation deliberately preserves it and refuses automatic retry; it
never overwrites a partial file or promotes it to an old successful receipt.
Synthetic tests exercise a mid-done write after unlink and show the next pair
remains untouched. Crash recovery is therefore conditional, not unconditional.

The required review protocol for a partial JSON file is:

1. Drain the failed owner, retain the complete old journal and actual exit/error
   evidence, and authenticate its exact partial bytes and current identities.
2. Independently classify every affected pair using the full original map,
   complete verification and whole intent. Check original presence/identity and
   reread keeper bytes. Missing original plus missing/incomplete pair intent is
   an unresolved external mutation and must remain blocked.
3. When the old pair intent is complete and original absent, a reviewed recovery
   successor may create a **fresh** journal and explicitly recovered completion
   receipts, retaining references to every old intent/done/partial file. If only
   the intent is partial and the original remains exact, it may recompare and
   create a new intent in that fresh journal. It must not claim an observed unlink
   from the recovery process. Already completed pairs need the same explicit
   old-to-new receipt lineage; no silent receipt copying/path substitution.
4. Re-run complete keeper verification and independent review before permitting
   continued deletion. The old partial files and failures remain immutable.

That exceptional partial-JSON migration successor is not implemented here.
Ordinary retry is intentionally blocked for this case. Its required scope is a
small explicit journal migration plus negative tests for incomplete intent,
replaced original, altered keeper and missing old receipt lineage; it is not a
reason to discard the evidence or restart an unreviewed remover.

## Validation and remaining actual work

Validation consists of source/AST preservation checks and synthetic file tests:
missing/replaced/wrong bytes, keeper/export integrity, wrong source closure,
post-admission source mutation, partial/full receipt states, interruption before
and after unlink, full-sized synthetic scientific comparison, and differences
in late cache/HF/layout results. These are not real-model qualification.

Before actual cleanup, root must review the final source map, bind a new current
serial admission after all current work is terminal, execute complete paired
verification and full original11859/full32 replay, independently observe exits,
run the full equality comparator, and authenticate current complete History and
exclusive writer/holder drainage. Only then can the production transaction run.
Cold-cache, long-decode, slot reuse, quality96, matched HTTP, original precision
and all model failures remain unchanged. No speedup is inferred from storage work.
