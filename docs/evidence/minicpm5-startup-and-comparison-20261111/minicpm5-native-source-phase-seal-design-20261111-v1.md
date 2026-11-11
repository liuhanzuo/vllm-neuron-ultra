# Native source-authentication phase: independent prototype

This is a source-only optimization prototype, not a runner, admission, live patch, or release recommendation. Current session 88265 and all frozen sources remain untouched. No original plan, source catalog data, payload, keyring, live lock, model or process was read or executed for this work.

## Source diagnosis

The dominant repeated static computation is visible directly in the frozen code:

1. `minicpm5_layer1_operator_native_runner_v7.py:71` runs `bootstrap(native)`. Bootstrap line 59 calls `runtime.resolve_admitted`.
2. `research/minicpm5_layer1_operator_native_process_adapter_v6.py:151` reconstructs the complete recovered history and overlay; lines 162–163 classify and verify all text sources. It also preserves the saved pair, fresh complete two-pass physical drain, actual execution handoff and loaded-source verification.
3. Runner lines 83–85 construct the original-derived context and call `RuntimeContext.verify_sources`. `minicpm5_layer1_operator_native_runtime_context_v1.py:32` classifies the full map again, and line 33 rehashes every original text source.
4. Each source calls `context.guard`, including the original process/environment checks. Its external guard at runner lines 79–81 calls `Lease.check`, disk floors, complete descriptor-relative output scan, and `life.usage`.
5. `life.usage` at lifecycle core lines 97–99 calls `binding`; `binding` rebuilds the contract at lines 57–59; `build_contract` hashes the **entire plan** at line 38 and reconstructs the output catalog. The live walker and inventory validator each independently reconstruct that catalog too. Thus each source causes one full-plan JSON serialization/hash and three complete catalog constructions, including 936 raw paths, 144 metadata paths and 97 directories.

For N text sources, plan encoded size P and static output catalog size C, that inner path costs O(N × (P + C)), in addition to the required source-byte hashing and dynamic checks. P itself grows with the source/input maps. It is not merely O(total source bytes).

Other preserved costs are real: `Lease.check` performs descriptor and current-process checks each time; once per second it checks all observed metadata identities, discovers the latest handoff using all matching names, rechecks all old owners and active-heavy state. `sources.classify` and recovered completion recursively parse historical metadata. Named envelopes can be parsed for shape and again by `load`, and `text(expected)` plus its caller often hashes the same bytes twice. Those costs are not changed in this prototype.

The parent reported no torch/native framework mappings for PID 1494935/start 418180432 at 13:22 UTC. That supports “not yet loaded torch” but does not locate a Python frame. No stack was sampled. Absence of `native-operator` alone is weaker: the directory is created only at driver `run_loaded` line 340, after `loader.load` and `inputs.load`. Neither rchar nor syscall totals identify the responsible function or physical disk traffic. No actual speedup is established.

## Narrow integration boundary

The prototype has no automatic integration. A separately reviewed successor could change only the native context's text-authentication interval:

1. Complete original bootstrap/history/physical checks and construct the original context first. The context constructor deep-copies the plan; both that copy and `resolved['plan']` are separately checked against the same complete digest.
2. `phase_partition(context, sources)` invokes the original classifier with a detached plain plan and original guarded metadata transport. All source classes and pins remain unchanged. The original complete `check_partition` validates its returned plain partition before sealing.
3. `SourcePhase` performs the original full `life.binding`, then recursively seals the plan, contract, partition and output catalog using private `MappingProxyType` dictionaries and tuples. It accepts only exact JSON dictionaries/lists/scalars, rejects custom subclasses, unknown values, non-finite numbers and cycles, and does not expose backing dictionaries. It never uses a writable dict subclass as a freeze substitute.
4. During this interval `context.plan` points to the sealed copy, so inherited `OriginalContext.guard` reads the sealed value. All per-source hashes, original alias validation, three Python-byte fixtures and the original guard cadence remain. The original live lease callback, storage floors, descriptor-relative output walk and full allocation validator run at every guard. Only static catalog construction and whole-plan serialization leave the inner loop.
5. The dynamic walker and inventory function bodies are copied unchanged except that their already-validated static catalog is an argument; an AST control verifies those bodies match the frozen originals exactly. No live output entry, allocation, directory, symlink, pending-writer or cap check is removed.
6. Normal exit rechecks the entire original resolved object, original context-plan copy, original contract, original partition and permit against entry digests, then reruns the original binding and partition validator. Only after success is the original plain context plan/API restored. The existing native authentication receipt and all nine native stat identity checks still run before `_sources_checked` may become true. The prototype itself always leaves `_sources_checked` false and never grants model permission.
7. Any failure permanently binds the original exception to both `context.guard` and `context.external_guard`. Failed exit does not restore the plain API. Catching an error inside the phase cannot allow another guard, source pass, successful exit, `verify_input`, or new phase to proceed.

The imported source package has historical import-time JSON reads. The prototype therefore accepts the already-loaded original source module explicitly. Tests execute only unchanged pure function definitions from source against synthetic inputs. The first development test's audit hook rejected an attempted historical JSON open before reading it; the final prototype does not import that package.

## Mutation guarantee and deliberate limit

During the interval, mutation of the consumed nested snapshot is denied by genuine read-only containers. Context plan/callback/writer rebinding is rejected by the live guard. Mutable aliases retained outside the phase are checked **at exit, before any model/weight operation**, not on every guard. A transient mutation of an old alias that is completely restored before exit is not detectable and does not change the sealed data used by this phase. This is not a continuous mutation proof. Detecting arbitrary old-alias changes on every call would require retaining a full traversal/hash or an invasive mutation-tracking design; neither is claimed here. Deliberate reassignment of the implementation's own private controls is outside the trusted Python callback contract.

The source-only interval must contain no model callback or weights operation. The original `verify_input` gate requires `_sources_checked`, and the phase holds it false. Subsequent receipt validation and model admission remain separate original steps. This prototype does not remove the two complete source passes, cache source bytes or historical metadata, alter a physical drain, change a lease cadence, or introduce a generic cache.

## Verification and small comparison

Command from the repository root:

```bash
python3 -B -s research/check_minicpm5_native_source_phase_seal_v1.py
```

Nine methods passed in session 70743, terminal `55ad97`, exit 0. Tests use memory-only source/metadata/output fixtures and the actual original `RuntimeContext` constructor/guard with only process/filesystem observations mocked. They cover recursive immutability, unsupported values, source/partition completeness, context's separate copy, every old alias category, nested source/input/catalog changes, callback rebinding, source hash failures, output and disk failures, and permanently latched failure even when caught internally. An audit hook rejects actual network, subprocess, lock and non-source opens.

Both comparisons retain the full 936/144/97 static output contract but use a small synthetic plan. Timings include phase entry/exit and exclude fixture/context construction; they are illustrative, not a model benchmark.

| Synthetic sources | Encoded plan | Original / sealed catalog builds | Original / sealed full-plan serializations* | Original / sealed live scans | Text + Python-byte hashes | Original / sealed seconds |
|---|---:|---:|---:|---:|---:|---:|
| 41 total | 220833 B | 42 / 5 | 14 / 8 | 14 / 16 | 9 + 3, unchanged | 0.598424 / 0.128108 |
| 49 total | 222595 B | 66 / 5 | 22 / 8 | 22 / 24 | 17 + 3, unchanged | 0.936755 / 0.132050 |

*Sealed count includes three original lifecycle plan digests, three additional boundary plan digests and two resolved-object digests that contain the full plan. Additional snapshot traversal and equality are also O(P) at fixed boundaries; they are not hidden as zero-cost work. Lease and storage checks equal live scans in both versions. The extra two sealed checks are entry/exit checks. Historical and native payload source categories remain pinned but are not read by this text phase, matching the original policy.

## Minimal future phase observations

A later reviewed runner may emit one small bounded JSON line to the existing runtime log at each of these boundaries: bootstrap resolve begin/end; source authentication begin/end; loader begin/end; saved inputs begin/end; first arm begin. Include monotonic elapsed time, exact pid/start and the phase name only. These are progress observations, never completion or external-exit evidence. No per-source log, background monitor, new directory or service is needed. No such logging is added to the live r1 or integrated by this prototype.
