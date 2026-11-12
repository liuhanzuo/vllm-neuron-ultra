# Qwen complete-history successor v9

This is source and authenticated JSON preparation only. No native program,
model, service, physical drain, or serial lock was started. The accepted v8
runtime and its 128 tests, and preparation v2 and its 30 tests, were left intact
and were not rerun. There is no new model qualification or speedup result.

The old preparation v2 cannot consume the real LFM r2 actual-exit record: it
expects output/source/drain/lock fields that belong to separate saved audit and
scope records. Preparation v3 reads the immutable actual exit, rebuilds the
independent full audit/scope with `lfm3_full_cpu_successor_history_v3`, and never
rewrites or relabels an actual-exit record.

The current catalog deliberately stores only LFM's two main owners, because its
per-row limit is 256. Runtime v8's overlay consumes those catalog owners and
does not automatically add the separate 874 children. Runtime v9 appends every
exact child owner to the resolved original owner list. The old owner list,
complete known-service list, full-tree physical collector, all historical
success/failure statuses, and all original checks are retained.

The v8 generic physical reader already permits a collected `not-found` service;
that is not itself a defect. V9 additionally authenticates the three specific
retirement proofs needed for this completed history: LFM r2, Mini native r1,
and Mini native r3-sourceclosure. The existing frozen Mini retirement v2
validators retain each original invocation, the actual success or failure, and
independent recorder evidence. Future admitted v9 physical collection adds
fresh two-sample checks for precisely these three services and checks every
complete LFM/Mini owner. There is no generic unknown-invocation exception and no
change to other historical services.

The six additive runtime v9 files preserve four v8 bodies exactly apart from
versioned imports/CLI references: adapter, IO, service, and CLI. Contract v9 adds
the explicit `lfm_r2_scope` reference and the supplement source. History v9
rebuilds/extends that scope and records the extra retirement observations.
`qwen35_sampler_history_supplement_v1.py` contains this small bridge. Scientific
phases, native environment, CPU/Neuron/HTTP/cold/long/reuse/quality checks,
resource floors, output limits, original lock, and 45-second cleanup are
unchanged. The source diff records every runtime and preparation change.

The first candidate source map missed three Python dependencies imported by the
LFM scope validator. The unchanged `loaded_local_sources` rejected it at
`lfm3_nki05_device_matrix_v1.py` (tool chunk `6d6b52`, exit 1). The final source
manifest v2 adds exactly this file, `lfm3_nki05_full_cpu_metadata_v1.py`, and
`lfm3_full_cpu_successor_history_v3.py` with their existing hashes. It preserves
the six v8 sources and all 47,959 prior pins. Actual runtime import closure then
passed (session 94512, terminal `e7a5a3`, exit 0). The final v81 candidate has
47,996 pins; no original pin was replaced. The compact host map is 7,398 bytes.

Preparation API:

```python
import prepare_qwen35_sampler_v9_candidate_v3 as candidate
reader = candidate.JsonMetadata(metadata_guard)
result = candidate.prepare(explicit_future, reader.read, metadata_guard)
```

`explicit_future` contains the original ten fields (`terminal`,
`prerequisite_exit`, `catalog`, `run_name`, `output_root`,
`overlay_output_root`, `running_handoff_path`, `overlay_path`, `spec_path`,
`review_candidate_path`) plus `lfm_r2_scope` and `successor_sources`. Use the
current root-provided complete catalog and idle handoff; the code does not bind
to v79/v81 or discover the latest handoff. The final source manifest is
`qwen35-sampler-complete-history-v9-sources-20261112-v2.json`, SHA-256
`4f286d816c5247d9d644f714712b1caf778030584b0c3da360cfa3efc1a922b0`.

Actual preparation using v81/v538 passed, including all 126 serial rows, the
876-owner LFM scope, and three retirement proofs (session 14076, terminal
`1a926d`, exit 0). The 156 authenticated metadata inputs preserve all 150 old
spec inputs, 159 old overlay inputs, original 36/137 subsets, and 8,791 base
inputs. The prospective overlay is 844,133 bytes and the spec 8,781,938 bytes,
within the original separate envelopes. No runtime candidate artifacts were
written; only a source-validation report was saved. A candidate review remains
`passed=false`, `launch_authorized=false`, and runtime `validate_spec` rejects
it. The actual exit, prior failure statuses, and large preparation-only
envelopes were not recast as ordinary runtime inputs.

The attempted v80 preparation correctly rejected its extra `bytes` fields in
catalog references (tool chunk `fec2bc`, exit 1). Root preserved v80 and issued
v81 with strict references; the helper's reference rules were not relaxed.
Generic historical SDK readers can accept optional bytes, but this Qwen/LFM
preparation path requires exact path/hash references.

Ten new focused JSON/mock tests passed (`3cca1f`, exit 0): complete owner objects,
missing child/audit/recorder/owner rejection, altered actual exit, live final
child, unexpected invocation for all three retired units, and unstable service
observations. Two final preparation negatives rejected a missing dependency map
and a false unchanged-science declaration. All were bounded to CPU 7, nice 19,
512 MiB address space, CPU 60 seconds, system Python `-B -s`, and a native-import
blocker. These mock checks are not physical observations.

Before any future Qwen run, root still needs independent source review, a fresh
explicit candidate using the then-current serial catalog, complete clean-process
`Bundle.resolve` including recovered/recorder witness closure, actual physical
and lock checks, and ordinary admission. Do not load the preparation helper in
that runtime-resolution process or bypass the loaded-source check. The Qwen r6
API exception remains unknown; no new root-cause claim is made.
