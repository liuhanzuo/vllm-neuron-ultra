# LFM writeback characterization: report bound and serial runner successor

This is source-only delivery. No spec, review, admission, handoff, worker,
service, lock action or native/simulator execution was created or performed.
The prior diagnostic v2 and probe v1 remain unchanged.

## Complete report fits the actual writer

The frozen serial runner's `once()` calls its `encode()`, which uses
`json.dumps(sort_keys=True, separators=(',', ':'), allow_nan=False)` plus a
newline. It does not use `indent=2`. The successor retains both functions
byte-for-byte at the AST level.

A complete synthetic three-arm specimen includes:

- Original identity failure with all three host-operand bit summaries and four
  extraction calls; restored-original arm retains the same complete evidence.
- All 32 candidate movement cases with their original observer-stage summaries,
  hashes, input/output types/shapes and all three host-operand summaries.
- All 132 candidate extraction calls and all 40 BF16 writeback substitutions,
  each with both explicit-false and explicit-true precision/copy controls.
- Full bit-position counters, all-word counts and all eight illustrative changed
  words wherever a changed operand is represented; no event or counter dropped.
- Original source maps, fixture/version evidence, exhaustive-encoding evidence,
  negative/arithmetic/cast controls and qualification flags.

For size conservatism, every operand summary uses the probe's full 256 KiB
operand bound, and changed summaries use maximum word/bit counts and eight
large-index examples. The real complete candidate has zero changed host words,
as represented. The two original error strings are padded to 4,096 characters,
larger than the observed identity assertion. This specimen is explicitly
synthetic; its inflated counts are a serialization envelope, not measurements.

| Serialization | Bytes |
| --- | ---: |
| Exact frozen `encode()` for the specimen | 225,977 |
| Diagnostic v2's existing default `json.dumps` cap check | 246,038 |
| `indent=2` comparison, not used by the runtime writer | 489,829 |
| Original single-file cap | 262,144 |

The original `once()` source definition was then actually used to write the
specimen into a new research fixture file, with two additional synthetic-only
markers. The saved file is **226,045 bytes**, leaving **36,099 bytes** below the
cap. The saved bytes equal the exact `encode()` output, and JSON parsing returns
the full original specimen without loss. This exercised the existing exclusive
temporary-write/fsync/link publication logic only on a new synthetic research
artifact, not on a runtime output or evidence namespace.

Artifact:
`research/lfm3-nki05-three-arm-serialization-envelope-20261111-v2.json`

SHA256: `cdeb3f118524759cf7a1a9ff82a17c20b9182d281864166740724227255e40ce`

This proves the complete expected characterization representation fits the real
writer. Unexpected runtime failure details still remain subject to the existing
hard cap and fail closed; no truncation or alternative unbounded file is added.
Pretty-printing this representation would exceed the cap and must not replace
the unchanged runtime writer.

## Runner successor and failure semantics

New source: `run_lfm3_nki05_transport_serial_v2.py`

SHA256: `7029adf0cb320c9857db0de5ca92b1d3dea52ec4003324b45b7d120def3910c7`

The changes are limited to importing diagnostic v2, an explicit reviewed-map
source-preparation interface, a characterization validator, and the worker's
diagnostic publication boundary. Every other original function/class has an
identical AST, including outer/history resolution, physical drain, lease/lock,
metadata checks, service ownership, resources, actual-exit observation and all
cleanup/failure paths. Resource constants and output caps are unchanged.

After an admitted worker obtains the complete diagnostic, it first saves all of
`diagnostic-result.json`. `require_characterization` then requires the exact
original failed identity fixture and its restored reproduction; candidate
success passes through the unchanged original 32-case success validator. It
also checks all three restoration records, 4/132/4 extraction counts, 0/40/0
substitution counts and the precise BF16 writeback evidence.

Only after those checks and another original live admission check does it write
`characterization-status.json` with:

- `characterization_complete=true`
- `original_passed=false`
- `candidate_passed=true`
- `nonzero_worker_exit_required=true`
- `actual_tool_exit_verified=false`
- The exact owner, admission and diagnostic reference, with no self-exit or
  model/device qualification claim.

The worker then deliberately raises `RuntimeError` rather than publishing a
successful worker result. The unchanged failure reporter writes
`worker-failure.json` with the distinct characterization-complete flag and status
reference. The unchanged outer observes the actual failed service and preserves
the nonzero outcome, writes its original failure/drain evidence, and releases
ownership only through the original cleanup path. Python's unhandled failure
is intended to produce exit 1; only root's future real observation may record
the actual exit.

If characterization is incomplete, a candidate gate fails, or the live check
fails, the full diagnostic remains saved when available. No complete status is
published, and the worker failure states `characterization_complete=false`.
Thus a completed causal experiment and an interrupted/incomplete experiment can
be distinguished without turning the original implementation's failure into a
successful scientific result.

The added small status file remains within the unchanged finite 64-entry / 4 MiB
output-tree checks; there is no new directory, raw payload or expanded cap.

## Complete preserved source map

The source closure contains **561 entries**, retaining every original 558 hash
verbatim and adding exactly three sources: runner v2, diagnostic v2, and probe
v1. All original finite dynamic targets and eight diagnostic SDK pins remain.
The source-only bounded AST collector parsed 553 local files / 6,682,612 bytes,
within its original limits. No source hash was rebound.

Artifact:
`research/lfm3-nki05-characterization-source-closure-20261111-v1.json`

SHA256: `59feb3573c3811738813247d37272152de6e11ede52e0ecc7b8d8099bc157250`

Root should use its complete `source_sha256` field for future binding. The
successor's `execution_source_paths(reviewed_source_sha256)` now requires that
reviewed full map explicitly, authenticates it through the existing verifier,
and returns every supplied path. It no longer infers closure from initially
imported modules. Preservation of the exact old 558 pins is established by the
hash-authenticated closure report, rather than inferred from a count alone.

## Pure verification

`/usr/bin/python3 -B research/check_lfm3_nki05_characterization_delivery_v1.py`
passed **8 tests**, actual exit 0, terminal `300a9e`, 0.157 seconds. Tests check
complete envelope coverage and exact writer encoding; unchanged original
lifecycle ASTs; candidate/full encoding gate enforcement; rejection of missing,
promoted or unrestored evidence; full publication followed by a real Python
exception; incomplete status behavior; and mandatory full-map preparation.

Test source SHA256:
`5fda747401bdcce74513448295f4b4f86fcf6a815f8ca32dfa4688838574f634`.

The actual synthetic `once()` write and lossless roundtrip completed with exit
0, terminal `896741`. The full 561-source closure collection and hash checks
completed with exit 0, terminal `52c79c`. No native execution was used by these
checks. Root's independent review and later serial binding remain separate.
