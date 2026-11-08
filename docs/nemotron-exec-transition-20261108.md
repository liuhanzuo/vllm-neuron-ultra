# Nemotron package startup: retained failure and bounded fix — 2026-11-08

The third package run failed in our startup monitor after 42 successful stages.
The 16 `tensor_split` cases passed all 96 complete output byte comparisons;
all nine alias stages passed 70 device checks. These are partial correctness
results, not a completed model or performance qualification.

The compile worker raised `Unexpected process during exec transition` and
exited 1. Its server wrote an exact exec intent, but no accepted post-exec,
readiness or request receipt. The server log is empty. Cleanup initially refused
to signal the unexpected identity; the controller subsequently drained the
owned service group. The original outer exit is 1. Supported explicit recovery
released the failed run without changing its result. The history contains all
735 planned services and 107 standalone identities at that terminal point.

## What the diagnostic establishes

The old monitor required each `/proc` command-line read to equal either the
wrapper argv or the exact serving argv. A separate, serial, standard-library-only
probe made 32 real `os.execve` transitions with the original Python interpreter.
It observed **115 empty command lines**, no unexpected nonempty argv, stable
PID/start-time/parent/group, and exact target argv afterward. All 32 children
exited 0. No model or compiler was imported.

This reproduces a mechanism that the old monitor rejects. The original rejected
process snapshot was not retained, so it does not prove the historical r3 cause.
The failure receipt was written about 10 ms after the exec-intent receipt; those
file timestamps are supporting context, not an atomic process trace. Installed
vLLM contains process-title support, but the inspected API entry did not call
it and the retained log provides no title-change evidence.

## Prepared successor

Sixteen coherent v8 modules preserve the model package, 52 stages, scientific
checks and resource limits. Only the exec observation behavior changes: an empty
argv may be resampled for at most 250 ms, only while the same child is alive and
its exact exec intent is authenticated. An exact target argv, interpreter,
private session, parent and service group are still required. Independent
controller observation, health, requests and deliberate successful shutdown
remain required. Nonempty unexpected argv, identity changes, disappearance,
exit, persistent emptiness and reversion to the wrapper are rejected. The new
monitor retains the rejected observation.

Validation: **15 transition tests plus 34 HTTP lifecycle, filesystem-race and
source-preservation tests passed**. The HTTP fixture exercises the actual new
worker through an empty-argv transition and the existing terminal validator.
This is source preparation; no v8 runtime has been admitted or started.

## Storage and remaining work

Two inactive editor distributions were fully archived to NVMe and verified
member by member before retirement, recovering 1,382,281,216 allocated bytes
(about 1.29 GiB) on the root volume. All 7,444 regular-file hashes are preserved
in recoverable archives; the active editor, user settings and experiment data
were retained. The first archive helper retired its source successfully but
failed while writing its final report because an old metadata reader rejected
the 8.9 MB catalog. Its actual exit 1 and uncaptured process-identity limitation
remain recorded. A separate recovery reverified every archive member and the
source's absence; the second helper completed with exit 0.

Root free space returned to about 30.5 GiB. A later verified archive of sixteen
completed global Neuron compilation-cache entries reclaimed another
2,148,040,704 allocated bytes (about 2.00 GiB) on NVMe. All 64 original files were
rehash-verified in the retained archive before retirement. Model weights,
artifacts and experiment outputs were untouched. NVMe free space was then about
48.196 GiB, above the unchanged 48 GiB floor-plus-reservation requirement, with
little margin; each new run still requires a fresh space check.

The archive receipt is in the [admission evidence](evidence/smollm3-admission-20261108/manifest.json).
Nemotron v8 remains source-prepared and unlaunched. SmolLM3 sio12 ran first but
failed before readiness and completed no inference requests; its groups are now
drained. Native experiments remain serial. No storage threshold or model
validation requirement was lowered; no new model speedup is claimed.

## Completed Smol history compatibility prepared

While SmolLM3 sio13 was running, a source-only successor extended the prepared
Nemotron runner to consume the complete later history. The v8 reader actually
rejects the retained 9,070,236-byte Smol catalog at its ordinary 8 MiB bound and
rejects the wheel's extensionless `REQUESTED` receipt. These source-level errors
were reproduced without starting a model.

The coherent v9 successor retains the exact `60ab7f5` model package, all 52
stages, resource/storage limits and the bounded authenticated exec-transition
fix. Its metadata reader adds only these routes:

- Canonical completed Smol catalog filenames under the experiment's research
  directory require a reviewed SHA256 digest before reading. They retain the
  original 16 MiB Smol writer cap and must bind the filename to the catalog
  schema, complete service count, unique units, finalized status, source map and
  retained terminal-audit hash. Later physical drain and source checks still run.
- The two original wheel receipts have exact path/hash pins and a 64-byte bound.
  Other extensionless files and unlisted large JSON keep the original behavior.

All 16 metadata controls and 34 HTTP/lifecycle/inventory/source-preservation
checks passed. They include the actual retained 760-service/146-process catalog,
wrong/missing hashes, changed bytes, schema/count errors, oversized files,
symlinks and the original exec-transition handling. A synthetic future catalog
checks transport only and is never presented as a completed experiment.

The [source evidence](evidence/nemotron-history-20261108/manifest.json) preserves
the v8 parents and the complete v9 patch. At that v9 preparation snapshot, no execution specification or
runtime admission existed. Actual Smol termination, independent finalization,
all latest predecessor release proofs, full source binding and a fresh 48 GiB
NVMe admission check were still required before launch. There was no
new model qualification or speedup claim.

## Latest serial run: r4 admitted after Smol finalization

SmolLM3 sio13 has now failed at its title-handshake deadline and was independently
finalized. The completed history contains 785 planned services and 147 standalone
identities. A v10 runner explicitly follows both completed Smol producers (sio13
and sio12), authenticating each original audit, execution, live owner, actual
external exit, release, controller result and supervisor exit. It checks original
row/process extensions and source hashes, preserves failure outcomes, and does
not reinterpret inherited Nemotron release fields as the latest Smol exit.
Fresh physical drain remains mandatory at launch.

Twenty actual-history and rejection controls plus 34 lifecycle, inventory and
source-preservation checks passed. The v9 parent sources, original model package,
all 52 stages and all resource/storage limits remain unchanged. An initial test
selected the wrong predecessor glob; its error is retained and the corrected
whole suite passed. The source-preservation checks compare distinct old and new
modules. An unlaunched specification also failed on underscores in service names;
a successor normalizes only those service identifiers and retains the failure.

The real v10 bundle check revalidated **37,606 source entries**. Fresh root and
NVMe free space were 32,579,350,528 and 51,750,019,072 bytes respectively, meeting
the unchanged reservations. The r4 external launcher is running under session
50501; its controller has authenticated ownership and written actual runtime
admission. This is an admitted experiment, not a completed model result.
Compilation, numerical/state checks, cold cache, long decode, quality and paired
workloads still have to finish. No new speedup is claimed.

The [r4 evidence](evidence/smol-terminal-nemotron-r4-20261108/manifest.json) preserves
the v10 source changes, regressions, failed Smol receipts and actual admission.
Large complete source/specification records remain in the workspace with their
exact hashes listed in that manifest. Native/model/compiler tasks remain serial.
