# SmolLM3: bounded experiment records — 2026-11-08

The prepared SmolLM3 observer test could not safely carry the growing experiment
history: three acknowledgements and one admission record were projected to use
about 76.5 MB, exceeding a 64 MiB off-mode run budget before response output.
The successor stores repeated source hashes and file identities in shared,
SHA-addressed chunks and expands them losslessly when reading. Original record
hashes continue to identify the physical bytes on disk.

Nine runtime modules are integrated. The source reader accepts only the named
large Nemotron metadata envelopes and the three original non-UTF8 Python
fixtures; other source restrictions remain unchanged. Model weights and raw
payloads are excluded from runtime source reads. A separate namespace for final
catalog chunks avoids changing a catalog merely by writing its own dependencies,
so recovery can compare the same semantic catalog.

Validation completed: 71 focused codec/store/integration checks, 433 historical
JSON roundtrips, and 36,588 source/metadata hashes. The complete current-source
check used about 145.6 MB RSS under a 512 MiB address-space limit. A further
24 finalizer and transport test methods passed, including original exit checks,
retention of all 682 completed service entries and 106 standalone process
identities, corrupted/missing chunk rejection, and preservation of interrupted
writes. Non-JSON extra bytes stay physical evidence and cannot become source pins.

The prepared root finalizer authenticates the compact catalog, retains its full
physical dependency closure, inventories all experiment output and separately
requires actual external exit and physical drain. Pending or extra files need a
concrete root disposition tied to their hashes; saving those bytes never proves
scientific success. The finalizer has **not** run on a new Smol experiment.

The scientific protocol is unchanged: three configurations, 12 off/on/on/off
rounds, 264 requests and 33,696 output tokens, with the original cold-cache,
long-decoding, request-slot reuse and quality requirements. The total 2 GiB,
control 128 MiB and per-run 64/256 MiB caps remain in force.

The complete-category forecast has now been recomputed against the actual
final Nemotron r3 history plus completed maintenance: 735 services and 142
recorded standalone process identities. It rehashed 37,080 source/metadata entries,
roundtripped all 433 historical JSON records, and kept the original indented
encoding for ordinary records. The 512 MiB address-space-limited forecast
process peaked at 199.5 MiB RSS.

| Forecast category | Allocated bytes | Original cap |
|---|---:|---:|
| All 12 rounds plus control | 1,697,579,008 | 2 GiB |
| Control, chunks, logs and terminal reserves | 99,549,184 | 128 MiB |
| Each off round | 65,011,712 (62 MiB) | 64 MiB |
| Each on round | 201,326,592 (192 MiB) | 256 MiB |

The forecast includes response files, both logs, observer/export output,
interrupted writes and failure allowances; it retains the original 8 MiB
control terminal reserve and adds 32 MiB outside-runtime finalization allowance.
These empirical allowances fit the original caps. The off rounds have only
2 MiB forecast headroom. Unexpected output still fails under the original
inclusive monitors; this is not a guarantee that every hypothetical maximum
response can fit.

The retained-data external-audit memory probe has now passed under the original
512 MiB address-space and 16 MiB file-size limits, peaking at 282,914,816 bytes
(269.8 MiB) RSS. It reparsed original raw HTTP/SSE and observer records, regenerated
exact export bytes, and retained twelve capture/proof slots plus six exports.
Those slots repeat the two genuinely fully audited historical reference fixtures
(r00/off and r01/on). They are **not twelve new model runs**, nor coverage of all
three configurations. The failed first probe, which attempted to use r02 without
its missing completed lifecycle audit, is retained. No r02 audit was fabricated.

Final admission exposed a source-reader mismatch: four already reviewed model
metadata symlinks were accepted by the complete source-closure check but rejected
by the handoff loader. The successor routes that loader through the same exact
alias-aware validation. Ordinary reads remain strict; changed targets, redirected
links, wrong hashes and unreviewed aliases are rejected. Nine successor runtime
modules and the finalizer now bind consistently. Thirteen alias/preservation,
24 finalizer/transport and nine budget checks passed; frozen predecessors remain
unchanged. An intermediate missing successor test-file preparation failure is
also retained.

The final forecast binds **735 completed services and 145 recorded standalone
process identities**, and reverified **37,106 source entries**.
Its total is **1,697,599,488 bytes** and control allocation is
**99,569,664 bytes**. Original caps, the complete scientific matrix and
fail-and-retain behavior are unchanged. Exact source acceptance, specification
and review were created and the real `load_spec` check passed.

The fresh sio12 experiment **failed before the first server became ready**;
zero inference requests completed. Source, input and compiled-payload admission
had passed. The actual server traceback rejects the installed wheel's extensionless
`REQUESTED` receipt; the original reader independently reproduces that error.
`INSTALLER` is the second unsupported receipt in the same 44-file installed list.
The external process exited 1 (session 78142), and all planned groups are drained.

A source-only successor now permits these two exact original receipt paths and
hashes, bounded to 64 bytes. All 44 real installed files verify; eleven focused
receipt controls passed. The new reader has **not** been integrated into a new
runtime closure or launched. The failed sio12 namespace will not be reused.

The terminal finalizer wrote its audit and complete catalog, then itself exited 1
while printing a reference: the new 9,070,236-byte catalog exceeds the ordinary
8 MiB reader cap. An independent follow-up authenticated both retained outputs,
verified all prior history and repeated physical drain checks. The completed
catalog contains 760 planned service entries and 146 recorded standalone process
identities; 23 of the 25 new planned units were never launched. Original finalizer
failure and successful follow-up verification remain separate records.

Reader v3 additionally accepts only this exact completed catalog path/hash and
schema under the existing 16 MiB writer cap. Sixteen tests passed, including the
44 installed files, real completed catalog, changed hash/schema rejection and
continued rejection of unlisted oversized JSON. This is preparation for a future
run, with no new inference result, model qualification or speedup.

The [admission and terminal evidence](evidence/smollm3-admission-20261108/manifest.json)
retains the startup snapshot, actual exits, failure log, post-finalizer verification
and successor reader controls. The sio12 native process is terminal. Its successor status is recorded below;
Nemotron v8 remains unlaunched. Original science and resource limits remain in force.

The [evidence manifest](evidence/smollm3-metadata-20261108/manifest.json) binds the
source checks, runtime patch, finalizer source, regression checkers, the older partial estimate and the new
complete-category forecast. These are experiment-harness sources that require the
retained workspace and fixtures; they are not standalone installation tests of
the unified model package. The unified model implementation remains `e0a01d3`.

## Successor sio13 integrated; now failed and finalized

The exact installation-receipt reader is now connected through all nine runtime
modules in a fresh sio13 namespace, with fresh service names and nonces. The
original source files and failed sio12 outputs are unchanged. The two receipt
files are pinned in the initial source closure as well as the wheel check.
The actual worker's complete pre-native source phase verifies all 44 installed
files before importing inference libraries.

Four integration checks, 26 finalizer/transport checks and nine budget checks
passed. The finalizer now returns a reference computed from the exact bytes it
just durably wrote, avoiding the previous 8 MiB console reread. A real >8 MiB
write is tested, along with unchanged no-overwrite and 16 MiB output limits.

The retained-data memory probe was rerun against the complete 760-service /
146-process predecessor history under the original 512 MiB address-space limit.
It peaked at 283,824,128 bytes (270.7 MiB) RSS. As before, its twelve slots repeat
two actual fully audited reference fixtures; this is not twelve model executions.
The final budget rehashed 37,315 source entries and projected 1,697,529,856 bytes
total, including 99,500,032 bytes of control data, within the unchanged caps.

The real external process (session 76500) exited **1**. Initial source,
input and compiled-payload admission passed, and both parent and engine child
passed the previous installation-receipt rejection. The controller then failed
at the process-title handshake with `before ACK deadline`, before readiness or
any inference request. There were zero completed requests and output tokens.

The retained intent fixes a 15-second title deadline. Its file modification time
is approximately 19.4 seconds after creation, using a wall/monotonic conversion
sampled after exit. That estimate is supporting evidence, not a direct trace or
proof that the wall clock never changed. Source inspection establishes that
`begin()` starts the clock before `context.record()`, whose writer performs a
complete guard poll before publishing and another afterward. There is no
before-ACK, after-call or commit receipt. This is a validation-harness failure;
it does not establish a model numerical defect or completed title change.

The independent finalizer exited 0, retained all output, and confirmed physical
drain. The finalized history contains **785 planned services and 147 standalone
process identities**; 23 of the 25 added services were never launched. The
original failed outcome remains failed. The old namespace and sources are frozen.

The three configurations, twelve rounds, 264 requests and 33,696 tokens remain
the required future matrix, including cold cache, long decode, slot reuse and
quality. No new completed matrix, model qualification or speedup is claimed.
Nemotron r4 subsequently received serial admission using this finalized history.
The [terminal and successor evidence](evidence/smol-terminal-nemotron-r4-20261108/manifest.json)
includes original failure receipts, the timing diagnostic and actual launch.

## Guard cost repair prepared; complete transaction still pending

The [new source report](smollm3-guard-cost-20261108.md) records 71 passing
checks and a retained-metadata component reduction from about 18.6–18.8 seconds
to 5.1–5.2 seconds. These are validation-harness timings, not model speedups.
The original 15-second title deadline and complete scientific matrix remain.
No fresh Smol runtime is admitted by these component results.
