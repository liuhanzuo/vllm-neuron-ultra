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

The fresh sio12 external process has started (tool session 78142); its actual
identity is retained in the new [admission evidence](evidence/smollm3-admission-20261108/manifest.json).
Input/source and compiled-payload admission passed; the authenticated controller
has entered the first round, r00. The full twelve rounds and terminal review have
not completed. This is no new
model qualification or speedup claim. Follow the retained execution handle; do
not start a duplicate experiment.

The [evidence manifest](evidence/smollm3-metadata-20261108/manifest.json) binds the
source checks, runtime patch, finalizer source, regression checkers, the older partial estimate and the new
complete-category forecast. These are experiment-harness sources that require the
retained workspace and fixtures; they are not standalone installation tests of
the unified model package. The unified model implementation remains `e0a01d3`.
