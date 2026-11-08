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

Root free space is now about 30.5 GiB after verified, recoverable archival of
inactive software. **Runtime admission remains pending:** the forecast does
not prove the complete external-audit path fits its 512 MiB address-space limit,
and final source/spec/review bindings and live disk checks remain required.
No `sio12` directory or execution specification has been created. The full
scientific request matrix is unchanged and there is no new speedup claim.

The [evidence manifest](evidence/smollm3-metadata-20261108/manifest.json) binds the
source checks, runtime patch, finalizer source, regression checkers, the older partial estimate and the new
complete-category forecast. These are experiment-harness sources that require the
retained workspace and fixtures; they are not standalone installation tests of
the unified model package. The unified model implementation remains `e0a01d3`.
