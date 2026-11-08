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

Source preparation is complete through the finalizer component, but runtime
admission remains pending. The known-history control projection is at most
43,225,088 bytes after adding a separate catalog-chunk allocation; it excludes
parts of the full 12-round output and later Nemotron history. It is **not** a
complete budget admission. The next run must bind the actual final Nemotron
catalog, recheck all output/storage/memory budgets and satisfy the existing
30 GiB root-free admission floor. No `sio12` runtime directory or execution spec
has been created, and no new speedup is claimed.

The [evidence manifest](evidence/smollm3-metadata-20261108/manifest.json) binds the
source checks, runtime patch, finalizer source, regression checkers and explicitly
partial budget estimate. These are experiment-harness sources that require the
retained workspace and fixtures; they are not standalone installation tests of
the unified model package. The unified model implementation remains `e0a01d3`.
