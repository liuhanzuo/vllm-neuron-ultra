# MiniCPM5 source-phase bounded snapshots

Source-only successor: runner v9, runtime context v3, and source-phase seal v3.
The frozen v7/v8 runners and v1/v2 contexts/seals remain unchanged. The actual
native r1 session 88265 is still running its original v7.

## What changes

The old seal recursively copied the entire resolved history, including the
same native bundle reachable through several aliases. That full snapshot was
not used by the source loop except for its output catalog. Exit validation
also created and discarded another full snapshot before serializing the tree.

The successor snapshots only `context_plan`, `contract`, `partition`, and
`native_runtime_output_catalog`. These are the actual loop consumers. A
private memo preserves shared container identity within that snapshot; lists
become tuples and dictionaries become read-only mapping proxies. Cycles,
nonfinite numbers, custom containers, and non-string keys are rejected.

Complete original `resolved`, context plan, contract, partition, and permit
values still receive entry and exit SHA256 checks. Exact JSON type validation
visits shared containers once. `JSONEncoder.iterencode` then hashes every
serialized value using the original sorted-key, compact, ASCII-escaped JSON
representation. It avoids a whole serialized string and its whole byte copy;
it does not omit repeated values from the canonical digest. The largest
encoder chunk and sorted dictionary keys still require memory.

The original binding and partition checks remain. The per-source hashes,
lease/storage checks, complete live output walk, output allocation caps,
permanent failure latch, plain-plan restoration, and final native source
receipt validation are unchanged. No model permission is granted by the
source phase. Normal completion immediately releases temporary state without
depending on garbage collection. Transient changes restored before exit
remain outside the original boundary-check guarantee.

Context v3 and runner v9 differ from the accepted v2/v8 only in their import
wiring. They keep the original six bounded startup progress records.

## Tests and admission boundary

The root ran 12 source/memory controls in real tool session 49688, terminal
`c4e1b6`, exit 0, 8.958 seconds. They include the seven original integration
controls, digest parity for Unicode/escaped strings/floats and 100 randomized
objects, shared-container/cycle/type checks, every old mutable alias, and
preserved context/runner source wiring.

A small synthetic shared-history digest comparison measured Python-traced
temporary peaks of 2,039,368 bytes for the original serializer and 21,984 bytes
for the streaming serializer. This is a memory microtest, not a full-process
RSS measurement, actual startup benchmark, or inference speedup.

No real plan, historical tensor, model weight, device, serial lock, admission,
or successor model was accessed by these controls. Full-size memory and
startup time remain unmeasured. A separately reviewed preparation successor
must select v9, preserve all historical source pins, and bind a fresh complete
catalog before a real run. The original service/resource/scientific limits
are unchanged.
