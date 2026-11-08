# Nemotron input and environment capture — 2026-11-08

The first actual preparation capture for the original Nemotron package
`60ab7f54a2c54ef0bc9b498621a45719ee7f0e84` completed with exit code **0**.
This is input and environment verification for that package. It does not add
model inference, serving or speedup qualification to the unified branch.

| Captured evidence | Result |
|---|---|
| Original model and reference inputs | All 16 paths: 10 official files, including all four weight shards, and six reference outputs |
| Installed Python sources | 30,410 files; three non-UTF-8 fixtures verified by original bytes |
| Installed native libraries and interpreter | 1,337 SHA-256 hashes and filesystem identities |
| Complete source/history fingerprints | 36,281 retained entries, including the 14 previously recorded Phi-4 raw fingerprints |
| Prior task cleanup | All 576 services and 103 standalone process identities checked; the completed capture adds one standalone predecessor |
| Saved capture | 14,505,713 bytes; hash pinned in the evidence manifest |

All pre-existing input hashes and exact official file sizes matched. Three
historical reference hashes were first bound during this capture; this does
not establish that those hashes had been recorded in the earlier experiment.
The root reviewer confirmed the original process had exited, independently
reacquired the serial lock, checked the recorded cleanup evidence and rechecked
all input and binary file identities. Canonical state files remained unchanged.

## Preparation problems fixed

The preparation reader had treated the complete historical fingerprint map as
UTF-8 source. That map also contains 14 historical raw tensors. The preparation
now authenticates their exact existing provenance and retains every fingerprint,
while reading current text separately. No historical raw payload is read again.
The ordinary reader admits the actual text metadata formats present in the
history; arbitrary model/native payloads remain excluded.

Three installed Python test fixtures use non-UTF-8 encodings. Their original
bytes are hashed under the same fixed site-packages directory, stable file
identity, no-symlink rule and 8 MiB per-source cap. All other text stays subject
to UTF-8 validation.

The complete evidence exceeds the former 8 MiB JSON limit. Explicitly named
preparation/runtime JSON envelopes now have a 32 MiB cap and structural checks;
ordinary sources remain capped at 8 MiB. The capture used one CPU core and a
1 GiB memory limit. Model-stage resource limits are unchanged.

Actual draft creation also exposed a catalog reference shape mismatch and a
missing v4 lifecycle-check filename. The original three-field catalog reference
was retained. The missing checker was created by changing only three imports
in its existing v3 counterpart; all 13 lifecycle regression tests passed.

## Remaining qualification

Preparation preserved all **51 scientific stages plus the raw verifier**,
the original 16 inputs, cold cache, long decode, state/slot reuse, quality and
paired workload requirements. NVMe entry/floor/output limits remain **48 / 30 /
18 GiB**. This capture did not create any model outputs; available NVMe storage
was 52,097,224,704 bytes at completion.

The old v4 draft was superseded by the reviewed v5 runtime. Its first actual
qualification run stopped in the experiment controller after alias workers
reported 70 passing checks; 62 had normal retained-exit proof. The external
exit was 1. A process-exit observation race was reproduced and a v6 successor
passed 23 source regressions. A fresh full-protocol run was launched on
2026-11-08; see the [failure, fix and timestamped launch evidence](nemotron-exit-race-20261108.md).
Full-model and paired performance qualification remain pending. No new Nemotron
speedup is claimed.

SmolLM3 preparation also advanced: the accepted controller and supervisor-stop
fixes are wired through nine coherent sio11 modules, with all 12 focused source
checks passing in the author's run and the root rerun. The original 12-round,
264-request, 33,696-output-token protocol remains unrun in sio11. The failed
sio10 attempt completed three rounds, 66 requests and 8,424 tokens; its matching
partial outputs cannot qualify the full protocol. No new Smol speedup is claimed.

## Evidence

The [manifest](evidence/nemotron-preparation-20261108/manifest.json) pins the
[actual exit](evidence/nemotron-preparation-20261108/actual-exit.json),
[root validation](evidence/nemotron-preparation-20261108/capture-validation.json)
and [completion review](evidence/nemotron-preparation-20261108/completion-review.json).
Weights, native libraries and tensor payloads are not included in this report.
Existing [speedup tables](multi-model.md) retain their stated workloads and
validation boundaries.
