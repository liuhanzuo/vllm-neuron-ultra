# MiniCPM5 native/SDK comparison: bounded lossless report transport

Source/mock delivery, 2026-11-11. This extends only the representation of the complete reviewed v3 comparison report. It does not execute or authenticate an actual native/SDK comparison and does not admit a runtime caller.

## Interface

`research/minicpm5_native_sdk_report_transport_v1.py` exports `encode_report(report) -> bytes` and `decode_report(bytes) -> report`. The module has only stdlib imports: base64, hashlib, json, re, zlib. It has no file readers, CLI, model imports, process inspection, scheduler or runner. The caller supplies an already completed and authenticated v3 comparison; the existing guarded stable nofollow source reader and terminal/audit requirements remain external and mandatory.

The existing `packed_physical` and `unpack` definitions were copied verbatim from their historical sources; function AST equality passed without importing either historical module. No original helper was changed. The format remains `zlib-base64-json-v1` with exactly encoding, uncompressed_bytes, sha256, compressed_bytes, data and all_samples_retained.

The canonical complete JSON has sorted keys, compact separators and no NaN. The original uncompressed cap remains 32 MiB; compressed and base64 payloads each remain strictly below 2 MiB. The outer serializer additionally requires the complete stored bytes, including indentation and final newline, to remain strictly below 2 MiB. Decoding permits only one complete bounded zlib stream, exact declared lengths, matching SHA and valid base64. The complete decoded JSON must reserialize to the original canonical length/SHA. Duplicate envelope keys and noncanonical inner JSON are rejected.

Shape checks enforce the existing v3 schema, 936 native raw + 624 SDK raw + 72 metadata unique references, 24 frames and 48 rank/call/slot rows. Original boundaries, coverage, unknowns, SDK padding, scale provenance and alignment survive whole-object equality. The representation is the entire original report: no source projection, deduplication or summary replaces evidence. Shape checks do not authenticate evidence or independently prove that a declared report is complete.

## Measured synthetic capacity

Exactly one complete synthetic comparison was generated during session 64230. The final tool chunk is 60395f, exit 0; seven tests passed in 32.623 s. Canonical JSON was 8,161,901 bytes; zlib data 470,304 bytes; base64 627,072 bytes; the complete indented stored file 627,301 bytes. The remaining original 2 MiB allowance is 1,469,851 bytes; the remaining 32 MiB uncompressed allowance is 25,392,531 bytes. Complete decoded object equality, all 1,632 references and full canonical SHA passed. Exact digests and controls are in the accompanying metrics JSON. The synthetic bytes were verified in memory; this delivery does not claim a persisted or actual model report of that size.

Actual values may compress differently. If a real complete report exceeds any original cap, encoding must fail; this result does not authorize increasing limits or dropping evidence. There was no second full generation and no new actual experiment.

## Integration boundary

The comparator's independent root source review covers saved-byte comparison only, not runtime or this transport. The transport remains a candidate for parent review; this author delivery cannot declare independent acceptance. Root reported four further small codec controls passed in tool call 99d7ed, exit 0, 0.011 s; their source/review ownership stays with root.

Preserve the original model/weights/BF16, native 16 GiB/14 GiB/7,200 s limits, output 128 MiB/256 MiB limits, root 28 GiB/NVMe 30 GiB floors, and cold-cache/long-decode/request-slot-reuse requirements. This pure report transport changes none of them. Original SDK model exit 1 remains 1; full-model qualification, causal localization and any new speedup claim remain false. Unknown native PV-prior, PV-active and o-projection local-before-reduce boundaries remain unknown.
