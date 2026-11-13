# Source-only r7 call contract

This delivery has performed no actual preparation, current binding selection, History, lock, compiler/model work, SDK binary read, cleanup, queue edit or repository edit. None of its tests or source manifests authorizes admission or asserts model qualification/speedup.

## Delivered sources

- `qwen35_package_serial_runtime_v9/source_delivery.json`: `7fc6b85ff1f241dbc548e85d6bbc3ef3a208b10d5592ad4559307cb3a8a5e12a`.
- `qwen35-package-serial-preparer-v8-source-delivery-20261113-v1.json`: `5963fed6d71e5db8220b0356ed7471fd3d8a6e84c2337090c7527f4dc81ae145`.
- `render_qwen35_package_r7_root_helpers_source_v1.py` implements pure `render(predecessors, bindings, audit_sources)`; it reads and writes no files.
- Static audit adapters: `qwen35_package_post_audit_v2.py` and `qwen35_package_audit_physical_v2.py`. Their only changes are runtime v8 → v9 and physical adapter v1 → v2 import routing.

The exact SDK manifest has 20 assets: existing 16 PWP entries and four DVE entries from the independently observed root r6 receipt. Its SHA is `d46ae4f1e70e5b2f735704b0572429e3a6c9c44394a31cf50e59da0d2f97567d`. The four DVE contents were never read by this author. Their receipt hashes match wheel RECORD text. Aggregate logical bytes 387197 and allocated bytes 417792 are counted accurately; installed root assets never reduce NVMe reserve. All original budget classes, entry and single-file limits, root/NVMe floors, headroom, six model binary identities, precision and all 25 scientific stages remain unchanged.

## Independent root procedure

1. Authenticate the two deliveries and static adapter sources; run independent source/synthetic review as appropriate. Publish a runtime review with schema `qwen35_package_serial_runtime_v9_root_source_review_v1`, reviewer `/root`, `passed:true`, `source_only:true`, `runtime_admitted:false`, `blockers:[]` and exact two-field `source_delivery`. Publish a preparer review with schema `qwen35_package_serial_preparer_v8_root_source_review_v1`, same source-only fields, `source_accepted:true`, exact two-field `delivery`, and the existing required `independent_review` reference. The source author does not invent the review references or their hashes.
2. Resolve the genuinely current catalog/idle metadata in root's original serial workflow. The minimum accepted catalog/idle suffixes are v119/v598 and minimum catalog rows 166; these are lower bounds from the observed predecessor, not selections. Do not reuse them if a later actual handoff/catalog exists. Renderer `bindings` has exactly `catalog`, `idle`, `running_handoff`, `catalog_rows`, `runtime_review`, `preparer_review`; all refs have exact `path` and `sha256`. `running_handoff` must be a fresh r7 name with strictly newer suffix/date. Review filenames must match exact v9/v8 root-review versions dated 20261113 or later. Pure rendering validates only shape; actual source helper and History checks authenticate current contents.
3. Supply all six frozen actual r6 predecessor bytes from renderer `PINS`, plus the two exact static `AUDIT_PINS` byte payloads. Invoke `render` only after these source inputs and actual refs are independently verified. The returned six files must be reviewed and exclusively created; fake test bindings must never be published. `exact_predecessor_reversal` reconstructs every entire r6 helper byte-for-byte. Budget/stat metadata differs only in fresh output/case names and the new budget reference.
4. Returned names are `prepare_qwen35_package_full_root_r7_20261113_v1.py`, `accept_qwen35_package_full_root_r7_20261113_v1.py`, `ack_qwen35_package_full_r7_root_20261113_v1.py`, `/home/ubuntu/qwen35_package_full_r7_post_exit_20261113_v1.py`, `qwen35-package-full-r7-20261113-resource-budget.json`, and `qwen35-package-full-r7-20261113-stat-inputs.json`. Publish all six and retain the static v2 adapters **before** executing preparation. ACK/observer byte hashes and both adapter hashes are computed/pinned in preparation sources; observer independently requires the same pre-enrollment.
5. Run actual prepare with every exact CLI ref from `required_actual_refs` using `python3 -B`, preserving actual-exit records and original source guards. Prepare uses preparer v8 and runtime v9, stat-checks all 20 installed SDK assets without opening binary contents, preserves all six model binary inputs and original source/history lineage. It still returns a false candidate. No copied stat identity is treated as a fresh observation.
6. Root acceptance remains bounded by RLIMIT_AS 2 GiB, RLIMIT_CPU (600,605), alarm/guard 600 seconds, original serial lock, complete History and all 874 inherited children. Recheck resource floors and reserve before execution. Actual root admission and runtime run/ACK must follow the original caller protocol. Runtime prelock hashes all exact 20 SDK assets with nofollow ancestry and stable identities; it does not accept arbitrary absolute SDK links. Capture actual tool session and exit through root's tool API, never by child inference.
7. After outer exit, use the exact r7 observer with explicit request schema `qwen35_package_r7_post_exit_request_v1`. Preserve every actual child/outer terminal, full owned-scope drain, complete physical pair, freshness and loaded-source checks; stage25 retains exact pure package imports. Only an independent actual successful 25-stage audit and root decision can qualify the model or support a speedup claim. Full testing must still cover cold-cache buckets, long decode and request-slot reuse.

## Limits of this fix

The source search for DVE/PWP generator names found no matches in installed `neuronxcc` `.py/.cc/.cpp/.h` files (actual rg chunk `86f4ac`, exit 1 meaning no matches). Some compiler implementation is compiled; this is not proof of generator completeness. These exact 20 observed targets are supported. Any additional SDK file, generation, arbitrary path, relative target, wrong-root basename or target mutation is rejected. A later compiler stop may therefore require new observed evidence. All 12 HLO generation and zero NEFF completion reported for r6 remain failure evidence; this source delivery does not change that result.

## Tests actually observed

- Runtime source/mock suite: 192 tests, exit 0, actual chunk `c7ece4`.
- Preparer source/mock suite: 65 tests, exit 0, actual chunk `b07b90`; original v2–v8 support and new exact v9/20 stat-only path retained.
- Pure r7 renderer: 15 tests, exit 0, actual chunk `018de7`.
- Preserved independent observer / physical/scientific checks: 25 tests, exit 0, actual chunk `7ee760`.
- Stage25 pure package import probe: five checks, exit 0, actual chunk `170202`; no native imports or model work.
- Frozen v8/preparer/adapter predecessor preservation: all 37 recorded file hashes and full filesystem identities unchanged; delivery hashes verified, chunk `5d7e3d`.

Two intermediate test expectations were corrected: unknown future runtime negative fixture moved from v9 (now supported) to v10; a renderer test now checks the basename used by `Q / filename` instead of assuming an absolute literal. Neither correction changed production logic.
