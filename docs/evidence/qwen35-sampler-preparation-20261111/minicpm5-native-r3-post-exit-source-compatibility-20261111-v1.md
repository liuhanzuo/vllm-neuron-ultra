# MiniCPM5 r3 post-exit compatibility

**The old science/receipt/true-exit checks are compatible with runner_v12 and driver_v5, but recorder_v2/audit_v3 cannot be reused unchanged.** Their required entry/core pins are already inherited by the current r3 model; their validators require those same pins in a disjoint new extension. This contradicts the no-overlap merge rule.

## Exact blocker and compatible behavior

[Recorder v2 line 52](/home/ubuntu/experiments/trn1-latest-models-20261031/research/record_minicpm5_layer1_operator_native_model_exit_v2.py:52) and [auditor v3 line 45](/home/ubuntu/experiments/trn1-latest-models-20261031/research/audit_minicpm5_layer1_operator_native_result_v3.py:45) require entry/core pins in the new extension. [merged_plan line 51](/home/ubuntu/experiments/trn1-latest-models-20261031/research/audit_minicpm5_layer1_operator_native_result_v3.py:51) rejects any inherited key, including an identical hash. All three paths already appear in the actual r3 overlay and retained catalog rows 87/91.

The lifecycle contract, command, owner, result-schema and unwrap functions in v2/v3 are AST-identical. Outer v4/v5 `verify_drained` is identical. [driver_v5 line 35](/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5_layer1_operator_native_driver_v5.py:35) returns the unchanged v4 scientific driver result; its schema remains v4. No change is needed to core_v5/core_v4 or the 24-frame/72-arm/936-raw-file science gates.

## Minimal new source paths

- [Recorder v3](/home/ubuntu/experiments/trn1-latest-models-20261031/research/record_minicpm5_layer1_operator_native_model_exit_v3.py)
- [Auditor v4](/home/ubuntu/experiments/trn1-latest-models-20261031/research/audit_minicpm5_layer1_operator_native_result_v4.py)
- [44 pure checks](/home/ubuntu/experiments/trn1-latest-models-20261031/research/check_minicpm5_native_post_exit_inherited_sources_v1.py)

The successor keeps the extension at **0–2048 only-new local text pins**, rejects every inherited-key overlap, and authenticates entry/core pins in the complete inherited-plus-new map. The current r3 can use just the two newly authored source entries; its package and old helper pins remain inherited and fully verified.

A roughly 2.4 KiB `source_binding` records exact original model spec/review/overlay references, full inherited-plan/map hashes and counts, the additive-map hash/count, the complete-map hash/count, and explicit entry/core hashes. The original complete source verification still runs. A future consumer must reconstruct the authenticated complete inherited map and check the binding; a digest assertion alone is insufficient. This keeps provenance without repeating all package pins in the recorder extension.

The [actual audit runtime](/home/ubuntu/experiments/trn1-latest-models-20261031/research/audit_minicpm5_layer1_operator_native_result_v4.py:225) now reads `ended.recorder_spec` and `ended.recorder_review` through its stable metadata reader. Before raw auditing, it checks their accepted review, model references, output and true exit, validates `ended.source_binding` using the fully reconstructed `resolved['plan']` and actual recorder_v3 producer, and verifies that the auditor source map covers every recorder source. This validation is called by the consumer, not left as an unused helper.

The original scientific, true-exit, physical-check and lifecycle predicates, source classes, descriptor release paths and result-output limits are preserved. Entry source-extension validation intentionally changes to accept inherited-or-new entry/core binding and a 0–2048 only-new map. No running source or configuration was edited.

## Catalog input size differs from result-output size

Catalog v50 is **2,001,785 bytes with 92 rows**, leaving **95,367 bytes** below 2 MiB. Its actual input path is `Metadata.read → sources.load → nemotron9b_package_metadata_v27.text`, whose ordinary text bound is **8 MiB**. `serial_catalog` separately limits rows to 256; it has no 2 MiB file-input cap.

The **2 MiB** limits belong to individual preparation/audit/recorder outputs and current handoff content. A pretty-printed copy of the r3 overlay's 3291-pin map is 702,073 bytes. Adding that map crosses 2 MiB; even two such map copies plus the existing catalog total 3,405,931 bytes before new-row overhead, still below the 8 MiB input bound. The actual future complete file still needs its size and hash checked.

Keep every old row unchanged and complete new provenance. Do not truncate history or reject a complete catalog because an unrelated root writer uses the 2 MiB result-output helper. This source increment does not change actual catalogs, catalog schemas, ordinary input limits or result-output limits.

## Future preparation successor is still required

[prepare_native_v6 lines 70/90](/home/ubuntu/experiments/trn1-latest-models-20261031/research/prepare_minicpm5_layer1_operator_native_v6.py:70) allow only the old exact producer paths. [Lines 105–121](/home/ubuntu/experiments/trn1-latest-models-20261031/research/prepare_minicpm5_layer1_operator_native_v6.py:105) also require producer/core entries inside the additive map. Consequently v6 will reject the future actual recorder_v3/audit_v4 exits.

A new preparation source version must explicitly support these two producers, authenticate their real terminal/spec/review/owner argv, reconstruct the complete original model plan, and validate `source_binding` before accepting inherited entries. Preserve all older validation branches and all existing rows. Current v6 and actual catalogs remain unchanged. No new `source_map_reference` catalog-row field is implemented or presently legal.

The 44 checks passed (tool chunk `851621`, exit 0), using the actual 3291-entry r3 overlay subset and complete 92-row catalog. They do not reconstruct the full historical base plan or run any recorder/auditor. Full source hashes, source excerpts and restrictions are in the [JSON review](/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-native-r3-post-exit-source-compatibility-20261111-v1.json).
