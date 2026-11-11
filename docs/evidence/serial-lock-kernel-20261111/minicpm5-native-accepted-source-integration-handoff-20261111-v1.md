# Accepted native operator source interfaces and next wiring

All three source increments are accepted by independent root review: verifier
v2, actual reader v1 and complete overlay v2. Actual runtime remains unadmitted.
This handoff performs no new implementation, model/raw read or process/lock
operation. Existing native driver_v4/journal_v1 and recovery completion/context
remain unchanged. No model or speedup qualification follows from this status.

## Exact accepted callable boundaries

| Component | Callable | Output / obligation |
|---|---|---|
| recovered_completion_v1 | completed(BINDING, read_json_ref, guard, read_source) | Original immutable base plan, real recovered binding, native donor bundle and complete historical scope; original SDK model exit stays1. |
| native_overlay_v2 | resolve(overlay, recovered, read_json_ref, guard) | Memory-only plan/plan_sha256, native_bundle, current_scope, complete native_runtime_output_catalog and its digest. Eight small fields; no large spec copy. |
| native_runtime_context_v1 | RuntimeContext(plan, plan_sha256, permit, external_guard) | Existing native permit/source/input/environment guards. Permit binds overlay ref, resolved digest and source receipt. |
| native_driver_v4 | run(context, guard, native_bundle) | Admitted actual native direct attention capture,72 calls/24 frames/936 raws/72 journals, unchanged BF16 and official full-cache controls. |
| native_audit_io_v1 | NativeAuditIO(plan, native_bundle, guard) | Pure construction. After actual model exit: final_inventory(), read_bytes(ref,path,size), mandatory unchanged() final rescan. |
| native_results_v2 | verify(result,plan,native_bundle,sdk_results,read_bytes,read_source,guard,entries) | Independent936 raw+72 result+72 journal+96 donor byte verification,2,736 journal records; process exit/fresh drain flags remain false in this pure core. |

Module names above share prefix minicpm5_layer1_operator_ except the existing
runtime context at root and the accepted files under research. Exact paths and
hashes are in the companion JSON. Original source partition is still used for
read_source; a generic binary reader must not replace its classification policy.

## Correct recovered SDK rank inputs

Read these refs with the original authenticated metadata reader and current
guard, in rank order:

- recovered['recovery']['original_model_outputs']['rank0_result']
- recovered['recovery']['original_model_outputs']['rank1_result']

Their original files are:

- /mnt/nvme/minicpm5-layer1-operator-sdk-r2-20261111/rank0-result.json
- /mnt/nvme/minicpm5-layer1-operator-sdk-r2-20261111/rank1-result.json

Use the hashes from those authenticated refs, not freshly invented references.
The path prefix follows the frozen original operator-r2 spec; this handoff did
not reopen those actual rank outputs. These are the failed outer run's successful
rank outputs recovered by the separate successful raw audit. Do not substitute
completed batch/layout rank outputs from another experiment. The original
layer1_operator_boundary remains the completed batch/layout binding and must
not be rewritten to disguise the operator outer's exit1.

Pass resolved['native_bundle'] and resolved['plan'] to both driver and audit.
The two authenticated recovered rank results supply the official projection
matrix proofs for native_results_v2.verify; original SDK operator raw data stays
at its original paths for the later SDK/native boundary comparison.

## Remaining concrete wiring, in order

1. Root resolves the actual lock-observation/lifecycle issue and determines the
   reviewed unchanged-lock launch path. No deduplication, guard weakening or
   blind rerun is supplied by these source increments.
2. Add small concrete spec/review and exact source extension for accepted
   components. Use root's fresh current serial catalog, complete historical
   source/input maps in memory, original model/precision and original floors.
   Keep all historical rows; bind actual recovery and recorder terminals.
3. Wire a new outer/coordinator/one-native-child runner using the old native
   runner only as a topology reference. Outer owns original lock through cleanup;
   actual PID/start/argv/cgroup/service invocation acknowledgements precede
   child admission. Native RuntimeContext checks and full current physical
   drainage precede model import. Keep original resource/cap envelope.
4. Runtime usage must apply complete catalog_v2 and include output-root directory
   allocation in total256 MiB. All97 native directories count in driver128 MiB
   and total; arm JSON+journals stay8 MiB. Retain exact staging writer identities.
   Use compact references for native/coordinator results to avoid repeating the
   oversized nested-summary failure. Persist failures without replacing the
   original exception; retain lock until actual children/group drainage.
5. Root separately observes actual model/outer exit and recorder exit. Only after
   their actual terminal records are in current serial scope may the independent
   native auditor acquire original lock and perform fresh paired physical checks.
6. Auditor uses final_inventory → verify with the correct recovered rank refs →
   unchanged. All three must pass. It reports its independent lifecycle evidence
   alongside the pure proof without rewriting the proof's scope flags. Root then
   records actual auditor exit and separate recorder exit; presence of result
   files cannot replace actual external terminal observation.
7. Compare native/SDK saved operator boundaries; retain full cold4096,long
   decode,request-slot reuse/inactive,quality96 and matched HTTP qualification
   before claiming any new qualified model or speedup.

The detailed design remains in the earlier reader/runtime integration design;
its proposed old packed-payload-reader reuse was superseded by accepted
NativeAuditIO's direct stable descriptor reader within unchanged manifest caps.
No runtime entrypoint is created by this handoff.
