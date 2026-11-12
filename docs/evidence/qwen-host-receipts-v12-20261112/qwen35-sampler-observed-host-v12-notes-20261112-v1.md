# Qwen v12: real startup settings and initial usage accounting

Source-only successor to the actual failed r9/v11 run. The old sources and
outputs remain immutable. This delivery neither recasts r9 as successful nor
supplies missing historical fields. It does not launch or qualify a model.

Eight new runtime sources are pinned: the six v12 runtime modules, driver v3,
and `qwen35_sampler_host_settings_v1.py`. The new preparation wrapper is
`prepare_qwen35_sampler_v12_candidate_v1.py`; it returns candidate objects in
memory and leaves the review unapproved. The original scientific delivery,
every inherited source/input, original lock identity, finite resources, short
IPC paths, and storage failure observations are retained.

The v12 adapter's `api_ready` already reads the actual API PID's `/proc/.../environ`
and checks the complete digest against the exact exec intent. After the existing
PID/start/argv/cgroup, ready API, and worker checks, it now validates the complete
observed environment against the original clean `native_environment` plus the
exact `phase_environment`. The complete observed map and comparable projection
are persisted in the same startup proof. Every later readiness observation must
match the first one. These are actual runtime observations only when a future
real run executes this path; tests supply explicitly mocked process values.

The comparable projection retains every API argument except the independently
validated final `--worker-cls` pair, and every environment entry except exactly:

- `QWEN35_SAMPLER_HOST_IDENTITY`, checked against the original recomputed host
  identity, including arm, phase, sampler, observation flag and source map;
- `QWEN35_SAMPLER_HOST_RECEIPT`, checked against this arm/phase's exact output;
- `TMPDIR` and `VLLM_RPC_BASE_PATH`, both checked against the original short
  control directory for this precise arm/phase;
- `QWEN35_SAMPLER_CAPTURE_ROOT`, required to equal this phase's packets directory
  only for capture, and absent for both measured phases.

These are omitted only after complete equality to the intended clean launch is
established. Unknown extra environment keys, changed thread or logging settings,
wrong arm/temp/output bindings, extra argv or changed model flags fail. An
environment digest recomputed after tampering does not allow these changes.
No default DEBUG behavior is changed. Absence of an override remains an observed
absence under the existing pinned vLLM sources; the direct `parallel_config`
object and runtime thread-count objects are still not claimed as observed.

Driver v3 validates this evidence at **every live server observation**, before
any client request. It recomputes the projection from the complete observed map,
checks its digest and the actual process argv, and compares it with the recorded
projection. It then carries `effective_host_settings` into its yielded host
receipt and final `host.json`. The original worker attachment file is untouched.
Before the first candidate HTTP request, it additionally runs the unchanged
`check_host_pair` against the saved baseline host. The original final
`compare_full_http` still runs, followed by the complete final graph-reuse check.
All original model, vector, capture, native replay, boundary controls, quality96,
initial HTTP and full HTTP phases remain in the same order.

Driver v3 preserves driver v2's initial-usage correction exactly: all three
count fields must have the expected integer type and value; both full usage
objects and greedy texts must agree. Optional API fields are no longer confused
with changed budgets. Original root regression coverage was 26 checks; an AST
contract proves this corrected function is identical in v3.

Validation: 17 guarded source/mock tests, tool `f2b6d3`, exit 0. The suite
blocks native/model imports. It executes the actual `Phase.api_ready` method
with supplied proc/environment/health mocks, and a full driver route using the
saved original r9 JSON behind explicitly fake execution ports. The full route
checks eight model calls, six server contexts, both original HTTP reports,
before-candidate host comparison, settings in saved host outputs, and the
original final graph-reuse call. Missing settings fail before the first client;
tampered candidate startup fails before candidate timing; cleanup remains
executed. No native outcome is inferred from mocked fresh-execution flags.

AST/source comparisons (`ea4712`, exit 0) establish that after version/import
normalization only `Phase.api_ready` changes in the adapter, only `run` changes
relative to driver v2, contract functions remain exact, and IO/history/service
and CLI only change version references. The source manifest and diff are:

- `qwen35-sampler-observed-host-v12-sources-20261112-v1.json`, SHA256
  `910eecb1b9e57f478e7c635fb98677de534a016037f4a00518e484a45119840d`;
- `qwen35-sampler-observed-host-v12-source-diff-20261112-v1.patch`, SHA256
  `0c7ab872a1559c44a3249a8fff6e64083c99e3303cf085584dfc3475741ba8d4`.

For root's next metadata preparation, retain the original explicit v9 future
fields. `terminal` and `catalog` must bind the **then-current complete idle**
handoff/catalog. The existing `prerequisite_exit` is the original completed LFM
r2 proof required by the original builder and its `lfm_r2_scope`, not a relabelled
Mini or Qwen receipt. All later actual terminals remain in the complete catalog.
`successor_sources` still references the original v9 manifest. The new API is:

```python
import prepare_qwen35_sampler_v9_candidate_v3 as v9
import prepare_qwen35_sampler_v12_candidate_v1 as v12

reader = v9.JsonMetadata(metadata_guard)
candidate = v12.prepare(explicit_future, v10_short_ipc_manifest_ref,
                       v11_storage_manifest_ref, v12_observed_host_manifest_ref,
                       reader.read, metadata_guard)
```

The unchanged v10 and v11 manifest paths are
`qwen35-sampler-short-ipc-v10-sources-20261112-v1.json` and
`qwen35-sampler-storage-diagnostic-v11-sources-20261112-v1.json`.
Root owns the actual candidate-file writes, exact independent source review,
physical admission and original-lock run. The explicit future runtime command
is the same interface with `research/run_qwen35_sampler_runtime_v12.py --mode
outer --spec SPEC --spec-sha256 SHA --review REVIEW --review-sha256 SHA`.
No current spec/review/handoff/catalog/admission was created by this delivery.

Root's independent final auditor must import driver v3's corrected
`check_initial_pair` and replay the new `from_live_observation` check for all six
server observations, compare its result with measured `host.json`, and retain
the original complete science, final-result, actual-exit and drain requirements.
The original r9 terminal auditor must not be relabelled as a v12 auditor.
