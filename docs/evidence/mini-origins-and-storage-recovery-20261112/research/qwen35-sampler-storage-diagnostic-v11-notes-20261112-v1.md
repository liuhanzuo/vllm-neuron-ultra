# Qwen v11 storage rejection observations

The actual r8 v10 supervisor failed its storage guard before API startup, then
killed the still-running numeric child during cleanup. Its saved exception did
not name the failing filesystem or record its instantaneous free space. This
source change cannot reconstruct that missing historical observation or prove
OOM. It only makes a future storage rejection interpretable.

The six additive v11 runtime files preserve the v10 short IPC paths. Contract,
history, service, adapter and CLI differ only in versioned references. After
normalizing those references, the only changed IO function is `resources`.

`resources` still calls the original `storage(spec)` first, then samples root
and NVMe in that order, at most once per filesystem, failing immediately at the
same original comparison. Root still requires 28 GiB. NVMe still requires
30 GiB and, only with `reserve=True`, the unconsumed portion of the same 2 GiB
output cap. The comparison, reserve expression, memory checks, successful
return schema, monitoring frequency and cleanup are unchanged.

On failure, the existing ValueError prefix is followed by a compact JSON object
with `path`, `available_bytes`, `floor_bytes`, `remaining_reservation_bytes`,
`required_bytes`, `shortfall_bytes`, `failed_constraints` and `reserve`.
`device` is explicitly `{kind: "statvfs_fsid", value: ...}`: this is the
filesystem ID from the **same** statvfs result, not a fabricated `st_dev` or a
second filesystem sample. `failed_constraints` reports the floor and/or the
requested reservation. No new output file, service or probe frequency is added.
The existing traceback/error capture retains the detailed message.

Twenty focused source/mock tests passed, actual tool `63d2a6`, exit 0. They
compare old and new decisions, successful returns and observation ordering for
600 boundary combinations; separately cover first/second filesystem failure,
exact equality, reservation exhaustion, both failed constraints, actual byte
arithmetic, unchanged memory/storage failures, and no additional `stat` call.
Source AST checks establish the narrow change and exact retained IPC functions.
Preparation tests establish old pin retention, six additive files, unapproved
review, fixed limits, source conflicts, wrong manifests/identity and path
collision rejection. Tests use system Python `-B -s`, CPU 7, nice 19, 512 MiB
address space, 60-second CPU cap and an import blocker for native/model stacks.
They do not call real statvfs, read proc memory or start a process/service.

The preparation API keeps the original v9 `explicit_future` fields, including
`successor_sources` pointing at its frozen v9 manifest. Supply the v10 short IPC
manifest and the new v11 manifest separately:

```python
import prepare_qwen35_sampler_v9_candidate_v3 as v9
import prepare_qwen35_sampler_v11_candidate_v1 as v11

reader = v9.JsonMetadata(metadata_guard)
candidate = v11.prepare(explicit_future, v10_manifest_ref, v11_manifest_ref,
                       reader.read, metadata_guard)
```

The new manifest is
`qwen35-sampler-storage-diagnostic-v11-sources-20261112-v1.json`, SHA-256
`8f61e23f0cab7dc1f03252d47d4d43d9b95c6874447b0bc99fd35e1e6fec4e5d`.
The wrapper delegates to the original v10 preparation, then retains every
returned source/input and adds six v11 source pins plus the authenticated
manifest identity. It rebuilds overlay/spec/review references in memory and
keeps `passed=False`, `launch_authorized=False`. It neither chooses the current
catalog nor writes any actual spec or review. Root must bind the latest complete
idle history, audit this source and admit a fresh serial run.

The source-only union with the executed r8 map is 48,031 old pins plus six new
pins, 48,037 in total; every old pin and the original v10 source bytes remain.
This is not a future bound spec. The complete runtime Bundle was not resolved;
no native/actual filesystem failure, API, quality96 or HTTP check was executed.
No new model qualification or speedup follows from this delivery.

One initial metadata construction attempt (`b7a153`, exit 1) tried to apply
`ast.literal_eval` to the arithmetic expressions in LIMITS. It failed before
writing any output. The corrected source-only check (`b3d6e4`, exit 0) compares
the exact old/new LIMITS AST and uses the unchanged authenticated v10 manifest
values. The runtime and test sources were not changed by that correction.
