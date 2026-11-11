# MiniCPM5 native operator reader and runtime integration design

Status: source design only. The native verifier v2 has independent source
acceptance. No runtime admission, model import, raw read, lock/process probe or
real historical completion replay was performed for this design. This is not a
concrete launch spec and does not authorize a run.

## Frozen components and contracts

Keep driver_v4, journal_v1, native manifest_v1, recovered completion_v1 and
RuntimeContext_v1 unchanged. Use accepted output_catalog_v2/results_v2. The
associated design manifest records exact source hashes and accepted reviews.
The original SDK operator model exit remains1; recovered624-file data is a
separate successful audit prerequisite. The native result still requires a real
native execution, original official BF16 weights and completed donor metadata.

The present overlay_v1 derives only a driver catalog and explicitly returns
full_runtime_catalog_pending=true. Prepare overlay_v2 as a successor that imports
output_catalog_v2.catalog(contract), binds its complete digest in the in-memory
resolved plan and returns the complete finite runtime contract. Preserve original
source/input maps; extensions cannot replace existing entries. Preserve the old
successful SDK batch/layout binding and the separate recovered-operator binding.
Do not write the full resolved plan or duplicate the historical source map into
new giant spec/review files. Small overlay/spec/review references, extension pins
and resolved-plan digests are sufficient; complete historical validation still
runs on the original plan under admission.

The final current serial catalog must be supplied by root during concrete
preparation, preserve all historical rows and include actual recovery/recorder
terminals plus every later serial operation. No catalog version is fixed here.

## Actual reader and final inventory adapter

Implement a source-only `NativeAuditIO` adapter with explicit guard and admitted
root/native donor scope; no constructor IO. Its future audited methods are:

- `read_bytes(ref, absolute_path, exact_size)`: authenticate the exact allowed
  output/donor reference, canonical parent chain and owned singleton regular
  file; snapshot lstat identity; open with O_RDONLY|O_NOFOLLOW|O_NONBLOCK;
  compare fstat with the snapshot; read in bounded chunks with guard before
  each read, demand exact extent and EOF, verify full SHA256, fstat again, close,
  recheck path lstat/parent identities and guard. Never follow a substituted
  parent or a symlink. Admit only936 raw,72 result JSON,72 journals and96 donor
  files. Every path is read exactly once by the pure verifier.
- Raw/tensor/journal refs already carry captured identity. Use the original
  `payload_io_v1.read_owned_payload` after an explicit mapping from raw_bytes /
  sha256 to payload_bytes / payload_sha256 and a check that the declared size
  fits that helper's existing MAX_PAYLOAD. Do not raise its cap. For any declared
  extent that cannot use it, stop preparation for separately reviewed handling.
- Arm result JSON refs carry path/bytes/hash but no captured identity. Bind their
  current identity to the independent post-exit inventory and apply the same
  stable descriptor checks. They are bounded by2 MiB individually, and all arm
  JSON plus journals remain within8 MiB. Parse with results_v2.json_object so
  duplicate keys, nonfinite literals and exponent overflow reject.
- `read_source(path, sha)`: preserve the existing partition-aware text policy;
  no historical payload or shared library may enter a text reader. Compile the
  native source for its trace descriptor only; never execute that module.
- `final_inventory()`: only after all native processes have exited, walk the
  exact owned root with no-follow descriptors; retain file and directory
  identities before/after traversal. Emit the exact catalog entry fields and a
  separate identity map. Refuse extra paths, symlinks, hardlinks, changed entries
  or partial staging files. Feed entries into output_catalog_v2.inventory with
  successful_driver=true. Snapshot the output root's own allocation separately
  and add it to the total cap in the runtime monitor; the catalog's97 native
  directories already count toward both128 MiB driver and256 MiB total caps.

The final inventory is not a producer list. Its identity map is the authority
for result JSON reads and a cross-check for captured identity refs. A failed or
changing tree stays failed; do not regenerate hashes to make it pass.

## Three-process runtime ownership

Use the existing native_cache_runner_v1 outer/coordinator/native topology as
source reference, with new entrypoints, schemas and exact command vectors.
Do not copy its scientific validator or old SDK usage loop: the new accepted
catalog includes directory allocation, whereas SDK runner_v5 usage skips it.

1. Outer authenticates concrete small spec/review and all loaded text sources,
   acquires the original serial lock and keeps its actual descriptor until
   cleanup is proven complete. It resolves the recovered boundary and overlay
   in memory, verifies latest-terminal freshness and performs the complete paired
   current physical drain. It checks no competing model/compiler/performance
   work and original floors (root28 GiB; NVMe30 GiB plus256 MiB output reserve).
   Authenticate the exact inherited native-binary source partition once under
   this lock and retain its receipt/identities. Only then create admission and
   launch one constrained coordinator service.
2. Coordinator is acknowledged by outer using actual PID/start/argv/cgroup and
   service invocation identity. It spawns exactly one native child with the
   original native interpreter/environment and acknowledges the child's actual
   PID/start/parent/argv/cgroup before permitting work. There is no TP2 rank or
   rendezvous port requirement in the native child.
3. Native constructs the original permit with role=native, rank=-1, arm=HF_native,
   actual process identity, overlay reference, resolved-plan digest, inherited
   source-authentication receipt and recovered binding. Its external guard
   retains actual owner/parent/service/lock/current predecessor checks. Context
   source verification and a full preimport physical guard must pass before
   driver_v4.run(context, guard, native_bundle) can import/load native weights.
   All72 direct calls,24 checkpoints,936 raw files,72 journals, full saved cache
   history, scale/RoPE provenance and original precision remain unchanged.
4. Native saves its result with self_exit_claimed=false. Coordinator uses its
   child's actual wait status and refuses a result-only success. Coordinator
   saves a compact reference to native-result.json plus actual child identity /
   exit and bounded metadata checks; it must not inline every per-arm record.
   This is lifecycle success pending independent raw audit, not verified model
   qualification. Outer observes actual service exit, validates coordinator and
   native refs, proves owned group/children drained and then records its result.

Keep the existing native service envelope:16 GiB MemoryMax,14 GiB MemoryHigh,
no swap,128 tasks,CPUs6-7,two native threads,Nice19,KillMode=control-group,
7200s runtime,15s stop timeout and16 MiB per-file hard limit. Exact interpreter,
environment, argv and properties must be bound by the new concrete review.
No resource increase is implied. Bounds may only be tightened with reviewed
consequences and actual observations; driver work itself remains unchanged.

Runtime monitoring uses catalog_v2 with successful_driver=false during writes,
accounts for every directory/file allocation and accepts only exact acknowledged
atomic staging names. A live scan can observe an expected pending-to-final rename;
handle only that same identity and declared destination, never general missing
files or unknown PIDs. Final scan after drainage is immutable and complete.
All16 declared lifecycle names and15 staging slots remain finite. Root allocation
is an additional total-budget charge, not a new descendant path. Each JSON cap
remains8 MiB; runtime.log16 MiB; native tree128 MiB; entire tree256 MiB. Compact
lossless references and bounded physical-pair encoding prevent another oversized
nested coordinator report. Exceeding a cap fails with retained evidence.

On error, preserve the original exception and all saved journals/raws. Retain
cleanup-pending until actual children and owned group are drained; retain the
original lock through cleanup. A success-looking report, child exit0 or missing
unit alone is insufficient to release. Preserve lock-failure evidence; do not
weaken the original exact-one-row guard or add blind retries. Parent fd proof
and kernel investigation remain root-owned and must be resolved or separately
reviewed before a concrete launch.

## Independent model and auditor terminals

After the outer tool session really ends, root's separate recorder authenticates
its actual session/chunk/exit, exact owner gone, complete service/child drainage,
all result/spec/source/overlay refs and original lock reacquisition. A native
exit0 report must not stand in for this external terminal. Record the recorder's
own actual exit separately. Append both actual operations to the current catalog.

Only then prepare the independent post-exit raw auditor. It acquires the same
original lock, freshly authenticates its reviewed source closure and latest full
serial scope (including native and recorder), proves both physical passes, binds
real native exit and executes results_v2.verify with the stable reader above.
It must reread936 raw+72 results+72 journals+96 donor files, check2,736 journal
records and both SDK official-shard projection proofs. It writes a bounded report
with self_exit_claimed=false. Root independently observes the auditor's terminal,
reacquires the lock and records its result and separate recorder terminal.

The raw verifier itself deliberately returns actual_process_exit_verified=false
and fresh_physical_drain_performed=false. The admitted auditor's report must keep
that core scope and add independently authenticated lifecycle/physical evidence;
never mutate those flags to imply the pure function performed external checks.

## Required source controls before concrete admission

Use pure-memory/fake-descriptor tests first: symlink/parent replacement, changed
inode or timestamps, short/growing/trailing read, hash mismatch, failure at every
guard, directory allocation and root allocation, pending-name identity races,
wrong permit role/overlay digest, stale terminal, missing predecessor owner,
changed source/native identity, duplicate child, child failure/hang, service
failure, cleanup failure while lock retained, malformed result, cap overflow and
external exit forgery. Retain the22 accepted verifier tests unchanged.

Review actual reader, overlay successor, runner/worker, compact receipts, complete
source graph and exit-recorder adapters independently. Only then build a small
concrete spec using a fresh current catalog and measured disk space. Synthetic
checks do not establish a real model result. After actual native/audit completion,
SDK/native operator comparison is the next diagnostic stage; quality96,cold4096,
long decode,slot reuse/inactive and matched HTTP gates still precede any new
model qualification or speedup claim.
