Finite Mini retained retirement caller v2: retained foreground helper scope

This is an immutable successor to the frozen v1 delivery. Its primary functional
change is ownership of the privileged read-only holder inspector. v1 used
subprocess.run(timeout=90), whose timeout cleanup could kill/reap only sudo and
leave its helper or sudo monitor alive before unconditional lock.release(True).
v2 registers each foreground scope before Popen construction, retains its pipes
and Popen object, and invokes communicate(timeout=90) without killing anything.
The owner waits under the original retained lock for the complete authenticated
scope to disappear before publishing a terminal or releasing that lock. A
90-second failure stays a TimeoutExpired failure even if the helper later exits0.

The helper installs a 90-second real-time alarm before permit/plan IO and an
audit hook rejecting fork/forkpty/posix_spawn/subprocess/system/exec. It emits a
fresh nonce-bound start line containing actual helper -> optional sudo monitors
-> direct sudo -> exact retained caller identities (PID/start/PPID/argv/cgroup).
A valid direct sudo exec of the helper is represented by the two-identity
helper -> caller chain. Every sampled identity and the original lock descriptor
is rechecked before the ACK. The ACK is followed by the original two complete
all-UID inspection results. stdout remains capped at1 MiB and stderr at64 KiB.
The parent requires exact permit, nonce, expected helper and sudo argv, an
unbroken ancestry ending in its actual Popen PID and retained caller, successful
foreground pipe completion/wait, and independent gone() checks for every
captured non-caller identity. It never treats only Popen.wait()/exit as drainage.

The owned scope here is this exact foreground sudo execution chain plus the
helper, whose Python source forbids creating or replacing a child process. No
new service, daemon, cgroup manager, general process killer or cleanup framework
is introduced. Detached/unobserved execution is not admitted. Missing/invalid
ACK, ancestry mismatch, signal-killed sudo, uncertain Popen construction or
observation errors keep ownership pending. Normal OSError pre-exec failures
are accepted as child-free only with the exact executable filename and the
bounded known exec error codes; all other constructor errors remain unknown.
A sudo policy rejection without the helper ACK is also pending, not proof that
nothing remains. It requires a separately reviewed recovery after independent
scope inspection; there is no automatic recovery switch or lock-release bypass.

Cleanup, verification, notice and retry exceptions are contained by the existing
retain_until_drained helper. SIGINT/SIGTERM set the stop condition in both run
and finalize while drainage continues. Root must preserve this foreground owner
while drainage is pending; forcibly killing it is not a supported recovery.
The original execution/ACK/resource/storage limits remain unchanged; retaining
ownership after a failure deadline is cleanup, never permission for additional
retirement work. No unrelated Claude/xiaogou/user process is signalled.

Every holder result includes its drain proof. run's terminal stores the bounded
rolling chain digest, completed-scope count and last full scope evidence.
finalize validates the producer proof, includes its current proof in the final
receipt and writes one finalizer-helper-drain.json under the existing control
root before release. No per-scan files are created, and all evidence remains in
the unchanged128-file/128-MiB control reservation. The4706-file journal and
2352-pair scientific/storage contract are unchanged. Specs use
minicpm5_retained_retirement_spec_v2, successful helper results use
minicpm5_retained_retirement_all_uid_holders_v2, and run terminals use
minicpm5_retained_retirement_owner_terminal_v2. ACK/finalize CLI shapes and all
frozen transaction/replay/review/publication consumer schemas remain unchanged.

The v2 execution closure is the union of the frozen660-source v1 closure and
the newly discovered661-source static closure:664 explicit Python source pins.
All overlapping hashes are required to agree. It conservatively retains the
old v1 sources while adding exactly the four new implementation files. Actual
loaded-source and independent global-map checks still execute unchanged. The
four SOURCES entries are contract, caller, holders, scope in that order. Future
binding must enroll all664 execution pins into the full History overlay and use
the exact four new mandatory pins when calling render_spec(). This delivery
selects no current refs, obtains no lock, runs no History and deletes nothing.

The original completed_outputs predicate also incorrectly required every old
catalog source pin to occur in the new overlay extension. The recovered source
map has3484 prior pins that can belong to the authenticated base/global map.
v2 validates every old catalog pin against that complete global map, preserves
every prior row verbatim, validates all overlay pins against the same map and
requires all new retirement sources in the extension. The new catalog row
contains only extension pins not already in prior rows. Prior pin corruption or
missing new pins is rejected; the extension is never substituted for the global
map. A synthetic3484-pin fixture exercises the actual original catalog consumer
without reading current History or selecting a current catalog.

The audited synthetic suite has41 tests: the original20 scientific/source/
publication/lifecycle checks, plus recovered3484-prior-pin global-map/overlay separation and success/timeout/interruption, live helper or
monitor after direct sudo exit, cleanup/notice/retry errors, exact pre-exec and
unknown spawn failures, missing/forged ACK, signal-killed sudo, helper alarm and
no-child audit, bounded rolling evidence, both real entry functions' terminal/
publication/release ordering and immutable-v1 preservation. All process, lock,
History and filesystem fixture behavior is injected in memory. The test audit
rejects actual JSON/raw/proc/sys/NVMe access, process creation/signalling, flock,
network and native model imports. These are source/lifecycle tests; actual
privileged launch compatibility and actual removal are not claimed.


The accepted deletion core already exists at
minicpm5_retained_raw_transaction_v1.retire(plan, verification_ref, intent_ref,
journal, guard). This delivery adds only its finite root caller, pure binding
contract and privileged read-only all-UID holder inspector. No frozen source is
modified. No actual cleanup, current History, current catalog selection or raw
read occurred during implementation/testing.

The new files are minicpm5_retained_retirement_contract_v2.py,
minicpm5_retained_retirement_caller_v2.py,
minicpm5_retained_retirement_holders_v2.py and
minicpm5_retained_retirement_scope_v1.py. There is no automatic successor from
the current replay. Actual full replay, independent observer exit and completed
catalog/idle finalization must exist before the future root selects references.

Binding is explicit and pure: call contract.render_spec(run_name, references,
history_spec_value, original_replay_plan_value, new_four_source_sha256,
execution_source_sha256). The execution source map is the delivered complete
static local import closure; dynamic actual local imports are additionally
checked before transaction entry and by the frozen transaction itself. The
transaction text map is exactly original1869 sources plus this executing Python
closure. It is not the mixed full current source map. The complete global map
stays unchanged in history_spec/source_sha256 and is authenticated by the
original complete History resolver, including every binary/source/input role.

references has exactly: history_spec, history_review, idle, replay_outer_exit,
replay_observer_exit, replay_completed, verification, replay_equality,
writer_exclusion. verification and replay_equality retain the actual three-field
{path,sha256,bytes} receipt form. Other refs have {path,sha256}. All must be real
future root-selected refs, never placeholders from this source delivery.

The history_spec must be prepared against the latest complete post-replay
catalog, with the original overlay/recovered maps and every new execution
source enrolled. Existing metadata-only preparer v2 can build that complete
future overlay/spec; only its History view is used here. It must receive an
updated source closure containing these exact new sources. No raw replay mode
is launched from that history spec. history_review must be an independently
accepted /root review of that exact spec and complete source map. The original
latest_handoff and full physical History consumers execute unmodified.

Concrete offline binding sequence (all values are supplied after real closure):

1. Preserve the accepted metadata-preparer-v2 seed_spec/seed_review/seed_binding
   refs. Use the real latest complete catalog/idle, with prerequisite_exit/tool
   equal to that catalog's exact last normalized actual-exit row. Immediately
   after Mini closure this is the independent observer row, not a guessed numeric
   catalog version. Its empty units list is valid and retained.
2. Form a new preparer source-delivery manifest with the original accepted v2
   source map union this retirement delivery's execution_source_sha256. Require
   equality for every overlap. Keep the v2 preparer schema and source_only=True,
   launch_authorized=False. Root reviews that exact extended manifest using the
   existing preparer_root_source_review_v2 schema. The accepted lifecycle
   delivery/review and old mini_source_review refs remain the same.
3. Supply a fresh future run_name such as retirement-history-r1 and all seven
   future path fields to prepare_minicpm5_retained_replay_binding_v2.prepare (or
   its CLI). This yields the complete overlay and spec with all old sources,
   metadata identities, new retirement Python pins, exact large envelopes and
   the real post-Mini catalog. Keep its false launch review draft unchanged.
   Independently review the returned exact History spec/source map at the
   reserved review path; leave launch_authorized=False because its pair/replay
   modes are never admitted by the retirement task. The retirement caller only
   consumes the complete History view and this accepted source-map review.
4. Reserve a fresh retirement spec path. Publish the explicit writer_exclusion
   attestation naming that path and run_name; it does not contain the eventual
   spec SHA. Build the ten-reference dictionary above from authenticated actual
   files, including the real successful replay/observer receipts, completed Mini
   handoff, paired-verification and equality refs with their complete byte sizes.
5. Pass the authenticated history-spec object, unchanged original replay-plan
   object, exact four source pins and this delivery's execution source closure
   to render_spec(). Serialize it with base.publication_bytes(), calculate its
   SHA and write exclusively. Root then publishes the independent removal
   review binding that exact spec ref and the source/replay facts below.
6. Before run, authenticate every explicit reference and ensure all output paths
   are absent. No shell placeholder, predicted owner, predicted tool session or
   predicted successful exit is an admissible binding. The actual run repeats
   complete physical History/freshness under the original lock.

writer_exclusion is root-produced metadata with schema
minicpm5_retained_retirement_writer_exclusion_v1, reviewer=/root, spec_path equal
to the future retirement spec path, run_name, storage_plan equal to the frozen
map ref, original_lock_identity equal to the original device/inode,
other_task_agents_excluded=True, external_namespace_writers_excluded=True and
retained_for_entire_transaction_and_finalization=True. spec_path deliberately
avoids a content-hash cycle. This is an explicit operational exclusion promise;
neither idle state nor a one-time scan implies that all other writers are
excluded. The parent root coordinates its other agents and external writers.
The all-UID scan is additional observable evidence, not an atomic unlink-if-inode
claim. Linux's documented stat/unlink interval condition remains unchanged.

The separate run review retains the frozen schema
minicpm5_retained_raw_replay_and_removal_review_v1 and all its required fields:
plan_sha256, verification, remover_source, replay_source, source_sha256,
replay_equality, passed=True, blockers=[], full11859_logical_inputs_preserved,
full32_replay_and_all_scientific_outputs_equal, all_original_failures_preserved,
original_weights_precision_and_remaining_gates_unchanged (all True). It adds
reviewer=/root, spec=<exact retirement spec ref>, launch_authorized=True,
production_deletion_admitted=True and execution_source_sha256. source_sha256
is the exact text submap above. These true statements are produced only after
the root independently verifies actual successful replay/observer/finalization;
this delivery does not generate or invent them.

Use the exact run argv from the original experiment root:

    /home/ubuntu/venvs/vllm_neuron/bin/python -B -u \
      /home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5_retained_retirement_caller_v2.py \
      --mode run --spec SPEC --spec-sha256 SHA --review REVIEW --review-sha256 SHA

The one root process owns the original retained serial lock. It launches no
model or compiler service. It applies CPU7, Nice19, 6 GiB address-space and
32 MiB per-file caps, a six-hour execution deadline and 180-second actual-tool
ACK deadline. This bounded metadata/removal process is separate from the
unchanged replay worker/cgroup resource contract.

Before any deletion it checks all actual replay/observer receipts, the complete
current catalog rows/owners/units and unchanged scientific equality; resolves
complete current History; checks zero candidate intersection across all resolved
current inputs, input_extension and both full source maps; performs the original
complete two-pass owner/service/cgroup observation; and runs two all-UID holder
passes through sudo -n /usr/bin/python3 -B holders_v2.py. The inspector proves
the exact caller lock descriptor before/after scanning, checks all process argv,
target/keeper inode FDs, target-parent directory FDs and mapped inode identities,
and rejects missing permissions or other scan errors. Only the exact current
remover PID/start is excluded from its own intentional transaction descriptors.
It neither stops nor signals unrelated Claude, xiaogou or other processes.

After the actual live owner is printed, root observes the true outer tool
session and invokes --mode ack with request {permit,actual_tool,updated_at}.
actual_tool is exactly {kind:tool_session,session_id:<actual integer>}. The ACK
entry independently proves the owner/lock/previous idle before publishing the
complete running state. No deletion occurs before this exact ACK.

The caller then writes complete History compatibility and exclusive lease
receipts, creates the whole durable intent while every original still exists,
and calls the frozen public transaction. That transaction independently hashes
its complete text sources, recomputes the full original/new replay comparator,
fully rechecks each original/keeper pair, records durable per-pair intent,
performs exact dirfd unlink and parent fsync, records each completion, and
rereads every keeper after all originals are absent. Sources, lock, handoff,
resources and holder state remain checked during the transaction. Blocking
holder scans are followed by forced metadata/source/latest-handoff checks.

Only the fixed14x168 original restore-after raw files can be removed. Keepers,
all98 serialized outputs, weights, manifests, original failures and qualification
gates remain. Expected original allocated bytes are 10,173,284,352
(9.474609375 GiB), checked again against the actual verification accounting.
Filesystem-wide free-space change is not asserted to equal this byte sum.

The retirement journal is a separate root-level
minicpm5-retained-raw-retirement-<run> namespace. First run allows exactly2352
intent files,2352 done files,postcondition.json and complete.json:4706 files,
128 MiB total. Control evidence has its own128 MiB/128-file budget. The original
replay512-entry budget is unchanged. Original root28 GiB/NVMe5 GiB floors and
actual remaining reservation are enforced. There is no implicit retry/migration
of an occupied namespace. All partial/failed evidence is preserved; incomplete
JSON recovery still requires the separately reviewed successor described in the
frozen retirement contract.

After root observes the actual run exit, invoke --mode finalize with request
{spec,review,terminal,actual_tool,updated_at}. actual_tool includes the real
session_id/exit_code/terminal_chunk. The independent finalizer checks the actual
remover is gone, reacquires the original lock, repeats complete History and
holder inspection, and on exit0 independently rereads all2352 keeper bytes and
verifies original-path absence. It validates the transaction result, complete
done-path selection and exact allocated-byte total. Failed runs remain failed
and retain partial journals. An unacknowledged preflight failure requires an
explicit root failure publication; it is not silently promoted to completed.

Final publication preserves every old catalog row, adds one actual retirement
tool/owner row with only new source pins from the authenticated full overlay
extension, and writes actual-exit -> catalog -> idle. The original catalog
consumer validates the pending complete values before publication. The catalog
points to the exact final idle hash; idle carries next catalog path to avoid
hash cycles. Partial publication cannot advertise a usable new idle. Written
receipt identities are retained before the next publication. Qualification and
speedup claims stay False.

Validation is source-only and synthetic: exact original consumer roundtrip,
successful/failed actual tool checks, all prerequisite scientific gates,
candidate intersections in every current full map, source-submap preservation,
explicit writer exclusion, all-UID scan denial and appearance of another writer,
full mocked run and finalization sequencing, post-scan handoff change, exact
journal limits and the sole public frozen deletion-call boundary. Real History,
locks, processes, raw files and deletion calls are replaced by injected fixtures;
the audit hook rejects any accidental actual access.
