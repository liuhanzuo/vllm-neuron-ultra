# MiniCPM5 failed retirement finalizer: source-only recovery v2 with stage telemetry

This delivery is source and synthetic-test evidence only. It does not accept a recovery binding, run History, inspect current lifecycle state, operate the original lock, start a helper, read raw/model/weight payloads, publish an idle handoff or qualify a model. The parent owns the actual failed-finalizer observation, request/review, execution, terminal capture and independent acceptance. No runtime request or acceptance review is created by this delivery.

## Corrections to the preserved design v1

The design `minicpm5-retirement-finalizer-recovery-source-design-20261113-v1.md` remains unchanged. Two of its proposed details are superseded here:

1. **Audit sources never become original-remover execution sources.** The historical remover's exact tool, owner, units and source delta remain unchanged. The three new pins appear under `post_exit_audit_source_sha256` in recovery provenance, its separately reviewed source union and idle audit metadata. A future preparer must enroll that audit closure explicitly and separately. It must not infer that the old remover executed these new files.
2. **The read-only global-check cadence covers more than resolve.** The initial resolve-only draft was insufficient to cover all relevant source paths. The delivered source explicitly cycles the external global guard during resolve, complete physical collection, both selected-file metadata passes and final composition/consumer validation. This changes observation timing; it is not equivalent to checking every global predicate on every callback.

The parent reported actual finalizer terminal `26060 / 64c33a / exit 1`, with the 7,200-second check raising inside `History.physical_pair` through a historical metadata callback. That proves where the total deadline was reached. There are no phase-duration measurements: it does not prove the entire two hours was spent in physical collection, nor that the separately prepared v4 resolve-only caller must fail. This child did not independently read the actual observation or failed-state files.

## Exact callback path and bounded timing change

The relevant original source path is:

`History.physical_pair` → current full physical collector → `minicpm5_native_r12_completion_physical_v1.physical` → `guard = authority.check` → `nemotron9b_r12_completion_v1.metadata(read, guard)` → `nemotron9b_r12_r11_metadata_v1.metadata_completed` → `minicpm5_after_nemotron_package_v2.Reader.path` / `Reader.__call__` → `Authority.check` → the caller's external global guard.

`Reader.path` invokes that guard once and then `Reader.__call__` invokes it twice more around its original complete JSON authentication. Once the History object exists, the old global guard calls `history.unchanged()`, whose metadata implementation visits every previously recorded identity. Thus N calls can repeat a complete M-file identity inventory O(N×M), in addition to the actual reads and checks. This is a source-level work count, not a measured speedup.

Changing only `history.meta.guard` would not change the r12 `Authority.check` path. The delivered caller instead passes the same bounded guard to the unchanged History constructor and unchanged Authority constructor. It does not alter or monkeypatch their class/function bodies.

Each callback still checks the original retained lock, stop flag, original total 7,200-second deadline, and that both original remover and failed finalizer owners are gone. A complete global check runs on the first callback, explicit force, and first callback at least 0.5 seconds after the **start** of the preceding complete check. A slow complete check cannot defer its next due boundary by resetting the clock on completion. This is callback-based observation, not a background timer or a hard 0.5-second wall-clock guarantee during blocking operations. A transient global mutation between observations may be observed later or missed if fully restored; the original writer exclusion and retained lock remain required. No previous source hash or metadata verdict is substituted for a required real read.

The explicit read-only stages are:

- `resolve`: the original full source/metadata resolution;
- `physical`: the original entire two-pass proc/service/cgroup and owner observations;
- `selected_metadata_first` and `selected_metadata_second`: all 2,352 original/keeper identity pairs and original allocated bytes on both passes;
- `composition_consumer`: original overlay read, unchanged original composer, and complete unchanged original `serial_catalog` validation of the virtual new chain.

Every stage has complete checks on entry and exit, including exceptional exit. Holder scans and output publication retain full checks. Global checks also recheck the fast predicates after their work. The original physical body still performs every direct process/service/cgroup observation; Authority still performs its original `check`, `disjoint`, service and `active_heavy` operations. Selected-file metadata still performs every original/keeper identity read, the original block count comparison, and a repeated original identity read across that observation. Neither direct physical observations nor per-file predicates are skipped.

The consumer stage matters independently: before this correction, `rebind` supplied `lambda: guard(True)` to each original catalog callback, which forced an all-UID holder scan as well as full source/history checks on every callback. It now receives the explicit read-only guard; the unchanged consumer still checks its entire chain. The callback leaves that stage before each actual publication, which always receives a full check.

There is no runtime performance result and no assurance the recovery will finish within the unchanged deadline. Actual source review and execution must establish that.

## Three new runtime sources, unchanged historical inputs

The separately reviewed source supplement is exactly:

- `research/minicpm5_retained_retirement_recovery_v2.py`
- `research/minicpm5_retained_retirement_recovery_holders_v2.py`
- `research/minicpm5_retained_retirement_recovery_scope_v2.py`

Old `caller_v2.load`, `prerequisites`, `source_check`, `storage`, `Saved` and `contract_v2.completed_outputs` execute unchanged. The new scope source is mechanically identical to the old scope except the exact helper command target. The helper's entire `inspect` AST is identical to the old helper. Its permit validator requires the exact new recovery owner argv; it cannot authenticate an old run/deletion argv as recovery.

Only the temporary History input copy receives old pins plus the new three-file union. The returned resolved historical map must still exactly equal the original spec map. The original spec, reviews, running handoff and original control files never change.

## Supported branch and authenticated inputs

The input request has exactly `original_request`, `interruption`, `recovery_review`, `updated_at`. All references are explicit complete path/hash pairs; only the existing bounded metadata fields retain their original three-field path/hash/bytes form. The caller never selects current references automatically.

The original finalizer request is authenticated through unchanged v2 predicates and contains its exact spec, review, remover terminal, actual remover tool and timestamp. The observation must bind a separately observed **failed** original finalizer tool, distinct from the failed remover tool; its exact owner argv; original source; original lock identity; current original running hash and identity; complete old control namespace; complete finalizer transcript hash/identity; and the existing original finalizer-helper-drain.

The retained original terminal must have `failed=True`, `transaction_result=None`, exact spec/review/running/lock bindings and its original complete helper-scope drain. The finalizer drain must independently validate against the failed finalizer owner.

This implementation intentionally supports only **entirely prepublication, zero-journal** failure:

- old actual-exit, catalog, idle and all their `.publishing` paths are absent;
- the old journal and its `.publishing` path are absent;
- old control may contain partial History/permit/drain files, but its complete bounded directory/file name set, identities and hashes are captured;
- all 2,352 originals and keepers remain present with the original exact identities, and original allocated blocks equal the original verification on two complete passes.

Any old publication, journal, missing/replaced original, changed allocation, old snapshot mutation or unresolved owner/helper rejects this narrow branch. A different recovery must be designed if any such condition occurs. No raw bytes are reread. The supported successful audit records `reclaimed_bytes=0`, never resumes removal, and never changes the original failed retirement into success.

The independent `/root` review must have the exact schema and fields accepted by `validate_review`, including exactly three new pins, the original request/observation/spec/review references, exact fresh paths, `launch_authorized=True`, `deletion_authorized=False`, `model_or_raw_replay_authorized=False`, and explicit `original_writer_exclusion_retained=True`. The old exclusion is separately checked unchanged. A review produced by this child is not accepted; none is created.

## Fresh namespace, resources and process ownership

The bounded recovery name produces a separate root:

`R/minicpm5-retained-retirement-finalization-recovery-<name>`

It contains a fresh History directory, permit, helper drain and actual-exit receipt. Catalog and idle use separate exact fresh names under research. Every destination and `.publishing` destination must initially be absent. The old control tree remains untouched, including directory identities.

Resources retain CPU 7, nice 19, 6 GiB address-space limit, 32 MiB file limit, the original 7,200-second total finalizer deadline, original root/NVMe floors and original control/journal reservations. The recovery adds a separately counted **128 MiB** root output reservation. The exact free-space test includes the remaining old control and journal budgets plus the remaining new recovery budget. The new root allows only itself and its one History subdirectory, at most 128 files including separately published catalog/idle, and bounded enumeration before unbounded entries could accumulate. It does not borrow the old control or journal budget.

The lock remains device 66306 / inode 8294421. The new recovery owner is distinct from the old remover and old finalizer. Exact invocation is the original configured Python, `-B -u`, the new recovery source, `--request <path> --request-sha256 <digest>`. The parent supplies the actual request and observes the actual tool exit; the caller reports that its own exit is still pending.

All-UID scans keep the original two-pass argv/fd/maps checks and exact privileged ancestry/nonce/descriptor binding. Scope/sudo/helper drainage uses the original foreground behavior. Failure is not permission to signal unrelated processes or release an unresolved helper scope. The recovery has no deletion, transaction-retire, model, compiler or raw-replay call.

## Publication and future acceptance

Original `completed_outputs` constructs its original three objects in memory. Pure `rebind` keeps all prior rows unchanged, preserves the failed remover's original source delta/tool/owners/units, adds separately named audit provenance to receipt/idle and recomputes the fresh reference chain. The unchanged original consumer checks the entire virtual chain against old pins plus the separately reviewed audit supplement.

Actual writes occur receipt → catalog → idle, with full prepublication checks and exact post-write saved bytes/identities; idle is last. Immediately after idle publication the expected tip changes to that exact newly saved idle identity. Final drainage is written to the new root and the original lock is released only after real helper drainage.

The parent must subsequently observe recovery's actual terminal and independently verify the new three-object chain, unchanged old snapshot, preserved original failure, zero reclaimed bytes, complete real History/physical evidence, current tip and actual lock/owner/helper drainage. A partial new publication or nonzero recovery exit is not silently accepted. This source delivery does not claim a new catalog length or publish a model speedup.

## Synthetic verification and retained development failures

The test suite executes the real new `recover()` body using injected filesystem/process/lock/History observations, real old `Saved/load/prerequisites/completed_outputs`, real Authority and the unchanged original catalog consumer functions loaded from source. It checks original-source attribution, both complete selected-file passes, idle-last, fresh output chain, mutation/holder/resolve/physical/deadline/signal/write failures, unresolved helper retention, explicit cadence at every read-only stage, due/exit mutation detection, start-based slow-check timing, mandatory publication fences and additive storage reservations.

A separate source-extracted original r12 physical function plus original Reader class confirms the actual metadata path enters original Authority's external guard. Exact source/AST comparisons preserve the scope command-only change and entire helper scan body. A test audit rejects actual subprocess/network/namespace/flock calls, native imports, metadata/model/raw opens and file mutations; the test process only reads Python sources. Only the outer test tool writes its new synthetic test log.

Development failures are retained as failures, not counted as successful runs:

- Initial 13-test run: tool 68949, terminal chunk `364bce`, exit 1; one failure and nine errors. The fake stat incorrectly supplied `st_inode` instead of `st_ino`; a test invoked the real broad consumer import instead of its existing source-extracted test module, and the audit correctly blocked import-discovery `os.listdir`. Both were fixture problems; no native/model/state work occurred.
- The fixed 13-test run: tool 81865, terminal `549b19`, exit 0.
- Full-entry 16-test run: tool 37196, terminal `457184`, exit 0. This was an earlier draft, not final-source acceptance.
- Explicit-stage 19-test run: tool 15123, terminal `255b1f`, exit 0. The later consumer-path correction was not yet included.
- Consumer-path 20-test run: tool 74003, terminal `e89f48`, exit 1. An arbitrary fixture assertion expected more than ten catalog callbacks; its synthetic catalog produced seven. The final assertion checks at least the actual complete catalog's row count and verifies real original-consumer callbacks plus forced fences before each publication. No production predicate was weakened.

The frozen v1 final suite passed 22 tests in 23.068 seconds: actual tool 13223 / terminal 20d3de / exit 0. Its source-delivery SHA is `8c7b4ce9b0cb5bf7dcb7745fdaab041aa4b3d0b85cb1287e11126fb3c5a1aff3`. All v1 files remain unchanged. The v2 final test result, full source hashes and actual terminal belong to the adjacent v2 source-delivery JSON and test log. Passing source tests does not constitute runtime acceptance.


## V2-only observational successor

After v1 was frozen, the parent requested stage timing before any real recovery, because the failed original finalizer did not identify phase durations. V2 therefore uses three entirely new source namespaces; it does not modify the frozen v1 sources, tests, notes, logs or delivery. V2 preserves the v1 recovery schemas and predicates. The future root review must bind exactly the v2 three-file supplement, and the actual owner/helper argv must target v2. V1 pins must not be substituted for v2 pins.

The only behavioral addition beyond namespace retargeting is bounded stdout telemetry. Each named stage emits one `mini_retirement_recovery_stage` enter and one exit event with stage name, success (`null` at entry, boolean at exit), exception class on failure, monotonic elapsed seconds since the retained-lock start, stage elapsed seconds, cumulative full-global-guard count, cumulative full-global-guard seconds and a count of telemetry write failures. Fast per-callback checks do not emit output and are not counted as full-global checks. A failed full guard is still counted and its elapsed time is accumulated through `finally`.

The eleven stages are `initial_fence`, `resolve`, `physical`, `history_snapshot`, `holder_permit`, `verification`, `selected_metadata_first`, `selected_metadata_second`, `composition_consumer`, `publication`, and `final_drain`. Their scopes surround the existing checks and work, including exceptional stage exit. Initial authentication before lock acquisition remains outside this elapsed clock, as it was outside the original retained finalizer deadline. The timer adds no sleeps and does not extend the original 7,200-second limit or alter the 0.5-second callback cadence.

Ordinary stdout `OSError`/`ValueError` during these observational events is counted and does not bypass validation, publication fencing or helper drainage. The next successfully emitted event reports that count. Telemetry loss is not fabricated timing evidence; an independent root must use the actual available log and actual terminal when deciding acceptance. No per-callback stdout, background process or telemetry file is created by the runtime code.

Additional v2 synthetic tests cover paired success-stage events and monotonic cumulative counters, positive accumulated guard time under the injected slow-check clock, failed physical-stage exit followed by real cleanup flow, and simulated broken telemetry stdout preserving original idle-last and drain/release behavior. The preceding v1 failure/mutation/cadence/consumer tests all remain in the v2 suite. Actual runtime timings and any claim about whether full checks exceed the cadence remain unknown until separately reviewed execution.
