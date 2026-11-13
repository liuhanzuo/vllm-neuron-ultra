# Geometry independent auditor freshness successor v4

This is a source-only, standalone future auditor, `qwen35_writer_geometry_audit_v4.py`, using the frozen geometry runtime v3. It does not modify or replace the existing running auditor, and does not run History, acquire a real lock, scan the real handoff directory, inspect native payloads, or choose a new catalog/binding. Frozen geometry v1/v2/v3 files remain unchanged.

## Static cost finding

The frozen v2/v3 auditor's nested `guard` calls `latest_handoff()` on every invocation. The accepted v2/v3 outer runtime checks the same full discovery only when at least 0.5 seconds have elapsed. The original latest-discovery implementation enumerates every handoff candidate, validates and obtains its identity, sorts the complete set, rejects ambiguous newest timestamps, and returns the newest path and identity; its bound is 4096 candidates.

The complete source verifier calls its guard for every text/Python-byte source. History reconstruction and physical checks add more guard calls. Consequently, the auditor repeats full discovery once per guard: for G guard calls and H handoff candidates, identity work is proportional to G times H, with an additional complete sort for each call. This static cost mechanism is demonstrated by the source bodies; it does not establish what fraction of live audit 9791's elapsed time it caused. No timing speedup is claimed. Parent-reported CPU/read progress is contextual observation only; this task does not inspect or stop that process.

## Minimal change and timing semantics

`FreshnessGuard` calls the original `lock.check()` on every invocation. It deep-copies the already authenticated running ACK's complete path-and-identity observation. Complete latest discovery runs on the first guard call, on forced calls, and whenever the current guard call starts at least 0.5 seconds after the previous complete scan started. Each full discovery must equal the original ACK observation; a same-path replacement is rejected as well as a different latest path. The original auditor compared only the path in its recurring guard.

The timestamp records the scan start, so time spent scanning cannot postpone the next due check. Nonfinite or backwards clocks fail closed. The cadence is checked at guard boundaries, matching the existing outer runtime. A blocking source or physical operation cannot promise a wall-clock callback every 0.5 seconds; the next guard call performs the due check. No timer thread or metadata cache is introduced. This is not a hard real-time scheduling guarantee.

Full checks are forced before and after History reconstruction, after retained report collection, at both boundaries of each of the two explicit post-exit drain passes, before the success/report validation branch, before and after the complete inherited History physical pair, after final identity/source checks, immediately before and after each result publication, and before successful return. The original calls inside inherited History, source verification, report collection and physical work still execute the guard and lock check; their complete latest discovery uses the 0.5 second cadence. Original metadata identity validation and full source hashing are unchanged.

The original complete History, 874 original children plus later serial scope, every source in the full approximately 48k source map, all metadata identities, complete two-pass physical collectors, explicit two current drains, report validation, actual-exit reconciliation, failed-state retention, source checks, original retained lock and final release remain intact. The final complete-source verification still uses the accepted exact-envelope dispatcher. No source list, historical record, report, owner, unit, scope or check is dropped.

Every original non-guard statement in the audit function is AST-identical after removing only the old guard definition, the new guard construction and guard invocations. `saved_binding` adds only the already validated ACK observation to its return dictionary. `latest_handoff`, `observe_layouts`, `predicted`, and `main` function ASTs are identical to frozen v3. Runtime, History, Authority and RetainedLock are the same imported objects. Output schemas and the original audit publication cap remain unchanged.

If a change is detected after the actual-exit file has been published, that file remains as evidence, the auditor fails, and no qualification is published. A later independent consumer still requires the actual successful auditor exit. This change does not roll back or silently replace evidence.

## Validation and delivery

Actual source/mock test session 37445, terminal c9b0d7, exit 0: 46 tests passed in 9.832 seconds. This includes 16 new freshness/boundary/preservation controls and all 30 inherited geometry controls. The six original auditor tests use the same mock fixture with its running ACK observation added; their substantive assertions are retained. Native imports, including ctypes, are blocked.

Clock tests cover first call at zero, the exact 0.5 second boundary, forced checks within an interval, unchanged lock-check frequency, changed latest path, same-path identity replacement, clock reversal/nonfinite values, deep-copy binding, and slow full-scan start-time accounting. A synthetic 48627-call schedule over 4.8626 clock seconds performs all 48627 lock checks and ten full discoveries. Those are mocked operation counts, not measured audit or inference speed.

Integration tests record complete history/report/source calls and both explicit physical passes, force checks at a constant clock, reject a change at a physical boundary, block all writes if a change appears before publication, and block qualification if a change appears immediately after exit publication. Actual failed model exits remain failed. AST checks preserve all original non-guard audit work, including original physical-pair and final identity/source calls.

The additive helper-compatible delivery includes the complete previous 654 Python source pins plus the new auditor and its test. It retains the original geometry scientific delivery and prior scanner/History successors. Future use still needs root source acceptance and a fresh explicit concrete binding naming runtime v3 and auditor v4 with the complete then-current lineage. This delivery selects no current/future state and authorizes no launch.
