# Geometry independent auditor v5: namespace and physical boundaries

This source-only successor corrects two root-review findings in rejected v4. It preserves every sealed v1/v2/v3/v4 file and does not change or interrupt audit 9791. No real History, lock, native work, physical observation, handoff-directory scan, experiment preparation or queue action is executed by this delivery.

## Correct result namespace

The standalone v4 file lives directly inside `research`, but copied the package auditor's `Path(__file__).resolve().parents[1]`. That computes the experiment root for a standalone file. V5 uses `.parent`, restoring the original research-directory namespace for both actual-exit and qualification publications. V4 was rejected before actual execution; its source and rejection record remain preserved.

The new test checks the actual imported v5 source location, compares `audit.Q` with the runtime's research path, and executes the complete audit publication flow with mocked IO while retaining the real `audit.Q`. Both returned publication paths must be in `research`, not its parent. No result is written to those paths during this test.

## Force each complete physical pass

V4 forced full freshness before and after the inherited physical pair as a whole, but each internal pass had ordinary 0.5-second cadence guards. That is insufficient for a strict requirement to force full checks at every pass boundary. V5 adds `qwen35_geometry_audit_physical_v5.py`, an audit-only adapter; no original module is patched.

The adapter's `_collect_pair` preserves the complete original `minicpm5_sdk_physical_evidence_v3.collect_pair` body. Only the guard at each pass's start and the guard after that pass's supplements and validation gain `force=True`. Both full trees, all additional owners, every original 1308 service and exact later unit, all missing-field supplements, both complete observations, equality checks, original lock checks and bounds remain. Mid-pass guard calls still check the lock every time and perform complete freshness when due at 0.5-second guard boundaries.

The adapter's outer collector keeps `authority.disjoint`, the original current full physical function and the same properties/group callbacks. Its History subclass inherits the original constructor, complete `resolve`, metadata identity checks, freshness and scope. The overridden `physical_pair` method changes only its collector import; its original complete pair validation, source-bound scope, retirement supplement, packed physical evidence, original failed-state retention and complete result remain AST-identical. The runtime still uses its original History v3; this subclass is selected only by the future independent v5 auditor.

The original 874 children plus all later history are retained, separately from the 1308-service physical collector's original unit set. No source, child, owner, service, metadata identity, report or check is omitted.

V5 otherwise preserves v4's freshness guard and audit body: every guard checks the original lock; full discovery compares the authenticated running ACK's complete path and identity; first, due and forced checks use scan start times; clocks must remain finite and monotonic. Full discovery is also forced at the existing audit/report/source/publication boundaries. The 0.5-second cadence applies at guard calls; it is not a hard wall-clock scheduling guarantee during a blocking operation.

## Source binding and validation

The helper-compatible delivery uses schema `qwen35_writer_geometry_binding_source_delivery_v1` and extends the accepted v3 scanner delivery's 654 Python pins with exactly three source files: the v5 standalone auditor, physical adapter and test. All prior pins remain unchanged. The rejected v4 delivery and root rejection are evidence references, not the active auditor selection.

The existing `prepare_qwen35_writer_geometry_binding_v1` can merge this map through its accepted `preparer_delivery`/`preparer_source_review` inputs. A future concrete spec must include the v5 standalone and adapter pins before startup, so `stage.load_spec`/`loaded_local_sources` can authenticate both `__main__` and the imported adapter immediately. The delivery records exact paths/hashes and verifies direct local-import coverage. It creates no concrete spec or current catalog selection. Future root review must explicitly select runtime v3 and standalone auditor v5; until then the original v3 auditor remains available as a separately selectable alternative. V4 must not be selected.

Actual source/mock tests: session 40587, terminal 3b648e, exit 0; 51 tests passed in 9.908 seconds. This includes the previous 46 controls, one actual-source publication-namespace test, and four pass-boundary/wiring/preservation tests. At a constant clock, the two full inherited passes produce exactly four forced latest checks: before/after pass 1 and before/after pass 2. Injecting a changed latest observation at any boundary fails before subsequent work. Original callbacks, disjointness and full pair-validation arguments are checked with fake data. AST comparisons show the pair body changes only at those two guard sites and the History method only at its collector import.

Initial test session 32648, terminal 81fa96, exit 1 attempted to import the historical current-physical module in a wiring test. Its transitive ctypes import was rejected before native loading. That test now supplies a fake module and callback namespace. The strict native-import guard remains active; the production adapter retains the same original physical import lazily for a future authorized audit. No actual History or native collector ran.

Independent root source acceptance, fresh complete concrete binding, actual source/identity/physical audit and actual terminal observation remain future work. No audit or inference speedup is claimed.
