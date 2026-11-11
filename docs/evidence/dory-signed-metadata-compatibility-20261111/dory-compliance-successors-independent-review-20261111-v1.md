# Independent source review: Dory compliance-parser wiring

Conclusion: no actionable blocker found in the scoped six-file successor integration. Source/mock acceptance only; no real metadata retry, cryptographic verification, network call, original data read, lock acquisition, process observation, model execution, or failed-outcome promotion was performed or authorized by this review.

## Reviewed scope and exact differences

Reviewed the actual files and `dory-ubuntu-metadata-compliance-integration-20261111-v2.patch`. Independent AST comparisons confirm the complete modules differ only as follows:

- Adapter v3 replaces the old `parse_status` definition with an import of the reviewed pure compliance helper. All transport, bounded input/output, stable byte/hash checks, process command/group/timeout/cleanup, key authority, and receipt code is identical to adapter v2.
- Input binding v2 retains every old source pin and adds exact adapter v3/helper pins. Its complete validation, input/trust binding, output accounting, receipts, and flags are unchanged.
- Worker v2 selects binding v2 and adapter v3.
- Parent/root v2 selects binding v2 and launches the explicitly pinned worker v2; all other code is unchanged.
- Saved audit v2 selects binding v2 and adapter v3.
- Exit recorder v2 selects binding v2 and saved audit v2.

The original source pins remain hash-correct. The imported chain is acyclic: adapter v3 → compliance helper → frozen adapter v2 → unchanged package-discovery source. Both worker/parent use the new binding, recorder uses the new saved auditor, and saved auditor uses the same adapter v3 helper callable as the actual verifier. Each new executable/module must still appear in the future concrete plan/review source map; the binding's four fixed pins do not replace complete import/source verification. Existing `verify_sources` and `verify_loaded` calls remain unchanged, including the worker check immediately after importing adapter v3.

## Actual verification and saved replay agree

`Verifier.__call__` still saves the entire original stdout/stderr and emits their exact references before parsing. Its new `parse_status` callable is the reviewed helper; the helper's original parser is frozen adapter v2. Only the optional exact `VERIFICATION_COMPLIANCE_MODE 23` annotation is omitted from an in-memory parse view, with the reviewed per-NEWSIG-group count and placement restrictions. Saved bytes, hashes, status references and receipt shape do not change.

`saved_audit_v2.replay_saved` calls the same adapter v3 parser on the fully authenticated saved status. The independent synthetic replay exercised all 19 events / 27 nested files, all three signature statuses containing the annotation, and original discovery/replay equality. It made no cryptographic subprocess call and did not alter saved status bytes.

The integration does not accept unknown flags/statuses, duplicate annotations, mismatched signer/key fingerprints, weak hashes, expired signatures/keys, revoked signatures, bad signatures or nonzero gpgv exit. All other status bytes still reach the frozen validator. The independent live-verifier and saved-replay mocks both rejected representative unknown, bad, expired-key, signer-mismatch, expired-signature, duplicate-annotation and unsupported-flag cases. Existing helper tests cover the remaining strict syntax, grouping and refusal cases.

## Original protection and failure behavior

Full-module equivalence establishes unchanged retained descriptor policy, parent/child ACK and original-fd noninheritance, historical owner/service checks, resource limits, root/NVMe floors and output reservation, complete allocated-tree accounting, subprocess group ownership, drain/release, exception persistence, and original external-exit/recorder logic. No resource value, trust authority, input pin, digest rule, source gate, success flag or exit handling was weakened.

The original failed run remains exit 1. The independent tests verified a nonzero verifier permits only its observed failure prefix, cannot continue into success replay, and that producer exit 1 stays failed even with success-shaped output. The recorder's event audit retained the actual four-event failing verifier prefix and did not claim stage success.

## Validation evidence and limits

`check_dory_compliance_successors_independent_20261111_v1.py`: 6 tests, actual tool call terminal chunk `182c25`, exit 0, unittest 0.151 seconds. Tests cover complete-module AST equivalence, actual import identities/all fixed hashes, the real Verifier class with memory-only transport/process mocks, complete three-signature saved replay, matching negative cases on both paths, and retained failure outcomes. The test process installs the prior source-only audit hook, rejecting network/process/lock operations and any non-module file read. Historical fixture import-time JSON reads are replaced by the existing synthetic source fixture.

`check_dory_ubuntu_metadata_integration_v2.py`: directly executed as requested, 7 tests, terminal chunk `cd1743`, exit 0, unittest 0.033 seconds. This confirms the actual worker dispatch block, binding, unchanged resource/process checks and pins. No broad directory discovery was used.

The earlier 8 helper tests remain a separate source/mock result. These results do not authenticate a new Release, admit a new real task, replace current-catalog/freshness checks, or qualify package bytes/native ABI/model behavior. Root owns the concrete successor source review, actual input/output references, runtime admission, external terminal observation and publication.
