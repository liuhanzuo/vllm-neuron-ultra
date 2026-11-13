Mini retirement recovery v3: separate periodic full checks from forced holder scans

This is a SOURCE ONLY successor to frozen recovery v2. The three new runtime
sources use v3 names. Existing v2 sources, tests, deliveries, reviews and runtime
outputs are not modified. No actual lifecycle, History, proc, service, lock,
model, compiler or raw payload is inspected by this delivery. No actual recovery
request, catalog, idle or acceptance is created, and no recovery is launched.

The concrete regression is the coupling of two meanings of force. Frozen v2
history_guard calls guard(True) whenever its 0.5-second global cycle is due.
full_guard uses that same force both for complete old-control byte hashing and
for the holder condition (force or now-last_scan >= 10). Once the permit exists,
every periodic global cycle therefore launches the privileged two-pass holder
scan, bypassing its original ten-second cadence. If a full check takes at least
0.5 seconds, start-based timing makes the next row callback immediately due.
The two required 2352-row no_unlink_state passes have 2353 callbacks apiece, so
this can cause a holder invocation on each of 4706 callbacks, plus boundaries.
This is a source-level possibility, not a measured attribution of elapsed time.

The original caller_v2.finalize success branch passes guard without force to
transport.audit_retired_state. Its two keeper/postcondition loops and Replay
callbacks use guard(), retaining the ten-second holder condition between forced
boundaries. The original failed branch has no equivalent selected-keeper audit;
recovery's two complete no-unlink metadata passes remain necessary. They must
not be removed as a performance workaround.

The minimal v3 correction is:

    full_guard(force=False, *, force_holders=None)
    guard(force=False, *, force_holders=None)

An omitted force_holders inherits force. The periodic history_guard calls:

    guard(True, force_holders=force)

Its ordinary callback force is False, so that due cycle still performs ALL full
source/global/old-snapshot byte checks while the holder scan uses its unchanged
ten-second due condition. An explicit history_guard(True) forces both. All
existing guard(True) calls at stage boundaries and before actual publications
also force both, because their force_holders argument remains omitted.

The holder condition is now:

    permit_ref is not None and (
        force_holders or time.monotonic() - last_scan[0] >= 10)

last_scan still records completion of a successful holder invocation. It is
not advanced on a skipped or failed scan. The full global 0.5-second cycle still
measures from the START of the preceding full check. Its first-callback, forced
callback, exceptional-exit, and every-callback fast checks are unchanged. No
clock is reset to completion to hide slow work. The change does not replace
guard(True) with guard(False), which would also reduce old-control hashing.

The frozen helper inspector and scope behavior are preserved byte-for-byte
after namespace substitutions. Each invoked holder check still has both real
all-UID scans and unchanged argv/fd/maps checks, permit/nonce/ancestry/descriptor
authentication, foreground wait and complete scope drainage. Neither a timeout
nor an unknown helper scope authorizes lock release or signaling other work.

All other caller predicates and function bodies are AST-identical to v2 after
reversing only the three namespace changes and the explicit force_holders
signature/forwarding/condition change. The original 7200-second work deadline,
lock, writer exclusion, floors/reservations, full source maps, full old-control
snapshot, real two-pass physical observations, both complete 2352-pair metadata
passes, source attribution, and receipt/catalog/idle order remain unchanged.

The original remover is tool session 4693. That number is NOT an execution-source
count. Its entire original execution map must stay unchanged; its actual count
comes from the authenticated original spec. V3 sources remain separate post-exit
audit provenance and must never be attributed to that old remover row.

Validation is six new focused synthetic regressions; the previously passed
25-test suite is not repeated. The real v3 recover entry runs with injected
external boundaries and the original source-extracted serial consumer. Tests
exercise 4706 selected callbacks and 14112 identities; 0.7-second simulated
holder scans; 0.6-second simulated slow global checks; explicit force inside
both selected passes; stage and every-publication holder fences; and periodic
holder failure followed by forced exceptional exit and final drainage without
idle publication. Snapshot checks inside periodic phases remain full hashes.
The tests verify the minimal AST delta and helper/scope namespace-only changes.

Actual author test: tool session 57730, terminal 895d6b, exit 0; six tests passed
in 4.976 seconds. The audit was installed before local test imports and allowed
only Python source/bytecode reads and importlib source-directory resolution.
Old and new implementation/test hashes and stat identities were identical
before and after the run. No real runtime actions were attempted. The shell
alone created the new synthetic test log. The default v3 test entry selects only
HolderCadenceRegression; inherited fixture/test definitions remain available
but the old 25 tests are not selected by that entry.

These tests do not establish actual recovery speed or completion within 7200
seconds. A slow global check may still run on every callback; that no longer
automatically forces a holder scan before ten seconds since its last completion.
Actual full checks taking ten seconds or longer can legitimately make a holder
scan due on each subsequent callback. Physical drainage can retain ownership
past the work deadline if a scope remains unresolved, as before.

The parent owns the current v2 tool's termination/drain observation and any
future source acceptance, fresh request/binding and launch. This minimal source
repair does not broaden the original supported zero-journal/prepublication
branch or auto-resume an existing output namespace. Any prior recovery actual
exit and partial outputs must remain separately preserved and explicitly
reviewed before root selects a fresh successor. It cannot be accepted merely
from the source tests or from the preceding recovery's partial stage progress.
