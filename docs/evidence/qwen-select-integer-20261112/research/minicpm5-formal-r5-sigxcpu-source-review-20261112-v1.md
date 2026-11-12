# Formal r5 SIGXCPU: source-only boundary and minimum optimization review

This is a source review, not a runner successor or new admission. It reads only
the two supplied saved metadata files and relevant Python source. It does not
execute History, native authentication, SDK import, ptrace, any child, a lock,
current catalog selection, or model/capture/native payload reads. No frozen
source was changed.

## What the evidence locates

Root supplied the actual terminal: formal r5 session 58506, chunk `d8b842`,
external exit 152; launcher.wait observed outer PID 948519 return -24, SIGXCPU.
The original limits remain CPU soft 600/hard 605 seconds and wall 900 seconds.
This review does not independently repeat root's exit/physical audit.

The saved files are:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `minicpm5-fixed-input-pv-sdk-import-formal-r5-20261112/admission.json` | 44,438 | `8e5b1b0959f3270e831963637c3bfd0e2e794593cdc1a3a53ef2dac0839677f0` |
| `minicpm5-fixed-input-pv-sdk-import-formal-r5-20261112/phase-1-owner.json` | 990 | `95ae403eaecef64ab7eb47c86c241805f99ae7db053ea9bb4bc37a246339d1cb` |

The owner names actual SDK PID 960133, parent 948519, the complete original
SDK310 literal argv, admission hash, and inherited absolute deadline. Its
presence on the frozen source path establishes actual child spawn and parent
exec/argv acknowledgement. `source_adapter_v3.start` writes this file at line
55, after complete prerequisites and before constructing the whole source gate
at lines 56–62. The admission witness records 48,223 complete source-map rows;
the inline SDK bootstrap closure has 172 rows. These are saved counts, not a
newly selected current source/catalog domain.

The last durable point therefore precedes all of the following possible work:

1. Whole release-packet serialization and the final child identity check.
2. `observed_adapter_v5.finish` construction of the observer, backend native
   revalidation, and explicit wait handoff.
3. `observer_logic_v1.run` authority/budget checks, SEIZE, interrupt, wait events,
   and gate pumping.
4. Potentially, a released worker's separate SDK authentication and import.

No phase-exit, observer-exit or trace artifact distinguishes those possibilities.
Trace records live in memory until `observer_logic_v1.py:476` writes them after
drain. Gate pumping occurs earlier at lines 195–207 and 464. Thus **gate release
is possible but unproved**; neither SDK import success nor “SDK never started”
is supportable. The exact instruction/function active at SIGXCPU is unknown.

The reviewed entry/descriptor installs SIGALRM handling, not SIGXCPU handling.
Default SIGXCPU termination need not run Python `finally` blocks, explaining
why a parent exit receipt can be absent without locating the interrupted step.
Root's independent complete physical audit remains the source for drain/lock
facts. SIGXCPU concerns the outer process's CPU accounting; it does not identify
the child import outcome or establish a particular expensive function.

## Repeated work actually present in source

| Source location | Actual repeated computation | Boundary that must remain |
| --- | --- | --- |
| `nocache_entry_v6.py:271–304,359–370` | `read_source` hashes the complete returned bytes, then `SourceLoader.get_code` hashes the identical immutable bytes again for its receipt. | Complete first SHA, EOF, before/open/after identity, exact source pin, no-cache loader and checks. |
| `formal_literal_codec_v1.py:47,159–196` | Every encode/decode hashes the unchanged 37,178-byte literal; encode hashes the expanded object, decode hashes wire plus the same expanded object. | Exact literal, canonical whole JSON, wire/expanded SHA, fixed positions and lossless roundtrip. |
| `observer_backend_v3.py:29–69`, `nocache_launch_v3.py:13–27` | Prospective admission is encoded/hashed at initial reservation, reservation verification and pre-spawn; argv construction repeats admission encoding. Writing/readback repeats representation checks. | Fresh durable file readback/identity and exact admission/argv binding before spawn. |
| `formal_literal_codec_v1.py:223–260`, `formal_literal_result_v1.py:33–52` | Runtime and stdout reservations independently decode the same admission and reconstruct the same maximum-size prospective terminal. | Both full budget calculations, conservative scalar widths, six-name domain and original caps. |
| `fixed_input_pv_catalog_auth_v2.py:83–116` | Whole descriptor-plan digest is recomputed for each of six report checks and later expected/witness fields; complete source maps are also encoded/digested repeatedly. | All six complete report comparisons, descriptor equality, complete source bindings and saved 1,284-capture scientific proof. |
| `operator_sdk_sources_v1.py:148–160`, `native_process_adapter_v7.py:124–129` | For ordinary pinned text, `text(path, expected)` already verifies SHA; its immutable returned bytes are hashed again by the caller. | Original text transport, alias semantics, complete read/EOF/identity and exact expected SHA. |

The expensive domain is larger than these local duplicates:

- `NativeAuthenticator.authenticate_bootstrap` at lines 215–225 hashes every
  `.py` row in the whole map before historical imports. The source loader then
  rereads/hashes each actually imported file. Original `reconstruct` later calls
  `verify_text_sources` for the complete classified text domain.
- `StableReader.read` invokes its guard before the read, on every read iteration
  including EOF, and after the read. The bootstrap guard performs actual source
  fence/resource/owner/descriptor checks. After adoption, `ContinuedPVGuard`
  calls that full bootstrap guard before and after the original PV guard, whose
  lease also proves the descriptor. This amplifies work across a large domain.
- After the owner is saved, `backend_v4.make_ports` calls
  `NativeAuthenticator.authenticate`: this performs full source/native
  **identity scans**, then mapping validation. `verify_loaded_native` repeats a
  complete source-identity scan. It does **not** redo the full source-content
  hash pass or historical reconstruction at this point. `StableReader.unchanged`
  checks canonical ancestry and identity for every row.
- `Metadata.read` has an identity registry but no decoded-object cache. Repeated
  authenticated references are reread, hashed and JSON-decoded. Historical and
  current catalogs can share prefixes, but their bindings are different and
  cannot simply be treated as one catalog.

The full `prior.reconstruct` call appears once in the actual outer authentication
path (`catalog_auth_v2.py:60`). The second call site in the old prerequisite
module is not also called by this successor. Removing an alleged second whole
reconstruction would be based on a false call count. Independent old/current
source-plan and catalog comparisons remain meaningful validation.

## Bounded pure-code check

Tool chunk `88de3a`, exit 0: AST-selected frozen codec/result definitions were
evaluated with a counting stdlib SHA256 oracle and only the saved admission,
owner and literal source. Both complete saved files round-tripped byte-for-byte.
No original module import, runtime replay, guard/lock/native action or timing
measurement was performed.

| Pure operation | Digest calls | Bytes submitted to the frozen pure-digest interface |
| --- | ---: | ---: |
| Encode saved expanded admission | 2 | 156,759 |
| Decode saved wire admission | 3 | 201,197 |
| Runtime reservation | 10 | 550,055 |
| Stdout reservation | 8 | 361,898 |

These counts prove deterministic duplicate computation. They do not measure
elapsed/CPU seconds or prove that hashing caused all 600 seconds.

## Smallest reviewable successor proposal

Start with local reuse of already authenticated **immutable bytes and pure
results**, without a new historical executor or general filesystem cache:

1. Return/reuse the digest established by `Entry.read_source` in the immediate
   loader receipt. Keep the first complete hash and all current file checks.
   Preserve source/deadline guard checkpoints when removing the second digest.
2. Prepare the exact literal/admission representation once per owned preparation:
   full expanded bytes, full wire bytes and both digests. Use those same bytes
   for argv binding and both reservation calculations. Build the conservative
   prospective terminal once and feed it to both budget calculators. Continue
   to freshly read/hash/check the actual persisted admission and compare all
   bytes before spawn. Changes to admission, literal, argv, source pin or budget
   inputs must invalidate reuse. Do not cache authority or physical observations.
3. Compute the full descriptor-plan digest once for the immediately consecutive
   six-report validation and expected/witness construction, while keeping all
   six full comparisons. Similarly avoid hashing immutable bytes twice inside
   a single authenticated text-read operation. Preserve exact alias/non-UTF8
   transport and the complete classification checks.

These edits need new versioned files and independent root review; this analysis
does not implement or authorize a launch. They retain the original CPU600/605,
wall900, CPU affinity, memory/output limits, source domain, nine-native checks,
and the 1,284-capture scientific gate. The import-only diagnostic still reads no
capture/model tensors and does not substitute an import result for science.

Do not initially remove a complete source pass, collapse catalog versions, add
an unbounded memo table, throttle descriptor checks, or reuse a parent's native
receipt in SDK310. Sharing an authenticated `.py` verification ledger across
the early and original text passes could save more work, but it requires an
explicit narrow callback contract, complete set/SHA/identity equality, fresh
boundary checks, and alias handling. That is a broader follow-up, not a justified
shortcut in the smallest successor. Whole live identity scans at separate
native-use boundaries are not duplicate byte hashing and must remain.

Before any real retry, verify the proposed pure changes with saved-file golden
roundtrips; mutations to every referenced input; byte/hash/identity mismatch
negative controls; exact reservation equality; unchanged native/scientific gate
call counts; and bounded instrumentation of digest calls/bytes. For an eventual
root-authorized finite retry, small bounded stderr phase markers containing
process CPU/wall deltas and counters would locate pre-auth, pre-spawn,
backend-revalidation and gate-EOF boundaries without replacing receipts or
expanding output caps. No source-only analysis can promise that the unchanged
600-second CPU limit will suffice; that remains to be measured.

## Frozen entry source identities reviewed

| Source | SHA256 |
| --- | --- |
| `minicpm5_sdk_mapping_outer_source_entry_v9.py` | `c498fadfb42957b9e8ae66f461be4137c5546b27046de7b1ab7f4c9843624c57` |
| `minicpm5_sdk_mapping_nocache_entry_v6.py` | `bbf5affece28e3d6356ce27d48ccdb7f4d707fbf6717808a69ca1f61222763d0` |
| `minicpm5_sdk_mapping_source_adapter_v3.py` | `245f15f52cbfbd36a345d0ac16ed6000118026d055eff5fb9dc17f7ff792506f` |
| `minicpm5_sdk_mapping_observed_adapter_v5.py` | `561688b8012a97867f990ebd8f43ae29551fe0d37450bcbf75e7fdddb5ead378` |
| `minicpm5_sdk_mapping_observer_backend_v4.py` | `e944fe7cb35194fe052934d337c5d5d48e94aa40820742af68939010b7db26ef` |
| `minicpm5_sdk_mapping_observer_logic_v2.py` | `c7672fa51553edd3f46cc90049ee2c0f2c46c76e816fd9869a97d596f32fcbf8` |
