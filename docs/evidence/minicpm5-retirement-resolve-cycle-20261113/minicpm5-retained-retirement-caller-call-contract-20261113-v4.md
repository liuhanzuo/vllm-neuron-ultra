# Retirement v4: bound repeated checks during read-only History resolution

This is a source-only successor for a future complete retirement binding. It
does not replace or interrupt an admitted caller/finalizer. No current lifecycle
JSON, candidate/keeper bytes or captured directory, process, lock or physical
History was accessed while producing it. Root retains the exclusive actual
retirement transaction and finalization.

## Concrete cause and scope of the change

The v2 and v3 `finalize.guard` bodies are identical. Every callback performs
`Saved.unchanged`, all caller/frozen source hashes, the complete control/journal
storage scan, exited-owner check, latest-handoff discovery, and, once resolved,
the complete historical metadata identity check. `run.guard` already limits
its corresponding full checks to a 0.5-second cycle.

History calls its guard before/after each metadata read, around inherited reads
again, and for every text/Python source pin. For K callbacks, J journal entries,
L candidate handoffs, M saved inputs and S caller source bytes, this multiplies
otherwise independent work by approximately K*(J+L+M+S). The full source pass
still has to read every source; repeatedly scanning the unrelated journal does
not authenticate an additional source. The journal cap is 4,706 files. At the
resolution stage `history` is still None until `.resolve()` returns, and the
finalizer permit has not been created; neither `history.unchanged` nor the
privileged holder scan causes that stage's multiplication.

An AST-only mock executed the real old guard bodies for 10,000 callbacks on a
synthetic five-second clock: old finalize performed 10,000 storage/source/latest
passes; run performed 10; a resolve-only cycle plus exit check performed 11.
For a hypothetical complete 4,706-file journal, those passes entail 47,060,000
versus 51,766 journal-file visits. These are operation counts under a synthetic
clock, **not measured runtime speedup or evidence about the live finalizer**.

The v4 implementation changes only the guard passed to `History.resolve`:

1. Every resolve callback checks the original retained lock, interruption flag,
   original 7,200-second deadline, and actual original remover owner absence.
2. Initial callback, explicit `force=True`, and the first callback starting at
   least 0.5 seconds after the preceding full-check start invoke the complete
   unchanged finalizer guard. Source hashes are recomputed normally; no hash,
   metadata identity, directory result or successful verdict is cached.
3. The interval starts before the full check. A slow scan cannot push the next
   deadline out by another interval. Lock/deadline/owner are rechecked when the
   full check returns.
4. Full checks occur before entering resolve and in `finally` after resolve,
   including failures. The resolved History is assigned only when resolve
   succeeds, as before. A failure exits through the original helper drain and
   lock-release path, and a new guard failure retains exception chaining.
5. After resolve, the retained History callback directly calls the original
   full guard for every invocation. Physical passes, blocking holder scans,
   keeper postcondition auditing, original downstream consumer and each write
   keep their existing callbacks and forced checks unchanged.

The observable tradeoff is explicit: persistent changes to global
source/metadata/storage/latest state during resolve can be detected at the next
due callback or the forced exit instead of at every callback. This is not
strict observation-by-observation equivalence. The cycle is evaluated at
callback boundaries, not by a background timer; a blocking operation may delay
that boundary. Every source/read still performs its original direct content and
identity checks, and no physical work or publication can proceed past the stage
without the forced full check. Owner/lock/deadline/interruption remain checked
at every callback. All limits and scientific predicates stay unchanged.

## Four necessary namespaces

`holders_v3.main` requires the exact owner command to name
`contract_v3.SOURCES[1]`, the old caller. A standalone finalizer cannot satisfy
that accepted helper protocol. Four mechanical namespaces therefore move
together:

- `minicpm5_retained_retirement_contract_v4.py`
- `minicpm5_retained_retirement_caller_v4.py`
- `minicpm5_retained_retirement_holders_v4.py`
- `minicpm5_retained_retirement_scope_v3.py`

Contract, holders and scope change only their mechanical module/spec names.
Caller changes those names plus the resolve guard adapter and forced entry/exit.
Its original `guard` body and all other functions match the predecessor AST.
The unchanged public remover is `minicpm5_retained_raw_transaction_v2.retire`;
`minicpm5_retained_raw_source_io_v1.py` and all six frozen science sources remain
unchanged. The exact historical shared-source exception is unchanged.

The spec schema is `minicpm5_retained_retirement_spec_v4`. Holder nonce,
helper/sudo/monitor ancestry, exact finalizer command, complete source hashes,
scope drain and lock retention are unchanged except for those names. Failure or
timeout never signals an unrelated process or invents a drainage proof.

## Future binding and invocation

The new delivery distinguishes:

- `mandatory_new_source_sha256`: exactly four newly authored source files.
- `current_contract_source_sha256`: all six `contract_v4.SOURCES`, including
  the existing transaction v2 and source reader v1.
- `mandatory_lineage_source_sha256`: those four plus all six prior v3 sources,
  ten distinct files with all predecessor bytes retained.
- `execution_source_sha256`: complete prior 670-file execution map plus the
  four new files, 674 pins. This preserves older v1/v2 helpers even where they
  are not imported by the latest entry.
- `static_source_closure`: 667 statically reachable Python files, including
  complete previous static closure; dynamic sources still belong to the full
  original seed map and are checked by the existing loaded-source verifier.

Root must merge the complete prior global/source/input/catalog lineage with
these pins before rendering the future History spec and retirement spec. Do not
replace the full global map with the execution submap. Preserve original 1,869
science pins and all existing text pins. The previous delivery's 2,453 expected
text pins become 2,457 if the same original seed is used; current exact counts
must come from the future authenticated builder. No current count is asserted
by this source-only task.

Use `contract_v4.render_spec` with the original exact nine reference roles,
the complete future History spec, original plan, exact six-source current
contract map and full 674-source execution map. Its review, global map,
execution submap, writer exclusion, fresh paths and actual tool ACK retain the
existing contract. This delivery creates no accepted review or runtime request.

Future exact run command:

```text
<c.base.PYTHON> -B -u <Q>/minicpm5_retained_retirement_caller_v4.py --mode run --spec <fresh-spec> --spec-sha256 <sha> --review <fresh-review> --review-sha256 <sha>
```

Root ACK uses caller v4 `--mode ack --request <exact-permit-and-tool-request>
--request-sha256 <sha>`. After the real remover exit, the independent finalizer
uses:

```text
<c.base.PYTHON> -B -u <Q>/minicpm5_retained_retirement_caller_v4.py --mode finalize --request <fresh-final-request> --request-sha256 <sha>
```

`c.base.PYTHON` is the unchanged reviewed interpreter. Do not substitute a
different command spelling: the read-only holder helper checks the complete
actual owner argv. The final request still has exactly spec, review, terminal,
actual_tool and updated_at. Both modes retain original source/spec enrollment,
resources, journals, output schemas and full History requirements.

This is for a fresh r2 binding after root has reconciled the active predecessor.
It cannot retrofit an existing v2/v3 spec, migrate a partial journal, skip a
previous failed result, or replace an already running finalizer.

## Validation and limits

62 audited source/mock tests pass. They include all previous lifecycle, helper
scope and exact shared-source tests, plus eight resolve-cycle tests covering
10,000 callbacks, full exit/failure checks, source/metadata/storage/latest
mutation, slow full checks, forced callbacks, signal/deadline failures,
unthrottled post-resolve physical callbacks, holder/publication mutation, and
exact four-namespace helper command compatibility. No actual JSON/raw/plan,
captured namespace, process, lock or native library access is allowed by the
test runner.

Run deadline remains 21,600 seconds, finalizer deadline 7,200, helper 90, ACK
180. CPU 7, nice 19, 6 GiB address space, 32 MiB files, 128 MiB control and
journal budgets, 4,706 journal entries and disk reservations remain unchanged.
Physical and postcondition callbacks deliberately remain unthrottled and may
still be expensive; this change targets only the identified resolve-stage
multiplication. No model or end-to-end speedup is claimed.
