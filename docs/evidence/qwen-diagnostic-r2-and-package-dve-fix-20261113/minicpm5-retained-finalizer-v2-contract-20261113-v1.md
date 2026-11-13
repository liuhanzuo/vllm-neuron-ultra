# Mini retained replay: v2 final publication successor

This is a source-only correction to the independently reproduced v1 final publication race. The frozen v1 caller, History, contract, observer, raw transport, replay, comparator, delivery files and test logs remain unchanged. No current binding, actual History, lock, proc/service state, payload, native execution or cleanup was used in this work.

## Defect and exact correction

The v1 `publish_completed()` checked metadata and latest handoff, then performed a final blocking `drained()` check, and published completion immediately afterward. An independent source/mock reproduction inserted an unknown newer handoff during that last drain and demonstrated that v1 still published completion. The reproduction is `check_minicpm5_retained_replay_final_guard_independent_20261113_v1.py` (SHA-256 `d3637201acb536e88be860563308234d0b2f1c520bd5b99ae5b2b11ab547f83a`); its recorded output is `minicpm5-retained-replay-final-guard-independent-probe-20261113-v1.json` (SHA-256 `ae232a14d60791054b289af1eb21ee0d520cc0cb18f98aab7d726d131006566c`). Old positive tests did not exercise this boundary.

New `minicpm5_retained_replay_finalizer_v2.py` retains the entire old final publication function, with three narrowly tested additions: authenticate the new finalizer source before admission; recheck that source during forced/cadenced guards; and repeat `history.unchanged(); guard(True)` after the final `drained()` and immediately before `io.once(...)`.

The post-drain checks reject a changed inherited metadata identity, the original running-handoff identity changing, or an unknown newer handoff. Nothing can publish completion from these regression scenarios. The ordinary unchanged scenario still publishes, with the regression event trace proving that both the complete metadata unchanged-check and a fresh latest-handoff observation occur after the last drain and before publication.

This is a bounded correction at the existing publication boundary; it does not claim a new atomic filesystem transaction or change the original retained-lock cooperation model. All inherited original-lock, actual observer owner-gone/external tool exit, complete History/scientific equality, per-phase typed ownership, physical drain and no-retirement constraints remain.

## Concrete future source enrollment

Use this new module's pure wrapper at the original binding step:

```python
from minicpm5_retained_replay_finalizer_v2 import render_binding
result = render_binding(context, v1_source_bytes, original_plan_bytes, finalizer_bytes)
```

- `context`, all four `v1_source_bytes`, and `original_plan_bytes` obey the unchanged v1 contract.
- `finalizer_bytes` is the complete exact byte content of `minicpm5_retained_replay_finalizer_v2.py`.
- The wrapper copies the context, adds the v2 file's exact SHA to the complete inherited source map, and invokes the frozen v1 pure renderer. Every old source/input/namespace/resource field remains intact. A conflicting inherited v2 pin is rejected; no source is replaced.
- Returned `mandatory_source_sha256` includes all four v1 files plus v2. Returned `final_publisher` identifies the exact v2 path/SHA. Runtime enrollment remains false.
- Root independently reviews the exact resulting spec and complete source map before any pair/replay run. The existing exact root review schema includes that map; the v2 source must therefore be enrolled from the start. A later finalizer cannot append its own source to an already executed frozen spec.
- Actual v2 execution refuses a missing v2 pin, wrong path or changed v2 content. The original `caller.refs()` full loaded-source validation remains active.

The pair/replay outer, ACKs, independent v1 observer, root live capture, and actual observer terminal remain exactly the v1 orchestration. The only replacement is the last invocation, after root has actually observed the independent observer's exit 0:

```text
/usr/bin/python3 -B /home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5_retained_replay_finalizer_v2.py --request <exact final-request path> --request-sha256 <exact SHA>
```

The final request remains `{observation, observer_live, actual_observer_tool, updated_at}` with the actual observed observer session/exit/terminal chunk. The finalizer acquires the original lock, rechecks complete History and physical scope, performs the original final drain, then executes the new post-drain metadata/freshness check before publishing. A failed guard preserves existing evidence and publishes no terminal handoff. The old v1 observer `--mode finalize` is not the reviewed successor entry.

## Validation and limits

Ten focused source/AST/mock tests exercise stable publication, earlier unknown handoff, last-drain unknown handoff, last-drain inherited metadata mutation, last-drain running identity mutation, pure full-map extension, conflicting pin rejection, actual self-source enrollment/hash checks, mechanical preservation of the old finalizer body, and reuse of the original actual-exit/full-History constraints. All synthetic filesystem work uses `/tmp`; no actual observer or finalizer is launched. A separate combined rerun retains the original 44 caller checks.

This delivery is not permission to run or delete data. Retirement remains separately unadmitted, and the prior partial-JSON retirement migration limitation is unchanged.
