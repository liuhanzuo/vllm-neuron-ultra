# MiniCPM5 prepublication recovery finalizer: source delivery

This delivery authors and tests sources only. It did not execute actual History,
inspect actual locks/processes, read raw/model inputs, replay a model, delete
anything, or publish a lifecycle result. A separate root runtime review is
required by the explicit entry contract. The root already owns the task and can
make that review without asking the user for another permission.

## Frozen new sources

All paths below are under
`/home/ubuntu/experiments/trn1-latest-models-20261031/research` (`Q`).

| File | SHA-256 |
| --- | --- |
| `minicpm5_retained_replay_finalizer_v5.py` | `17ffdb97e201bae58b668618e80f0c7b6cde301ed748ce4471adf413786ca8e6` |
| `minicpm5_retained_replay_catalog_v2.py` | `5819a8c56adc0ae93f6e38dda38c1dcb5c978c8f09cb55b885534f1143f2e649` |

These **two sources only** are the runtime supplement. Tests, runner, this note,
and delivery manifest are not imported by the finalizer. Frozen v3, catalog v1,
the old spec/review/overlay, and all original history remain unchanged.

Catalog v2 differs mechanically from v1 in one predicate and its error text:
prior catalog pins must occur with the same digest in the **full authenticated
source map**, which includes both recovered and extension pins. v1 incorrectly
required every prior pin in the extension. Actual saved metadata independently
reproduced 3,484 valid prior recovered pins outside that extension. This is a
separate diagnosis; tool 15003's actual exit/cause is still unknown.

The finalizer repeats all frozen `caller.refs` spec/review/source/accepted
predicates. It then explicitly authenticates the separately reviewed supplement
and checks loaded sources against the union. It does not call old refs with a
forged review or patch the old function. Full unchanged History receives an
in-memory spec differing only in the source map, while its resolved historical
map must still equal the original spec. Composer v2 receives the original spec.
A pure rebinder then updates both receipts, source/delta counts and digests,
outer-to-observer-to-idle-to-catalog references, and the outer row's new sources.
All previous rows, actual tools, phase owners, and science flags remain unchanged.

## Exact future review and request

Create a **new** root-authored review with exactly these keys and values. Every
reference is the two-key `{path, sha256}` shape. No extra review fields are
accepted. `recovery_name` matches `[a-z0-9][a-z0-9-]{0,47}` and names a fresh
directory under the original control root.

```python
review = {
    "schema": "minicpm5_retained_replay_recovery_source_review_v1",
    "reviewer": "/root",
    "passed": True,
    "blockers": [],
    "original_request": {
        "path": Q + "/minicpm5-retained-r1-20261113-finalizer-request.json",
        "sha256": "dc87ae97860b0fba7dca1084b374e76207e221dcc30902c8cabca36064a22b61",
    },
    "interruption": {
        "path": Q + "/minicpm5-retained-r1-20261113-finalizer-interruption-observation-v1.json",
        "sha256": "0e13bc1c1fe920addf9cfadfde061dc5eaeb5f2323ad134222fc62ada84f4d15",
    },
    "original_spec": {
        "path": Q + "/minicpm5-retained-r1-20261113-spec.json",
        "sha256": "7fc0600ee1ad83a3c293c2134df372d45e18858ab02837a1b6e1a424a6e91bf6",
    },
    "original_review": {
        "path": Q + "/minicpm5-retained-r1-20261113-review.json",
        "sha256": "e78b315a8a5fd1825f3b4534a09fa581612f348a86f3e29220c9d5903937e56f",
    },
    "source_sha256": {
        Q + "/minicpm5_retained_replay_finalizer_v5.py":
            "17ffdb97e201bae58b668618e80f0c7b6cde301ed748ce4471adf413786ca8e6",
        Q + "/minicpm5_retained_replay_catalog_v2.py":
            "5819a8c56adc0ae93f6e38dda38c1dcb5c978c8f09cb55b885534f1143f2e649",
    },
    "recovery_name": "r2-20261113",
    "launch_authorized": True,
    "deletion_authorized": False,
    "model_or_raw_replay_authorized": False,
}
request = {
    "original_request": review["original_request"],
    "interruption": review["interruption"],
    "recovery_review": {"path": NEW_REVIEW_PATH, "sha256": NEW_REVIEW_SHA256},
    "updated_at": ROOT_CURRENT_ISO_TIMESTAMP,
}
```

Future entry (root captures the actual tool session, stdout/stderr and terminal):

```text
/usr/bin/python3 -B /home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5_retained_replay_finalizer_v5.py --request NEW_REQUEST_PATH --request-sha256 NEW_REQUEST_SHA256
```

The entry emits a bounded `mini_retained_recovery_publisher_live` JSON event
after original-lock acquisition and initial guard/drain, before History. It
contains actual `lock.owner` (PID/start/argv), request reference, original lock,
and no claimed exit. Redirect stdout/stderr to a durable fresh log from launch;
capture the actual helper terminal externally. The helper does not invent its
own final exit receipt or a terminal for tool 15003.

## Runtime boundaries

The original `final-publication` directory must contain exactly the three saved
files, with their recorded identities and hashes. The entire directory and all
three files are checked before/after blocking drains and after every publication.
No `.publishing` remnants may occupy the four final output paths. All four final
paths must initially be absent. Recovery History goes into a distinct fresh
`final-publication-recovery-<recovery_name>` directory. Never rerun v3 or this
recovery into an occupied namespace.

The original retained lock, complete History and physical pair, full scientific
comparison, owner/unit checks, source and metadata freshness, storage budgets,
7200-second deadline, and exclusive writer are retained. Under the lock a bounded
process scan rejects any still-running exact v3 executable argument. Each write
checks all remaining paths absent and all published objects unchanged. Idle
remains the final write; it legitimately replaces the old running latest handoff.
Final drain releases the lock only on the original physical-drain proof.

Expected current counts after the two additional sources:

| Quantity | Expected |
| --- | ---: |
| Original complete sources | 49,069 |
| Original recovered sources | 44,532 |
| Original extension sources | 4,537 |
| Original prior catalog sources | 7,684 |
| Original new extension sources | 337 |
| Recovery complete sources | 49,071 |
| Recovery extension sources | 4,539 |
| Recovery newly recorded sources | 339 |
| Result catalog rows | 169 |

The two appended lifecycle rows still describe actual outer
`79116 / 55d39b / exit 0` and independent observer
`28931 / c33ec1 / exit 0`. Complete replay equality preserves 11,859 inputs,
32 frames, 14 scientific reports, and 98 layouts, including original scientific
failures. This is not full-model qualification or a new speedup claim.

## Validation

`python3 -B run_minicpm5_retained_recovery_tests_audited_20261113_v2.py` passed
17 tests in 18.523 seconds. Actual test tool session 62547, terminal chunk 595395,
exit 0. Log: `minicpm5-retained-recovery-synthetic-tests-20261113-v2.log`.

The audit hook rejects actual JSON/raw/model file opens, `/proc`, `/sys`, NVMe,
native imports, subprocesses, locks, network and filesystem writes. Tests use
synthetic in-memory JSON and source reads only. They check exact old admission
predicate parity, explicit supplemental admission, the recovered/extension bug,
missing/conflicting sources, four object SHA values and downstream consumer,
all actual phase/observer owners, unknown interrupted exit, entire old-directory
names/identities/bytes, occupied/transient outputs, full mock History, mutation
after final drain or each write, and idle-last latest discovery.


## Narrow v5 correction and preserved failed v4

The actual v4 launch exited 1 before lock acquisition or History, with observed
tool-call terminal `888c9b`. The saved sorted spec lists `observer_actual_exit`
before `outer_actual_exit`; v4 incorrectly compared dict insertion order with
the independently saved semantic outer/observer/catalog/idle path list.

v5 changes exactly its SELF source path and the two output-list constructions.
Both now explicitly enumerate `outer_actual_exit`, `observer_actual_exit`,
`successor_catalog`, then the terminal handoff. All other v4 source text remains
identical. New fixtures roundtrip specs through `json.dumps(sort_keys=True)`.
Tests reproduce v4 failure, prove v5 success, and still reject reordered,
missing, duplicated or substituted paths. Other order-sensitive sites were
checked: phase and partial-file arrays intentionally retain their semantic
order; declared units are sorted; map and review comparisons are order-neutral.

Exact saved metadata was independently authenticated and passed into the pure
`validate_interruption` function: v4 fails the expected predicate and v5 passes
all remaining fields. Actual tool call `0b6793`, exit 0. No History, process,
lock or raw operation was executed by this check.

The failed attempt remains in the root's immutable receipt:
`Q/minicpm5-retained-recovery-actual-failed-attempt-20261113-v1.json`, SHA-256
`f25bbb734c7b3224447b04acfbfed2312b486e7ade7120c740c666cee5d6b502`.
It binds the real exit, durable log, original v4 source/request and root's
subsequent physical observations. Frozen v4 is preserved unchanged at SHA-256
`bd935bf8eaf0caa350109f8b9b25095fa0ccd1ee221cc480878e99f61dfd4e49`.
Per root's scope, the failed before-lock publisher remains documentary evidence;
it does not become an invented experiment row or add a runtime schema field.
The runtime supplement remains precisely v5 plus catalog v2, and expected
source/catalog counts above remain unchanged. Old tool 15003's exit and cause
remain unknown, distinct from this actually observed v4 exit 1.
