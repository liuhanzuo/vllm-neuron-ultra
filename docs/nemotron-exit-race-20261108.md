# Nemotron qualification: exit-observation race — 2026-11-08

**Later result:** v6/r2 completed all 70 alias checks with normal exits, then
failed on a separate temporary-file inventory race before candidate execution.
See the [follow-up result and v7 fix](nemotron-storage-race-20261108.md).


The first 52-stage qualification attempt stopped in our experiment controller,
after the last alias test had written a passing result. The external task exited
with code **1**. This is an experiment-tool failure; it does not establish a
Nemotron numerical failure or an upstream vLLM-Neuron/Neuron defect.

| Evidence from the failed attempt | Observed result |
|---|---|
| Input preparation | Passed with retained original exit 0 |
| Alias/state checks reported by device workers | 70 passing checks across nine stages |
| Checks with the normal retained-exit record | 62 across the first eight alias stages |
| Last alias stage | Eight passing checks; cleanup captured original exit 0, but the normal retained-exit record was never written |
| Controller / external task | Original exit 1 / actual tool exit 1 |
| Recovery | All planned groups drained; explicit failed release completed |
| Complete predecessor history | 629 planned services and 105 captured standalone process identities |
| Full-model or speedup qualification | Not reached; no new speedup ratio |

The 70 worker checks and 62 normally retained checks are overlapping evidence,
not two additive totals. The last stage's cleanup exit does not repair the
failed controller or qualify the whole experiment.

## Failure mechanism and fix

The controller queried a service's `MainPID`, then read `/proc` to verify the
worker. A worker can finish between those reads. The old implementation rejected
an absent process as `Actual process argv/group` before recording its retained
service exit. Saved evidence contains that traceback, passing worker output,
and a subsequent original service exit 0. The failing instantaneous process
sample was not saved, so absence versus zombie state cannot be reconstructed.
The absent-process sequence reproduces the same rejection in a regression test.

The v6 successor retries for at most one second, and only after a previously
acknowledged original process disappears or is verified as the exact original
zombie. Every new service sample still checks invocation, original PID,
installed command and resource limits. A live process with changed arguments,
cgroup or start identity is rejected. A missing service cannot manufacture a
successful exit, and an original nonzero exit remains a failure.

All **23 source regressions passed**, including the old failure reproduction,
recovery, identity replacements, failed exits, bounded retry, original source
fingerprints and unchanged scientific/resource contracts. These are source
checks, not evidence of model inference passing.

## Successor run

At **2026-11-08 01:35:10 UTC**, the separately launched v6/r2 controller was live
and was at stage `alias-nemotron-conv-r4`. It repeats all **52 original stages** in fresh
output directories using original model package `60ab7f5`. Cold cache, long
decode, state/slot reuse, quality, paired workloads, raw verification and all
storage/resource limits remain required. No earlier partial stage is treated as
a completed successor stage. This timestamped launch status is not a final result.

The unified branch's model implementation remains `e0a01d3`. The attached
patch changes the external qualification harness; it is not a model-kernel
optimization or an additional speedup claim. Existing
[speedup tables](multi-model.md) keep their original scopes.

## Evidence and reproduction scope

The [manifest](evidence/nemotron-exit-race-20261108/manifest.json) binds a
[compact outcome](evidence/nemotron-exit-race-20261108/r1-outcome.json),
[source-check result](evidence/nemotron-exit-race-20261108/source-check.json),
[complete successor patch](evidence/nemotron-exit-race-20261108/runtime-v6.patch)
and [regression checker](evidence/nemotron-exit-race-20261108/check_exit_race_v6.py).
The patch and checker target the retained experiment workspace, including its
original v5 files and saved fixtures; they are not a standalone package test.
The compact outcome pins the full local terminal audit. Model weights and raw
tensors are not included in Git.
