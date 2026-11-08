# Nemotron qualification: storage-inventory race — 2026-11-08

The v6/r2 run completed all nine alias stages: **70 device checks passed with
normal retained original exits**. It crossed the earlier controller failure
point. A later candidate worker then stopped before executing its split test,
when storage monitoring encountered a temporary JSON file being renamed. The
external task exited with code **1**; the failed run and physical cleanup are
retained. Full-model and speedup qualification remain pending.

| Result | Evidence |
|---|---|
| Preparation | Passed |
| Nine alias stages | All 70 device checks and original exits passed |
| First original split control | CPU checks and compilation passed; device output/address mismatch reproduced, as required by this negative control |
| First candidate split | Not executed: initial storage guard raised FileNotFoundError |
| Completed predecessor history | 682 planned services and 106 captured standalone process identities |
| Recovery | Exact original audit confirmed drain and explicitly released the failed run |

## Independent device finding

For original projection split input `[1, 128, 11328]`, with split sizes
`[5120, 6144, 64]`, the compiled original path mismatched **1,143 of 1,152**
sampled addresses. For output 0 at token row 1, the observed source offset was
5,120 rather than the expected 11,328. Complete device-output byte comparisons
also differed for both test patterns, while CPU checks passed. The raw failure
and compiled graph fingerprints were retained.

This is evidence about this exact original split consumer, shape and installed
compiler path. It does not qualify the candidate or a full model, and a failing
original implementation cannot be used as a speedup denominator.

## Storage-monitor fix

The controller saves high-water records through a `.json.tmp` file and atomic
rename. Another process can list that temporary name just before the rename,
then fail when it tries to stat it. The candidate worker traceback shows this
exact missing temporary path inside its initial storage guard.

The v7 successor retries the **whole directory inventory**, at most eight
attempts, when an entry disappears. It starts the sum again and includes all
remaining temporary/pending files. Permission errors, root replacement or
removal, invalid links and persistent churn remain failures. No file is silently
skipped and none of the storage budgets or floors is reduced.

**39 source/filesystem regressions passed**: 16 inventory checks, including an
actual atomic-rename reproduction, plus all 23 earlier exit-observation tests.
All v6 source files remain unchanged. The complete v7 patch is attached for review.

## Successor status and scope

At 2026-11-08 01:54:21 UTC, actual v7/r3 external and controller identities were
confirmed, with stage `prepare`. The new run repeats all 52 original
stages in fresh directories with package `60ab7f5`. This is a timestamped launch
observation, not a completion claim. Cold cache, long decode, state/slot reuse,
quality, paired workloads and raw verification remain required.

The unified model implementation remains `e0a01d3`. These changes fix the
external experiment harness and introduce **no new model speedup ratio**.
The [historical speedup table](multi-model.md) retains its existing scope.

## Evidence

The [manifest](evidence/nemotron-storage-race-20261108/manifest.json) binds the
[compact outcome](evidence/nemotron-storage-race-20261108/r2-outcome.json),
[original worker traceback](evidence/nemotron-storage-race-20261108/candidate-worker-failure.log),
[source-check result](evidence/nemotron-storage-race-20261108/source-check.json),
[complete patch](evidence/nemotron-storage-race-20261108/runtime-v7.patch),
[regression checker](evidence/nemotron-storage-race-20261108/check_inventory_v7.py)
and [actual successor observation](evidence/nemotron-storage-race-20261108/r3-observation.json).
The checker/patch require the original retained experiment workspace and fixtures;
they are not standalone tests of the unified installation package.

The later [projection milestone](nemotron-projection-qualification-20261108.md) records eight completed candidate shapes and their paired original controls.
