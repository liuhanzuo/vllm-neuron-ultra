# SmolLM3 startup guard repair: source preparation, 2026-11-08

最新进展：Nemotron r4 已以失败状态完成回收；Smol 下一代读取器和九模块衔接已通过源码检查。详见 [2026-11-08 后续记录](nemotron-r4-terminal-smol-reader-20261108.md)。

**Latest status:** the third codec candidate and delayed intent publication are
connected through all nine successor source modules. There are 71 codec/stat,
nine writer and nine integration checks. Complete source-level title transactions
with real writes and retained metadata workloads pass the original deadline,
including both acknowledgement orderings. This remains source preparation:
no fresh Smol native run is admitted, and there is no new model speedup.

SmolLM3's last run (`sio13`) stopped before serving a request because its
15-second process-title handshake expired. The original failure remains a
failure. The new work reduces the cost of the checks surrounding that handshake;
it has not yet established a successful live-worker handshake or a model speedup.

## What changed

The candidate metadata codec validates absolute POSIX paths using string
operations instead of constructing `PurePosixPath` objects for every entry.
It retains the original two-leading-slash semantics, canonical path rules,
suffix exclusions, type and length checks, wire schema and size limits. Source
chunks are still reread and authenticated, with no cache across metadata reads.

The candidate file-identity reader uses Linux `O_PATH`, `fstat` and
`/proc/self/fd` to obtain metadata and the resolved target from the same open
file descriptor. It reads no file bodies, checks identity again before returning,
and closes the descriptor on both success and failure. This helper is Linux-only.
It replaces repeated Python path resolution; it does not remove file checks.

The first candidate used `os.stat` and `os.path.realpath`; it remains archived.
The second candidate is `metadata_codec_v4`, `metadata_store_v3` and
`stat_identity_v2`. Neither candidate is connected to a new admitted runtime.
Frozen sio13 and live Nemotron sources have not been modified.

## Verification and measured scope

The second candidate passed **71 tests**. Coverage includes 20,000 randomized
path strings, POSIX corner cases, all **37,315** retained source paths and their
resolved targets, original adversarial codec/transport checks, symbolic-link
redirection, file replacement, mutation, missing files, link loops, access without
file-read permission, and descriptor cleanup. The first candidate passed 68 tests.

The component diagnostic used the actual Ray trace dispatcher with explicitly
synthetic dispatcher state. Each sample performed **three retained permit reads**
and **two complete source-stat scans** (74,630 observations). All expanded
metadata hashes and all observed source identities matched the original.
The sequence was original / candidate / candidate / original.

| Implementation | First sample | Second sample |
| --- | ---: | ---: |
| Original components | 18.756 s | 18.597 s |
| First candidate, separate earlier diagnostic | 11.427 s | 11.477 s |
| Second candidate | 5.135 s | 5.183 s |

The second candidate used about **72% less time** for these components.
These are maintenance timings, **not inference timings**. The first candidate
was measured in an earlier diagnostic, so the table does not imply one jointly
paired comparison of all three implementations. Two samples are insufficient
for a statistical latency claim.

The replay did not execute a complete live guard, process-title transaction,
Ray import, model, or native inference framework. Moving the intent timestamp
alone also cannot establish correctness: post-publication checks, controller
acknowledgement, and after-call/commit checks still consume the same fixed
15-second deadline. Complete transaction validation is required before admitting
a fresh successor.

## Experiment status

As of 2026-11-08T04:39:22.241729+00:00, Nemotron r4 had **15/52** stages with retained successful
exits and passing worker results. The observed stage was `split-nemotron-projection-b1-t512-tensor-split`.
The experiment was still active; this is a timestamped progress observation,
not a final qualification or a new speedup result. No concurrent Smol native run
was started.

All 15 model families remain in scope. Smol's three configurations, twelve
rounds, 264 requests, 33,696 output tokens, cold-cache checks, long decode,
request-slot reuse and quality gates remain required. The 15-second title,
7,200-second run and 86,400-second matrix limits are unchanged.

The [evidence manifest](evidence/smollm3-guard-source-20261108/manifest.json)
contains the exact candidate sources, test/probe scripts, measured outputs,
exit receipts and progress references. These scripts require the retained
experiment workspace and fixtures; they are not standalone package tests.
The unified model implementation remains `e0a01d3`; historical model speedup
tables are unchanged.

## Third candidate and complete title source validation

The latest codec inlines 44 existing constant-message validation calls. Every
condition and rejection message is preserved; this removes Python function-call
cost under the trace dispatcher. The existing 71 codec/stat/transport tests pass,
and a structural check proves that the only change from codec v4 is this inlining.

A new original / candidate / candidate / original component replay measured
19.569 / 3.142 / 3.113 / 18.768 seconds. Candidate metadata reads took about
2.40–2.43 seconds, and both complete stat scans about 0.709 seconds. This is the
same three-read/two-scan diagnostic scope, not a model benchmark.

Two full permit copies plus two hashes cost about 0.39–0.40 seconds. A plain-JSON
copy candidate saved about 0.12 seconds and passed nine tests, but it was **not
integrated**. Its restricted object domain and small benefit do not justify
changing the existing copy behavior here.

The larger source probe executes the original EngineCore Python entry, title
wrapper, actual Ray dispatcher, original atomic durable writer and original
monotonic deadline. Each worker poll performs three real retained permit reads,
the actual runtime-source and full-source identity scans, payload **metadata**
checks, two permit copies, and the source-plan/base hashes. Process identity,
the native setter and controller scheduling are explicitly fixtures. The native
setter is called once as a stub; no inference request runs.

With the original publication ordering, the state machine reported commit, but
the commit file finished publishing **15.151 seconds** after intent creation.
Its decision timestamp had been sampled earlier. This does not demonstrate
publication within the 15-second budget.

The new dedicated before-intent writer creates the intent **after** the ordinary
mandatory pre-publication poll. It retains both full polls, size/name checks,
exclusive durable publication, byte count/hash/path verification, and the same
15-second duration. Generic writer behavior and every other title-state-machine
function remain unchanged. It does not introduce a generic unchecked callback.

| Source probe | Worker polls | Commit file published after intent creation | Within 15 s |
| --- | ---: | ---: | --- |
| Original ordering, latest codec/stat workload | 9 | 15.151 s | No |
| Candidate, acknowledgement before worker state check | 9 | 11.782 s | Yes |
| Candidate, acknowledgement after worker state check | 9 | 12.238 s | Yes |
| Integrated nine-module source, acknowledgement after check | 9 | 12.112 s | Yes |

These are four separately executed source fixtures, not repeated live service
trials or a confidence interval. Total fixture wall time was about 33 seconds;
pre-intent and post-commit checks also ran outside the title interval. A live
controller would be scheduled separately and real service/environment/lock
checks remain to be exercised in the future admitted run.

Nine writer controls cover actual no-overwrite publication, bad receipts,
pre/post-check failure, original run deadlines, and exact preservation of the
ordinary writer checks. Nine source integration checks verify all predecessor
hashes, actual peer-module bindings, unchanged model science/resources, and
preservation of the other six module bodies after namespace/import changes.
The new `sio14` runtime directory has not been created.

The [latest evidence manifest](evidence/smollm3-title-guard-20261108/manifest.json)
contains all candidate modules, source checkers, original and successor probe
outputs, and the actual fixture receipt bytes. Admission still requires the
completed Nemotron r4 predecessor catalog, an accepted metadata reader, the full
source closure, budget/finalizer and execution specification/review. The original
three configurations, 12 rounds, 264 requests, 33,696 tokens, cold/long/slot/quality
checks and resource limits remain in force.

At 2026-11-08T05:02:05.630101+00:00, Nemotron had 28/52 retained successful stages and was observed at
`split-nemotron-conv-b1-t128-tensor-split`. Its native experiment remains the sole serial enrollment. Historical
model speedup tables and unified model implementation `e0a01d3` are unchanged.
