# SmolLM3 startup guard repair: source preparation, 2026-11-08

SmolLM3's last run (`sio13`) stopped before serving a request because its
15-second process-title handshake expired. The original failure remains a
failure. The new work reduces the cost of the checks surrounding that handshake;
it has not yet established a successful complete handshake or a model speedup.

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
