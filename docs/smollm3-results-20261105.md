# SmolLM3 packaged comparison — 2026-11-05

The opt-in NKI cache reader/writer passed the defined full-model comparison at
implementation commit `e2d1e28b83411863acef5f12ff3db1f887866a76`. Fresh reference
and candidate artifacts were compiled and loaded, then both completed all
correctness, quality and HTTP workload stages. NKI remains disabled by default.
[Setup and supported scope](smollm3.md) are unchanged.

This result covers original BF16 SmolLM3-3B weights, revision
`a07cc9a04f16550a088caea529712d1d335b0ac1`, on Trn1 with TP2, physical decode
batch4, prefill batch1, context2048 and minimum decode bucket2048. Both variants
use the standard CPU sampler. Only the candidate enables NKI cache movement.
The baseline is this branch's qualified reference adapter with the same
minimum decode bucket; it is not an unmodified upstream or natural-bucket result.

## Measured throughput

Values are medians of three sequential rounds, in output tokens/second.
Speedup is candidate median divided by reference median. The four-request column
is aggregate throughput. HTTP timing includes the work measured by the unchanged
benchmark client, with the same 600-second socket timeout and output budgets on
both sides. Reference ran first, then candidate; there is no randomized-order or
statistical-significance claim.

| Workload (input/output tokens) | Reference, single | NKI, single | Ratio | Reference, 4 concurrent | NKI, 4 concurrent | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Short decode (128/512) | 1.8519 | 15.1059 | 8.16× | 7.3912 | 59.4799 | 8.05× |
| Medium prompt (512/128) | 1.8610 | 15.0433 | 8.08× | 7.4042 | 57.7739 | 7.80× |
| Long prompt (1536/128) | 1.8583 | 14.8737 | 8.00× | 7.3624 | 55.2979 | 7.51× |
| Short stochastic (128/512) | 1.8142 | 12.9312 | 7.13× | 6.8441 | 36.3668 | 5.31× |

Mixed lengths measured **5.8601 → 32.2224 tokens/second (5.50×)**. Each round
contains16 requests, at most4 in flight, with input/output pairs
`(32,32), (128,128), (512,256), (1536,128)` repeated four times:2176 generated
tokens per round on each side. Stochastic and mixed workloads use temperature0.7,
top_p0.9 and top_k40; their sampled texts are not expected to match.

The fixed request sets establish these observed improvements. They do not
establish gains for different batch sizes, contexts, sampling implementations or
hardware. Cache read/write contributions have not been separated.

## Correctness and lifecycle coverage

All15 stages completed with zero exit codes and no recorded OOM or timeout.
Native preflight independently exercised wheel installation, default/reference/NKI
plugins in parent and spawned processes, and configuration validation.

- **Original numeric32:** Both variants passed the unchanged HF comparison,
  including32/32 top-1 matches, minimum cosine0.9991750, maximum KL0.0111071 and
  maximum absolute error0.66015625. Eight requests contain six unique prompts.
- **Cold state:** Seven independent initializations,84 frames,42 duplicate-slot
  comparisons,1008 initial-zero checks and1008 post-decode finite-state checks
  passed per variant. Context buckets128–2048 were covered; decode used2048.
  This is cold prefix-length coverage, not prefix-cache support.
- **Long decode and reuse:** The original1537 operations include1529 main
  teacher-forced steps and1532 decode calls. All84 checkpoints passed against
  the exact native cached-HF trajectories, covering slot reuse, decode after
  reuse, partial batches and inactive-slot resume.
- **Quality:** Both scored60/96, with all96 texts and scoring decisions equal.
  Results retain36 failures: GSM8K52/64, arithmetic0/8, JSON0/8, retrieval0/8,
  and list8/8. Two responses reached the length cap; neither side had request
  errors. This finite suite does not establish general accuracy equivalence.
- **Deterministic serving:** All76 compared decoded texts and token-usage records
  matched:19 each for initial120, short decode, medium prompt and long prompt,
  including streaming requests. Stochastic/mixed text equality is not a gate.

The original numeric thresholds remain cosine≥.995, KL≤.02, maximum absolute
error≤2, top-1 match fraction≥.9, and required top-1 agreement for reference
margin>.5. No failed frames were dropped or thresholds relaxed.

## Reference limits and remaining work

Long validation uses exact native **cached** trajectories. The historical
cached-native versus full-causal-native crosscheck still fails two position510
rows (minimum cosine0.9920796, maximum absolute error2.109375). The inherited
`original_full_causal_reference_crosscheck_passed` field is hardcoded in the old
long-run harness and is not evidence that this crosscheck passed. The audit
explicitly rejects that interpretation. Old natural/small-bucket reference
failures remain unresolved.

Memory snapshots were taken while idle after workload evaluation; they are not peak
memory measurements. Prefix/block KV, chunked prefill, LoRA, speculation and
other precision/TP/batch/context settings remain outside this qualified scope.
No automatic best-known configuration promotion was performed, and this result
does not mean the model family is fully optimized.

Next work should attribute cache read/write costs and profile the qualified
end-to-end path, then repeat full qualification for any new change. Broader
bucket support must address the preserved failures rather than bypass them.

## Evidence

The local experiment is `smollm3-branch-full-v3`, with plan SHA256
`eaeda53c3f2c3ef3e84fc5b0da27435597673adc515fc95eb4a9db0fe5789372`.
[Machine-readable evidence](evidence/smollm3-20261105.json) preserves raw round
throughput/timing, correctness summaries, protocol identity and source/report
hashes. The [independent terminal audit](evidence/smollm3-terminal-audit-20261105.json)
is included alongside it. Model weights,
compiled graphs, SDK binaries and benchmark response texts are not bundled.
