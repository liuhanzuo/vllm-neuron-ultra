# SmolLM3 cache ablation: write-only output regression

The four-variant experiment at implementation commit
`7da79bc7903238ce19d8afc9e1c061e15fd3d9fa` **failed deterministic output
equivalence**. The write-only variant produced different text for 57 of 76
deterministic requests. It must not be promoted on the strength of its measured
throughput. The selected best configuration remains unchanged.

The [previous qualified combined read/write result](smollm3-results-20261105.md)
applies to its recorded commit and scope. This failed experiment does not replace
that result or qualify the newer commit as a whole.

## Scope and measured throughput

All four variants used original BF16 SmolLM3-3B weights, revision
`a07cc9a04f16550a088caea529712d1d335b0ac1`, Trn1, TP2, physical decode batch4,
prefill batch1, context2048, minimum decode bucket2048 and the standard CPU
sampler. Each variant used a separate compiled artifact. The reference is the
same branch's adapter with the same decode floor, not unmodified upstream.

Values below are medians of three rounds, in output tokens/second. Four-request
figures are aggregate throughput. These are **measurements from a failed
qualification**, not an accepted speedup claim. There is no randomized-order or
statistical-significance claim. The [machine-readable evidence](evidence/smollm3-ablation-20261107.json)
includes all workload medians and raw rates.

| Variant | Short single | Short, 4 concurrent | Medium single | Long single | Mixed |
|---|---:|---:|---:|---:|---:|
| Reference | 1.8524 | 7.3947 | 1.8615 | 1.8585 | 5.8646 |
| NKI read only | 1.8187 | 7.2603 | 1.8276 | 1.8246 | 5.7662 |
| NKI write only **(text mismatch)** | 17.8098 | 70.0553 | 17.7507 | 17.4933 | 35.6573 |
| NKI read and write | 15.1204 | 59.5416 | 15.0412 | 14.8813 | 32.2247 |

Short, medium and long workloads use input/output budgets128/512,512/128 and
1536/128. Mixed uses16 requests, at most4 in flight, with32/32,128/128,512/256
and1536/128 repeated four times. The nominal short-single ratios are0.982×,
9.614× and8.163×, respectively. The write-only ratio is disqualified by output
differences. Read-only offered no measured gain in these runs. These timings do
not isolate a particular kernel's causal contribution.

## Correctness findings

All four variants completed the numeric, cold-cache, long-decode/reuse, quality
and workload phases. The read-only comparison completed. The write-only
comparison failed; the combined comparison and finalization never launched.
In total,29 of32 planned phases have successful completion records.

Independent analysis of the saved response records found:

| Candidate vs reference | Identical deterministic texts | Identical usage records | Quality texts/scoring/usage |
|---|---:|---:|---:|
| Read only | 76/76 | 76/76 | 96/96 |
| Write only | **19/76** | 76/76 | 96/96 |
| Read and write | 76/76 | 76/76 | 96/96 |

Every variant scored60/96 on the fixed quality suite. That suite did not expose
the deterministic serving regression. Write-only matched all19 initial120
requests, then differed on all19 requests in each of short, medium and long
workloads. Within each workload and variant, deterministic text was stable.
The first differing character offsets were2359,365 and174; these are **not token
positions**. The old responses did not retain generated token IDs.

The combined saved-output comparison is useful evidence, but cannot create the
unexecuted combined-comparison or finalization records. No variant is promoted
by this failed four-variant run. The previous cached-versus-full-causal reference
failures and small-bucket failures remain unresolved.

## Failure reporting and next experiment

The worker asserted equivalence before saving its comparison detail. The queue
then attempted to read an absent success record, surfacing `FileNotFoundError`
over the original assertion. The retained worker failure and raw responses
establish the primary error: deterministic output mismatch. This was not an
OOM, timeout or disk-exhaustion failure.

Future code must save failed comparison details before raising and inspect a
worker's exit status before demanding a success record. The frozen failed
runtime and its sources remain preserved.

The next diagnostic will capture actual engine token IDs while preserving the
request history that triggers the failure, then replay the identical common
prefix and physical slots across reference, write-only and combined variants.
Logit margins and cache observations can then distinguish possible causes.
Re-tokenizing old decoded text cannot reconstruct the old generated sequence.
No near-tie, BF16-rounding or cache-corruption explanation has yet been proven.
Any eventual fix still needs the full numeric, cold-cache, long-decode, slot
reuse, quality and serving protocol, with the existing equivalence gate intact.

## Evidence identity

Runtime: `smollm3-cache-ablation-full-v8`; frozen runner/protocol sources: v9.
Plan SHA256:
`3555f02906095452584223a8887bb5a77278e2d98617bbaac25da6eb210f8baf`.
Root failure-audit SHA256:
`4dbbfe6ddb8f64cb25bedb52263013411d3bc7a306135ae6b16bf94e183350da`.
The audit passing means that failure diagnosis and process cleanup were verified;
it does not mean model qualification passed. No weights, compiled graphs,
response texts or credentials are included in the evidence export.
