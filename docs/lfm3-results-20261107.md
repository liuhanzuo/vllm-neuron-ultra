# LFM3-1.7B packaged comparison — 2026-11-07

The packaged NKI predicated decode writer passed the defined paired full-model
test at implementation commit `822e1bd826321dc078226c2da3f199e1d385146f`.
Three fresh measurement rounds show **2.02× short-input single-request throughput**
and **8.51× long-input single-request throughput**. The independent final audit
also checked all 81 deterministic workload request pairs, including streaming:
decoded texts and token usage are identical.

This comparison covers original BF16 `LiquidAI/LFM3-1.7B`, revision
`ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`, on Trn1 with TP2, physical decode
batch 4, prefill batch 1 and context 2048. Both variants use the same packaged
fast CPU sampler. The baseline is **this branch's reference adapter**, not
unmodified upstream vLLM-Neuron. The candidate enables
`NEURON_LFM3_NKI_WRITER=1`; other admitted settings remain equal. NKI stays opt-in.
[Setup and restrictions](lfm3.md) apply.

## Throughput

Values are three-round medians in output tokens/second. The four-request values
are aggregate throughput. Speedup is candidate median divided by reference
median. Each role used a separate, newly compiled artifact. Compilation is
outside the warm HTTP measurements. The role order was reference/candidate,
candidate/reference, reference/candidate; it was not randomized and does not
establish statistical significance.

| Workload (input/output tokens) | Reference, single | NKI, single | Ratio | Reference, 4 concurrent | NKI, 4 concurrent | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Short decode (128/512) | 30.1602 | 60.9265 | 2.02× | 105.4957 | 189.9923 | 1.80× |
| Medium prompt (512/128) | 29.3587 | 54.9311 | 1.87× | 99.7571 | 167.6554 | 1.68× |
| Long prompt (1536/128) | 5.3259 | 45.3239 | 8.51× | 20.4737 | 137.2682 | 6.70× |
| Short stochastic (128/512) | 29.6335 | 59.0103 | 1.99× | 99.2996 | 170.9155 | 1.72× |

Four-request streaming throughput was 105.0678 → 189.5181 tok/s for short
decode (1.80×), 99.2799 → 165.2053 for medium prompts (1.66×),
20.4854 → 137.3175 for long prompts (6.70×), and 99.5214 → 171.6905
for short stochastic generation (1.73×).

Mixed lengths measured **20.7745 → 122.8816 tok/s (5.92×)**. Each round
contains 16 requests, at most 4 in flight, with input/output pairs
`(32,32), (128,128), (512,256), (1536,128)` repeated four times:
2176 generated tokens on each side. Stochastic and mixed workloads use
temperature 0.7, top_p 0.9 and top_k 40. Their generated texts need not match.

Inputs are fixed token-trimmed native chat prompts. Output budgets are forced
with minimum tokens and EOS ignored; these are throughput workloads, not quality
questions. HTTP timing includes prefill, decode and client overhead, with a
240-second socket timeout on both sides. Streaming TTFT means time to the first
nonempty text chunk. Mixed per-request latency excludes client queue time;
aggregate elapsed time includes all 16 requests. These results are not device
kernel timings or measurements of cold compilation latency.

## Correctness and lifecycle checks

All original numeric gates were retained: finite metrics, cosine ≥ 0.995,
KL ≤ 0.02, maximum absolute error ≤ 2, top-1 match fraction ≥ 90%, and exact
top-1 agreement whenever the reference margin exceeds 0.5.

- **Numeric32 against the original HF reference:** both variants passed, each
  matching top-1 in 29/32 rows. Minimum cosine was 0.9981968 for reference and
  0.9985716 for NKI; maximum KL was 0.0080034 for both. All required high-margin
  top-1 comparisons matched.
- **Cold state:** both completed six independent prefixes (64, 160, 320, 768,
  1280, 1536), with 96 finite output rows and finite final states. Candidate
  versus reference matched top-1 in 96/96 rows; **90/96 were bitwise equal**.
  Minimum cosine was 0.9999706, maximum KL 0.000001825, maximum absolute error
  0.125. This passes the original tolerance; it is not universal bit equality.
- **Long decode and reuse:** both completed 1529 teacher-forced decode steps
  and all 84 checkpoints, including slot reuse, partial batches and inactive
  slot resume. Both matched top-1 in 84/84; minimum cosine 0.9995099 and maximum
  KL 0.0067298. The original oracle is full causal HF inference with a separately
  checked incremental segment, not an independently rerun HF trajectory here.
- **Quality:** all 96 texts, prompts, usage, scoring decisions and finish reasons
  match. Both scored **63/96**: GSM8K subset 46/64, arithmetic 0/8, JSON 8/8,
  retrieval 6/8, list 3/8. There were zero request errors. This custom finite
  suite is not the full official GSM8K benchmark or proof of general equivalence.
- **Deterministic serving:** all 81 decoded-text and usage pairs match across
  three workloads, three rounds, and single/four-request/streaming modes.
  All 312 request usage records and 78 rates were independently checked against
  raw counts and elapsed time. This is text equality, not a token-ID trace.
- **Writer primitives:** the actual records pass 1648 CPU cases, 9 device graphs
  with 43 checks, and 12 persistent-alias graphs with 60 checks.
- **Sampler migration:** 90 synthetic records match the historical and packaged
  implementations in sampled tokens and RNG state. The 30 top-k/top-p records
  also have equal shortlists; upstream-token equality was not asserted for those
  records. Explicit request generators and HTTP seed support remain unqualified.
- **Lifecycle:** the independent audit authenticated all 30 launch/owner records,
  checked 20 successful worker exits, and confirmed 10 server groups and the
  standalone controller had drained. Retired server absence proves physical
  drain; completed result records establish that serving ran successfully.

## Evidence and limits

[Evidence manifest](evidence/lfm3-paired-20261107/manifest.json),
[final independent audit](evidence/lfm3-paired-20261107/root-audit.json), and
[paired summary](evidence/lfm3-paired-20261107/qualification.json) accompany the
raw per-round workload, numeric, cold, long, quality and primitive JSON records.
The copied audit source is an archival script for the original local experiment
tree; it does not make this evidence directory a standalone inference runner.
Tensor payloads and compiled binaries are not bundled or rehashed by the final
JSON audit. The frozen source checkout and all 146 source pins were verified.

The new result reproduces the historical optimization in the packaged branch.
It does not qualify other batch sizes, contexts, sampling options, model
families, precisions or hardware. The largest relative gain is against the
slower long-input reference path; it is not an 8.5× improvement for every request.
No model family is declared fully optimized.
