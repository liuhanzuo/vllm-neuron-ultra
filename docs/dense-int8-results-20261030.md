# Historical INT8 comparison: four dense models

On 2026-10-30, the original Trn1 experiments measured **1.53×–2.08× aggregate
throughput at concurrency four** when changing these models from BF16 weights
to per-channel symmetric INT8. Both precisions used TP2, batch capacity 4,
context 2048 and **on-device greedy sampling** (`dynamic=false`, `do_sample=false`,
`top_k=1`). The earlier Feishu description incorrectly called this CPU sampling;
the saved launch parameters and all eight saved Neuron configurations agree
on the device sampling route.

These are historical experiment results, not new measurements of the unified
branch. The old small quality/path checks do not establish that INT8 is lossless
or that the later full numeric, cold-cache, long-decode and quality gates passed.

## Measured rates

Each number is the median of three warm HTTP rounds, in output tokens/second.
Every request forced 120 output tokens; C4 is the aggregate of four concurrent
requests. The timing includes prefill, decode and client overhead, but excludes
compilation. Model-specific chat prompts contain 19, 36, 12 and 17 input tokens,
respectively. Comparisons across models are not equal-prompt quality benchmarks.

| Model | BF16 single | INT8 single | Ratio | BF16 C4 | INT8 C4 | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 29.7014 | 45.3709 | 1.53× | 115.2669 | 176.4441 | 1.53× |
| Qwen2.5-7B-Instruct | 23.3669 | 48.0433 | 2.06× | 91.4602 | 184.9556 | 2.02× |
| DeepSeek-R1-Distill-Qwen-7B | 24.1490 | 50.5899 | 2.09× | 94.1531 | 196.2946 | 2.08× |
| Mistral-7B-Instruct-v0.3 | 32.5331 | 53.2569 | 1.64× | 126.7001 | 205.6425 | 1.62× |

The denominator is each model's historical BF16 vLLM-Neuron path, and the
numerator is its INT8 path at the same concurrency. Higher-batch, two-replica
and fast CPU sampler experiments are separate comparisons. No statistical
significance or universal speedup is claimed.

## What changed and what failed

The quantization utility loads official models as BF16, invokes the SDK's
per-channel symmetric INT8 conversion and leaves `lm_head` unconverted. The
Qwen2.5 and DeepSeek INT8 runs additionally used a local compatibility adapter
for unpacked Q/K/V bias tensors. The adapter is preserved in the archive for
traceability; it is not enabled by the unified package or recommended as a
general SDK patch. Qwen3 and Mistral's measured cases did not select it.

The [old quality review](evidence/dense-int8-20261030/quality-review.json)
retains limitations and errors: Mistral's generated `list(set(values))` did not
preserve first-occurrence order in either precision; a Qwen2.5 INT8 example
contained a comment with a value absent from its input. Several explanations
hit their output budget. The finite objective checks and static code reviews
cannot support a broad accuracy conclusion, and generated code was not executed.

The source metadata retains the exact official revisions:

| Model | Official revision |
|---|---|
| Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | `916b56a44061fd5cd7d6a8fb632557ed4f724f60` |
| mistralai/Mistral-7B-Instruct-v0.3 | `c170c708c41dac9275d15a8fff4eca08d52bab71` |

## Evidence and reproduction limits

The [archive manifest](evidence/dense-int8-20261030/manifest.json) binds 33 original
files: eight raw HTTP reports, eight launch records, eight saved configurations,
the original result/catalog/quality/verification records and four archived source
files. The [2026-11-07 metadata re-audit](evidence/dense-int8-20261030/reaudit.json)
recomputed all 48 nonstreaming round rates and checked all 120 nonstreaming
request usage records. Median rates and speedup ratios are unchanged. Streaming
records are preserved but were not included in this new arithmetic audit.

Some early HTTP records omit input token IDs. Equal recorded prompt lengths
and the benchmark's shared prompt construction do not fill in missing per-request
ID evidence. The archive authenticates saved files, not current compiled bytes
or historical process ownership. The historical `passed` fields refer to the
checks available at that time.

Weights, compiled binaries and environment credentials are not bundled. Three
large official checkpoints were previously removed for storage; reproducing
those cases requires downloading the pinned revisions and regenerating INT8
outputs. The archive's paths and scripts describe the historical machine; it is
an evidence package, not a standalone installer or qualified serving recipe.
Any new branch run must use fresh admitted artifacts and complete the current
numeric, cold-cache, long-decode, request-slot reuse, quality and serving gates.
