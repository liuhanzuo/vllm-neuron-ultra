# Trn1 optimization branch

This branch packages the Trn1 experiments into the vLLM Neuron plugin. It is
based on upstream commit `ad82965376f1eab5dcfecb6b1bae99ae8e262a8d`
(`0.2.2+lts`), with **Neuron SDK 2.27 and vLLM 0.11.0**. The repository's
`main` branch uses a different backend and compatibility matrix. Install this
branch in a separate environment from active benchmarks.

The first change includes an optional CPU top-k/top-p sampler and avoids
calculating diagnostic tensor statistics when DEBUG logging is disabled.
Model weights, device graphs, KV caches, and the standard greedy sampler are
unchanged. The fast sampler is disabled by default.

## Usage

In an environment with the matching Neuron SDK and vLLM versions:

```bash
git clone --branch trn1-sdk227-optimizations \
  https://github.com/liuhanzuo/vllm-neuron-ultra.git
cd vllm-neuron-ultra
pip install --no-deps -e .

NEURON_CPU_SAMPLER_FAST=1 vllm serve /path/to/model \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --additional-config '{"override_neuron_config":{"on_device_sampling_config":null}}'
```

The variable only affects CPU sampling. Remove it or set it to `0` to use the
original sampler. Requests must specify a supported `top_k` (1–64) to use the
bounded candidate path. Greedy requests use vLLM's original greedy implementation.

The candidate list retains ties at the top-k threshold. Unsupported inputs,
processed-logit modes, explicit per-request generators, excessive ties,
nonfinite values, nucleus-boundary ties, and near-boundary cumulative
probabilities use the existing sampler. The replacement is local to each
Neuron sampler instance; it does not patch vLLM classes globally.

Unseeded stochastic requests consume random values differently on the fast
path. Identical generated text between implementations is not promised.
Explicit per-request generators use the original whole-batch implementation.
This does not add HTTP seed support to a request builder that lacks it.

## Validation status and evidence

The fast CPU sampler is an experimental integration; its tests do not qualify a model release.
The branch has focused tests for the actual compact probability reduction,
eligibility, fallback behavior, explicit generators, and per-instance wiring.
On 2026-11-04, commit `69b498f20b0f2ad598396683e3ade35b19edb661` passed
all **64 CPU runtime tests**, with zero failures, errors, or skips. The run
included captured-logit replay from Granite BF16 and INT8 models, **32 frames
each**. The fast sampler remains opt-in and disabled by default. This CPU-only
run did not execute a model or device, qualify an end-to-end configuration, or
measure a sampler speedup.

Run the tests in the compatible environment, supplying local capture files to
include the replay test:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS='' PYTEST_PLUGINS='' \
  VLLM_PLUGINS='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NEURON_SAMPLER_CAPTURE_PATHS='/path/to/bf16-captures.pt:/path/to/int8-captures.pt' \
  python -m pytest test/cpu_sampler -q
```

Without `NEURON_SAMPLER_CAPTURE_PATHS`, the optional replay test is skipped.
On the shared Trn1 host, native CPU tests use the same serial queue as existing
experiments. There is no branch-specific end-to-end throughput result for the fast CPU sampler yet.

Earlier experimental evidence motivates this port:

| Experiment | Recorded result | Limit |
|---|---|---|
| Granite BF16/INT8 captured-logit replay | 335 fixtures, 563 rows; 555 bounded-path rows; fallback cases exercised | The old probability check reconstructed a full-vocabulary vector before softmax. It does not prove the compact softmax has zero rounding error. |
| Granite BF16 greedy quality | 78/96 before and after, all 96 texts identical | Fixed suite and experimental implementation |
| Granite INT8 greedy quality | 77/96 before and after, all 96 texts identical | Fixed suite and experimental implementation |
| LFM3 sampler plus debug-log optimization | Short stochastic median 21.982 → 29.530 tokens/s for one request; 47.132 → 98.559 tokens/s at concurrency four | Combined change, three sequential rounds; not a sampler-only or branch-specific speed claim |

Recorded sources:

- `ultra-sampler-runtime-tests-terminal-20261104-v1.json`: SHA-256 `bd50c8be8e0cb0a5aa36fadc6db4de16e6673386a558d0b4ab78c2ce002df2dd`
- `cpu_bounded_sampler.py`: SHA-256 `6ec785c4d29c5cd633d96bede33fa52476f75f427be69c2ee428a26567c6f80f`
- `cpu_debug_guard.py`: SHA-256 `a02400eaa224d73a698396fd5321af20c54b73cdcf4e5021c8bbc17e3c4b73ab`
- `cpu-bounded-sampler-both-precisions-validation.json`: SHA-256 `60487f4556c4478b0e946a735cd43fb177614bf232529a3f9f14cbf2947ae624`
- `lfm3-optimization-retry/paired-comparison.json`: SHA-256 `061f10abda61c0a79afdb8d74ec0da0a2bd9e5cdfc4337342038444d4ffaa529`

Original reports remain in the experiment workspace. Model weights, compiled
artifacts, logs, and credentials are not included here. The separate SmolLM3
terminal audit records local evidence paths alongside hashes for traceability.

## Further integration

The LFM3 model adapter and NKI cache writer have now passed a fresh packaged
BF16/TP2/batch4/context2048 comparison at implementation `822e1bd`, including
the original numerical, cold-cache, long-decode, slot-reuse and quality checks.
Both sides used the same fast CPU sampler; see the
[paired results](lfm3-results-20261107.md). The [unified branch](multi-model.md)
combines this package with SmolLM3 and Nemotron, and still needs fresh runtime
qualification of its merged source identities.

An opt-in [SmolLM3 integration](smollm3.md) now packages the model adapter and
combined NKI cache candidate. Its fresh packaged comparison passed the defined
BF16/TP2/B4/context2048/floor2048 protocol; see the [measured results and
reference limits](smollm3-results-20261105.md). Natural-bucket and historical
full-causal reference failures remain unresolved.

Phi-4, Qwen3.5, MiniCPM5 and other ongoing work remain separate until
their original gates pass. New integrations retain full-vocabulary numerical
checks, cold caches, long decoding, request-slot reuse, quality evaluation, and
matched workloads. A local operator test does not qualify a complete model or
establish a speedup.
