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

This is an experimental integration, not a newly qualified model release.
The branch has focused tests for the actual compact probability reduction,
eligibility, fallback behavior, explicit generators, and per-instance wiring.
Run them in the compatible environment:

```bash
VLLM_PLUGINS='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest test/cpu_sampler -q
```

On the shared Trn1 host, native CPU tests use the same serial queue as existing
experiments. Unexecuted tests are not recorded as passed. There is no
branch-specific end-to-end throughput result yet.

Earlier experimental evidence motivates this port:

| Experiment | Recorded result | Limit |
|---|---|---|
| Granite BF16/INT8 captured-logit replay | 335 fixtures, 563 rows; 555 bounded-path rows; fallback cases exercised | The old probability check reconstructed a full-vocabulary vector before softmax. It does not prove the compact softmax has zero rounding error. |
| Granite BF16 greedy quality | 78/96 before and after, all 96 texts identical | Fixed suite and experimental implementation |
| Granite INT8 greedy quality | 77/96 before and after, all 96 texts identical | Fixed suite and experimental implementation |
| LFM3 sampler plus debug-log optimization | Short stochastic median 21.982 → 29.530 tokens/s for one request; 47.132 → 98.559 tokens/s at concurrency four | Combined change, three sequential rounds; not a sampler-only or branch-specific speed claim |

Recorded sources:

- `cpu_bounded_sampler.py`: SHA-256 `6ec785c4d29c5cd633d96bede33fa52476f75f427be69c2ee428a26567c6f80f`
- `cpu_debug_guard.py`: SHA-256 `a02400eaa224d73a698396fd5321af20c54b73cdcf4e5021c8bbc17e3c4b73ab`
- `cpu-bounded-sampler-both-precisions-validation.json`: SHA-256 `60487f4556c4478b0e946a735cd43fb177614bf232529a3f9f14cbf2947ae624`
- `lfm3-optimization-retry/paired-comparison.json`: SHA-256 `061f10abda61c0a79afdb8d74ec0da0a2bd9e5cdfc4337342038444d4ffaa529`

Original reports remain in the experiment workspace. Model weights, compiled
artifacts, logs, credentials, and private workspace paths are not included here.

## Further integration

The LFM3 NKI cache writer is the next device-side candidate. Its existing
BF16/TP2/batch4/context2048 experiment passed the original numerical, cold-cache,
long-decode, and paired quality checks. Both sides of that cache comparison used
the same fast CPU sampler. Porting its model adapter and NKI dependencies still
requires fresh qualification in this branch.

SmolLM3, Phi-4, Qwen3.5, MiniCPM5 and other ongoing work remain separate until
their original gates pass. New integrations retain full-vocabulary numerical
checks, cold caches, long decoding, request-slot reuse, quality evaluation, and
matched workloads. A local operator test does not qualify a complete model or
establish a speedup.
