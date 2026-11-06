# Experimental LFM3 v4 integration

This branch prepares the historical LFM3-1.7B v4 predicated decode writer for repository review. It is **not yet repository-qualified**: only standard-library source/admission/mock checks have run. No native import, tokenizer execution, compilation, installation, device test or workload was run for this migration.

The integration is disabled unless `NEURON_LFM3=1`. Its reference variant is the default; `NEURON_LFM3_NKI_WRITER=1` selects the historical multi-row decode writer. The one-row prefill writer is preserved. No later typed-writer pilot is included. The kernel retains capacity/rows ≤4 and width divisible by 128.

## Scope and dependencies

Only original BF16 `LiquidAI/LFM3-1.7B` revision `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7`, Trn1, TP2, decode batch4, one-row prefill and context2048 are admitted. CPU sampling and `apply_seq_ids_mask=true` are required; prefix caching, quantization, LoRA, other parallel modes, speculative decoding and chunked prefill are rejected.

| Component | Historical environment to reproduce |
|---|---|
| Python | 3.10 |
| vLLM | 0.11.0 |
| Torch | 2.8.0 |
| torch-neuronx | 2.8.0.2.12.22436+0f1dac25 |
| neuronx-distributed | 0.17.26814+4b18de63 |
| neuronx-distributed-inference | 0.8.16251+f3ca5575 |
| neuronx-cc | 2.23.6484.0+3b612583 |
| Transformers | 4.57.6 |

The broad repository requirements do not replace this matrix. The package fingerprints relevant installed SDK sources and NKI modules through distribution metadata, without hard-coded environment or `.so` paths. This review did not import or hash those binaries.

The historical saved root `neuron_config.buckets` was `[2048]` (SHA256 `243df3eba317cf196efcde6c5eee47da39c1edaf13adb4ccab9d022a3751a0ba`, pinned by the v4 resume plan). This is **not a minimum decode floor**: the historical cold checks observed decode buckets 128, 256, 512, 1024 and 2048. The package keeps LFM's original NxDI routing and does not install Smol's `get_target_bucket` floor override.

## Prepare a fresh view

The following are future validation commands, not records of execution. Fetch the pinned model separately under its own license. Do not enable remote Python execution. Run preparation outside any timed/shared benchmark window:

```bash
python -m vllm_neuron.models.lfm3.prepare_view /data/LFM3-1.7B /data/LFM3-1.7B-lfm3-view-v1
```

Preparation refuses an existing destination, verifies pinned Hub Git-blob/LFS identities and streams the checkpoint hashes. It translates only `TokenizersBackend` to `PreTrainedTokenizerFast` and maps named special-token fields into TF4 `extra_special_tokens`. Before writing a completion manifest it requires all original 6 HF prompt inputs, 96 quality inputs, special-token IDs and recorded decode strings to match. Token strings and model configuration are not edited. A failed partial directory is retained and is not admitted.

## Matched reference and candidate

Use the same pinned package checkout, input fixtures, sampling and resource limits for both variants. Each variant requires a new, distinct artifact directory. The existing repository CPU sampler is enabled by `NEURON_CPU_SAMPLER_FAST=1`; the historical experiment used `TRN1_CPU_SAMPLER_FAST=1` and a different global patch. That patch is not copied here, and packaged sampler equivalence remains a required paired test.

Reference, initial explicit build:

```bash
NEURON_LFM3=1 NEURON_LFM3_NKI_WRITER=0 NEURON_CPU_SAMPLER_FAST=1 \
NEURON_LFM3_ALLOW_COMPILE=1 NEURON_COMPILED_ARTIFACTS=/data/artifacts/lfm3-reference-v1 \
NEURON_RT_VISIBLE_CORES=0-1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
vllm serve /data/LFM3-1.7B-lfm3-view-v1 --served-model-name lfm3-1.7b \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 2048 --max-num-seqs 4 \
  --no-enable-prefix-caching --host 127.0.0.1 --port 8010 \
  --additional-config '{"override_neuron_config":{"on_device_sampling_config":null,"apply_seq_ids_mask":true}}'
```

Stop and independently confirm the reference service and descendants are drained before the candidate:

```bash
NEURON_LFM3=1 NEURON_LFM3_NKI_WRITER=1 NEURON_CPU_SAMPLER_FAST=1 \
NEURON_LFM3_ALLOW_COMPILE=1 NEURON_COMPILED_ARTIFACTS=/data/artifacts/lfm3-nki-v4-v1 \
NEURON_RT_VISIBLE_CORES=0-1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
vllm serve /data/LFM3-1.7B-lfm3-view-v1 --served-model-name lfm3-1.7b \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 2048 --max-num-seqs 4 \
  --no-enable-prefix-caching --host 127.0.0.1 --port 8010 \
  --additional-config '{"override_neuron_config":{"on_device_sampling_config":null,"apply_seq_ids_mask":true}}'
```

For reuse set `NEURON_LFM3_ALLOW_COMPILE=0`. The default is off. Existing missing/incomplete/mismatched manifests, changed source/config/model/software/variant, and artifact load failures reject without deleting or recompiling. Compilation publishes a completion manifest only after load succeeds. Revert to the reference by restarting with writer `0` and its separately admitted reference artifact; variants cannot switch within one process. Historical artifacts lack the new identity and cannot be imported as qualified package artifacts.

The historical services used reference MemoryMax/High 22/20 GiB and candidate 26/24 GiB, 8 GiB swap, and startup bounds 3600/5400s. Those are reproduction context, not an implemented service manager. This package adds no service controller and does not arrange benchmark scheduling.

## Historical evidence and pending repository gates

[Historical evidence](evidence/lfm3-20261031.json) records the original 2026-10-31 experiment. Original numeric32, cold96, long1529/84 and quality96 passed; both variants scored 63/96 with 96 identical texts. In three sequential rounds, historical short_decode was 29.783→61.292 tok/s single request and 104.136→188.828 tok/s at four concurrent requests. These are **historical experiment results, not a retest of this branch**. Measurements were sequential, not randomized, and do not establish statistical significance or whole-family optimization completion.

`lfm3-nki-predicated-slot-resume/qualification-audit.json` has `performance_qualified=false` because it preceded workload verification. The later `lfm3-nki-predicated-slot-full/paired-comparison.json` completes that historical audit; the two reports are not conflicting repository results.

The original numerical gates require finite outputs, cosine ≥ 0.995, KL ≤ 0.02, max absolute error ≤ 2, exact top1 when reference margin > 0.5, and aggregate argmax match ≥ 90%. They remain the thresholds for all new numeric/cold/long validation.

Before admitting this package:

- Review the source diff, registry collisions, spawn registration and scope/artifact rejection checks.
- Re-run original CPU1648 cases, 9 device graphs/43 checks and 12 persistent-alias graphs/60 checks.
- Validate actual tokenizer equality, numeric32, independent cold96, long1529 steps/84 checkpoints and cache reuse.
- Re-run 96 quality cases with exact input IDs, per-case outcomes and text comparison.
- Run five workloads×three matched sequential rounds; retain raw counts/times and audit the packaged fast sampler on both sides.
- Keep cold compilation separate from warm serving; do not reuse old controller/resume scripts or overwrite historical evidence.

The historical runner names and checks are recorded for rule migration; parameterized runtime validation drivers remain a separate review task. None is silently run by installing or importing this package.

## Source attribution

The port derives from `modeling_lfm3.py`, `configuration_lfm3.py` and `lfm3_conv.py` at the pinned model revision: Copyright 2026 The HuggingFace Team, Apache-2.0. Their exact official hashes and the original v4 experiment hashes are retained in `vllm_neuron/models/lfm3/provenance.py`. Only the functional slot writer, partitioned-parameter helper and tied-weight/one-token-prefill route are extracted from the Qwen experiment; no Qwen architecture is imported. The model weights have the separate LFM Open License 1.0 and are not bundled.

The tokenizer fixture includes 64 selected GSM8K prompts, Copyright (c) 2021 OpenAI, under the MIT License. The [complete license and copyright notice](../vllm_neuron/models/lfm3/licenses/GSM8K-MIT.txt) is included in the source and wheel distributions. Its bytes match the [upstream license at the fixture's pinned revision](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE), SHA256 `86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b`. The fixture retains its source revision and selection; no input text or token IDs were changed.
