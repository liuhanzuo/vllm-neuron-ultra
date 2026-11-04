# Experimental SmolLM3 integration

The source integration adds original-weight SmolLM3-3B support and an optional
combined NKI decode-cache reader/writer. **This branch integration has not passed
native or full-model qualification.** NKI is disabled by default. There is no
new branch throughput result.

Initial scope is Trn1, original BF16 SmolLM3-3B revision
`a07cc9a04f16550a088caea529712d1d335b0ac1`, TP2, physical decode B4, prefill B1,
context2048 and minimum decode bucket2048. Original 36-layer RoPE/NoPE, tied
embeddings and per-rank `[4,2,2048,128]` caches are preserved. Prefix/block KV,
chunked prefill, LoRA, speculation and other precision/TP/batch/context routes
are unsupported. Keep `NEURON_CPU_SAMPLER_FAST=0` for initial qualification.

## Preparation and interface

Use an isolated installation and serialized build window. Preparation streams
SHA256 checks of original model/tokenizer files, including checkpoint shards,
without changing them. The destination must not exist. It contains the custom
Neuron config and tokenizer cleanup=False setting; remaining files are symlinks.

```bash
python -m vllm_neuron.models.smollm3.prepare_view /path/to/SmolLM3-3B /path/to/new-view
```

The initial software versions are pinned:

| Distribution | Version |
|---|---|
| vllm | 0.11.0 |
| torch | 2.8.0 |
| torch-neuronx | 2.8.0.2.12.22436+0f1dac25 |
| neuronx-distributed | 0.17.26814+4b18de63 |
| neuronx-distributed-inference | 0.8.16251+f3ca5575 |
| neuronx-cc | 2.23.6484.0+3b612583 |
| transformers | 4.57.6 |

Reference and NKI variants need distinct new artifact directories under an
existing parent. Compilation additionally requires `NEURON_SMOLLM3_ALLOW_COMPILE=1`;
remove it for cached startup. This interface still requires native validation:

```bash
export NEURON_SMOLLM3=1
export NEURON_SMOLLM3_NKI_CACHE=0
export NEURON_CPU_SAMPLER_FAST=0
export NEURON_SMOLLM3_ALLOW_COMPILE=1
export NEURON_COMPILED_ARTIFACTS=/path/to/new-reference-artifact
vllm serve /path/to/new-view --tensor-parallel-size 2 --max-num-seqs 4 --max-model-len 2048 --no-enable-prefix-caching --additional-config '{"override_neuron_config":{"on_device_sampling_config":null,"apply_seq_ids_mask":true}}'
```

For the candidate, set `NEURON_SMOLLM3_NKI_CACHE=1` and select another new artifact
directory. Both variants retain floor2048. General-plugin registration is a
no-op unless `NEURON_SMOLLM3=1`. When restricting `VLLM_PLUGINS`, include both
`neuron` (platform) and `neuron_smollm3` (general plugin).

Manifests bind variant, source files and relevant SDK sources, original model/tokenizer hashes, full
requested config, scope, exact software and relevant environment settings.
Startup verifies model and compiled bytes, including explicit artifact paths.
Existing incomplete/mismatched/corrupted directories are rejected before load.
The saved config body must exactly match the admitted requested config after
replacing only the embedded identity metadata with its original null value.
No other saved field is ignored, including unlisted or newly introduced fields.
The comparator preserves the SDK's positive-Infinity default only at
`neuron_config.quantize_clamp_bound`; other NaN/Infinity values are rejected.
Every value carries a type tag during comparison, so an ordinary string or
container cannot imitate that SDK sentinel, and booleans remain distinct from
numbers. Invalid configurations are rejected before model hashing or compiling.
Expert parallelism is fixed at1 and total world size at2.
This path never enters generic deletion/recompilation logic. The SDK compiler
submission pool is limited to one worker during the build and restored afterward. Failed builds stay
available for diagnosis; use another new path for a corrected build. No existing
artifact is automatically repaired or replaced.

To revert, stop the worker, select its reference artifact, set the NKI flag to0
and restart with fresh live KV state. In-request switching is unsupported.

## Evidence and remaining qualification

The frozen `smollm3-samefloor-full-v1` experiment combined NKI DMA cache reads and
predicated-copy writes, preserving prefill and remaining attention arithmetic.
It passed numeric32, cold84, long84 checkpoints, smoke, quality96 and matched
workloads. Its original runner measured about7.92–8.07× deterministic single and
7.32–7.85× concurrent throughput over three sequential rounds. There is no
read-only/write-only attribution or statistical significance claim.

Those ratios do not describe this branch: its DEBUG statistics handling already
differs, and packaged model/registration/admission require fresh checks. Old
small-bucket NaNs and other failed routes remain failed evidence.

SDK source identity includes live bucket generation, padding, TP layers,
collectives/casts and relevant configuration/build helpers. The first build
records the inspected source hashes; it does not prove those sources equal
every source of the old experiment. NKI/compiler distribution versions are
pinned, but this source-only change has not executed or qualified their native
binaries. Fresh qualification must freeze the actual complete environment.

The focused stdlib admission tests use tiny fake artifacts, without model/SDK
execution. They can be run with:

```bash
python -m unittest discover -s test/smollm3 -p 'test_*.py'
```

Before qualification, freeze the integrated commit and run both variants through
original numeric32 thresholds, seven independent cold starts/84 frames and the
complete1529-step trajectory/84 checkpoints. Retain all1537 forwards, including
slot reuse, decode after reuse, partial batch and inactive resume. Run smoke,
quality96, initial120 and five workloads×3 rounds. Verify actual decode mask2048
and cache alias order. Compare76 deterministic decoded texts/usage plus real
finalizer mutation rejection controls; report stochastic differences. Both sides
use the same branch runner/sampler. No automatic best-known promotion occurs.

## Source and license

Derived architecture code retains Hugging Face's Apache-2.0 copyright/license.
`vllm_neuron/models/smollm3/provenance.py` records pinned upstream and experiment
sources. The repository remains Apache-2.0. Model weights, compiled graphs and
SDK binaries are not bundled.
