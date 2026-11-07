# Nemotron Nano 9B v2: experimental Trn1 package

**Source migration only: this package has no fresh device, full-model,
end-to-end or performance qualification.** The explicit opt-in is intended for
validation work. It does not certify production readiness. Historical rates
are documented separately in [the historical record](nemotron9b-results-historical-20261103.md).

## Exact scope

The package targets `nvidia/NVIDIA-Nemotron-Nano-9B-v2` at revision
`6533e8de2c68e4536bf7c411d7a3ce5734111476`: original BF16 weights, FP32 Mamba
SSM state, BF16 convolution and attention KV state, Trn1 TP2, physical decode
batch 4, prefill batch 1, context 4096. The 56 layers contain 27 Mamba,
4 attention and 25 MLP blocks. Six prefill and six decode routes cover
128, 256, 512, 1024, 2048 and 4096 positions. The 124 buffers across two ranks
occupy 845,283,328 bytes; 54 are FP32.

The standard CPU sampler is required. Set `NEURON_CPU_SAMPLER_FAST=0` and
`XLA_DISABLE_FUNCTIONALIZATION=1`. Historical `TRN1_*` experiment switches are
rejected. The full Neuron configuration is pinned, including disabled secondary
kernels/layouts and numeric switches. Unknown, missing or changed fields are
rejected; this version does not support changing context, TP, dtype, batching,
quantization, LoRA, prefix caching, speculative decoding or sampling location.

The model's native CPU reference uses the disclosed Mamba timestep limit
`[0, +Infinity]`. This is not a claim of CUDA kernel numerical equivalence.
The prepared view preserves the original tokenizer and template: the original
template **ignores `enable_thinking=False` and still opens `<think>`**. This
package does not add `/no_think`. The explicit `/no_think` behavior is tested
as a tokenizer control only.

## What was moved

The explicit-state model math, packed TP sharding and NKI cache read/write
operations are carried over from the qualified experiment. Four admitted
last-dimension `split` sites become `tensor_split` through exact cumulative
boundaries: two in the original Mamba path and two in its NKI decode path.
Prefill still uses the original forward. No runtime AST rewriting is needed.

Small generic helpers and the original one-token prefill forwarding method
are extracted without importing or registering the Qwen architecture. The
historical custom vLLM flags `has_inner_state=False` and `is_hybrid=False`
are preserved. Opt-in registration checks collisions before modifying any
registry and also runs in spawned workers.

Normal serving retains the original state lifecycle. No request-level clear
was added. `state.clear_for_cold_validation` is an explicit diagnostic helper
for an already loaded model: it copies zeros into existing state tensors and
checks their binding and raw bytes. It is never called by serving.

## Pinned environment

Use a separate Python 3.10 environment or an isolated package path; do not
repoint a shared editable installation running other experiments. Runtime
admission requires these exact distribution versions and the SDK Python
source identities recorded in `provenance.py`:

| Distribution | Version |
|---|---|
| vLLM | 0.11.0 |
| Torch | 2.8.0 |
| torch-xla | 2.8.1 |
| torch-neuronx | 2.8.0.2.12.22436+0f1dac25 |
| neuronx-distributed | 0.17.26814+4b18de63 |
| neuronx-distributed-inference | 0.8.16251+f3ca5575 |
| neuronx-cc | 2.23.6484.0+3b612583 |
| Transformers | 4.57.6 |

The original native oracle used Torch 2.9.1+cpu / Transformers 5.23.0.dev0.
Changing that oracle requires separate reference qualification. The package
records runtime NKI binary identities when explicitly compiling/loading;
source tests do not import or hash native binaries.

## Preparation and serving interfaces

These are **not-yet-runtime-validated commands** for a separately scheduled
worker, after resource admission. Source tests can run with system Python:

```bash
python3 -B -m unittest discover -s test/nemotron9b -p 'test_*.py'
```

Obtain the original model from the pinned official revision under its model
license. Use ordinary fresh destination directories. Preparation verifies
every original file, including all four weight shards, then validates 96
chat inputs, 22 raw encode/decode cases, the complete vocabulary, special
tokens and both thinking controls. It does not execute model remote code.

```bash
export PYTHONPATH=/home/ubuntu/src/vllm-neuron-ultra-nemotron9b-v1
/home/ubuntu/venvs/vllm_neuron/bin/python -B \
  -m vllm_neuron.models.nemotron9b.prepare_view \
  /mnt/nvme/models/NVIDIA-Nemotron-Nano-9B-v2 \
  /mnt/nvme/models/nemotron9b-ultra-view-v1
```

The view references original payload files and writes only its transformed
config/tokenizer config. The original directory is not modified. Failed
preparation retains a partial directory without a completion manifest; use
a different fresh destination after diagnosing the cause.

The root controller must enforce exclusive model/compiler ownership and
memory, runtime and disk limits. The package serializes compiler submissions
within its process; it does not reserve the machine or arbitrate other jobs.
The historical 26 GiB memory / 12 GiB swap / 14,400-second compile envelope
is a planning reference, not a package peak-resource measurement.

```bash
export NEURON_NEMOTRON9B=1
export NEURON_NEMOTRON9B_ALLOW_COMPILE=1
export NEURON_CPU_SAMPLER_FAST=0
export NEURON_COMPILED_ARTIFACTS=/mnt/nvme/neuron-artifacts/nemotron9b-ultra-v1
export NEURON_COMPILE_CACHE_URL=/mnt/nvme/nemotron9b-ultra-v1-compile-cache
export NEURON_RT_VISIBLE_CORES=0-1
export XLA_DISABLE_FUNCTIONALIZATION=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
/home/ubuntu/venvs/vllm_neuron/bin/python -B \
  -m vllm_neuron.models.nemotron9b.serve \
  --model /mnt/nvme/models/nemotron9b-ultra-view-v1 \
  --served-model-name nemotron-nano-9b-v2 \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs 4 --no-enable-prefix-caching \
  --host 127.0.0.1 --port 8010 \
  --additional-config '{"override_neuron_config":{"on_device_sampling_config":null,"apply_seq_ids_mask":true,"context_encoding_buckets":[128,256,512,1024,2048,4096]}}'
```

After an admitted compile and load, set `NEURON_NEMOTRON9B_ALLOW_COMPILE=0`.
New artifacts require explicit compile opt-in and exclusive directory creation.
The manifest is published only after successful compile and load. Existing
partial, modified or mismatched artifacts are rejected without deletion,
recompilation or fallback. Full config, model, source, hardware, software and
artifact identities are bound. Historical artifacts cannot be imported or
relabeled as package artifacts. Admission hashes payloads only during explicit
preparation/load; do not run it in another experiment's measurement window.

## Required fresh runtime qualification

Passing source/mock tests proves source and control contracts only. Before any
package performance claim, an independently reviewed serial controller must:

1. Verify original source/model/tokenizer identities and the complete tokenizer
   fixture suite. Preserve the exact native reference identities and timestep
   disclosure.
2. Run the original NKI read/write and alias protocol: six composition graphs,
   46 checks, plus 32 Nemotron split control/candidate cases. Failed original
   cache variants are not valid speed denominators.
3. Compile/load all 12 fresh package routes; verify source/route witnesses and
   all 32 numerical frames against the original oracle. Also compare to the
   historical candidate as a migration check.
4. Freshly execute the two-pattern state-copy proof: 124 buffers, 248 fills,
   248 zeros, unchanged bindings. Then run cold prefixes 32/128/256/512/1024/
   2048/3070 in one loaded process, 84 frames and 868 zero-buffer checks, with
   all 12 routes exercised. Do not inherit the old v2 copy-proof exception.
5. Run 2,041 long-decode steps / 93 checkpoints with original teacher-forced
   tokens and cached oracle, including permuted, reused and inactive slots and
   partial batches. Retain finite outputs, cosine >= 0.995, KL <= 0.02,
   maximum absolute error <= 2, top-1 >= 90% and exact high-margin top-1 checks.
6. Run identical 96 quality prompts, budgets and template; compare outputs and
   scoring decisions. Historical quality was 55/96, with 41 length-capped.
7. Run smoke plus five workloads × three rounds including streaming, concurrent
   and mixed-length requests. Retain counts, elapsed time, TTFT and reuse
   records; physically drain all workers before the terminal audit.

The ultra base also avoids expensive DEBUG logits statistics even with the
fast sampler disabled. Therefore historical runner rates cannot be transferred
to this package. Report fresh absolute performance after the gates; speedup
requires a separately qualified matched baseline.

## Licensing and provenance

Repository source uses Apache-2.0. Model weights remain subject to the
[official model license](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/tree/6533e8de2c68e4536bf7c411d7a3ce5734111476).
No weights, compiled binaries or native remote implementation are distributed.
Tokenizer fixture prompts include GSM8K material; its MIT license is retained
in `vllm_neuron/models/nemotron9b/licenses/GSM8K-MIT.txt`. Source hashes,
arithmetic AST contracts and the historical evidence JSON retain attribution
and the experiment-to-package mapping.
