# Sampler class import for vLLM-Neuron 0.5.3

`0001-import-sampler-class.patch` fixes the `NeuronCausalLM` constructor when
`NEURON_ON_DEVICE_SAMPLING_DISABLED=1`: the original import resolves to a module,
which cannot be called. The patch imports the class from that module.

Target: upstream commit `f9ba933e9591e3cef163740516b1977aa1100493`, with the
vLLM 0.16 sample package layout. Apply in a separate checkout of that revision:

```bash
git apply --check /path/to/0001-import-sampler-class.patch
git apply /path/to/0001-import-sampler-class.patch
```

The current unified experimental branch uses an older stack and already has the
correct class import. This patch is not applied automatically or installed into
the active benchmark environment.

Run the six isolated source regression tests from this repository:

```bash
python3 -B docs/evidence/sampler-and-runtime-reviews-20261112/sampler/check_vllm053_sampler_class_import_v1.py
```

The tests execute complete original/candidate loader modules with dependency
stubs. They reproduce the old exception, verify the corrected constructor and
check default/explicit device sampling branches. They do not execute Torch,
Neuron, a model or HTTP serving. The normal runner's CPU sampling route has a
correct import; this finding does not establish the cause of the Qwen r6 API
startup failure. See the [full report](../../docs/sampler-and-runtime-reviews-20261112.md).
