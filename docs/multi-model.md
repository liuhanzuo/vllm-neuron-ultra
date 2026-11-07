# Unified Trn1 experimental branch

`trn1-multi-model-experimental-20261107` combines the LFM3, SmolLM3 and
Nemotron Nano 9B v2 packages in one vLLM-Neuron checkout. All three general
plugins are installed, and each remains disabled until explicitly selected.
Their artifact loaders run before generic artifact deletion or recompilation,
so an invalid saved artifact continues to fail instead of being silently rebuilt.

This is a source integration release. Fresh device and end-to-end qualification
of the merged branch is pending. Use the recorded source commits for reproducing
the existing measured comparisons; new merged source identities need new
artifacts and a new full comparison.

## Install and select a model

Use a separate environment matching Neuron SDK 2.27, vLLM 0.11.0 and the exact
dependencies in the selected model's guide. Do not install into a running
benchmark environment.

```bash
git clone --branch trn1-multi-model-experimental-20261107 \
  https://github.com/liuhanzuo/vllm-neuron-ultra.git
cd vllm-neuron-ultra
pip install --no-deps -e .
```

Then follow the model-specific preparation and serving instructions. Choose one
model per server process. If restricting `VLLM_PLUGINS`, include the `neuron`
platform plugin and the appropriate general plugin below.

| Model and guide | General plugin | Model opt-in | Optimized path |
|---|---|---|---|
| [LFM3-1.7B](lfm3.md) | `lfm3` | `NEURON_LFM3=1` | `NEURON_LFM3_NKI_WRITER=1`; paired reference and candidate both use `NEURON_CPU_SAMPLER_FAST=1` |
| [SmolLM3-3B](smollm3.md) | `neuron_smollm3` | `NEURON_SMOLLM3=1` | Combined `NEURON_SMOLLM3_NKI_CACHE=1`; standard sampler required |
| [Nemotron Nano 9B v2](nemotron9b.md) | `nemotron9b` | `NEURON_NEMOTRON9B=1` | Packaged cache path remains unqualified; standard sampler required |

Model views and compiled artifact directories are model- and variant-specific.
Follow each guide's explicit compile opt-in, fresh destination and cached-load
rules. The unified package does not convert old artifacts or enable unsupported
batch sizes, contexts, precision, prefix caching or other serving features.

## Existing results and provenance

These gains use matched reference adapters with the same weights and settings,
not unmodified upstream vLLM-Neuron. C4 means aggregate throughput for four
concurrent requests. These are warm HTTP workload results, not compile times.

| Model / recorded implementation | Short single / C4 | Long single / C4 | Mixed lengths | Status |
|---|---:|---:|---:|---|
| LFM3 `822e1bd826321dc078226c2da3f199e1d385146f` | 2.02× / 1.80× | 8.51× / 6.70× | 5.92× | [Paired full-model gates passed in the stated scope](lfm3-results-20261107.md) |
| SmolLM3 combined `e2d1e28b83411863acef5f12ff3db1f887866a76` | 8.16× / 8.05× | 8.00× / 7.51× | 5.50× | [Qualified at decode floor2048 against cached native reference](smollm3-results-20261105.md); natural-bucket and full-causal-reference failures remain |
| Nemotron `60ab7f54a2c54ef0bc9b498621a45719ee7f0e84` | — | — | — | No matched packaged speedup; historical 13.18 / 51.06 tok/s are absolute rates |

The model package trees are preserved from these source branches:

- LFM3: `lfm3-paired-results-20261107`, commit `fe398c2a93b89f18328d0db7f2a6f655d8a6936b`.
- SmolLM3: `smollm3-ablation-results-20261107`, commit `e3dc962`.
- Nemotron: `nemotron9b-experimental-v1`, commit `60ab7f54a2c54ef0bc9b498621a45719ee7f0e84`.

The included SmolLM3 successor adds experimental ablation variants. Its
[write-only candidate failed deterministic output equality](smollm3-ablation-20261107.md),
so its measured speed is not a qualified improvement. The older combined
result does not certify these successor sources. The ongoing observer on/off
diagnostic measures whether instrumentation changes outputs; it supplies no
new speedup result.

## Integration validation and remaining work

On 2026-11-07, all **100 standard-library tests passed**: 33 LFM3, 26 SmolLM3,
37 Nemotron and 4 shared integration tests. A PEP 517 isolated wheel build
succeeded with the declared build dependencies. All 49 model-package files,
including six JSON/license resources, matched the checkout byte for byte in
the wheel, and all three plugin entry points were present. Setuptools warned
about the two license directories' namespace discovery; their license files
were explicitly checked in the built wheel.

The integration checks cover all three installed entry points, disabled-plugin
imports, required package resources, model-specific loader selection and
propagation of artifact rejection before generic recompilation. Existing model
tests exercise tokenizer/config identity, source contracts, registration,
cache/state contracts and artifact handling using small inert fixtures.

```bash
python3 -B -m unittest discover -s test/lfm3 -p 'test_*.py'
python3 -B -m unittest discover -s test/smollm3 -p 'test_*.py'
python3 -B -m unittest discover -s test/nemotron9b -p 'test_*.py'
python3 -B -m unittest discover -s test/multi_model -p 'test_*.py'
```

These checks do not load native model libraries or execute device inference.
The CPU sampler's native tests are a separate serial task, with historical
results in [Trn1 optimizations](trn1-optimizations.md). Before promoting the
merged branch, repeat native plugin/spawn checks, compile/load and corruption
rejection, original numeric gates, cold cache, long decode, request-slot reuse,
quality and matched serving measurements for each model. No model family is
declared fully optimized.
