# Unified Trn1 experimental branch

`trn1-multi-model-experimental-20261107` combines the LFM3, SmolLM3 and
Nemotron Nano 9B v2 packages in one vLLM-Neuron checkout. All three general
plugins are installed, and each remains disabled until explicitly selected.
Their artifact loaders run before generic artifact deletion or recompilation,
so an invalid saved artifact continues to fail instead of being silently rebuilt.

The exact `e0a01d3` wheel passed [nine native installation/plugin/spawn/configuration checks](unified-native-checks-20261108.md) on 2026-11-08, after fixing a JSON-input bug in the probe. Fresh device and end-to-end qualification
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
| Nemotron `60ab7f54a2c54ef0bc9b498621a45719ee7f0e84` | — | — | — | [Input/environment capture passed](nemotron-input-capture-20261108.md); no matched packaged speedup; historical 13.18 / 51.06 tok/s are absolute rates |

The model package trees are preserved from these source branches:

- LFM3: `lfm3-paired-results-20261107`, commit `fe398c2a93b89f18328d0db7f2a6f655d8a6936b`.
- SmolLM3: `smollm3-ablation-results-20261107`, commit `e3dc962`.
- Nemotron: `nemotron9b-experimental-v1`, commit `60ab7f54a2c54ef0bc9b498621a45719ee7f0e84`.

The repository also preserves the [historical four-model INT8 comparison](dense-int8-results-20261030.md)
for Qwen3-8B, Qwen2.5-7B, DeepSeek-R1-Distill-Qwen-7B and Mistral-7B: C4 gains
of 1.53×, 2.02×, 2.08× and 1.62× respectively. These used device greedy sampling
and an earlier small validation suite. They are not new merged-branch results
or evidence of lossless quantization.

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
results in [Trn1 optimizations](trn1-optimizations.md). Native plugin/spawn/configuration checks for the exact wheel are recorded above. Before promoting the
merged branch, complete compile/load and corruption
rejection, original numeric gates, cold cache, long decode, request-slot reuse,
quality and matched serving measurements for each model. No model family is
declared fully optimized.

最新实测进展： [MiniCPM5 第 1 帧完成 15 个张量重放和独立数据复核](minicpm5-frame1-diagnostic-20261108.md)，固定输入的 down 路径差异与实际 MLP 输入差异已分别量化；整模型仍未合格。Nemotron r5 的失败与回收见 [结束记录](nemotron-r5-compile-interruption-20261108.md)；[v14 编译资源链接修复](nemotron-v14-compiler-links-20261108.md)已通过 94 项集成检查和 13 项配置检查，[r6 已因遗漏 DVE 资源链接而在编译阶段失败并释放；Dory 的 192 次位置编码比较及运行后完整数据复核通过](dory-r1-nemotron-r6-terminal-20261108.md)。整模型验收仍未完成。模型提速倍数仍以已合格的配对性能结果为准。

[2026-11-08 小模型候选复查](recent-small-models-20261108.md)记录官方仓库更新、草稿模型排除依据与原精度候选的准备状态，保留全部 15 个模型家族。

最新局部实测：[MiniCPM5 完整有效前缀的注意力输入一致，差异从注意力输出出现；固定 CPU 布局替换后恢复原生残差/归一化，独立字节复核通过](minicpm5-upstream-result-20261108.md)。没有新增整模型候选或提速倍数。

最新运行器进展：[Nemotron v15 核验完整 PWP/DVE 资源，113 项检查通过，已发起 r7 复验](nemotron-v15-compiler-resources-20261108.md)。原模型实现、52 阶段和提速表保持原有结论，尚无新整模型合格结果。

最新诊断准备：[MiniCPM5 注意力内部采集、完整结果复核、串行运行、配置生成、结束核验及退出后独立复核源码通过 144 项检查](minicpm5-attention-source-20261108.md)，尚未执行新的模型采集。Nemotron r7 继续串行运行，提速表没有新增结果。

[Nemotron r7 投影与卷积拆分对照](nemotron-r7-split-progress-20261108.md)：32 个局部对照阶段全部完成，累计核对 42/52 个工作阶段。16 组候选的 96 次完整设备输出比较全部逐字节一致；原写法 14 组多行输入失败、2 组单行对照通过，CPU 两种写法均正确。随后进入整模型编译。报告保留具体地址差异、环境与范围；整轮独立原始数据复核和完整模型验收仍待完成，没有新提速倍数。

[Dory 原权重参考加载准备](dory-metadata-and-optimization-20261108.md)：原加载器已接到完整参考候选，15 项源码/合成存储检查通过；新增存储共享与缓冲寿命探针，22 项源码/标准库检查通过，实际 PyTorch 探针尚未运行。保留原 507 个 BF16 物理张量、两次循环与 56 份独立历史。原权重实际加载和整模型验证尚未执行，位置编码 192 次局部比较的既有结果已同步修正到概览。
