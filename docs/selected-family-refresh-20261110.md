# 所选小模型家族的官方信息复核（2026-11-10）

此次复核覆盖既定 15 个家族、25 个仓库，以及它们所属 14 个官方命名空间中各最新 30 个仓库。25 个仓库接口和 14 个命名空间查询全部成功。这次没有下载权重、执行外部模型代码或切换实验版本。

## 近期改动是否需要改变实验

对 LFM3、MiniCPM5 和 Spark-X2.6-4B，按既有研究版本与当前版本获取了完整递归文件树，均无后续分页。逐文件比较 Git 对象标识、LFS 摘要和大小，权重与运行所需文件的元数据均未变化。

| 模型 | 比较版本 | 发生变化的文件 | 对当前实验的影响 |
|---|---|---|---|
| LFM3-1.7B | ac2586a → 1da06be | README、IFBench 评测记录、4 个导航图标 | 1 份权重和 9 份配置／分词器／代码类文件元数据未变，继续固定版本 |
| MiniCPM5-2B | f974000 → 42433e9 | README；提交说明为更新文档链接 | 1 份权重和 7 份运行相关文件元数据未变，继续当前采集 |
| Spark-X2.6-4B | aa074aa → 64ca8f2 | README、推理力度示意图 | 5 份权重和 10 份运行相关文件元数据未变，仍需原生参考与 Neuron 基线 |

这里的“未变”是托管方文件树元数据的比较结果，没有重新读取权重字节做独立核验。仓库创建时间和最后修改时间不等同于模型发布日期。

14 个命名空间的有限查询中，11 月 4 日后创建的仓库记录只有两个 allenai 机器人模型和 NVIDIA MetricTrail；本轮没有找到可直接替代既定语言模型实验的新热门候选。查询上限为每个命名空间 30 个仓库，不能据此断言全网没有其他新发布模型。保持原 15 家族目标和串行实验顺序。

## 25 个仓库的规模和热度

参数量取自 Hugging Face API 的 safetensors 总参数字段，属于托管方报告，未重新计数权重。所有 25 项均低于 10B。下载和点赞是查询时的 API 计数，不是本机验证或性能成绩；同家族的多个尺寸仍需分别完成适配与验收。

| 模型仓库 | API 报告的总参数（B） | 下载计数 | 点赞 |
|---|---:|---:|---:|
| [ibm-granite/granite-4.2-3b](https://huggingface.co/ibm-granite/granite-4.2-3b) | 3.660 | 37,414 | 132 |
| [ibm-granite/granite-4.2-8b](https://huggingface.co/ibm-granite/granite-4.2-8b) | 8.792 | 59,763 | 117 |
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2.274 | 3,781,825 | 475 |
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4.660 | 5,236,460 | 1,213 |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9.653 | 6,370,574 | 2,341 |
| [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | 5.123 | 3,113,008 | 1,122 |
| [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | 7.996 | 4,255,026 | 1,846 |
| [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) | 1.729 | 78,750 | 1,580 |
| [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | 3.849 | 66,175 | 317 |
| [mistralai/Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) | 8.918 | 93,502 | 238 |
| [mistralai/Ministral-3-3B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) | 3.849 | 48,212 | 49 |
| [mistralai/Ministral-3-8B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16) | 8.918 | 41,781 | 42 |
| [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | 3.836 | 329,199 | 886 |
| [microsoft/Phi-4-mini-reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning) | 3.836 | 36,480 | 252 |
| [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3.075 | 341,164 | 1,077 |
| [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4.022 | 2,553,310 | 1,049 |
| [deepseek-ai/DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) | 8.191 | 389,929 | 1,127 |
| [Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) | 4.170 | 112,574 | 858 |
| [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16) | 7.996 | 24,719 | 67 |
| [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) | 8.888 | 100,332 | 532 |
| [allenai/Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) | 7.298 | 131,687 | 159 |
| [EssentialAI/rnj-1.5-instruct](https://huggingface.co/EssentialAI/rnj-1.5-instruct) | 8.311 | 1,302 | 21 |
| [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | 2.517 | 1,055,655 | 1,843 |
| [nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) | 3.975 | 13,143 | 246 |
| [XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) | 4.112 | 24,979 | 1,032 |

模型名称中的 2B／4B 不总等于完整权重参数量。例如 Gemma 4 E2B 约 5.123B、E4B 和 Nemotron 3.5 Bolt E4B 约 7.996B，Nanbeige4.2-3B 约 4.170B，Qwen3.5-9B 约 9.653B。选型继续按完整参数规模和实际存储／内存条件判断。

当前优先完成 MiniCPM5 的数值差异定位、Nemotron 的整模型验收恢复，以及已选家族的有效基线和配对性能验证。优化不能以失败对照作为提速分母，也不能以单算子通过、API 热度或仓库版本更新代替整模型正确性。

证据：[全部元数据及来源清单](evidence/selected-family-refresh-20261110/manifest.json)、[25 仓库汇总](evidence/selected-family-refresh-20261110/selected-family-metadata-refresh-20261110-v1/summary.json)、[逐文件版本比较](evidence/selected-family-refresh-20261110/selected-family-revision-delta-20261110-v1/comparison.json)、[14 命名空间查询](evidence/selected-family-refresh-20261110/selected-family-author-refresh-20261110-v1/summary.json)。
