# 小模型候选更新与原生基线准备（2026-11-08）

本轮继续保留全部 15 个模型家族。对与 11 月 6 日相同的 19 个发布者，复查每家最新 16 个仓库，共 293 项；发现 22 个版本变化或新仓库，完成 42 次元数据、模型卡、配置或源码请求。检索范围有限，不能据此断言所有新模型都已覆盖。

| 候选 | 本轮核实情况 | 实验安排 |
| --- | --- | --- |
| [Tencent DSpine for Qwen3.8-27B](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16/blob/ea909c0f5fab8865f9e14600334f55f7ce47308f/README.md) | 11 月 7 日新建仓库，文件元数据约 2.026B；模型卡明确是五层草稿模型，运行依赖 Qwen3.8-27B | 不作为独立的 &lt;10B 基线；仅保留推测解码研究线索 |
| [Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B/blob/64ca8f2a295b8ceb2757b9d4749a6558c4759e5b/config.json) | 原 BF16 元数据 4,112,079,360 参数，约 2.30 万次下载、1,023 个赞；模型类和配置源码与之前已审阅版本逐字节相同 | 保留既有 Spark 家族与原实验版本；新权重和分词器是否相同尚未核验 |
| [Spark-X2.6-1.7B](https://huggingface.co/XHToken/Spark-X2.6-1.7B/blob/0e60633b345b19ba9f56ed1fda197acd271e8546/config.json) | 原 BF16 元数据 1,707,657,216 参数；28 层、8 个 Q 头、2 个 KV 头 | 可作为 Spark 家族的较小后续变体，需要自己的原权重参考和 Neuron 基线 |
| Nemotron Nano 4B | 119 项既有准备源码校验值和共享库文件身份仍与已审阅计划一致；原先文件头核验为 3,973,556,832 参数 | 原生依赖验证的完整启动器仍未完成，不能直接启动旧的半成品；之后还需完整权重、原生参考及设备验收 |

DSpine 的 2B 是附加草稿网络，并非独立语言模型的总规模。配置声明 `DFlashDraftModel`，官方启动命令同时加载 Qwen3.8-27B；后者的仓库元数据已达到 27,781,427,952 参数。即使草稿文件低于 10B，完整推理所需模型也超出当前筛选范围。模型卡所列的加速来自 SGLang、FlashAttention 3 和 GPU 相关实现，不能填入我们的 Trn1 实测速度表。

本轮也看到了 Qwen3.8-Flash-Next、Qwen3.8-27B 和 2.4T-A95B 的仓库更新。其参数元数据均超过 10B。FP8 不改变这一规模判定；INT4 文件中的整数张量元素数也不能当作原模型参数量。仓库修改时间不等于新模型发布日期。

Spark 的 4B 配置继续使用 36 层，滑动窗口注意力与全注意力按 3:1 排列，窗口 512、head dimension 256、按 head 的输出门控、GELU，以及两类注意力不同的 RoPE 参数。模型代码和配置不变，意味着本轮没有发现可以直接替换原生适配代码的新修复；现有滑动窗口边界、缓存状态、门控和位置编码验证仍然需要完成。官方 CUDA 或 Ascend 部署说明也不能证明安装的 vLLM-Neuron 已支持这个架构。

后续优化优先级继续围绕实际基线：先完成当前 Nemotron9B 的完整复验，再进行 MiniCPM5 长提示首帧的 MLP 误差定位。对 Spark，可在原生与 Neuron 输出一致后评估滑动窗口缓存访问、门控计算融合、GELU/投影及分桶开销；推测解码需要另行实现批量验证、拒绝后的缓存恢复，以及混合状态模型的状态回滚。以上为待验证方案，均未形成新的速度结果。

本次仅访问公开元数据和小型文本文件，没有下载模型权重、执行远程代码或启动新的设备实验。完整原始响应与校验清单保存在研究目录；公开证据保留本轮筛选结论、目标元数据、源码比较及 Nano4B 的当前准备检查。

[筛选结论](evidence/recent-small-models-20261108/selection.json) · [元数据与原始来源](evidence/recent-small-models-20261108/targeted.json) · [源码比较](evidence/recent-small-models-20261108/spark-source-comparison.json) · [证据清单](evidence/recent-small-models-20261108/manifest.json)

Dory 的实际官方权重头与完整分词器结构已补齐核验：507 个 BF16 张量、3,975,229,440 参数，23 项元数据检查通过。固定源码显示原生参考与 CPU 融合 RoPE 有不同舍入路径；尚未运行模型或得到提速。见 [Dory 元数据、后端区别和优化方向](dory-metadata-and-optimization-20261108.md)。
