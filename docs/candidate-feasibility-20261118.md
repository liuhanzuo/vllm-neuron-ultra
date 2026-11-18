# 下一模型选择：LFM2.5-230M 的结构准入与两个 8B 候选

2026-11-18，在完成 Qwen3.5 正反序 HTTP 证据发布后，核对官方固定版本模型卡、配置和 safetensors 元数据。选择 **LiquidAI/LFM2.5-230M-Base** 作为下一接入对象：体量较小，适合当前磁盘余量；先建立正确的 BF16 基线，再考虑优化。不替代原 15 个模型家族，也不继承 LFM3 的模型资格。

| 官方候选 | 总参数（按官方 safetensors 元数据） | BF16 checkpoint 字节 | 架构与当前判断 |
|---|---:|---:|---|
| LiquidAI/LFM2.5-230M-Base | 229,693,184 | 459,401,112 | Lfm2ForCausalLM；作为下一接入对象 |
| ibm-granite/granite-4.2-8b | 8,791,592,960 | 17,583,228,032 | GraniteForCausalLM；保留候选，尚未完成新编译预算与基线验收 |
| nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 | 7,996,156,490 | 15,992,612,746 | Gemma4ForConditionalGeneration；需要 Gemma4 适配，不能套用旧 Nemotron 路径 |

以上均按**总参数**判断小于 10B。Bolt 模型卡明确为约 8B 总参数、4B effective；它声明 text/text 用法，但配置仍包含 vision/audio 子配置，须明确文本运行范围及依赖。不能根据 E4B 名称或 pipeline 标签假设它是原生 4B、或已能在 Neuron 运行。

候选来自当天已完成的 1,506 个仓库有限搜索，不重跑搜索。该快照中 LFM2.5-230M-Base 下载数为206,638，Granite 8B为60,416，Bolt为30,834；下载数只用来辅助选择，不作为质量证据。权重大小不包含编译缓存、checkpoint 转换或运行临时文件，不能直接作为完整设备运行预算。

## 已实际完成的 LFM2.5 结构检查

固定版本 `9d2be5519834990d30996f878b6771cccbd24f2c`。通过 HTTP Range 读取官方 safetensors 文件头，在服务器用现有 Transformers 4.57.6 建立完整模型的 meta 结构并逐项比较：

- 全部132个保存张量均为BF16，名称与形状匹配；合计229,693,184个参数。唯一额外 state_dict 键 `lm_head.weight` 与 embedding 为同一绑定参数。
- 文件头为14,736字节；全部张量数据区连续，大小与官方459,401,112字节文件一致。完整权重没有下载或哈希，官方LFS摘要仅作为后续下载校验目标，不能当作本次已验证的内容摘要。
- 14层由8个卷积层与6个GQA注意力层组成；卷积宽度3，attention head dimension 64，RoPE theta 1,000,000 与官方配置一致。官方文件声明 Transformers 5.2.0；现有4.57.6的结构能够匹配，前向语义仍需完整验证。
- 当前 NxDI 原生 registry 没有 `lfm2`。现有 LFM3 插件使用宽度7卷积、sandwich norm和注意力输出门控，不能直接复用为 LFM2 基线；LFM2需要保留自身的预归一化残差结构、无输出门控GQA和原权重名称。

本次执行正常返回，**没有读取完整权重值、运行前向、编译或使用 Neuron**。这项结构检查不是数值、冷缓存、长解码、槽位复用、采样重放、quality96或HTTP资格，没有新增模型提速倍数。[检查结果](evidence/candidate-feasibility-20261118/lfm25-checkpoint-preflight.json) · [源码、固定版本配置与文件摘要](evidence/candidate-feasibility-20261118/manifest.json)

## 后续实验入口

为LFM2建立独立基线适配器，先用官方BF16原权重对照完整CPU参考，覆盖冷状态、不同输入长度、长解码和槽位复用，再经精确存储预算与原串行锁准入后编译。随后执行原完整验收和匹配HTTP，确认可用基线后再测缓存写入及CPU采样优化。Base模型的质量绝对值须独立报告；不能把Base与指令模型混在同一配对中。

接手时 NVMe 可用42,476,666,880字节，低于KV256原40GiB门槛42,949,672,960字节。没有降低该门槛或扩容；MiniCPM5的10,173,284,352字节仍未实际回收，DeepSeek旧HLO到归档绑定仍缺。两者继续作为独立待办保留。
