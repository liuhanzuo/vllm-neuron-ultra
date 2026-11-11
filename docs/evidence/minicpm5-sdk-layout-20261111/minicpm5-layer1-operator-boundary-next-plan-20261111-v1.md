# MiniCPM5 layer1 下一步：在原注意力路径中定位首个数值分歧

当前只完成源码设计，没有运行模型、导入原生库、读取张量文件或获取串行锁。r2 的下列结论来自 coordinator JSON，必须等根线程完成独立原始字节审计后才能作为下一轮已完成前提。

r2 记录：B4 和 B1 各自改变 stride/contiguous 都没有改变 24 帧的输出。B1 相对 B4 只改变 frame 8 的 232 个 BF16 字和 frame 11 的 1 个字；相对官方的整体 relative L2 从 0.00446357787123175 变成 0.004465040877035361，仍没有完整相等的官方帧。own/restore 及之前 official_both 控制重现。这个结果不支持继续扩展布局实验，也没有证明根因。

## 最小新实验

保留全部 24 个 decode 帧、6 个原始 call（4、5、6、11、12、13）、TP2 两个 rank、原物理 slot 顺序及 bucket（前三个 2048、后三个 128）。只为已经测过的 **B4 strided / official current + official history** 增加真实内部张量观测。其他历史/布局实验继续作为不可变控制引用，不重跑其矩阵。

SDK 使用原完整 42 层模型、原 rank-local BF16 权重、BF16 输出归约、单线程和原 SDK attention 类。每个 rank 的每个 call 按以下顺序执行：

1. 既有无观测 own 控制，逐字节等于原 saved own。
2. 无观测 official_both，逐字节等于 r2 b4_strided 和之前 official_both。
3. noop 包装/trace 的 official_both，同一输入必须重现第 2 项。
4. 完整内部观测的 official_both，同一输入必须重现第 2 项。
5. 移除全部包装后 restore 控制，逐字节等于 own。

每次从同一完整原缓存重新准备，不累计上一次调用产生的状态。完整原缓存 backing（包括 padding）、input stride/offset、原 mask/positions/seq_ids、B4 产生的 RoPE、四份官方权重 shard 均保持前后检查。原有禁止未知 forward 覆盖的检查仍在安装前和移除后执行；观察中的调用必须显式验证只有本次已审查包装，不能简单删掉原 no-hooks/no-override 检查。

官方侧只重放 **单个原始 layer1 attention**：同一 native 解释器、原 eager 实现、BF16、2 线程、B1、完整 16 个 Q heads / 2 个 KV heads；使用既有官方 normalized input 及 step-1 完整 layer1 cache 恢复原 DynamicCache 的 layer 1，保留真实长度和位置，再用原 rotary module 产生 position embeddings。对所有 24 帧分别做 unobserved、noop、full 三种调用，每次独立恢复 cache。不要重新执行已完成的 32 次整模型轨迹或 prefill。

官方直接调用的 attention output 必须逐字节等于既有 `layers.1.self_attn` 完整 2048 维输出，更新后的完整 layer1 K/V 必须等于已有本帧 native cache。若重现失败，本次仅报告隔离调用不等价，不能把差异归因给 SDK，也不自行换精度、mask 或运行方式。当前源码尚未实现这层 admitted driver。

## 保存哪些完整张量

复用既有经过审查的 Python trace 与 clone-only SnapshotSink。每个 decode query 的长度为 1，原 score 的 prediction-row 选择因此覆盖本次调用的完整 Q 轴；不截短 key 轴，不取头/维度样本。

| 边界 | 保存内容 | 主要比较 |
|---|---|---|
| 输入、权重 | 完整 normalized tensor、位置/mask/slot、缓存引用及完整有效 cache view、官方四个权重哈希 | 确认输入先相同；输入不等时不能定位算子 |
| Q/K/V | 原投影输出，reshape 后 Q/K/V | native 16/2 heads 按固定 rank head 映射，对应 SDK 8/1 heads |
| RoPE | 完整 cos/sin、旋转前后 Q/K、实际 source/stride/offset | 区分 projection 与旋转分歧；本轮真的保存 cos/sin 字节，不能沿用 r2 未独立保存 RoPE 的声明 |
| QK/缩放/掩码 | 原 matmul 返回值、Python 原赋值产生的 scaled tensor、原 prior/active mask 前后值 | 保持 native multiply 与 SDK divide 的真实实现，不代算 scaled 值 |
| softmax | 原 FP32 概率及实际用于 PV 的 BF16 概率，SDK prior 与 active 均保存 | native 有效长度 p+1 对 SDK prior 有效 p 个位置 + active；完整 padded 张量也保存并核验 |
| PV | 原 matmul 返回的完整 native context、SDK prior/active context，以及实际合并后的 `context_before_o` | 区分概率差异与分段 PV/相加差异 |
| 输出投影 | native 完整输出；SDK 每 rank 的原 local BF16 partial、真实归约结果及 attention output | 先比较同一 context；context 不等时不能宣称 o_proj 根因 |

两个 rank 的完整输出继续核验一致。分析可以在已保存字节上做 head/slot/有效长度映射，但原始保存必须保留完整源 shape、stride、offset、dtype、byte count、SHA 与调用身份。mask/padding 的不同计算域必须明确标注，不能把不存在的 native padding 当成计算证据。

## 已准备的最小源码后继

- `minicpm5_layer1_operator_capture_observer_v1.py`：保留原完整 observer/sink，native cache 索引与层身份从 0 改到 1，限定 decode，额外记录真实 PV 返回对象及 SDK 输出归约前后对象。
- `minicpm5_layer1_operator_native_trace_observer_v1.py`：原 native trace 的 layer1/import 后继。
- `minicpm5_layer1_operator_sdk_trace_observer_v1.py`：原 SDK trace 的 layer1/import 后继，复用同一原函数完整源码/字节码/line-table 认证。
- `research/check_minicpm5_layer1_operator_observer_v1.py`：8 项 stdlib/inert-object 检查通过（chunk c1403a，exit 0）。验证层 0 被拒绝、层 1 cache 定位、仅 decode、原对象返回及包装恢复；没有运行 torch 或张量算术。

所有旧 observer、r2 callable、runner、builder、canonical 文件均未修改。这三个 observer 还没有实际运行资格；源测试不能代替 noop/full 的真实 endpoint 控制。

## 大小与下一轮命令要求

按现有张量几何估计，完整新 capture 原始数据约 **75–80 MiB**：SDK 两 rank 六 call 约 60 MiB；native 24 帧约 15 MiB；控制输出约 1.3 MiB。这里包含既有 observer 对 cache view 的重复记录，不靠删去历史/重复边界来压缩范围。不同 native mask dtype、元数据和文件系统块占用仍须在 concrete plan 中计算，不能把此估计冒充实际字节数。

沿用原总输出 **256 MiB**、单张量 **16 MiB** 上限，并在新 admitted plan 中按真实 event manifest 做精确分区/字节预算。最大 SDK cache view（B4×1×2048×128 BF16）为 2 MiB；完整 cache backing 继续由原持久证据和前后校验认证，不必复制每个 backing 为新输出。所有新增实际原始文件须由独立进程完整重读，重新计算 SHA、计数与比较。

本交付没有伪造可运行的 model 命令。唯一现在可以运行的检查命令是：

```text
python3 -B /home/ubuntu/experiments/trn1-latest-models-20261031/research/check_minicpm5_layer1_operator_observer_v1.py
```

真正下一轮需要先完成这些具体接线，仍使用原串行 runner/worker/独立审计形态：

1. 根线程真实观察 r2 model 与独立 audit 的退出，发布其 immutable completion；不能只依赖 coordinator 的 `complete`。
2. 创建新的轻量 callable/driver 后继，嵌入上述 scoped observer；native 与 TP2 SDK 两个阶段分开串行，不能把新 observer 直接塞入冻结的 r2 callable（它正确地拒绝 forward 覆盖）。
3. 完整继承 r2 和之前 1308 历史服务/owner 证据及原 lock，构建 fresh output namespace、source closure、精确 capture manifest 和 resources。源代码独立审查通过后才填入真实 spec/review path+SHA。
4. 运行命令继续采用原模式 `...cpu_runner_<新版本>.py --mode outer --spec <新spec> --spec-sha256 <真实SHA> --review <新review> --review-sha256 <真实SHA>`；native 使用原 native runner 的对应新后继。这里的“新版本”尚未实现，不得把占位符视为已存在的执行入口。
5. 每阶段真实退出后分别记录 exit，再独立全字节审计；没有接入任何自动 successor launch。

看到“最早不同的保存边界”只能缩小检查范围。若 Q/K/V 与 RoPE 全相同而 PV 前开始不同，下一步才做同输入算术定位；若只有 o_proj 之后不同，才沿用既有 full-shape 投影方法。不要在本轮扩展新的因果矩阵，也不要把局部诊断替代原 cold4096、长 decode、slot reuse/inactive behavior、quality96 和 matched HTTP 性能门槛。
