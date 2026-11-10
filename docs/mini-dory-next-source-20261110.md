# MiniCPM5 与 Dory：下一轮诊断已接入源码，Nemotron 完整运行继续

2026-11-10 的新进展分为两部分：Nemotron r12 已实际完成 52 阶段中的前 8 个阶段；MiniCPM5 与 Dory 在它运行期间推进轻量源码准备，尚未启动新的真实模型执行。所有外部检查都通过性能窗口保护，模型、原生库验证和编译仍按原串行队列进行。

## MiniCPM5：固定输入与历史，区分批大小和布局

上一轮 180 次真实局部 attention 调用已证明，同时换成官方当前归一化输入和有效历史 K/V 后，原严格对照汇总相对 L2 从 1.2720% 降到 0.4464%，但 24 帧仍无一帧与官方完全一致。这不是整模型提速或资格结果。

下一轮在同一个 SDK 路径内比较 B4/B1 × 原 stride/连续布局，并保留完整 B4 own、restore 和上一轮 both-official 输出作为逐字节对照。固定原 bucket、有效历史、物理请求槽位、BF16 权重、分片投影和 BF16 规约；四种变体共用同一次 B4 RoPE。SDK B1 不等同官方原生 attention，不同时改变历史长度或 softmax/PV 算法。

源码已串起原完整模型构造、官方权重分片核验、输入读取、调用、持久化输出与独立结果复核。每个 rank 完整校验原 42 次历史调用、18 个解码输入，实际仅重放 anchor 的 6 个解码调用；另外两个候选的 12 个解码输入明确记录为已校验、未重放。计划产生 144 次 attention 调用，保存 72 份完整四行输出，共 1,179,648 字节，覆盖全部 24 个独立解码帧。

结果审核会重读新输出及原 own/restore/both 完整控制，检查每个 B1 分片对应原物理行、实际 view 的 shape/stride/offset 和双 rank 输出一致性，再统计四组 batch/layout 配对差异。合法新变体允许与原输出不同，不能把输出一致作为所有变体的强制要求。RoPE 原始字节由调用阶段检查；当前档案仅存其元数据，退出后的复核不能声称独立重读了未保存的 RoPE 原始载荷。

| MiniCPM5 源码检查 | 通过数 |
| --- | ---: |
| 原始字节、布局和物理槽位核心 | 15 |
| 注入框架接口的 attention callable | 17 |
| 完整输出分析与拒绝检查 | 17 |
| driver、存档和跨 rank 验证接入 | 18 |
| 合计 | 67 |

合成接口不会执行真实 attention 数学。独立审阅发现分析代码用普通四项累加时，可能把极小非零交互抵消为零；现改为 `math.fsum`，保留完整 24 帧回归用例，并通过独立复验。该问题是在合成输入中发现的，没有证据表明旧真实数据恰好存在同样模式；它也不是模型误差的贡献比例。

真实执行仍需接入最新 Nemotron 实际退出链，固定新的 worker/runner/审计器及完整源码、输入身份。原整模型结论仍为对照数值 29/32、首选 token 32/32；两个候选各 28/32、31/32，继续拒绝采用。

## Dory：完整观测、按原位保存字节与独立文件核验

Stage A 观测核心保存六次原始完整输入形状，在三个目标访问点记录完整 input、mixer output 和 residual output；同时保留全部 224 个抽样边界及 114 个原始输出。43 项合成检查和独立边界控制通过，尚未执行新的原生模型采集。

新张量适配器显式接收原 Torch 接口，按逻辑行序读取非连续 stride、storage offset 的 BF16/F32 张量，保持原始位模式和完整列数。每块至多 64 KiB，不为了存档先复制整个大张量，也不把切块后的张量作为模型输入。47 项合成接口检查通过；真实固定版本 Torch 的接口和内存开销仍须独立原生验证。

独立文件审核增加了从偏移 0 读取、完整文件长度、原 producer 的设备/inode/大小/修改时间/状态变化时间身份、文件稳定性及 EOF 检查。审核发现并补齐了两处草稿边界：读期间祖先目录被改成符号链接时再次拒绝，以及文件流构造失败时的句柄清理和原异常保留。最终 18 项小文件检查及独立复验通过。这些测试只使用临时合成字节，不读取模型权重或旧实验档案。

Stage A 全部原始存档预算为 1,467,113,472 字节，未来三个诊断阶段共约 1.906 GiB，仍需额外 3 GiB 预留。当前没有占用 Nemotron 的完整模型预留。Dory 原 87 个数值检查点中的 7 个失败继续保留，后续全部 87 点、冷缓存、长解码、槽位复用、质量和真实 HTTP 验收未被缩减。

## Nemotron：8/52 为已完成阶段，不代表整模型通过

本轮独立复查了前 8 个阶段的 worker 通过记录、实际保留的成功退出及进程／资源组排空。原外部会话与控制器仍存活，并持有原串行锁；随后阶段继续运行。当前没有新的整模型性能结果，既有所有模型提速总表保持原样。

证据：[Dory 稳定文件核验独立审阅](evidence/mini-dory-next-source-20261110/dory-stage-a-saved-files-independent-review-20261110-v1.json)、[完整清单](evidence/mini-dory-next-source-20261110/manifest.json)、[MiniCPM5 根代理源码审阅](evidence/mini-dory-next-source-20261110/minicpm5-layer1-sdk-batch-layout-root-source-review-20261110-v1.json)、[分析问题与独立复验](evidence/mini-dory-next-source-20261110/minicpm5-layer1-sdk-batch-layout-analysis-independent-review-20261110-v1.json)、[Dory 核心独立审阅](evidence/mini-dory-next-source-20261110/dory-stage-a-source-core-independent-review-20261110-v1.json)、[Dory 张量适配器交付](evidence/mini-dory-next-source-20261110/dory-stage-a-tensor-adapter-delivery-20261110-v1.json)、[Nemotron 八阶段实测](evidence/mini-dory-next-source-20261110/nemotron9b-r12-actual-progress-20261110-v2.json)。
