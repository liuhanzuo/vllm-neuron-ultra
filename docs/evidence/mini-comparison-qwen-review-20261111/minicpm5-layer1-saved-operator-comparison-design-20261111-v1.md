# MiniCPM5：已保存算子边界的全域比较设计

本交付是 **源码与小型合成测试**。没有读取实验 tensor payload、导入 torch/numpy、运行模型、操作进程/服务/锁或改写 canonical 状态。真实比较必须等根线程观察当前 native 运行真退出、完成独立全字节审计，并认证已有 SDK 恢复材料之后，进入原串行槽位。没有提供绕过这些条件的独立执行入口。

实现：`minicpm5_layer1_saved_operator_comparison_v1.py`。它只接收内存中的已认证 bytes；调用者负责原 descriptor、实际退出链、独立 audit、权重/输入来源、旧控制与原始文件 SHA 的认证。这些认证不能由本函数返回的 `runtime_admitted: false` 或源码测试替代。

## 完整范围与字节预算

| 范围 | capture 文件数 | capture 原始字节 |
|---|---:|---:|
| native：24 帧 × 30 leaves | 720 | 14,897,124 |
| SDK：6 calls × 2 ranks × 47 leaves | 564 | 62,893,440 |
| 全部比较输入 | 1,284 | 77,790,564 |

包含控制的原始 native 数据仍是 936 文件 / 28,490,724 B；SDK 仍是 624 文件 / 63,876,480 B。控制的认证由已有独立 auditor 完成，不能因为本比较只消费 capture 就删掉原控制。

每次消费一整个 call 的 4 native 帧和 SDK 两 rank，共 **214 完整 leaves**。总 capture 上限 128 MiB；单 call 输入上限 32 MiB；每 call JSON 报告上限 2 MiB，超限直接拒绝，不能截断。所有 6 calls、24 帧、1,284 文件和精确原始字节总数均须汇总成功。

固定 calls 为 4、5、6、11、12、13；bucket 为 2048、2048、2048、128、128、128；native 原索引为 4–15、20–31。Native manifest 的 canonical JSON SHA 固定为 `3671f05b8d8cb29e47a03a7efc26caf19e12937454f351b0c5dd3f08baa4ddc2`，因此变更 frame、fixture、slot、call、长度或控制来源均拒绝。SDK 必须逐字段等于原完整 manifest。

## 对应关系

SDK 物理行来自原 `frame.slot`，不能用 `call.frames` 的列表次序代替。例如 call 4 的物理行 0、1、2、3 对应 native frame 6、5、7、4。

| 保存边界 | 完整对应方式 |
|---|---|
| normalized input、位置、有效历史 | 两 rank 对应同一完整 native 输入；位置、实际 mask 和所有有效 prior KV 先逐字节一致，否则停止定位 |
| Q/K/V projection 与 RoPE | rank 0 对应 Q heads 0–7、KV head 0；rank 1 对应 Q heads 8–15、KV head 1；每 head 的 128 维全部保留 |
| 实际更新后的有效 K/V | native 的完整有效 cache 与 SDK 保存的完整有效 prior 加实际 returned current K/V 做字节拼接后比较；不生成新的张量算术 |
| QK、缩放、softmax、概率 | 每 head 按 `[全部 p 个 prior keys, 当前 active key]` 拼接原 SDK 保存值，与 native 的 `p+1` 个有效 keys 比较 |
| mask 后有效 prior scores | native 保存的 softmax 输入有效 prior 部分，对 SDK 保存的 prior mask 输出；不重算加法或缩放 |
| PV | native 保存的完整 `pv_context` 对 SDK 实际合并后保存的 `context_before_o`，只按 head/维度布局重排 |
| 输出投影 | native 保存的完整 attention output 对 SDK 实际归约后的完整输出；SDK 两 rank 的实际归约终点必须相同 |

SDK 的 `pv_prior`、`pv_active` 和 `o_proj_local_before_reduce` **没有保存的 native 一对一对应项**。它们做完整独立统计，并显式标记 `native_counterpart: false`；不会人工相加、重做投影或伪造 native 中间值。

所有 SDK-only padding 都留在完整 inventory 内。额外按 frame/rank 对完整 prior KV padding、unscaled/scaled score padding、masked score padding、softmax 输入 padding、两种概率 padding 做统计。概率 padding 必须数值全零；其 signed-zero 数量仍保留。不存在的 native padding 不补零充当证据。

## 字节与数值规则

- 原始 capture 已被保存为 canonical packed logical bytes。切片地址只由逻辑 shape 的 row-major 索引计算；**绝不使用原 tensor 的 source stride 或 storage offset 访问 packed 文件**。原 source geometry 仍须由已有 auditor 独立认证。
- native softmax 输入是 BF16，SDK 是 FP32。对 native 做精确位嵌入：little-endian 的每个 BF16 word 前加两个零字节；不重做 softmax、加法、缩放或浮点四舍五入。正负零的符号位不丢失。
- 所有完整 floating leaf 均检查 NaN/Inf 并拒绝；bool 必须是规范的 0/1 bytes。每个 leaf 都保存字节数、字数、SHA、完整有限值/零/负零计数及范围。
- 每个可比域遍历全部 words，报告不同 word/bit 数、最大绝对误差、误差平方和、native 平方范数、RMSE 和 relative L2。误差累计用 Python FP64；逐位相等与 SHA 独立于误差舍入。native 范数为零且误差非零时 relative L2 用 null，不能伪造 0 或输出 Infinity。
- 每域最多列 8 个说明性差异位置；计数和误差仍覆盖完整域，例子不是抽样覆盖率。额外核验重复保存的 dispatch、位置、mask、KV、prep、cache 与归约终点身份。
- “first differing saved boundary” 按 native 阶段的保存顺序报告；prior 与 active、Q 与 K 等分支不能据此推出因果。首个差异只缩小检查范围，**不是算子根因证明**。

## 已做的小型测试

唯一测试入口：

```text
python3 -B /home/ubuntu/experiments/trn1-latest-models-20261031/research/check_minicpm5_layer1_saved_operator_comparison_v1.py
```

测试只读取两个现有 manifest JSON。合成 payload 总计 **368,380 B**，最大单 leaf **16,384 B**，使用 production contract 不接受的 **test-only bucket 4**。这些 bytes 不来自真实模型，不能当成已执行原实验。完整 77,790,564 B capture fixture 未构造、未比较；实际 tensor payload 读取量为 0。

9 项测试覆盖：固定原始 24 帧/6 calls 与错误身份拒绝；packed shape 寻址；BF16→FP32 与 signed zero；完整计数/指标及 8 例上限；NaN/Inf/bool 拒绝；完整 30/47 leaves 及 missing/extra/short 拒绝；两 rank/head/slot 与 prior+active 拼接；错槽位、错 rank、错 active、非零概率 padding 和重复输入变更；完整六 call 报告结构的元数据序列化。

报告预算测试只合成 24 帧和完整 leaf inventory 的 **元数据**，每比较域放满 8 个例子，并放入长数值表示。最大单 call 约 0.4 MB，低于 2 MiB；它不是实际 capture 的测量结果，也不证明真实 reader 或独立 auditor 已通过。Python 对象/RSS 的额外内存未测量；payload 规模和序列化规模是确定的。

所有 BF16、原模型与 cold4096、长 decode、slot reuse/inactive behavior、quality96、matched HTTP 门槛保持原约束。本交付不新增提速倍数，不把局部边界比较当成全模型合格或 Neuron 服务资格。
