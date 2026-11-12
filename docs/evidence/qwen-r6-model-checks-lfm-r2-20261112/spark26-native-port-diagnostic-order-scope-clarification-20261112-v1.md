# Spark native core v2：请求顺序与物理输入顺序澄清

主任务指出的问题准确。`cache_trace()` 中的确存在乱序请求槽位，但 `cache_diagnostic()` 在 core v2 第 210–235 行先构造 `by_slot`，然后以 `range(4)` 生成 `seq_ids`、position 和 K/V 行。送到 `candidate.decode_cache()` 的有效物理行始终按照槽位升序；无效物理行使用 -1。core 没有执行真实 vLLM `_reordered`／`restore`，也没有把任意排列的物理 B4 行交给 tensor 函数。

因此，本诊断当前范围应写为：**trace 含乱序请求元数据，真实 tensor 核心拟验证排序后的 B4 物理缓存语义、dummy 写隔离、历史与槽位复用；它不直接覆盖物理输入行排列，也不能证明服务入口的请求排序和输出还原正确。** 以前 notes 中“乱序请求”仅描述计划元数据，若理解为真实 tensor 输入排列或端到端重排已覆盖则过强。本说明是原交付的范围勘误，原 v1/v2 与交付不改。

固定 SDK 源码支持这种排序后的物理布局：

- vLLM-Neuron `neuronx_distributed_model_runner.py:233` 在非 prefix caching 且非 chunked prefill 时启用 `is_reorder_needed`；Spark v2 拒绝这两条非目标路径。
- `neuronx_distributed_model_loader.py:111` 的 `_reordered` 在第 122 行排序 block/slot IDs；`_sort_inputs` 用同一 indices 重排批次张量/列表。第 125–129 行通过逆排列恢复输出，`NeuronCausalLM.forward` 最终调用 restore。
- NxDI `model_wrapper.py:599–621` 在普通连续 KV 模式下补齐缺失槽位并排序，`apply_seq_ids_mask` 再将无效 ID 改成 -1。第 641–645 行以相同 indices 重排重复首行补齐的输入，返回时按原 seq_ids 选择相应行。
- 满 B4 的 wrapper 路径在第 1345 行直接执行 `_forward`；该路径依赖前面的 vLLM 排序。core 的 B4 顺序与目标路径的排序后布局一致，但这是源码推导，不能代替真实 host 执行。

本轮仅做源码阅读和标准库符号推导：64 个非空有序槽位子集的映射与上述“先 host 排序、再 SDK 补齐排序”算法一致。这不是 64 项 SDK/tensor 测试，不是原生验收。

## 后继最小方案（未实现、未准入）

保留原 534 步全部案例，新增两项不同层级的检查：

1. **实际 host 路径**：在认证后的 serial worker 中调用固定 `_reordered` 与 `_sort_inputs`，将实际整数 position/KV identity tensor 与请求一同排序；验证同一 tuple 的各行关联、SDK padding 的实际输出，以及 `restore` 的逆映射。必须通过真实固定函数，不以此处符号算法替代。它直接对应服务承诺，必须纳入后继冷缓存／长解码／HTTP 完整验收。
2. **独立 tensor helper 的行排列覆盖**：另选 full B4、B3、B2、B1 四种原 trace 状态的调用前缓存快照，分别对四个物理输入行执行全部 24 个排列，同步排列 K/V、position 和 seq_ids。每次从相同输入缓存重新调用实际 `decode_cache`，用逆排列比较逐行返回 K/V/position/mask；persistent K/V 按物理槽位存储，必须与未排列对照逐位相同。dummy 行只要求合法读取和无写入，不把其数值当真实请求结果。这是辅助健壮性诊断，不要求生产 host 保持乱序物理行。

方案须新增不可变 contract/core 版本，原科学门槛、全部原案例和失败记录保留。还需按新增缓存快照/调用次数核算串行内存、时间和输出预算；本澄清没有增加本轮运行许可，也没有启动 native、Torch、服务或权重读。
