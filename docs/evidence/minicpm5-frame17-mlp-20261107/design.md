# MiniCPM5：第 17 帧第 0 层 MLP 的一次有限诊断

状态：源码与小 JSON 设计；尚未实现、准入或执行。只检验一个操作差异，不提出新合格模型或提速结果。

## 要回答的问题

在第 17 帧（fixture 1，28-token 预填充，原 call 8、slot 1）中，已有“第 0 层输出投影仅预填充用 FP32”候选在 embedding、input RMSNorm、attention 输出、post-attention RMSNorm 的**预测位置向量**哈希均与原生参考相同，随后 MLP 输出哈希不同。其最终 KL 为 0.028214486，原门槛为 0.02。进一步逐项核对发现，全部 8 个预填充样本均在 post-attention RMSNorm 预测向量相同、MLP 预测向量不同；这里只选择已新增失败且长度短的第 17 帧。

**单一假设：同一 MLP 输入上，SDK 的 down projection 将 6,144 维输入拆成两份 3,072 维，各自产生 BF16 局部输出再 BF16 求和，这一计算路径是否足以解释本例 MLP 差异？** 原生使用一次完整 BF16 linear。不能预先排除 gate/up 的分片输出宽度、矩阵形状或运行库也产生差异。残差、RMSNorm、SiLU 的源码计算顺序相同，当前没有依据修改这些语义。

这增加了此前没有的信息：original88 定位 attention 内部差异；全部 42 层 o_proj 干预、仅第 0 层干预都使完整 CPU 检查从 29/32 降到 27/32；最近仅四个观测点的数值重放说明局部误差下降不保证最终改善。它们都没有保存 MLP 内部的完整输入与输出，也没有同输入分离 MLP 前半段与 down projection。

## 缺少的观测和一次最小采集

**完整 postnorm 输入目前未保存。** HF32 档案保存每个 tap 的 [1,1,2048] 预测向量；目标候选的四-tap bundle 同样只保存预测向量，且不包含 postnorm/MLP。original88 保存 layer0 attention 内部张量，未保存 postnorm/MLP。源张量的完整 shape/stride 只是元数据，不能证明完整输入相等。不得用复制末行、补零重建、原生输入替换候选输入或重新计算的张量冒充观测。

新增一个有限入口，复用已通过的 native loader、CPU constructor、七输入读取/整数无损转换、层 0 实例 o_proj patch、TP2 传输与退出流程。native 与 CPU 串行运行；只执行 fixture 1 的原形状 embedding 和第 0 层，在第 0 层正常返回后用明确私有结束信号停止，保证第 1 层、最终 norm/head、设备编译及服务都未执行。无需重跑 14-call 轨迹或添加新的通用审计框架。

- **原生：**官方 BF16 权重、原生 interpreter、eager/eval、2 threads；原 28 个 token、全 1 mask、positions 0..27、use_cache=True，首次 prefill 无历史。保留 [1,28,2048] 的完整张量。
- **CPU 候选：**原官方 TP2 分片、原 SDK interpreter、每 rank 1 thread；取原 call 8 的全部七个输入，原 slot 1、bucket 128、右侧 padding 和整数转换均不变。仅复用已经运行的 layer0 prefill o_proj FP32 实例干预；MLP 完全不改。保留 [1,128,2048] 及实际 stride [1,1,128] 的 MLP 输入。
- **历史边界：**这是 prefill 层前缀诊断，不声称重放完整候选历史。原普通 prefill 路由不消费 past KV；实际入口必须确认 past_key_value=None、get_kv_per_layer=False、无 prefix/window 路由、无 KV lookup/update，且在后续模型/KV 写回之前结束。若观测到任何历史读取，停止，不以零缓存替代。补充元数据证据：旧 C_Hc call 7 的 layer0 两项缓存哈希在两个 rank 均等于目标候选 call 8 的 before 哈希；本设计不需要读取缓存载荷。
- **采集 5 个完整张量：**MLP 输入（postnorm 输出）、gate 输出、up 输出、down_proj 输入（实际 SiLU×up）、MLP 输出。用实例模块 pre/post hooks 保存实际值，不修改返回值或运算；原生 5 份、CPU 每 rank 5 份，共 15 份。记录 shape、stride、offset、dtype、所属 frame/rank/模块、字节数和哈希。CPU gate/up/down_input 为 [1,128,3072]；原生为 [1,28,6144]。
- **旧锚点必须精确复现：**六个预测 tap（embedding、input norm、attention、postnorm、MLP、layer0 输出）逐位哈希和原 source geometry，分别匹配原生 frame17 与两个候选 rank/call08；不得要求 native 与候选 MLP 相等。确认仅选中层/phase 执行、两 rank 的完整复制 MLP 输入/输出一致、所有输入/权重未修改。此时形成的是新采集的完整输入/输出锚点，旧证据仍只有预测向量。任何锚点失配都使重放不准入。

## 完整形状重放与唯一干预

两项采集结束后，复用已接受 saved-oproj full-shape 重放的有限权重区间读取和数值统计办法，只读取官方 layer0 gate/up/down 三个 BF16 矩阵（各 24 MiB，共 72 MiB）。gate/up 官方连续行按 3,072 分片，down 按连续列分片；新运行须用实际模块权重逐字节证明映射，不只检查 shape。载荷是 packed 存储时按记录的 stride/offset 还原实际输入布局，验证往返字节，禁止直接将非连续输入当作 contiguous 控制。

先以各自**完整输入、原形状和线程数**重放原生 BF16 linear→SiLU×up→linear，以及 SDK 的原 einsum spelling→SiLU×up→两个局部 BF16 down 输出及 BF16 和。全部五个已采集边界都必须逐位重现，CPU 两 rank 的完整输出分别校验；原生控制使用 2 threads，CPU 控制使用 1 thread。可以先在 SDK interpreter 检验原生数学重放，但只有实测完整边界精确才允许使用；这不等于相同原生运行环境。单进程 BF16 和重现 Gloo 结果也不等于重跑了 Gloo。若任何控制不精确，停止并报告具体第一个失败边界，不缩成单行、不放宽阈值。

真正的对照固定为**候选完整输入 Xc=[1,128,2048]，实际 stride，1 thread，同一进程/库、官方 BF16 权重**：

1. **S（原路径）：**SDK gate/up 分片及实际 SiLU×up，两个 BF16 down 局部结果再 BF16 相加。必须精确匹配新采集的候选完整 MLP 输出。
2. **F（同输入参考）：**同一个 Xc 用完整 gate/up 的 BF16 linear、相同 SiLU×up，再完整 BF16 down linear。F 是新反事实参考，不是原生 [1,28,2048] 的旧观测。
3. **D（唯一干预）：**直接复用 S 的实际 gate/up/SiLU×up 结果，只按已证明的 3,072+3,072 列顺序拼接，改用一次完整 BF16 down linear。不改 FP32，不改层数，不改任何输入或权重。

先逐位比较 S 与 F 的 gate/up/乘积，再比较 S/D/F 输出；相同 down 输入还必须形状、stride、dtype 相同。分别报告完整 128 行、真实前 28 行、预测行 27 的不同元素数、max/mean absolute error 和 cosine；同一复制输出的两个 rank 只计一次。另行比较两个新观测 Xn 与 Xc[:,:28,:]，清楚记录完整输入是否相同，不把预测行相同外推。

## 结果如何决定下一步

| 结果 | 允许的结论 |
|---|---|
| 精确控制失败 | 本次机制比较无效；最小缺失事实是该边界的精确重现，保留失败，不开展新精度扫描。 |
| S/F 前半段乘积完全相同，S≠F，而 D=F | 在这个固定输入、形状与运行环境内，分片 down 运算路径足以解释 MLP 输出差异。仍未证明它导致最终 KL 失败或精确区分内核累加与局部 BF16 舍入。 |
| 前半段乘积已有差异，D 仍≠F | down 路径不是唯一来源；最早观测差异具体落在 gate、up 或乘积。下一步只围绕该实际证据设计，不自动启动。 |
| D 比 S 更远或无改善 | 拒绝 down-only 修复方向；不把 BF16 完整 linear 或更宽精度当成全模型保证。 |

即使局部严格复现并解释成功，也不运行新的完整模型候选；先评估结论再单独决定。既有 all42 与 layer0 候选仍被拒绝，CPU 严格 29/32、设备严格 30/32 分开保留。原阈值（cosine≥0.995、KL≤0.02、maxabs≤2、总体 top1≥0.9、参考 margin>0.5 必须 top1 相同）以及冷缓存、长解码、槽位复用/空闲槽位、96 题质量、配对服务性能检查全部保留。

## 成本与交付边界

模型加载串行复用原 16 GiB/no-swap/2 CPU 上限；仅执行三个层前缀（原生 1 次、CPU 两 rank 各 1 次），加载仍会读取原完整 checkpoint，这是最少源码改动的路线，不承诺只加载一层权重。离线重放只需 72 MiB 权重、少量矩阵乘法，可设 2 GiB/1 CPU。保守预计几分钟至 15 分钟，建议总硬上限 20 分钟；尚未实测。15 个观测张量恰为 8,077,312 字节，结果/日志总 cap 32 MiB；不保存全模型缓存、整套 32-frame 新 payload 或权重副本。沿用现有串行准入与资源空闲检查，root 观察实际退出；无后台自动执行。

设计阶段仅读取源码/JSON；未读取、哈希或 stat 原始张量/权重载荷，未导入 Torch/NumPy，未启动任何实验。精确证据路径和源码哈希见同名 handoff JSON。
