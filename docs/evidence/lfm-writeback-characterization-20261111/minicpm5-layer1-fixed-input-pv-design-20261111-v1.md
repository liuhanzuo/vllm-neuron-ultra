# MiniCPM5：固定 P/V 输入的有限重放设计

这是源码与元数据层面的实验设计，不是 runtime、运行 spec 或准入。未读取真实 capture/native 字节、导入模型框架、解析实际 history、操作锁/服务或启动任务。新 builder 只产生有限数据矩阵，没有执行器、生命周期或监控代码；所有冻结来源不变。

## 已有证据与待区分的问题

Root 已观察独立审计 session 39819 真正 exit 0（chunk `49846e`）。它完整重读 1,284 capture / 77,790,564 B，六报告和整体 summary 精确重建；audit result SHA `8b50014cd4ce8756733e648fcce9698732f68b15dff373db78ffc0061e59a2d1`，外部 terminal SHA `c1696463a10393d285564890b6e89d593762579a94b0dea9c9af40fc1f774b0d`。

已审计报告包含 48 个 frame/rank 对：BF16 probability 有效域 48/48 全字节相等，effective V 为 47/48，merged context 为 0/48。唯一 V 不同是 frame 27 / call 12 / fixture 3 / slot 2 / rank 1，共 2 个 BF16 字。保留全部 48 对及 native、SDK 两套原输入；不得去掉该异常对或把所有情况当作共同 V。

这些结果定位了下一实验入口，尚不能证明某个算子或舍入步骤是原因。待区分的是分块/合并 reduction、返回 BF16 的中间舍入、最终相加和 cast，以及 batch、head、padding、stride、原解释器与线程环境。

原 native `modeling_llama.py`（SHA `363df58c077a8fcfd3593b2817538e53f90eeab32294d3c6bbb7241f094669de`）在 `eager_attention_forward` 对 repeat_kv 后 V 做一次完整长度 matmul。原 SDK `attention_base.py`（SHA `c436c003161c9fd8d8d8a2a258284026b727aabca64d596c845d54ca776de5e8`）2068–2071 行先把 prior/active probability 转为 Q 的 BF16 dtype，再分别 matmul，最后相加。SDK repeat_kv 来自 `utils.py`，SHA `282142ea4a2a761e4287d8d6962f7e11d8ca8e57261feb74c038535d13319016`。两个实现都用 expand/reshape，但 KV head 数不同，实际 view/materialization 不应预先假定相同。

## 先完整复现各自原输出

第一道门槛是 **36 次路径调用、60 个完整 tensor-output 等式**：

| 原环境 | 调用 | 原始操作尺寸 | 必须全字节相等的输出 |
|---|---:|---|---:|
| native Python 3.12，Torch threads=2 | 24 帧 | P `[1,16,1,L]`；V `[1,2,L,128]` 按原 repeat_kv 扩展至 16 heads | 24 个完整 `pv_context` |
| SDK Python 3.10，Torch threads=1 | 6 calls × 2 ranks = 12 | prior P `[4,8,1,K]`、V `[4,1,K,128]`；active P `[4,8,1,1]`、V `[4,1,1,128]` | 12 ×（`pv_prior`、`pv_active`、merged context）=36 |

Native 使用 `/home/ubuntu/venvs/model_reference_2026/bin/python`；SDK 使用 `/home/ubuntu/venvs/vllm_neuron/bin/python`。每个环境保留其原 Torch/build/source、CPU 与线程设置、inference mode、无 autocast 等执行合同，记录实际 backend/precision flags。既不把 SDK 放进 native 解释器，也不把二者线程数统一后冒充原控制。

P 必须取真正送入 PV matmul 的保存 `probabilities_bf16`；不重新 softmax、不换成保存的 `probabilities_fp32`。V 使用各自实际有效 history/current V。Native 的另 24 项保存 `pv_context` 与 `context_before_o` 全字节 flatten 身份仍验证，但它们不是另外 24 次计算控制。SDK merged context 按原 transpose/reshape 与 `[4,1,1024] context_before_o` 比较；所有四个物理行均保留，不能只取某个行后把结果称为原 B4 复现。

输入由 packed logical bytes 重建，必须保留记录的 shape/stride/storage_offset，再执行原 repeat_kv 并记录其输出布局。SDK prior V 的 stride 在 K=2048 和 K=128 时都为 `[540672,540672,128,1]`；按连续数组恢复会丢掉原 batch 间隔。只恢复可见 logical bytes和布局，不能声称已重建未保存的原分配地址或隐藏 backing 内容；未访问间隔独立初始化，不从旧文件越界取字节。每次重建必须全字节 roundtrip，所有输入结束后仍不变。

两环境的所有初始控制都通过之后，才允许后续解释。任何一个原输出不能全比特复现，就停止本设计的因果判读，先调查环境/布局复现；不改 tolerance、不补算上游投影、不改原输入。各环境的变体结束时，再重复相同 36 路径 /60 tensor 控制，并比较前后执行指纹；最终共 72 路径、120 个计算输出等式。

## 再拆开 shape、layout 与 padding

每个 frame/rank 都做以下单因素顺序桥接，共 240 次路径，始终使用来源自己的解释器与线程数：

| 链条 | 保持的公式 | 被改变的因素 |
|---|---|---|
| Native 原 B1/H16 → 从原完整 repeat 后 P/V 取对应 8 heads | full concat BF16 | head 调用尺寸；保留切片 stride/offset |
| Native 上一步 → 完整 contiguous P/V clone | full concat BF16 | layout；逐字节输入相同 |
| SDK 原 B4/H8 → 对重复后的 P/V 取单个原物理行 | padded split BF16 | batch 调用尺寸；保留原 view stride/offset |
| SDK 上一步 → 完整 contiguous P/V clone | padded split BF16 | layout；逐字节输入相同 |
| SDK 上一步 → prior 截到该行 `L−1`，再连续化 | split BF16 | 去掉 masked padding 的 reduction 长度；active 独立保持 |

桥接输出若变化，记录为实际 shape/layout/padding 敏感性，不抹掉或筛除。各桥的输入字节关系必须先证明。必须检查完整 padding probability 的零位与 finite V，保留原 padding V；不能悄悄把 V padding 置零。零概率消除实数贡献，并不自动证明不同 reduction 形状或 signed zero 的逐位结果相同。

## 固定同一有效域后的六式比较

统一比较域仅用于诊断：B1/H8、完整有效长度 L、所有 expanded V 与 P 都为明确 contiguous 布局。每个原 frame/rank 保留 **native 与 SDK 两套 BF16 原字节**，每套均在两个原环境执行同一六式；不是把原模型改为 FP32。总数为 48 × 2 origins × 2 environments × 6 = **1,152**。

令 `M16` 为 BF16 operand/result 的当前环境 matmul，`M32` 为从同一 BF16 字节精确扩为 FP32 operand/result 的 matmul；`R16` 为一次 BF16 cast。prior 为有效前 L−1 项，active 为最后一项。矩阵输入顺序严格是每个 head 的 `[prior..., active]`，V 的位置顺序相同。

| 变体 | 计算形式 | 需要保存 |
|---|---|---|
| concat BF16 | `M16([Pp,Pa],[Vp;Va])` | 完整 BF16 context |
| split BF16 | `M16(Pp,Vp) + M16(Pa,Va)`，BF16 add/result | 两个完整 BF16 partial 与结果 |
| BF16 partial、宽相加 | `R16(float32(partial16p)+float32(partial16a))` | 两 partial、FP32 sum、BF16 结果 |
| FP32 product、各自先舍入 | `R16(M32p) + R16(M32a)`，BF16 add/result | FP32/BF16 两套 partial、最终结果 |
| FP32 product、最后舍入 | `R16(M32p+M32a)` | 两 FP32 partial、FP32 sum、BF16 结果 |
| concat FP32、最后舍入 | `R16(M32([Pp,Pa],[Vp;Va]))` | FP32 context 与 BF16 结果 |

这里精确区分 API 的 operand/result dtype 与内部 accumulation。不能仅凭 BF16 输出宣称内部就是 BF16 accumulation，也不能仅凭 `.float()` 宣称已控制了 backend 的内部 reduction/precision 策略。FP32 partial 和 cast 前后全对象使舍入发生在哪个显式边界可检验；各环境 backend 状态必须记录。

另对全部 48×2=96 套固定有效输入，设计有限的纯整数参考：将每个 finite BF16 解码为整数乘二次幂，乘积和点积精确累加，再按明示 round-to-nearest-even 生成理想 BF16 结果。它共含 **17,731,584** 个标量乘积；只是一条数学参考，不能据其吻合反推底层使用了同样 accumulation。实数总和为零时单独记录 signed-zero 信息，不能借参考把 −0 与 +0 合并为位相等；非有限值按原检查拒绝。

两个解释器的跨环境差异同时包含原 native=2、SDK=1 线程设置。当前设计明确保留这个 bundle，不把差异单独归因于 Torch 版本、编译器或 backend；若后来需要分开线程因素，应另立有限对照，而不能改本次原控制定义。

## 原来源、有限范围与输出判据

原 24 帧、calls 4/5/6/11/12/13、两 ranks、fixture/physical-slot/head 映射全部保持。Native Q heads 0–7/8–15 对应 SDK rank 0/1，KV head 0/1 对应相同 ranks；不能按 frame 列表顺序假定 B4 行。全部 normalized input、Q/K/V projection、位置、RoPE、缓存和原源码/退出/控制证明继续引用完整原 descriptor plan 和已完成独立审计，不以 PV 局部来源替代原完整证明。

局部算术只需要重读明确的 **180 文件 /17,064,480 B**：native 每帧 P、effective V、pv_context、context_before_o（96 文件）；SDK 每 call/rank prior P、active P、prior V、active V、pv_prior、pv_active、context_before_o（84 文件）。这是新计算的有限读取子集，不能声称重新审计过全部 1,284 个文件。未来 admission 仍必须认证所有完整来源与先前真实终态、当前完整 serial catalog、准确原 metadata/文件身份；每个 PV 文件继续完整长度、EOF、SHA、形状、dtype 检查。

完整最大矩阵共 **1,560** 次有限求值，其中 Torch matmul 2,400 次、context add 936 次；不是全模型 forward。原单 tensor 16 MiB 上限保留。按每个完整结果/partial 分别写一文件的保守估计，产生 4,392 个小型派生 tensor，原始字节共 12,288,000 B，4 KiB extent 模型分配 18,284,544 B。设计另外限制 metadata 总量 8 MiB、总实际 artifact 分配 32 MiB、每文件 2 MiB，并检查目录和实际 st_blocks，超限停止。这是更窄的 prospective 数据预算，不是已测量的 RSS、时长，也不允许提高原 worker/service 资源上限；未来仍复用既有串行准入，不新增框架。

每项输出保留原输入 refs/全 SHA、完整 layout、解释器/线程/来源指纹、所有输出完整 raw/SHA、全域 differing words/bits、最大误差、平方误差及 signed-zero 计数；例子最多 8 个且明确不是 coverage。拒绝只保存平均误差、只看首个边界或只报告变好的 cases。输入改变、head/slot 错配、删除一个 frame、替换 dtype 或来源均需失败。

只有两套原输出均准确复现，且明确某个固定输入、固定环境、固定几何的受控变化产生对应完整输出差异，才可报告“该局部机制解释了哪些 frame/rank”。若多个变体都接近、shape 桥本身改变输出或异常 V 情况不同，就保留这些区分，不写唯一原因。所有阶段继续 `causal_operator_proven: false`，直到另有充分对照；原 SDK model exit 1 保留。整模型 BF16、cold4096、长解码、slot reuse/inactive、quality96、matched HTTP 与提速门槛完全不变。

## 本交付验证

`minicpm5_layer1_fixed_input_pv_design_v1.py` 是无 IO 的有限矩阵 builder；`minicpm5-layer1-fixed-input-pv-finite-design-20261111-v1.json` 是 641,391 B 的设计枚举，不含当前 runtime binding。它的全部几何已与已审计 summary 的完整 contract 中相应字段相等校验。

8 项纯测试通过（chunk `647567`，exit 0，0.038 秒），覆盖严格 24 帧与 JSON 类型、非自然 slot 顺序、36/60 控制计数、完整双 origins/双环境六式、异常 V case 保留、五条桥接、原 threads/stride、独立输入/输出预算和零准入声明。初次测试发现手算 4 KiB extent 预期有误，改为独立按 4,392 文件及较大的 SDK 输出计算；生产 builder 的计算未变。没有真实 PV 算术结果或因果/提速结论。
