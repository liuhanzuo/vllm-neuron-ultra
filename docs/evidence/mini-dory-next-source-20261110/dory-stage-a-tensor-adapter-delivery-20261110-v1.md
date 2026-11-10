# Dory Stage A 注入式张量适配器交付

本次新增 NativeTensorAdapter，实现 describe、integers、select、top1、iter_bytes。构造函数接收外部传入的原始 CPU torch 接口，本文件既不导入 Torch/NumPy，也不加载模型、构造运行器、申请准入或启动实验。既有 Stage A core 交付保持原样。

## 字节与行序

describe 返回原 shape、dtype、stride、storage_offset，不复制数据。仅接受 CPU strided 的 BF16、F32、I64、正尺寸 rank 1–3；拒绝量化、稀疏、设备转移、负 stride 和延迟 conj/negative 标志。实际元素字节宽度与完整元素数必须一致。

iter_bytes 仅序列化实际 BF16/F32。完整模式 rows=None 覆盖原张量的全部逻辑元素；抽样模式仅接受二维张量中真实、排序、无重复的行，并保留每一列。行序按逻辑索引排列，支持转置、列步长、三维维度置换、非零 storage_offset 和 stride=0 广播；不把 backing storage 的 padding 当作张量数据。

为了避免整个非连续张量的连续化副本，序列化器用真实 narrow / select 视图递归分块。每个 native clone 和输出字节块均不超过 65,536 字节。clone 明确使用 contiguous_format，之后通过 view(uint8) 和 uint8.tolist() 取得原始字节，无 BF16→float 转换、数值重新打包或 NumPy。输出必须是小端、按元素字节对齐的完整字节流；每个块及流结束时核对原 shape/stride/offset/data_ptr 未变化。

分块视图只用于复制观察结果，不会替换原模型前向输入，不创建零填充、缩短或补齐的模型输入。数据位模式原样保留，包括符号零、subnormal 和 NaN payload；有限值门槛仍由既有 ArchiveWriter 与独立 reviewer 检查。

integers 读取完整 int64 向量，保持大于 2^53 的整数精度，上限 2060 个元素。select 用原 int64 行索引执行 index_select(dim=0)，结果保持原 dtype 和全部列，要求独立连续存储。唯一 torch.tensor 分配用于这些行索引。top1 对原完整 F32 矩阵直接执行 argmax(dim=-1)，不截短词表，不做 softmax、转换或采样。

## 已实际执行的验证

47 项 stdlib-only 合成接口测试全部通过。实际测试退出码为 0，工具会话 55261，最终工具 chunk 07d387，报告运行时间 2.427 秒。

测试数据全部是在当前进程中构造的少量合成字节；FakeTorch 模拟 storage、逻辑索引和视图语义，不代表原生 PyTorch 执行。没有读取真实张量、权重、模型或旧二进制存档，没有写出张量文件。

覆盖包括：

- BF16/F32 原始位模式，包括符号零、subnormal、正负 Inf 和 NaN；随后传给既有 writer 时按原门槛拒绝非有限值。
- 转置二维、列步长、三维置换、非零 offset、单元素维度与广播 stride。
- 完整 131072 列 F32 输出，2560 列隐藏状态和四个缓存 head 的逻辑顺序；只复制真实指定行且保留全部列。
- 64 KiB 整边界、额外一个 BF16 word、宽行跨块、最后一个块和最后一列；任何 clone/输出块都不得超限。
- 只在消费 iterator 时复制第一块，不预先复制全张量；中途停止不继续分配后续块。
- select 保持 dtype、全列、独立存储；top1 的最后词表位置、相等最大值的首索引和非连续布局。
- int64 极值、超过浮点精确范围的大整数，以及错误 dtype/形状/元素数。
- 错行、重复行、乱序、bool 索引、越界、空集合、错误 narrow 返回、native clone 异常、复制后 alias/dtype 错误。
- 分块间或最后一次 yield 后 shape/storage_offset/data_ptr 改变；guard 失败必须中止。
- 适配器→既有流式 writer→独立 saved verifier 的小字节端到端回读对照。
- AST 与模块状态检查：没有 Torch/NumPy 导入，serializer 无 tensor/zeros/pad 等输入构造，没有浮点 cast、NumPy 或磁盘入口。

整个任务所有命令均从实验根目录经 nemotron9b_package_window_v27.py --observe --timeout 45 -- 包裹执行。未创建生产 runner、准入或 continuation handoff，也未占用 Nemotron 的模型资源预留。

## 剩余原生验证缺口

1. **真实 Torch 行为**：未来在允许的串行窗口，用冻结的原 CPU torch 版本验证 clone(contiguous_format) 的实际 stride、storage_offset 和新 storage，特别是 singleton / 非连续 / broadcast 视图；验证 BF16/F32 view(uint8) 的原始位模式、完整 narrow/select/index_select 行序。当前 FakeTorch 不证明原生 ABI。
2. **真实尺寸**：本次覆盖完整词表宽度 131072 和隐藏宽度 2560，但没有执行 Dory 的真实 2060 行前向或真实完整缓存。六次原始完整模型调用及原 224 个边界 / 114 个输出的字节对照仍待执行，不以小接口样例认定整个模型正确。
3. **源/模型/权重绑定**：注入接口由未来原生调用者负责身份绑定。适配器不证明原 torch / native library / oneDNN / 原权重 / 官方模型的实际来源，也不构造或修改原模型。
4. **并发值稳定性**：每块核对元数据和 data_ptr 可检测形状或存储改变，无法证明其他代码没有原地改写相同 storage 的数值。未来 worker 必须独占观察值，保持同步复制并绑定真实进程退出；禁止同时进行模型推进或外部修改。
5. **真实资源与开销**：64 KiB 是每次 raw chunk / clone 的上限，不是整个进程的内存上限。uint8 Python list、bytes、clone、调用栈及被观察模型会有额外内存与 CPU 开销，需另行测量和限定。当前没有吞吐、延迟或提速测量。
6. **实际持久化审核**：未来 worker 负责文件独占创建、容量、flush/fsync/close、异常保留、实际退出与稳定文件句柄；此前独立字节 core 仍需完整审核程序接入。这些不由适配器提供。
7. **连续整体门槛**：原 87 个检查点、七个 FP32 失败、冷缓存、长解码、槽位复用、完整质量及 HTTP 门槛继续保留。Stage B/C 与生产资格没有新结论。

新增源码与测试不改变原 Stage A 计划存档字节数。Dory 原七个失败仍保留，没有新增精度通过、根因或提速结论。
