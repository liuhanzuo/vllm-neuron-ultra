# Qwen3.5 精确形状缓存写入实验：源码交付，尚未运行

本交付准备了一个独立的缓存写入原语实验，用原始 Torch `store_slots` 对照已冻结的 predicated NKI 内核。没有导入 Torch、NumPy、Neuron 或 NKI，没有编译、设备运行、加载权重或修改运行中的 Qwen r10。下面的图数、调用数和资源值是未来实验合同，不是已经完成的实验结果。

交付文件：

- `qwen35_predicated_slot_protocol_v1.py`：精确形状、完整正确性矩阵、计时矩阵、计时边界、完整字节比较、资源预算和冻结源码指纹。可用标准库读取。
- `qwen35_predicated_slot_device_leaf_v1.py`：供已有串行 runner 调用的 `run_case`；无 CLI、无自行启动队列、无新的审批/生命周期框架。
- `test_qwen35_predicated_slot_source_v1.py`：16 项标准库源码/模拟检查，全部通过。包括完整矩阵、字节比较负例、计时边界、配对输入、磁盘下限以及在 native 导入前拒绝执行的路径。这些检查不构成内核或设备正确性证据。
- 同名 JSON manifest 保存交付文件和原有证据的 SHA256，以及可机器读取的完整矩阵。

## 保留的真实形状和精度

每个任务针对 TP2 配置中的一个真实 rank、一个缓存张量。每种布局分别编译 original Torch / predicated NKI，以及普通输出 / 真实持久状态别名两种模式，共 4 张图。K/V 共用相同形状，测试一份原语布局；这不是运行全模型的 48 个缓存张量或 TP2 通信。

| 布局 | 实际形状 | 类型 | 一个张量大小 |
|---|---|---|---:|
| 卷积状态 | `[4,3072,4]` | BF16 | 96 KiB |
| 循环状态 | `[4,8,128,128]` | FP32 | 2 MiB |
| KV 128 | `[4,1,128,256]` | BF16 | 256 KiB |
| KV 256 | `[4,1,256,256]` | BF16 | 512 KiB |
| KV 512 | `[4,1,512,256]` | BF16 | 1 MiB |
| KV 1024 | `[4,1,1024,256]` | BF16 | 2 MiB |
| KV 2048 | `[4,1,2048,256]` | BF16 | 4 MiB |
| KV 4096 | `[4,1,4096,256]` | BF16 | 8 MiB |

保持 B4；部分请求仍以 `-1` 标记，不能缩成另一个 batch shape。FP32 循环状态不降精度。模型原来的 KV bucket 切片及最大长度尾部拼接不修改；这个原语实验也不声称已经验证尾部拼接、prefill、GDN、attention 或完整模型。

## 正确性：625 种映射 × 两种数据配置

对每张图使用完整 `{-1,0,1,2,3}^4`，共 625 种映射，包括全部 24 种排列、所有部分活动/全不活动组合、重复目的槽位和全部更新写向同一槽位。顺序严格保留：原始 `store_slots` 逐行更新，最后匹配该目的槽位的更新行获胜；不匹配的槽位保留原字节。

每个布局分别使用 raw storage 和 finite 数据。直接复用冻结的 `probe_slot_cache_nki_read.array_for` 生成器：raw BF16 包含完整或显式异常位型，raw FP32 包含正负零、无穷、NaN payload、subnormal 和一般位型；finite 配置包含有符号零。转换采用整数视图，避免先转换浮点数而改变 payload。

普通模式把编译结果与原始 Torch `store_slots` 输出逐字节比较，并检查 cache、updates、slots 输入没有被改写。持久状态模式将真实 `torch.nn.Parameter` 注册到 `input_output_aliases={source.cache:1}`；同一图同时返回 `cache[slots.clamp_min(0)]` 的旧值和更新后的完整状态。每一步验证旧读值、新输出、实际 `compiled.states._parameters` 中的 Neuron 状态、输入不变性。必须实际处于 `privateuseone`，不能用 Python 的状态副本代替设备状态检查。连续 625 次调用推进状态；切换数据配置时通过真实全槽写入复位，并再次检查输出及实际持久状态。

比较检查 shape、dtype 和完整 byte buffer；SHA256 仅记录证据，不替代直接完整比较。正负零或不同 NaN payload 必须判为不一致。出现不一致立即失败，不放宽为误差阈值、不跳过案例、不自动重试。

## 计时：独立的代表映射，匹配的 A/B 条件

完成当前布局和模式的全部 raw/finite 正确性检查后，才使用显式列出的 16 种代表映射计时。每个分支 2 个 warmup cycle、10 个测量 cycle，每 cycle 16 次调用。两支使用完全相同的初始状态和输入序列；持久状态在每 cycle 前经真实全槽写入复位。10 个测量 cycle 中 original-first 和 candidate-first 各 5 次。

计时从 `compiled(...)` 调用前的 `perf_counter_ns` 开始，到所有返回张量的 `detach().cpu().clone()` 完成后结束。包含主机调度、输入传输、设备执行、输出传输及主机复制。guard、输入生成、原始 Torch oracle、比较、持久状态读取、复位、编译、文件写入均在计时外。每个 cycle 的时间是 16 次独立调用的计时之和，中间有正确性检查；不是连续满负荷吞吐，也不是设备纯内核耗时。

两种模式分开报告各 cycle 的 reference/candidate 比值及这些配对比值的中位数。普通模式输出一个完整张量，alias 模式输出旧读值和新状态，因此不直接把两种模式混为一种计时。即使全部通过，也只能报告对应原语的同步主机调用比值，不能宣称全模型提速、TP2 提速、HBM 实测节省或零拷贝。

如果同步主机调用时间基本相同，也不能据此证明编译器已经融合了原始 writer；这种结论需要单独的编译或设备证据。本交付不新增 profiling 框架。

## 有界工作量和资源要求

| 项目 | 每种布局 | 全部 8 种布局 |
|---|---:|---:|
| 编译图 | 4 | 32 |
| 正确性显式调用 | 5,000 | 40,000 |
| warmup 显式调用 | 128 | 1,024 |
| 计时显式调用 | 640 | 5,120 |
| 已检查的持久状态复位调用 | 28 | 224 |
| 显式运行调用合计 | 5,796 | 46,368 |

每个布局一个新进程，8 个进程串行；最多同时加载两张图。编译器内部工作不计入上表的显式调用数，仍受整个进程的资源限制约束。

拟定边界为单个 NeuronCore 0、2 CPU、Torch 1 线程、MemoryMax 6 GiB / MemoryHigh 5 GiB / 无 swap、每布局 900 秒，总超时上限 7,200 秒。每布局输出、编译目录、cache、temp 和日志合计最多 1 GiB，8 布局合计最多 8 GiB；单个 `model.pt` 上限 128 MiB，完整报告序列化上限 16 MiB。叶函数在边界调用 guard，但编译期间的 cgroup、时间、存储监控必须由已有 runner 持续执行，不能依赖叶函数轮询。

原有 **root 28 GiB / NVMe 30 GiB 空闲下限保持不变**。安排整个独立实验时，输出所在存储还需保留原来 30 GiB reserve，加 8 GiB 输出预算、2 GiB 额外余量，即 **40 GiB 初始可用空间要求**。不能将 2 GiB 余量误作新的全局下限；未来调度必须按实际文件系统逐项核对。LFM 的独立 105 个任务和 **54 GiB 要求完全保留**，不能由此实验替代、折算或降低。

旧 predicated device 实验的 cgroup 峰值为 1,331,191,808 bytes（约 1.24 GiB）、30.12 秒，9 图/43 检查；旧 alias 为 2,633,613,312 bytes（约 2.45 GiB）、108.46 秒，12 图/60 检查。只读核对旧目录的逻辑文件总量分别为 3,523,208 和 117,803,218 bytes。这些是较小旧任务的测量，**不证明新的 625-map Qwen 任务能在上述拟定上限内完成**。尤其 raw BF16 生成器逐行检查位型覆盖，全部字节校验和报告序列化也消耗主机时间。实际超限应报告失败并保留证据，不缩减625映射、不放宽字节检查，也不静默放宽全局资源下限。

## 已复用的机制与未完成的串行接入

冻结内核是 `slot_cache_nki_predicated.py`，wrapper 是 `slot_cache_nki_predicated_torch.py`。内核仍读取原值和更新值并写出完整输出，不声称原地零拷贝。原始 oracle 为冻结的 `qwen35_text_port.store_slots`。alias 绑定和实际设备状态检查沿用已有 `probe_slot_cache_nki_predicated_alias.py` / `probe_slot_cache_nki_read.py` 的模式。使用原有 legacy `neuronxcc.nki`，拟定解释器 `/home/ubuntu/venvs/vllm_neuron/bin/python`，没有把内核改成 NKI 0.5。

**目前没有可直接使用的兼容串行 dispatcher。** `run_slot_cache_nki_device_queue.py` / `run_slot_cache_nki_read_queue.py` 绑定旧前序任务、路径和固定任务表；`run_lfm3_nki05_device_serial_v2.py` 校验 LFM105任务与 NKI0.5 SDK，不能直接派发本 Qwen legacy 叶函数。本交付没有改写它们，也没有另造审批或生命周期系统。

后续由现有 owner 增加明确的 Qwen 派发分支，使用当前完整前序历史和原串行锁，核对新的源码指纹、解释器依赖、原有空闲下限及独立预算，绑定独占临时/cache/输出路径和真实外部监控。它逐布局调用 `run_case(name, output_dir, guard=..., save_report=..., record_artifact=...)`。这三个回调必须复用 owner 已有的归属检查、完整报告保存和 artifact 记录；传入普通回调本身不构成运行授权或串行所有权证明。最终状态仍须由已有 owner 的进程退出、子进程清理和外部结果检查确认。

本次只完成源码和16项标准库检查。Qwen 精确形状设备正确性、FP32持久状态别名、32图的编译/存储/时间测量、原语配对比值以及完整模型原有 cold-cache / 长解码 / 请求槽复用 / 质量 / HTTP 门槛均仍待实际验证。本次没有新增可发布的提速倍数。
