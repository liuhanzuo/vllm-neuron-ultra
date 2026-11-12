# Spark 原生 provider / 串行 worker 源码交付

本次交付的是两个具体实现和 source-only 检查：`spark26_native_provider_v2.py`、`spark26_native_serial_worker_v2.py`、`check_spark26_native_provider_source_v2.py`。26 项 stdlib/mock 检查通过，真实工具 `8de814` / exit `0`。没有导入 Torch、safetensors、Neuron SDK、vLLM 或 ctypes，没有打开模型权重或既有 native 文件，没有操作真实进程、systemd 服务和串行锁，没有启动、下载或发布。本次检查仅创建并读取了临时目录内新建的三字节文本和 JSONL mock 输出。

## 具体入口与复用

`native_providers(Runtime)` 真正导入原 Torch、`safetensors.safe_open`、`neuronx_distributed.trace.trace.create_local_weight_qkv` 和固定 loader 的 `NeuronModelBase`，并加载原 `spark26_text_port_v2`。官方配置/模型加载沿用已经执行成功的 `run_spark26_native_meta_v1.load_official` 机制：原 revision 命名 package、原配置与模型文件、实际 `spec_from_file_location`/`exec_module`。没有 AST 提取、替代 SDK 函数或修改官方模块，也没有解除 v2 adapter/registration guard。

`WorkerContext` 自行验证实际 Python、独立 ACK 的 PID/start/argv/cgroup、原 parent-only fd 的锁证明、实际 systemd InvocationID/ExecStart 和资源属性、完整历史与现时 drain、磁盘/内存/output 限额。`Runtime` 仅接受这个具体 context 类型，完成实际文件哈希后才建立可用于科学 core 的 authority。科学 core 所需 `load_native` callback 在这里返回已由真实 factory 构建并验证的 providers；认证工作已经实现，不再交给未定义 callback。

原 descriptor 证明直接调用 `research/descriptor_serial_lock_proof_v1.prove`。进程观察、systemctl 输出解析、active-heavy 检查和 recursive empty-group 沿用 `minicpm5_layer0_execution_protocol_v4`。完整历史直接调用 `lfm3_nki05_serial_history_v1.resolve/collect_drain`，并使用其原 `history.serial.sources.load` 读取大历史 envelope/alias，不用新 worker 的 8 MiB receipt 限额裁剪历史。

本 worker 不获取/释放锁，不执行 systemd-run。`service_command` 只生成准确的命令 argv；外层控制器仍必须独立持有原锁，完成完整 predecessor proof，执行该命令，观察并 ACK 实际 worker，等待真实退出和两次稳定 drain，之后才释放锁。worker ACK 由外层在服务启动后生成，绑定 admission hash 和实际 owner；admission 不预先引用 ACK hash，避免不可构造的循环 hash。没有声称 worker 自己的 result 可以证明它已经退出。

旧 Qwen Service 的 `properties` 和 `argv` 固定 Qwen namespace，因此没有 monkeypatch 或直接借用旧 service 身份。旧 LFM runner 的 source/runtime/namespace 与 Spark 不同，因此只把它作为外层接入的参考，不能直接传入 Spark spec。新 spec/review/admission 具有 Spark 独立 schema；尚未生成真实 spec、review、admission 或任何 launch intent。

## 真实运行时认证范围和剩余闭包

旧 Spark meta worker 认证七个基础包、37 个 native 文件。新真实 sharder 与 loader 入口引入 torch-xla、torch-neuronx、libneuronxla、neuronx-cc、NxD、NxDI、vLLM、vllm-neuron、regex 等依赖，因此旧认证不能证明新路径。

本次仅读取 16 个必要直接包的 RECORD 文本，统计为 **6,390,223,265 B** package payload，其中声明 **849 个 native 文件、5,014,367,978 B**。数据位于 `spark26-native-provider-direct-distribution-metadata-20261112-v1.json`，SHA `1336881856135b8e2aec17bfdbede77c7dff9df475892638f2f69346a291604b`。**这些数字不是完整传递闭包，也不是这些二进制已认证的证据。** RECORD 声明数包括包内并未实际加载的库。

`merge_manifest` 从 root 冻结的 RECORD 重新构造实际 payload map；保留 package Python/native/data，不执行或重写包外 console scripts，不读取旧 pyc。原 SDK 的已审阅 Python 改动必须独立列在 `python_overrides` 并同时存在于 source pin map；禁止 override native hash。editable vllm-neuron、官方文件、worker/helper 来源必须在 `additional_files` 中独立 pin。额外第三方 ELF 依赖必须列在 `linked_native_files` 并预先哈希。

`Runtime.authenticate` 在真正串行 worker 内逐项读取全部冻结 payload 并校验 SHA256/尺寸/身份。其后 origin finder 委托真实 import finders，检查第三方 module 来源属于预先认证的 map；未知来源 fail-closed。真实 imports 后检查实际非系统 native mappings 必须已经哈希；它是加载结果核对，不能伪称阻止任意 ELF 初始化代码。系统 Python/stdlib 和 `/lib`、`/usr/lib` ELF 依赖明确属于 host 信任边界，未被本实验重新认证。

当前 **没有完整真实 manifest**：16 直接包之外的传递包、editable source、已改动 SDK Python 源码、额外 ELF 依赖仍需 root 的完整冻结/独立 review。新代码明确要求该 manifest，不会以“已有 16 个 RECORD”自动放行。动态 finder 安装、私有 loader 或未固定的依赖若超出当前 origin guard 会失败；这必须先由真实 import 前置诊断记录，不能绕过 guard 宣布成功。

新 worker 使用原 Python 3.10 的 `-S -B`，避免 `.pth` 在认证之前执行；认证后再显式加入原 SITE 和 `/home/ubuntu/src/vllm-neuron`。使用命名空间内永不创建的 pycache prefix。这里选择单进程 CPU native diagnostic 环境 `PJRT_DEVICE=CPU`、`NXD_CPU_MODE=1` 和空 Neuron visible cores，避免 import 默认选择 Neuron。这个组合尚未实测；不能冒用 Mini 原 torchrun TP2 的环境转换证明。检查保留真实 sharder 的两个 rank 数值路径，但未建立 TP2 distributed process group，未执行 collective 或芯片图。

## 不变的科学范围

继续调用冻结 `spark26_native_port_diagnostic_v3.run` 和 contract v2，完整保留：

- 36 层全部原 QKV、三种原输入形状、两个 rank：216 组投影诊断。
- 原 534 事件 × 两种 cache × 两 rank：2,136 步；包括 520 连续 decode、511 边界、4095 最大位置、slot reuse/dummy isolation。
- 每个原 decode 的真实 host `_reordered`/`_sort_inputs` 与 restore：2,108 组。
- 原 steps 4/5/6/7 的 24 种物理行排列 × 两 cache × 两 rank：384 组。

共 4,844 条 emit。新的 sink 对四个真实事件名、各类别数量和输出字节数分别限制；不能通过少跑事件生成完整 receipt。原 logits 门槛、cold-cache、long-decode、slot、quality96 和 matched HTTP 待执行门槛保持。BF16 投影差异继续仅作诊断，不新增严格 gate。本源码和这些 mock 测试都不构成全模型/芯片正确性或提速证据。

## 明确预算提案

| 项目 | 拟固定限额或需求 | 证据/范围 |
|---|---:|---|
| 原权重永久存储 | 8,224,192,408 B | 一套原五 shard；本 worker 不下载/复制 |
| 运行时认证读取 | 最多 16 GiB | 逐个文件、1 MiB buffer；16 直接包声明 6.390 GB，完整闭包尚未固定 |
| 原权重认证读取 | 8,224,192,408 B | 沿用 OriginalWeights；完整 hash 后才 safe_open |
| MemoryMax / High | 16 / 14 GiB | systemd 强制；不是已实测峰值 |
| swap / virtual AS | 0 / 24 GiB | 强制禁止 swap，AS hard/soft 相同 |
| CPU | affinity 6–7、nice 19；库线程 1 | 诊断，不计性能窗口 |
| 整个 worker wall time | 10,800 s | finite deadline 与 systemd RuntimeMaxSec；未实测完成时间 |
| 输出总 allocation | 48 MiB、最多 64 entries | 包括 admission/proof/ACK/result/log/tmp；不能遗漏原历史证明成本 |
| 单文件 | 8 MiB | RLIMIT_FSIZE 与 sink/scan 同时约束 |
| events | 16 MiB、每条 3,072 B | 最多 4,844 条共 14,880,768 B；最大类别 6,561,792 B 小于单文件上限 |
| logs / receipts / temporary+dirs | 8 / 16 / 4 MiB | 分类 allocation scan；没有 tensor payload 或编译预算 |
| 根盘 floor | 28 GiB | 不扩容 |
| NVMe floor + Qwen reservation | 30 + 2 GiB | worker 监控都保留；当前 free 未在本子任务观察 |
| MemAvailable floor | 2 GiB | 每次完整监控读取实际 meminfo |

外层启动前必须另外预留完整 48 MiB 输出空间；若原权重尚未落盘，还需按一套原权重 + 下载过程峰值另作下载准入。本 worker 不提供下载器、不隐含下载额度，不包含 TP2 派生 checkpoint、编译缓存或 NEFF 配额。历史“放一套权重后还有 4,023,498,856 B”只是一条旧外部观察，不能用于当前 admission。

缓存中的一个 full K/V pair 是 32 MiB；原步骤和排列重放同时持有 prior/new/expected/returned/clone 多组张量，另有官方 DynamicCache 和临时 inverse/gather。QKV 原权重单层 30 MiB，还有 clone、rank partitions 和张量计算临时值。16 GiB hard limit 包括这些张量、runtime imports、页缓存、完整历史/manifest 的 Python 对象，不以单个 tensor 尺寸冒充实测峰值或完成保证。若预算不足，应保留失败并提出新的完整预算，不能删减科学范围。

每次 core guard 均重新核对真实 retained-fd 和 worker 进程；较昂贵的 systemd、latest-handoff、output/storage/memory 和 runtime identity 全扫描以一秒间隔执行，关键开始/结束强制 full。hash/科学不是性能窗口。间隔内不是连续监控，这个限制没有隐去。

## 本轮检查和 root 待办

26 项检查覆盖 RECORD 数据保留和外部脚本/pyc排除、未知/重复/未哈希条目拒绝、16 个真实直接包必需、native hash override 拒绝、遗漏固定 host 源拒绝、错误 map 预算拒绝、新建小文本的实际 hash reader、假 authority 不能创建 Runtime、未知 import origin 在返回 loader 前拒绝、真实 import 目标和无 AST 执行机制、全部 emit 名/数量、事件超量/超长与不完整 receipt 拒绝、实际 service argv 与预算、错误 Python 在 proc 前拒绝、无 ACK hash 循环和无 launch/acquire/release。

下一步先 root 独立 source review。随后仍需完整依赖 manifest、原始 checkpoint 的存储准入及落盘证明、完整当前历史绑定、真正外层 retained-lock/start/ACK/exit/drain/release 控制器的 Spark 接入与检查、全输出证明预算验证。之后才能考虑排入原串行队列执行 import 前置诊断和完整 native tensor diagnostic。当前 LFM r2 继续拥有原串行锁；本交付 **不可直接准入，不声称可运行或已执行**。

## v2 补充与不可变性

v1 已经过 20 项检查（99381d/0），因此保持原文件及摘要不变；本交付创建 v2 后继，26 项检查通过（8de814/0）。真实 native 映射核验现在解析 /proc/self/maps 的 device major/minor 与 inode，要求同时等于预先哈希的对象身份和当前磁盘 stat。拒绝 deleted 映射、零/无效 inode、设备不匹配、同名旧 inode 以及哈希后替换的文件。不能再拿当前同名文件的身份替代已加载对象。首次 import 后、科学结束及每次一秒完整监控都会重新执行 native mappings 核验。系统 Python/stdlib 与系统 ELF 信任范围不变。新增五项 mock 映射案例和一项预算作用域检查；不读取本机实际 maps 或 native 对象。

预算字段现在明确为 max_runtime_authentication_read_bytes / MAX_RUNTIME_AUTHENTICATION_READ，数值仍为 16 GiB，只限制一次完整 runtime manifest payload 哈希所列文件字节之和。它不包含原五 shard 的 8,224,192,408 B 哈希、不包含 safe_open/header/tensor 的后续读取、guard 反复读取的 proc/metadata/source，也不包含 OS page-in 或应用再次打开已认证文件的 I/O。**整次诊断总读取量没有被宣称小于 16 GiB。**现有 6.390 GB 直接包统计与 8.224 GB 权重哈希相加尚且不包括完整闭包和后续读取；不能把这一和数用作整次 I/O 上限。运行时 wall/resource/output 限额不因此放宽，剩余闭包和真实 admission 仍未完成。
