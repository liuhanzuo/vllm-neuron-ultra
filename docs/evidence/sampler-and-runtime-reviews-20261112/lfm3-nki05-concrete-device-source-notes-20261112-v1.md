# LFM3 NKI 0.5 具体设备执行接线：源码交付，尚未运行设备

本交付把此前已验收的科学叶子接到实际 serial outer、worker、91 个有父进程 ACK 的子任务及具体 `Context(DeviceContext)`。它不是另一个只有回调的接口。新设备任务尚未产生实际 spec、root runtime review、admission 或启动命令；没有获取串行锁、创建服务、导入原生库、编译设备图、读取模型权重或旧二进制 artifact。本次运行的仅是有资源限额的标准库/mock 测试，以及旧 JSON、Python 源码和安装 metadata 文本读取。

`run_lfm3_nki05_device_serial_v1.py`、pipeline、runtime、protocol、budget、SDK 和 full-CPU metadata 共 7 个生产文件在首次测试前冻结，之后没有改动。此前交付并已由 root 独立验收的 11 个 device scientific 源文件也未改动。扩展叶子及补充控制继续使用 V2。原始 r2 的 595 个源码 pin 全部保留；本次 additive map 为 616 个 Python 源文件，新增 21 个，均已逐文件只读复核。44 个可静态解析的本地 import 源文件进行了传递检查；该数不被宣称为 native 动态执行闭包。

## 实际接线与拒绝条件

outer 在导入串行 history、连接 user bus 或尝试原 singleton lock 之前，先独立消费固定实际 r2 的完整科学结果、所有 874 个 child exit/outcome/ACK、原/new/candidate 全量 fixture 和 scientific byte gate、独立三臂 control、warning receipt、worker/service/outer/cleanup 及 root 外部记录的真实工具退出。固定 r2 spec SHA 为 `955c2267d1ac27bb1af5f680b4e0acefa4355c05bdfa874cf4f46f4381116630`，admission SHA 为 `bcc175f2aeb21fdfb0fd4c0fa585336ef6b7996c3666b6596ffc4a78f7b026c5`。仅 original_fixture 或 new_fixture 完整通过不能满足这一关。

未来 root 的外部退出 JSON 必须放在 research 下，包含：`reviewer: /root`、`actual_tool_exit_observed: true`、精确 `spec` 引用、`actual_tool` 中 `kind: tool_session`、`session_id: 42892`、非空实际 `terminal_chunk` 和 `exit_code: 0`，以及原 outer 和 worker 的实际完整 owners。该 JSON 不能在当前仍 live 的 r2 上提前填写成功。它只是新核验器明确的未来字段契约，不是本交付创建的结果。

取得原 descriptor lock 后，outer 继续独立解析完整旧 history 和未来新 catalog/handoff，要求旧 V76 全部 121 行保持原序，精确 r2 的实际工具退出只出现一次，fresh 检查 outer、worker、874 children 已 gone，且 r2 cgroup 为空。再做完整历史 drain、磁盘实际空闲/挂载检查、独立资源检查、创建全新 namespace 和保存 admission。原 `descriptor_proof`、`acquire`、`drain_and_release` 及 OwnedService 的 launch/sample/cleanup/drained 保持原函数 AST 一致。

OwnedService 的 acknowledge 是明确改动：允许精确固定 `/usr/bin/env -i ...` 到固定 Python worker 的正常 exec 过渡；该过渡必须保持实际 PID、start ticks、PPID、cgroup 和 service InvocationID，仍限原 60 秒，未知 argv 直接失败。这一变动修复了 service-type=exec 先看到 env 的竞态，未放宽其他 PID 或服务身份。

worker 顺序运行 7 phases。`run_one` 实际 Popen、固定 argv/clean env、实际 child_identity ACK、父观察 wait/exit、原 stop_child 的 pidfd/descendant drain 全部接入。未取得 ACK 的 PID 不由 child helper 信号操作；失败转到原 service 整体 drain，并持续保留锁直到物理 drain 证实。失败 outcome 被保留，后续科学任务不继续。

具体 Context 每次科学入口/存盘/编译入口/产物认证检查实际 child 和 parent 身份、descriptor、service、active-heavy scope、最新 handoff、source/canonical pins、磁盘/时间界限，并明确拒绝 simulator CompactWriteback/WritebackProbe。其科学进度 JSON 只能写当前 task 的精确路径；产物只能来自当前 fresh phase 的 `model.pt`，用 bounded streaming SHA 认证并保存包含实际 child/admission/task/shape/dtype/software/source 的独立 receipt。phase 和最终审计再独立重读实际 outcomes、ACK、SDK manifest 绑定、artifact receipts 和科学完整性。

## 原矩阵全部保留

| Phase | 编译图 | 严格检查 | 实际子任务 |
| --- | ---: | ---: | ---: |
| raw_device | 4 | 36 | 1 |
| regression_device | 9 | 43 | 1 |
| regression_alias | 12 | 60 | 1 |
| extended_device | 42 | 2520 | 42 |
| extended_alias | 24 | 192 | 24 |
| fp32_alias | 6 | 222 | 6 |
| wrapper_controls | 16 | 418 | 16 |
| 合计 | 113 | 3491 | 91 |

第一张实际设备图仍为 BF16 `(4, 7168)`。原 BF16/FP32、全部 65,536 个 BF16 bit patterns、单行原始 dispatch、多行 typed dispatch、int64 越界先归一化再 narrowing、最后写者、inactive、persistent alias、状态复用、conv 字节一致和注意力有限值等条件不删减。注意力舍入差异仍仅为诊断信息；conv 输出字节一致仍是硬门槛。新增独立报告验证还核对完整 output hash、连续 step、完整 alias input/state hash 及 reference/candidate fixture 配对。

CPU 前置仍是 1,648 + 2,068 个原案例三次全量 pass，各 291 批，另 1 个 controls child，共 874 次实际退出。新设备阶段不会替代原模型冷缓存、长解码、请求槽位复用、参考/候选性能矩阵；原 model required_gates 完整放入 protocol manifest。LFM 4096 target 继续待实现/验证，旧 2048 或 RNJ 4096 不赋予 LFM 4096 资格。

## 有限预算和估算依据

新设备预算独立于 CPU 小元数据预算：MemoryMax 16 GiB、MemoryHigh 14 GiB、swap 0、AS 24 GiB、TasksMax 256、CPU 6-7、nice 19；单 `model.pt` 256 MiB、单 compiler file 2 GiB、全部 fresh output 24 GiB、32,768 files、元数据/日志 8 MiB。root 空闲下限 28 GiB，NVMe 空闲下限 30 GiB；启动时另预留全部 24 GiB 输出。whole 12 小时、每 phase 4 小时、indexed child 30 分钟，前三个原 bulk phase child 各最多 4 小时。全部是拒绝超额的有限上限，不是成功所需资源已经测量或证明。

只读历史 JSON 依据：旧 device plan 使用 14/12 GiB、swap 2 GiB、每 phase 4 小时；旧 9 图 standalone 最大 compile_seconds 为 3.617127252276987、合计 25.233170510735363；旧 12 图 alias 最大 30.67484271293506、合计 89.10859228856862。FP32 B16 SSM 单份状态按形状为 40 MiB，旧 B4/H4/4096 两份 BF16 KV 为 32 MiB。这些只约束粗估，不代表新 SDK 的编译峰值。`measured_new_compiler_peak_bytes` 和 `measured_new_artifact_bytes` 仍为 null。

安装前缀有 171 个 dist-info。仅由完整 RECORD 声明计算，去重非 pyc 文件为 39,825 个、4,953,771,217 字节。每 fresh child 都重新完整认证，因此 42-child extended_device phase 声明重复读取量为 208,058,391,114 字节、1,672,650 次文件校验；全 91 child 为 450,793,180,747 字节、3,624,075 次文件校验。这不含 metadata/source/PID/descriptor 重读开销，不是实测磁盘读量或耗时，热页缓存影响也未知。没有为减少此开销删减科学矩阵或省略 fresh-child 校验。

SDK_VERIFY_SECONDS 为 3600，但它不能延长 child/phase/whole 的界限：真实约束为 SDK 剩余、parent child deadline、当前 phase 剩余和 whole 剩余的最小值；indexed child 最多 1800 秒，且这包括 CPU predecessor audit、SDK 认证、编译和全部检查。原 bulk child 的 SDK 认证仍最多 3600 秒。任何超时都按失败保留，不能被解释为科学通过。实际认证/编译成本是否能落在这些预算内尚未知。

## 安装前缀与 editable 的边界

`lfm3_nki05_device_sdk_v1.py` 提供具体 metadata_manifest/validate_manifest/verify。准备 manifest 仅 bounded 读取 METADATA 和 RECORD 文本；当前生成的文本 manifest 不声称 native payload 已验证。未来真实 owned child 才逐文件按完整 RECORD SHA/size 有限流式认证，并只接受明确 root review 绑定的 `.py` 源码修正，不接受二进制 override。SDK 认证证明必须绑定当前 admitted manifest 和同一 child。

本前缀所有 RECORD 中实际 pyc 行数为 0。新 verifier 明确拒绝出现任何 `.pyc` RECORD 行，不扩大旧 NumPy 单个 source-backed bytecode 豁免。Context 仍持续检查原 absent PYTHONPYCACHEPREFIX/`-B` 隔离，并拒绝已经加载的 sourceless `.pyc` 模块。

两份 editable finder 在 Python site 启动期执行，现已加入 additive source map。它们分别将 vLLM 映射到 `research/vllm016-source/vllm`，将 vllm-neuron 等映射到 `research/vllm-neuron-053`。`vllm 0.16.0+empty` metadata 不表示缺失源码。当前 primitive route 直接使用 Torch、Torch-NeuronX、NKI/NeuronX-CC、NumPy；对 Torch-NeuronX、NKI、NeuronX-CC、Torch-XLA 普通 Python 源码的静态搜索未发现直接 vLLM import，这不证明未来动态加载闭包。

RECORD census 只覆盖安装包所记录的前缀文件，不覆盖 editable finder 所指向的外部源码，也不宣称认证整个 OS、system Python 或 `/opt` 动态库。两份 finder 本身已明确 pin；若实际新设备 route 动态载入其他 research 下源码，现有 loaded-local-source guard 要求其在 additive map 中，否则失败。真正 serving route 的 vLLM/vllm-neuron 外部源码、动态依赖及加载兼容性仍须独立绑定和验证，不能据此交付宣称已经完成。

## 实际 source-only 验证与保留的失败

V2 检查脚本通过 51 项，实际 session 99225，terminal chunk `8fbb6c`，exit 0，13.464 秒。测试使用 CPU7、nice19、AS 256 MiB、CPU 60 秒、单文件 2 MiB 上限，并在新模块导入前阻断 Torch/NKI/NumPy/ctypes/vLLM。完整 CPU 世界是合成 JSON，不读取 live r2 的科学输出。设备科学值同样是用于测试拒绝条件的合成记录，不是设备结果。

覆盖包括 874-child 完整 CPU metadata 正例和 partial/missing-exit/fixture漂移/byte失败/ACK漂移/outer非零阴性；全 91 task 的完整正例与最后一行丢失；每个 phase 的严格字段失败；所有实际 receipt/SDK pin 绑定；同 PID/start 的 env 过渡及未知argv/超时/PID复用；child实际非零/未ACK/失败drain；原锁机制 AST 等价；bounded artifact hash（仅mock字节）、owned JSON 防覆盖及在 drain 未证实时不释放。

首次 V1 测试自身错误也完整保留：它试图用 StopIteration 逃离保留锁的 drain loop，而原实现正确吞掉等待中的任意异常，导致测试循环；22 项已完成后该受限过程实际 exit 137，session 37829，terminal `84b5dd`。V2 只改测试为有限两次 drain：第一次失败，重试时断言尚未 unlock，第二次实际 mock drain 成功。生产源码未改。早先对已完成测试数误报 25，失败 receipt V2 已明确修正为 22，同时保留 V1。预解析阶段发现的测试 dict 参数语法笔误在首次导入/冻结前修正，记录 `ae5a98/1` 保留在 pretest manifest。

接下来需要 root 独立读取全体源码/diff，复跑指定 V2 测试，等待实际 r2 完整退出后生成精确 future exit/catalog/handoff/spec/review，再在新鲜独占 serial lease 下做真正的 SDK 和第一张设备图。当前 source-only 成功不赋予 runtime admission、device/alias/model/4096 资格或新增提速倍数。
