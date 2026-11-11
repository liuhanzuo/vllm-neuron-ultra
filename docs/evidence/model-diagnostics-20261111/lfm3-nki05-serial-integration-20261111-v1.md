LFM3 NKI 0.5 搬运诊断现在有独立的单进程 worker 和根串行启动入口。新增的是可调用源码；本轮没有准入、获取锁、启动服务、运行 native/NKI/模型，也没有生成未来 Mini 退出结论、实际 LFM spec/review、catalog 或 handoff。

入口：`run_lfm3_nki05_transport_serial_v1.py`。根任务未来用 `python3 -B ... --mode outer --spec <实际spec路径> --spec-sha256 <实际SHA> --review <实际review路径> --review-sha256 <实际SHA>` 调用。worker 模式由该 outer 的唯一 systemd 服务启动，不能直接绕过它运行。spec/review 的格式定义位于 `research/lfm3-nki05-serial-interface-schema-20261111-v1.json`；这是 JSON Schema，不是已批准的运行实例。

准入需要当前 Mini 的实际外部退出及排空记录。`lfm3_nki05_serial_history_v1.py` 复用现有 SDK completed metadata validator、serial catalog validator 和 historical physical evidence validator。固定保留 v45 的全部 85 条历史记录，额外要求 catalog 中出现当前 Mini session 58352 的真实退出条目、原 outer/coordinator 身份和 service。退出码 1 等失败状态原样保留，不转成成功。最新 catalog/idle handoff 必须由根在实际结束后绑定；不能使用目前只包含 running 信息的 handoff。

历史验证会读取原 SDK 科学计划以供其已有 completion validator 核对，但其模型/库 source map 仅是历史来源记录，不成为 LFM 执行 source closure。新执行依赖是入口实际导入的本地纯元数据 helpers、原输入生成器及其 protocol、诊断源码和相关 SDK 文本文件。`execution_source_paths()` 在新鲜的、只导入此入口的准备进程中返回这些文件名；根在准备实际 spec 时散列它们。完整数量必须小于 1024，运行前及诊断后还会验证所有已加载本地 helper 均在 reviewed source map 中。当前 import-only 检查为 241 项，没有 native 导入。

串行实现复用原 `serial-enrollment.lock`，device/inode 固定为 `66306/8294421`，只打开现有文件，不创建或更换 inode。通过原 `descriptor_serial_lock_proof_v1.prove` 和原 `protocol_v4.proc` 验证实际父进程、fdinfo、完整 `/proc/locks` 与文件身份。outer 连续保留同一个 CLOEXEC descriptor，worker 不继承它；worker 核验该实际父进程持有的 descriptor。代码不再使用会把无关全局锁数量变化当作本锁变化的旧简化 guard。

先在锁内完成完整历史 owner/unit 两次 fresh drain，每个已知 owner 都检查 PID/start 已消失，每个 unit 都读取实际 terminal 属性和完整空 cgroup；两次结果必须相同。原始历史树不会被冒充为本次观察。outer 在整个工作期重新查询原 latest-handoff 身份，除检查已读文件未变之外，还拒绝新出现的更新 handoff、替换 inode 和身份变化。worker 在前后准入检查也执行同一检查。根在运行过程中应通过实时消息/实际工具状态报告进度；若另行发布新的 latest handoff 或改动 spec 固定的 canonical 文件，此 worker 会按冲突失败并安全排空，不能把正常并发发布悄悄放行。

资源保持原要求：CPU 6–7、nice 19、服务 MemoryMax 10 GiB / MemoryHigh 8 GiB / swap 0 / TasksMax 32、wall 与 CPU 上限 1800 秒、地址空间 12 GiB、单报告/日志 256 KiB。服务属性由 outer 和 worker 同时验证；worker 核对实际 hard/soft rlimit。outer 自身也固定 CPU/nice/CPU/AS/文件上限，其整个有用工作窗口最多 1800 秒；清理阶段继续保留锁直到证实排空。系统解释器只负责元数据 outer；唯一 native worker 固定 `/home/ubuntu/venvs/vllm_neuron_053_py312/bin/python`、`-B`、clean environment、单线程数学库、trn1 target 和 precise FP。没有模型、设备或编译器任务入口。

输出在根目录下全新的 `lfm3-nki05-transport-...` 目录，总量上限 4 MiB，保留原 28 GiB 根盘余量要求。完整 predecessor 证据使用最多 16 个 128 KiB 压缩块，每个块单独 SHA，清单固定顺序、压缩总 SHA、原始总 SHA 和长度；解码上限 32 MiB。每个 JSON 仍不超过 256 KiB；worker 在 native 导入前重新验证并完整拼接。不得删字段、截断、提高上限或把旧采样结果替代 complete proof。任何超限均停止。

已用当前 Mini 的真实保存历史 metadata 做容量验证：通过固定 live → admission → observed_drains 引用核对 `/mnt/nvme/minicpm5-layer1-operator-native-r2-catalog256-20261111/predecessor-drain.json` 的 SHA `df20574f3de355bad20e90daf36b6aec5c63bb069accdd960b6c91cf6bce9855`，文件 618,037 字节。完整解码 1,319 个历史 unit、1,753 个 owner，并加入全部 85 条 catalog、721 条来源及重复 owner，得到保守代表样本 13,486,365 原始字节 / 796,076 压缩字节。分为 7 块，最大块 JSON 174,826 字节，总块和清单 1,063,063 字节，完整还原成功。缺块、乱序和内容破坏均被拒绝。该验证仅处理已保存元数据，不是本次 fresh drain。

失败行为有明确的实际代码边界 `execute_diagnostic`：`diagnose()` 返回 `complete=False` 或 `passed=False`，先原样保存 `diagnostic-result.json`，再记录失败并让 worker 非零退出；不能生成 `worker-result.json` 成功记录。outer 除实际 service exit0 外，还要求诊断两标志均为真正的 `True`、32 条结果完整、65,536 BF16 编码、原 SHA fixture、六项负控和算术/转换控制全部通过。任何异常、信号、超时、资源/源码/状态变化均保存失败，禁止模型资格或新提速声明。

清理沿用 `retain_until_drained`：先验证完全相同的启动命令、attempt、service invocation、owner，再停止自己的服务并检查全部子进程/cgroup。工作 guard 失败不能阻碍这种有明确归属的清理。如果无法证明排空，保持原锁并等待显式重试信号；绝不在普通 finally 中直接解锁。服务实际退出由 outer 记录，outer 自身的实际退出仍须根通过原工具独立观察；没有自证退出或自动后继实验。

标准库验证命令：`python3 -B check_lfm3_nki05_transport_serial_source_v1.py`。验证包含真实 CLI 分派到合成失败诊断的非零子进程（实际退出码 1、失败结果保留、成功文件缺席）、资源/argv/环境拒绝、完整 85 行和当前 Mini 真实 terminal 必需、最新 handoff 新增/替换拒绝、排空重试完成后才调用 unlock，以及上述完整历史 metadata 的分块容量/损坏检测。测试子进程只执行标准库和被替换的合成 callback；没有真实 worker 准入、NKI/native、lock、service/proc/cgroup 查询或 payload 读取。

运行准备完成不等于 simulator 通过。实际执行还要验证新 NKI API、间接 gather 和所有 typed 位模式。原 B4 包、旧 typed v5 failed gate、所有原 CPU3716/291、device/alias、B8/B16 冷缓存/长解码/槽位复用/质量/匹配 HTTP 门均不变。
