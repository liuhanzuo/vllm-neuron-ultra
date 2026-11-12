# Spark 独立串行 outer：源码交付，尚未准入

本次新增 `run_spark26_native_serial_outer_v1.py` 和 `check_spark26_native_serial_outer_source_v1.py`。完整 26 项 stdlib/mock 检查通过：真实工具 `351fe3`，exit `0`。测试在导入实现之前安装 blocker，拒绝 ctypes、_ctypes、Torch/Neuron/vLLM 等应用包以及完整 `lfm3_nki05_serial_history_v1` 入口。没有实际导入完整历史、ctypes 或应用 native，也没有观察真实 proc/cgroup、获取原锁、操作服务、读取权重/native payload 或启动模型。测试只使用已有源码/固定模型元数据和新建临时 JSON/JSONL 文本。

worker v2、provider v2、科学 core v3、contract v2 和已有 source tests 保持原字节。新的 outer 调用这些已审阅版本，不替换模型或 gate。26 项 outer 检查不是原 worker/provider 的 native 实测。

## 完整外层流程

入口 `outer(binding_ref, review_ref)` 要求原 Python 3.10、`-S -B` 和新的 absent pycache prefix。没有默认 binding、默认队列或自动创建的“允许执行”配置。outer binding 必须与 worker spec 使用同一完整 source map，并分别绑定 exact outer root review 和 exact worker root review。此前 component-only root review 的 schema 不足以生成实际 admission。

1. 核对固定 worker/provider/core 摘要和完整 source map；解析真实冻结 runtime manifest 的所有 RECORD，但不读取其 native payload。当前缺少真实 manifest 时在获取锁之前失败。
2. 通过 `O_NOFOLLOW|O_CLOEXEC` 打开原 existing inode，nonblocking 获取一次 whole-file flock，保留 parent-only fd。持续使用原 descriptor proof 检查 PID/start/argv/cgroup 与原 device 66306/inode 8294421。
3. **持锁后**才通过实际完整 `lfm3_nki05_serial_history_v1.resolve/collect_drain` 解析全集并进行两轮全部 predecessor PID/service/cgroup 观察。保留所有历史 owners/units/source provenance，原大 envelope/alias 仍走 `history.serial.sources.load`，不会被新 receipt 的 8 MiB 限制截断。
4. 冻结实际 latest terminal identity，要求运行期间保持不变。现时核对无其他 heavy unit、原 root/NVMe floor 和完整 48 MiB output reserve。仅 stat 原五 shard 的存在、完整尺寸和身份；真正 full hash 留给 worker。
5. 先检查完整 predecessor proof 能否放入原单文件/receipt 总预算。创建唯一 0700 output namespace，记录完整 predecessor proof、outer binding/初始 descriptor/所有读过的历史 metadata identities 和 admission。
6. 写 durable launch intent 后，只安装 `worker.service_command` 生成的固定服务。systemd-run client 完成仅证明安装尝试，不证明 worker 成功。独立观察 `/usr/bin/env -i` 到实际 Python argv 的转换，要求 PID/start/ppid/cgroup 连续，然后才发布 worker-owner ACK。
7. 监控实际固定 InvocationID、ExecStart、资源和 logging mode。只有已 ACK 的 worker 真正 clean exit、其 PID/start 已消失、两次稳定完整 recursive cgroup 都空，才验证输出。独立读取原 result、event receipt 和四个实际 JSONL 文件，核对完整 4,844 条事件及真实文件字节/hash。
8. outer-result 保留 `outer_self_exit_proven=False`。外层自己的真实工具退出仍由 root 独立记录；代码不能自证已经退出。finally 再做清理/稳定 drain，之后才释放原 fd。

## 启动失败和 cleanup

外层保留未 ACK 的失败 InvocationID、实际 ExecMainPID/退出字段；没有观察到 start ticks 时明确记为 unknown，不补造 process owner。若服务从未创建，或是 ExecMainPID=0/ExecMainCode=0 的实际启动资源失败，可以证明其 physical drain，但不会声称观察到进程退出。已经观察过的服务若在 terminal 审计前消失，拒绝把当前 absence 冒充已知退出。

cleanup 每次先核对原 durable intent、实际命令与 InvocationID。仅给这个已认证服务的完整 cgroup 发 SIGTERM，30 秒后必要时 SIGKILL；不用全局 pkill，也不用可能提前回收 terminal 字段的 systemctl stop。之后独立证明原 PID 已消失及两次稳定 recursive empty group。

若任一步失败，复用原 `research.minicpm5_layer1_drain_hold_v1.retain_until_drained`，每秒重试并继续持原锁。pending receipt 写失败、等待被中断或清理命令失败都不会直接 unlock。catchable signals 在 cleanup 阶段不会把控制流展开出 retained-lock 区域。不可捕获的进程终止/机器故障不是这个 Python 生命周期能排除的情况；原 descriptor 模块的“不证明持续所有权、ABA 或 fork 排除”限制没有被扩大宣称。

ACK 与其他 outer receipt 在私有 run directory 中先写完整临时文件、fsync，再 atomic rename，避免 worker 读到半截 JSON。只允许唯一声明 writer，创建前拒绝已有目标。这不声称可以防御同 UID 的另一写者违反私有目录所有权约定的 check/rename 竞态。所有已发布 receipt 都不复写。

## 明确的 host 历史入口与 native 范围

root 已指出：完整历史模块会间接导入 ctypes。新 outer 没有把这条路径称为“完全无 native”。`RealHost.load_history` 必须先证明 retained lease，然后打开专门的 history import gate。`PinnedHostFinder` 在 gate 之前拒绝 ctypes/_ctypes，始终拒绝应用 ML/SDK 包，且在 local source 执行前核对实际 pin。系统 Python/stdlib 的 ctypes/_ctypes 和系统 ELF 属于已经声明的 host 信任边界，不属于 worker 应用包 runtime manifest 的覆盖证明。实际导入的 ctypes origin 将记录在 outer-binding receipt 中。

本轮测试不触发这一真实入口：只对假 finder、假 history module 和 fake process/service 验证顺序；global blocker 保持安装。没有用 `import history` 作为一次无害的“source-only 检查”。将来真实执行的历史验证仍需完整 source closure 和 root 当前绑定，不能把测试用 mock providers 放进实际 CLI。

## 预算与未完成绑定

原 worker 限额保持：MemoryMax 16 GiB、MemoryHigh 14 GiB、swap 0、AS 24 GiB、wall 10,800 s、CPU 6–7/nice19，库线程1。原 48 MiB output、单文件8 MiB、events16 MiB、logs8 MiB、receipts16 MiB、temporary+dirs4 MiB、最多64 entries 全部保持。新 outer receipts 与 worker 输出共用这套 output 预算。

外层另显式提出 **2 GiB RLIMIT_AS**，用于 host-only history/control 进程；它不在 worker 的 MemoryMax cgroup 内，因此不能声称二者合计硬上限仍是16 GiB。这是本次 outer source review 需要确认的独立资源边界。outer 使用同一个10,800秒工作 deadline，但不设置会在 pending cleanup 中直接杀死锁 holder 的有限 RLIMIT_CPU。cleanup 必要时超出工作 deadline 继续持锁；不因此继续运行科学任务。

完整历史 proof 若超过8 MiB，或 proof加剩余control/result预留不能满足16 MiB receipts，在 launch 前拒绝。不能用少列 owners/units、删除原 source provenance 或缩小两轮观测解决预算失败。此处还没有以当前真实 idle binding 实测完整 proof size，不能声称它必然能装下。所有真实 reference/manifest/proof 输出预算仍待具体绑定检查。

16 GiB runtime authentication read budget仍只限制 worker一次 manifest payload 哈希总字节，不含原8,224,192,408 B shard hash、后续 safe_open/tensor读取、历史metadata反复读取或页加载。outer不会借此预算下载任何内容。原 root28 GiB、NVMe30 GiB、Qwen2 GiB reservation均保留；启动前还需完整48 MiB输出reserve。本次未观察当前物理free或读取原checkpoint。

## 科学范围与检查证据

result validator 保留36层×3输入形状×2 rank的216 QKV；两种cache×两rank各534步；每组527次host排序/restore和96次物理排列。四个真实event文件必须完整包含216、2,136、2,108、384条，合计4,844条，实际字节数必须等于worker receipt。BF16 projection不完全相等仍是诊断结果，outer不增加精度阈值。原cold/long/slot/quality96/HTTP全模型门槛仍待执行，full_model_qualified和new_speedup_claim保持false。

26项检查覆盖：双review真实binding；阻断native/ctypes与持锁后history入口顺序；安装ACK不冒充worker ACK；env→Python连续身份及替换拒绝；错误argv/cgroup/InvocationID拒绝；提前失败/never-created/no-process failure如实记录；已观察服务消失不冒充exit；成功必须PID gone与两轮empty；非空后代及不稳定第二轮拒绝；cleanup只发给绑定组且保留terminal字段；错误invocation不发信号；pending报告失败和等待中断不unlock；无launch路径正常release但不声称exit；完整atomic metadata；原大history reader不减小cap；完整科学计数且无新BF16 gate；四个实际新建fixture文件共4,844条；原output预算未放宽。

本交付仍不可准入：还缺完整真实传递依赖manifest、原权重storage/placement准入、当前已结束任务的完整idle catalog/terminal绑定、实际完整proof尺寸和source closure冻结、exact worker与outer binding的独立root review，以及真实Python3.10 import前置验证。本轮没有锁或服务操作，也没有排入新实验。
