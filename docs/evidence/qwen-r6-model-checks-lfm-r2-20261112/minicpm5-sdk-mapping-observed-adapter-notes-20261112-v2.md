# MiniCPM5 SDK mapping observer：alarm-safe 交接后继源码

本版后继使用 `minicpm5_sdk_mapping_observer_backend_v2.py` 与 `minicpm5_sdk_mapping_observed_adapter_v3.py`；纯逻辑 v1 未变。v1 交付里发现的 wait-lock / alarm 窗口已作源码修复，并通过两组纯 mock 检查。没有 native authority、完整 outer / CLI 或真实运行准入；原历史依赖提前导入 ctypes 的接入阻塞仍待解决。

## 修复的具体行为

旧 backend 的 `make_ports()` 在返回前已经取得 Popen `_waitpid_lock`，adapter 却在函数返回后才保存 ports。若 alarm 在 backend 返回处或调用方赋值前触发，原清理可能看不到真正的所有者。还有 release 已完成、owned 标记尚未清除的对称窗口。

新 backend `make_ports()` 只准备 ports，不取得 wait lock。adapter 先将同一个 ports 保存到 `self.native_ports` 与 `self.observer.ports`，再显式调用 `begin_wait_handoff()`。因此认证失败、factory 返回之前、调用方赋值之前，以及开始 handoff 之前的 alarm 都还没有 wait ownership，可以安全进入原 pre-handoff 清理。

`begin_wait_handoff()` 和最终归还锁共同使用 `_transition_wait_lock()`：只在非阻塞 acquire/release 与其 ownership 标记更新之间安装暂存 SIGALRM 的 handler。原 900 s timer 继续计时，不调用 alarm、setitimer 或 pthread_sigmask。原 handler 恢复后立即处理暂存 alarm，确保异常传播时 ports 已有强引用、锁状态已一致。认证、导入、文件 IO、等待和 ptrace 都不在这一短窗口内。延迟的是一次不可分割的 bookkeeping 提交，不重新开始计时或扩大预算。

其他不可预期异常若导致取得/归还状态未知，`wait_transition_uncertain` 保持 True；adapter 明确拒绝旧 Popen cleanup 并保留句柄与原串行 lease，不能猜测所有权或释放串行描述符。这个保守路径没有自动解除功能，后续真正 outer 必须保留原 cleanup-pending 行为。普通可解释的 acquire(False)=False 保持未接管状态。

factory 和 adapter 的源码后继均保留实际 child terminal 与观察成败分别计账、六文件 Store、原完整 prerequisites、相同 2 GiB / 900 s 与全部输出 caps。只有实际 terminal、所有已知任务及原组物理排空后，才归还 wait ownership。zero child exit + trace/receipt 失败仍返回失败诊断。原 v2/v4 与 backend v1、logic v1 均未改动。

## 检查结果与边界

- `check_minicpm5_sdk_mapping_observer_backend_v2.py`：实际 terminal `5d68fc`，exit 0，37 项，0.012 s。包括原 backend/Store 的 27 项语义检查及 10 项新边界检查。
- `check_minicpm5_sdk_mapping_observed_adapter_v5.py`：实际 terminal `a0f1c8`，exit 0，26 项，0.044 s。覆盖原协调行为及 5 项新交接/清理检查。

测试使用 rejecting finder 禁止 ctypes、_ctypes、Torch、NumPy 和 Neuron 模块；只执行原 SingleSDKAdapter 的选定 class AST 与 fake dependencies。native/process/IO/signal/lock 端口全部为 fake，没有真实进程、锁、signal handler 或 timer 操作。

新检查明确在实际 backend 函数 RETURN_VALUE opcode 之前，以及 adapter 的 STORE_ATTR native_ports opcode 之前抛出模拟 alarm。另覆盖 handler 安装前/后、acquire 内已取得假锁但尚未返回、handler 恢复处、显式 handoff 调用前、以及 release 内的 alarm；断言两处强引用、原 handler 恢复、正确清理路线与只释放一次。opcode 检查使用本测试进程的 sys.settrace，不使用真实信号。

这些测试证明上述 Python 协调分支在 mock 条件下的行为；不证明实际 ptrace、CPython/native ABI、内核 task 生命周期或真实模型数值。未重复已通过且未改动的逻辑和历史源码检查。原 12 项 ABI 签名检查属于前版记录；本版实际 backend 和 adapter fake 调用覆盖新增接口，不把旧 12 项标成新执行。

## 接入仍需完成

外层仍须记录与原 limits 同时开始、早于 prerequisites 的绝对 monotonic deadline，并传给 adapter；须使用新 `ObserverStore`、`adapter.result()` 以及其 `exit_code`，并显式核对六文件。`make_adapter_class(original)` 依旧不隐式导入原 diagnostic，调用方必须提供经过完整认证的原模块。

原 diagnostic 的历史模块仍会从 `minicpm5_python_trace_lifecycle_v4.py:7` 提前 import ctypes。新 factory 的纯导入不能解决实际 caller 提前导入这一点；full-auth outer 必须先解决历史入口，再提供独立明确的 observer authority，完整保留原九项 native 认证。当前原解释器/ctypes/libc 已存 archive 与 package 证据可复用，但当前字节、稳定身份、已加载 provider/pyc、原路径依赖闭包及 ffi 归属尚待原规定方式验证。

先前测试误导入 ctypes 的范围偏差记录完整保留：`minicpm5-sdk-mapping-observed-adapter-source-test-scope-deviation-20261112-v1.json`，SHA `9cd1bd2aeed58e31900d848d9847a3fd5d1580d1d097cfa7eac5694666a35a27`。不能把早期 cae836 的 mock 通过改称纯源码合规；没有补造当时 PID、_ctypes 来源或全进程 syscall 证明。本轮后继没有再次跨过该边界。

本版为源码交付，待 root 独立审阅；未启动模型、native、SDK 子进程、真实 signal/锁、服务、下载或发布，未取得主线串行锁，没有新增模型资格或提速结论。
