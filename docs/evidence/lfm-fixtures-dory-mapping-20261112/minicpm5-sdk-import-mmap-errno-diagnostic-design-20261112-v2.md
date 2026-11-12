# MiniCPM5 映射 errno：同一父进程观察的有限设计

2026-11-12；仅可行性与边界设计，未实现、未运行、未生成当前绑定。替代 v1 的 `strace -D` 建议；冻结的诊断 v1 与证据不变。

**为何撤回 `-D` 路线。** 官方 strace v5.16，commit `69f66eb864ffa547eda77fa594c4b612b800e8ea`，`src/strace.c` SHA-256 `f2d2e4b7c34bcde35d278fd0d72645c4bb916e0329a89313592797843ec317b3`：1661–1662 明确执行 `prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)`；123–133、1363–1406 是双 fork 与中间进程；1779 起还有 PTRACE_SEIZE 自检子进程。仅检查成功时 TracerPid 不覆盖这些启动阶段。精确摘录存于 `minicpm5-sdk-import-strace-d-source-boundary-20261112-v1.json`（SHA-256 `e41a93742869129f1f08a15176389cfd64e8861ca689dec54cc0865b699eebb7`）。这是固定上游源码证据，不宣称已认证本机 ELF 与其等价；不能把其权限/拓扑变化默认为可接受例外。

**建议的最小路线。** 原 outer 仍用原 SDK argv/env 启动唯一直接子进程；在尚未发送 source gate 的阶段，outer 通过经认证的 libc `ptrace(PTRACE_SEIZE)` 观察这个确定的 child。不启动 tracer 子进程，不向 SDK 导入 ctypes，不设置 PR_SET_PTRACER、LD 环境或 sysctl/CAP。父子关系通常满足 Yama 的父进程限制，但实际内核/LSM 可以拒绝；失败时清理并报告 errno，不修改策略。先只实现有注入端口的纯逻辑与 mock，再审核未来 native 入口，不在本次做 native 实现或试探。

1. **认证与顺序。** outer 保留完整 `auth.authenticate`、descriptor 重建、原九项 native 认证和原 worker 完整导入前路径。outer 所需 ctypes 纯源码、`_ctypes`、libffi/libc 及实际必要依赖必须作为新增的显式未来认证输入，不能用包版本或当前进程中的符号名替代内容/身份凭据，也不能代替原九项。先完成该新增认证再加载 ctypes；SDK 的 worker、解释器、环境、ppid 和源 gate 语义不变。此次未读取任何这些库的 ELF/RECORD。
2. **附加、事件与 gate。** 验证实际 child PID/start/exe/argv/ppid/cgroup 后，以 `SEIZE` + `TRACESYSGOOD` 和经过审查的 fork/clone/exec/exit 事件选项附加，再 `INTERRUPT` 并由唯一事件循环收取真实 stop。调用 `PTRACE_GET_SYSCALL_INFO`，要求实际支持与正确 ABI；不支持即未决，不悄悄改为寄存器猜测。随后 `PTRACE_SYSCALL` 推进，同时以非阻塞短写完整发送原最多 16 MiB 的 source gate，写全才关闭 stdin 形成 EOF。不能先同步写完整 gate 再处理 stops。附加时可能先收到一个没有 entry 的 exit；只标记为附加边界，不推断调用/errno。完整 gate 的验证在原 worker 中先于 Torch import，因而目标 import 不处于该边界缺口。
3. **唯一真实 wait 所有者。** 从附加开始禁止 `Popen.poll()`、`Popen.wait()`、`communicate()` 和原含这些调用的 drain helper。本机 outer 的 Python 3.12 `subprocess.py:1961` 会把 `WIFSTOPPED` 设成负 signal 的 returncode，2008 起还可能把 ECHILD 转成状态 0；均不能用于本诊断。只对已知 PID/TID 做 `waitpid(..., __WALL | WNOHANG)`，保留原始 status；仅 `WIFEXITED/WIFSIGNALED` 是真实退出，`PTRACE_EVENT_EXIT` 仍只是 stop。只有从实际 terminal status 得到 exit code 后才同步 Popen.returncode，避免析构再竞争收割；不得以 ECHILD、消失或 stop 伪造退出。
4. **后代、信号与清理。** 不使用 `waitpid(-1)` 或同组扫描来吸收未知进程。新增 task 只由已知父 task 的内核 fork/vfork/clone event + GETEVENTMSG 获得并绑定身份，数量有明确小上限；意外 exec/ABI 切换、超额或无法解释的事件使诊断失败。真实 signal delivery 原样传递；只有可证明的 ptrace 合成 stop 才不传递，不吞掉普通 SIGTRAP。group-stop 必须正确 LISTEN 或失败清理。考虑启用 EXITKILL 让 tracer 意外消失时终止 tracees；这是必须明示并审核的新增故障行为。失败时关闭 gate，停止采集，继续由同一 wait 所有者处理 TERM/KILL 与事件，直到真实 SDK terminal、所有已知任务消失且原拥有组为空，再允许原锁释放。sticky guard 仍可阻止持久化；不另开弱化 writer。未知成员阻止释放，不能自动接纳或宣称 drain 成功。
5. **输出与解释。** 逐 task 区分 ENTRY/EXIT，仅为 `mmap/mprotect/munmap/mremap/brk` 留存请求参数、实际有符号 return、is_error、原始信息字节和时间顺序；不靠交替布尔位猜 entry/exit。内部 syscall restart 返回必须单独标记，不能算最终映射 errno。FD 路径只可作停点处 readlink 元数据观察：有并发共享 FD 的可能时注明关联未证明；匿名映射不由相邻库名推断归属。不读 tracee 内存，不改寄存器、syscall 或返回值。`ENOMEM` 仍只证明对应调用被拒绝，不能单独判为物理 OOM、RLIMIT_AS 或 libcusparseLt 的某个加载边。
6. **原预算与完成边界。** 保留 AS 2,147,483,648 B、CPU 6、CPU 600/605 s、nice 19、FSIZE 2,097,152 B、wall 900 s；raw 4,392 files / 12,288,000 B、metadata 8 MiB、allocated 32 MiB、单文件 2 MiB、summary 16 KiB、运行 metadata envelope 79,008 B、source pipe 16 MiB。outer 本地有界 trace buffer 最多 2 MiB，计入原预算，真实结束后通过原 Store 一次写入；没有 tracer 输出管道及其额外 EOF 生命周期。超额、超时、附加拒绝、丢 entry/exit、SDK 行为改变或目标调用不明确，都记录未决；不提高限额，不声称模型执行/正确性/提速。追踪的调度与 outer 开销是新增观测扰动，不能声称与未观测运行完全等价。

**有限实现次序。** 下一份纯源码只需要覆盖：已知 task 的 ptrace stop/terminal 状态机，stdin 短写/EAGAIN 与事件交错，按 task 配对的 syscall 记录，2 MiB/原 envelope 预算，以及 sticky guard 后仍真实 drain 的控制。mock 必须覆盖附加失败、孤立初始 exit、fork 事件前后顺序、普通/合成信号、SDK 先退、未知 PID、ECHILD、超额和清理失败。之后再审核具体 ctypes ABI/来源与真实 wait 适配；未完成这些边界不接入当前运行绑定。

最小接口可限定为三个部件，以下仅是设计签名：

```python
# 纯数据校验；authority 必须来自未来新增 observer 来源认证，不能自签。
validate_observer_authority(authority, original_nine_source_receipt)
# 唯一有副作用的可替换端口；mock 禁止触及真实 OS/native。
ports.seize(pid, options); ports.interrupt(tid)
ports.wait_known(tid)                    # pid + 原始 wait status，仅 WNOHANG|__WALL
ports.syscall_info(tid); ports.event_message(tid)
ports.identity(tid); ports.resume(tid, mode, signal)
ports.write_gate(view); ports.close_gate(); ports.signal_owned(owner, signal)
ports.now(); ports.owned_group_members() # 只作核验，不自动接纳成员
# 纯状态机拥有所有 task、entry、gate 游标、trace 缓冲与真实 terminal。
ObservedSDK(child, owner, gate, authority, guard, store, ports).run()
```

`run()` 先认证和附加，唯一循环同时推进 stops 与 gate；任何失败均进入同一清理状态机。只有真实 terminal 才更新 child.returncode；只有 drain 后才请求原 Store 持久化。端口不提供 `poll`、`wait_any`、任意进程接纳或替代 writer。已认证的 original worker/full-auth 路径由既有 outer 调用，无额外 launcher、tracer service 或第二套源码地图落盘。

**备选的适用范围。** 同父进程 PTRACE_TRACEME 需要在 SDK 或新 pre-exec helper 中新增 native 入口，改动面大于 outer SEIZE，暂不优先。若 SEIZE 无法获准或不满足原预算，可另做有限静态 ELF/PT_LOAD 需求分析：必须先认证目标文件与相应加载器规则，只能说明段跨度/对齐及可能请求大小，不能给出实际失败 errno，也不能把估算升级为 OOM 根因。本次不进行此类 payload 读取或 native 前置工作。
