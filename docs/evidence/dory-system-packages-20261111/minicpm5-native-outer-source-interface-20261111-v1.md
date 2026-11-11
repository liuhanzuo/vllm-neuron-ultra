# MiniCPM5 原生 outer 主路径与 native 认证兼容实现

本交付为可审查的实际操作源码及 mock。当前候选是 `minicpm5_layer1_operator_native_outer_runner_v2.py`、`research/minicpm5_layer1_operator_native_outer_support_v2.py`、process_adapter_v5 和 child runner_v6。未产生具体 spec、overlay、admission、intent、running handoff 或 serial catalog；未读取实际 proc/raw/native 二进制，未获取锁或启动服务/模型。所有已执行旧版本保留。

## 实际实现范围

1. outer 首先验证原 UID/user bus，然后通过 stdin FIFO 接收 root release：先输出带实际 outer 完整身份、spec/review 引用和随机 nonce 的 challenge，外部 root 在实测工具 session 后回复同一 challenge、session handle 与新 running handoff 路径。包上限 4096 字节，120 秒期限，单换行加 EOF，非 FIFO、错 nonce/owner/spec/review 拒绝。包被保留在 admission.outer_release，intent/current_execution 继续绑定实际 tool handle。进程不能自行证明外部工具 session，可信边界仍是 root 真实观测与回复；这不是凭空生成的外部成功凭证。
2. support.acquire 独立实现 Mini 原锁的一次 O_NOFOLLOW/O_CLOEXEC/LOCK_EX|LOCK_NB 获取，校验规范路径、原 device 66306/inode 8294421、打开前后身份、实际非继承属性。获取后使用已接受 descriptor.prove 并验证完整初始证明。它没有借用 Dory inventory-only 授权，没有旧 unique-global-row fallback。任何获取阶段失败发生在 child 创建前，关闭该 fd 并传播原错误；工作期间只保留这一 fd。
3. outer.resolve 执行原 recovered completion、完整 overlay、具体 source review/plan digest、完整 source 分类/文本验证/实际加载源码 pin、全部历史身份与原 paired physical collect/validate。实际 outer/support 和原 descriptor/proc pin 也必须出现在完整来源集合中。原 source map 不裁减，不把动态闭包当成全部输入。
4. native-auth 兼容实现保留原 check_partition、native_identities、query_sources.authenticate_native_sources 和 validate_native_receipt，只把旧内层 p.lock_proof 替换成绑定 outer/原锁/fd 的 descriptor guard。它仍认证原九项依赖的全部 3,920,408 字节，每个文件前和全程前后检查；身份变化、短读/摘要失败、额外 native pin 或非真实 outer 均拒绝。receipt schema/字段/分区摘要和科学语义不变，模型/raw 不在这个认证路径读取。
5. 在原空间底线加全量输出预留后创建新 NVMe root；保存 source-authentication、完整无损 predecessor pair，然后按已接受无循环顺序产生 intent → admission → systemd-run → 实际 coordinator ACK/owner → running handoff。运行 handoff 完整继承已认证前驱，只包含已接受的本次执行变化。发布前再次调用原 freshness 并要求与 intent 的完整证明相同。child 启动后使用 current-execution 校验，不冒充 idle。
6. OwnedService 记录 launch_attempted，原有 command helper 会实际 wait/reap systemd-run helper，包括失败和超时路径。即使命令失败，清理仍保留该次已尝试服务的身份边界。ACK 验证实际 argv、cgroup、PID/start、unit/invocation；monitor 使用原资源、命令、cgroup、状态与时限，保留并发检测、输出 inventory 和空间监测。
7. 正常结束复用 lifecycle_core_v2.outer_terminal_step 验证实际 service exit、coordinator/native wrapper 和完整输出 inventory。仅其 OuterPorts 管理 cleanup-drained.json 的一次发布。finally 使用原 retain_until_drained，在明确完整服务/所有 owner/整个 cgroup 排空前不能关闭原锁；timeout、等待中断和失败凭证写盘异常不授权释放。清理不重用锁存失败的工作 guard，而独立重新检查精确 service invocation/command、native owner 和完整 cgroup。调用的是 actual cleanup 实现，没有 success boolean 替代原物理检查。

## 接入时发现并修正的两个实际问题

原低层 p.properties 的 KEYS 没包含 native PROPS 中的 Nice。adapter_v4 起新增 native 专用属性查询，把全部原 KEYS 与 life.PROPS 的实际属性名合并，包括 Nice、RuntimeMaxUSec、TimeoutStopUSec；没有改动资源值。adapter_v5 保留此修正。

原低层 p.once 会直接创建最终 JSON 路径，child 可能在文件尚未写完时读取 owner/handoff。最终候选改用已有 `minicpm5_python_trace_lifecycle_v4.once`：写有限 pending 文件、完整 fsync、renameat2(NOREPLACE)、父目录 fsync。outer、coordinator/native owner/result/failure 都使用这个已有实现。原有限 runtime staging writer slots 已覆盖这些文件，不增加 runtime 文件名。research running handoff 的 staging 不属于模型输出树，也不会匹配 continuation-handoff 最新扫描。

## 保持的约束

原模型、原权重、BF16 和原生解释器不变。服务仍为 MemoryMax 16 GiB、MemoryHigh 14 GiB、无 swap、TasksMax 128、CPU 6–7、Nice 19、运行 7200 秒、停止 15 秒、单文件 16 MiB。输出 256 MiB/native 128 MiB、根盘 28 GiB/NVMe 30 GiB 底线保持。原 SDK 模型 exit 1 不改写，恢复审计成功不转换成模型成功或性能结果。冷缓存、长解码、请求槽位复用验证要求未被删除。

## 实际 mock 证据

最终候选联合运行 46 项纯 mock：session 73506，initial chunk 6d5a9e，terminal 15c9e7，exit 0；`Ran 46 tests in 13.935s`。

- support 8 项：单次/CLOEXEC/初始证明、失败关闭不重试、原九文件合成完整字节循环和原 receipt validator、错误依赖/短读/身份变化/非 outer 拒绝、无 drain 不释放、完整服务属性请求。
- 服务/故障/原子发布 10 项：challenge、实际 ACK、命令/资源/invocation、ambiguous launch 保留、publication 前 freshness、锁存失败后的独立清理、live owner/group 拒绝、等待中断和证据失败继续保留，以及 child-visible 发布全走原 atomic once。
- outer 主路径 3 项：内存文件和 mock adapters 下执行真实 outer 函数，检查 auth/intent/admission/ACK/running/terminal/cleanup/release 次序；认证失败不启动；launch/monitor 失败先排空再 close；cleanup-drained 不重复创建。这些 mock 不是实际 systemd 集成测试。
- current-execution 10 项、adapter/runner 15 项回归继续全部通过。

先前未加入原子发布的版本也保留其真实通过记录：support_v1 8 项 107783/0；outer_runner_v1 9 项 18343/f96ab9/0；main_v1 3 项 c69d58/0；current_v2 10 项 b07329/0；adapter_v6 15 项 14023/a88693/0。它们没有被覆盖成最终版本。

静态 runtime 闭包报告 v2 包含 572 个本地 Python 文件、7,034,904 字节，分析 session 86895/terminal 1667da/exit 0。71 个动态执行位置和外部 import 根明确保留，动态导入/外部包/native 二进制认证仍不能从静态闭包推断通过。原完整科学 source map 继续必需。

## 当前未接边界

当前源码尚待 root 独立审查。没有具体新 spec/review/source extension/overlay/input identity 绑定，真实外部 root FIFO challenge responder 也未执行。需将最终 outer/support/adapter/runner/原 atomic helper 的实际完整闭包全部纳入原科学 pin，并选择当前已完成 serial catalog 及新运行路径；不允许复用 mock 引用。

没有执行真实完整 completion/source/physical/child bootstrap/systemd 路径，不能保证实际历史输入或系统差异已被此轮验证。输出分配量/目录 live snapshot 的既有并发增长限制仍由后续监测和最终不可变审计处理；重复 guard/source 校验的实际成本未测。

outer-result 明确保留 self_exit_claimed=false、actual_external_tool_exit_verified=false 和 independent_raw_audit_pending=true。外部实际 outer/tool exit recorder、成功/失败结束后的 root terminal handoff/catalog 更新、独立 post-exit native raw auditor 入口及其实际 recorder 仍需接入。当前 outer 不把自身返回结果当成外部成功，不执行或自动启动 raw auditor，也不自动释放任何后继任务。

因此 source_review_accepted、runtime_admitted、launch_ready、full_model_qualified 和 new_speedup_claim 当前全部为 false。
