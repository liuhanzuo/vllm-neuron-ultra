# MiniCPM5 当前执行 handoff 契约

这是源码设计，不是实际准入。外层启动前仍必须在原锁下调用原 `require_latest_terminal`，得到最新 idle terminal 或完整 publication-only 链；本次 child 启动后检查的是随后明确建立的本次执行记录。完整历史恢复、前驱物理 pair、并发活动检测与原锁检查不变。

## 无循环的生成顺序

1. 已审查的外层持有原始串行锁，完成原 freshness 检查，保留完整返回值。将结果与 contract/plan 摘要、原 outer 身份/lock identity、前驱 catalog、预先选定的 running handoff 路径和由 root 实测的外部 tool session handle 一起写入既有 `launch-intent.json`。此文件不包含 admission 或未来 handoff 的摘要。
2. `admission.json` 加入 `current_execution`：launch-intent 引用、同一未来 handoff 路径和同一 tool handle。其他已接受绑定保持不变。admission 不包含未来 handoff 的摘要。
3. 外层创建协调服务并确认其实际进程身份与 invocation，写入既有 `coordinator-owner.json`。root/外层随后写完整 running handoff；该 handoff 引用已完成的 admission、intent 和 coordinator-owner 摘要。child 在此记录存在前等待，不能用旧 idle 状态通过。

引用方向为 running handoff → admission → launch-intent → 已完成的前驱 handoff。coordinator-owner 只在末端的 running handoff 中引用，无循环摘要。没有增加 runtime 输出文件名。

## running handoff 的精确变化

完整复制原 freshness 绑定的最新 handoff，只能改 schema、updated_at、previous、live_tool_sessions、active_heavy_units、publication_only_since_bound_terminal 以及本次 `mini_native_current_execution` 对象。schema 使用专用值，previous 指向准入前最新 handoff，live_tool_sessions 必须恰好为本次外部 session，active_heavy_units 必须恰好为本次 native unit，publication_only_since_bound_terminal 明确为 false。其余字段完整保留，禁止删改前驱状态或清空 live 字段伪装 idle。

`mini_native_current_execution` 完整绑定 admission/intent 引用、contract/plan 摘要、outer 完整进程身份、coordinator-owner 引用及完整对象、原 lock identity、前驱 catalog 和实际 session handle。未来路径或 tool handle 单独均不足以准入。

## child 校验与持续观察

child 验证原 intent/admission 的认证字节与身份，按已保存引用重新验证准入前 publication-only 链，并读取全局最新 handoff。该最新项必须恰为预定 running 路径，其整个对象必须等于上述唯一变换。实际 outer 与 coordinator 的 PID/start/argv/parent/cgroup 必须分别匹配；原 descriptor lease 和服务 unit/invocation/命令/资源仍由实际适配器验证。

首次成功读取的 running handoff 摘要和文件身份被锁存。后续 lease 监测检查同一文件身份、最新路径和当前 outer/coordinator 身份；同路径替换、新未知 handoff、不同 admission/owner、额外 live sessions/units 或前驱改动均拒绝。正常本次 running handoff 也不能绕过完整已完成前驱检查。

outer 尚未实现；它仍须负责真实 tool handle 的外部观测、准入前原 freshness 调用、真实 intent/admission/owner/running handoff 的顺序发布和失败清理。该设计和 mock 结果不构成实际启动许可。
