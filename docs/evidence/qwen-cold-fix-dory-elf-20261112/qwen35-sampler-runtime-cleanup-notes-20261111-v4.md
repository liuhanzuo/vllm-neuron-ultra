# Qwen sampler runtime v4：严格复用已回收 phase 终止证据

源码后继包含 contract、io、history、service、adapter、CLI 六份 v4。冻结 v3 和原科学 delivery v1 不变；此交付不创建当前 spec、admission、catalog 或 handoff，也不启动实验。

## 问题与修复

旧 service 清理使用本机不支持的 `--kill-whom=all`；新版本对 TERM/KILL 都使用 `--kill-who=all`，仍先认证原 service invocation，不调用可能回收服务状态的 `systemctl stop`，不使用全局进程清理。

旧 outer 在 declared_services 读到已记录 supervisor owner、当前 unit 已被回收时立即拒绝，未检查同阶段已保存的 root-terminal。新版本只在这些保存记录齐全时接入历史终止证据：同目录完整 phase-spec、supervisor-owner、root-terminal 均通过实际摘要读取；phase 必须等于原 expected_phase，supervisor argv 明确绑定该 phase-spec 的路径与摘要，完整 PID/start/ppid/argv/cgroup 和 InvocationID 相同。

保存的 root-terminal 必须恰好保留原 acknowledged schema 的全部字段，完整原 verify_properties 和 terminal 校验通过，真实 ExecMainCode/Status/Result 与原 properties 一致，两个相同的空组记录属于这个阶段，remaining_processes 为空，exit_observed/cleanup_complete 均为 True。成功不能掩盖非零或信号退出；已保存失败仍保持失败。未 ack、缺失、额外字段、改动记录、身份不符、缺排空事实都拒绝，不会降级成 never_loaded。

真正复用前重新读取三份摘要绑定记录并核对原 inode/内容身份，然后顺序做两遍当前 not-found/no-Job/no-Invocation/no-PID、原 owner gone 与当前空组核对。返回的 terminal 保持原保存内容，outer 另外记录 `historical_terminal_reuses`，明确标识退出状态是历史事实，当前观察仅证明服务不存在和进程已排空。新 unit 再次出现或 InvocationID 改变不会借用旧证据。仍 loaded 的 coordinator/phase 即使失败且没有 saved receipt，继续走原实际 observe/drained 路径。

## 科学与来源边界

adapter 的原 native leaf import 指向另行审阅的 qwen35_sampler_native_leaves_v2，所有科学工作仍通过同一个完整入口。RUNTIME_FILES 同时列出保留的 v1 和新 v2；host_sources 添加重复路径摘要必须相同的约束，避免 v1 同时出现在 runtime 清单与 frozen delivery 时覆盖冻结摘要。原完整历史、原解释器/环境、原图、全部科学 gates、资源/日志/输出预算和原 descriptor 锁规则保持。

历史复用不证明新的模型正确性，不制造成功结果，不补造没有保存的失败退出。在 unit 缺失又没有完整终止记录时继续失败并保留锁，仍需 root 独立排空处理，不能泛化接受未知失败或无证据回收。

## 验证

原 runtime33、bootstrap5、namespace6 合计44项通过；新24项覆盖旧 GC/kill-option 失败复现、成功/失败/信号历史状态、缺/改记录、owner/phase/invocation/资源/退出字段不匹配、未 ack、当前存活/后代/Job/重新出现、双观察变化以及 v1来源冲突/v2遗漏。所有进程、服务、锁观察均被mock；原测试的独立解释器只做禁止native导入的import/help检查。

正常化 diff 显示 io/history/CLI 仅版本引用变化，原 service 顶层函数全保持。service 变更仅初始化、observe/drained/cleanup 与三个严格历史证据方法；adapter 变更为 native leaf 接入、declared terminal 绑定和外层显式历史复用记录；contract 仅来源清单和不覆盖原pin规则。原六文件及相关10份v3源码/测试、32份冻结科学和依赖来源摘要已全部核对。
