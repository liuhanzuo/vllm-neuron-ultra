# Qwen sampler runtime v6：env 到 Python 的一次重认证

已完成六个继任源码和纯 mock 验证。v5 的六个源码及固定测试文件保持原摘要；没有当前运行绑定、准入或实验启动。

## 窗口复现与修复

Service.observe 读到 /usr/bin/env 时保存 pending_process；紧接着 scan 再读 proc，如果同一 PID 已 exec 为精确 Python argv，v5 会因完整字典不同而误拒绝。工具 4cd83a（exit 0）先复现了该问题：第三次阶段 proc 回调发生单向切换，PID/start ticks/ppid/cgroup 全不变，没有 supervisor-owner 文件。新增测试再按同一真实函数调用顺序复现。

v6 仅对“已有合法 env pending 身份 → 精确 service.argv，四个身份字段不变”增加一次 Service.observe 重认证，重新检查全部 properties、InvocationID、资源、命令和 pending 身份，并以末次 proc 读取核对正式 owner。补充认证前后均调用原 guard。不修改身份，不循环等待 exec。陌生 argv、身份置换、资源/InvocationID 改变、回退 env 或补充认证后的进程变化均拒绝。

## 5 秒门槛评估

20 个已启动阶段的 unit 保留为已加载终止状态时，稳定双扫描精确需要 122 次 properties 查询和 2 次活跃名单查询，共 124 次 systemctl 命令。5 秒要求每次查询平均小于约 40 毫秒，还没有计入元数据、进程和 cgroup 检查成本；没有实测能保证该条件。

虚拟时钟假设每次命令 50 毫秒，则完全稳定的扫描累计 6.2 秒。v5 的 5 秒门槛拒绝，v6 接受。这是合成成本反例，不代表本机实际延迟。

按 root 明确决定，v6 删除这项新增门槛，保留最多 4 轮完整重核、原每命令超时、原 outer 总期限和 guard。虚拟时钟逼近原 outer 截止时间时仍停止，exec 重认证前 guard 失效时不继续 IO。所有原资源常量不变。

v5 的未知/未声明服务拒绝、当前 admission/coordinator 完整认证、至多一个活跃阶段、身份保留、旧阶段实际排空、持续变化有界拒绝、失败快照持久化及 SIGTERM 与逻辑成功分离规则均保持。

## 全量验证

工具 88cabe、exit 0：109 项通过，unittest 报告 1.866 秒。原 v5 的 97 项全部保留/迁移，新增 12 项：8 项执行切换及拒绝控制、3 项虚拟时钟/原期限检查、1 项补充认证前 guard 拒绝检查。原 leaf v2 的 8 项继续执行原文件。新增 12 项先单独通过 f68d09/0；最终采用 109 项全量结果。

native_leaf、Phase、coordinator、Adapter、LeaseObserver、graph_reuse、worker_bootstrap、admit_shape、load_context 的 AST 与 v4 一致；leaf v2、科学驱动、全部原 LIMITS/资源属性、20 阶段和 21 unit 规则不变。六个继任文件中实质修复仅在 adapter；其余五个文件仅更新版本接线。

没有实际模型/native import/proc/服务/锁操作，没有读取或修改当前 handoff/catalog。原 r5 的实际失败原因仍未证明，没有新增模型合格或提速结论。

在 /home/ubuntu/experiments/trn1-latest-models-20261031/research 复跑：

```sh
python3 -B -m unittest test_qwen35_sampler_runtime_source_v6 test_qwen35_sampler_bootstrap_self_service_v6 test_qwen35_sampler_virtual_namespace_v6 test_qwen35_sampler_collected_terminal_v6 test_qwen35_sampler_active_scope_v6 test_qwen35_sampler_split_install_v2 test_qwen35_sampler_env_exec_window_v6
```

交付同时含 v5→v6 最小差异与 v4→v6 累计差异。下一步由 root 独立审核，随后再准备新的完整运行绑定。
