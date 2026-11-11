# MiniCPM5 nested physical 后继独立复核

结论：**接受此前唯一 NATIVE-OUTER-B1 的源码修复**。范围为冻结 outer_v3 / support_v3 / adapter_v6 / runner_v7 与六个 authority / physical 后继模块；旧 outer_v2 / adapter_v5 的拒绝结论保持。此结论不表示 runtime 已准入或可立即启动。

本次只核验原阻断，没有扩大到新的执行框架。

- adapter 在 fresh physical pair 前已重新认证 coordinator-owner 和本次 running handoff。authority 每次通过原 descriptor Lease、admission/intent 稳定读、current handoff 身份锁存、实际 outer/coordinator/caller 身份，以及完整 service command/resource/unit/invocation 验证。只允许已确认 coordinator 或其 direct native child 使用唯一 admission.unit 排除，并拒绝任何历史 unit 重叠。
- outer 使用 descriptor-owner 模式，排除集合始终为空；实际调用者必须为原锁 owner。本次 native 服务即使名字正确也不能在此分支豁免。
- 五层新旧 physical 函数的 AST 对照只改变 authority/guard/排除传递点，并增加互斥和原锁检查。所有历史 metadata 验证、owner/group 检查、单服务双采样、全树两遍、supplement 与输出 schema 保持。原 child 自身持锁假设已替换成明确 outer descriptor observer。
- 没有对全局 low 模块或函数做 monkeypatch；生产代码显式传递 authority。旧 per-model / mini / current-history 的相关路径已继续检查，没有遗留本次 blocker 所涉及的当前 PID 自持锁或全局 idle 要求。

独立重跑 52/52 通过：session 71663，terminal 27f087，exit 0，16.343 秒。其中 6 项真实调用 strict collect_pair 和五层 physical，在 1308 个合成历史 unit 上执行两遍完整树，每个 unit 至少 4 次 properties 采样。

另新增 2 项最少独立控制，terminal e43069，exit 0，0.019 秒：真实 `Lease.check`、活动列表解析及完整 service 属性检查联合运行，只替换底层系统/metadata/descriptor IO。确认 coordinator/direct native 通过；错 invocation、PID、resource、outer 冒充 child、额外服务拒绝；outer 零排除；descriptor 错误锁存且不能绕过。

交付 source pins、interface/closure 引用全部匹配，上一轮记录的源码摘要全部未变。没有 actual proc/systemctl、锁、服务/模型、native/raw payload IO，也没有具体 plan/overlay/handoff/catalog。实际完整历史 bootstrap、原生运行、外部 exit recorder 和独立 raw audit 仍是后续工作。
