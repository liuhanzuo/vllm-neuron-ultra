# Native 当前运行与完整历史物理检查兼容后继

本交付修复独立审查发现的实际启动阻断，不运行模型。旧 outer_v2/adapter_v5 的完整历史链把本次 native 服务视为未知重任务；更深的 r12_completion 还误把 child PID 当作原串行锁 owner。旧源码和拒绝结论保留。

## 原阻断链与后继范围

原链：adapter.resolve_admitted → operator_sdk_runtime_completion.collect_pair → 原 physical_evidence_v3 双采样 → batch_layout_completion.physical → recovered_r12.physical → startup_drain_v27.physical_drain → r12_completion.physical。

startup_drain 独立解析 systemctl list-units，默认要求所有任务 idle，且其 allowed_units 只接受旧 Nemotron controller；后面另有三个活动检查，以及 `low.lock_proof(low.proc(os.getpid()), lock)` 的当前 PID 持锁假定。只改最上层 active_heavy 无法修复。

新增一个显式 Authority 和五个小的 physical 后继，复制原函数体，仅替换当前操作 guard/排除参数及向下传递。所有原 metadata 校验、1308 历史服务、全部历史 owner、旧 cgroup、每服务双样本、完整两轮树、补充观察、返回 schema 和严格 pair validator 保持。

## 两种固定 Authority

- descriptor_owner：只供实际持有原 descriptor 的 outer/后续独立 auditor；要求当前实际进程等于 lease.outer_owner，allowed 为空，不能排除任何活动任务。
- current_native：只供已 ACK 的协调进程或其确切直接 native 子进程。每次嵌套排除前后验证原 descriptor lease、不可变 authority 摘要、已认证 current-execution handoff 锁存、admission/intent 原字节及身份、实际 outer/coordinator PID/start/argv/cgroup、当前调用者身份和完整 service unit/command/invocation/resource 契约。只允许 admission.contract.unit；该 unit 必须与全部历史 unit 集合无交集。

Authority 不接受自由形式 allowed_units，不修改任何共享模块或全局 low。startup 原 list-units 的真实解析结果交给同一 Authority 校验，其他 active_heavy 也只接其精确允许项。旧嵌套锁检查变为实际 descriptor-owner lease，未删除串行锁检查。

接入候选为 adapter_v6、child runner_v7、outer_support_v3、outer_runner_v3。outer 准入前使用零排除 Authority；child 先完成原 current-execution 校验，再将确切 admission/owner 绑定传入全链物理检查。原 BF16、权重、16/14 GiB/7200 秒、完整 native 九项认证、原子发布、失败 drain 和已接受有限输出 catalog 不变。

## 验证证据与边界

新增真实嵌套链内存回归 6 项：session 5603、initial 05811a、terminal d86427、exit 0，2.368 秒。未 mock collect_pair 或中间五层 physical；实际执行原 strict pair，两次完整树包含 1255 旧服务和 53 r12 服务，每个 unit 至少四次 props/group 采样。startup 实际 list-units 解析和后三层 active 检查均执行。原错误 child-PID low.lock_proof 被设为必抛，正常全链仍成功。正常 native 精确绑定接受；额外 service、历史重叠、不同 owner/admission/invocation、第二轮树变化均拒绝；outer 零排除模式也实际执行该链。

测试只 mock 系统 IO/metadata enrollment，以及原五个旧 per-model 物理函数和原 mini/current-history 验证的返回值；这些旧分支没有本次修正的全局 idle/lock 检查，源码保持调用，不声称所有实际历史数据已在内存构造中重新验证。真正 1308-unit 扫描及完整 pair 语义没有被 mock。

迁移后的原 46 项 support/服务/outer 主路径/current-execution/adapter 回归全部通过：session 77922、initial c6ee23、terminal ef4dda、exit 0，13.876 秒。AST 标准化函数 diff 仅显示上述依赖传递和 guard/exclusion 变化，没有删除历史观察循环。

本交付待独立审查；没有实际 proc/raw/native/锁/服务操作，没有具体 spec/overlay/admission/handoff/catalog 更新。原 post-exit auditor 任务被本次阻断修复优先打断，目前仅留下未执行的 post_exit_core_v1 草稿，未计入本交付或认证闭包；独立 raw auditor 真实入口仍待继续补齐。新候选整体 launch_ready=false，不声明整模型通过或提速。
