# Package r2 post-exit integration — source-only plan

本文件仅保存接入方案；没有实现新 wrapper，也没有执行实际 audit、锁、History、native 或资格升级。实际 prepare 成功与 accept session 由 root 报告；不据此推断 runtime outer 的 session 或退出状态。

1. root 获得真正 outer tool 的 session_id、terminal_chunk、exit_code；绑定 root-live、running、running-ack、binding/spec/review/admission/coordinator-owner/outer_owner 全身份。子进程自称退出不是工具终态。
2. 重新取得原串行锁并持续核对 lock identity、精确 latest running handoff 和无 active heavy workload。用原 `declared_services` 重建全部固定 phase scope，两轮当前稳定观察；started 服务用原 `drained()`，未启动/未声明 phase 核对 not-found 和递归空组。不能只遍历 outer-terminal 中较短的完成名单。保留历史 terminal 复用标记及当前 absence 证据。
3. 实际 exit 0 且原 outer-terminal 成功时，写精确 external tool receipt，然后直接调用未改的 `runtime_v4.final_audit.audit`。它保留完整 History、24 个阶段、child wait/exit/drain、203 向量、sampler/quality/HTTP、source/inputs/build seal、physical pair 和稳定六树检查，成功才写第25阶段。返回仍未独立授予 full_model_qualified。
4. 失败保留实际 code、outer-terminal.failure/active_scope_failure、真实 completed prefix 和既有 metadata refs；同样要求完整 scope 排空，不写成功第25阶段、不升级模型或提速。缺原 outer terminal 或 scope 证据不能补造；实际 nonzero 与 terminal.failed=false 的不一致也原样保留。若独立审计失败，另记录审计失败，不改写原 outer tool 状态。
5. Catalog 追加真新 source delta：先完整证明 recovered + overlay == spec sources；prior = recovered + 所有旧 row sources（冲突拒绝）；delta = spec 中 prior 未有的项。再把独立 root auditor/ACK 的实际完整 SHA 并入 executed sources，只向 delta 增加 prior 未有项，断言 prior + delta == 全 executed sources。旧 rows 原样保留，追加严格五字段 row；全部输出和 catalog 保持原 8 MiB 上限，依序发布 actual-exit、idle、catalog。不能追加整个 overlay。

成功 stage25 接收的 external tool receipt **只允许**十个字段：`schema`、`reviewer`、`spec`、`admission`、`outer_owner`、`tool`、`exit_code`、`completed`、`observation_source`、`observed_unix_ns`。schema 固定 `qwen35_package_root_actual_tool_exit_v1`，reviewer `/root`，tool 只含 `kind=tool_session` 与真实 session_id，exit_code=0、completed=true、observation_source=`actual_external_tool_result`。`terminal_chunk` 放另一个 root post-exit receipt，不能塞入这个精确十字段对象。

另一个 post-exit receipt 绑定完整 actual_tool（含 terminal_chunk）、上述 external ref、root binding/live、running/ACK、spec/review/admission/coordinator/outer terminal、全scope及双排空证据、completed-prefix/failure、成功stage25 ref（失败null）、单独root资格决定、完整source composition和真新增delta、auditor/ACK source pins与执行前后identity。

独立 observer 如以后需要实现，可放在实验 ROOT 外明确的外部路径，因为最终 `loaded_local_sources` 不允许新 Q 下的 `__main__` 混入已经冻结的 spec source map。这不意味着它已在原闭包：必须先单独源码交付和审阅，明确绑定 SHA，执行前后完整重验 SHA 与稳定canonical identity，写入真实 post-exit receipt 并纳入 catalog executed source delta。不修改原spec、不伪装 `__file__`、不跳过 loaded-source 检查。暂不新增 wrapper，待真实 outcome 后决定最小实现。

完整字段和源码 SHA：[qwen35-package-full-r2-post-exit-source-plan-20261112-v1.json](/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-full-r2-post-exit-source-plan-20261112-v1.json)，SHA256 `1c9c1883e1dbf197739c62d3512e8f98e2183cacdf6391a5eca9e95a3e5a4ba9`。

已对 root 新 ACK 源码作一次简要独立对照（工具读取 `6d5b47`）：保留真实 session 参数、outer/coordinator 全身份、原 descriptor/live/active-heavy、完整 expected_running 和 exclusive publication；package 改用 control_root/coordinator-owner/admit_shape，并增加精确 binding SHA 与最新idle完整identity核对。源码范围接受，无实际ACK执行；外部session真实性仍须root实际工具证据，ACK source需独立绑定并加入真新增delta，不能称其原已在worker spec内。详细对照及两份源码SHA已存入JSON。
