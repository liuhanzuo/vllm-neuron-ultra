# MiniCPM5 native outer 独立源码审查

结论：当前 outer_v2 / support_v2 / adapter_v5 / runner_v6 **不能接受为可启动的 runtime integration**，发现 1 项明确 P1 阻断。既有 46 项 mock 全过，但覆盖不到真实嵌套历史 physical 路径。

## 唯一阻断：child 的当前执行上下文没有传入历史 physical 全链

`runner_v6.bootstrap:59 → adapter_v5.resolve_admitted:168 → latest.collect_pair → latest.physical → sdk_batch_layout_completion_v3.physical → after_nemotron_r12_recovered_failure_v1.physical → startup_drain_v27.physical_drain → r12_completion_v1.physical`。

coordinator 已经是本次活动 systemd 服务，当前执行 handoff 也已正确绑定。此时历史链仍要求全部服务 idle；startup_drain 的自行解析扫描会拒绝这一本次服务，直接把新 native unit 填进 allowed_units 又被旧 nemotron controller 名称规则拒绝。更深一层 `r12_completion_v1.physical` 把 child 自身当成锁 owner 调旧 `low.lock_proof`，与 outer 唯一持锁、child 不继承 fd 的已接受约定冲突。上层另有无参 `active_heavy()`。仅改 adapter 的 Lease.check 排除列表或两个上层扫描不足以修复。

最小语义正确修复：新增显式 current-execution-bound physical 后继，将准确绑定的本次 unit / invocation / PID-start / argv / cgroup 和 outer descriptor observer 沿全链注入。保留原完整历史单位、owner、group、成对完整采样与验证；outer 启动前仍要求全局 idle；不全局 monkeypatch 旧模块、不省略旧 physical pair、不修改 frozen 文件。

## 复核通过的边界

- support 保留原四类完整来源分区和九项 3,920,408 字节认证循环、原 receipt validator，只替换旧内层锁证明。
- native 专用系统属性 query 包含 Nice 和所有原资源；已有真实 SDK r2 metadata 中 `RuntimeMaxUSec=2h`、`TimeoutStopUSec=15s`、`CPUAffinity=AllowedCPUs=6-7` 可被当前 parser / service_identity 正确接受。本轮没有实际查询 systemd。
- 原 atomic once 的实际函数在内存 IO 中按写入→文件 fsync→renameat2(NOREPLACE)→目录 fsync 执行；最终路径没有半写内容可见区间。所有原 staging writer slots 与命名一致，owner / native-result / running-handoff 发布后读取顺序未发现新的正常完成阻断。
- 16/14 GiB、7200/15 秒、native 128 MiB / total 256 MiB 和磁盘底线保持。失败清理不再使用锁存 work guard，精确服务身份、所有已确认 owner 和完整 cgroup 排空后才关闭 outer fd；正常 cleanup-drained 只有一个发布者。
- intent→admission→launch→coordinator ACK→running handoff 保持无循环；没有把外部工具 exit、独立 raw audit 或原 SDK model exit 1 改成成功。

## 可复核证据

既有五组纯 mock：46/46，session 63412，terminal d3c7f2，exit 0，13.863 秒。

独立控制：`check_minicpm5_layer1_operator_native_outer_independent_20261111_v1.py`，5/5，terminal d401ad，exit 0，0.447 秒。它真实执行历史 startup 的双采样后 current-unit 拒绝、r12 child self-lock 拒绝、上层全局 idle 拒绝，并检查系统字段表示和 atomic 发布时序；系统调用均为内存替身。

未读取 raw/native payload、未申请串行锁、未启动服务/模型、未创建具体 plan/handoff/catalog，旧源码没有修改。结论只针对此冻结候选；源码后继须另行验证上述嵌套真实控制流后才能改变此结论。
