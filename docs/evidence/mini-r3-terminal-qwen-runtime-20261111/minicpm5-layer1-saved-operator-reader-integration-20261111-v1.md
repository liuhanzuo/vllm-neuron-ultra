# MiniCPM5 已保存算子比较：有界读取与串行接线

本交付仅有源码、完整范围的合成元数据和每份 6 B 的内存文件测试。**未读取实际 tensor payload，未操作锁、进程、服务、目录扫描或当前 catalog；未绑定尚不存在的 native terminal/audit 引用。** 已审查的 `minicpm5_layer1_saved_operator_comparison_v1.py` 保持不变。新增 helper 不提供独立命令行入口、后台任务或自动 successor。

## 复用的既有接口

- `minicpm5_layer1_operator_native_results_v2.Reader.get`：完整独立读取、精确长度、完整 SHA256 和每文件一次读取；`event_context`、`capture_metadata`：原完整事件身份、shape/stride/offset/dtype/selection 与 clone 元数据。
- `minicpm5_layer1_operator_native_trace_observer_v1.validate_capture_summary`：原 native scale source/code/line 及实际操作记录。
- `minicpm5_layer1_operator_sdk_trace_observer_v2.validate_sdk_capture_summary`：原 SDK 两个 Python scale 观测与完整 trace 身份。
- `minicpm5_layer1_operator_native_audit_io_v1.NativeAuditIO._parents/_directory`：从 `/` 到最终父目录逐层 no-follow 打开并保持句柄，前后验证每个目录链接身份。

SDK 既有 `operator_sdk_results_v2.snapshots` 把结构校验和直接文件读取组合在一起。新 helper 的 `sdk_descriptors` 保留相同 manifest、路径公式和完整嵌套 leaf 集合检查，把读取延后；没有改写或全局替换旧 auditor。新增 metadata 检查复用 native 通用 snapshot 校验器，因此原始 version-counter 字段也保留。

## 输入与输出

新增 `minicpm5_layer1_saved_operator_reader_v1.py` 提供：

1. `prepare(...)`：接受已经由 root 认证的 native plan/driver、24 份 full_trace JSON、SDK plan/两份 rank-result、两份原始 source-derived trace contract 和真实 prerequisite refs。重建固定 comparison contract，并仅提取 capture 描述；输出 JSON 可序列化的 descriptor plan。每个原 leaf 保留事件名、嵌套路径、完整 logical spec、原 source metadata 和稳定文件 reference。
2. `validate_plan(plan)`：重新检查全部 six calls、24 帧、两个 rank、每 call 214 leaves、所有唯一路径与 inode、完整 manifest geometry 和确切总字节；返回唯一 capture 文件 allowlist。
3. `StableCaptureIO(allowed, guard, filesystem=...)`：构造时无 IO。实际调用 `read_bytes` 才读取 allowlist 中的单个文件；测试显式传 memory FS，未来生产显式传 `os`。没有目录遍历或 inventory 生成方法。
4. `compare_all(plan, read_bytes, guard, emit_report, expected_plan_sha256=...)`：必须提供经 root 认证的 **descriptor plan canonical JSON SHA**。按原顺序串行读取/比较全部 six calls，调用原比较 v1，并逐 call 向外部 metadata sink 交付报告；完整穷尽后返回 summary。任意 guard、read、comparison 或 sink 失败立即抛错，不返回全局完成。sink 获得副本，不能修改保留用于最终汇总的报告。

native capture reference 格式为 `{path, raw_bytes, sha256, identity}`，SDK 相同。identity 固定包含 `realpath/device/inode/size/mtime_ns/ctime_ns`。每次实际读取检查完整原 identity、uid 1000、regular file、nlink 1、exact size、EOF、完整 SHA，并在读前/读后检查文件与所有父目录绑定；`O_NOFOLLOW` 防止链接替换。每 64 KiB 及关键边界调用 guard，异常会关闭全部句柄。

原 source stride/offset 原样保存在 descriptor plan。它们 **不用于 packed 文件字节寻址**；原比较 v1 只按 logical shape 访问已保存的 packed 数据。源布局与官方 donor/backing 的完整对应仍由既有全控制独立 auditor 认证，本 helper 不重复模型或 donor 算术。

## 原完整范围和资源边界

| 范围 | capture 文件 | 字节 |
|---|---:|---:|
| Native 24 帧 × 30 leaves | 720 | 14,897,124 |
| SDK 6 calls × 2 ranks × 47 leaves | 564 | 62,893,440 |
| 总比较输入 | **1,284** | **77,790,564** |

每 call 是 4 native 帧与 SDK 两 rank 的 **214 完整 leaves**。峰值 call payload 为 **23,920,808 B**，原比较上限仍为 32 MiB。只在当前 call 持有 raw maps，比较完成后释放；全六 call raw 不同时缓存。稳定读取拼接会产生最多一 leaf 的临时副本，比较 v1 的分域切片、浮点统计与 Python 元数据也有额外内存；这里没有宣称上述 payload 上限就是完整 RSS。

完整 descriptor plan 上限 16 MiB；单 call 原比较报告上限仍 2 MiB。所有范围均拒绝缺项、额外项、重复路径/inode 或超限，不能截断。原三臂 native **936 文件 / 28,490,724 B** 和 SDK 五控制 **624 文件 / 63,876,480 B** 的完整独立审计继续是前提；仅提取 capture 不代表删除控制。

## Root 后续实际接入步骤

当前 Mini native 尚未真正退出时，不调用下面的 raw reader。不存在可直接运行的“现成成功 spec”。

1. 沿原流程等待实际 native model terminal；执行已准备的 recorder v3，观察 recorder 真退出；然后在原串行条件下执行 audit v4，并由 root 观察 **auditor 自己**真退出。audit JSON 的 `complete` 或 `actual_auditor_tool_exit_verified: false` 不能代替外部 terminal。
2. 用原 metadata reader 认证完整 model/recorder/auditor refs 与源闭包、原始 full control proof、所有 owner/service scope，以及最新 catalog。SDK 使用已实现的 `minicpm5_layer1_operator_recovered_completion_v2.completed` 完整回读恢复链；保留原模型 **exit 1**，不能改成原 runtime 成功。SDK 的 recovery actual exit 和 recorder terminal 已在该既有模块绑定；本 helper 不复制新的常量。
3. 认证 native plan、driver result 和 driver 中每帧 full_trace `result` reference；其余两个 native arm 和完整缓存控制由 audit v4 proof 保证。SDK 两份 rank-result 必须来自被恢复审计认证的 `original_model_outputs`。SDK 完整原 `calls` 来自原认证 plan，不能按物理槽位重排其列表。native/SDK trace contracts 继续由原 source/code validators 构建，不从未经认证的 capture 自我接受。
4. 向 `prepare` 传入这些真实已认证对象与下表 refs。把产出的完整 descriptor plan 存为新批准输出，记录文件 reference 和 canonical JSON SHA；新 helper、比较 v1 及所有实际加载依赖纳入单独审查的 source extension。不得改写原 frozen sources、模型运行结果、当前 catalog 或已有 completion。
5. 在 root 的原串行执行框架中建立 fresh owner/lease 和完整当前 guard。guard 的责任包括真正的 prerequisite/控制审计、全部原 owner/service 范围、新近串行记录与当前资源；每个读取或计算阶段继续复查。`StableCaptureIO` **不会**自行认证这些条件；`prepare` 返回的 `runtime_admitted: false` 也不会授予运行资格。
6. 构建 `allowed = validate_plan(plan)`、`StableCaptureIO(allowed, guard, filesystem=os)`，调用 `compare_all`；metadata sink 使用既有独占/有界 JSON 输出规则保存全部六份报告。根线程应在运行前后重新确认已认证原输入/metadata/source 稳定性及原目录 inventory；新 reader 不扫描目录来替代已有完整 audit。
7. 观察本次比较进程真退出后，按原框架独立审计全部报告/summary、全部 1,284 原始文件的完整 descriptor/SHA 与计数，并做一致性复核，再发布科学结论。部分 per-call 报告不能被当成全局完成。调用者也可在既有独立审计阶段复用此 reader，不必引入新锁、监控系统或另一套生命周期协议。

`prerequisite_refs` 必须全部是来自以上真实认证链的 `{path, sha256}`；此交付不填假值、不提供“待填”的可运行 spec：

| 键 | 必须来自 |
|---|---|
| `native_model_exit` | root 观察 native model 真退出后的原 terminal |
| `native_recorder_terminal` | recorder v3 的独立真退出记录 |
| `native_audit_result` | audit v4 的完整原 936 payload/control 审计报告 |
| `native_auditor_terminal` | root 外部观察 audit v4 进程真退出的 terminal |
| `native_plan`、`native_driver` | 与上述链完全一致的原计划和 driver result |
| `sdk_recovery_actual_exit`、`sdk_recovery_recorder_terminal` | 既有 recovered completion 验证的恢复链 |
| `sdk_plan`、`sdk_rank0_result`、`sdk_rank1_result` | 被恢复链认证的原 SDK plan 与两 rank 原结果 |
| `source_review` | 批准本次 reader/comparator 以及实际依赖的独立 source review |

这些 refs 在 summary 中保留供追溯，但 **helper 不会把 refs 的存在冒充终态认证**；terminal/control/source 检查由原 root 接线承担。测试中的 `/synthetic/...` 只存在于内存夹具，绝不复制到实际 spec。

## 测试及未覆盖范围

```text
python3 -B /home/ubuntu/experiments/trn1-latest-models-20261031/research/check_minicpm5_layer1_saved_operator_reader_v1.py
```

9 项 stdlib/injected-FS 测试覆盖完整 descriptor 元数据、原 geometry/身份/event/slot 错误、missing/extra/重复 inode、真实 SHA/长度校验的 6 B 小型内存文件、未知路径/guard 提前拒绝、short/grown/hash/owner/link 故障、读中 file/parent/guard 变化及全部句柄关闭、六 call 调度顺序/完整分组/完整穷尽/metadata sink 隔离、错误 plan SHA 和读取失败不返回全局完成。

完整合成 descriptor plan 的序列化大小为 **1,640,025 B**；完整 1,284 个 payload 没有生成。每个独立内存 FS 只有 **6 B**，另有单字节增长故障。全 six-call 调度测试 **mock 了 raw reader 和 scientific comparator**，只证明元数据控制流；其模拟数量不是实际读取/比较证据。实际实验 raw 读取量为 **0 B**，未测量完整 RSS，也未测量真实存储性能。

保持 BF16、原模型、cold4096、长解码、请求槽位复用/inactive、quality96 和 matched HTTP 的原门槛；首次不同的保存边界仍不证明根因。本交付不新增提速倍数、模型合格或服务资格。
