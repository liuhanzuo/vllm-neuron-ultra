# MiniCPM5 原生进程适配器：源码交付与接入边界

本次交付的是 `minicpm5_layer1_operator_native_process_adapter_v2.py`、`minicpm5_layer1_operator_native_runner_v3.py` 和其纯 mock 验证。它们将已接受的 lifecycle_core_v2 接到真实操作的实现代码，但没有创建实际运行的 admission，没有持有或操作串行锁，也没有运行 MiniCPM5。`launch_ready=false`。

## 实现范围

- `Lease` 观察外层进程保留的原始文件描述符，复用 descriptor_serial_lock_proof_v1 与 Mini 原协议的进程身份检查。任何失败都会锁存，不会自动重试或释放锁。明确的 Mini 描述符策略要求父进程独占、CLOEXEC、子进程不继承，且所属服务完全排空前不得释放。
- Admission 必须保留首次 descriptor 证明的完整 fdinfo/global 文本及前后身份。适配器重新检查其解析结果和外层 owner、原始 device/inode、fd、规范路径绑定。
- `resolve_admitted` 接收明确的 spec/review 引用，恢复原来的 SDK 成功数据审计，解析小 overlay，保留完整科学输入与来源集合，绑定完整 plan/catalog 摘要，并复用原协议验证保存的完整物理排空对、进行新的物理检查及检查最新 terminal。SDK rank 引用只来自 `recovered['recovery']['original_model_outputs']`。
- `Ports` 提供实际子进程启动、完整身份 ACK、显式释放消息、有限输出目录枚举、空间监测、子进程回收以及服务属性验证的实现。实际方法仅在 mock 边界内执行过。
- 新 runner 只有 `coordinator` 和 `native` 两个模式。原生子进程需先收到 parent 写下的 owner 记录，再从 stdin FIFO 收到绑定 admission/owner/plan 的完整单包及 EOF，才进入完整恢复检查和模型工作。包上限为 4096 字节，等待上限为 60 秒。
- 原生路径使用原 RuntimeContext 和 driver_v4，将 driver 返回值原样放进已接受的认证 wrapper。独立 raw verifier 仍应接收解包后的原 driver 对象。

## 外层接口

外层需产生经独立审查的 spec，其字段恰为 `schema, overlay, runner, unit, lock_policy`。对应 review 必须绑定同一 spec、overlay、lock_policy 及 resolved_plan_sha256，reviewer 为 `/root`，passed 为 true，blockers 为空。当前交付没有产生这些实际对象。

运行时 admission 需提供当前代码消费的以下绑定：schema、runtime_admitted、predecessors_checked、spec、review、overlay、contract、resolved_plan_sha256、unit、attempt_id、outer_owner、retained_fd、lock_policy、lock_identity、initial_descriptor_proof、source_contract、sdk_operator_recovery、latest_completion、completed_sdk_layout、serial_predecessors、observed_drains、source_authentication。外层必须根据真实操作建立这些数据，不能从 mock 构造准入记录。

`observed_drains` 指向保存原始完整物理 pair 的记录；需保留既有无损压缩格式、完整观察摘要、外层 owner 和 lock 身份。`source_authentication` 指向原始九项 native 依赖的一次完整认证记录。owner、source receipt 和 admission 的身份与内容在后续阶段继续检查。

外层需完成原锁的单次获取和保留、CLOEXEC、完整前驱排空、源认证、服务创建和 ACK、服务结束后的完整组排空，以及自身之外的实际退出记录。coordinator/native 不提供其中任何替代授权。原锁身份仍为 device 66306 / inode 8294421。

## 保持的模型和资源条件

原模型、权重、BF16、原始解释器和科学 manifest 不变。原生服务保持 MemoryMax 16 GiB、MemoryHigh 14 GiB、无 swap、TasksMax 128、CPU 6–7、Nice 19、7200 秒运行上限、15 秒停止超时、16 MiB 单文件上限。输出总分配量上限 256 MiB，native 部分 128 MiB；根盘 28 GiB 和 NVMe 30 GiB 可用空间底线继续检查。

完整原生验证仍包括 24 frames、72 个 attention 调用、936 raw 文件及原有 journals/results；live 输出枚举涵盖 97 个内部目录及根目录分配量，严格识别已知写进程的 pending→final 原子更名。冷缓存、长解码和请求槽位复用的既有研究要求没有被此基础设施交付替代或删减。

历史 SDK 模型进程的 exit 1 保留。独立恢复审计成功只允许使用恢复的数据，不能改写模型运行结论，也不能直接用于性能或整模型合格声明。

## 实际通过的验证

恢复摘要记录过 tests_v4 的 15 项测试成功，但本次没有直接读取那次会话的终端。因此仅复跑了同一份纯 mock 测试，未修改任何已执行源码：

`python3 -B research/check_minicpm5_layer1_operator_native_process_adapter_v4.py`

本次真实工具 session 为 71010，初始 chunk 为 b3da49，最终 chunk 为 e649b0，exit code 为 0；终端显示 `Ran 15 tests in 9.344s`、`OK`。

覆盖无副作用构造、原 fd/proc 绑定、失败锁存、单子进程环境和无锁继承、实际 ACK 条件、释放消息/EOF/FIFO/时限及拒绝路径、完整服务资源与命令/Invocation/noJob 检查、原进程组 drain 边界、初始 descriptor 证明篡改拒绝、1080 文件和 97 目录的合成枚举、合法原子更名与额外路径/外来 PID/symlink 拒绝，以及原 Context/driver/wrapper 的接线。进程、文件系统输出、proc、模型执行和 driver.run 均在 mock 边界内，未访问实际 raw 或启动原生/模型任务。

此前 runner_v1 的语法失败与 tests_v3 的两项 fixture 错误继续保存在各自 failure JSON 中；没有覆盖失败版本或把失败记录改为成功。tests_v4 只修正了 mock 对 Path 参数的兼容，adapter_v2/runner_v3 未因此改变。

## 依赖闭包

`minicpm5-native-process-static-source-closure-20261111-v1.json` 使用既有 `prepare_dory_residual_spec_v3.local_imports` 对声明的 `research, ROOT` 搜索顺序进行递归 AST 分析，覆盖函数内延迟的静态 import。runtime 闭包为 569 个本地 Python 文件、6,997,972 字节；测试闭包为 586 个文件、7,223,017 字节，均有每个路径的 SHA256 和 import edges。实际静态分析 session 52655，terminal 41c9ae，exit 0。

这份集合是附加本地源码依赖，不能替代或裁减完整恢复后的科学 source map。原来源分类、所有既有 pin、外部包及 native 二进制认证义务保持不变。报告单独列出外部 import 根与 71 个可能动态执行位置；动态导入、外部包完整闭包及运行中的实际导入身份没有由静态分析证明，相关布尔值明确为 false。未读取模型权重、raw 或 native 二进制来生成此报告。

## 仍未接入的部分

1. 本次没有 outer/acquire/service-launch/admission producer。原描述符持有期限、失败后的无限保留和实际释放仍必须由外层实现并审查，不能依赖 Lease 的失败提示实现安全清理。
2. 原 `sources.authenticate_native_sources` 内部仍调用旧 `p.lock_proof`。在采用新 descriptor 策略的外层中不能不加审查地直接复用；需要单独的兼容实现，保持原九项依赖、完整字节认证和同一 receipt 语义。
3. 原生完整 resolved/source/physical 恢复路径及 bootstrap 尚未使用实际历史输入执行；当前 tests 覆盖的是适配器隔离行为和原生入口接线。实际 catalog 的最新性、source extension 的完整性及 source review 需由外层结合当前状态完成。本交付不生成具体 overlay/plan，不修改 serial catalog 或 handoff。
4. 外层与 `Ports.assert_service_drained` 必须明确谁独占写 `cleanup-drained.json`，防止两个清理路径重复创建同一 exclusive receipt。coordinator 的 child cleanup 已由 lifecycle_core_v2 的单个 finally 管理。
5. live inventory 是运行中观察，目录分配量采样后可能继续增长；后续监测与最终不可变审计仍不可省略。native guard 每次可能重复枚举输出，其实际开销未测量，不能作为优化性能结论。
6. 仍需外部实际模型进程退出、完整服务排空、独立 post-exit raw 审计及其实际 recorder/auditor 退出。没有这些证据，模型通过状态和提速倍数均不得更新。

当前源码交付待 root 独立审查，既非一次实际原生运行，也非新的模型合格或提速结果。

## Root 复核新增的启动阻断：运行中 handoff 与 freshness 不兼容

这是纯文档后继；v1 文档、交付 JSON 和全部已执行源码保持不变。Root 已重跑同一份 15 项 mock：session 19441，terminal c4efeb，exit 0。通过的隔离测试并未覆盖下面的整体启动问题。

当前 adapter_v2 在 coordinator/native 启动后的 `resolve_admitted` 内调用 `completion.latest.require_latest_terminal`。后者的源码说明限定为“Fresh preparation/outer only”；函数仅提供 paths/stat/read_text 注入，没有针对本次已准入运行的 owner、admission 或 active-unit 例外接口。最新 handoff 如果不是 catalog 绑定的 idle terminal，就必须为 publication-only 链，而且 `live_tool_sessions` 和 `active_heavy_units` 必须都为空。

因此，外层按既有流程记录本次 native running handoff 后，即使本次运行已获得合法串行授权，当前 child bootstrap 仍会拒绝。Root 已使用原 completion_v3 的纯 mock 实际复现：tool call terminal 57c714，exit 0；预期拒绝信息为 `Every later handoff is explicitly publication-only with unchanged complete serial state`。该工具成功退出证明复现检查通过，并不表示 native bootstrap 成功。

这是交付清单中的第 8 项 remaining gap，属于 launch blocker。当前 adapter_v2/runner_v3 不能标记为整体 accepted 或 launch-ready。后继源码需将外层准入前的 freshness 检查和 child 对本次已准入运行身份的检查分开，同时保留完整前驱排空、并发活动检查、原锁身份、owner/admission 绑定和失败时的所有权保留。不应通过省略检查、保持过时 idle 状态或伪造 publication-only handoff 绕过它。

本次只补充文档，不实施该接口改变，不创建实际 spec/overlay/admission，不修改 serial catalog/handoff。
