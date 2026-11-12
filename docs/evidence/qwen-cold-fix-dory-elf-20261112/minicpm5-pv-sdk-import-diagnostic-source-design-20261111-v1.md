# MiniCPM5 SDK 导入诊断：独立源码实施设计

本文件是待实施设计，没有可调用的运行入口，也不是运行准入。它诊断原 SDK 环境在当前有限预算下能否完成原导入前路径和框架导入；原五阶段 PV 实验仍须完整执行，不能由这个诊断替代。当前 Qwen 实验结束并核对完整排空之前不得运行。

## 最小源码边界

建议新增三个文件：

1. `minicpm5_pv_sdk_import_diagnostic_v1.py`：独立诊断协议、父进程和叶子实现。重用原 prerequisites v2 的完整认证、原进程/锁观察工具、原 Store 的字节预算和已审 PV v3 退出观察逻辑。
2. `run_minicpm5_pv_sdk_import_diagnostic_v1.py`：只接收两个明确的未来 spec/review 引用；导入模块不启动任务。
3. `minicpm5_pv_sdk_import_diagnostic_worker_v1.py`：只允许原 SDK Python 3.10、`-B` 和诊断中明确绑定的阶段 1。

后续源码交付须包含纯检查和逐块正常化差异；不能在本设计中选定当前 catalog、admission、进程身份或输出路径。

## 认证与运行顺序

顶层诊断 spec 使用独立 schema `minicpm5_fixed_input_pv_sdk_import_diagnostic_spec_v1`，绑定完整的原 PV 前置认证 spec/review 引用、诊断 worker、原固定 limits/artifact limits 和源码扩展。前置认证仍使用原 `auth.authenticate`，重建原已完成比较/独立审计、完整描述计划和全部来源；其 spec 使用未来最新完整串行历史、新输出目录及包含诊断源码的新增来源集合。不能通过沿用 r4 的旧 current catalog、删来源或只验证框架几个入口来代替这一步。

父进程继续使用原 native outer 解释器及原锁，在认证前施加原 2 GiB AS、600/605 CPU 秒、900 秒 wall、CPU6/Nice19、2 MiB 单文件限制。采用新诊断 admission schema，但保留原 lease/witness/原生来源 receipt 字段及原完整 SHA、descriptor、真实外层所有权绑定。

只启动一个明确标记为诊断的 SDK 子进程。`SingleSDKAdapter.start` 保留原 SDK `phase_argv(..., index=1)`、原 `p.environment({}, 'cpu_control', attempt)`、原 `close_fds/start_new_session`、实际可执行文件/父进程/argv/cgroup 核验、所有权记录、整份来源通过原有限 stdin 包释放和 EOF gate。它使用独立的“只允许一次 SDK child”状态机；**不能通过给 `last_exit` 填 0 来伪造已完成 phase0，也不能调用 FivePhases 后跳过任何科学阶段。**

诊断子进程显式重用或逐行保留 PV v3 `worker` 从入口直到原 SDK 导入位置的完整逻辑：

- 精确原解释器、断言与 `-B`，再次施加同一原限制；
- admission/lease/actual owner/argv/完整 release 包核验；
- 完整来源字典摘要和数量、来源分类、全部文本校验、原九项 native 来源身份检查、已加载源码检查；
- 从原六份报告与 summary 完整重建原描述计划；
- 原 `validate_plan`、`StableCaptureIO` 对象构造、原 Store 全目录检查；
- 初始 own-before 阶段的 `own_gate is None` 断言及原环境/stage/plan 绑定。

原 `StableCaptureIO` 仅构造，诊断不调用其原始张量读取接口。任何“裸 import torch”、不同解释器、跳过完整来源/描述重建的快速试验，都不得标记为该诊断的等效复现。新的独立目录不会冒充 r4 的历史目录；本诊断复现的是完整原代码路径和约束，不声称能重建已经失去的 r4 内存快照或相同虚拟地址布局。

## 导入观察与输出

在原 `import torch` 前写入/缓存一份有界状态，随后按原顺序执行 `import torch`、原 `repeat_kv` 导入、`authenticate_framework`、原 `set_num_threads` 和 `verify_loaded`。记录明确的阶段标识，区分 Torch 导入、SDK attention 模块导入及后续来源认证失败；不加载模型、不执行 PVLeaf 或张量计算、不切换 allocator，不预加载/禁用任何库。

状态内容只包括：实际解释器、原完整环境字典、`getrlimit(AS/CPU/FSIZE)`、CPU affinity/nice，以及原始 `/proc/self/status` 中 `VmSize/VmPeak/VmRSS/Threads` 的存在性与实际值。读取 status 有固定 64 KiB 上限并核对 EOF，不能把缺少字段补成零。保存导入前和成功后/异常后的观察时间及顺序。

异常记录包含实际异常类型、消息、完整 traceback 和 `__cause__/__context__` 链上的实际 errno；errno 不存在时明确记为未取得，不猜测 ENOMEM。不得只保留最后一行 traceback 或把库名解释成实际 CUDA 执行。检查所有异常对象的有界、无环遍历，禁止无限递归。

保留原 Store 可识别的 `admission.json`、`phase-1-owner.json`、`phase-1-summary.json` 和 `phase-1-exit.json` 文件名，但 admission/summary 使用独立诊断 schema；新目录必须明确为 SDK import diagnostic。不生成 own-gate/final 科学成功文件。summary 总量保持原 16 KiB 保守上限；原运行 metadata 保留 79,008 字节上限，整个目录仍受 8 MiB metadata、32 MiB allocation、2 MiB 单文件上限约束。若完整错误记录装不下，诊断失败并报告完整错误记录未保存，不能截断后声称完整。源字典继续走原 16 MiB 输入管道，不能另写一份扩大输出预算。

正常导入返回 0，导入异常返回真实非零值；子进程只报告自己的观察，不能报告自己的实际工具退出。父进程复用 PV v3 的严格身份转换、真实 wait、全部 owned group 排空、原 owner 消失与空组检查，持久化真实 returncode。外层保持该返回状态；失败实验仍是失败，完整诊断观察不等于导入成功。报告写入或 guard 自身失败时，同样排空并保留真实退出；缺失报告不能补造。进程真实退出记录也不能被 summary 里的布尔字段替代。

## 需要的纯检查

- 对原 v3 worker 的完整导入前 AST/源码片段做明确逐块对比，只允许诊断 schema 字面量和状态采样插入；确认所有原认证调用和顺序保留。
- 单 SDK child 状态机不构造 phase0 完成记录，不调用 FivePhases，不释放 own60 gate。
- 原 SDK 环境和 argv 完全一致；父子限制调用值和先后顺序保留。
- 原错误竞争的 17 项检查针对实际新诊断结束逻辑复跑；非零 import 不能作为成功，真实 wait/drain 先于退出记录。
- mock status 缺字段、错误单位、超限/未 EOF、异常链环、长 traceback、磁盘分配越界均拒绝；不存在的 errno 保持未取得。
- mock Torch/SDK 导入及认证错误标记正确；所有真实 native 导入、proc/锁/服务/模型调用在纯检查中明确禁止或替换。

只有完成上述源码交付和 root 独立审阅后，才绑定当时的完整串行历史并排队运行。若有界状态和异常链仍没有给出失败系统调用的 errno/库提供者，则继续保留根因未确定；后续如需系统调用或 loader 观察，必须另行设计有界、来源已认证的工具，不能私自换工具、提高预算或换框架。
