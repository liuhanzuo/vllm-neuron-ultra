# MiniCPM5 后继实验接入审查，Nemotron 完整运行继续

2026-11-10，本轮核实 Nemotron r12 已完成 16/52 个阶段，原控制器仍持有串行锁并继续运行。MiniCPM5 的后继实验完成了退出记录、运行器和工作进程的源码接线与针对性检查；审查发现全量历史证据与旧源码读取策略不兼容，正在修正，因此尚未启动这轮真实实验。没有新增模型提速结果。

## MiniCPM5：等待真实结束记录，保留全部前序实验

新的结束记录校验要求 Nemotron 的真实工具退出，并区分成功与失败；若失败，必须另有原支持的审核及释放命令的真实成功退出，不能用状态文件代替。它保留 1,255 个历史服务，加上本轮 Nemotron 的 53 个服务，共 1,308 个服务；此前 MiniCPM5 的 180 次真实调用及独立审核记录同样保留。进程身份按完整记录保存，包括同一进程在不同命令阶段的记录。

独立审查发现部分早期进程记录只有 PID 和启动时间，原草稿强制要求命令行会错误拒绝这些真实记录。现已修复并加入回归检查，保留所有历史字段；实际检查进程是否消失仍依赖 PID 与启动时间。

后继运行器保留 TP2、原 16 GiB 内存限制、CPU 6–7、进程启动确认、完整清理和确认排空后才释放串行锁。完整前序排空由外层在启动前检查；工作进程验证该检查的完整哈希及文件身份、同一串行锁和全部前序进程，避免把自己的运行服务误判成其他实验。

| 本轮新增源码检查 | 通过数 | 说明 |
| --- | ---: | --- |
| Nemotron 真实结束与完整历史接入 | 19 | 根代理独立复跑通过 |
| MiniCPM5 完成记录、锁与历史身份 | 30 | 包含历史 PID/启动时间记录回归 |
| worker/runner 模拟生命周期 | 33 | 包含启动确认、失败清理、排空记录防替换 |

这些检查使用模拟接口或小型临时元数据，不执行真实 attention 数学。根代理另重新核对了已完成的 MiniCPM5 180 次调用元数据，确认五个实际进程身份与原独立审核链；未重读模型权重或张量载荷。

## 审查发现：旧“源码文件”规则不能直接承接全部历史

本轮 Nemotron 的完整来源映射有 43,912 个条目，其中含 14 份 Phi-4 原始观测、6 份 Dory 原始观测、9 个系统查询依赖库，以及 40 份超过旧 8 MiB 上限的元数据。新运行器第一版沿用了只允许普通小型源码的规则，实际使用完整历史时会拒绝启动。测试通过不能代替这一真实接入检查。

第一版源码及交付记录保留不动，后继版本将复用既有严格分类：保留全部来源及哈希，分开处理普通文本、已固定的特殊编码源码、历史载荷与查询依赖库；只对明确命名和结构的元数据使用已审查的大文件规则。准备阶段不读取模型载荷或系统库原始数据，原串行锁内才执行必要的有限认证。该接入与完整运行准入尚未完成。

独立退出后审核核心也已起草，要求 144 次调用、72 份完整输出、24 个独立解码帧及原 own/restore/both 控制；当前仍待独立审查和可执行审核器接线。未保存的 RoPE 原始字节不能被写成已独立重读。

## Dory 与 Nemotron 的实际边界

Dory 的静态审查确认固定 Torch 2.14.0+cpu 中有关复制、按原位重解释字节、选行及 argmax 的接口签名与原调用方式一致。但真实 CPU 路径的单例维度步长、零偏移、完整字节一致性和资源开销仍未验证。原 87 个检查点中的 7 个失败保留，不能把静态接口核对称为原生验证通过。

Nemotron 的 16 个已完成阶段逐项核对了成功结果、保留的实际退出、进程及资源组排空。观察时正在执行长度 1024 的投影对照，外层会话及控制器仍存活，原科学配置未变。整模型编译、数值、冷缓存、长解码、请求槽位复用、质量及真实 HTTP 性能仍等待完整流程。本轮没有产生可加入总表的新提速倍数。

证据：[本轮完整清单](evidence/minicpm5-runtime-lineage-20261110/manifest.json)、[完成记录核心审查](evidence/minicpm5-runtime-lineage-20261110/minicpm5-layer1-sdk-batch-layout-completion-root-source-review-20261110-v1.json)、[Nemotron 后继接入审查](evidence/minicpm5-runtime-lineage-20261110/minicpm5-layer1-sdk-batch-layout-r12-helper-root-review-20261110-v1.json)、[worker/runner 交付](evidence/minicpm5-runtime-lineage-20261110/minicpm5-layer1-sdk-batch-layout-worker-runner-source-delivery-20261110-v1.json)、[来源读取问题](evidence/minicpm5-runtime-lineage-20261110/minicpm5-layer1-sdk-batch-layout-source-policy-findings-20261110-v1.json)、[Dory 静态兼容性审阅](evidence/minicpm5-runtime-lineage-20261110/dory-stage-a-tensor-adapter-static-compatibility-review-20261110-v1.json)、[Nemotron 16 阶段实际核验](evidence/minicpm5-runtime-lineage-20261110/nemotron9b-r12-actual-progress-20261110-v4.json)。
