Qwen3.5-2B 的独立 CPU sampler 后继已完成源码准备和 20 项纯 Python / AST 检查。没有启动 native、模型、Neuron、服务或队列，没有新的性能结果，也没有修改冻结的模型、安装目录、continuation state 或交接状态。当前交付包含有限串行 driver、真实评测客户端调用、原科学流程的 native leaf 实现、稳定 worker 入口和采集/重放代码；实际执行仍需根任务绑定现有资源与进程所有权保护。

交付证据：[JSON](qwen35-sampler-pair-source-delivery-20261111-v1.json)，SHA256 `2b50426e39728dc20800aa0030544af9b37307bca01c61e39de5dd4320bef04b`。测试记录：[JSON](qwen35-sampler-pair-source-tests-20261111-v1.json) / [日志](qwen35-sampler-pair-source-tests-20261111-v1.txt)。所有 native 测试与真实完整配对实验仍待执行；纯测试中的成功回执和速率使用明确的替身输入，不能作为实验数据。

选用的比较基线是已经完整通过声明测试的 Qwen/Qwen3.5-2B 文本适配器，官方 revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，原 BF16 权重、FP32 recurrent state、TP2、B4 decode / B1 prefill、context4096、修正后的 tensor_split 和标准 CPU sampler。它属于实验性 vLLM-Neuron 文本适配器。视觉、MTP、其他模型规格以及通用准确率没有因此获得资格。

候选只替换每个 runner 实例的 `cpu_sampler.topk_topp_sampler`。bounded operation 与仓库中的 sampler 源码逐字节相同，SHA256 `9638cd6189a819684efe02fb25b959065c9c5ea5cfc3c45ea3a9153b9ce4289a`。原权重、GDN、split、图缓存、温度/惩罚/processor、greedy 路径和 DEBUG 行为保留。旧 `TRN1_CPU_SAMPLER_FAST` 开关同时安装 DEBUG guard，不能用于这个独立对照；新入口拒绝该开关及其他旧 sampler/observer 开关。

原基线短随机 C4 约 20.91 tok/s，短 greedy C4 约 39.58 tok/s。这只能说明值得测量采样开销：两个工作负载的采样条件不同，不能把差值归因于 sampler，也不能当作预期提速倍数。

| 源码 | 实际职责 |
| --- | --- |
| `qwen35_sampler_pair_protocol_v1.py` | 原模型 case 与新 host identity 分开，固定全部科学门槛和有限采集预算 |
| `qwen35_sampler_instance_v1.py` | 调用原 worker factory 后，仅附着当前 sampler 实例；测量臂没有采集 hook |
| `qwen35_sampler_bounded_operation_v1.py` | 从已合入仓库复制的原 bounded operation；不改安装目录 |
| `qwen35_sampler_worker_v1.py` | 四个稳定、可导入的 worker 名称，实际启动附着回执，以及有限 NVMe packet writer |
| `qwen35_sampler_native_capture_v1.py` | 在原 top-k/top-p 调用边界采集真实完整 processed logits、有效 metadata、request/slot 与前后 RNG |
| `qwen35_sampler_native_replay_v1.py` | 对新采集做完整分布和自身 token/RNG 重放；14 项 native rejection controls 的实现 |
| `qwen35_sampler_full_checks_v1.py` | 源码 pin 验证后复用原验证函数体，校验完整模型 metadata 和全部质量文本 |
| `qwen35_sampler_native_leaves_v1.py` | 原 numeric32、cold84、long1541/87、state-copy 科学流程及全部203原始向量比较；入口首先要求根执行 guard |
| `qwen35_sampler_pair_driver_v1.py` | 有限串行阶段顺序、真实原 HTTP/quality 调用、fresh NVMe 输出、独立 root live/terminal 审计接口 |
| `test_qwen35_sampler_pair_source_v1.py` | 20 项 stdlib / AST / 替身测试；包括新解释器导入与完整 driver 的失败顺序 |

服务继续执行冻结的 `server_with_qwen35_split_v4.py`，通过 `--worker-cls` 加载以下稳定名称，而不是向 spawn 传递局部类对象：

- `qwen35_sampler_worker_v1.BaselineMeasuredWorker`
- `qwen35_sampler_worker_v1.CandidateMeasuredWorker`
- `qwen35_sampler_worker_v1.BaselineCaptureWorker`
- `qwen35_sampler_worker_v1.CandidateCaptureWorker`

原 Neuron platform 仅在 `worker_cls == "auto"` 时选择默认 worker；显式名称可以保留。两臂都需要相同的 research 路径加入 Python 模块搜索路径。模块按名称解析时才导入 native NeuronWorker，缓存导出的类，并赋予稳定 module/name/qualname。测试已在两个新 Python 解释器中验证四个类的 pickle 和重新导入；使用的是 stdlib NeuronWorker 替身，真实 vLLM/Neuron spawn 仍须验证。

根执行适配器必须为每个服务提供新的 `QWEN35_SAMPLER_HOST_IDENTITY` 和 `QWEN35_SAMPLER_HOST_RECEIPT`；采集服务额外使用 `QWEN35_SAMPLER_CAPTURE_ROOT=<阶段目录>/packets`。实际 worker 会校验 host source pins，并写入自身 PID、实际 worker/operator/reference 类和附着方式。根审计须独立确认该 PID 的 start ticks 与服务 cgroup / invocation 归属，不能把启动配置视作实际加载证据。

服务存活状态与结束状态已经拆开。driver 在 context 内独立调用根 `server_audit`，要求真实 owner PID/start ticks/invocation、ready、worker 归属、`exit_observed=false`，不接受运行中服务的 `returncode=0`。context 退出后，包括客户端抛错路径，再调用根审计确认同一 invocation 已终止、实际 systemd exit status、受控停止与空的剩余进程列表。替身测试覆盖了这一顺序、提前声明终止、owner 变化、清理不完整和客户端失败后的清理。

模型 case 必须原样共享，旧 artifact-provenance 文件及图 source map 不能加入新的 host 文件。根执行适配器使用原 cache validator 校验图复用，host pins 独立记录。当前 8 个大文件的全部原 stat identity 均匹配；旧 graph source pins 也匹配。没有重新读取权重/原始张量或重哈希 model.pt，因此这里不构成新的内容验证或执行准入。

原运行器还有一个必须绕开的复用行为：`orchestrate.run_case(case)` 会返回同名已通过结果，并可能读取旧同名 HTTP JSON。两臂保持同一冻结 case，不能通过它获得新计时。后继 driver 直接调用原 `bench_http.benchmark`、`benchmark_workloads.evaluate` 和 `evaluate_quality.evaluate`，只向新的 arm 目录写结果；已有输出根目录会被拒绝。不会重写全局 results 或 best-known 文件。

完整流程对两臂都保留以下内容：

- state-copy 的 96 bindings、192 次 fill 比较与192次零检查。
- numeric32、cold84、672 次原始零检查、42 组重复槽比较、全部12图路由和 context4096；cold 保留七个 prefix，包括2052，以及原72帧重现。
- 原 long1541/87、原 full-causal oracle 与 incremental crosscheck、槽位复用、partial batch 和 inactive resume。
- 两臂全部203原始向量都必须与已合格基线逐字节相同。
- 原 smoke、quality96；重新评分所有保存文本，基线与候选均须复现69/96且96文本相同，无 lost-correct。
- 原 initial120 三轮 single/C4 和 streaming，以及原5类×3轮完整 HTTP；每臂124条计量请求、30848输出 token、全部57条 greedy 文本相同，保留 warmup、streaming、mixed16、原 timeout/force/ignore_eos 和 seed 缺省。

采样正确性额外使用独立的有限诊断：单请求和四并发两波，共5条请求、每条128输入/32输出、每臂160输出 token，温度0.7、top_p0.9、top_k40，不添加 seed。它不替代任何原完整工作负载。只采集每种实际行数 B1–B4 的前16次 top-k/top-p 调用：每臂最多64个 packet、160行完整248320词表分布，B1/B4各16次为必需；B2/B3仅在实际调度出现时记录。达不到必需配额则诊断失败，不自动重试，也不虚构覆盖。

每臂 packet 总量上限256 MiB，单 packet12 MiB，JSON header1 MiB。总量包含 packet header、metadata、所有输入/输出张量和 RNG。溢出、未知有效字段、schema变化或 writer 短写都停止诊断，不截断后宣称完成。两臂测量阶段均不安装观察 hook。

每个新 packet 保存温度和 penalties 等实际 effective metadata、原空 processor 链的实际观测、请求历史/槽位、原始调用 k/p、完整 processed logits，以及全局 CPU 和显式请求 generator 的前后状态。缺失的历史 repetition penalty 或 RNG 不做补推。native 重放要求原支持集一致、total variation≤1e-6、最大概率误差≤1e-6，以及本臂实际 token/RNG 可重现。显式请求 generator 必须整次回退原调用，保持 token 和 RNG；未显式指定 generator 的随机 token 序列不要求两臂相同。

截至交付，根盘剩余约28.604 GiB、NVMe约44.025 GiB，仅为只读观察。原28/30 GiB floor 保持。每臂203原始 FP32 向量为201,635,840字节，两臂加最大 packet 总量合计940,142,592字节；这个数不含序列化、状态/HTTP/审计 metadata、日志、临时文件、文件系统分配以及峰值内存。新的运行输出要求在 NVMe。根任务必须在当前 native owner 结束后重新检查资源，并给总输出/日志/临时空间、RSS、swap 和期限设定实际边界，不能用本报告的空闲观察代替准入。

剩余工作是根任务将 `GuardedPorts` 绑定到现有真实 owner/resource/服务管理实现，pin 该适配器及独立根审计源码，验证真实 native worker 与 packet roundtrip，然后执行完整配对流程并独立审核结果。当前 driver 故意不生成 `full_model_qualified=true` 或自动推广 best-known；真实所有 gate 与最终审计完成前，不发布候选提速。

此前 GDN 候选的43条质量文本变化和57条 greedy 工作负载变化仍保留为不合格记录。Smol 的2048证据没有升级为4096，其他模型家族、原错误范围和长期目标没有被替换。
