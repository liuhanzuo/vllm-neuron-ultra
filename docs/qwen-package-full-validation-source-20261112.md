# Qwen3.5 独立包整模型验证源码接线

Qwen3.5-2B 独立包的整模型验证 v2 已完成源码审阅并获主任务接受。它保留原数值、状态、槽位、采样器、质量和 HTTP 门槛，接入 package 自身的 build/load/serve 和独立 artifact 准入。**本页记录源码准备与 mock 检查；尚未运行这套新 package 的12图整模型验证，没有新增模型资格或性能结论。** 已完成的独立包初始化检查另见[原生预检结果](qwen-package-preflight-passed-20261112.md)，研究实现的结果另见[r10 结果](qwen35-r10-results-20261112.md)。

完整冻结源码、测试输出、交付清单和主任务接受记录均在[证据 SHA256 清单](evidence/qwen-package-full-validation-source-20261112/manifest.json)。复制品保留原字节和来源路径；仓库内的归档没有重新改写冻结源码中的绝对路径，也没有安装新的运行调度器。

v1 曾遗漏原 r10 的 **实际 sampler capture/replay 和13个 controls**。只有数值、quality96 或 HTTP 接线不能弥补这个遗漏。此次明确保留 [v1 交付](evidence/qwen-package-full-validation-source-20261112/v1/source_delivery.json)、[v1 模型入口](evidence/qwen-package-full-validation-source-20261112/v1/entry.py)与 [v1 HTTP 入口](evidence/qwen-package-full-validation-source-20261112/v1/http.py)，在新的 v2 目录补齐；没有修改 v1、package 或原研究来源。冻结检查确认 v1 交付覆盖的19个文件仍为原哈希。

v2 的具体变更见[后继版本差异](evidence/qwen-package-full-validation-source-20261112/v2/successor_v2.patch)、[科学模型入口差异](evidence/qwen-package-full-validation-source-20261112/v2/scientific_port.patch)和 [sampler 科学差异](evidence/qwen-package-full-validation-source-20261112/v2/sampler_science.patch)。完整说明见[模型接线审阅](evidence/qwen-package-full-validation-source-20261112/v2/REVIEW.md)与 [sampler 接线审阅](evidence/qwen-package-full-validation-source-20261112/v2/SAMPLER_REVIEW.md)。

v2 使用新 case、新 view、新 artifact、新 compiler cache 和 workdir。配置固定为原 BF16 权重、FP32 recurrent state、TP2、context B1、decode B4、context4096，保留六个 context 与六个 decode bucket：128/256/512/1024/2048/4096。完整 [Neuron 配置](evidence/qwen-package-full-validation-source-20261112/v2/neuron_config.json)同时用于 build 和 serve，包含与实际 loader 一致的 `pa_num_blocks=4`、`pa_block_size=4096`。只有新 build 进程允许编译；后续各阶段必须通过 package 独立 artifact identity 准入并禁用编译，不调用旧研究 `graph_reuse`。

build 还必须从实际加载的模型记录12个图路由、12份新 NEFF 和96个实际状态布局。配置数量、manifest 或文件哈希不能替代真实编译进程退出、数值检查或资格审计。具体调用在[模型入口](evidence/qwen-package-full-validation-source-20261112/v2/entry.py)与[配置和加载入口](evidence/qwen-package-full-validation-source-20261112/v2/runtime.py)。

| 必须保留的门槛 | 完整范围 |
|---|---|
| 状态复制 | 两组模式、96状态、192次逐字节 fill 比较、192次 raw-zero 比较；binding 不变；独立探针进程不加载图 |
| 数值 | 32个完整248320词表帧；混合长度、重复 prompts、permuted slots、3步 teacher-forced decode；cosine≥.995、KL≤.02、maxabs≤2、margin>.5 的 argmax 全匹配、总匹配率≥.9 |
| 冷状态 | 7 prefixes、84帧、672次 raw-zero、42组 duplicate pairs、7次原地 clear、96 binding 不变；全部12路由；本次实际逐字节复现原 partial 的72帧 |
| 长解码与槽位 | 完整1541步、87帧；原边界、slot reuse、B3 partial/inactive -1、inactive resume；原完整 causal HF oracle 和 incremental crosscheck |
| 张量复现 | numeric32+cold84+long87，共203个新保存向量逐字节对照原合格基线，不能从源码相同推定 |
| 实际 sampler capture | 两臂各5请求、160输出 tokens；真实 processed logits、完整 metadata、request/slot、输出和前后 RNG；每臂必须取得16次B1和16次B4调用；B2/B3只记录真实出现的调用 |
| 完整 replay | 每个实际包、整个248320词表；support 相同、TV≤1e-6、最大概率误差≤1e-6；本臂 token/output/global与request RNG 精确复现；显式 generator 整次调用 fallback；每臂须有实际 fast rows |
| 13 controls | 原 named controls 全部执行；每项输出或原异常及 RNG 一致；fallback 恰一次原 reference 调用；最终恢复 global RNG |
| Smoke 与 quality96 | serial4/concurrent4/identical-prompt4；原96题、c4、temperature0、无截断、全部文本重新评分；两臂均69/96、全部96文本与原基线一致、零 lost cases |
| 初始 HTTP | 原120输出、3 rounds、single+c4+streaming c4、8-token warmup；prompt/usage/greedy文本一致 |
| 完整 HTTP | 5 workloads×3 rounds、124请求、30848输出 tokens、57 greedy文本对；16-request mixed/c4、streaming、4×8-token warmup；原不传seed、240s timeout、min_tokens/ignore_eos |

sampler capture 只观察 package runner 已安装的实际 sampler 实例，不重复包装候选。完整 replay 源码只将两处候选 import 从研究模块改为 package `cpu_sampler`；撤回这两处 import 后，整个文件 AST 与原文一致。原 writer、两个 native replay/control leaf、package `_shortlist` 和 bounded sampler 的科学 AST 也核对一致。运行时还必须验证实际 package operation FQN，源码一致不代表新实测通过。

原13项 controls 为：`partial_explicit_generators`、`all_explicit_generators`、`k_absent`、`k_zero`、`k_above64`、`non_float32`、`invalid_p`、`processed_logits`、`processed_logprobs`、`excessive_kth_ties`、`nonfinite_max_or_kth`、`cdf_near_boundary`、`cdf_boundary_tie`。它们共享执行一次，与原 r10 相同。capture 每种行数最多16次调用，每臂最多64包/160行、每包12MiB、每臂256MiB；缺少必须配额时失败，不补 seed 或重试。

[Sampler 入口](evidence/qwen-package-full-validation-source-20261112/v2/sampler.py)把两臂 replay 与全部13 controls 设为每个 quality/HTTP client 的前置门槛。[HTTP 入口](evidence/qwen-package-full-validation-source-20261112/v2/http.py)在计时前还要求两臂 quality pair 通过。正式计时使用全新、无 capture observer 的 package server；测量臂出现 capture worker 参数或专用 capture 环境变量会被拒绝。

以下是[冻结阶段清单](evidence/qwen-package-full-validation-source-20261112/v2/stage_order.json)的完整25阶段。baseline/candidate 是同一份新 package 图上的 `NEURON_CPU_SAMPLER_FAST=0/1` 两臂。root 必须在相邻重阶段间取得真实退出与进程组清理证据；这张表本身不授予运行权限或资源准入。

| 序号 | 阶段 | 臂与必要结果 |
|---|---|---|
| 1 | prepare_view | 独立新视图，验证原官方文件 |
| 2 | build12 | 新 artifact/cache/workdir；实际编译、加载并记录12图 |
| 3 | state_copy | baseline：96状态、192 fill、192 zero |
| 4 | numeric | baseline：32帧 |
| 5 | cold | baseline：84帧与完整冷状态门槛 |
| 6 | long | baseline：1541步、87帧与槽位门槛 |
| 7 | vectors203 | baseline：全部203新向量逐字节复现 |
| 8 | state_copy | candidate：96状态、192 fill、192 zero |
| 9 | numeric | candidate：32帧 |
| 10 | cold | candidate：84帧与完整冷状态门槛 |
| 11 | long | candidate：1541步、87帧与槽位门槛 |
| 12 | vectors203 | candidate：全部203新向量逐字节复现 |
| 13 | correctness_capture | baseline：实际 worker/环境核对，5请求；完成后停止并清理 server |
| 14 | sampler_replay | baseline：独立进程完整重放所有实际包 |
| 15 | correctness_capture | candidate：实际 package sampler，5请求；完成后停止并清理 server |
| 16 | sampler_replay | candidate：独立进程完整重放所有实际包 |
| 17 | sampler_controls | shared：独立进程执行原13 controls |
| 18 | sampler_pair_gate | 两臂包/分布/RNG与13 controls 门槛；root 认证实际退出和清理 |
| 19 | quality | baseline：全新无 observer server，smoke与quality96 |
| 20 | quality | candidate：全新无 observer server，smoke与quality96 |
| 21 | quality_pair_gate | 两臂全部96文本/分数/输入核对，先于任何计时 |
| 22 | matched_http | baseline：全新无 observer server，initial120及完整workloads |
| 23 | matched_http | candidate：同样完整请求，实际 host/debug/thread 设置匹配 |
| 24 | full_pair_gate | 完整quality、初始及完整HTTP配对检查 |
| 25 | independent_final_audit | 原 root 独立核验来源、输入、artifact、实际生命周期与资格 |

本次证据是源码层面的接受：

| 检查 | 已记录结果与限制 |
|---|---|
| 作者最终检查 | pinned Python3.10，30项通过，exit0，工具 terminal `1a325b`；额外 `sys.meta_path` 阻断 native 依赖；原始[测试记录](evidence/qwen-package-full-validation-source-20261112/v2/source_checks.json)与[输出](evidence/qwen-package-full-validation-source-20261112/v2/source_checks.log)完整复制 |
| 独立复核 | root 接受记录载明：独立 reviewer 在 pinned Python3.10 下完成最终30项检查，工具 `43d9f7`/exit0，并启用 native import blocker；未另存独立测试日志 |
| Root 接受 | [接受记录](evidence/qwen-package-full-validation-source-20261112/qwen35-package-full-validation-v2-root-source-review-20261112-v1.json)标记 source_accepted=true、无源码 blocker，记录60个原输入 pins、39个 capture pins、25阶段；root 未重复运行测试，也未启动 native runtime |

独立 reviewer 没有另存冻结 JSON 或测试日志；独立复核一行仅引用 root 接受记录中的 `independent_reported_final_test`，没有重建独立实测文件。30项检查包含15项原整模型 source/metadata 和15项 sampler source/mock，涵盖完整 AST、v1冻结、12路由、配置匹配、缺包/配额/词表、分布越界/NaN、RNG失败、遗漏/改名control、错误包装/FQN、缺前置门槛阻断以及测量污染拒绝。mock 正例不生成新的模型或性能通过结论。

独立包仍须由原 root 接入实际生命周期、完整 History/来源绑定与资源准入：在运行前核验内存、NVMe、时间及已有任务清理；认证 producer 的 PID/start/invocation/cgroup、实际 API/worker 参数和环境；记录实际编译/子进程退出和递归进程组清理；最后核验 artifact 未变并完成独立资格审计。不能用旧研究图、旧 sampler 通过记录或旧吞吐量代替上述新运行。

同期槽位原语诊断是另一项工作。主任务报告其 session `31827` 首轮编译 identity 通过后遇到 protobuf HLO parser 失败、实际调用数为0，正在准备后继修复；这不构成本页任何整模型、采样或性能结果。

本次仓库归档只新增本页及其证据目录，不改 package、冻结研究来源或已接受交付。v2 原交付 SHA256 为 `020b4032a594a2a9dddee0e530e8b596456529540b4ed32937de49f9473e1b0d`；root 接受记录 SHA256 为 `980ace7cdbf1430b4b80de4da0493b2d6f65e050ec2cc0f76c538d82bafdc35b`。
