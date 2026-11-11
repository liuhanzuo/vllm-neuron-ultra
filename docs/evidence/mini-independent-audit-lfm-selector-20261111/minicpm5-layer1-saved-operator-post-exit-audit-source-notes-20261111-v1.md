# MiniCPM5 保存算子比较：退出后独立审计源码交付

本交付只有新 auditor、纯测试与本文。没有运行 auditor，没有重建真实 history，没有读取实验 capture/native 字节，没有检查进程、服务或锁，也没有修改 catalog、handoff、原始报告或任何冻结的 v1 源码。这里不是实际执行成功记录，也不是 root 对当前运行 spec 的批准。

入口是 `audit_minicpm5_layer1_saved_operators_v1.py`，测试是 `check_audit_minicpm5_layer1_saved_operators_v1.py`。它复用原 retained descriptor lease、ComparisonGuard、source/metadata reader、完整 scope 和 physical pair 验证，以及冻结的 comparison/reader v1。没有模型运行、子进程、新服务、新锁或另一套监控框架。

## 必须先有真正的比较退出

入口只接受 root 独立记录的 `minicpm5_saved_operator_comparison_actual_exit_v1`。终态恰好包含 14 项：`schema`、`reviewer`、`spec`、`review`、`live`、`result`、`owner`、`actual_tool`、`actual_tool_exit_observed`、`exact_owner_gone`、`lock_identity`、`self_exit_claimed`、`full_model_qualified`、`new_speedup_claim`。

`actual_tool` 必须为真正 tool session 的正整数 session ID、真实终态 chunk、严格整数 0；布尔值 False 不能冒充退出码 0。终态必须确认外部观察到退出和准确 owner 消失，不能宣称模型合格或提速。独立 auditor 重新检查 owner 已消失，并核对原 live 的 session、owner、argv、spec/review、实际 native executable。终态的 result 必须是原批准路径的完整 summary；六份 call 报告的存在不能代替整体成功退出。

实际 MiniCPM5 比较退出记录由 root 提供，源码没有硬编码某一 session、当前 catalog 或 handoff。LFM 后续任务已扩展当前队列；root 运行前须绑定届时完整 idle catalog，不能沿用比较开始时的旧 catalog 当作当前状态。

## 完整来源、退出与历史重建

`reconstruct` 独立重建原比较的全部先决证据，不调用会写结果的 `previous.compare`。它认证原 native model/recorder/native auditor 的真正终态，原模型 admission、intent、overlay、完整 source plan，原 SDK recovered completion 与原 source contract。原 SDK 模型 exit 1、失败 native auditor 的旧行、成功重试的独立行，以及所有后续 serial rows 都保留。

完整 source closure 经原 source classification、native receipt、loaded-source 检查和文本 SHA 验证，包含原 model 来源、已完成 native auditor/recorder 的继承来源、原比较增量及本次仅新增来源。`merged_plan` 禁止继承来源重叠、替换或遗漏。当前 catalog 新增来源若不在继承 map 内，必须全部加入本次精确 `source_extension`，不能只加入新入口。

Root 检查完整最新 catalog 后发现，已结束 LFM namespace 诊断的 567-source map 含四份尚未继承的 SDK 文本：原 `vllm_neuron_053_py312` 环境的 `torch/__init__.py`、`torch/_ops.py`、`torch/_classes.py`、`nki/framework/_torch_convert.py`。入口只对这四个**准确绝对路径和准确原 SHA 的组合**允许非 ROOT 增量，依据 r4 spec `d6b0f4548ea67a469d87afb92dae419f585c06abff8a9a8d4d77912faa0b5a94`，不允许任意 venv、相邻文件、不同 SHA 或 native extension。它们仍必须通过原分类、完整内容 SHA、完整 current catalog 和 source contract 验证；没有放宽其他来源规则。

原 native model 的 predecessor physical pair、完成 native audit 的 pair 和原比较的 pair，分别按各自当时的完整 owner/unit scope 解码验证。旧比较的 freshness 走原 `freshness_shape` 与 `validate_predecessor_chain`，完整读取它的历史 publication-only previous chain；它证明当时 idle，不冒充现在 latest。当前 audit 单独验证最新完整 idle handoff，在此次读取前后重做完整 physical pair，要求观察相同，最后复查 latest freshness、所有来源和 metadata identities。

原 native audit 的 936 payload / 28,490,724 B 官方 attention/cache 控制证据仍须完整认证，但本次不重复读取这些控制 payload。原 native inventory 在前后必须完全一致；SDK 的全部 60 个 endpoint control 描述与身份前后重查，其原字节相等关系继续依赖真正完成的 recovery audit。

Native trace 只在原 native Python 解释器重建。SDK trace 复用原成功 SDK recovery 独立审计确证的两 rank 原 JSON、12 份完全一致的原合同及其实际 loaded verifier 来源；没有在 native Python 下重新编译 SDK，没有删除任何 line-table 字段。

## 独立全量重读与精确结果比较

`reader.prepare` 从认证后的 native full-trace 及 SDK rank metadata 重建完整 descriptor plan。读取固定为原 **1,284 capture / 77,790,564 B**：native 720 / 14,897,124 B，SDK 564 / 62,893,440 B。六 call 固定为 4、5、6、11、12、13，保留 24 帧、两 rank、原物理槽位对应关系。

在打开六份报告前，先检查 summary 的 report refs 恰好六项且顺序、路径、字段和 SHA 形状完全符合原输出。各报告和 summary 用原稳定 SHA reader 读取，身份不变，完整文件均不得超过原 2 MiB 限制。

`reader.compare_all` 在本进程再次读取全部允许的 capture，冻结的 comparator 逐域计算全部统计和比较。每份原报告的完整 envelope、完整 descriptor、完整科学比较对象及其 canonical digest 都必须与此次重算相等。最终要求实际读取集合恰好等于全部 1,284 个允许文件，每个文件恰好一次。没有抽样、缓存原统计数或只核对“首个不同边界”。

完整 summary 的所有字段随后独立构造并比较，包括来源、controls、inventory、SDK trace 证明、完整 descriptor 公共 header、六 refs/digests、所有科学统计、原 SDK exit 1、各项 scope 声明。字典相等和 canonical JSON digest 同时检查；因此 True 不能冒充数字 1，-0.0 不能冒充 +0.0，未知字段或遗漏字段都拒绝。

历史 storage 的可用空间数值无法在事后重新测量。入口保留认证后的原值，只验证其严格整数类型、原 root 28 GiB + 16 MiB reserve / NVMe 30 GiB floors、严格整数 1 秒检查间隔，以及与此次检查完全相等的整个 native-tree usage（含 JSON 类型）。结果明确写明 `historical_storage_telemetry_bound_validated_not_remeasured: true`。本次实时 storage 由原 guard 另行检查并保存在新报告。

## 保持原资源约束和证明边界

资源不变：2 GiB address space、600/605 秒 CPU、900 秒 wall、CPU 6、nice 19、每文件 2 MiB。仍使用原 native interpreter 和 `-B`；原存储 floors 和 16 MiB 完整输出 reserve 保持。唯一新报告 exclusive/no-overwrite，写出后完整 SHA readback；不截断，不修改任何原结果。

新报告可证明原比较真正退出、全部 capture 此次重读、六报告与完整 summary 精确重建。它仍保留自身 `actual_auditor_tool_exit_verified: false`、`self_exit_claimed: false`，必须由 root 之后观察真实 tool terminal。模型未重跑，`causal_operator_proven`、`full_model_qualified`、`new_speedup_claim` 全为 false。“首个不同保存边界”仍不是算子因果证明。原 BF16/模型、cold4096、长解码、slot reuse/inactive、quality96、matched HTTP 门槛均不变。

## 纯测试与 root 后续接线

13 项测试通过，准确最终工具 exit/chunk 记录于配套 source-delivery JSON。测试从 AST 隔离提取纯函数，不导入 auditor runtime；生命周期调用全部显式 mock。范围覆盖准确终态/spec、六原报告路径、字段/类型/指标/descriptor/digest 的篡改拒绝、完整 summary schema、失败 SDK 和自身退出边界、原完整历史 publication-chain 验证函数、storage floors、执行 gate 顺序和原限制/锁释放 mock、三个冻结源 SHA，以及四份准确 LFM 文本 pins 的接受与错误 SHA/相邻路径/native extension 的拒绝。

重算测试使用 **12 个内存 6 B BF16 数组，共 72 B/次合成 replay**，通过真实冻结的 `compare_words` 验证 signed-zero 位差。它不是 1,284-file reader 的实际执行，不是原模型/张量运行、真实性能或 RSS 测量。另有静态检查确认原完整生产 summary 字段与新重建器完全一致。

Root 可独立复跑：

```text
python3 -B /home/ubuntu/experiments/trn1-latest-models-20261031/research/check_audit_minicpm5_layer1_saved_operators_v1.py
```

未来真实 spec 恰好是 `schema`、`comparison_exit`、`current_serial_catalog`、`source_extension`、`output`、`limits` 六项，schema 为 `minicpm5_saved_operator_post_exit_audit_spec_v1`。真实 source review 恰好是 `schema`、`reviewer`、`passed`、`blockers`、`spec`、`source_extension`，schema 为 `minicpm5_saved_operator_post_exit_audit_source_review_v1`，reviewer 为 `/root`，必须精确绑定 spec 和全部新增来源。

未来输出名限定为 research 下 `minicpm5-layer1-saved-operator-comparison-<name>-post-exit-audit-<name>.json`。CLI 接收 `--spec`、`--spec-sha256`、`--review`、`--review-sha256`。Root 先独立完成 source review，绑定届时的真正 idle catalog，再在原 native interpreter 中串行运行；本交付没有创建真实运行 spec、review、catalog、handoff 或 terminal。
