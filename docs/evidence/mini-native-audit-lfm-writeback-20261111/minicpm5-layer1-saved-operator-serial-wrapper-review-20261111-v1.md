# MiniCPM5 保存算子比较：实际入口源码审查说明

交付 `compare_minicpm5_layer1_saved_operators_v1.py` 及纯源码测试。没有运行该入口，没有重建真实完整 history，没有读取实验 raw/native 文件、获取锁、检查服务或修改当前 catalog。已审核 reader/comparison v1 的 SHA 保持不变。本文件不是执行许可；必须等 root 所选 native audit 真成功退出并独立登记后，才能创建新的真实 spec/review。

## 原接口的实际复用

入口复用 `audit_minicpm5_layer1_operator_native_result_v4` 的原 descriptor lease、`AuditGuard`、输出 inventory、完整物理 pair、源分类与 `OwnedService.verify_drained`。原锁由 `outer_support_v3.acquire/release_after_physical_drain` 持有与释放，没有子进程/新服务/另一套锁或监控协议。

模型和 recorder 的完整外部终态继续由 `post_exit_core_v5.validate` 认证。SDK 使用既有 `recovered_completion_v2.completed`，完整保留原 **44,503 sources / 8,791 inputs** 及 SDK 模型 exit 1。完整 native plan 仍经原 overlay resolver 在内存重建；用 `prepare_v7.inherited_source_plan` 再验证实际 spec/review/runner 与完整 digest。由同一次认证重建出的 cache 避免重复解析相同历史；不以精简 map 替代原 plan。

所选成功 auditor 的真实终态、准确命令、spec/review、recorder v3 的完整来源绑定和 auditor v4 的完整继承来源证明由 `prepare_v7.completed_inherited_extension` 认证。审计控制 proof 必须覆盖 native **24 帧 / 72 次调用 / 936 payload / 72 result / 72 journal / 2,736 journal records** 及全部官方 attention/cache 相等控制。

比较源仅通过与继承来源不重叠的 `source_extension` 加入。原 source map、所有非入口 package pins、原 source contract 和九个 native dependency 身份均保留；真正使用的本地模块也仍全部校验 SHA。未来 spec 不得复制、替换或遗漏继承 pins。

## 两个解释器的合同认证

Native capture 使用 `model_reference_2026` 解释器；root 的元数据检查发现 host Python 3.10 重建的 line-table hashes 与原生 Python 3.12 不同。入口因此检查当前解释器与原 `life.PYTHON` 解析到相同目标，并只在该解释器下重建 native trace contract。没有删除 `co_linetable_sha256`、`co_lnotab_sha256` 或更改原 source/code 校验。

SDK 的合同不能在 native 编译器里重新生成。新增纯函数 `authenticated_sdk_trace` 只在以下条件全部满足时使用原 SDK 保存的合同：

1. 既有完整 recovered completion 已验证真实 recovery 与 recorder 终态，原模型 exit 1 保留；完整 raw pair proof、所有 endpoint/journal/RoPE 审计保证必须成立。
2. 原 recovery result 的 `original_model_outputs` 精确绑定两份 rank-result reference；它们也必须在 recovered 的认证 metadata 集合中，并经同一稳定 SHA reader 再次读取相等。调用者替换 capture 字段但继续沿用原 ref 会被拒绝。
3. 原完成审计的 **实际 loaded source closure** 必须包含 `operator_sdk_results_v2.py`；该源码 SHA 与恢复后的完整 source map 相同，并重新验证源码内容。该旧 verifier 确实在其原解释器下通过 `expected_trace_contract`、`validate_sdk_capture_summary` 与完整 snapshots 检查，实际 recovery 代码通过 `validate_rank_pair` 执行它。
4. 原 attention source path/SHA 必须等于固定 SDK trace provenance，并再次校验完整源文件。两 rank × 六 call 的 **12 份**原合同、其完整 trace 与 summary 全相同；保存的 route/function/source、first line、两个 line-table hashes、QK0/QK1 local/assignment/event 字段全部保留。原 scale provenance 继续由原 `validate_sdk_capture_summary` 校验。

输出保留原合同全对象、两 rank 原文件 refs、12 个 call 绑定、真实 recovery completion/result ref、完整 raw proof digest 和实际 verifier source ref。这里的依据是此前真实独立审计对准确 SHA 绑定的原 JSON 做过完整认证，**不是保存记录自我认证**。缺少这条证据链就拒绝；没有使用不同解释器重新编译 SDK，也没有声称存在原合同未保存的独立 bytecode hash。

## 失败审计与完整当前历史

此前失败的 auditor row 可以保留真实 exit 1 和明确失败 terminal schema，且不需要补造成功 result。原 `serial_catalog` 本来就允许非零真实退出；新入口只对**本次选择的成功 auditor**调用成功消费者。所有旧 catalog rows、source pins、owners 和 units 都按完整相等关系保留，失败行不被替换为重试成功行。

测试实际执行原 `serial_catalog` 的纯元数据函数：失败 schema 为 `minicpm5_native_independent_audit_failed_actual_exit_v1`，没有 result 字段；它可与独立的成功重试行共存。尝试丢弃失败行或把其 exit 1 改成 0 会拒绝。只要失败行 source hashes 已被既有成功 recorder 等真实来源见证，不需要为失败任务伪造成功证明。

完整 current catalog 还必须包含所选模型、recorder、auditor 的真实 ref/tool/owner，保留 audit catalog 及 model predecessor 的所有行。原最新 idle handoff/freshness 在开始和结束时保持一致；lease 继续监测原所有 owner 消失和没有新重任务。原模型 predecessor pair、旧 audit pair 都按其当时完整 scope 解码验证，新的完整 scope 在比较前后各做两次物理观察，要求一致。

## 读取与输出范围

真正读取的 capture 固定为 **1,284 文件 / 77,790,564 B**，六 call 依次执行，每 call 214 leaves。每个文件由已审核 reader 完整验证身份、路径、长度、EOF、SHA；全域比较仍用已审核 comparison v1。

Native 原 936 payload 的独立字节控制审计不重跑；原完整 output inventory 在前后复查相同。SDK 的 60 个 endpoint control 做完整 descriptor/路径/身份前后复查，不重新读其 payload；其完整逐字节控制关系依赖已认证的真实 recovery 审计。

六份 call 输出各含本 call 的完整 descriptor（含 source geometry）和原完整比较报告；summary 含 descriptor 公共 header、整个 plan canonical SHA、六份报告 refs/digests、完整来源/终态/物理绑定和完整科学汇总。实际 native plan ref 指向**生成它的原 overlay**，driver ref 指向**嵌入 driver 的原 native-result wrapper**，均明确标注 kind，并分别保留重建对象 canonical digest。没有伪造两个不存在的文件。

原限制保持：**2 GiB address space、600/605 秒 CPU、900 秒 wall、CPU 6、nice 19、每文件 2 MiB**。原 root 28 GiB / NVMe 30 GiB floors 继续检查；root 额外保留 16 MiB 完整输出空间。所有输出 exclusive/no-overwrite、完整 JSON readback SHA、超限失败且不截断。若后续 call 失败，之前的 call 报告仍是部分证据，不能当成 complete summary。

进程不宣称自身已经退出，保留 `actual_comparison_tool_exit_verified: false`、`external_comparison_audit_pending: true`、`self_exit_claimed: false`。原模型、BF16、cold4096、长解码、slot reuse/inactive、quality96、matched HTTP 门槛及“首个不同边界不是因果证明”均不变。

## Root 审查与后续接线

建议独立复跑唯一源码测试：

```text
python3 -B /home/ubuntu/experiments/trn1-latest-models-20261031/research/check_compare_minicpm5_layer1_saved_operators_v1.py
```

重点检查 `compare` 的门槛顺序、`authenticated_sdk_trace` 的真实审计证据绑定，以及所有实际操作只会由未来显式调用的 `run` 进入。对两个旧 v1 helper 的 SHA 检查也是测试的一部分。

实际 spec 只包含 schema、真实 model/recorder/auditor exit refs、届时完整 current catalog ref、仅新增 source map、fresh output prefix 和原 LIMITS；source review 必须准确绑定该 spec 和新增来源。本交付没有生成任何当前 spec/review/catalog/terminal。未来 CLI 接受 `--spec`、`--spec-sha256`、`--review`、`--review-sha256`；应在原 native 解释器下执行。若通过 PATH 选择该解释器，仍由 root 观察实际 argv/executable，不以 PATH 文本冒充进程证据。

11 项纯测试通过。测试只执行 AST 提取的纯函数、显式 mock 的 runtime 调用与合成元数据；没有导入该 runtime 入口、读取真实张量或创建输出文件。六份完整元数据 envelope 的最大序列化大小 **1,343,675 B**（每比较域 8 个例子，包含完整 descriptor），公共 header **280,892 B**。真实 summary 的完整物理/来源元数据和实际报告仍须未来运行时再次受 2 MiB 限额检查。没有构造完整 raw fixture，没有测量真实运行时 RSS、时长或存储性能。
