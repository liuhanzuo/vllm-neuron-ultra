# v2 mandatory sampler 接线审阅

v1 的整模型候选遗漏了原 r10 的 actual sampler replay 和 13 controls。v2 已补齐，HTTP 或质量结果不能替代这两个门槛。v1 的 delivery 及其19个文件全部保持原 hash；本轮仅源码/metadata/mock，没有读取 tensor/packet/NEFF payload，没有 native、编译、设备或锁操作。

## 原 r10 独立复核

`r10_sampler_review.json` 保存8个原科学来源与实际 r10 spec 的 SHA256 join，以及 actual-exit、qualified-selection、两臂 replay/capture receipts 和 shared controls 的 metadata 引用/hash。

| 原研究臂 | 实际 packet / rows | B1/B4 calls | fast rows | 最大 TV | 最大概率误差 | 实际 child exit |
|---|---|---|---|---|---|---|
| baseline | 32 / 80 | 16 / 16 | 77 | 6.705522537231445e-8 | 1.1920928955078125e-7 | 0 |
| candidate | 32 / 80 | 16 / 16 | 78 | 7.450580596923828e-8 | 1.1920928955078125e-7 | 0 |

shared controls 的13项及实际 exit0 已核对。这些是研究执行的范围证据；没有将其复制为 package 通过结果，package 实际 replay/control 仍待新运行。

## 最小源码变更

`sampler_replay.py` 完整复制原 `qwen35_sampler_native_replay_v1.py`。唯一科学入口差异是两处 import 从研究 `qwen35_sampler_bounded_operation_v1` 改为 `vllm_neuron.worker.cpu_sampler`。撤回这两处 import 后，整个文件 AST 与原文相同：Decoder、所有 metadata/tensor/stride/hash/coverage 检查、完整词表 support/TV/maxprob、自己的 token/output/RNG 复现、显式 generator fallback、13 controls、所有 finally RNG restoration 均保留。

`sampler_leaves.py` 完整保留原 `replay_captures` 和 `rejection_controls` 函数；只把 native replay import 接到上述 package port。保留每个实际 packet 的完整文件顺序、SHA、大小、调用ID、arm/rows join、完整 quota accounting；不抽样替代全包 replay。

`capture_writer.py` 的 nvme_destination、atomic_metadata、CaptureFileWriter AST 与原 worker 相同。实际 CaptureObserver、NativeSnapshotter、PacketBuilder、request_context、CaptureBudget 直接复用 pinned 原 `qwen35_sampler_native_capture_v1.py`。该文件使用已通过 r10 的 flat allocation singleton-stride 修正，不创建简化 codec。

package 的 `_shortlist` 和 `BoundedTopKTopPSampler` 两个 AST 与 r10 科学实现逐项一致，但这不等于新 package 运行已通过；replay 运行时仍检查实际 packet.operation_class。candidate 必须是 `vllm_neuron.worker.cpu_sampler.BoundedTopKTopPSampler`，reference 必须是 `vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler`。

不能直接复用旧 instance.attach：它禁止 `NEURON_CPU_SAMPLER_FAST`，并会再安装一个研究 wrapper。新 `capture_worker.attach` 验证 package runner 已安装的实例及其单一原 reference，再安装原 observer。它不赋值 topk_topp_sampler、不调用 maybe_enable、不创建新的 bounded operator。测量 server 不使用这个 worker。

## 必须执行的科学范围

每臂的原诊断完全保留：固定 prompt128，先B1后B4，共5个请求，各32输出，合计160输出；temperature=.7/top_p=.9/top_k=40；不补seed，不因quota不足重试。要求实际 B1/B4 各16次 top-k/top-p 边界调用。可保存真实出现的 B2/B3，各 row count 最多16 calls，最多64包/160 rows，每包≤12MiB、JSON≤1MiB、每臂总packet≤256MiB、两臂≤512MiB。每包包含整个248320词表的实际processed logits、全部有效sampling metadata、真实request/slot顺序及历史、全局和每请求generator的前后RNG、实际sampled IDs及processed输出。

每个实际 packet 都要完整 decode/replay：support严格相同；TV≤1e-6、最大概率差≤1e-6；本臂自己的 output/token/global RNG/request RNG 完全复现；有显式 generator 时，整个原/候选调用的 output/token/RNG 相同。无seed的两种算法不要求随机token序列相同。每臂必须有实际 fast rows；不能靠 control 代替实际 model logits。

原13 controls 共享运行一次，与 r10 一致：

1. partial_explicit_generators
2. all_explicit_generators
3. k_absent
4. k_zero
5. k_above64
6. non_float32
7. invalid_p
8. processed_logits
9. processed_logprobs
10. excessive_kth_ties
11. nonfinite_max_or_kth
12. cdf_near_boundary
13. cdf_boundary_tie

所有原seed、构造、dtype、k/p、whole-call或row fallback检查不变；每项须恰一次原reference调用，输出或原异常相同，全局及每请求RNG相同；最终恢复原global RNG。异常控制保留原异常，不把异常记为合法分布成功。

## 新 package 来源与阶段绑定

`capture_sources.json` 是 worker 实际认证的明确源码集，包含 package sampler/runner/worker/loader/qwen来源、原capture/replay/protocol来源、新sampler接线及原vLLM/torch相关Python来源。root 用最终 delivery 冻结此映射；capture identity 绑定完整新 case、该source集合、arm、worker FQN、operation/reference FQN 和当次新 package artifact manifest SHA。

root 为每个 capture server 提供：

```python
argv = sampler.capture_server_arguments(case, arm)
capture_env = sampler.capture_environment(case, arm, new_run / arm / 'capture', manifest_sha256)
```

它沿用 package `-m vllm_neuron.models.qwen35.serve`，只增加 capture worker FQN。env仅增加 `QWEN35_PACKAGE_CAPTURE_ROOT/RECEIPT/IDENTITY` 三个精确绑定；标准 package flags 和 `NEURON_CPU_SAMPLER_FAST=0/1` 由root既定clean environment提供。packets目录必须预先由root建立、为空且属于本次新capture。

API ready后，root调用 `check_capture_attachment`，把实际独立观察的worker PID/start/cgroup、API完整argv/env、实际worker attachment与上述identity匹配。它不自我认证观察来源；root仍负责原invocation/owner及终态审计。运行诊断后停server并取得实际exit/drain，才在独立子进程调用 `entry.run_sampler_stage('sampler_replay', case, new_run, guard, arm)`。shared controls另起子进程调用 `entry.run_sampler_stage('sampler_controls', case, new_run, guard)`。

只有两臂所有包replay和13 controls实际执行、exit/drain经root认证，才进入quality；`check_saved_sampler_gates`会检查新case/artifact、完整quota、每包vocab/TV/maxprob/own token+RNG、fast rows及全部13controls。所有quality和HTTP入口强制先调用这个检查；timed HTTP另外检查两臂完整quality pair。capture专有环境或capture worker不能出现在measured阶段。

完整顺序见 `stage_order.json` 和 `REVIEW.md`：new view → fresh12图build → 两臂全部state/numeric/cold/long/vectors203 → baseline capture/drain/replay → candidate capture/drain/replay → shared13controls → 两臂quality96 → quality pair → 无observer的匹配initial/fullHTTP → 原root独立终态与artifact资格审计。所有报告保留 full_model_qualified=False，只有外部最终审计可以作资格结论。

## source/mock 验证

推荐运行 `test_sampler_source.py`；它同时收集15项原基础检查和15项sampler检查，总30项。新增负例覆盖旧研究operation FQN、错误arm、nested wrapper、重复observer、缺B1/B4配额、删包/行/词表、TV/maxprob越界及NaN、own-RNG失败、fast rows缺失、缺/重排/改名control、reference多次调用、fallback范围错误、guard拒绝、缺sampler门槛阻止所有客户端，以及measured捕获环境污染。所有正例明确为mock或只读历史metadata；没有native执行。
