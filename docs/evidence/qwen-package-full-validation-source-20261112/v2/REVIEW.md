# 独立包整模型验证完整接线 v2

本目录是可审阅的源码候选。没有运行 native、模型、编译、prepare_view、HTTP 或锁操作，也没有修改 package、原研究文件或已绑定的 slot 来源。源码检查通过不等于新编译通过、模型通过或速度结论。

v2 补齐 v1 遗漏的 mandatory actual sampler capture/replay 与 13 controls，v1 全部文件保持冻结。`SAMPLER_REVIEW.md` 说明原 r10 实际证据、完整科学 AST 接线和所有新负例。`stage_order.json` 是原 root 必须执行的有限阶段顺序；它不创建调度器或锁。sampler 两臂 replay 与 13 controls 缺失时，quality 和每个 HTTP 入口会在 client/native 导入前拒绝；timed HTTP 另外要求两臂 quality pair 已通过。

## 具体差异

`scientific_port.patch` 展示四个科学入口和 metadata checker 的全部差异；`port_edits.json` 列举四个入口逐项文本替换。测试重建完整文件并比较 AST，保留完整科学循环和数值门槛。新增文件只位于本目录。

| 原入口 | 新文件 | 必要接线 |
|---|---|---|
| validate_real_logits.neuron | numeric.py | 去除其他模型及研究注册选择；经 runtime.load_admitted 加载 package；args.adapter 仍固定 qwen35_text，以保留原 fixture 顺序 |
| validate_qwen35_long_decode.neuron | long.py | 显式新 case/output；package 准入加载；原 HF oracle 不变 |
| diagnose_qwen35_state_reset_cold_v1.main | cold.py | 新 case/output；package clear_states；新图实际 input_shape_map；保留原 84 帧及 72 帧新鲜逐字节比较 |
| probe_qwen35_state_copy_v1.main | state_copy.py | 外层已有 root guard 取代旧 service 名；布局来自新 build 实际 96 状态记录；独立进程内仍只分配 copy 探针状态，graph_models_loaded=0 |
| sampler full checks | checks.py | 模型/质量新报告显式绑定新 case；只有 historical 对照使用原 case；所有原数值/质量门槛完整保留 |

原函数 `metrics`、`passed`、`graph_routes`、`raw_zero_states`、`verify_cold_reproduction` 的 AST 完全相同。没有 sys.modules 研究别名、运行时 AST 重写、研究 graph_reuse、orchestrate.validate_cache/run_case 或旧 serving shim。`checks.original_validators` 保留 r10 原来的纯 stdlib、已钉住源码的 validator 提取，仅作科学检查，不准入研究图。

`runtime.load_admitted` 直接调用现有 package `build.build`，但强制 `NEURON_QWEN35_ALLOW_COMPILE=0`、已存在的明确 artifact 路径。该入口的 existing 分支重新构造 build_identity，再执行 package admit_existing 和 load。没有绕过 manifest、配置、官方模型字节、SDK、硬件或 package 源身份检查。未发现需要修正 package 的代码。

## 12 图与配置

`case.example.json` 给出明确的新 view、artifact、cache、compiler workdir；这四个目录尚未由本次工作创建。`neuron_config.json` 是 build 和 serve 共同使用的完整 kwargs。

- 原 BF16 权重，TP2，decode B4，context B1，context/seq_len/max_length 4096，连续 batching，CPU sampling。
- context buckets 显式为 128/256/512/1024/2048/4096；token_generation_buckets 保留原 SDK 的 None，SDK 派生同样六桶。`graph_routes` 必须从加载的新模型实际证明 context 6 + decode 6，不能从配置数量推定成功。
- pa_num_blocks=4，pa_block_size=4096 与 package 平台禁用 prefix cache 后的实际 loader 默认一致。源码测试执行原 loader 的默认配置与 override 函数，并确认追加的三个 quantization 字段等于 pinned SDK 默认值。完整 metadata 中只允许 SDK 原有的 +Infinity 量化默认值，遵循 package 自身 canonical_configuration。
- build_fresh 要求 artifact/cache/compiler workdir 三者原先不存在，并复用 package serial_compiler。完成后记录实际 12 个 `compiler_workdir/{context_encoding_model,token_generation_model}/_tp0_bk{0..5}/graph.neff` 的大小/SHA256、实际路由、96 个原地清零后的状态布局、artifact manifest SHA256。
- NEFF 文件与新 cache 是工作产物证据；root 仍需保留同一次真实编译的日志、外部实际退出及完整 drain 证据。build inventory 不自行认证 OS 进程，也不证明任何数值或性能门槛。

## 原范围完整保留

| 阶段 | 必须完整完成 |
|---|---|
| state_copy | 两组模式，96 状态，192 fill bitwise + 192 raw-zero，比对前后 tensor binding 相同，0 graph models |
| numeric | 32 个 248320-vocab 帧；重复 prompts、混合长度、permuted slots、3 teacher-forced decode；cosine≥.995、KL≤.02、maxabs≤2、margin>.5 必须 argmax 匹配、总匹配率≥.9 |
| cold | 7 prefixes=32/128/256/512/1024/2048/2052；84 帧；672 raw-zero；42 duplicate pairs；7 原地 clear；96 binding 不变；新图全部 12 routes；前 72 帧本次实际复现原 partial tensors |
| long | 1541 teacher-forced steps，完整 87 帧；四槽初始长度 23/127/511/1、slots 3/1/0/2；所有原边界；slot reuse、B3 inactive -1、inactive resume；原 full-causal HF oracle 与原 incremental crosscheck 同时通过 |
| vectors203 | numeric32+cold84+long87 全部新保存 float32 张量逐字节对照合格研究基线；不能由源码一致推定结果 |
| correctness capture | 每臂原 5 requests/160 output tokens，B1/B4 两波，prompt128、output32、temperature.7/top_p.9/top_k40，不补 seed、不重试；真实 processed logits、完整 metadata/request-slot context、所有 RNG 前后及实际输出；各 B1/B4 必须捕获16 calls |
| sampler replay | 两臂每一实际 packet 全部重放；248320-vocab support 相同，TV≤1e-6、maxprob≤1e-6；各臂自己的 token/output/global+request RNG 精确复现；显式 generators 整次调用 token/RNG fallback；必须存在 fast rows |
| sampler controls | 与 r10 一样共享执行一次完整13 named controls；package Bounded 类及原 TopKTopP reference，逐项输出/异常/RNG相同，fallback恰一次原调用，global RNG finally恢复 |
| smoke | 原 serial4、concurrent4、identical-prompt4 |
| quality96 | 原 suite、c4、temperature0、无截断、所有已保存 text 重新 grade；两 package arm 都须 69/96，全部 96 文本与历史一致，零 lost cases |
| initial HTTP | 原 120 output、3 rounds、single+c4+streaming c4、8-token warmup；全部 greedy text/usage/prompt 相同 |
| full HTTP | 原 5 workloads×3 rounds，124 requests/30848 output tokens/57 greedy pairs；16-request mixed rounds/c4、streaming、4×8-token warmup、原不传 seed、240s timeout、min_tokens/ignore_eos |

当前实现按 r10 的 baseline/candidate 配对：同一份新 package 图，`NEURON_CPU_SAMPLER_FAST=0/1`，两臂均须重新走模型和 HTTP 门槛。它不会把新 package 与旧研究性能历史直接组成匹配速度比。若下一步要比较研究与 package 的性能，必须另行取得两份实际匹配的新运行；不能把旧 benchmark 时间当作匹配臂。

## root 的具体调用顺序

所有重操作都由原串行 root 放入独立子进程/既定 cgroup，并传入现有 `execution_guard`。本目录不提供假 guard、不提供新 History/锁框架，也不降低任何内存/NVMe/时间门槛。执行前 root 将本交付及实际输入绑定到原完整 History，并在实际编译前重新验证资源预算；本源码不作资源准入判断。

解释器固定 `/home/ubuntu/venvs/vllm_neuron/bin/python`，PYTHONPATH 顺序为 package checkout、research、实验 ROOT。先完成现有已授权工作及物理 drain，再为本整模型工作单独做原 admission。

```python
# 以下由真正的既有 root 执行适配器调用；不是可自我授权的 Python 脚本。
from qwen35_package_full_validation_v2 import entry, http, runtime, sampler
case = runtime.read('/.../qwen35_package_full_validation_v2/case.example.json')

# 一个 admitted 子进程：只建立并验证新 view，官方 source 固定到原模型目录。
entry.prepare_view(case, execution_guard)

# 另一个 admitted 子进程：此过程才允许编译。
# NEURON_QWEN35_ALLOW_COMPILE=1，其余设置见下文。
entry.build_fresh(case, new_run / 'build-inventory.json', execution_guard)
# root 取得本次真实 exit=0 和 drain 后才能继续。

for arm in ('baseline', 'candidate'):
    # root 分别在新 run/{arm}/model 中写入完全相同的新 case.json。
    # 每次调用在新的 admitted 子进程，ALLOW_COMPILE=0；上一进程先 exit/drain。
    for phase in ('state_copy', 'numeric', 'cold', 'long'):
        entry.run_model_stage(phase, new_run / arm / 'model',
                              new_run / 'build-inventory.json', execution_guard)
    entry.compare_vectors(new_run / arm / 'model', execution_guard)

for arm in ('baseline', 'candidate'):
    # root 启动 fresh package capture server，使用 capture_server_arguments 和
    # capture_environment 的精确绑定。这里只观察 package 已安装的 sampler。
    # root 等待 API ready，并以真实 worker PID/start/cgroup/argv/env 做 attachment 审计：
    sampler.check_capture_attachment(actual_attachment, root_live_observation, case, arm,
                                     new_run / arm / 'capture', new_manifest_sha256)
    sampler.run_capture_diagnostic(case, new_run / arm / 'diagnostic-http.json', execution_guard)
    # root 停止 capture server、取得实际 exit/drain 后，另起 admitted 子进程：
    entry.run_sampler_stage('sampler_replay', case, new_run, execution_guard, arm)

# 与 r10 一样，共享 controls 只执行一次，但必须完整13项；另一个 admitted 子进程。
entry.run_sampler_stage('sampler_controls', case, new_run, execution_guard)
sampler.check_saved_sampler_gates(case, new_run)

for arm in ('baseline', 'candidate'):
    # root 在 fresh quality server 上调用；质量 server 退出/drain 后再启动 HTTP server。
    http.run_http_stage('smoke', case, new_run / arm / 'quality/smoke.json', execution_guard)
    http.run_http_stage('quality96', case, new_run / arm / 'quality/quality96.json', execution_guard)

# 两臂完整数值和 quality pair 门槛先通过，之后才能进入 HTTP timing。
http.check_saved_quality(case, new_run / 'baseline', new_run / 'candidate')
for arm in ('baseline', 'candidate'):
    http.run_http_stage('initial120', case, new_run / arm / 'http/initial120.json', execution_guard)
    http.run_http_stage('workloads', case, new_run / arm / 'http/workloads.json', execution_guard)

# 仅纯 metadata gate；root 另行认证每个 producer、实际 exit/drain、最终 artifact 不变。
http.compare_saved_pair(case, new_run / 'baseline', new_run / 'candidate',
    root_observed_live_hosts, {'baseline': str(new_run / 'baseline/http'),
                               'candidate': str(new_run / 'candidate/http')})
```

根进程给每个 server 采用 `runtime.server_arguments(case)`，其具体执行入口是：

```text
/home/ubuntu/venvs/vllm_neuron/bin/python -m vllm_neuron.models.qwen35.serve
  --model <new view> --served-model-name qwen3.5-2b-text
  --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 4
  --no-enable-prefix-caching --host 127.0.0.1 --port 8010
  --additional-config <override_neuron_config = neuron_config.json 完整内容>
```

上面是 quality/measured server 的参数。capture server 用 `sampler.capture_server_arguments(case, arm)`，仅额外添加稳定 `--worker-cls qwen35_package_full_validation_v2.capture_worker.{BaselineCaptureWorker,CandidateCaptureWorker}`。capture 专有的三个 `QWEN35_PACKAGE_CAPTURE_*` 环境键必须在新 quality/HTTP server 中全部消失，measured checker 会拒绝任何此类键或 capture worker argv。capture 抓包/写盘/replay 时间绝不进入性能比值。

各 native stage 设置 `NEURON_QWEN35=1`、`NEURON_ON_DEVICE_SAMPLING_DISABLED=1`、`XLA_DISABLE_FUNCTIONALIZATION=1`、显式 `NEURON_COMPILED_ARTIFACTS`、`NEURON_COMPILE_CACHE_URL`、`BASE_COMPILE_WORK_DIR`。只有 build 子进程 ALLOW_COMPILE=1，其余必须为 0；测量 arms 的 `NEURON_CPU_SAMPLER_FAST` 分别为 0/1。保留 package 要求的 spawn；不得携带旧 TRN1_QWEN35* 或 TRN1_CPU_SAMPLER_FAST。所有 compile identity 环境必须与实际新 manifest 一致。

root 应使用现有干净 host 环境/线程/DEBUG 配置，不在此凭空选新值。HTTP 配对检查比较完整实际 API environment 和完整 argv；只允许 sampler arm 以及已精确绑定的 `{arm}/http/tmp`、`{arm}/http/rpc` 差异，拒绝研究 capture hooks。root 对这些 live observations 的 owner/PID/start_ticks/invocation/cgroup/worker module 做原独立认证，并在退出后做原终态审计。纯 metadata checker 不自我声明观察来源真实。

原 HF reference 和 baseline tensor 路径是对照输入，必须纳入 root 当次输入 hash/原 provenance 检查；它们不是新 package 图的 provenance。所需 payload 是 `qwen35-2b-hf-reference.pt`、`qwen35-baseline/long-decode-hf-reference.pt`、`qwen35-split-corrected-full-v4/cold-buckets.pt` 及 `qwen35-state-reset-full-v1/{real-logits-validation.json.pt,cold-buckets.pt,long-decode-validation.pt}`。本次未读取这些 payload。

## 审阅和验证

`input_sources.json` 钉住实际依赖的原科学/HTTP/package 来源；`source_delivery.json` 钉住本目录候选。运行 stdlib/source-only 检查：

```text
/home/ubuntu/venvs/vllm_neuron/bin/python -B /home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_package_full_validation_v2/test_sampler_source.py
```

测试包含原15项整模型 source/metadata 及新增15项 sampler source/mock：v1 全文件冻结、replay/13controls完整AST、package数学AST、既有 sampler 实例观察、错误/重复包装拒绝、稳定spawn FQN、全部包/分布/RNG/controls负例、缺 sampler gate 阻止所有客户端、原5请求160tokens和capture配额、measured污染拒绝。正例使用明确标识的 metadata/mock，保留 fresh_execution_proven=False/full_model_qualified=False；没有把这些测试作为新实测。

交付只完成本阶段具体源码准备。实际编译、完整数值/冷/长/槽位/quality/匹配 HTTP、真实编译进程审计和最终独立资格审计均待原 root 正式 admission 后执行。独立包不能继承研究图或 r10 性能结论；任何失败保留现场，不删除修复或自动重试。
