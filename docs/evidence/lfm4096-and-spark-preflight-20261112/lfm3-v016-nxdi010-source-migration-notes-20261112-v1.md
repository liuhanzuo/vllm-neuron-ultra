LFM3 在 vLLM 0.16 / vLLM-Neuron 0.5.3 / NxDI 0.10 上的最小迁移方案

2026-11-12。结论：旧 LFM3 的数学实现、TP 分片和扁平状态接口可以作为迁移起点；需要补齐的是注册、独立版本身份、NKI 0.5 writer、采样接线以及明确的请求槽位契约。本次仅完成源码分析，没有实施安装、导入原生 SDK、编译或运行模型，也没有产生新的提速倍数。16 个所依赖的底层接口中，14 个完整函数 AST 一致、15 个签名一致，不能把升级概括为“所有底层 API 都不兼容”。

主证据是同目录 `lfm3-v016-nxdi010-source-migration-evidence-20261112-v2.json`，SHA256 `9866da97aeb71ab309cd6067231aa8156103b289dd79ba667eabfeec025f8416`。其中 `source_sha256` 固定 43 个文本文件，`evidence` 收录 68 段源码及路径、行号和 SHA；下文方括号引用其证据键。API 比较位于 `api_comparison`，35 状态列表位于 `state_ledger`。交付清单另外固定 SDK 文字清单、补充源码和本说明。

旧适配器根目录为 `/home/ubuntu/experiments/trn1-latest-models-20261031/lfm3-package-native-core-v1/work/target/vllm_neuron`。新插件实际 editable 源码是 `research/vllm-neuron-053/vllm_neuron`，vLLM 实际 editable 源码是 `research/vllm016-source/vllm`，新 Python 环境为 `/home/ubuntu/venvs/vllm_neuron_053_py312`。distribution 显示 `vllm==0.16.0+empty`，但 finder 确实指向有版本 0.16.0 源码的目录；不能把 `+empty` 当作没有可用源码或据此重装。[`__editable___vllm_0_16_0_empty_finder`、`__editable___vllm_neuron_0_5_3_finder`、source_sha256 中的 `_version.py`]

| 项目 | 旧包固定范围 | 新环境/目标 | 迁移含义 |
|---|---|---|---|
| Python / vLLM | Python 3.10 / 0.11.0 | Python 3.12 / 0.16.0 源码 | 新的源代码和软件身份；旧验收不自动继承 |
| 插件 | 旧统一优化包 | vLLM-Neuron 0.5.3 | 新目录尚无 `models/lfm3`；pyproject 仅有 platform plugin |
| Torch / Torch-NeuronX | 2.8.0 / 2.8.0.2.12.22436+0f1dac25 | 2.9.1+cpu / 2.9.0.2.15.32035+de43f57c | 真实 native ABI 和运行仍待串行验证 |
| NxDI / NxD | 0.8.16251+f3ca5575 / 0.17.26814+4b18de63 | 0.10.18399+ed62453e / 0.19.28492+435aae2b | 保留已确认稳定的适配接口 |
| NKI / compiler | `neuronxcc.nki` / 2.23.6484.0+3b612583 | `nki` 0.5.0+28631259367.ga768afa6 / 2.26.6360.0+6f180f47 | 改接独立 typed writer，使用新的编译产物目录 |
| 模型 / tokenizer | LiquidAI/LFM3-1.7B，revision `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7` | 保持原权重、tokenizer 与视图身份 | 不因迁移替换模型或放宽模型字段核对 |
| 范围 | Trn1、BF16、TP2、B4、prefill B1、2048 | B4/B8/B16 和 4096 为分别待验的目标 | 不是给旧 scope 改一个数就获得资格 |

版本来自旧 `identity.py` 以及已固定的 `lfm3-nki05-device-sdk-text-manifest-20261112-v1.json`；该 SDK 清单只证明读取到的文本身份。本次没有重新散列 SDK 二进制载荷。

API 比较覆盖 `InferenceConfig.__init__/get_text_config/get_neuron_config_cls`、`NeuronBaseModel.__init__`、`NeuronBaseForCausalLM.forward/_is_prefill`、`NeuronApplicationBase.__init__/get_neuron_config_cls/load/get_state_dict` 和四种 NxD 并行层构造函数，以上 14 个函数 AST 一致。`NeuronConfig.__init__` 内部不同但仍为 `self, **kwargs`；`NeuronApplicationBase.compile` 末尾增加可选 `disable_fail_fast=False`，旧调用形式仍可绑定。其他调用链、Torch ABI、trace lowering、编译及服务行为不在“14 个相同”的证明范围内。

旧 `Lfm3CausalLMBase` 对位置 0 的单 token prompt 保留 padding 路径；旧注释称 NxDI 会按长度路由，但两个版本 `_is_prefill` 实际 AST 一致，检查的是 `position_ids.min().item() == 0`。保留已经受测的 padding 行为，不能以这条旧注释声称 NxDI 0.10 改了路由，更不能在迁移里顺手删除它。[`old_one_token_prefill`、api_comparison]

建议未来在隔离分支中实施下列变更；本报告没有改这些文件。

| 文件/接线 | 最小具体变更 | 源码依据与边界 |
|---|---|---|
| `pyproject.toml` 和 `models/lfm3/__init__.py` | 加入 opt-in 的 general plugin；保留关闭时不导入模型/原生依赖的行为 | `new_platform_entrypoint`、`old_optional_plugin` |
| `models/lfm3/registration.py` | 注册 Transformers 的 `lfm3` config、vLLM 的 `Lfm3ForCausalLM`、NxDI `MODEL_TYPES['lfm3']['causal-lm']`；保留已有注册冲突即拒绝、一个进程不切换 writer | `old_registration`、`new_nxdi_model_types`、`new_arch_resolution`。仅加 MODEL_TYPES 不够 |
| `modeling.py`、`text_core.py`、`helpers.py` | 保留原模型计算、TP2 权重分片、tied head、FP32 RoPE 和扁平状态输出；接新 scope 校验 | `old_nxdi_model`、`old_nxdi_app`、`old_attention`、`old_conv`、`old_partition_helper`、API 比较 |
| 新版 `identity.py`、`provenance.py` | 使用新的独立 schema/版本、精确新软件与真实 editable 源码散列、writer 标签、capacity、context、完整生效配置；保留原官方 model/tokenizer 身份 | `old_identity_model_scope`、`old_software_identity`、`old_source_identity`、`old_variant`。旧 v1 的固定身份和旧编译目录保持有效且不可冒用 |
| `cache_writer.py` 和新的 typed NKI 模块接线 | 使用已审查的 typed factory、wrapper、dispatch；容量明确为 4/8/16，int64 slot 先归一化再转 int32，one-row 走 original 路径 | `typed_normalize`、`typed_production_dispatch`、`new_nki_jit`；真实设备证明仍未完成 |
| loader 的 `_load_weights_common` | 在通用缓存路径删除、hash 与异常自动重编译之前进入 LFM 专属 `load_or_compile`；完整不匹配直接拒绝，仅 fresh 路径显式允许编译 | `old_artifact_admission`、`old_artifact_config`、`new_loader_generic_fallback`、`new_loader_generic_deletion` |
| `worker/cpu_sampler.py` 和新 runner 构造函数 | 移植原有局部 wrapper，在创建 `self.cpu_sampler` 后调用一次 opt-in hook；保留新版 `execute_model`/`sample_tokens` 两阶段协议 | `old_sampler_wrap`、`old_sampler_enable`、`new_runner_sampler_instance`、`new_deferred_sampling`、`new_execute` |
| LFM 专用 serving scope 校验和 runner/worker 校验 | 明确单组 contiguous 请求槽位代理、B+1 含 null block 的预算、实际 B 个 Neuron slots、完整状态独立账本；拒绝未实现的抢占重建和其他执行模式 | `new_proxy_cache_specs`、`new_proxy_cache_memory`、`new_request_slot_allocate`、`new_request_slot_cached`、`new_request_update` |
| 新的 4096 supplement 与 full-model 检查 | 追加 LFM 2 KV heads/rank 的 4096 writer/alias 检查及跨 2048 边界长解码；原矩阵和门槛全部保留 | `current_device_matrix`、`current_alias_matrix`、`preserved_full_gates` |

新 loader 对 `Lfm3ForCausalLM` 的名字拆分已经能得到 `lfm3` / `causal-lm`，因此不需要为解析函数另造特殊字符串规则。它在普通文本模型分支实际使用通用 `NeuronCausalLM` 包装器；registry 类主要用于模型识别和混合状态配置。继续核对包装器的实际 forward 输入和输出，不能用 registry 注册成功冒充完整加载成功。[`new_arch_resolution`、`new_model_loader_entry`]

CPU sampling 必须显式设 `on_device_sampling_config=None`；新 `_get_default_neuron_config` 默认创建 on-device config。新 runner 的 deferred sampling 在恢复请求顺序、应用 grammar mask 后调用 CPU sampler，不能把旧整份 runner 覆盖上去。现有 bounded wrapper 的入口仍匹配新 `topk_topp_sampler(logits, generators, k, p)` 四参调用；保留 CPU FP32、top-k 1..64、候选上限 128，ties/nucleus 临界值回退、显式 generator 回退以及 processed logits/logprobs 回退。新栈还要实测输出类型、logprobs、混合 greedy/random、grammar 和随机分布等价性；旧栈的提速倍数不得直接搬过来。[`new_default_config`、`new_deferred_sampling`、`new_cpu_sampler`、`old_sampler_wrap`；完整旧 `cpu_sampler.py` 第 15 行起]

另一个已独立确认的上游问题是 loader 第 54 行把 `sampler` 模块当成类导入，显式 `NEURON_ON_DEVICE_SAMPLING_DISABLED=1` 后构造器调用 `Sampler()` 会失败。root 已用完整模块源码和 stub 依赖完成 6 个检查，并准备单行修复 `from vllm.v1.sample.sampler import Sampler`，本任务不重复实施。新 runner 自身已正确导入类，普通 CPU sampling 路径是否成功与这条 legacy 开关问题必须分开判断；它也不是已证实的 Qwen r6 API 故障根因。见 `vllm053-sampler-class-import-root-reproduction-20261112-v1.json`，SHA256 `25fb5fcdbe19cc624851a5cc99eedb4e5815ec4ce6462b601cade8881d09a54e`。[`new_legacy_sampler_import`、`new_runner_sampler_setup`]

LFM 的真实状态布局由 24 层组成：13 个 conv、6 个 full attention、5 个 sliding attention。每个 TP rank 有 13 个 conv state 和 22 个 K/V state，共 35 个；TP2 共 70 个。conv 为 `(B,1024,7)`，每个 K 或 V 为 `(B,2,L,128)`。这些按层顺序放在 `kv_mgr.past_key_values`；NxDI 0.10 的 alias 构造在无 output logits/capture/spec/LoRA 的本范围仍把输出 `1+i` 绑定至第 i 个 state。因此保留原扁平接口，重点检查顺序、长度和 MUST_ALIAS，不需要擅自改成 GPU Mamba cache。[`old_hybrid_cache`、`old_nxdi_model`、`new_alias_output_contract`]

新版 runner 却为全部 24 层创建 `FullAttentionSpec`，并以 `tensor_parallel_size` 作为其 `num_kv_heads`；在本模型 TP2 条件下这个数字恰好等于每 rank 的 KV heads。worker 用 `(B+1) × 24 × attention_page_bytes` 返回给调度器；contiguous 路径不读取真实剩余 HBM。`initialize_kv_cache` 在没有 KV connector 的范围由 NxDI 管理物理状态，因此这套数值可作为请求槽位预算代理，但不能说它精确描述 LFM 混合状态内存。[`new_proxy_cache_specs`、`new_proxy_cache_memory`、`new_cache_no_allocation`；`neuron_worker.py:119`]

最小建议是保留现有单组代理构造，并为 LFM 显式命名、限制和检查其含义：`block_size == max_model_len`，一个请求占一块，最终调度器共有 B+1 blocks，其中一个 null block；NxDI 实际 capacity 为 B，`max_num_seqs == max_batch_size == tkg_batch_size == B`，`ctx_batch_size == 1`。同时独立核对全部 35 state/rank。不要把代理预算写入“真实 HBM 消耗”报告。将代理改为单个 synthetic spec 是另一种设计，但会改变调度行为，不属于这里选择的最小迁移。

新 runner 的 `InputBatch` 当前只接一组 block size；多个路径还有 `len(block_ids)==1` 断言。因此直接生成 13 个 MambaSpec 加 11 个 attention spec 会引出多组调度/状态搬运的更大改造。新的 `IsHybrid` 还增加 prefix align 用的 `get_mamba_state_copy_func`；初始范围关闭 prefix 后，不需要伪造 GPU state-copy 函数。未来若支持 prefix caching，必须实现并验证真正的 conv/KV checkpoint、拷贝和恢复语义。[`new_hybrid_interface`、`new_hybrid_config`、`new_mamba_config`；新 runner 第 151、1366、1415、1627 行]

请求生命周期还须额外核对：finished request 在 `execute_model` 释放 Neuron slot，随后 `_update_states` 删除旧记录、允许同 request ID 作为新请求进入；未调度请求只从 input batch 移除而保留 slot。cached request 恢复路径会更新 block IDs，但 continuous decode 仍使用已有 slot 和最后一个 output token，没有建立被抢占状态的重新 prefill/恢复证明。初版应显式拒绝 unsupported preemption/resume，并通过足额 slot 预算避免正常负载触发；不能仅凭“更新了 block IDs”就声称支持抢占。超长 prompt 的校验应在占用 slot 前完成，或明确回滚，否则异常路径需要单独释放证明。检查 finish/new 同 ID、abort、错误请求、间歇调度、不同长度复用和 slot 耗尽。[`new_execute`、`new_request_update`、`new_request_slot_allocate`、`new_request_slot_cached`]

初始 admission 要同时覆盖 vLLM 和 NxDI：prefix caching、block layout、chunked prefill、KV transfer、speculation、LoRA、其他并行模式、量化和 async 路径均沿用原拒绝范围。vLLM 0.16 可能默认启用 `scheduler_config.async_scheduling`；平台的 `is_async_output_supported=False` 不是同一个开关。显式设 vLLM async scheduling false，另设 NxDI `async_mode=False`。prefix false 应在 hybrid config 验证前生效，因为 Mamba align 会要求 chunked prefill。保留 `apply_seq_ids_mask=True` 以及 padding slot `-1` 的规则；不得把无效行映射到真实 request slot。[`old_scope_admission`、`new_async_scheduler_default`、`new_platform_config`、`new_mamba_config`、`new_seq_ids_mask`]

编译产物需要新的独立身份和全新目录。旧 identity 明确固定旧软件、B4/2048 和 `nki_predicated_decode_v4`，修改其校验以接纳新产物会混淆已验收记录。新 manifest 应覆盖完整生效 config、原模型/视图、writer 代码、全部 typed 依赖、真实 SDK/source closure 和编译设置。现有不完整或不匹配目录直接拒绝，不能被通用 loader 的 `except (FileNotFoundError, ValueError)` 转为重编译，更不能进入自动 `rmtree`。[`old_artifact_admission`、`old_artifact_config`、`new_loader_generic_fallback`、`new_loader_generic_deletion`]

NxDI `get_builder` 仍从 `BASE_COMPILE_WORK_DIR` 环境变量取目录，默认 `/tmp/nxd_model/`。未来 full-model 运行需显式绑定独立工作目录、编译缓存和资源预算；primitive 的隔离目录不能自动保护 full model。旧 `serial_compiler` 将 builder 模块内 ThreadPoolExecutor 限制为 1；新 builder 仍调用该符号，文本上可接，但真实完整模型编译尚未验证。[`new_builder`、`old_serial_compiler`、`new_parallel_compiler`]

NKI 0.5 的直接不兼容点是 `nki.jit(platform_target='trn1')`：新 API 会显式抛 ValueError，需使用 `NEURON_PLATFORM_TARGET_OVERRIDE=trn1` 和 `NEURON_LOGICAL_NC_CONFIG=1`。已审查的 typed factory 使用新的公开 `nki.jit(function)`，并要求真实 owned admission 后才导入 native。保留 BF16/FP32 typed 位模式契约、负值/过大 int64 slot 归一化、重复 slot 写入语义、输入不变和每次 store 恰好一次；不能用 simulator 的成功替代设备。原 one-row prefill 仍走 original writer。[`old_nki_import`、`old_nki_cast`、`new_nki_jit`、`typed_normalize`、`typed_production_dispatch`；补充 `slot_cache_nki05_typed_jit_v1.py`]

4096 的迁移必须同时改变 `seq_len`、`max_length`、`max_context_length`、vLLM `max_model_len` 和完整状态长度，并以新的 scope 身份准入。显式保持 `allow_input_truncation=False`，超限输入应被拒绝。原始模型 `max_position_embeddings=262144` 只是官方配置字段，不能代替本机 4096 验证。保持 sliding window 256、conv 宽 7、原 RoPE 和 BF16/TP2；不要为了扩大 context 改模型结构。[`old_identity_model_scope`、`old_scope_admission`、`old_attention`、`old_conv`]

在 `enable_bucketing=True` 且 stage bucket override 为 None 时，NxDI 会生成 CTE/TKG 各 `[128,256,512,1024,2048,4096]`，stage config 会重新写入 buckets。根配置写 `buckets=[4096]` 不能代表只编译一个图。保留旧五个长度并新增 4096，对每个 capacity 是至少需要覆盖的 12 个 stage/bucket 组合；这不是实测 compiler artifact 个数，不能拿它估算总编译耗时。[`new_cte_buckets`、`new_tkg_buckets`、`new_generate_buckets`、`new_stage_configs`]

以下仅为单份逻辑 BF16 state pool 的字节数，未包含模型权重、CTE/TKG 是否共享/复制、临时张量、编译器、host staging 或运行时保留内存。计算为每 slot/rank `13×1024×7×2 + 22×2×L×128×2`。因此这些数值不能证明 B16/4096 一定能在本机完成编译/推理。

| context | capacity | 每 rank 字节 | TP2 合计字节 |
|---|---|---|---|
| 2048 | 4 | 93,020,160 | 186,040,320 |
| 2048 | 8 | 186,040,320 | 372,080,640 |
| 2048 | 16 | 372,080,640 | 744,161,280 |
| 4096 | 4 | 185,294,848 | 370,589,696 |
| 4096 | 8 | 370,589,696 | 741,179,392 |
| 4096 | 16 | 741,179,392 | 1,482,358,784 |

当前已冻结的 LFM primitive 扩展矩阵只到 2048，已有其他模型的 4-KV-head 4096 图不能替代 LFM 的 `(B,2,4096,128)`。后续必须以 supplement 增加 B4/B8/B16 的 standalone 和 MUST_ALIAS 4096 覆盖、有效/空/重复/乱序及 int64 越界 slots，保持原 113 图和其他检查全部存在。full model 还要覆盖跨 2047/2048、接近 4095、达到 capacity 后准确拒绝且不截断、prefill 旧 tail 清零、70 状态生命周期和新旧请求隔离。[`current_device_matrix`、`current_alias_matrix`、`preserved_full_gates`]

原门槛继续保留：B4/B8/B16 numeric frames 分别 32/32/64，cold frames 96/200/400；1529-step 长解码及相应 84/168/336 frames；每个候选 96 个 quality cases；7 个新冷进程、70 states、140 fill 和 140 zero roundtrips、5 次 snapshots，以及全部旧 buckets 和请求槽位复用。对比保持 original B4、selected writer v4 B4、typed B4/B8/B16；5 种 workload、3 轮，共同并发 1/4，容量并发 8/16 分开报告；每 variant 测量 704 requests、189952 output tokens、16 warmup requests，每 candidate 384 greedy text pairs。4096 是追加检查，不能取代这些旧要求。[`preserved_full_gates`]

准入顺序仍由 root 控制：先等待当前 full CPU r2 的真实全部 874 个进程退出及固定 session 的最终 exit 0/收据，完整排空并重新确认空闲；再按已审查 serial device pipeline 验证 typed 原语；随后才具备讨论 full-model migration 的实际 native 接线、冷缓存、长解码、slot reuse、质量和性能试验的基础。本报告不创建锁、service、handoff、设备运行绑定，也不改已冻结矩阵。

本次 inventory 的 V1 因 `constants.py` 实际只有 68 行而摘录终点写为 70 行退出 1（工具 chunk `bba219`），未输出证据文件；原脚本保留。V2 只修正摘录边界、改用独立输出名并记录此前失败，实际 chunk `f58b9d`、exit 0，完成 43 文件/68 段/16 API 的源码分析。这不计入模型或 SDK 运行通过数。
