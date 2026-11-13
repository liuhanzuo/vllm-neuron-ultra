# DeepSeek 捕获图：权重分片与编译存储的源码结论（v2）

本版本保留 v1 文件，并补充重试配置与执行模式的适用条件：默认重试次数为 0，当前 ModelBuilder 的 LAZY 失败路径直接抛异常，不进入缺少工作目录参数的递归。下文问题仅在可达的重试分支成立；本次未检查或选择实际运行环境。

固定对象是现有 `deepseek_capture_graph_entry_v1.py`、DeepSeek-R1-0528-Qwen3-8B 原 BF16、TP2、B4、context 4096，以及 `/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages` 的当前源码。此分析只读取 Python 源码和已有源码交付说明；没有读取模型配置正文、权重、图、原始张量、运行记录或 `/proc`，没有导入原生库或运行编译器。完整源码摘要见同名 JSON。以下分支结论仍以运行时认证这些源码和原配置为前提。

**可证明的窄结论：原配置的编译与加载 Python 路径不会调用保存 TP 权重 checkpoint 的分支。** 历史两份 TP 文件共 16,382,184,072 字节，不能直接当作本次新生成的磁盘副本，也不能据此证明实际运行不需要任何临时空间。此分析不修改当前预算，不签发运行资格。

## 原始调用链与权重寿命

捕获应用继承 `DeepSeekCaptureForCausalLM → NeuronDeepSeekQwen3ForCausalLM → NeuronQwen3ForCausalLM → NeuronBaseForCausalLM → NeuronApplicationBase`。捕获子类只替换内部模型类、配置和输出处理，没有重写 compile/load；见[捕获适配器](/home/ubuntu/experiments/trn1-latest-models-20261031/deepseek_qwen3_capture_adapter_v1.py:118)、[原适配器](/home/ubuntu/experiments/trn1-latest-models-20261031/deepseek_qwen3_neuron_adapter.py:13)、[Qwen 应用](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/qwen3/modeling_qwen3.py:194)和[基础应用继承](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/model_base.py:3074)。

1. **构造与编译。** 叶子的原配置校验要求 `save_sharded_checkpoint=False`、`skip_sharding=False`、`skip_warmup=False`，并保持 `_name_or_path` 指向原 YaRN 视图；[入口第 96 行](/home/ubuntu/experiments/trn1-latest-models-20261031/deepseek_capture_graph_entry_v1.py:96)。其 build 调用 `model.compile(..., pre_shard_weights_hook=None, dry_run=False)`；[第 268 行](/home/ubuntu/experiments/trn1-latest-models-20261031/deepseek_capture_graph_entry_v1.py:268)。SDK compile 明确调用 `builder.trace(initialize_model_weights=False)`，保存 `artifact/model.pt`，再调用 `shard_weights`；[application_base.py 第 292 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/application_base.py:292)。因此 ModelBuilder 中本会在工作目录写出 `tp{rank}_sharded_checkpoint.safetensors` 的 `if initialize_model_weights` 分支不执行；[model_builder.py 第 764 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/model_builder.py:764)。随后 `shard_weights` 因 `not save_sharded_checkpoint` 直接跳过预分片，未传入 `artifact/weights/` 保存目录；[application_base.py 第 240 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/application_base.py:240)。叶子结束检查还拒绝存在 `artifact/weights`；[入口第 216 行](/home/ubuntu/experiments/trn1-latest-models-20261031/deepseek_capture_graph_entry_v1.py:216)。

2. **加载和内存分片。** SDK load 先读取 `artifact/model.pt`，然后 `load_weights` 选择 `builder.shard_checkpoint()`，没有传入 `serialize_path`；[application_base.py 第 317 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/application_base.py:317)和[第 374 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/application_base.py:374)。该参数默认 `None`；builder 只在非空参数下建目录，每个 rank 的 `save_file` 也只在非空参数下执行；[model_builder.py 第 802 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/model_builder.py:802)、[第 979 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/model_builder.py:979)。`checkpoint.copy()` 是字典的浅复制，接下来的切分、拼接和 contiguous 等操作可以产生实际张量分配；不能从浅复制推断内存共享或零峰值。

3. **原权重读取。** `checkpoint_loader_fn` 优先使用配置的 `_name_or_path`，原目录存在时进入 `get_state_dict → load_state_dict`，加载完整官方 checkpoint 后做名称、Q/K norm 和分片转换；[application_base.py 第 628 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/application_base.py:628)、[第 690 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/application_base.py:690)、[checkpoint.py 第 24 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/modules/checkpoint.py:24)、[Qwen 转换第 206 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/qwen3/modeling_qwen3.py:206)。这一流程不是读取那两份历史 TP 诊断文件。未核对实际 shard 身份或运行时内存峰值。

4. **分片后的寿命。** 两个 rank 的字典先同时保存在 `sharded_checkpoints` 列表，随后传给 `nxd_model.initialize`；其权重字段被设置为设备 layout transform 或 `_parallel_load` 的返回值，全部桶模型使用相同的传入权重列表；[spmd.py 第 166 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/spmd.py:166)。Python 局部变量生命周期不证明底层 C++ allocator、布局转换、页缓存、共享内存或设备内存已释放；本次没有检查这些原生实现或实际占用。

## 编译输出仍需独立预算

ModelBuilder 为每个图调用 `generate_hlo(inline_weights_to_neff=False, return_weights=False)`；[第 889 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/model_builder.py:889)。Torch-NeuronX 因而在返回 HloArtifacts 时把 weights 与 constant-parameter tensors 设为 `None`；[trace.py 第 484 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/xla_impl/trace.py:484)。其 `setup_compiler_dirs` 只有在 inline 为真时调用写权重参数函数；[第 241 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/xla_impl/trace.py:241)。这排除了“把完整原 BF16 checkpoint 原样写进每个桶的参数文件”这一特定 Python 分支，不能推出 HLO、NEFF、调度信息或编译器临时文件大小不随模型几何增长。

| 存储位置 | 由源码确认的内容和寿命 | 仍不能从源码确定的量 |
|---|---|---|
| 独立 artifact | `model.pt`、模型配置、capture-build 报告；成功和失败产物由外层保留 | 12 个 NEFF、额外捕获输出、可选布局图经 JIT 序列化后的实际大小 |
| `BASE_COMPILE_WORK_DIR` | 每个 `tag/_tp0_bkN` 的 HLO、编译 flags、NEFF、日志、metaneff、配置；可有 `layout_opt`。trace 开始会删除既有 workdir，所以叶子要求目录原先不存在 | 编译器 scratch、链接资源、layout 图和失败现场的峰值 |
| `NEURON_COMPILE_CACHE_URL` | `<cache>/neuronxcc_<version>/<key>/` 中 HLO、flags、NEFF、可选 wrapped HLO、完成标记或失败日志；FS 实现用 copyfile 临时文件后原子替换，并非硬链接去重 | 缓存与工作目录、序列化 artifact 同时存在时的总分配；不可把同样字节只计一次 |
| Python 临时根 | `neuron_xla_compile` 第 251 行的 TemporaryDirectory 保存 `model.hlo` 与返回用 `model.neff`，正常离开上下文时清理；WLO 第 289 行还有对应 HLO/NEFF/wrapped-HLO 临时目录。`call_neuron_compiler` 第 63 行是另一个临时目录，在 LAZY/no-dump 分支中为空且在调用编译器前结束，不应把它算成实际 HLO/NEFF 副本 | TMPDIR 所在文件系统、异常强制退出残留、编译器自身额外临时根 |
| RAM / 共享内存 / Neuron 内存 | trace 设置 multiprocessing 的 `file_system` sharing strategy；load 包含完整 checkpoint、两个 rank 分片和设备加载/布局转换 | 该设置是否在实际执行中产生共享内存文件、各阶段峰值与真正释放时刻 |

关键实现：[ModelBuilder 编译输出与 NEFF 包装](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/model_builder.py:708)、[layout_opt](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/model_builder.py:1218)、[缓存上传与临时 NEFF](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:180)、[HLO/NEFF TemporaryDirectory](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:251)、[WLO TemporaryDirectory](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:289)、[LAZY/no-dump 空临时目录](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:63)、[缓存路径选择](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_cache.py:74)、[缓存复制与原子写入](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_cache.py:354)。

叶子认证的 **39 份文件不是全目录存储清单**，没有枚举 cache、HLO、编译日志和临时文件。外层还必须对实际所有输出根计费。两个完整 on/off 输出的 **47,150,344,192 字节**保持；总峰值还要加各阶段同时保留的 artifact/cache/workdir，以及临时、共享内存和编译资源预留。没有提出压缩比例、复用旧分片或新的数字上界。

## 发现的 SDK 重试工作目录问题及其触发条件

当前 [ModelBuilder 第 604 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/model_builder.py:604)明确传入 `lazy=True`、`use_cache=True`、`create_subdir=False` 和专用 bucket 目录；[WLO 调用](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/model_builder.py:651)也选择 LAZY。所审源码的 `NEURON_PARALLEL_COMPILE_MAX_RETRIES` 来自环境变量，**默认值为 0**；见[第 24 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:24)。本次没有读取实际进程环境，因此“源码默认值”不是对当前配置的认证。

LAZY 的 subprocess 返回非零时，[第 114 行起](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:114)会抛出 `CalledProcessError`；`compile_cache_entry` 在 [第 227 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:227)发现 `retry == maximum_retries` 时记录错误并直接抛出。**默认 0、初次调用 retry=0 的 LAZY 失败不会递归，不会由此切到回退目录。** 不能把下述有条件问题写成默认编译必然触发的故障。

只有执行到重试递归时，[第 235 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:235)才暴露未传递 `work_dir` 的缺陷。新调用使用默认 `None`，随后 [get_work_dir 第 150 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:150)选择 `/tmp/<USER>/neuroncc_compile_workdir`。`create_subdir=False` 保留，因此不是自动生成 UUID 的目录；[第 153 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:153)还会对已存在的目标调用 `shutil.rmtree`。这个回退来自字符串拼接，**仅设置 TMPDIR 不能约束它**。例如主动配置 maximum_retries=2 后，LAZY 首次异常就能进入此分支。

另外，EAGER 的 [client.run 第 101 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:101)可以直接返回非零值；它不走上述异常次数检查，会落入递归。模拟中，即使 maximum_retries=0，EAGER 的非零返回后仍发生第二次调用。这是另一个执行模式的源码行为，**不是当前 ModelBuilder LAZY 失败的实测路径**，也没有在此测试真实 EAGER 客户端。v1 的非零返回控制未区分此模式；v2 已明确区分。

[隔离复现脚本](check_deepseek_compile_retry_workdir_source_20261113_v2.py)提取并执行当前 SDK 的 `get_work_dir` 与 `compile_cache_entry` 原函数 AST，函数体未改写；真实 SDK 文件 SHA-256 固定，文件、编译器、进程和目录操作均为内存假对象，审计钩子禁止实际进程与目录变更。**八项控制通过，实际工具 `13d396 / exit 0`**：

| 控制 | 观察到的模拟行为 |
|---|---|
| 默认 0，LAZY 首次成功 | 只用指定工作目录 |
| 默认 0，LAZY 首次异常 | 记录错误并抛出；仅一次调用，无递归 |
| 配置 2 次，LAZY 异常后成功 | 指定目录转为硬编码回退目录 |
| 配置 2 次，EAGER 非零后成功 | 指定目录转为硬编码回退目录 |
| 配置 2 次，LAZY 且 create_subdir=True | 两种根目录分别加 UUID，根目录仍改变 |
| 配置 2 次，LAZY 三次异常 | 第三次保留异常，首次目录加两次回退目录 |
| 配置 2 次，LAZY 且假设回退目录存在 | 记录一次模拟 rmtree；无真实删除 |
| 默认 0，EAGER 非零后成功 | 仍递归一次，区别于默认 0 的 LAZY 异常 |

完整结果见[源码控制 JSON](deepseek-compile-retry-workdir-source-tests-20261113-v2.json)。没有真实编译器失败、目录逃逸、磁盘事故或删除的观测，没有因本结论立即修改 SDK 的必要。后续应先认证实际选定路径及重试配置；若选择可达的重试分支，才需要经过独立审阅的修复或完整原路径保护与预算。正常 HLO/NEFF 临时目录、持久 cache 和 bucket 工作目录仍需各自绑定与计费；只监控三个 capture 目录不足以证明包含全部写入。此分析不选环境、不改重试设置、不降低资源或科学门槛。

原 native42/device84、全部 9,072 份 KV、144 状态、14 个失败点，以及后续长解码 2,041 步/93 位置、槽位复用、quality96、匹配 HTTP 均保留；本次没有模型资格或提速结果。
