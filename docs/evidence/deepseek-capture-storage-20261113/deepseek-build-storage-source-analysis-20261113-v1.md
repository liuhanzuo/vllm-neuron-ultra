# DeepSeek 捕获图：权重分片与编译存储的源码结论

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
| Python 临时根 | `neuron_xla_compile` 的 TemporaryDirectory 内保存 `model.hlo` 与返回用 `model.neff`，正常离开上下文时清理；WLO 路径也有对应临时目录 | TMPDIR 所在文件系统、异常强制退出残留、编译器自身额外临时根 |
| RAM / 共享内存 / Neuron 内存 | trace 设置 multiprocessing 的 `file_system` sharing strategy；load 包含完整 checkpoint、两个 rank 分片和设备加载/布局转换 | 该设置是否在实际执行中产生共享内存文件、各阶段峰值与真正释放时刻 |

关键实现：[ModelBuilder 编译输出与 NEFF 包装](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/model_builder.py:708)、[layout_opt](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/model_builder.py:1218)、[缓存上传与临时 NEFF](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:180)、[TemporaryDirectory](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:240)、[缓存路径选择](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_cache.py:74)、[缓存复制与原子写入](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_cache.py:354)。

叶子认证的 **39 份文件不是全目录存储清单**，没有枚举 cache、HLO、编译日志和临时文件。外层还必须对实际所有输出根计费。两个完整 on/off 输出的 **47,150,344,192 字节**保持；总峰值还要加各阶段同时保留的 artifact/cache/workdir，以及临时、共享内存和编译资源预留。没有提出压缩比例、复用旧分片或新的数字上界。

## 发现的 SDK 重试工作目录问题

`compile_cache_entry` 的首次调用接收 ModelBuilder 指定的 bucket 工作目录；失败后的递归调用在[neuron_cc_wrapper.py 第 235 行](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:235)没有传递 `work_dir`。新调用使用默认 `None`，随后[第 142 行的 get_work_dir](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:142)选择 `/tmp/<USER>/neuroncc_compile_workdir`。`create_subdir=False` 会保留，故这不是每次自动分配 UUID 的目录；get_work_dir 还会删除已存在的目标。这个路径来自字符串拼接，**仅设置 TMPDIR 不能约束它**。

[隔离复现脚本](check_deepseek_compile_retry_workdir_source_20261113_v1.py)直接提取并执行当前 SDK 的两个原函数 AST，函数体没有改写，文件、编译器、进程和目录操作均用内存假对象。五项控制通过，实际工具 `1e713c / exit 0`：首次成功仍用指定目录；非零返回和 CalledProcessError 重试都会改到默认目录；开启子目录模式只在两种根目录后添加 UUID；重试耗尽保留原异常。结果在[完整 JSON](deepseek-compile-retry-workdir-source-tests-20261113-v1.json)。没有运行真正编译器，也没有实际发生目录逃逸或删除。

这是已复现的 SDK Python 路径传递问题，不是 DeepSeek 数值失败或已观测的磁盘事故。后续受控运行必须在接入时处理该路径：给正常临时文件显式绑定独立 TMPDIR，同时对这个不遵循 TMPDIR 的重试入口采用经过独立审阅的修复或完整原路径保护与预算。不要把只监控三个 capture 目录当作覆盖全部写入。此分析不修改 SDK、不创建修复、不选择实际环境，也不降低原重试、资源或科学门槛。

原 native42/device84、全部 9,072 份 KV、144 状态、14 个失败点，以及后续长解码 2,041 步/93 位置、槽位复用、quality96、匹配 HTTP 均保留；本次没有模型资格或提速结果。
