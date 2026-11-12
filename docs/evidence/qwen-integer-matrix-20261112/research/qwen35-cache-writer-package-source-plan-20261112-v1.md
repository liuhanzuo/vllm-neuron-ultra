# Qwen3.5 缓存写入的独立包接入方案

这是只读源码方案，未修改包、导入 native、编译或准备当前运行。先完成当前独立原语矩阵的实际退出审计，再完成现有独立包 **采样优化 25 阶段资格验证**，最后另建 cache writer 整模型对照。原语矩阵的参考是新命名的整数 bitcast writer；其比值不能作为原模型 baseline 提速，也不能与已有采样提速相乘。

精确 SHA、字节数和逐项行号在 [JSON 源清单](qwen35-cache-writer-package-source-plan-20261112-v1.json)；记录了 22 个包、研究和 SDK 源文件。AST 核对证明包和研究的原 `store_slots` 函数完全相同。

## 实际调用与覆盖边界

| 写入 | 包 / 研究调用行 | TP2 下每 rank 的 cache / decode updates | 类型 | 当前矩阵对应 |
|---|---|---|---|---|
| Conv | `text_core.py:104` / `qwen35_text_port.py:102` | `(4,3072,4)` | BF16 | conv |
| Recurrent | 同上 | `(4,8,128,128)` | FP32 | recurrent |
| Attention K、V | `text_core.py:151` / `qwen35_text_port.py:149` | `(4,1,L,256)`，L 为 128…4096 六个 bucket | BF16 | 六个 KV 布局 |

包有 18 个 linear attention 层和 6 个 full attention 层，每 rank 48 个缓存，TP2 共 96 个。分配在 [modeling.py:77](/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/modeling.py:77)，FP32 保留在 [modeling.py:155](/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/modeling.py:155)。方案保持这些分配、精度和别名关系。

三项差异需要明确保留：

1. **Prefill 未被本矩阵覆盖。** 包固定 `ctx_batch_size=1`，cache 容量仍为 4，但 updates 和 slots 只有 1 行；矩阵是 cache 4 / updates 4 / slots 4。第一版只替换 decode writer。若以后替换 prefill，需增加 capacity 4 / updates 1 的独立验证。
2. **包的图输入 slots 为 int32。** 实际文本路线是 runner 的 `input_block_ids` → [NeuronCausalLM.forward:363](/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/worker/neuronx_distributed_model_loader.py:363) → SDK `model_wrapper.py:1337` 的 int64→int32 转换；SDK trace 本身也在第 254 行构造 int32。矩阵入口是 int64，NKI wrapper 内再转 int32。因此还需验证实际 int32 入口。loader 第 531 行属于 multimodal 分支，不作为本文文本路线证据。
3. **KV 实际输入是切片。** [modeling.py:187](/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/modeling.py:187) 把最大缓存裁到 decode bucket，之后拼回未改动尾部。矩阵的独立完整张量不能证明切片 stride、reshape 是否产生拷贝、old gather 与写入的依赖、尾部保留及 TP2 多状态别名的整图行为。

这里的 slots 是序列缓存行号，**不是分页注意力的 token `slot_mapping`**。实际值为 `-1` 和 `0..3`；SDK 将不活动行置为 `-1`。625 个四行映射覆盖这些值及重复目的槽位；原 loop 与 NKI 都采用最后一次写入生效。正常服务分配的活动槽位通常互不重复。

## 显式接入及对照

建议新开关 `NEURON_QWEN35_CACHE_WRITER`，只接受 `torch_where` 和 `nki_predicated_decode_v1`，默认原 writer。这个开关尚未实现。

在 `LinearAttention`、`FullAttention` 上保留默认原 `cache_writer` callable；只把四处写入改为调用该字段。由 `NeuronQwen35Model.init_model` 在原 TP 分片完成后、trace 构造时绑定后端。Prefill 和 CPU 参考始终绑定原 writer。避免全局 monkeypatch、运行中读取环境切换或未声明的静默 fallback。

候选把已审阅的 [NKI wrapper](/home/ubuntu/experiments/trn1-latest-models-20261031/slot_cache_nki_predicated_torch.py:6) 和 [predicated kernel](/home/ubuntu/experiments/trn1-latest-models-20261031/slot_cache_nki_predicated.py:8) 作为包内模块，使用相对导入；不能让服务依赖研究目录。保持 BF16/FP32 payload 和 int32 索引，不改 old gather、卷积、delta、attention、scatter、KV 尾部拼接或状态清理。

两臂使用同一份源代码，仅显式 writer 模式不同：baseline 为原 `torch.where` writer，candidate 为 decode NKI writer。CPU sampler 必须两臂固定同一个值；可在采样 25 阶段独立通过后同时设为 `1`，不能再用 baseline `0` / candidate `1` 混合测量。整数 bitcast writer 仍仅作为另有名称的诊断参考；原 baseline 若失败，应保留失败。

`identity.py:217` 的 artifact identity 目前没有 writer 选择字段，不能只加环境开关后复用旧图。新后继必须将 writer、作用阶段、版本和 kernel/helper 哈希绑定到 identity，并验证父进程和 spawn worker 一致。两臂分别使用全新的 artifact、compiler cache、workdir，分别构建与验证 12 个图。源码改动与额外存储先走独立审阅；现有采样实验共享同一 artifact 的假设不能照搬。若 kernel 放子目录，还要扩展目前仅 `glob('*.py')` 的包内源清单。

## 后续资格与计时

先保持现有采样实验的含义和全部 25 阶段不变。其独立最终审计完成后，另建 writer 实验和相应的双 artifact 证据链；保持原模型字节、分词器、BF16/FP32、TP2、B4、ctxB1、context4096、六 buckets、SDK/compiler flags、请求与生成设置、CPU/线程/资源限制一致。

Writer 整模型资格仍须覆盖：96 个实际状态及原地 raw-zero 清理、原数值门槛、六冷 bucket、1541 步长解码、请求槽位复用、partial batch / inactive resume、完整 203 向量及原精确比较、采样/capture 控制、96 个质量样本及原 69/96 基线复现、全新无 observer 的 HTTP smoke / initial120 / matched workloads，以及实际退出和所有后代清理后的独立最终审计。

只在这些门槛通过后发布新测得的整模型吞吐与延迟比值，并写清 writer baseline 和负载。不能从原语调用时间推算整模型收益，也不能相乘采样、模型或原语的历史提速。
