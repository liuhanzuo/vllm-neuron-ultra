# Qwen3.5-2B 实验包接入

最新结果：[第四轮原生预检通过](qwen-package-preflight-passed-20261112.md)。实际父进程和全新 spawn 的导入、分词、配置、BF16 拆分与采样器接线全部通过，退出及完整清理已经独立核验。独立包的编译、整模型和性能验收仍待完成。

统一分支现已加入 Qwen3.5-2B 的模型注册、文本适配器、split 修正、模型加载与显式服务入口。**目前完成的是源码接入和模拟验证；新包尚未编译或做设备、质量、性能验收。** r9 已因实验汇总问题退出；修正后的 [r10](qwen35-r10-results-20261112.md) 已完成研究路径整轮验收。这些研究运行使用原实验适配器，其结果不能直接作为新包结果。

## 实现范围

固定官方版本为 `Qwen/Qwen3.5-2B` 的 `15852e8c16360a2fea060d615a32b45270f8a8fc`，保留原 BF16 权重、FP32 循环状态、TP2、物理 batch 4、prefill batch 1、context 4096，以及 128／256／512／1024／2048／4096 六个桶。仅支持已研究的文本路径。

数值实现位于 [models/qwen35](../vllm_neuron/models/qwen35/)，保留单 token prefill、缓存槽位写入和 inactive slot 行为。`state.clear_states` 为冷缓存诊断原地清零现有 96 个设备缓冲区，并检查原始字节与绑定身份；它不会重新初始化计算图，也不会改变正常请求调度。

模型注册通过显式 `serve` 入口在主进程和 spawn 子进程执行。通用加载器新增仅针对 `qwen35_neuron_text` 的分支，在普通缓存删除／重编译回退之前调用专用产物校验。正常插件 worker 继续使用已有的 [CPU sampler](../vllm_neuron/worker/cpu_sampler.py)，没有再复制一套采样器。

标准与快速采样共享同一份编译产物。`NEURON_CPU_SAMPLER_FAST` 只选择主机端采样路径，记录在主机配置中；采样器源码仍被绑定。改变这个开关不会单独改变图或权重身份。

## 实验入口

在独立安排的固定 Python 3.10／Neuron 环境中安装该分支后，先创建新的文本配置视图。以下是后续运行模板，**本轮没有执行这些编译或服务命令**：

```bash
python3 -m vllm_neuron.models.qwen35.prepare_view ORIGINAL_MODEL FRESH_TEXT_VIEW

NEURON_QWEN35=1 \
NEURON_ON_DEVICE_SAMPLING_DISABLED=1 \
XLA_DISABLE_FUNCTIONALIZATION=1 \
NEURON_COMPILED_ARTIFACTS=FRESH_ARTIFACT_PATH \
NEURON_QWEN35_ALLOW_COMPILE=1 \
python3 -m vllm_neuron.models.qwen35.serve \
  --model FRESH_TEXT_VIEW --served-model-name qwen3.5-2b-text \
  --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 4 \
  --no-enable-prefix-caching --host 127.0.0.1 --port 8010 \
  --additional-config '{"override_neuron_config":{"on_device_sampling_config":null,"apply_seq_ids_mask":true,"context_encoding_buckets":[128,256,512,1024,2048,4096]}}'
```

首次编译需要显式开关、尚不存在的产物目录和已存在的父目录。后续加载可移除编译开关；不完整、来源不符或配置不符的已有目录会报错，不会自动删除重建。独立构建入口为 `python3 -m vllm_neuron.models.qwen35.build FRESH_TEXT_VIEW NEURON_CONFIG_JSON`，后一个文件必须提供完整、符合固定范围的配置。

省略 `NEURON_CPU_SAMPLER_FAST` 使用标准采样；设为 `1` 使用现有快速采样实现。这个入口使用正常插件 worker，不能混用原实验研究 worker 和 `TRN1_QWEN35*` 开关。

新包的源码身份与实验目录不同，旧实验产物不能直接继承。模型视图中的权重和 tokenizer 字节保持不变，但必须为新包重新编译并验证图。完整配置包含视图路径，移动视图可能使旧产物校验失败。共享加载器的源码变化也会改变其他已绑定该文件的模型产物身份；本次没有改写旧清单来隐藏差异。

## 验证与发现的问题

主任务独立运行以下命令，25 项源码／模拟测试通过；运行器禁止导入实际 Torch、Neuron、Transformers、vLLM 和 NumPy：

```bash
python3 -B -s test/qwen35/run_source_tests.py
```

另独立核对了 11 份原始源码、23 个数值函数或类的 AST 对应关系，以及 SDK 配置转换测试数据与真实固定源码的一致性。保存的两侧 r9 数值数据各 203 行通过原判据，87 个长轨迹帧的身份和顺序一致。这些保存数据检查没有运行新包的模型。

接入过程中修正了两项具体错误：

- 初版检查曾增加 KL 非负要求，但原计算存在极小的负数舍入结果。现恢复原 `strict_rows` 函数和原阈值，没有改变科学验收标准。
- SDK 会移除模型配置顶层的 `dtype`／`torch_dtype` 别名。运行时检查现按这个实际转换比较配置，原始视图仍逐字节核对，Neuron 配置仍严格要求 BF16。

日志中的 Mistral 分词正则告警属于固定 Transformers 的模型类型误判，告警分支不会修改正则。现有六条原始 HF token 序列对照通过，全部 quality96 输入的官方分词对照仍待补充；详见[告警来源与验证范围](qwen35-tokenizer-warning-20261112.md)。

还通过模拟检查覆盖原地清零、其他槽位保持、注册冲突、显式编译开关、编译／加载成功后才发布产物清单，以及已有产物加载失败时不自动重编译。

后续仍需新包的真实导入和 spawn、12 图编译及 HLO 检查、tokenizer、32 帧数值／84 帧冷缓存／1,541 步长解码与槽位复用、真实采样重放和 13 项控制、quality96、配对 HTTP 及实际退出核验。这里没有新增合格提速倍数。

独立包的原生预检 v3 已通过源码复核。31 项源码／模拟检查通过，保留原主进程与 spawn 子进程的实际导入来源、三处模型注册、完整 SDK 配置保存／加载、错误配置拒绝、12 KiB CPU BF16 split 和两种采样器的独立实例接线检查，并增加全部已知输入的完整 token ID 比较。70 个插件源码文件与 20 个 SDK 源码文件继续绑定哈希。当前没有执行原生预检。

预检复用当前串行调度与锁，由外层监督器处理完整历史核对和全部子进程清理；它不加载模型权重或编译图。即使预检通过，也只证明导入、配置接线及指定输入的分词行为，仍需上述独立模型验证。[当前 v3 源码及独立复核](evidence/qwen35-tokenizer-preflight-v3-20261112/manifest.json)与[此前 v1 接入证据](evidence/qwen35-package-native-preflight-20261112/manifest.json)均保留。下一项设备端候选为[合并解码槽位写入](qwen35-slot-writer-candidate-20261112.md)，尚无实测提速。

完整证据见[固定源码清单](../test/qwen35/source_delivery_v2.json)、[依赖清单](../test/qwen35/dependency_inventory.json)与[独立源码复核](evidence/qwen35-package-source-20261112/manifest.json)。既有实测结果继续见[提速表](model-speedups-20261112.md)。

串行执行接入 v2 已通过 18 项源码／模拟检查及 6 项独立检查，修复了服务未安装时可能持锁循环的问题。真实 package 预检仍待执行，见[接入修复及验证范围](mini-formal-import-and-preflight-20261112.md)。
