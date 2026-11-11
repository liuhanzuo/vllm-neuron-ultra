# DeepSeek CPU sampler 中间值捕获：源码交付，尚未运行模型

当前实现补齐了 SDK 捕获开关与原 CPU sampler 不兼容的源码缺口。原模型仍为 DeepSeek-R1-0528-Qwen3-8B，BF16、TP2、B4、context4096。没有改动已执行的诊断文件、安装的 SDK 或已有编译产物，没有启动模型、编译、设备任务或读取原始张量。

建议根执行者审查以下当前入口，路径均相对于 `/home/ubuntu/experiments/trn1-latest-models-20261031`：

|文件|实现内容|
|---|---|
|`deepseek_cpu_capture_bridge_v1.py`|固定一份完整 logits、468 个具名捕获张量、72 个 KV 输出的顺序；序列化配置转换；预测位置选取；原 14 失败点和全 84 帧的元数据一致性检查。|
|`deepseek_qwen3_capture_adapter_v1.py`|独立配置、模型及应用子类。保留 CPU sampler，显式解析捕获输出和 KV 偏移；在图内选择预测位置后才汇集两个 TP 分片。|
|`deepseek_cold_boundary_capture_v1.py`|原生模型实际 forward hooks/位置 trace 与设备输出写入器。Q/K/V 分别保存，设备保留末尾 rank 维；KV 使用原 codec 保留完整历史。|
|`validate_deepseek_cold_boundary_v2.py`|保留原 cold42/84 全部循环、槽位、桶路由和数值判定，绑定独立生命周期观察器，并保存新 device logits 摘要。无 CLI、下载、加载、编译或排队入口。|
|`deepseek_cold_boundary_lifecycle_v1.py`|原状态复制、warmup/reset/clear 检查的独立输出路径版本；原校验器继续适用。报告明确标注新图，case 保留原比较配置。|
|`verify_deepseek_cold_boundary_neutrality_v2.py`|要求根执行者绑定四份原始 logits 和所有外层元数据/manifest 的 SHA256，然后比较全部原始 logits 与完整 KV 原始字节。v2 是推荐验证入口。|

源码检查证据保存在：

- `research/deepseek-cpu-capture-bridge-source-checks-20261111-v1.json`：实际冻结 SDK AST 的三个原始问题复现、配置往返、预测位置、Q/K/V 和 rank 保留、错误输出偏移拒绝、原循环一致性。
- `research/deepseek-cold-boundary-integration-source-checks-20261111-v2.json`：独立生命周期路径、完整缓存配对、遗漏/重排拒绝和完整存储预算；原 v1 源码与证据哈希保持。
- `research/deepseek-cold-boundary-admission-source-checks-20261111-v3.json`：缺失、错误或畸形调用方哈希拒绝；四份 logits 在反序列化前全部核验；外层报告和两个后端的原始/新增 manifest 均须绑定。

所有检查只用了标准库、真实源码 AST、已有小 JSON 和惰性哨兵对象。没有 import Torch、NumPy 或 NxDI。这些结果不是数值实验通过证明。

## 已固定的观测范围

保留原 native42、device84、四个物理槽位、两条不同输入流、全部七种冷缓存前缀，以及 14 个原失败帧。每层捕获 13 个边界，共 468 个具名张量。每个边界保存原预测位置的完整特征向量；设备另有长度为 2 的 rank 维。Q/K normalization 的序列轴为 1，prepared Q/K/V 的序列轴为 2。

原 SDK 为 CTE position_ids 填充值 1；七种前缀均至少 32，因此图内 argmax 选择真实前缀的最后位置。TKG 每行仅一个 query，取列 0。设备写入器只接受原 CTE 单槽或 TKG `[0,1,2,3]`，明确记录捕获行与恢复后的 logits 的槽位对应关系。全 36 层、全部有效历史 token 的 KV 仍保留。

其他 prefill token 的中间激活、attention probabilities 和 MLP gate/up/product 不在本次捕获范围。这足以观察预测位置的分歧，不能声称首个全局因果故障，也不能直接开展缺少完整输入的因果替换。

## 根执行者接入顺序

1. 保持串行所有权，重新绑定当前源文件、原模型和编译器版本、原输入、数据路径以及空间下限。先完成 untouched observer-off 的同协议控制；原失败点不得被新诊断“修正”。
2. 用 `instrument_config_dict(original_serialized_config)` 生成新配置，再由 `DeepSeekCaptureInferenceConfig.from_json_string` 构造。只允许新产物目录。验证序列化往返、权重键/分片不变，以及实际 trace KV aliases 为 469 至 540；不得加载原图并把它当捕获图。
3. 在实际 SDK 中验证 trace/hooks、graph outputs、468 个额外 collectives 和模型 warmup。上述行为尚未实测。编译及载入必须使用新的 admitted 工件，不得绕过原始权重验证或修改 sampler。
4. 原生控制仍由原官方模型和来源加载器提供；对已加载原生模型构造 `Capture('native', fresh_destination, fresh_native_payload_root, model=native_model)`，调用 `validate_deepseek_cold_boundary_v2.reference(model, protocol, source_hash, fresh_reference_directory, capture)`。调用方须用 finally 关闭 capture hooks。
5. 对已加载且经过独立入场检查的 `DeepSeekCaptureForCausalLM` 构造设备 Capture，调用 `validate_deepseek_cold_boundary_v2.main(model, case, fresh_destination, fresh_reference_directory, capture)`。它执行原 state-copy/lifecycle/cold84。原 strict cold gate 失败是预期结果；只能精确处理原错误文本，不能吞掉其他失败。
6. 将新终态文件和原证据的真实哈希绑定为独立 admission 输入，调用 neutrality v2 的 `verify(original_root, fresh_destination, fresh_reference_directory, admitted_sha256=...)`。所有外层 JSON 与四份 logits 必须已绑定；内层 KV 元数据/载荷沿被绑定 manifest 的哈希链核验。不能在验证器内临时从待验证文件生成“期望哈希”。
7. 只有原 native42/device84 logits、原 native3024/device6048 完整 KV、全部原状态及槽位检查保持，才能分析预测位置边界。长解码、请求槽位复用、完整质量与性能门禁仍保持原任务要求；此诊断不新增模型通过或提速结论。

## 空间预算和未获得的运行依据

边界张量共 572,866,560 bytes（约 0.53 GiB）。全 126 帧 logits 另需 76,575,744 bytes。两份全 144 状态的生命周期快照为 4,982,833,152 bytes。完整 KV journal 在历史前缀不变时为 5,366,808,576 bytes；如果每次都需完整替换，最坏为 16,081,846,272 bytes。额外保留 2 GiB 供序列化和元数据。

因此输出预算为正常前缀约 12.24 GiB、最坏约 22.22 GiB，尚不含新编译 scratch、新 NEFF/序列化图或权重副本。历史两份 TP 权重 shard 合计 16,382,184,072 bytes（约 15.26 GiB）；只有独立验证不可变权重共享后才可不计新增副本。历史 model.pt 的 176,072,171 bytes 不能作为新图大小上界。

可考虑根盘放 native payload 和生命周期快照、NVMe 放 device payload；最坏 native KV 约 4.99 GiB，device KV 约 9.99 GiB。调用方报告的根盘空闲 28.67 GiB、NVMe 空闲 44 GiB/下限 30 GiB 是状态参考，此任务没有重新扫描空间。编译及权重共享仍无实测大小，故当前没有总执行空间入场结论，也没有启动依据。
