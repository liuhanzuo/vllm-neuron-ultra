# DeepSeek observer-off 全量控制：路径参数化科学入口

新增入口 `deepseek_cold_control_entry_v1.py` 接受运行器已经验证并加载的原模型，不创建进程、队列、锁或编译任务。原生模型仍是官方 BF16/eager；设备模型必须是原 `NeuronDeepSeekQwen3ForCausalLM`，CPU sampler、TP2、B4、context4096，原 12 个桶图。设备模型必须由外层在新进程中完成原权重/产物校验，并以 `load(skip_warmup=True)` 加载；不允许复用上一 arm 的模型状态。当前没有实际运行绑定。

## 根运行器可调用接口

```python
from deepseek_cold_control_entry_v1 import reference, device
from verify_deepseek_cold_control_v1 import verify

# 不同阶段使用不同受控新进程；metadata_root、payload_root 分开，预先由运行器建立。
reference(admitted_native_model, metadata_root, payload_root, authenticated_guard)
device(admitted_original_neuron_model, original_case, metadata_root, payload_root,
       caller_authenticated_original_cold_metadata, authenticated_guard)

# 两个生产进程实际退出后，外层冻结真实完整输入摘要，再用独立审核阶段调用。
verify(original_completed_cold_kv_root, metadata_root,
       admitted_sha256=caller_authenticated_raw_and_outer_hashes,
       execution_guard=authenticated_guard)
```

guard 必须来自当前独占运行器的完整来源、原权重、产物、资源与进程所有权验证；这些 callable 不能自己签发准入。`original_case` 必须逐项等于原协议 `expected_case()`。`original_cold_metadata` 是外层已绑定的原完整报告，不能临时拿新结果当原对照。全部输出均要求新路径，旧 `deepseek-cold-kv-diagnostic-v2` 被保护。

## 精确保留的内容

`validate_deepseek_cold_control_v1.py` 从原 `validate_deepseek_cold_kv_v2.py` 机械派生：仅传入模型和路径，移除旧固定路径 loader/CLI，接入重定位后的 KV writer 和生命周期观察器。原 `main` 与 `reference` 中每个嵌套 for-loop 的完整 AST 均相等，包括七前缀、两输入流、原生 42 帧、设备 84 帧、四个物理槽位、原排序/还原、decode mask、六 CTE 和六 TKG 桶、所有 logits 指标及状态判断。

`deepseek_cold_control_capture_v1.py` 仅改变构造函数的输出根路径；`write/native/device/finish` 全文 AST 与原实现相等。所有 72 个 KV、完整有效历史、旧前缀改变时保存全量替换的 codec 与 manifest 约束保留。

`deepseek_cold_control_lifecycle_v1.py` 只重定位 144-state 快照及 state-copy 文件，原拥有者检查由外层实时 guard 承接。原 288 fill/288 zero、完整 before/after-warmup 两份快照、flag reset/clear、全部状态身份与原 validators 保留。

设备入口只接纳原精确终止文本 `Strict cold-bucket numerical/state gate failed`，随后调用原 `check_device_metadata`，要求所有 84 行/指标逐项相等和原 14 个失败索引保持；成功变成通过、遗漏失败帧、其他异常均拒绝。它还调用原 copy/lifecycle validators，要求 warmup 改变全部 144 个状态。该阶段不会把预期数值失败改判为模型通过。

## 全量独立字节复核

`verify_deepseek_cold_control_v1.py` 在任何 logits 反序列化前要求调用方认证的四份 raw-logit 哈希，并认证全部外层报告与 manifest；完整 KV 逐层沿 manifest/frame/payload 哈希链核验。它比较原生 42/device 84 完整 logits，原生 3,024/device 6,048 完整 KV 张量；所有旧前缀都参与，不能只比末 token。

只有全部比较通过，才用独占新文件模式写出 `result.json`、`native42-reproduction.json`、`device84-reproduction.json`。这些 receipts 与现有 `verify_deepseek_cold_boundary_neutrality_v2` 所需的 original-root 字段兼容，使后者可以拿本轮 fresh off 当 capture-on 对照。生产阶段终止/资源排空及这些审核输入的真实绑定仍由外层负责，不能单凭文件存在视作实际结束。

16 项源码/stdlib/mock 检查通过，工具 `e6ca57/0`：包括三文件全文机械重建、全部科学循环一致、KV writer 方法一致、整套 9,072 对模拟 KV 枚举、末层末帧旧历史损坏、遗漏/重排、缺失 raw admission、错误终止、未取得执行资格等拒绝检查。另读取原已保存的 955,746 字节 cold JSON，用原纯元数据 validator 重现全部 84 行/14 失败判定，并检查改动失败行、遗漏状态或帧均被拒绝。这是既有报告复核和模拟控制，未读取其 raw logits/KV，也未新增实机数值结果。

## 剩余工作与预算

capture-on 的原生加载 leaf v1 保持冻结。本次补齐 off 生产与复核叶子；新 capture 图 builder/load、实际 aliases 469..540、468 个附加 collective、完整源码/权重/编译产物准入和 root 串行接入仍未完成。

两臂完整输出的保守预算仍为 **47,150,344,192 bytes（43.912 GiB）**；另有未知的新编译 scratch/cache/图/NEFF，以及在尚未验证不可变共享时可能需要的 16,382,184,072 bytes 权重副本。没有假定压缩比例或删除已完成证据，未占用 Qwen 当前保留预算。原长解码 2,041 步/93 检查、slot reuse、quality96、匹配 HTTP 均仍是独立必需门槛。没有新模型通过或提速结论。
