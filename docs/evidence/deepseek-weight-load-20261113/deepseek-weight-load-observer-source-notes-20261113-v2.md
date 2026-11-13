# DeepSeek：实际权重加载观察入口与仍未闭合的布局证据

本次是源码和隔离测试交付，没有读取真实权重、model.pt、HLO/NEFF 或任何原始张量，没有调用 Torch/SDK、运行 History、取得串行锁、编译或运行模型。新增 `../deepseek_weight_load_observer_v2.py`，没有修改已冻结入口或 SDK 文件。v1 的 18 项 stdlib/fake-tensor 测试通过（`03e791 / exit 0`），但初始调用示例错误地放在现有 run 外层，主任务复核发现这一 fresh-import 冲突。本 v2 修正调用说明，实际函数逻辑保持相同；完整接线仍未完成。v1 源码、说明和交付清单原样保留。

## 可调用边界

```python
from deepseek_weight_load_observer_v2 import observe_load

with observe_load(
    exact_model_class,             # 原有 off/on 类，须继承原 SDK load_weights
    arm,                           # "off" 或 "on"
    authenticated_checkpoint_provenance,
    authenticated_serial_guard,
    validate_and_retain_observation,
):
    model.load(original_artifact, skip_warmup=True)
```

这段示例仅是后继入口内部的加载边界示意，不是当前运行绑定，也不表示可立即执行。必须由新的 load-boundary successor 先完成原 fresh-process 检查、外部认证和原 native 依赖导入，再在真实 model.load 前安装观察器。**不能**把现有 off/on run 函数直接放在此 with 内：观察器进入时导入 SDK，而现有 run 的开头会正确拒绝已导入 native 的进程。当前没有实现这个后继入口，不能删去 fresh 检查或隐藏提前导入来接通。外层仍须认证完整权重、配置、SDK/source、环境、资源与进程归属，在独立进程中安装观察器。它临时包裹原 SDK 的 `NeuronApplicationBase.load_weights` 和 `ModelBuilder.shard_checkpoint`，每个原方法只调用一次，保留原参数、原返回对象及异常，不另做第二次加载或分片。

`shard_checkpoint` 返回时，对 **rank 0、rank 1 的每一个原键**保存 dtype、shape、stride、设备、storage offset/bytes/address、tensor version，以及完整字节数和 SHA256。随后由原 `load_weights` 把同一份返回对象交给原 `nxd_model.initialize`；初始化返回后，对实际 `nxd_model.weights` 做同样的完整观察。rank 数据不能省略，未初始化的 `__neuronprivatetensor__` 哨兵被拒绝。

字节来自 `detach().cpu().contiguous().reshape(-1).view(uint8)`，没有数值转换，包含 NaN 位模式和有符号零。先展平再取字节视图支持标量 rank/buffer 张量。每个张量和每个 4 MiB 哈希块都有外层 guard，观察前后核对 storage/version。一次最多额外复制一个完整张量到 CPU；这不包括 SDK 自有的 checkpoint、两个 rank、转换和共享内存峰值，**不能降低任何已有资源预算**。

`observation_callback` 在真实初始化完成后、`load_weights` 返回原 `load`/driver 前执行。它可以保留失败证据并抛出异常，阻止后续推理；若独立图/布局门槛未满足，调用方必须这样处理。该模块本身没有创建任何放行凭据。正常结束、原 sharder/loader 失败、guard 失败、callback 拒绝或后续 driver 失败时，两个方法都会恢复。

父进程仍须独立观察真实子进程退出和服务排空，再认证并封存结果摘要。子进程中的 report 不自证退出，也不能作为“已完成完整验证”的凭据。当前七阶段串行入口**尚未接入此观察器**。

## 这些字节能证明什么

`compare_observations(off, on)` 纯比较完整两 rank、每个键的原 dtype/shape/完整字节 SHA 和外部 checkpoint 来源，分别返回加载前分片与加载后 CPU 表示是否相同。进程地址不用于跨运行相等性判断。相同 shape 但任一 rank 的值变化会被保留为不相等。

`.cpu().contiguous()` 是张量的 CPU 表示，**不是设备分配器物理 backing buffer 的直接转储**。实际设备拷贝或视图可能隐藏布局细节；相同 CPU 值不自动证明两个推理图以同样方式解释权重。因此全部观察和比较始终保留：

- `graph_weight_ABI_proven = False`
- `layout_semantics_proven = False`
- `inference_admitted = False`
- `full_model_qualified = False`
- `new_speedup_claim = False`

真实设备张量是否支持这些 storage/CPU 观察接口仍待实际执行。如果不支持，将明确失败，不能用空记录代替。观察器也会增加数据搬运，仍需后续完整采集开关中立性对照。

## 已核对的 SDK 事实

1. `application_base.py:292` 编译调用 `trace(initialize_model_weights=False)`。因此 capture-build 阶段没有真实加载分片值，不能在那里伪造已加载权重证据。
2. `application_base.py:401` 的真实加载路径为 `builder.shard_checkpoint()`，随后 `nxd_model.initialize(weights, start_rank_tensor)`。`model_builder.py:802` 在原模型下做完整 checkpoint 预处理、cast 和 rank 分片；没有额外的 serialize_path。
3. `spmd.py:194` 的 `initialize` 在 `weight_loader` 非空时调用 `weight_loader.forward(checkpoint, False)`，否则调用 `_parallel_load`。随后全部 bucket 模型共享 `self.weights`。新观察器包裹的就是这条原路径，而不是另行计算一个假定“等价”的分片。
4. `model_builder.py:1280` 有明确 TODO：优化后推理 metaneff 的权重形状和运行时形状尚待更新。因此仅看 metaneff 的 checkpoint key/dtype/shape 不能证明实际布局相同。
5. `hlo_utils.py:567` 在 CPU WLO helper 前明确保留 `WLO on CPU doesn't match WLO on device` 的 TODO。该 helper 的 `get_wlt` 仅接受 parameter→reshape→transpose（可再 reshape）形式；不能把它当已验证独立参考。本次没有实际重现 CPU/device 不一致，也不将该注释描述为本模型已发生的故障。

## 可能闭合的严格相同路径，尚未执行

源码搜索找到已有 `audit_qwen35_weight_binding.py:82`：它使用 `audit_qwen35_serialized_runtime.RuntimeReader`，从序列化 `model/data.pkl` 读取 `weight_loader.state` 中的完整 NEFF 字节摘要、metaneff、serialization tag 与 rank 数。`LayoutTransformation` 的四项状态在该已有审核中得到明确检查。这提供了一个不导入 Torch 的候选路线，不需要依赖 CPU WLO helper。

后继可在已有独占实际任务中认证原/新 model.pt 后，逐项比较实际序列化 WLT 的完整 NEFF、metaneff、tag/rank，绑定每个键的输入/输出顺序及全部 12 桶实际权重参数接口；再结合本观察器的完整分片输入值和加载后 CPU 表示，建立**严格相同实现与绑定路径**的证据。只有实物逐项相同时才能采用，不能因源码方法名相同而填写相等。序列化代码还需确认全部桶确实接收同一份权重映射，且观测的模型就是这些已认证产物。

这条路线尚未实现 DeepSeek 专用检查，也未证明原/新布局转换实际相同。若转换或绑定不同，不能据此宣布语义不同或相同；需另做完整可验证映射或局部受控对照。HLO/NEFF 身份和相同调用路径也不替代真实冷缓存、全部 126 logits/9,072 KV、144 state、14 个原失败帧的复现与采集中立性检查。2,041 步长解码、93 个检查位置、槽位复用、quality96、matched HTTP 均保持。

本交付没有产生模型通过结论或提速倍数，不能满足当前外部 `capture_weight_layout_admission` 门槛。
