# DeepSeek：完整序列化权重转换与十二桶绑定检查器

新增 `../deepseek_weight_serialized_descriptor_v1.py`，复用既有 `audit_qwen35_serialized_runtime.RuntimeReader` 和其惰性描述符解析器。没有调用原 Qwen/LFM 固定目录的 main，也没有修改这些历史源码。本次只创建工具、读取源码并使用内存合成 ZIP/pickle/protobuf 做测试，**没有读取任何真实 model.pt、HLO/NEFF 或权重**。

16 项源码／模拟检查通过，实际 `870c39 / exit 0`。首次测试 `93a843 / exit 1` 的 4 个断言失败、6 个错误来自合成类名使用 `__torch__.Executor`，其 module 为 `__torch__`，不满足原安全解析器要求的 `__torch__.` 前缀。只修正测试夹具的类名到 `__torch__.synthetic.*`，没有放宽运行时 global 过滤。

## 实际调用合同

```python
from deepseek_weight_serialized_descriptor_v1 import read_archive, compare_descriptors

descriptor = read_archive(
    {"path": exact_model_pt_path, "sha256": externally_bound_sha256,
     "identity": externally_bound_full_stat_identity},
    authenticated_serial_guard,
    {"archive_bytes": reviewed_cap, "pickle_bytes": reviewed_cap,
     "members": reviewed_cap, "code_bytes": reviewed_cap},
)
```

上面均是调用合同占位，不是可运行请求。外层必须先提供真实认证的模型产物、完整源码和资源／进程／串行执行绑定。`identity` 精确包含 device、inode、size、mtime_ns、ctime_ns、mode、nlink。四项容量都是外部正整数预算，工具没有新增或扩大实验容量。读前使用普通规范路径、`O_NOFOLLOW`、真实 fd/stat，先读取**整个归档**验证完整 SHA256，再在同一 fd 上解析。结束再次比较 fd 和路径身份；所有大读取都有 guard。

完整归档哈希会经过归档中可能包含的 tensor storage 字节，但工具不打开或反序列化 tensor storage 成员。它只解析 `data.pkl` 的惰性描述符，保存序列化代码完整字节的 SHA。没有 Torch、protobuf SDK、模型加载、设备调用、锁管理或结果发布。父进程须在真实读取工具退出、归属进程排空后独立封存返回值，当前组件不自证完成或调用方身份。

原解析器按 64 KiB 块读取大型嵌入 NEFF，并保存完整 graph-definition 的长度和 SHA。新入口验证整个归档和每个脚本成员的完整分母，拒绝重复成员、超限成员数／pickle／代码总量、未知 pickle global、EXT 注册表绕过和 STOP 后尾随字节。当前仅接受固定 SDK 已有路线使用的 C++ protocol 2；不同序列化格式明确失败，不能自动放宽。

## 返回的具体证据

- WLT 必须是实际 `LayoutTransformation` 的完整四字段状态：NEFF 全字节 SHA/长度、metaneff 全字节 SHA/长度及完整接口、serialization tag、TP2 rank 数；未启用 WLT 则明确记录 `None`。输入／输出必须覆盖相同全部键并保留原顺序。
- 从 `input_shape_map` 和实际 `SPMDModel` 五字段状态检查全部 **CTE B1/TKG B4 × 六个原桶（128、256、512、1024、2048、4096）**。记录每个图的 NEFF/metaneff 身份、原 tag/local/world、所有原始输入形状和每个权重的实际 input index、名称、键、dtype、shape 及额外元数据，不能排序后抹去实际位置差异。
- 每图保留并检查全部 **72 个 KV 名称和 72 个 alias 目标**，alias 对应输出的完整 shape/dtype 必须匹配。12 图按名字匹配的权重接口必须完整一致；WLT 存在时其完整键集合必须覆盖这些图。
- 保留归档中所有 `.py` 序列化代码的完整长度／SHA。不是只检索一个 `initialize` 字符串，也不借新源码替代实际保存的运行代码。

`compare_descriptors(off, on)` 分别报告完整 WLT、12 图逐位置权重接口、所有保存代码是否精确相同，并返回三者合取的 `strict_saved_components_equal`。完整代码可能因捕获额外输出而不同，即使这种差异合法，也只能返回不相同；不能删除差异后宣称完全相同。NEFF 或 metaneff 不同同样不自动说明数值不等价，只说明这条严格相同路径证据未成立。

## 仍不证明什么

该工具比较的是**实际保存的转换程序和声明的权重接口身份**。它不解析 NEFF 的内部调度，不从 metaneff 未更新的 shape 推断优化布局，不采用 SDK 已注明 CPU/device WLO 不一致的 CPU helper，不观察真实加载值，也未证明 C++ runtime 使用所有这些绑定时的语义。

即使三项完全相同，输出仍保留 `loaded_values_observed=False`、`graph_weight_ABI_semantics_proven=False`、`layout_semantics_proven=False`、`inference_admitted=False`、`full_model_qualified=False`、`new_speedup_claim=False`。原始 HLO/编译配置与实际序列化对象之间的完整对应还需要外层证据；这里没有假装产物中包含原始完整 HLO。

和已接受的 `deepseek_weight_load_observer_v2` 组合后，未来可进一步检查真实分片输入值、真实加载后 CPU 表示，以及它们绑定到哪些精确序列化对象。但还须实现 fresh-process 检查后的真正加载边界接线、父进程独立封存与放行、真实 12 桶绑定的语义验证。这两份工具当前都不能满足 `capture_weight_layout_admission`，也不能先花费前五阶段后再发现此门槛未闭合。

原 native42/device84、126 logits、9,072 KV、144 state、14 失败帧复现／采集中立性、2,041 步长解码／93 检查位置、槽位复用、quality96 和 matched HTTP 仍全部保留。没有真实模型通过或提速结论。
