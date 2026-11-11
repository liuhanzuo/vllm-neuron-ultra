# LFM3 紧凑模拟器写回适配器与原输入指纹：源码接入说明

本次交付两个惰性叶子模块和纯检查，没有运行 native 库、生成实际 fixture、运行模拟器、编译器、芯片或整模型，也没有新增生命周期框架、当前运行 spec、排队项、历史解析、进程／服务／锁操作或发布。原已执行源码和已安装 SDK 文件没有修改。根流程需要先独立审阅这两个叶子，再以现有串行机制安排实际步骤。

`lfm3_nki05_compact_writeback_v1.py` 仅在明确调用 `CompactWriteback.install(module, admit)` 后替换进程内 `_tensor_to_numpy`，且只改变精确输入写回调用位置的 BF16 `precise` 参数。`lfm3_typed_fixture_ledger_leaf_v1.py` 校验原输入函数和完整用例定义，每次只返回一个原始 fixture 的四个完整 SHA、dtype、shape、字节数及供调用者检查的原数组。两者均无 CLI、native import、子进程或持久化接口。

## 必须保持的原范围

| 阶段 | 用例 | 批次 | 批次约束 |
|---|---:|---:|---|
| regression_cpu | 1,648 | 123 | 小宽度最多 16 例，宽度至少 524,288 独立一批 |
| extended_cpu | 2,068 | 168 | 同上 |
| 合计 | 3,716 | 291 | 原顺序、原 seed、全部映射和真实宽度 |

完整矩阵 SHA 固定在叶子中，序列化明确为 sorted keys、compact separators；它们不是旧协议格式 SHA 的替代品。纯检查从原冻结定义仅提取 stdlib 元数据函数，证明完整矩阵相同，并逐项比较 `original_batches` 与原 `lfm3_typed_writer_cpu_pilot_protocol_v3.py:batches`，不是只核对总数。实际调用者仍需把叶子批次与自己的原计划逐项核对。

保持每批 1,800 秒、每阶段 14,400 秒、整 worker 30,000 秒、MemoryMax 10 GiB / high 8 GiB / no swap、TasksMax 32、12 GiB AS、256 KiB 单批报告，以及完整输入不变、逐字节输出、dtype/shape 和 stop-first。继续使用当前更严格的根盘 28 GiB、NVMe 30 GiB 余量要求；不恢复旧 NVMe 5 GiB 门槛。fixture 生成与新环境校验都走现有串行预算，不能把新工具当作模型运行期间可并行的轻量任务。

## 先产生原环境 fixture 指纹

在原 NumPy 1.26.4 / ml_dtypes 0.5.4 环境的独占、有限阶段中：

1. 现有调用者完成自己的运行环境、版本、源文件、资源与串行所有权校验。`admit` 是这个已经存在的验证回调；叶子要求其返回严格 `True`，不生成准入凭据。`versions` 必须来自本次已核验的真实环境；叶子记录／检查允许版本值，不独立扫描 native 包或证明二进制身份。
2. 调用者提供原 `run_lfm3_typed_writer_cpu_pilot_v3.py:numpy_inputs` 和两份完整 `p.cases(phase)`。叶子核对原文件 SHA、函数 code/defaults/closure、`p.ROOT`、`ast` 绑定，以及扩展原生成源码 SHA；没有重写随机数、special bits 或 seed 生成。
3. 构造 `FixtureLedger(admit, numpy_inputs, matrices, versions)`。对原 291 个 `[begin,end]` 依次调用 `.batch(phase, begin, end)`。每次 yield 为 `(record, operands, expected)`；只持久化 `record`，不保存数组。每例消费后释放原数组与参考输出。
4. `record` 含完整 `case` 和 cache／updates／slots／expected 四组 `dtype, shape, bytes, sha256`。`expected` 使用原 `cache.copy()` 和最后有效写入行赋值循环；生成后再验证三份输入没有变化。每例 metadata 至多 8 KiB，小批全部 metadata 至多 256 KiB；实际调用者的总报告仍须连同其他字段一起满足原 256 KiB 上限。
5. 原运行若中断，只保留实际完成前缀，记录真实失败并停止。不能给未执行用例填指纹、只保存 seed、丢弃 NaN payload 或把新环境结果写成原环境 ledger。

实际生成前没有新的 fixture 通过证据。本次纯测试中的 `Array` / `Tensor` 是小型 Python bytes 模拟对象，既非 NumPy 数值实现，也非模拟器结果。

## 新环境逐例比对及原始／候选／恢复控制

新环境使用 NumPy 2.5.4 / ml_dtypes 0.6.0，并由调用者继续完成原 NKI 0.5.0、编译器和相关 Python 源文件的完整运行身份核验。相同 seed 不是字节相同的证明。

新环境也通过 `FixtureLedger` 调用同一个原函数，先对每个 `record` 调用 `verify_same_fixture(old_record, new_record)`。此比较要求 case 内容和四组完整元数据完全相等，运行版本元数据留在外层，以便允许明确记录的新旧环境比较。任一差异均停止，不能替换原 oracle。

完整 CPU 模拟前，必须先用根流程独立审阅后的 `slot_cache_nki05_typed_selector_v1.py:make_kernel()`，实际执行原 v5 首例的三组控制，顺序为 original、candidate、restored_original。每组重新生成数据，并匹配固定的原四个 SHA：extended_cpu 的 case 0、capacity 8、width 256、BF16、slots `[0]`。三个 probe 都是独立的新对象与安装／恢复作用域。

每组调用顺序如下；这是接入说明，不是已执行记录：

```python
probe = CompactWriteback(mode)
probe.install(already_authenticated_simulator_module, admit)
try:
    probe.begin_case(phase, case_id, operands)
    actual = already_authenticated_nki.simulate(observer_free_kernel)(*operands)
    extraction = probe.end_case()
    # 调用者完成原 dtype/shape/全部输出字节/全部输入 SHA 检查，保存实际记录。
finally:
    probe.restore()
    probe_receipt = probe.report()
```

完整控制记录字段由 `validate_small_controls` 明确定义：`name, fixture, shape_equal, dtype_equal, bitwise_equal, actual_sha256, inputs_unchanged, input_after_sha256, extraction, probe`。原始与恢复组应有精确输出，但两个 BF16 主机输入发生变化，slots 不变；候选组的全部输入保持原四指纹。三组输出均须等于原 expected SHA；原始与恢复组输入变化摘要及提取事件必须完全相同。

调用者必须保存三组真实记录、真实执行状态与恢复凭据，再执行 `validate_small_controls(arms)`，并把返回的完整控制 SHA 与实际证据绑定到后续原串行调用者。这个函数是严格的记录校验器，不能单凭手写 JSON 证明实际执行。**没有通过该检查的真实控制，根流程不能准入全套候选模拟。** 叶子本身没有全套运行入口，也不会自行排队或替调用者决定准入。此前 r5 的原始失败、恢复失败和实际非零退出保持原样。

三组控制不计入 3,716。完整批次另用新 probe，不能把额外首例塞进原 16 例的同一个作用域；如果选择每批重复三组控制，也必须为控制使用独立 probe 且保持原资源总上限。

## 紧凑适配器具体约束

`authenticate_simulator` 只接受固定 `/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/nki/_simulator.py` 及 SHA `3c28f10492ef8c6c5413acbd811c059586f7ff16d3ef795430e794c57ed8f119`。它比较实际 `_tensor_to_numpy`、`simulate_kernel` 的 code、globals、defaults、kwdefaults 和 closure，与仅编译解析、未执行的固定原源码一致。输入提取位置为实际 `simulate_kernel` code 对象的第 215 行；输出为实际嵌套 `_validate_and_convert` code 对象的第 252 行。行号由固定 AST 推导，不能只按函数名称匹配。

每例只允许按 cache、updates、slots、output 的顺序发生四次提取。输入位置必须省略 `precise`，局部 `tensor` 和 `orig_arr` 必须分别是实际张量及预登记原输入对象；输出必须来自唯一精确返回位置并显式 `precise=True`。未知调用位置、额外调用、漏调用、输入错序、dtype／shape／storage 改变、只读输入、callable/defaults 被替换都失败。

BF16 候选仅给前两个输入提取传 `precise=True`，slots 和输出没有替代；FP32 companion 完全不替代。原始组保留旧提取的 BF16 → FP32 返回 dtype。适配器从不读取 `.tobytes()`，不保留 tensor backing、counterfactual 克隆或逐 tile 事件，不改变输入写保护，不把参考输出喂给模拟器。每例固定四条小元数据；原一批最多 16 例。单 operand 的原 128 MiB 上限没有提高。

每次调用不论成功或异常均由调用者在 `finally` 中恢复；出现身份篡改时适配器仍尝试恢复原 extractor 对象，同时报错。未完成 case 的至多四条事件保留为 `incomplete_case`，不计入完成数。probe 的完成数只表示提取控制完整，不能替代外层真实输出／输入数值检查，更不能替代真实子进程退出。严禁在异常后复用已失败 probe 继续后续用例。

## 完整候选调用与最终汇总

原调用者对同一批的每例先验证旧／新 fixture 完全相同，再用紧凑适配器包住无观测内核的 `nki.simulate`。保留原 `check_case` 的实际 `.tobytes() == expected.tobytes()`、dtype、shape、三份输入完整 SHA 比对及有限 first16 mismatch 报告。输入哈希一致性不能代替输出全部字节比较。第一项失败立即停止，不丢弃 inactive／冲突用例、不扩大阈值、不减少长度。

外层沿用原独立审核机制，逐一核对 291 个真实批次退出和完整有序的 3,716 例，而不是把本工具的记录字段当作终态。此后 trace/XLA、raw device、regression/extended device、所有 alias、BF16 原权重整模型、cold4096、长解码 1,529 步、槽位复用、quality96 和匹配 HTTP 都仍独立待验。现有 B4 已合格结果保留；B8/B16 没有因本次源码交付获得资格，没有新增提速数据。

## 本次验证

`check_lfm3_compact_writeback_and_fixture_leaf_v1.py` 最终 20 项通过：实际工具 `883ad4 / exit 0`，测试报告 0.671 秒。检查包括惰性导入、真实 SDK 源码／callable 认证但不执行、严格四调用及错位置／错 dtype／错 shape／错顺序拒绝、失败后恢复和不完整记录、原完整矩阵及批次逐项一致、原生成函数 code 绑定、fake bytes 上的最后有效行／NaN 和 signed-zero 保护、原／候选／恢复控制记录及十种变异拒绝。全套结束还检查没有加载 NumPy、Torch、NKI、neuronxcc 或 ml_dtypes。

前一版 20 项也通过（`346cb4 / exit 0`，0.538 秒）；随后新增失败 case 的有限 metadata 保留和每批重新验证完整矩阵，因有源码变化而执行最终检查。没有反复运行未变化的 native 测试。所有 pure/mock 结果只能证明接入代码约束，不能证明原 fixture、真实模拟器或硬件正确。
