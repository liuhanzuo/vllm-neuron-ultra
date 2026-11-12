# Qwen slot BF16 select：源码/HLO定位与有限后继

原 reference 的保存 HLO 已把故障范围缩小到**BF16选择图及其后续编译/执行路径**，但没有证实具体compiler pass。允许读取的 `graph.hlo` 为4084 B，SHA `be083f7f728dcdb6a744fbacadf840c647ab06b5eff0b1355004cdc573c67d5e`。独立按固定SDK descriptor解析为1个computation、61个instruction；其中4个BF16 select（ID18/32/46/60），没有convert、bitcast-convert或浮点算术。

`qwen35_text_port.py:47` 的原函数逐行执行四次 `torch.where`，保存图中的四层BF16 select保留BF16分支数据，slot条件是S64 signed equality。因此“前端显式BF16→FP32→BF16转换造成规范化”不符合这份HLO。SDK `torch_neuronx/xla_impl/trace.py:240` 的 `setup_compiler_dirs` 保存该图，后面的 `hlo_compile` 才调用neuronx-cc；它不是compiler最终机器IR。不能据此断言select最终使用哪条指令、哪项优化或哪一个pass。

已完成的r2六次调用不重跑：reference两profile分别139/155字节变化，完整规则是负零变正零、全部NaN变正7fc0；candidate和真实NKI DMA两profile全部字节保持。共享的无条件host/device传输规范化不能独自解释臂间差异；仍不能排除图相关传输/运行交互。

原flags为 `--target=trn1 --auto-cast=none --model-type=transformer -O2 --lnc=1`，另保留原tensorizer选项。未发现这些显式参数中存在fast-math开关。固定SDK Python的 `--xla_cpu_enable_fast_math=false` 是CPU XLA默认值，不是neuronx-cc数学行为契约；`--auto-cast=none` 也不证明select逐位复制。在本次允许的Python源码范围内没有建立可用的Neuron fast-math关闭开关，更不能声称-O2已被证明是原因。下一项先保持原flags，只改变被研究的数据选择路径。

## 已准备的独立原型

[`qwen35_slot_select_bitcast_proposal_v1.py`](qwen35_slot_select_bitcast_proposal_v1.py) 导入时无SDK/native动作，只有未来原owner在native-import边界调用 `make_modules()` 才注册两个局部XLA op并实例化三个模块。它不编译、不启动、不获取锁，不修改冻结 `store_slots`，也不把新图称为已修复的原baseline。

| 新图 | 数据路径 | 鉴别目标 |
|---|---|---|
| `one_bf16_select` | BF16接口，原row0逻辑的一次torch.where | 单次select是否已经足以出现规范化 |
| `internal_s16_select` | BF16输入→BitcastConvert(S16)→原四次store_slots→BitcastConvert(BF16) | 仅将选择域换成整数、保留BF16接口 |
| `external_s16_select` | 原样raw字节作为S16输入→原四次store_slots→S16输出 | 去掉图内bitcast及BF16接口，独立控制整数选择本身 |

SDK源码给出了可用接线依据：`torch_neuronx/xla_impl/base.py:15` 的 `xla_call` 委托 `xla_op_registry.register`；`torch_xla/core/xla_builder.py:434` 的 `Op.bitcast` 明确生成 `BitcastConvert`，而相邻 `Op.cast` 生成数值 `Convert`。BF16/S16的Torch↔XLA及metaneff类型映射存在。原型使用这一接线，不依赖未证实的trace内 `tensor.view(dtype)` 支持。**这些只是前端API源码依据，尚未证明Trn1编译或运行位型正确。**

## 下一项有限实验

固定三个新图、原有两份raw profile、两种mapping，共12次独立调用，零timing：

- `idle = [-1,-1,-1,-1]`：完整输出等于cache。
- `one_active = [3,-1,-1,-1]`：cache槽0/1/2不变，槽3逐字节等于update行0。

这两种mapping使三图共享同一完整raw oracle，同时覆盖所选与未选分支；不复跑已经完成的r2六次。cache/updates依然为98,304 B，slots为int64；外部S16臂只把同一raw存储解释成int16，不能使用数值`.to(int16)`。完整12条schedule及 `byte_oracle` 已在原型内固定。

未来独立successor应复用原lifecycle、SDKEnvironment、固定core0、CPU/内存/时间/NVMe限制及drain，保持最多一个live graph。保存每个实际HLO、forward_v2代码路径、模型完整hash、全量输入/输出和所有差异u32 offsets；raw+offset预算8 MiB（最坏6,291,584 B），原整体资源上限不改。每次输出mismatch继续记录后续调用；真实异常保留现场并中止，不重试、不换实现。

运行前先检查实际发给compiler的HLO：单select臂必须是一个BF16 select；内部S16臂须沿参数及called computation数据流证明两个BF16输入均经BitcastConvert成为S16、四次选择在S16域、输出经BitcastConvert返回BF16；外部S16臂须为S16输入/输出和四次S16 select。不能用总字面bitcast计数代替数据流，因为两个调用可能共享同一callee。若内部图被重写回BF16 select、发生数值Convert、退回host或不可编译，需记录为该干预未建立；不得默默fallback。

结果解释预先限定：

- 单select已规范化：一层BF16选择路径已足以复现，无须四层链才触发；仍未定位pass。
- 单select保留、原四层失败：链式选择/图布局相关差异仍需区分。
- 内部与外部S16均保留：支持整数选择是此诊断范围内可行的独立候选，不证明fast-math是原因。
- 外部S16保留而内部失败：应调查bitcast lowering或BF16图接口交互。
- 两个S16图失败或编译失败：候选未成立；保存证据，不能宣布修复。

即使12次全部符合oracle，新实现仍必须另行通过原八layout、standalone/persistent_alias、625 mappings、raw+finite、old-read/actual-state/输入不变及原完整timing协议；特殊值门槛不减少，原reference失败状态不改变。

## 可核验交付

- [机器可读审阅](qwen35-slot-select-hlo-mechanism-source-review-20261112-v1.json)：完整61条HLO摘要、18份来源SHA、固定12-call schedule与边界。SHA `7336600d22f3f9b9235bb26fd08ea8af2502bb3c62cb198e91fc6d78b091829d`。
- [可重算的stdlib检查器](inspect_qwen35_slot_select_hlo_v1.py)：只读指定HLO metadata、SDK Python/descriptor和已有审阅报告。
- [source/mock tests](check_qwen35_slot_select_bitcast_proposal_v1.py)：pinned Python3.10、native import阻断，5项通过，tool chunk `822bde` / exit0；覆盖实际原型12次mock调用及完整98304-byte oracle、全部65536种16位编码的host存储往返、冻结writer与实际SDK bitcast/convert接线。

本轮没有native、编译、权重/model.pt/NEFF读取、锁/History/current binding扫描；不是root acceptance、primitive/整模型资格或性能结论。
