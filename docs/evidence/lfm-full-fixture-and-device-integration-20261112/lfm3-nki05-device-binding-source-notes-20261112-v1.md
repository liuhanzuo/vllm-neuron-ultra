# LFM NKI 0.5 device integration — source delivery

这次完成真实设备集成的源码准备，并通过标准库 AST / mock 检查。没有执行 NKI、Torch、NumPy、编译器、设备或模型；没有操作服务、锁、权重、运行中的 r2、飞书或 Git。实际设备准入、资源预算及后续完整模型测试仍待 root 实现。

## 最终选择的源码

- `slot_cache_nki05_typed_jit_v1.py`：延迟 native import，实际调用公开 `nki.jit(function)`，不传已弃用的 platform_target 参数。要求 Trn1、LNC1、非模拟器及精确四个软件版本。
- `slot_cache_nki05_typed_torch_v1.py`：保留旧 wrapper 的全部 payload/dtype/shape/device/contiguous 约束。int64 槽位先做有效性判断，将非法值变成 -1，然后才转 int32；payload 只 reshape。
- `lfm3_nki05_typed_dispatch_v1.py`：实际生产分派，一行仍调用 original_store_slots，多行才调用 typed writer。primitive 矩阵的一行案例继续测试 typed primitive 本身；生产分派另有实际 trace 控制。
- `lfm3_nki05_device_context_v1.py`：抽象、尚未绑定的未来运行接口。必须由实际 owner 认证完整 CPU 三遍各 3716 案例 / 291 批次和独立 control、全部报告/退出/警告收据、终态及 idle/history/资源/软件/source。此接口没有现成 runnable context；只有 summary 检查不构成准入。
- `lfm3_nki05_device_oracle_v1.py`：复制原 package 顺序 Torch oracle，仅改成延迟 import，AST 已核对原可执行函数体一致。
- `lfm3_nki05_device_matrix_v1.py`：准备时复制原纯矩阵函数；运行时不提取、改写、exec 或 compile AST，调用者不能通过 protocol 替换科学矩阵。
- `lfm3_nki05_raw_device_leaf_v1.py`、`lfm3_nki05_regression_device_leaf_v1.py`、`lfm3_nki05_regression_alias_leaf_v1.py`、`lfm3_nki05_extended_device_leaf_v2.py`：完整真实 scientific bodies，真实 `torch_neuronx.trace`、保存 artifact、逐字节比较、持久 state / MUST_ALIAS。
- `lfm3_nki05_device_supplements_v2.py`：新增实际 FP32 persistent alias 与生产分派 / int64 overflow device 控制源码。
- `check_lfm3_nki05_device_binding_source_v3.py`：最终 47 项 stdlib / mock 检查，内置 native import blocker，CPU 7、Nice 19、256 MiB address space、60 CPU 秒、2 MiB 单文件上限。

新旧同名实验文件均未覆盖；已执行过检查的所有生产 v1 源字节仍与 pretest v2 manifest 一致。extended v2 只补 artifact receipt 的实际 shapes / dtype / rows / variant；supplements v2 只改 helper import。最终测试 v3 补一项收据 shape 身份检查。

## 完整保留的科学矩阵

|阶段|图数|检查数|保留内容|
|---|---:|---:|---|
|raw device|4|36|BF16 / FP32、宽 7168 / 65536、全 65536 个 BF16 word、NaN payload、Inf、signed zero、subnormal、输入不变|
|regression device|9|43|LFM 128–2048 KV、conv、单行、RNJ 4096 KV、FP32 SSM|
|regression alias|12|60|完整连续调用、reference / candidate、attention 与 cache 写回同图、实际 persistent state|
|extended device|42|2520|B8 / B16、行数 1 / 4 / capacity、selector / conv / KV、全部原 mapping|
|extended alias|24|192|B8 / B16、reference / candidate、8 次 lifecycle、bindings / inputs / persistent bytes|
|新增 FP32 persistent alias|6|222|B4 / B8 / B16，每组 reference / candidate，真实 SSM shape、37 连续调用|
|新增 wrapper / production dispatch|16|418|B4 / B8 / B16、BF16 / FP32、1 / 4 / capacity 行、超大 int64 正负值与实际单行 original 分派|

原 2068 扩展 CPU pure matrix 也精确保留，仅用于比较，不在设备 leaf 重复运行。原 1648 + 2068 三遍完整 CPU 矩阵必须由 root 独立认证。科学体检查允许的差异仅为固定 device 分支选择、writer / oracle import、输出路径与 artifact 收据；原 trace 参数、完整循环、oracle、比较、alias 状态推进未缩减。

补充 raw fixtures 每一行使用不同 bit patterns，避免相同行数据掩盖 permutation / duplicate-last-writer 错误。FP32 alias 实际检查写前状态、写后输出和持久状态逐字节相等，并检查 input / binding 不变。其数据包含 NaN / Inf 等原始位模式，不能用浮点 equal 替代 raw byte 对比。

## Artifact 与运行界面

每个新 artifact 保存后必须调用 context.artifact，检查新 schema `lfm3-nki05-typed-device-artifact-v1`、variant `typed_selector_nki05_v1`、精确 NKI 0.5 / Torch 2.9 软件版本、source map、phase、actual shape / dtype / reference-or-candidate case、path 与 artifact digest。只允许新 context 的 output root 内文件；旧 package identity / artifacts / cache_writer 字节已再次与 route v2 pins 核对。

实际 root context 仍需实现认证、fresh exclusive output directories、所有阶段完整 gate 验证、原 warning / owner / history / lock 路径、有限 compiler 峰值 / artifact / 时间预算、bounded atomic JSON IO 和 streaming artifact hash。context 中预算保持 None；不得把本次 source tests 当成实际准入。保存 / 失败 / drain 的最终父子流程也由未来 serial worker 负责；这些 leaves 没有 CLI 或自动启动路径。

未来第一张真实集成图就是 raw 原矩阵 BF16 `(4,7168)`；通过后继续同一完整矩阵，不追加额外 smoke gate。原 XLA trace / states / privateuseone 断言保持；如新 SDK 实际桥接或设备注册不同，真实失败须保留后按证据处理，当前 source/mock 无法证明 SDK 兼容。

## 完整模型与 4096 目标仍待执行

CPU 模拟器 BF16 input writeback workaround 不进入任何设备 / 服务源码。原模型 B4 / B8 / B16 numeric、冷缓存、1529-step long decode、槽位复用、96 quality、704 requests / 189952 output tokens、同负载并发 1 / 4 与 capacity 8 / 16 对照全部保留。

ctx4096 另列 pending：原 ctx2048 oracle / artifacts 和 RNJ 4096 四 heads 不构成 LFM 4096 认证。需新 HF full-causal + incremental oracle、新实际 LFM width 1048576 CPU / device / alias 形状、新完整 cold / >2048 long / 4095 边界 / reuse / HTTP 服务图身份；route v2 的详细计划保持不变。当前仅保留目标与接口 phase 名，没有实现或执行这些新的 4096 primitive / full-model gates。

## 已执行的验证与保留失败

首轮测试 v1：46 项，45 通过；原 oracle 函数 docstring 被 AST 比较计入，而新 oracle 使用模块 docstring。终端 b6a0bc，exit 1。仅测试 v2 忽略原 docstring，生产源未改：46 / 46，终端 5c1a82，exit 0，0.418 秒。

最终补充 artifact 完整 shape 身份后的测试 v3：47 / 47，终端 3a5f8e，exit 0，0.548 秒。fake Torch / JIT 只验证 host control 与调用契约，不是实际 tensor / kernel / compiler / hardware 结果。所有新 device / alias / full-model 资格与新提速结论仍为 false。
