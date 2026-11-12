# LFM 首组 CPU 回归完成与后续设备准备

2026-11-12，LFM3 完整 CPU r2 仍在运行。原环境的 regression 阶段已完成 123 个批次、1,648 个案例；主任务独立读取全部报告、启动确认和父进程记录的退出结果，原科学验证器全部通过，123 个子进程均实际退出 0，且已确认退出。原环境的 extended 阶段继续执行；新环境 fixture、对照和候选计算尚未完成，不能据此宣布完整 CPU 门槛通过。

本轮没有新增合格模型或推理提速。[各模型提速总表](model-speedups-20261112.md)保持原数值与范围。[上一轮记录](qwen-r6-model-checks-lfm-r2-20261112.md)中的 Qwen r6 整轮失败、API 根异常未知及真实排空结果仍然有效。

## LFM 下一步要验证实际设备行为

源码定位确认，新 typed selector 的 `make_kernel()` 目前返回普通函数，尚未接到真实 NKI 0.5 JIT。后继应把不带 observer 的函数接到 JIT，并保留原 Torch wrapper 的形状、dtype、连续布局检查，int64 越界先规范化再转 int32，以及单行请求的原 writer 分派。模拟器专用 BF16 写回修正不得进入设备路径。

CPU 完整通过后，第一个实际编译应直接使用原 raw-device 矩阵的首图；成功后继续原设备／持久 alias 矩阵，再推进 typed B4 和 B8/B16 整模型配对。原门槛包括 raw 4 图／36 检查、B4 设备 9／43、B4 alias 12／60、B8/B16 设备 42／2,520、alias 24／192。还需补齐原 typed 契约要求的 FP32 持久 alias、int64 大值越界和生产分派检查，不能以较窄的历史结果代替。

环境接入仍需验证：旧已验收 package 固定 Python 3.10、Torch 2.8、NxDI 0.8；新 NKI 0.5 环境为 Python 3.12、Torch 2.9.1、NxDI 0.10。更正本页先前的“尚无 vLLM 源码”判断：editable finder 实际指向已存在的 `research/vllm016-source/vllm`，其中有 1,311 个 Python 文件；`vllm-0.16.0+empty` 的 `+empty` 表示构建时不带编译扩展，不表示缺少 Python 源码。见[完整更正与静态依赖核对](editable-source-and-runtime-entry-20261112.md)。后续 HTTP 仍须验证真实导入、ABI 和模型接入，并使用新的软件／源码／artifact 身份，不能放宽旧 package 身份检查或直接继承旧提速。

旧整模型流程固定 context 2048；另一路设计提出的 context 4096 目标仍保留为未完成项。原语矩阵中的 KV4096 控制、CPU 案例和旧 2048 冷缓存结果，均不能证明 LFM 整模型 4096 已通过。对应的模型图、参考轨迹、冷启动、长解码及槽位复用仍需补齐。

## Qwen：后继配置准备通过独立检查

v8 配置准备工具 v2 已经主任务完整源码审阅，30 项独立检查通过。它保留已执行 r6 的 47,959 份来源摘要、完整 121 条有序串行历史，以及原输入身份；依据原 base plan 的 8,791 个输入避免重复添加历史终态，并把旧超大 Qwen spec 留作准备证据，避免再次误放进普通 8 MiB 元数据输入。

工具必须显式接收未来 LFM r2 的真实退出、完整来源与排空记录，以及实际 idle 交接。返回的 review 仍为未通过、不可启动。测试使用真实旧 JSON 与合成未来记录，没有创建真实后继运行绑定；完整来源重建、原生认证和实时准入仍待执行。

## MiniCPM5：确认提前导入的真实路径

静态图记录了 233 个本地源码节点、697 条导入边，找到原 diagnostic 经历史审核模块到 `minicpm5_python_trace_lifecycle_v4.py:7 import ctypes` 的链。旧 outer 和 SDK worker 都在各自认证函数之前触发这条链；外层 Python 3.12 与 worker Python 3.10 必须分别认证。

新增入口 gate 保留原 canonical 模块、完整历史来源和原九项认证，要求独立认证先于 ctypes／历史导入，再执行原完整认证并核对整个 source map 相同。主任务独立复跑 24 项模拟检查通过，包括实际导入拒绝器与 233 份已记录源码摘要复核。它仍只是入口接线：真正的 bootstrap 描述符／限额检查、原生来源和实际库提供方认证、最终 outer 与 worker 后继尚未完成。先前测试误导入 ctypes 的偏差记录继续保留。

## Spark：补上请求顺序与物理行排列的诊断源码

后继 core v3 保留原 534 步缓存轨迹和 216 项 QKV 诊断，新增调用固定 SDK `_reordered`、`_sort_inputs` 和返回的 `restore` 的代码，拟执行 2,108 组 host 方法检查。原 B1/B2/B3/B4 四类快照各执行四行的 24 种排列，两种缓存、两 rank 合计新增 384 次实际缓存函数调用。

主任务完整审阅后独立复跑 13 项标准库／list-mock 检查通过。运行源码要求使用实际导入的方法，不以 AST 提取替代；本轮测试仍使用模拟对象，没有执行真实 Torch、SDK 或模型。最小 receiver 也不覆盖完整 serving forward 或 SDK padding。权重下载、完整原生认证、实际峰值内存／时间和 4,844 条输出记录的预算仍待固定并准入。

[本轮保存证据与源码](evidence/lfm-regression-and-native-preparation-20261112/manifest.json) · [统一分支](multi-model.md)
