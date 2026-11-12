# Qwen3.5 解码缓存写入候选（尚未设备验收）

这个独立实验分支将 NKI 条件复制接入 Qwen3.5-2B 的解码缓存写入。默认模式 `torch_where` 保持原实现；显式设置 `NEURON_QWEN35_CACHE_WRITER=nki_predicated_decode_v1` 后，TP 分片完成的 decode 图使用候选。Prefill 和独立 CPU 参考保持原 writer。

接入包含卷积状态、FP32 recurrent 状态以及 Attention K/V。新模块只接受原 TP2/B4/context4096 的固定几何和 int32 槽位，保留原数学、旧状态读取、KV 前缀切片、尾部拼接及 FP32 缓存。不会根据示例 tensor 是否在 CPU 上静默回退。

编译产物身份新增 writer 模式、阶段、版本、形状与 helper/kernel 哈希；父进程和 trace worker 必须一致。旧图或另一模式的图会被拒绝。两臂需要分别使用全新 artifact、compiler cache、workdir，各构建六个 prefill 和六个 decode 图。两臂必须固定同一 CPU sampler，才能把差异归因于 writer。

20 项纯源码/mock 检查由作者和主任务分别执行通过，补丁已在独立 checkout 应用并逐文件核对。原 NKI kernel 与此前设备矩阵使用的版本字节一致。

**本分支尚未完成设备编译或整模型验证，没有可报告的整模型提速倍数。** 原矩阵的 1.671×～3.965× 比较对象是整数 bitcast 诊断参考，不能作为本分支相对模型基线的倍数。真实 int32 入口、KV 切片 stride/alias、TP2 多状态图仍需验证。数值、六冷 bucket、1,541 步长解码、槽位复用、203 向量、quality96、匹配 HTTP 和退出后独立审计的全部要求保持。

当前 sampler 独立包工作在原分支继续。此分支不修改那个已绑定实验，也不复用它的单 artifact 对照结论。附带 `pair_contract.py` 只检查未来两臂的元数据，不是已可执行的模型验证 runner。

[完整源码与检查证据](evidence/qwen-decode-writer-20261112/manifest.json) · [接入边界和待完成验证](evidence/qwen-decode-writer-20261112/REVIEW.md)
