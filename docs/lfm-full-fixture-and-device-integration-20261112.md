# LFM 全套基准数据完成与三路原生接入

2026-11-12，LFM3 完整 CPU r2 的原环境数据已全部完成：291 个批次、3,716 个案例，包括 regression 1,648 例和 extended 2,068 例。主任务独立重新读取每份报告，调用原验证器，并核对启动确认、PID/start/argv、父进程记录的实际退出码和进程结束状态；全部通过，291 个子进程均实际退出 0。

r2 仍持有原串行锁，正在继续新环境 fixture，随后才执行三路控制和候选算子的完整对照。原环境数据完成不等于候选算子通过，更不等于设备、整模型或性能验证完成。本轮没有新增合格模型或提速倍数；[各模型提速总表](model-speedups-20261112.md)保留原数值、基线与适用范围。

## LFM：真实 JIT 接线和完整设备测试源码已补齐

新实现将 observer-free typed selector 接到 `nki.jit`，固定 Trn1/LNC1 与新软件身份；Torch wrapper 保留原形状、dtype、连续布局检查，以及 int64 越界先规范化再缩窄的顺序。生产分派保留单行请求的原 writer。模拟器专用写回修正没有进入设备代码。

完整的设备与持久 alias 科学循环已写成独立后继源码，保留原检查：raw 4 图／36 项、B4 device 9／43、B4 alias 12／60、B8/B16 device 42／2,520、alias 24／192。另补 FP32 持久 alias 6 图／222 项，以及生产分派与大 int64 越界控制 16 图／418 项。后两组是待执行源码中的计数。新 artifact 记录实际形状、dtype、reference/candidate、软件和源码身份，不继承旧环境的验证资格。

主任务完整审阅并独立运行最终 47 项标准库／模拟检查通过，其中包含原科学体的静态等价检查；运行源码没有动态提取或重写科学函数。真实 NKI／Torch 编译尚未执行，具体运行上下文、完整 r2 成功及退出记录、空闲交接和编译资源预算仍待绑定。4096 上下文目标继续保留，旧 2048 整模型结果不能替代它。

## MiniCPM5：修复计时边界，补上实际认证逻辑

新的 bootstrap 实现原串行描述符和限额检查，并将同一绝对截止时间延续到原完整 guard。审阅发现的“同一 handler 重启计时器”缺口已修复；超时、源码导入、加载库检查或后续重验失败都会保留失败状态，不能捕获异常后恢复运行。

认证实现会核对完整文件字节和稳定身份、保存的上游 archive／签名包来源、实际 ELF 依赖及映射身份，并要求六个 FFI 函数的正向定义证据。后续 ctypes／历史源码导入强制从已核对源码执行。主任务独立运行描述符 19 项和认证器 26 项模拟检查，共 45 项通过。

这些检查使用合成 ELF、模拟系统接口和保存 JSON，没有认证当前进程的真实 native 闭包。bootstrap 自身的执行源码／旧 pyc 入口、外层 Python 3.12 与 SDK worker Python 3.10 各自的完整接入仍待实现。`-B` 只禁止写缓存，不能单独证明没有读旧 pyc。

## MiniCPM5 的 CUDA 依赖：已确认安装情况，尚未证明失败根因

只读版本源码与包元数据确认，现有环境是 Torch 2.8.0+cu128、torchvision 0.23.0+cu128 和 torchaudio 2.8.0+cu128。保存日志中的失败发生于 Torch 导入，消息为 `libcusparseLt.so.0: failed to map segment from shared object`，尚未执行模型。这个环境即使面向 Trainium，也会承担所安装 CUDA wheel 的导入依赖。

该诊断限制虚拟地址空间为 2 GiB，保存 VmPeak 约 1,659.80 MiB，失败后 RSS 约 196.68 MiB；它们不是失败瞬间的映射记录。包 RECORD 中 libcusparseLt 文件约 431.04 MiB、libtorch_cuda 约 968.33 MiB，也不能直接当作映射跨度。现有异常没有 errno，因此 CUDA 导入负担与地址空间压力是待验证解释，不能写成已经证实的 OOM。

可以准备同版本 CPU wheel 的独立环境候选，先在相同限额下验证导入，再验证 Neuron ABI 和全部原科学门槛；目前未安装或替换环境。当前 upstream 的新版 SDK／平台说明与现有 LTS 栈有版本差异，这本身也不是 repo 缺陷。[完整原因分析与证据边界](evidence/lfm-full-fixture-and-device-integration-20261112/minicpm5-sdk-cuda-mapping-source-analysis-20261112-v1.md)。

## Spark：provider 与 worker 已实现，完整运行入口仍待接入

新增 provider 实际调用官方模块、SDK QKV sharder 和 vLLM-Neuron loader；worker 核对原锁、独立启动确认、实际服务身份、完整历史和有限资源。动态库核验同时比较预认证文件身份、`/proc/maps` 的 device/inode 和当前磁盘身份，拒绝删除或同名替换对象。主任务独立 26 项标准库／模拟检查通过。

原 4,844 条诊断事件全部保留。16 个必要直接包的 RECORD 声明约 6.390 GB payload、849 个 native 文件，但这不是完整传递依赖清单，也不证明这些库已通过真实认证。16 GiB 的读取上限仅适用于 runtime manifest 的一次认证，不包含约 8.224 GB 原权重哈希及后续读取。

完整依赖 manifest、外层串行控制器、权重存储准入及当前运行绑定尚未完成；没有启动 Spark native 诊断或下载权重。下一步继续补齐这些具体实现，同时等待 LFM 完整 CPU 结果。Qwen v8 后继也仍须等待 LFM 实际结束和排空后才能绑定，r6 的 API 根异常未知及整轮失败记录保留。

[本轮证据与冻结源码](evidence/lfm-full-fixture-and-device-integration-20261112/manifest.json) · [上一轮记录](lfm-regression-and-native-preparation-20261112.md) · [统一分支](multi-model.md)
