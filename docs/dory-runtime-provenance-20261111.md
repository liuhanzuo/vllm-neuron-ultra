# Dory 原生环境来源核对与 MiniCPM5 下一步准备

2026-11-11。本轮继续准备可信的参考推理环境和注意力内部诊断。尚无新的整模型合格结果或提速倍数；全部已有性能结果保留在[统一结果表](multi-model.md#existing-results-and-provenance)。

## 已补齐的来源信息

在已完成的完整环境清点上建立安装包来源映射：163 个已安装发行包，其中 20 个不在保留的安装报告中。待验证 SITE 清单中没有无安装包归属或被多个安装包同时声明的文件。此前 9,082 项“未归属”包含官方工作区里的未跟踪构建产物，与这个仅限待验证 SITE 文件的结果范围不同，不能混用。

这 20 个包均已找到与本机版本、WHEEL 标签一致的官方发行候选。Torch 2.14.0+cpu 的官方 CPU 索引提供精确 wheel URL 和 SHA-256；其官方 METADATA 与本机一致。CPython 3.12.16 的 BUILD 日期为 20261026；对应 Astral 发行包的 SHA-256 在 GitHub 发行信息、SHA256SUMS 和当时 uv 下载目录中一致。来源线索需继续通过实际归档与安装内容比较确认。

Torch 与 Python 两份归档合计 230,759,432 字节，其完整下载内容和落盘内容已核对上游 SHA-256。第一次安装文件比较在 Python 阶段触发原串行锁检查错误，实际退出 1；不形成完整比较结果。失败和文件保留，严格检查未放宽。后继复用归档，重新检查完整哈希，并在再次出现异常时保留原失败当刻的锁表与稍后的描述符状态。该锁异常的根因仍未知。

## 已完成的归档与安装文件比较

复用原归档的第二次运行实际退出 0。退出后重新取得原串行锁，确认进程已退出，完整解压重读保存的逐文件比较记录并复算数量；没有宣称退出后再次读取全部安装内容。

| 对象 | 逐字节相同 | 保留差异 |
| --- | ---: | --- |
| Torch wheel 普通文件 | 12,395 / 12,396 | 仅 RECORD 安装清单 |
| CPython 归档普通文件 | 3,483 / 3,487 | `_sysconfigdata` 配置源码与 3 份 `.pyc` |
| CPython 归档符号链接 | 1,049 / 1,049 | 无 |
| Dory 虚拟环境中的解释器副本 | 3 / 3 | 无 |

Torch RECORD 的后续完整元数据比较：上游 12,396 条全部存在，哈希与大小字段全部不变；本机增加 2,437 条 `.pyc` 记录，以及 `torchrun`、`torchfrtrace`、INSTALLER、REQUESTED 四项。这个差异不意味着 Torch 执行源码被改动；新增项目仍分别记录来源，不直接当作上游归档原件。

Python `_sysconfigdata` 中 70 个配置字段不同，包括安装目录替换、编译器名称及编译参数整理。完整差异已保存，正在核对安装工具的对应变换；本轮不将其自动豁免，也不据此认定环境损坏。三份 `.pyc` 保留为字节码差异，未执行。其余依赖、安装后新增文件、系统库及预加载库的完整来源和真实 ABI 检查仍待完成。

第二次核对未再次触发锁异常；第一次失败的根因仍未知，不能把一次成功解释成故障已修复。

## MiniCPM5 观测驱动

已准备两侧的可调用源码和精确事件清单，尚未接入实际运行：

| 阶段 | 固定范围 | 原始保存量 |
| --- | --- | ---: |
| Neuron SDK 的 CPU 参考 | 原 TP2/B4、6 个调用、24 个检查点，5 组无观测/观测/恢复控制，共 60 次注意力调用 | 624 份文件，63,876,480 字节 |
| 官方原生参考 | 同 24 个检查点，单层注意力直接调用，3 组控制，共 72 次调用 | 936 份文件，28,490,724 字节 |

保存完整 Q/K/V、RoPE、注意力分数、softmax 概率、PV/context 和输出投影边界。每组独立恢复原缓存，要求观测输出及更新后缓存逐字节重现原参考；不重新执行已完成的布局矩阵或整模型轨迹。原生恢复显式选择第 1 层缓存；原 normalized tap 未保存的 storage offset 仍标为未知，不能声称已重现该属性。

主任务复跑两侧共 32 项源码与合成检查通过。执行器接入、运行前检查、实际采集和退出后完整原始数据复核仍待完成；这些源码检查不能替代真实数值或性能结果。冷缓存 4096、长解码、请求槽位复用、质量和匹配 HTTP 性能要求保持不变。

## 证据

[来源清单](evidence/dory-provenance-20261111/dory-upstream-provenance-discovery-delivery-20261111-v1.json)、[归档比较实际退出与完整记录复核](evidence/dory-provenance-20261111/dory-torch-cpython-archive-r2-post-exit-root-review-20261111-v1.json)、[完整元数据差异](evidence/dory-provenance-20261111/dory-archive-metadata-differences.json)、[首次失败保留](evidence/dory-provenance-20261111/dory-torch-cpython-archive-r1-failed-terminal-20261111-v1.json)、[MiniCPM5 驱动源码检查](evidence/dory-provenance-20261111/minicpm5-operator-driver-root-source-confirmation-20261111-v1.json)、[完整证据哈希清单](evidence/dory-provenance-20261111/manifest.json)。
