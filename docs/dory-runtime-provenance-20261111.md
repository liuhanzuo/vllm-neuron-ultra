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

Python `_sysconfigdata` 的差异现已完整解释：从保存差异恢复的上游 50,137 字节源码哈希与归档记录一致；按固定官方 uv 提交 `acee8d938b52225be3e90ccfcbe69fa4eb8ed215` 的安装算法重放，生成的全部 51,256 字节与本机完全一致，涵盖 70 个原字段变更及新增 `PYTHON_BUILD_STANDALONE=1`。主任务复核源码、重跑完整重建和 14 项检查通过。这证明内容符合官方安装变换，不能追溯证明历史执行的是哪个 uv 二进制或进程。三份 `.pyc` 保留为字节码差异，未执行。后续全包比较见下节；安装后新增文件、系统库及预加载库的完整来源和真实 ABI 检查仍待完成。

第二次核对未再次触发锁异常；第一次失败的根因仍未知，不能把一次成功解释成故障已修复。

## 全部 163 个依赖包的实际核对

全包比较实际退出 0。163 份选定 wheel 合计 925,561,110 字节，均核对完整归档哈希；复用已保存的 Torch 与本地构建 vLLM wheel，没有重新下载。vLLM 的比对基准是本机保存的构建产物，不能把它称为上游发布的二进制认证。运行记录共 152,788 个事件，退出后重新取得原锁，完整解压复读并独立复算全部汇总字段。

| 范围 | 实际结果 | 解释 |
| --- | ---: | --- |
| wheel 普通文件 | 39,960 / 40,128 完全一致 | 其余 168 项见下两行 |
| RECORD 安装清单 | 163 份大小不同 | 40,063 个同路径条目的原哈希/大小字段全部不变；另 65 个路径全部是已核实的安装迁移 |
| 包内脚本 | 5 份大小不同 | 全部仅首行改成当前虚拟环境解释器路径，正文及换行完全相同 |
| 65 个迁移条目 | 60 个只迁移路径，5 个同时改脚本首行 | 安装后清单的目标路径、哈希和大小全部符合预期 |

因此，在已比较的发行包成员中，未发现上述安装清单和解释器路径变换以外的代码或二进制差异。这不是整套运行环境或模型正确性的通过结论。

另有 22,247 项安装后内容：21,724 项字节码缓存、459 项安装元数据、62 项环境入口/解释器文件，以及 `lib64` 链接和 `pyvenv.cfg`。这些保持单独记录，不能直接算作 wheel 原件。新旧清点对照中唯一的 SITE 新名字是 SITE 根目录自身（两份清点的根目录纳入范围不同）；没有发现 SITE 文件身份变化或缺失。系统库、官方工作区和实际使用的字节码路径仍须收尾核对，然后才能进行真实 ABI 与模型数值对照。

[全包实际结果](evidence/dory-provenance-20261111/dory-full-wheel-actual-result.json)、[退出后完整记录复核](evidence/dory-provenance-20261111/dory-full-wheel-comparison-r1-post-exit-root-review-20261111-v1.json)、[全部清单/脚本差异汇总](evidence/dory-provenance-20261111/dory-full-wheel-metadata-differences-summary.json)、[65 个路径迁移逐项核实](evidence/dory-provenance-20261111/dory-full-wheel-record-relocations.json)。

## MiniCPM5 观测驱动

两侧的可调用源码和精确事件清单已准备；SDK 侧新一轮两个 rank 已完成，但汇总文件超过上限导致整体失败，独立数据恢复审计尚待完成。官方原生侧仍待接入：

| 阶段 | 固定范围 | 原始保存量 |
| --- | --- | ---: |
| Neuron SDK 的 CPU 参考 | 原 TP2/B4、6 个调用、24 个检查点，5 组无观测/观测/恢复控制，共 60 次注意力调用 | 624 份文件，63,876,480 字节 |
| 官方原生参考 | 同 24 个检查点，单层注意力直接调用，3 组控制，共 72 次调用 | 936 份文件，28,490,724 字节 |

保存完整 Q/K/V、RoPE、注意力分数、softmax 概率、PV/context 和输出投影边界。每组独立恢复原缓存，要求观测输出及更新后缓存逐字节重现原参考；不重新执行已完成的布局矩阵或整模型轨迹。原生恢复显式选择第 1 层缓存；原 normalized tap 未保存的 storage offset 仍标为未知，不能声称已重现该属性。

主任务复跑两侧共 32 项源码与合成检查通过。执行器组件修复进一步通过独立复核：18 项原检查与 6 项追加检查通过，解决临时文件计数和字段顺序依赖；具体启动配置、外部退出记录及完整原始数据审计接入已完成源码交付；这些源码检查不能替代真实数值或性能结果。冷缓存 4096、长解码、请求槽位复用、质量和匹配 HTTP 性能要求保持不变。


下一轮实际准备首先暴露了历史状态检查的接线错误：新任务把上一轮完成状态误当成当前串行终态，准备程序实际退出 1，尚未写配置。修正保留历史完成链，在三个新启动入口校验当前串行目录；69 项作者检查、68 项独立检查通过（独立检查未重复一项完整历史元数据组装）。新版准备程序实际退出 0，生成的完整配置与源码清单经主任务复核。

随后 SDK 采集执行程序实际启动，但在模型加载前再次触发 `Controller does not hold whole serial flock`，实际退出 1。失败证据核对确认进程已退出、服务从未启动、输出目录不存在，没有注意力调用或新数值结果。这次已经通过修正后的状态衔接检查，不能混为同一故障。原串行锁保护未放宽。后续 300,000 次单独原锁检查未复现；此结果不能证明间歇异常已经消失，也不能说明失败瞬间真的丢失了锁。另一次 50 秒有界对照中，独立子进程在独立临时文件上切换锁 36,535,040 次，原检查通过 342,143 次，仍未复现；子进程正常退出并回收。因此锁表变化这一假说目前也未得到验证，下一次实际启动应在原检查失败当刻保留原始锁表与描述符证据，继续保持原保护条件。

[实际准备失败](evidence/dory-provenance-20261111/minicpm5-operator-sdk-r1-preparation-failed-actual-exit-20261111-v1.json)、[状态衔接修正独立复核](evidence/dory-provenance-20261111/minicpm5-operator-sdk-freshness-correction-independent-review-20261111-v1.json)、[新版准备实际完成](evidence/dory-provenance-20261111/minicpm5-operator-sdk-r1-root-preparation-complete-20261111-v2.json)、[执行程序实际失败与退出](evidence/dory-provenance-20261111/minicpm5-layer1-operator-sdk-r1-failed-terminal-20261111-v1.json)、[完整失败证据](evidence/dory-provenance-20261111/minicpm5-layer1-operator-sdk-r1-model-failed-evidence-20261111-v1.json)。

诊断后继通过独立源码审查及 13 项针对性检查；原锁检查、失败传播及清理流程保持不变。它只在原检查失败时保存当刻证据，不重试或改用宽松条件。新一轮 r2 配置实际生成成功，主任务确认原科学输入、624 份采集文件范围、资源与验收条件均保留；生成配置后的完整读取检查也通过。实际执行随后通过加载前检查，启动了服务和两个 rank。两个 rank 随后各完成 6 个锚点调用、30 次注意力调用，并均以 0 退出。coordinator 在写入嵌套的两份 rank 结果和完整校验证明时触发原 8 MiB `Evidence cap`；coordinator 与外层程序实际退出 1。全部 624 份原始文件（63,876,480 字节）及 12 份日志保留，独立失败证据确认进程退出和服务清空。这里的数量来自完整文件清单与大小检查，尚未完成退出后的独立原始字节复核。下一步以无损引用替代重复嵌套，单独审计已保存数据；保留原 exit 1，不补造成功记录，也不为修复汇总而重跑模型。本次未产生锁失败诊断，不能宣布原锁异常已修复。

为保存新配置，在原串行锁内整理已发布仓库的 Git 对象，回收 34,603,008 字节；没有裁剪 Git 历史、删除实验文件或扩容磁盘。

[诊断后继独立审查](evidence/dory-provenance-20261111/minicpm5-operator-sdk-original-lock-diagnostic-independent-source-review-20261111-v1.json)、[r2 配置实际生成与主任务复核](evidence/dory-provenance-20261111/minicpm5-operator-sdk-r2-root-preparation-complete-20261111-v1.json)、[r2 实际启动身份](evidence/dory-provenance-20261111/minicpm5-layer1-operator-sdk-r2-actual-live-20261111-v1.json)、[r2 实际准入观察](evidence/dory-provenance-20261111/minicpm5-layer1-operator-sdk-r2-actual-admission-20261111-v1.json)。

Dory 的有界分阶段启动器已通过 17 项独立检查和新增最终文件预算检查。清单阶段统一由独立工作进程执行，父进程持锁至工作进程及其子进程组退出；最终结果文件计入占用预算。绑定上述 MiniCPM5 真实失败与完整历史服务范围后，清单检查实际运行并退出 1，失败发生在外部库读取的单硬链接检查。父进程和子进程已退出、进程组已清空，失败和 70,726 条事件完整保留。退出后持原锁完整复读压缩日志：40,522 条 SITE 清单、21,723 条排除项、8,464 条官方源码清单和 15 条已处理外部依赖均在记录内，尚无完整最终清单。根据完整已保存依赖队列与冻结代码推导，下一项是 libc.so.6；随后 stat 观察确认该文件有两个硬链接，文件类型与大小符合要求。失败当刻的描述符路径未直接保存，因此这是队列推导与后续观察，不冒充当刻记录。libstdc++.so.6.0.30 也观察到两个硬链接。下一版将只在这两个系统路径的外部读取边界支持严格稳定的双硬链接文件，系统库来源仍单独待核实。它复用已完成的 163 包比较，不重做下载或整包对比。原生 ABI 与 Stage A 仍需随后分别验证。

[MiniCPM5 汇总失败实际终态](evidence/dory-provenance-20261111/minicpm5-layer1-operator-sdk-r2-failed-terminal-20261111-v1.json)、[完整失败与文件保留证据](evidence/dory-provenance-20261111/minicpm5-layer1-operator-sdk-r2-model-failed-evidence-20261111-v1.json)、[Dory 启动器独立审查](evidence/dory-provenance-20261111/dory-serial-stage-independent-source-review-20261111-v3.json)、[Dory 清单实际启动](evidence/dory-provenance-20261111/dory-serial-inventory-r1-actual-live-20261111-v1.json)。

MiniCPM5 汇总超限已用实际两份 rank 元数据复现：旧嵌套序列化约 8,740,236 字节，引用两份原文件并保留校验证明的示例为 2,377 字节。用于这次尺寸复现的校验证明是固定结构模板，不能当作原始字节审计结果。紧凑格式与失败恢复纯验证模块的 10 项独立检查通过；实际恢复审计程序仍在接入。

[Dory 清单实际失败与退出](evidence/dory-provenance-20261111/dory-serial-inventory-r1-actual-exit-20261111-v1.json)、[完整保存事件复读](evidence/dory-provenance-20261111/dory-inventory-r1-failed-saved-events-review-20261111-v1.json)、[失败位置的队列推导](evidence/dory-provenance-20261111/dory-inventory-r1-external-failure-fifo-derivation-20261111-v1.json)、[MiniCPM5 紧凑汇总与失败恢复核心复核](evidence/dory-provenance-20261111/minicpm5-operator-sdk-compact-recovery-independent-source-review-20261111-v1.json)。

## 证据

[来源清单](evidence/dory-provenance-20261111/dory-upstream-provenance-discovery-delivery-20261111-v1.json)、[归档比较实际退出与完整记录复核](evidence/dory-provenance-20261111/dory-torch-cpython-archive-r2-post-exit-root-review-20261111-v1.json)、[完整元数据差异](evidence/dory-provenance-20261111/dory-archive-metadata-differences.json)、[首次失败保留](evidence/dory-provenance-20261111/dory-torch-cpython-archive-r1-failed-terminal-20261111-v1.json)、[MiniCPM5 驱动源码检查](evidence/dory-provenance-20261111/minicpm5-operator-driver-root-source-confirmation-20261111-v1.json)、[完整证据哈希清单](evidence/dory-provenance-20261111/manifest.json)。

[官方 uv 安装变换完整重放](evidence/dory-provenance-20261111/dory-uv-sysconfig-source-replay-20261111-v2.json)、[主任务完整复核](evidence/dory-provenance-20261111/dory-uv-sysconfig-root-confirmation-20261111-v1.json)。

[独立锁变化对照实际退出](evidence/dory-provenance-20261111/original-serial-lock-churn-actual-exit-20261111-v1.json)。
