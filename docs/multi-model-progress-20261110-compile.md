# Nemotron 编译进展、MiniCPM5 启动检查与 Dory 归档接入

2026-11-10 16:15 UTC 核验：Nemotron r12 的原运行会话仍在执行整模型编译，前 42/52 阶段已完成。MiniCPM5 下一轮诊断的完整来源检查通过，Dory 的归档事务源码完成独立检查。本轮没有新增通过完整验收的模型，也没有新增模型提速倍数。

## Nemotron：拆分算子对照完成，整模型编译继续

32 个拆分阶段覆盖 16 组原写法／`tensor_split` 配对，包括投影与卷积、不同输入长度以及单行控制。已保存 worker 结果显示：

| 检查 | 结果 |
| --- | --- |
| 候选完整设备输出 | 16/16 组通过，96/96 次完整输出逐字节一致 |
| 原写法完整设备输出 | 14 组多行输入失败；2 组单行控制通过 |
| CPU 原写法与候选控制 | 全部通过 |
| 整模型资格和配对提速 | 尚未获得 |

这里汇总的是已保存结果 JSON；各阶段的真实退出、进程排空已核对，但本次汇总没有重新读取原始输出字节。整轮结束后的独立原始数据审核仍须执行。局部候选修正不能证明整模型的基线 NaN 问题已经解决。

控制器仍持有原串行锁，编译 worker 身份匹配。模型权重与设置、原 52 阶段、数值、冷缓存 4096、长解码、请求槽位复用、质量及真实 HTTP 配对性能门槛全部保留。编译日志已生成全部 HLO 并开始优先解码图的编译；此日志不是编译完成凭据。没有重复启动实验或扩容 EBS。

## MiniCPM5：当前 builder v5 的完整来源检查通过

前一份报告测量的是 builder v2。本轮针对当前 v5 的 544 个本地依赖文件重新合并并认证全部历史来源，结果如下：

| 项目 | 结果 |
| --- | ---: |
| 合并后来源记录 | 44,021 |
| 文本条目，含一份仅在内存的测试 contract | 43,989 |
| 特殊编码源码／历史原始载荷／业务原生库 | 3／20／9 |
| 保留的元数据别名 | 4 |
| 实际认证返回的文本字节 | 7,602,764,186 |
| 特殊编码源码字节 | 411 |
| 墙钟／CPU 时间 | 71.495／71.492 秒 |
| 峰值 RSS | 308,948 KiB，约 302 MiB |

没有读取历史原始载荷、业务原生库或模型张量。该测量的 guard 只检查观察锁文件，并不包含真实生产启动的全部成本。为量化其中一项，另一次有界测量调用原始存活控制器锁检查 43,994 次：全部完成，循环耗时 12.244 秒，约 0.278 毫秒/次；每次确认原控制器及原串行锁，测量进程没有获取或释放该锁。这两项测量不能简单相加当作生产预算，也不能替代历史单位完整观察或真实模型启动。

上一轮诊断结论不变：同时使用官方当前输入与有效历史，局部汇总相对 L2 从 1.2720% 降至 0.4464%，24 帧均改善但没有完全匹配帧。整模型原对照数值 29/32、首选 token 32/32，两个候选各 28/32、31/32，均未通过完整验收。

下一步仍是固定原 bucket、槽位、B4 RoPE 与 BF16 投影的 B4/B1 × 原 stride/连续布局实验，保留 144 次 SDK 调用、72 份完整四行输出和全部原输出控制。SDK B1 不等同官方原生路径。必须先取得 Nemotron 的真实终止和独立释放凭据，才能绑定生产配置、启动新实验。

## Dory：采集、落盘与独立读取接成一条调用链

后继 v2 事务源码直接连接原被动观测器、张量适配器、三路归档 writer 和独立 saved-file reader。六次调用产生 18 份文件；全部元数据检查通过后，独立 reviewer 才能开始读完整文件。每份文件独占创建，依次 flush、fsync、close，再同步目录并复核身份。任何写入、关闭或身份检查失败均保留部分文件，不返回完成凭据。

作者与根代理分别执行同一组 52 项检查，全部通过；这不是 104 项不同覆盖。检查使用标准库、内存字节和模拟文件系统，没有执行 Torch 原生模型、真实 Stage A 采集或 NVMe 耐久性验证。固定 Stage A 数据量 1,467,113,472 字节，三阶段总计 2,046,031,872 字节，未来 3 GiB 预算不变。87 个标记点中的 7 个 FP32 数值失败继续保留。

实际执行前还需官方路径专用 worker/preparer、可持久化 manifest、真实 producer/reviewer 退出绑定及原生适配器验证。后继源码准备已继续；本轮没有创建生产准入。

## 近期仓库复查：保留原 15 个家族

在既定 19 个官方发布方、各最近 16 个仓库的有界范围内复查 293 条记录：7 条 revision 改变，未发现相对上次快照新建的仓库。这不是全网穷尽搜索，修改日期也不等同模型发布日期。

对三个目标固定 revision 的完整文件树及配置逐项复核，变化均仅在 README，配置不变：

| 仓库 | 对当前工作的意义 |
| --- | --- |
| MiniCPM-OCR-1.0 | 元数据约 1.300B BF16 参数，可保留为未来 OCR 候选；完整版面分析流水线参数和资源尚未核验，文本测试不能认证 OCR 质量 |
| MiniCPM-OCR-1.0-DSpark | 约 296.8M BF16 草稿参数，必须匹配主模型；不是独立模型。官方 `d2t` 是相对偏移，需 `d2t[i] + i` 得到目标 token ID |
| Qwen3.8-27B-DSpine-b16 | 约 2.026B 草稿，但依赖 27B 主模型，不能按独立 <10B 模型纳入 |

这是官方文档中的兼容要求和范围核验，没有发现可据此宣称的新实现缺陷，也未测得这些仓库的提速。完整权重内容未下载或读取。原 15 家族及串行实验顺序不变。

## 证据

- [Nemotron 实际进度](evidence/multi-model-progress-20261110-compile/nemotron9b-r12-actual-progress-20261110-v14.json)与[完整拆分结果汇总](evidence/multi-model-progress-20261110-compile/nemotron9b-r12-split-result-metadata-summary-20261110-v1.json)。
- [MiniCPM5 v5 完整来源实测](evidence/multi-model-progress-20261110-compile/minicpm5-layer1-sdk-batch-layout-source-compatibility-measurement-delivery-20261110-v2.json)与[真实锁检查成本](evidence/multi-model-progress-20261110-compile/minicpm5-sdk-builder-live-controller-guard-cost-delivery-20261110-v1.json)。
- [Dory 事务源码与接入说明](evidence/multi-model-progress-20261110-compile/dory-stage-a-artifact-transaction-source-delivery-20261110-v2.md)及[根代理独立检查](evidence/multi-model-progress-20261110-compile/dory-stage-a-artifact-transaction-root-review-20261110-v2.json)。
- [官方小模型复查](evidence/multi-model-progress-20261110-compile/official-small-delta-20261110-v1/selection.json)、[独立文件树核对](evidence/multi-model-progress-20261110-compile/official-small-delta-root-review-20261110-v1.json)和[完整证据清单](evidence/multi-model-progress-20261110-compile/manifest.json)。

已有模型提速倍数及其适用范围继续见[统一分支概览](multi-model.md#existing-results-and-provenance)，本报告不改写历史结果。
