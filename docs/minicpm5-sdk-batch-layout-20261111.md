# MiniCPM5：批大小与张量布局对照

2026-11-11，MiniCPM5 在 CPU 上的 Neuron SDK 注意力局部对照已实际完成。两进程共执行 144 次调用，保存 72 份完整四行 BF16 输出（含双 rank 副本，去重后为 36 份 case×variant），覆盖 24 个不同解码检查点（8 个样例 × 3 步）。原权重、精度、bucket、请求槽位和同一次 B4 RoPE 保持一致。

模型运行实际退出 0。修正后的退出后独立审计也已实际退出 0，完整重读全部输出并重现运行期汇总；最后的独立退出记录核验也已实际退出 0。

| 局部 attention 路径 | 对官方输出的汇总相对 L2 | 与官方逐字节相同帧数 |
| --- | ---: | ---: |
| 原自身输入与历史对照 | 1.27204% | 0/24 |
| 官方当前输入与历史，B4，原布局 | 0.44636% | 0/24 |
| 官方当前输入与历史，B4，连续布局 | 0.44636% | 0/24 |
| 官方当前输入与历史，B1，原布局 | 0.44650% | 0/24 |
| 官方当前输入与历史，B1，连续布局 | 0.44650% | 0/24 |

在本次固定输入与历史的范围内，将张量转为连续布局没有改变任何输出：B4 和 B1 各自的 24 帧全部逐字节一致。把 B4 改为逐行 B1，仅影响 2/24 帧、233 个 BF16 元素（不按 rank 或布局重复计数）；22/24 帧完全相同。两者输出间的汇总相对 L2 为 0.031317%，最大绝对差为 0.00048828125。相对于官方结果，1 帧改善、1 帧变差，整体误差未降低。批大小与布局的交互项为零。

这说明本轮布局替换没有带来数值改善，B1 也未消除残留差异。它不证明所有形状下布局都无关，不证明 SDK B1 等同官方原生 attention，也未定位到某个确定的算子根因。下一步应固定同一批输入与历史，比较 Q/K/V 投影、RoPE、注意力分数与 softmax/PV、输出投影等内部边界。

## 运行与复核

原 own/restore 和上一轮 both-official 全输出对照逐字节保持，双 rank 输出相同。退出后的审计已重新读取全部 1,179,648 字节新输出及原对照，重建 B1 物理行和 layout 公式，并复查历史进程与原串行锁。未保存的 RoPE 原始载荷不能被描述成已在退出后重读；该项只有运行时字节检查及持久化摘要/元数据。

首个退出记录器在核对原串行锁时失败，原模型退出码仍为 0。原锁的 20,000 次独立检查未复现，源码审阅未找到提前解锁路径；原记录器不改源码、不改检查规则的单次重试成功。首次失败原因仍未确定，失败证据完整保留，不能宣称该问题已修复。模型未重复运行。

审核程序还发现：旧历史记录包含 1,313 个服务，新物理状态扫描只识别 1,311 个。缺少的两项服务仅保存在历史进程 cgroup 身份中；严格集合相等检查因此停止。进一步完整读取 owner 的资源组路径又识别出 3 项无单独 unit 字段的历史服务，完整待检查集合为 1,316 项。修正后的审计已对完整集合完成两遍真实状态与资源组检查，保留全部历史范围；其中 3 项旧记录没有历史服务调用标识，保留这一证据限制；当前两遍终止状态、空资源组和原进程退出均已实际核验。原模型及输出不重跑。每遍真实历史树已提供原 1,308 项完整状态，另补充 8 项服务状态，共覆盖 1,316 项；完整保留 1,642 条历史进程记录。源码中基于旧树的合成检查曾需补 1,316 项，该合成数量不能当作实际补充调用数量。此问题属于我们的审计程序，不能归为上游模型缺陷。

## 对优化结论的影响

这是 CPU 参考局部诊断，没有进行整模型性能测试。MiniCPM5 整模型原对照仍为 29/32 数值通过，两个候选各 28/32；本轮没有新增合格模型或提速倍数。冷缓存 4096、长解码、请求槽位复用、质量与匹配 HTTP 性能要求全部保留。已有全部模型提速继续见[统一结果表](multi-model.md#existing-results-and-provenance)。

## 后续准备

MiniCPM5 的下一步计划比较 attention 内部完整 Q/K/V、RoPE、分数、softmax、PV 和输出投影边界。三个观测源码后继的 8 项检查已由主任务独立复跑通过；尚未接入实际运行，不能当作新采集结果。

Dory 的完整依赖清点程序通过 39 项独立复跑检查。它只列目录、文件身份和限定的小元数据；完整记录未知来源，区分全部观察文件与待认证的运行源码清单，避免把未跟踪的大型构建产物误计入 8 GiB 内容清单预算。真实清点现已实际退出 0，退出后完整解压重读及文件计数复算通过：79,789 个普通文件，共 5,632,828,570 字节；其中待认证清单 48,986 个、3,169,981,978 字节。完整保留 11 个链接、1,185 个空文件，未发现被多个硬链接引用的普通文件；官方 Git 索引里的文件没有缺名。9,082 个未归属清单项包含未跟踪构建产物，属于后续核对范围，不能据此认定仓库有问题。Torch 2.14.0+cpu 没有对应的已保存安装报告条目或 direct_url 元数据；Python 解释器也仅有位置线索，可信发行来源尚待绑定。缺少 direct_url 本身不是安装故障。清点没有读取原生库或模型权重内容，不能代替可信内容认证、原生 ABI 或模型验收。原 87 个数值检查点中的 7 个失败保留。

## 证据

- [退出后独立完整输出审计](evidence/minicpm5-sdk-layout-20261111/minicpm5-layer1-sdk-batch-layout-r2-post-exit-audit-repaired-20261111-v1.json)、[独立实际退出核验](evidence/minicpm5-sdk-layout-20261111/minicpm5-layer1-sdk-batch-layout-r2-audit-actual-exit-repaired-20261111-v1.json)、[全部实际工具终态](evidence/minicpm5-sdk-layout-20261111/minicpm5-sdk-completed-root-observation-20261111-v1.json)。

- [实际模型退出](evidence/minicpm5-sdk-layout-20261111/minicpm5-layer1-sdk-batch-layout-r2-actual-exit-20261111-v1.json)、[运行期统计](evidence/minicpm5-sdk-layout-20261111/minicpm5-layer1-sdk-batch-layout-r2-coordinator-summary-root-20261111-v1.json)。
- [审计失败保留](evidence/minicpm5-sdk-layout-20261111/minicpm5-layer1-sdk-batch-layout-r2-audit-failed-evidence-20261111-v1.json)、[完整历史集合发现](evidence/minicpm5-sdk-layout-20261111/minicpm5-sdk-historical-unit-scope-root-observation-20261111-v1.json)。
- [修正源码与 71 项检查](evidence/minicpm5-sdk-layout-20261111/mini-sdk-post-audit-unit-completeness-root-review-20261111-v1.json)、[独立审阅](evidence/minicpm5-sdk-layout-20261111/mini-sdk-post-audit-unit-completeness-independent-review-20261111-v1.json)、[来源补充范围](evidence/minicpm5-sdk-layout-20261111/minicpm5-sdk-post-exit-source-supplement-r2-20261111-v1.json)。
- [锁异常审阅及证据限制](evidence/minicpm5-sdk-layout-20261111/mini-sdk-exit-recorder-lock-source-review-20261111-v1.json)、[下一步内部边界设计](evidence/minicpm5-sdk-layout-20261111/minicpm5-layer1-operator-boundary-next-plan-20261111-v1.md)、[Dory 清点源码复核](evidence/minicpm5-sdk-layout-20261111/dory-stage-a-native-metadata-census-root-review-20261111-v2.json)。
- [证据来源、大小及 SHA-256 清单](evidence/minicpm5-sdk-layout-20261111/manifest.json)。

Dory 新证据：[退出后完整清点复核](evidence/minicpm5-sdk-layout-20261111/dory-stage-a-metadata-census-r1-post-exit-root-review-20261111-v1.json)、[真实清点结果](evidence/minicpm5-sdk-layout-20261111/dory-stage-a-metadata-census-r1-result.json)、[来源与大小清单](evidence/minicpm5-sdk-layout-20261111/dory-supplement-manifest-v1.json)。
