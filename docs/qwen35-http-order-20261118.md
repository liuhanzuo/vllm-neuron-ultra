# Qwen3.5-2B：分块 CPU 采样的正反序 HTTP 对照

2026-11-18 已保存的两轮结果没有显示稳定的单请求端到端收益，四并发增益也很小且有顺序影响。本次只整理并复核既有数据，没有重跑模型；两轮均为 `model_qualified_by_this_run: false`。

同一份 Trn1 BF16 研究适配器编译图，TP2、decode B4、context4096，输入 128／强制输出 64 token，temperature=0.7、top-k=40、top-p=0.9。每侧每负载预热后测三轮，表中为总输出 token/s 的中位数。prior 为 `e96918402600ff68be17e663429405116c38df67` 的已有快速 CPU sampler，block 为 `04f9cde24e321b5ed55142fd26f133440d686986`；基线包含研究适配器，不是未经修改的上游 vLLM-Neuron。

| 运行顺序 | 单请求 prior → block token/s | 比值 | 四并发 prior → block token/s | 比值 |
|---|---:|---:|---:|---:|
| r1：prior 后 block | 10.07765 → 10.19187 | 1.01133× | 36.85096 → 37.64958 | 1.02167× |
| r2：block 后 prior | 10.09387 → 10.03002 | 0.99367× | 36.93872 → 37.05482 | 1.00314× |

单请求从正序约 +1.13% 变为反序约 −0.63%；C4 从约 +2.17% 缩至 +0.31%。这两个顺序各只有一次配对、每负载三轮，不能据此作统计显著性或普遍提速结论，也不能只报告 r1 的正收益。CPU logits 回放中的约 1.20×／1.18× 只属于采样阶段。

每轮两侧四条贪心输出逐条一致。全部 24 个计时轮次、60 个随机采样请求和 16 个贪心请求的 token 数、耗时、吞吐计算、中位数及比值已从保存文件重新核对。随机文本一致性不作为资格依据。

## 来源与边界

- [r1 原样汇总](evidence/qwen35-http-order-20261118/r1/summary.json)与[r2 原样汇总](evidence/qwen35-http-order-20261118/r2/summary.json)均未改写；逐轮完整请求结果、贪心文本和文件 SHA-256 见[证据目录及清单](evidence/qwen35-http-order-20261118/manifest.json)。服务器原件分别保存在 `/mnt/nvme/neuron-work/qs35-n18-r1/`、`qs35-n18-r2/`。
- 两轮记录的 `model.pt`、`neuron_config.json`、`adapter-provenance.json` 摘要相同；两侧保存的 sampler 源码与指定 Git 提交及 worker 回执摘要相符。服务日志均记录加载该轮预编译图及应用关闭。
- r1 原汇总没有 `arm_order` 字段，顺序由保存的两侧服务启动 ticks 和当轮 `measured_driver.py` 确认。r2 的显式反序与两侧保存启动 ticks 一致。r1 当轮 driver 原样保存；pair driver 与 worker 的另两份源码明确标为本次复核时快照，不冒充当时固定的源码记录。
- 当前复核确认四个服务均不活跃，保存的 API／worker PID 与启动 ticks 对应进程均已不存在。这是当前退出状态核对，不补造历史协调进程的 PID、启动 ticks、完整 History 或退出码。私有系统记录留在服务器。
- 本次没有重做完整数值、冷缓存、长解码、请求槽位复用、采样重放、quality96 或完整匹配 HTTP 验收。旧 r10 研究适配器的完整 BF16 资格不能转给新采样器；独立包仍待验收。

Qwen3-4B 的 `881a149` 结果不重复发布。原 15 个模型家族及既有失败、资格范围保持，MiniCPM5 待回收 10,173,284,352 字节仍未实际删除，DeepSeek 旧 HLO 到归档的绑定证明仍缺。
