# Dory：原审查受资源限制退出，完整数据复核已重启

2026 年 11 月 9 日，原 Dory 保存数据审查实际退出，退出码为 1，报告“Full original resource bounds”。它没有写出完整审查结果。六份输出和真实退出记录已保留，原进程已确认退出；已有原生重放的成功记录以及原来的七个 FP32 失败也全部保留。

随后使用已准备的审查程序，在新的输出目录启动重试。22:53 UTC 已完成真实启动核验：1,565 个源文件与二进制文件、1,071,484,575 字节全部通过校验，历史记录中的 1,149 个服务和 348 个进程全部确认退出，重试进程正在运行。这些是启动校验数量，不是已经通过的模型事件或张量比较数量。

## 重试改了什么

沿用原有完整数值复核、检查频率及资源上限，包括四小时时限、原内存和磁盘下限、单 CPU 绑定。服务状态查询改用已核验的独立持久连接，每次检查仍获取新的服务状态，不缓存旧结果。

原审查频繁启动独立服务查询进程，会增加诊断开销。此次修改只针对诊断工具的服务查询方式，不修改模型计算，也不构成模型提速。运行时仍要完成全部原始数据检查，最终成功还要求依赖复核、连接关闭和从外部确认实际退出。

对应准备与限制见[服务查询准备](dory-saved-review-service-query-preparation-20261109.md)和[重试后的完整任务接续](dory-retry-nemotron-continuation-20261109.md)。此前实际服务查询对照覆盖的是无活动实验服务的情况，没有人为启动服务来验证正向状态变化；这一限制继续保留。

## 当前结果和后续顺序

原生重放已经完成；保存数据的完整数值审查正在重试，尚无最终结论。原来的七个 FP32 失败仍未定位根因，不能将它们改判为通过，也没有新增基线或提速数字。

重试成功并确认实际退出后，使用重试后的任务历史汇总结果，执行已授权的有限存储维护，再推进 Nemotron v23 和 MiniCPM5 保存张量分析。后续不能复用原直接成功路径中较少的进程计数，也不能跳过这次失败及重试记录。若重试再次失败，则先保存真实失败证据并诊断，不能自动认定完成。

- [原审查实际失败与完整输出引用](evidence/dory-saved-review-retry-launch-20261109/dory-residual-saved-review-r2-failed-actual-exit-20261109-v1.json)
- [重试具体运行方案](evidence/dory-saved-review-retry-launch-20261109/dory-residual-saved-review-r3-runtime-spec-20261109-v1.json)
- [对应源码复核](evidence/dory-saved-review-retry-launch-20261109/dory-residual-saved-review-r3-source-review-20261109-v1.json)
- [实际启动与完整校验记录](evidence/dory-saved-review-retry-launch-20261109/dory-residual-saved-review-r3-live-launch-20261109-v1.json)
- [证据清单](evidence/dory-saved-review-retry-launch-20261109/manifest.json)
