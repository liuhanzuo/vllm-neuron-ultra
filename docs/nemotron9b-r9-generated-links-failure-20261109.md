# Nemotron 9B r9：生成文件名误拒绝与失败释放

2026-11-09：r9 已失败退出并清空全部 53 个服务/进程组。52 个实验阶段中，准备、9 项状态别名原语和 32 项拆分原语共 42 项保留了成功结果；完整模型编译中断，未获得新的整模型或性能结论。

本次首先触发的异常是我们自己的编译产物库存检查：Reviewed generated numeric basename and link subgraph。检查只认识较短的数值文件名，漏掉了编译器实际生成的 tiled transpose 名称，32 个链接因此被拒绝。例如：

    value_sg0001_tiled_pf_transposetiled_pf_transpose_sg0001_13912_identity_11_CRSM.npy

链接位于同一编译单元的 sgLnk/sg00，目标按名称中的子图号指向同单元 sg01。不能把链接所在目录号当成目标子图号。此错误来自本实验的文件解析规则，不能据此认定模型或上游编译器算术错误；此前固定 split 消费者图的设备地址错误证据仍另行保留。

修正候选仅加入这项精确命名格式，并要求重复子图号一致。原有的目录不跟随链接、同单元目标推导、文件归属、大小上限、普通文件、稳定性和哈希检查继续生效。使用独立函数全局环境复用原检查主体，没有修改冻结的 v18 源码或原始模块全局变量。14 项检查通过；真实 32 个被拒绝链接全部通过元数据模式识别，完整工作目录库存为 2,175,258,624 字节。元数据识别不等于这些目标已在该步骤逐一读出并验证张量内容。

原控制器清理曾遇到进程命令行/进程组检查失败；按原接口触发一次清理重试。编译服务最终记录 timeout/SIGTERM，这是停止及清理记录，不能改写成“完整编译运行满四小时超时”。原外部任务退出码 1，失败释放任务实际退出码 0。最终审计 passed=false，释放记录 released=true、successful=false、failed=true，所有进程组清空；释放成功没有把模型实验改判成功。

下一轮尚未启动。需要把文件名修正正式接入新的完整协议版本，认证本轮实际退出、恢复和 Dory 完成历史，再按原 52 阶段执行。原 BF16、模型实现、冷缓存至 4096、长解码、槽位复用、生成/RNG/停止/文本、质量、真实 HTTP C4 和配对性能要求不变。失败对照不能作提速分母，本轮没有新增 Nemotron 提速倍数。

[修正源码](evidence/nemotron-r9-generated-links-20261109/nemotron9b_compiler_dynamic_links_v2.py)、[14 项检查与真实链接识别](evidence/nemotron-r9-generated-links-20261109/link-source-check-result.json)、[原失败退出](evidence/nemotron-r9-generated-links-20261109/nemotron9b-r9-launch-actual-exit-20261109-v1.json)、[失败释放实际退出](evidence/nemotron-r9-generated-links-20261109/nemotron9b-r9-failed-release-actual-exit-20261109-v1.json)、[保留失败的最终审计](evidence/nemotron-r9-generated-links-20261109/external-audit.json)、[最终释放](evidence/nemotron-r9-generated-links-20261109/root-release.json)、[完整发布清单](evidence/nemotron-r9-generated-links-20261109/manifest.json)。

这些文件用于审阅修正及本机证据，依赖已有固定源码、环境和原始实验文件，不是自包含的复现实验包。
