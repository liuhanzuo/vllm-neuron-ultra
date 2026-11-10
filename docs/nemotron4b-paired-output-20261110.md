# Nano 4B：保留双进程启动失败时的错误日志

2026-11-10：双 rank 监督器的后继 v2 已修复握手前失败时丢失日志的问题，现有进程组结果适配器的后继 v4 也已接入这个结果格式。**38 项不同检查分别在原 Python 3.10.12、3.12.16 下通过，共 76 次成功执行。真实 Nano SDK、原生模型与 Neuron 基线仍未运行，没有新增提速倍数。**

## 问题与修复

旧监督器要等两个 rank 都完成握手并放行后，才开始读取标准输出和错误输出。若其中一个进程提前退出、握手格式错误或父进程拒绝记录，清理会关闭两边的输出管道，已经写出的诊断内容也会丢失。新增回归用实际生成的两个 Python 子进程复现了旧行为：两个进程都输出错误并提前退出，旧版最终保存的四路输出全为空。

新版仍先记录两个实际进程的身份，再放行任何一个。失败后先回收这两个直接子进程，再用**两边共享的 0.25 秒期限**读取管道中已有内容。标准输出和错误输出分别受原大小上限约束，只保留上限内的原始字节；超限、未读到 EOF、等待超时各自记录。仍被其他进程持有的管道不会导致无限等待。

两边日志收齐不代表实验成功。非零退出、握手失败、资源检查拒绝、强制清理继续保留失败；整个服务进程组是否清空仍由原外层流程验证。原模型、精度、采样、科学函数及资源上限均未修改。这是我们实验工具的缺陷修复，不能归因于上游模型或算子错误。

## 已验证的范围

| 检查 | 每套 Python 通过数 | 内容 |
|---|---:|---|
| 双进程输出与生命周期 | 34 | 旧问题复现、四路原始字节、握手前后失败、超限、超时、等待中的另一进程回收、共享读取期限，以及原身份记录和放行顺序 |
| 工作进程与 helper 组合 | 4 | 两个实际生成的 rank 与各自 helper、helper 早期失败日志、原始结果保存，以及旧格式兼容 |

34 项中包含原 18 项生命周期检查；原“多保存一个超限探测字节”的断言改为“仅保存上限内字节，并标记截断”，失败与回收要求保留。组合测试中的科学计算、启动恢复和来源检查使用明确标记的测试替身，没有执行真实 SDK、模型或系统服务。

首轮两项测试失败也已保留：一项沿用旧版多存一个字节的断言；另一项生成样例在对端写出预期日志之前就制造超限，随后被正确清理。后者改为确认两边已写出日志后才触发超限。修正只涉及后继测试及样例，运行器源码没有因此改动。

## 仍需完成什么

未来真实根入口需要选择 paired supervisor v2 和 owned-group adapter v4。完整启动入口、依赖认证、原 Python 3.12 frozen site、真实串行资源与整组清理接入仍待完成；此次准备没有创建真实 Nano 运行规格或启动任务。

2026-11-10 09:34 UTC 独立核对确认 Nemotron r11 原编译工作进程与模型服务仍存活，继续会话 74703。MiniCPM5 的完整采集、独立审计和自身历史重放仍排在该任务结束之后。全部 15 个模型家族及原冷缓存至 4096、长解码、槽位复用、生成／随机数／停止／文本／质量、真实 HTTP 四并发和配对性能要求保留。

[此前 helper 日志与启动顺序修复](nano-integration-nemotron-all32-20261110.md)的 74 次测试保持原范围；本轮结果不增加模型性能结论。[既有提速表](trn1-optimizations.md)不变。

## 源码与证据

- [双进程监督器 v2](evidence/nemotron4b-paired-output-20261110/nemotron4b_gated_rank_pair_v2.py)与[结果适配器 v4](evidence/nemotron4b-paired-output-20261110/nemotron4b_owned_group_drain_v4.py)
- [回归和生命周期检查](evidence/nemotron4b-paired-output-20261110/check_nemotron4b_gated_rank_pair_v2.py)与[工作进程组合检查](evidence/nemotron4b-paired-output-20261110/check_nemotron4b_pair_output_integration_v1.py)
- [四次实际工具退出记录](evidence/nemotron4b-paired-output-20261110/nemotron4b-pair-output-v2-tests-actual-exits-20261110-v1.json)、[初轮失败与原因](evidence/nemotron4b-paired-output-20261110/initial-failure.json)、[源码审查](evidence/nemotron4b-paired-output-20261110/nemotron4b-pair-output-v2-source-preparation-review-20261110-v1.json)
- [Nemotron 实际进程观察](evidence/nemotron4b-paired-output-20261110/nemotron9b-r11-compile-startup-observation-20261110-v4.json)与[所有文件及 SHA-256](evidence/nemotron4b-paired-output-20261110/manifest.json)
