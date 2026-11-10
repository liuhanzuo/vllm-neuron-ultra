# Nano 4B：辅助程序放行前记录与原始输出保存

本轮把辅助程序的进程握手、持久保存和 SDK 调用接口连接起来，21 项检查通过。原有 119 项源码校验也重新通过，共读取 5,552,337 字节源码与配置。测试只运行轻量 Python 模拟程序，没有执行安装的 SDK 辅助程序、模型或性能实验，因此没有新增提速倍数。

## 放行之前先保存身份

此前的握手能在子进程运行前捕获实际 PID、启动时间、命令行、父进程和 cgroup，但没有要求先把捕获结果写入文件。新父进程流程先保存这些实际记录，再次检查父子进程身份，最后发送放行消息。保存失败或记录引用不完整时不放行，并回收直接子进程。

保存器只在已批准的现有输出目录中写入预先声明的文件名。它拒绝覆盖，完成文件及目录同步、完整回读和 SHA-256 核对后才返回引用；失败的部分文件保留。单文件上限仍为 8 MiB，总输出上限仍为 64 MiB，并为终止记录保留 1 MiB。进程间文件锁串行化记录写入；实际启动器仍需统一预算服务日志等其他写入。

原始 stdout/stderr 可以包含任意字节，不能简单按 UTF-8 替换错误字符。新格式分别保存 JSON 结构和字节位置，以 base64 无损保存字节，回读后仍保留原来的 bytes、字符串和 null 类型。

初次测试发现我们自己的编码问题：JSON 会排序字段，而字节位置表最初按插入顺序生成，导致 stdout/stderr 顺序变化后回读失败。现在两者都使用一致的字段顺序，新增了针对这个问题的回归检查；初次失败输出和当时源码均已保留。这不是 SDK 或模型的故障。

## SDK 请求如何衔接

新接口只接受 SDK 原来的 `libneuronpjrt-path` 请求和原参数，每个计算进程只允许一次。调用前后核对安装的辅助程序 Python 源码哈希，以及库文件的原路径、设备、inode、大小和时间信息。实际启动的是等待父进程放行的已审查包装程序，其真实命令行单独记录，不伪装成原安装脚本的操作系统命令行。

捕获记录和成功或失败的原始输出都会保存。成功还要求实际退出码为零，输出严格等于原库路径加换行。额外子进程请求和第二次发现请求会被拒绝。原来的 20 秒执行预算涵盖启动握手，未因新增握手而额外延长；失败清理另外受限。

整组清理适配器也增加了对新子进程结果格式的支持，核对记录保存时间早于放行时间，并保留原始结果。原有服务身份、整组清理和失败判定规则没有改变。

## 原环境与尚未完成的部分

| 阶段 | 原 Python 环境 | 本轮状态 |
| --- | --- | --- |
| 原生参考 native-meta | Python 3.12 | 原源码与配置保留，未执行 |
| Neuron nxd-meta 与 SDK 辅助程序 | Python 3.10 | 调用衔接使用模拟程序验证，真实 SDK 未执行 |

模拟程序能够输出同一路径，不能证明真实 SDK 的行为已经正确。实际辅助程序启动入口、site 初始化、解释器和原生依赖核验，以及完整主程序和工作程序接入仍待完成。原有模型结构、BF16 精度、TP2、B4、4096 上下文、标准 CPU sampler 和 FP32 SSM 缓存要求保持不变。

Dory r3 保存结果复核仍在运行。本轮没有启动另一项原生实验，也没有读取原生库内容或模型权重。所有后续模型仍需完成冷缓存、长解码、请求槽位复用、生成质量、HTTP C4 和配对性能验证。

- [上一轮整组清理](nemotron4b-owned-group-cleanup-20261110.md)
- [SDK 请求与记录衔接](evidence/nemotron4b-recorded-helper-20261110/nemotron4b_gated_sdk_helper_v1.py)
- [无损输出保存器](evidence/nemotron4b-recorded-helper-20261110/nemotron4b_worker_receipts_v1.py)
- [21 项检查与实际退出](evidence/nemotron4b-recorded-helper-20261110/nemotron4b-recorded-helper-tests-actual-exit-20261110-v1.json)
- [原 119 项源码复核](evidence/nemotron4b-recorded-helper-20261110/nemotron4b-recorded-helper-original-source-revalidation-20261110-v1.json)
- [源码审查与限制](evidence/nemotron4b-recorded-helper-20261110/nemotron4b-recorded-helper-source-preparation-review-20261110-v1.json)
- [发布清单](evidence/nemotron4b-recorded-helper-20261110/manifest.json)
