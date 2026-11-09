# Nemotron Nano 4B：辅助程序依赖缺口与入口执行准备

源码检查发现，我们此前的 Nano 4B 准备清单漏掉了 SDK 库路径辅助程序会经过的 5 个本地 Python 模块。它们都与安装包的原始 RECORD 内容记录一致；这是准备范围不足，不能据此判断 SDK 存在计算错误或性能问题。

本轮只检查源码，没有运行真正的 SDK 辅助程序、加载模型或读取共享库内容。新准备的入口执行核心及静态依赖分析共 24 项选定检查通过。基线、正确性和提速结论没有变化。

## 为什么返回路径的程序也有这些依赖

安装的 libneuronpjrt-path 入口通过 from libneuronxla.libneuronpjrt_path import main 加载函数。Python 会先执行 libneuronxla 的包初始化；包初始化又导入编译包装、缓存、日志和 protobuf 模块。不能只审查那个返回文件路径的小函数，就认为覆盖了实际导入过程。

对安装包的 15 个 Python 文件、共 143,542 字节，已逐个验证其内容与原始安装记录一致。保守的模块体导入图覆盖 13 个 Python 模块和一个 proto 命名空间。其中以下 5 个文件不在旧 v2 的源码指纹集合内：

| 缺少的模块 | 在导入链中的作用 |
|---|---|
| libneuronxla.logger | 日志初始化 |
| libneuronxla.neuron_cc_cache | 编译缓存相关定义 |
| libneuronxla.proto.hlo_pb2 | HLO 的 protobuf 定义 |
| libneuronxla.proto.metrics_pb2 | 指标的 protobuf 定义 |
| libneuronxla.proto.xla_data_pb2 | XLA 数据的 protobuf 定义 |

包外还涉及 boto3、botocore、google.protobuf 等依赖。本轮没有把它们及其原生扩展全部展开，因此不能宣称已有完整的运行时依赖清单。分析同时列出函数体内的延迟导入和动态执行点；静态图保守地检查条件分支，不代表这些分支都在实际运行中执行。

## 入口执行核心

在[上一轮握手与子进程监督](nemotron4b-process-handshake-20261109.md)基础上，新增核心负责核验入口源码、等待父进程允许、再次检查库文件身份，然后在同一被记录的包装进程中执行源码。库路径输出必须完全一致，Python 文本输出有 4 KiB 上限；父监督进程仍需限制实际标准输出和错误输出，包括原生代码写入。

选定的 v2 实现保留原始非零 SystemExit 值。早期 v1 会将其拒绝为 ValueError，失败分类正确，但异常值没有原样传递；v1 源码和记录保留，后续接入使用 v2。生成测试没有执行安装的 SDK 代码。

包装会临时设置脚本所见的 sys.argv 和 __file__，但操作系统记录的进程参数仍是包装程序。后续验收必须如实记录包装身份和被执行入口的指纹，不能假称已观察到原命令的 exec。

## 验证与下一步

11 项静态图检查覆盖隐式包初始化、相对导入、命名空间、循环、函数体延迟导入、动态调用和缺失本地模块。13 项入口检查覆盖握手前后顺序、源码变化、库文件变化、父进程拒绝、导入环境失败、输出超限、精确输出、非零退出值和参数恢复。

下一步需把新增的 5 个本地源码指纹及包外依赖审查接入不可变的新工作程序、方案和真正的辅助程序包装。完整根监督器、双分片监督器、运行条件检查和完整前序任务退出记录仍是必要条件。新核心接受调用方提供的审查回调，本身不构成运行许可。

现有 Dory、有限存储维护、Nemotron 串行链保持原顺序。Nano 4B 原 BF16、双分片、批量 4、上下文 4096、FP32 循环状态和所有后续正确性、生成、质量及配对性能要求均保留，全部 15 个模型家族继续推进。

- [完整源码依赖诊断](evidence/nemotron4b-helper-entry-20261109/nemotron4b-helper-import-source-diagnosis-20261109-v1.json)
- [选定 v2 源码准备复核](evidence/nemotron4b-helper-entry-20261109/nemotron4b-helper-entry-source-preparation-review-20261109-v2.json)
- [选定入口执行核心](evidence/nemotron4b-helper-entry-20261109/nemotron4b_guarded_helper_entry_v2.py)
- [源码、测试及退出记录清单](evidence/nemotron4b-helper-entry-20261109/manifest.json)
