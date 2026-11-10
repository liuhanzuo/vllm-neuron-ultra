# Nano 4B：保留 Python 3.10 与 3.12 各自的启动行为

为 Nano 4B 接入原生参考时，发现此前只针对 Neuron 辅助程序准备的启动恢复逻辑无法直接用于 Python 3.12。新模块分别保存和检查两个环境的实际前缀、路径别名及 `site` 来源；8 项测试、18 个子场景全部通过。这是接入层兼容修复，尚未运行真实模型或新增性能测量。

## 发现的差异

| 项目 | 原生参考环境 | Neuron 环境 |
| --- | --- | --- |
| Python | 3.12.16 | 3.10 |
| 禁用自动 site 后的基础前缀 | `prefix` 与 `exec_prefix` 字符串不同，后者经目录软链接指向前者 | 四个前缀均为 `/usr` |
| `site` 的实际提供者 | 解释器内置的冻结模块 | `/usr/lib/python3.10/site.py` |
| 已安装的启动钩子 | distutils 钩子，以及加入 Transformers 源码目录的 `.pth` | distutils 钩子，以及 vLLM-Neuron 的 editable finder |

旧恢复逻辑要求四个基础前缀相同，并按磁盘源码处理 `site`。这两项假设不适用于实际 Python 3.12 环境；这是本项目启动接入层的限制，不能归因于 Nano 4B、Transformers 或上游 vLLM-Neuron 的模型错误。

两个原环境分别使用 `-B -S` 做了标准库观察，保存实际解释器路径、版本、四个前缀、启动文件和路径别名。原生参考读取 5 个启动源码/配置文件、26,737 字节，Neuron 读取 6 个、27,273 字节；另各读取固定哈希的原计划。观察过程没有运行原环境的 `site.main()` 或安装包启动钩子，也没有导入模型、Torch 或 Neuron SDK。

## 修改后的行为

新恢复模块逐项比较四个前缀，保留原路径的拼写；对明确记录的目录软链接检查链接内容、目标和文件身份。同时核对完整 `.pth` 集合、配置和源码，防止启动时遗漏或多运行一个钩子。

调用方必须在导入 `site` 前检查其实际提供者，导入后再检查模块，随后在来源检查上下文内运行原 `site.main()`。Python 3.12 的磁盘 `site.py` 记录只说明该源码文件的内容，不能证明实际冻结字节码与它一致；真实解释器和冻结代码的来源检查仍需接入。

启动钩子可能打印异常后继续执行，因此恢复流程也检查启动输出；出现被 `site` 吞掉的错误时仍判失败。运行结束后再次检查原前缀、路径、配置和源码。

## 验证与保留的失败

测试使用两个真实 Python 解释器，但虚拟环境和 `.pth` 内容由测试生成。18 个实际子进程均先完成父进程身份记录与放行，再进入恢复流程，退出后均已回收。8 项测试覆盖正常恢复、Python 3.12 的冻结模块和不等前缀、缺失别名、错误前缀/来源、拒绝 site 执行、变更或新增 `.pth`、启动异常被吞掉，以及模型检查主体结束后前缀被修改。

第一轮 18 个子场景全失败，原因是测试把回执输出目录设为整个临时虚拟环境；其中的 `bin/python` 软链接被回执目录检查正确拒绝。改用独立 `receipts` 子目录后通过。原失败输出和源码已保留，生产回执检查没有放宽。

测试中的来源检查回调仍是明确标记的测试实现。这里验证了两个解释器的启动恢复行为，没有验证真实 SDK 的完整依赖，也没有证明保留 `sys.flags.no_site == 1` 与原启动方式在所有行为上等价。真实模型检查、完整启动入口、根启动器及运行准入尚待完成。

Dory r3 保存结果复核仍在运行，原七项 FP32 失败仍未解释。全部 15 个模型家族继续保留原精度及冷缓存至 4096、长解码、请求槽位复用、生成质量、HTTP C4 和配对性能要求，本轮无新增提速倍数。

- [上一轮模型检查与工作程序衔接](nemotron4b-native-worker-20261110.md)
- [启动恢复模块](evidence/nemotron4b-phase-startup-20261110/nemotron4b_phase_site_restore_v2.py)
- [测试及实际退出记录](evidence/nemotron4b-phase-startup-20261110/nemotron4b-phase-startup-tests-actual-exit-20261110-v1.json)
- [原生参考的实际启动观察](evidence/nemotron4b-phase-startup-20261110/nemotron4b-phase-startup-native-observation-20261110-v1.json)
- [Neuron 的实际启动观察](evidence/nemotron4b-phase-startup-20261110/nemotron4b-phase-startup-neuron-observation-20261110-v1.json)
- [首次失败记录](evidence/nemotron4b-phase-startup-20261110/nemotron4b-phase-startup-first-failure-20261110-v1.json)
- [源码复核与适用范围](evidence/nemotron4b-phase-startup-20261110/nemotron4b-phase-startup-source-preparation-review-20261110-v1.json)
- [发布清单](evidence/nemotron4b-phase-startup-20261110/manifest.json)
