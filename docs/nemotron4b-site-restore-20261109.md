# Nano4B：父进程放行后的 Python 启动恢复

为记录辅助进程的真实身份，新的 SDK 辅助程序入口需要先等待父进程放行，再恢复原 Python 环境。普通 Python 会在脚本正文之前执行 .pth 文件，因此使用 -S 推迟初始化，并在放行后调用原标准库的 site.main()。

本轮把启动恢复逻辑接到了既有的父子握手与辅助程序执行核心上。13 项真实子进程检查通过，测试使用临时虚拟环境和模拟 .pth、辅助脚本，没有执行安装的 SDK 或模型。

## 处理的实际问题

Python 的 site.main() 对部分 .pth 异常只打印错误并继续执行。因此，仅凭辅助进程最后退出为零，可能漏掉不完整的启动环境。

新的恢复核心会完整核对 .pth 文件集合、源码指纹和文件身份，在经过外部依赖检查后恢复环境，并收集有上限的启动输出。发生启动错误、额外输出或输出超限时，不接受该次恢复。它会保留错误信息，且保留原辅助脚本的非零退出值。

启动前后还检查原辅助脚本的参数、搜索路径、虚拟环境与基础解释器路径，以及用户目录排除设置。辅助脚本所在目录会替换包装程序的搜索目录。恢复只允许在同一个辅助进程中进行一次；初始化产生的副作用不会被假装回滚，失败需要保留记录并退出。

## 验证结果

13 项检查覆盖父进程放行后的执行顺序、被 Python 吞掉的 .pth 异常、启动输出与超限、文件变化、新增 .pth、父进程或依赖检查拒绝、恢复路径不符、辅助脚本修改环境、原始非零退出、重复恢复和禁止生成字节码。所有实际启动的测试子进程均已回收。

另外核验了真实 SDK 环境的八份启动文本，共 37,365 字节：两个 .pth、三个相关 Python 文件、pyvenv.cfg、系统 site.py 和 _sitebuiltins.py。记录了配置中的 Python 3.10.12、解释器链接和文件属性，未读取解释器二进制，也未执行 SDK 启动代码。原计划的 119 个源码指纹重新核验未变。

## 仍需完成的接入

这一核心还需要完整的导入来源检查，覆盖解释器、标准库、Python 包、动态导入器、原生扩展及链接库。真实辅助程序的完整运行策略、主启动器与模型工作进程仍未接好；本轮的模拟导入上下文不能代替这些检查。

调用 site.main() 后，sys.flags.no_site 仍然为 1。这一可观察差异继续保留在证据中，尚未证明与原 SDK 辅助程序完全等价。所有 Python 输出检查也仅补充父进程对实际输出管道的限制。

[双分片启动管理](nemotron4b-rank-pair-20261109.md)和[371 份包源码的依赖清单](nemotron4b-helper-optional-dependencies-20261109.md)继续作为后续接入材料。现有串行任务和全部模型验证要求保持原定范围。本轮没有新的原生基线、模型正确性结论或提速倍数。

- [启动恢复核心](evidence/nemotron4b-site-restore-20261109/nemotron4b_gated_site_restore_v1.py)
- [13 项检查及实际退出](evidence/nemotron4b-site-restore-20261109/nemotron4b-site-restore-actual-exits-20261109-v1.json)
- [真实 SDK 启动文本清单](evidence/nemotron4b-site-restore-20261109/nemotron4b-helper-site-startup-inputs-20261109-v1.json)
- [源码复核与限制](evidence/nemotron4b-site-restore-20261109/nemotron4b-site-restore-source-preparation-review-20261109-v1.json)
- [完整证据清单](evidence/nemotron4b-site-restore-20261109/manifest.json)
