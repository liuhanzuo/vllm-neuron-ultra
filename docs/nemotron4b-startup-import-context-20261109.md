# Nano4B：动态导入接入与启动集成检查

本轮把导入检查接入了父进程放行、Python 启动恢复和辅助程序执行的整条测试链路。选定 router v3 与 context v2，三组共 69 项检查通过：32 项导入规则、22 项动态导入接入、15 项实际子进程启动集成。

测试使用临时虚拟环境、真实系统 Python 启动源码和模拟包、辅助程序。真实 SDK、原生库及模型仍未运行，因此没有新增基线或提速数字。

实际安装源码表明，editable 的 .pth、distutils 启动组件和 six 都会增删或检查 sys.meta_path 中的对象。若用一组固定包装对象替换原导入器，就可能改变注册、去重和模块别名行为。新的上下文保留原对象及顺序，在标准导入分发处核验每次调用，同时覆盖 importlib.util 保存的入口引用。

这条接入路径保留了导入锁、模块重新加载、查找器在一次查找中追加另一个查找器，以及已有模块的有效 spec 选择。三种已观察到的 SDK 动态导入器都有明确对象绑定规则；实际使用时仍必须核验所属模块的来源，名字或类型相同不构成授权。

启动集成中发现并修复了两类真实环境清单遗漏：

| 发现 | 处理 |
| --- | --- |
| 标准库 site 初始化会加载内置模块 pwd | 测试策略增加明确的内置模块身份检查，不把它当成普通源码包 |
| 系统 sitecustomize.py 存在，且 /usr/lib 下的路径实际链接到 /etc | 新版核验该链接的原始文本和文件身份，再读取已核验的目标源码；模块仍保留原来的文件路径 |

之前的普通源码读取规则拒绝符号链接，导致在补入 sitecustomize 后，测试甚至无法完成父子握手。新版仅支持明确记录的一跳链接；链接变化、目标源码变化、未登记的链接和额外链接链都会失败。没有通过改变 Python 搜索路径或关闭来源检查来绕过问题。

三个失败测试版本和实际退出记录均已保留。它们反映我们准备的环境清单与读取规则不完整，不能据此判断上游模型或 vLLM-Neuron 存在计算错误。

15 项最终集成检查确认：父进程放行之后才运行启动代码；.pth 注册的新导入器能够用于随后辅助程序的导入；可选子模块的 ImportError 仍可正常回退；捕获所有异常不能掩盖未核验的模块；启动错误、输出超限、路径改变和原非零退出仍按原要求处理。所有测试子进程都已回收。

实际源码清单现在覆盖 12 份启动和导入相关文本，共 125,177 字节，并单独记录 sitecustomize 链接。原八份启动文本身份未变。这份清单中的磁盘 bootstrap.py 不能证明解释器内置冻结代码与其字节一致，真实运行策略还需要独立核验解释器来源。

真实 SDK 的预加载模块、源码与原生库完整清单、自定义加载器及模块别名检查、原生库部分初始化检查，以及主程序和工作进程接入仍待完成。当前上下文依赖调用方提供这些具体策略；模拟检查不能代替它们。启动后的 no_site 标志仍为 1，完整程序等价性也尚未证明。

当前 Dory 保存结果复核仍占用串行实验位置，原七项 FP32 检查失败尚未解释。本轮只推进接入准备，全部模型范围、原精度和性能统计口径保持原定要求。

- [上一轮可选依赖修复](nemotron4b-import-router-20261109.md)
- [动态导入上下文](evidence/nemotron4b-startup-import-context-20261109/nemotron4b_import_context_v2.py)
- [支持明确源码链接的导入核心](evidence/nemotron4b-startup-import-context-20261109/nemotron4b_import_router_v3.py)
- [69 项检查及实际退出](evidence/nemotron4b-startup-import-context-20261109/nemotron4b-startup-import-context-tests-actual-exits-20261109-v1.json)
- [实际启动来源清单](evidence/nemotron4b-startup-import-context-20261109/nemotron4b-helper-startup-import-dispatch-inputs-20261109-v1.json)
- [源码复核、失败记录与限制](evidence/nemotron4b-startup-import-context-20261109/nemotron4b-startup-import-context-source-preparation-review-20261109-v1.json)
- [完整证据清单](evidence/nemotron4b-startup-import-context-20261109/manifest.json)
