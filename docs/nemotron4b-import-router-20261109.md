# Nano4B：导入来源检查与可选依赖回退

本轮修复了我们新增的导入检查器中的兼容性问题：原版本会把“可选子模块自身抛出 ImportError”也记成整次检查失败，即使上层模块按 Python 原有逻辑捕获了这个错误并选择备用实现。新的 v2 保留这种正常回退，同时把来源核验、文件读取和资源检查的失败单独拦截。

这是辅助程序接入准备中的修复。27 项实际模拟导入检查通过，尚未运行真实 SDK 或模型，也没有产生新的提速倍数。该现象不是已证实的上游模型或 vLLM-Neuron 缺陷。

| 情况 | v2 的行为 |
| --- | --- |
| 模块确实不存在；可选子模块缺依赖或主动抛出 ImportError | 保留 Python 的原始异常和上层回退 |
| 已获授权的查找器或加载器执行时抛出普通异常 | 保留原异常对象；退出该调用前仍检查运行条件 |
| 源码指纹、文件身份、加载器授权或资源检查失败 | 记为不可恢复的检查失败；即使被导入代码捕获，也不能继续完成 |
| 自定义或原生加载器初始化失败 | 调用失败后的来源核验，检查可能发生的部分初始化；原非零 SystemExit 保留 |

源码模块直接编译已核验的源码字节，已有字节码缓存不能替代这些源码。包的相对导入、原路径和命名空间搜索顺序保留；自定义加载器继续使用原对象，支持模块别名身份。查找器必须先获得授权，才会执行其代码。

新的加载器接口要求调用方处理 create_failed 和 exec_failed 两种失败回调。加载失败也可能已经映射了部分原生库，因此后续真实 SDK 接入必须核验这些状态。对来源检查本身抛出的 ImportError，v2 会转为持续失败状态，防止它被误当成“可选依赖不存在”。

27 项检查使用临时源码模块和模拟加载器，覆盖父模块捕获子模块异常、缺失依赖、查找器与加载器的原异常、自定义加载失败后的检查、来源失败被捕获、已有缓存绕过、包与命名空间、别名，以及原非零退出。原生文件用的是无效测试文件，并在加载前被拒绝，没有执行真实原生扩展。

最早一版测试曾把“存在 __cached__ 路径”误认为“读取或写入了缓存”。Python 可以只提供路径而不使用缓存；此断言已修正，另有实际有效旧缓存的测试验证源码优先。第一次失败记录和 v1 源码均保留，后续接入选择 v2。

仍需完成真实 .pth 和 six.moves 查找器、预加载模块、解释器与标准库、原生扩展和链接库的来源策略，以及主进程、工作进程和辅助程序的完整接入。本核心不自行安装导入钩子，也不独立监控 sys.meta_path 的变更；不能据此宣称真实 SDK 已兼容或整个模型已合格。

[启动恢复](nemotron4b-site-restore-20261109.md)与[双分片启动管理](nemotron4b-rank-pair-20261109.md)作为后续接入材料。当前串行队列继续等待 Dory 保存结果复核；其原有七个 FP32 检查失败仍未解释。本轮未改动模型精度、测试负载或提速统计。

- [v2 导入核心](evidence/nemotron4b-import-router-20261109/nemotron4b_import_router_v2.py)
- [27 项检查](evidence/nemotron4b-import-router-20261109/check_nemotron4b_import_router_v2.py)
- [实际测试退出记录](evidence/nemotron4b-import-router-20261109/nemotron4b-import-router-v2-tests-actual-exit-20261109-v1.json)
- [复核、来源指纹和剩余限制](evidence/nemotron4b-import-router-20261109/nemotron4b-import-router-v2-source-preparation-review-20261109-v1.json)
- [保留的首次测试失败](evidence/nemotron4b-import-router-20261109/nemotron4b-import-router-test-expectation-failure-20261109-v1.json)
- [完整证据清单](evidence/nemotron4b-import-router-20261109/manifest.json)
