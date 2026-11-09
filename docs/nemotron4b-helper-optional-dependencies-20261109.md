# Nano4B：SDK 辅助程序的可选依赖与导入边界

Nano4B 的 SDK 库路径辅助程序会触发包初始化。为准备真实运行环境，本轮补查了此前导入图中出现、且本机已经安装的六个包：pyOpenSSL、cryptography、certifi、typing_extensions、cffi 和 pycparser。

新增的 108 份 Python 源码均与安装包 RECORD 记录一致，共 1,288,610 字节。累计清单覆盖 15 个发行包、371 份 Python 源码、5,155,150 字节。原计划的 119 个指纹、此前 263 份包源码和启动文件均未改变。这里的 RECORD 校验证明内容与本地安装记录一致，不等同于独立验证发布者签名。

7 项检查通过，覆盖顶层原生扩展不能误当 Python 读取、原生扩展不得因缺少记录而被静默遗漏、大小或源码指纹不符、路径替换，以及包与命名空间的区分。这些检查使用生成的安装包样例。

## 剩余导入项意味着什么

对剩余 13 类、23 处导入记录检查了原始源码、条件分支和异常回退。它们不能简单当作“缺包清单”。

| 导入项 | 源码中观察到的行为 |
|---|---|
| StringIO | six 的 Python 2 分支 |
| annotationlib、compression | Python 3.14 相关条件分支；compression 路径还保留异常回退 |
| awscrt、backports、bcrypt、brotli、brotlicffi | 可选功能，源码具有条件判断或 ImportError 回退 |
| botocore.customizations.useragent | 导入失败时返回未修改的用户代理组件 |
| js、pyodide | Emscripten 平台相关路径；urllib3 主入口按平台条件引入该集成 |
| dateutil.tz 的星号导入 | 目标模块已在清单中；此前分析没有求值星号导出的实际内容 |
| six.moves | 由 six 的动态导入器提供，不要求存在普通的 six/moves.py 文件 |

这些是源码观察，还不是实际 SDK 导入轨迹。它们不足以认定上游缺包或计算有误，也不要求现在安装不存在的可选包。后续运行必须保持原有导入和回退行为。

## 原生库与下一步

清单现在记录四个原生文件：libneuronpjrt、protobuf 的 _message、cryptography 的 _rust，以及顶层 _cffi_backend。本轮只核对文件属性和安装记录声明的指纹，没有读取或加载这些原生文件的内容。

保守导入图覆盖 267 个 Python 模块，但真实执行仍需验证解释器和标准库、启动恢复、动态导入器、原生库及其链接依赖，并接入主启动器与实际模型工作进程。[此前准备的双分片管理核心](nemotron4b-rank-pair-20261109.md)继续保留，尚不能单独启动真实模型。

本轮没有执行真实 SDK、模型或张量分析，没有新增基线或提速。全部 15 个模型家族和原精度、冷缓存至 4096、长解码、请求槽位复用、生成质量、真实 HTTP C4 及配对性能要求继续保留。

- [累计依赖清单](evidence/nemotron4b-helper-optional-dependencies-20261109/nemotron4b-helper-optional-dependency-inventory-20261109-v1.json)
- [23 处导入的源码与分支证据](evidence/nemotron4b-helper-optional-dependencies-20261109/nemotron4b-helper-remaining-import-boundaries-20261109-v1.json)
- [检查和实际分析退出记录](evidence/nemotron4b-helper-optional-dependencies-20261109/nemotron4b-helper-optional-dependencies-actual-exits-20261109-v1.json)
- [源码复核与限制](evidence/nemotron4b-helper-optional-dependencies-20261109/nemotron4b-helper-optional-dependencies-source-preparation-review-20261109-v1.json)
- [源码及完整证据清单](evidence/nemotron4b-helper-optional-dependencies-20261109/manifest.json)
