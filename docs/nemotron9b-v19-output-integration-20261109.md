# Nemotron 9B：tiled transpose 文件规则接入 v19 草稿

2026-11-09：把 r9 的文件名修正接入新的 v19 执行源码，编译期间的容量扫描与最终证据目录扫描共用同一套严格规则。**16 项源码和文件系统检查通过，尚未创建 v19 运行规格或启动新编译。** 真实后继历史及元数据依赖仍待接入；原完成检查会拒绝当前新增的 Dory 实验目录，因此该草稿还不能用于实际启动。

原有 52 个阶段、模型提交、BF16、所有资源上限及验收要求保持不变。14 个执行模块的函数体仅替换版本引用；协议函数体除必需源码清单外保持原逻辑。原 v18 文件未改动。新的根目录路由模块使用独立名称，避免研究目录中同名候选模块因 Python 搜索路径先后而意外替换实际导入。

文件系统检查重建了原 r8 的 44 个链接和本次 r9 的 32 个 tiled transpose 链接。生成文件正文使用明确标识的合成字节；已安装资源沿用原安装清单与哈希。旧 v18 容量扫描在新文件名处复现拒绝，v19 容量扫描及完整证据扫描均通过，分别识别 17 个生成调试文件、17 个安装资源、10 个旧格式数值文件和 32 个新格式数值文件。两条路径均拒绝跨编译单元目标、名称内重复子图号不一致、硬链接目标和未在清单中的安装资源。

最终证据扫描实际读取并计算合成目标字节的 SHA256；更改目标内容后，记录的哈希随之改变。容量检查只检查文件结构和大小，仍使用原磁盘预留标准；测试中模拟充足容量，不代表机器的实时可用磁盘。16 项是本套最终用例数量，不能与此前 14 项解析器检查相加宣称新增 30 项。没有重新读取真实 r9 数值张量或运行 Neuron。

下一步是在 Dory 当前任务实际退出并完成所需复核后，把真实 r9 失败释放和 Dory 历史接入新协议，完成所有新增元数据及源码依赖认证，再创建新的固定运行规格。完整实验仍须执行原 52 阶段，包括冷缓存至 4096、长解码、槽位复用、生成/RNG/停止/文本、质量、真实 HTTP C4 和配对性能。失败原实现不能作提速分母；本次没有新增 Nemotron 提速倍数或新的上游缺陷结论。

[协议源码草稿](evidence/nemotron-v19-output-integration-20261109/nemotron9b_package_protocol_v19.py)、[共同链接规则](evidence/nemotron-v19-output-integration-20261109/nemotron9b_compiler_tiled_links_v1.py)、[16 项检查](evidence/nemotron-v19-output-integration-20261109/check_nemotron9b_package_exec_v19.py)、[完整测试日志](evidence/nemotron-v19-output-integration-20261109/nemotron9b-v19-integration-tests-20261109-v1.log)、[检查记录](evidence/nemotron-v19-output-integration-20261109/nemotron9b-v19-output-integration-tests-20261109-v1.json)、[审阅与未完成事项](evidence/nemotron-v19-output-integration-20261109/nemotron9b-v19-output-integration-review-20261109-v1.json)、[发布清单](evidence/nemotron-v19-output-integration-20261109/manifest.json)。

源码依赖本机固定模块、环境、模型及历史证据，本目录不能直接作为完整可运行基准。
