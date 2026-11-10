# MiniCPM5 自身历史重放：串行运行器、双 rank 入口和完整输出复核

MiniCPM5 的自身历史重放现已完成**外层串行运行器、双 rank 原生入口和独立结果复核的源码接入**。实际模型尚未运行：必须先完成当前 Nemotron 任务，再执行 MiniCPM5 完整采集和独立保存数据审计，取得实际退出记录后才能生成具体重放计划。

## 从已有证据到真实重放

运行器先认证两组已完成工作：原 84 次调用的采集、以及覆盖全部 logits、官方参考和 168 份完整缓存的独立审计。两组工作都必须有实际工具会话、原进程身份、成功退出和清理证据；完整审计结果本身不替代进程退出。

新任务取得同一串行锁后，再检查全部 1,255 个前序服务、采集的四个实际进程及独立审计进程已经退出，并复核原服务进程组为空。双 rank 入口先固定完整来源和输入，随后沿用原 SDK 的 CPU 环境初始化及 gloo TP2，调用上一轮准备的完整模型构造器与重放核心。

每个 rank 保留全部 42 次历史调用，重放其中 18 次解码；两个 rank 共 36 次、144 个物理输出行，副本去重后为 72 行。重放核心使用原模型、原桶、原内存布局、完整缓存视图和原位置编码。完整四行注意力输出必须与原记录逐字节一致。

## 完整输出要独立回读

每次保存 16,384 字节 BF16 注意力输出。两个 rank 共 36 份、589,824 字节，由结果复核器重新读取，核对原调用、独立缓存审计来源、原构造器和桶选择记录，以及完整输出哈希。两个 rank 的副本输出还必须相同，不能把副本当成独立样本。

结果复核同时要求四个官方投影分片的范围、轴、形状和原字节校验记录完整，并核对两个 rank 引用同一个官方完整矩阵。缺少调用、跨 rank 取文件、只保留部分行、修改桶、参数精度变化，或把结果提升为整模型／因果成功，都会被拒绝。

## 修正了我们包装代码中的一个参数遗漏

前一版官方权重校验包装只允许 Q/K/V/O 四个参数。核对已固定的 SDK 源码和原模型构造元数据后发现，原注意力模块还持有 `rank_util.rank`：一个不可训练的 int32 参数，用于标识当前 TP rank。旧包装会误拒绝正常的原模型。

后继 `official_shards_v2` 明确验证四个 BF16 投影和这个原 int32 rank 参数，要求值与实际 rank 相同；额外参数、遗漏投影、错误 rank 或精度仍然拒绝。驱动 v2 只切换到此校验后继。旧源码和记录保留，没有改写 SDK、模型参数或原模型精度；这属于我们的接入代码修正，不能归为上游模型故障。

## 已完成的检查及其范围

| 检查 | 通过数 | 证明的范围 |
| --- | ---: | --- |
| 实际退出证据链校验 | 16 | 合成记录覆盖成功／失败退出、身份、工具会话、完整审计范围及同一采集绑定 |
| 串行运行器生命周期 | 16 | 模拟清理失败、进程未退、服务身份变化；确认独立清理证明前不释放锁 |
| 完整结果复核 | 12 | 合成 36 份完整输出文件的实际回读与各种错误拒绝；前序退出来源由夹具模拟 |
| 运行规格与源码绑定 | 7 | 检查实际固定源码文件、原资源／科学输入，以及不完整证据提前拒绝；未启动服务 |
| 官方分片校验 v2 | 19 | 原尺寸合成 BF16 字节及原 rank 参数元数据；取代前版 16 项，不能相加 |
| 驱动 v2 | 11 | 更换校验器后重跑模拟原生调用及保存测试；原生模型计算仍被模拟 |

源码准备固定 **2,374 个文件、160,743,353 字节**，保留全部原科学输入。没有在本轮读取真实权重、缓存或模型输出，也没有执行真实双 rank 注意力。

## 运行条件和后续验收

保留 CPU 6–7、16 GiB 内存、禁用交换、两小时任务上限；输出仍限制总计 256 MiB、每 rank 128 MiB、单文件 16 MiB 和 512 个文件。根盘 28 GiB、NVMe 30 GiB 可用空间门槛保留。异常路径也必须确认任务进程已退出、进程组已清空，才能释放串行锁。

实际启动顺序是：Nemotron 实际终止／释放 → MiniCPM5 完整 84 次采集 → 独立全词表／官方 32 帧／完整缓存审计 → 根据实际完成记录生成并审查具体重放计划 → 双 rank 自身历史重放。只有全部 36 次完整输出复现，才能继续设计配对历史干预。

MiniCPM5 原 CPU 对照仍为 29／32 数值通过、top1 32／32；两个 BF16 候选仍为 28／32、top1 31／32，拒绝结论不变。原精度、冷缓存至 4096、长解码、请求槽位复用、生成／随机数／停止／文本／质量、真实 HTTP 四并发和配对性能仍是整模型条件。

2026-11-10 09:09 UTC 独立进程观察确认 Nemotron r11 原编译 worker 和模型服务仍在运行，状态为 compile；全部 42 个准备／局部诊断阶段的结果保持不变。此轮没有新增合格提速倍数。

## 源码与证据

- [串行运行器](evidence/minicpm5-layer1-own-history-runner-20261110/minicpm5_layer1_own_history_cpu_runner_v1.py)与[双 rank 原生入口](evidence/minicpm5-layer1-own-history-runner-20261110/minicpm5_layer1_own_history_cpu_worker_v1.py)
- [前序采集及审计完成链](evidence/minicpm5-layer1-own-history-runner-20261110/minicpm5_layer1_after_saved_audit_v1.py)与[完整输出复核器](evidence/minicpm5-layer1-own-history-runner-20261110/minicpm5_layer1_own_history_results_v1.py)
- [修正后的官方分片校验](evidence/minicpm5-layer1-own-history-runner-20261110/minicpm5_layer1_official_shards_v2.py)与[驱动 v2](evidence/minicpm5-layer1-own-history-runner-20261110/minicpm5_layer1_own_history_cpu_driver_v2.py)
- [源码计划及实际运行候选生成](evidence/minicpm5-layer1-own-history-runner-20261110/prepare_minicpm5_layer1_own_history_runner_v1.py)、[准备检查](evidence/minicpm5-layer1-own-history-runner-20261110/minicpm5-layer1-own-history-runner-source-checks-20261110-v1.json)和[审查及全部实际测试记录引用](evidence/minicpm5-layer1-own-history-runner-20261110/minicpm5-layer1-own-history-runner-source-review-20261110-v1.json)
- [Nemotron 实际进程观察](evidence/minicpm5-layer1-own-history-runner-20261110/nemotron9b-r11-compile-startup-observation-20261110-v2.json)
- [所有文件和 SHA-256](evidence/minicpm5-layer1-own-history-runner-20261110/manifest.json)
