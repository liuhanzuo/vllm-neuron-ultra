# Qwen3.5 host sampler：串行运行适配器源码交付 v1

本次交付的是可审阅的运行源码和 33 项纯测试。没有创建当前执行 spec、review、运行 handoff 或排队项，没有查询或启动当前 MiniCPM 运行，没有获取原串行锁，也没有执行 torch/vLLM/Neuron 原生实验。本次不产生模型合格或提速结论。已冻结的 sampler science v1 十个文件保持逐字节不变。

## 运行拓扑与真实退出

外层显式 CLI 持有原 `serial-enrollment.lock`，仍为 device 66306、inode 8294421。只获取一次，不关闭后重新获取，不向子进程传递 FD；原 `descriptor_serial_lock_proof_v1.prove` 观察该 FD，外层代码负责连续生命周期。一个新 coordinator systemd unit 提供冻结 driver 所需的真实 root owner；任意时刻最多一个当前 phase unit 与 coordinator 并行存在。全套有限目录为 20 个 phase unit 加一个 coordinator。

每个 phase 的 stdlib supervisor 在真实原生导入前等待独立父方核对 PID、start ticks、argv、cgroup、InvocationID；随后通过实际 Popen 启动独立原生 leaf/API bootstrap，再次等待真实 child ACK。`systemd-run` 客户端成功不算模型成功。完成必须同时具备 child 的实际 Popen.wait 返回码、相同 systemd 主进程的实际 ExecMainCode/ExecMainStatus/Result，以及递归完整 cgroup 的两个稳定清空观察。

API bootstrap 最终 exec 原来不变的 `server_with_qwen35_split_v4.py`，使用原冻结 worker 类、模型 case、TP2/B4/context4096、原环境和配置。正确性 capture 与两个 measured 阶段各自使用新服务；正常结束由原 supervisor 接收专属 stop-request 后等待 API 真实退出。失败清理只向经过身份核对的当前 unit 发送 SIGTERM/SIGKILL，不使用全局 kill，也不使用可能使 transient unit 在终态审查前消失的 `systemctl stop`。

快速启动失败有明确分支：若未观察到活跃 owner，则保留真实 installed InvocationID/ExecMainPID，标记未取得 start ticks，不伪造进程成功；未确认 PID 完全消失和完整 cgroup 清空前不释放。已取得 owner、pending invocation 或未 ACK terminal 后，unit 突然 not-found 一律不能降为“从未启动”。无法确认清理时，停止科学工作并保留原锁等待 root 处置；有限科学运行 deadline 不意味着允许带着残留进程释放锁。

## 完整历史与实际来源

完整 lineage 使用原 recovered_completion_v2 → native_overlay_v3 → latest serial catalog_v4，保留全部历史服务、owner、真实成功/失败 tool exit、原始源码和输入。物理检查使用原 current_physical.collect_pair，两次完整遍历全部历史树；没有用摘要计数替代 1308 个 Nemotron 服务及后续完整 scope。

新 Mini recorder_v3/auditor_v4 的 source_binding 通过已审查 prepare_v7 的 executed_extension 分派重建完整继承源码图；不交给旧 producer-only 验证器。此处不调用 prepare/bind，不写任何新 Mini 绑定，也不在 live coordinator 阶段调用要求当前 idle tip 的准备包装器。外层 admission 前仍检查原完整最新 idle/publication chain，运行中则要求 root 新写的 lossless running handoff 与真实外层/协调器 owner 一致。

运行 handoff 是未来 root 的最后一次独立 ACK：CLI 先给出实际 coordinator owner 和 admission 路径，等待 root 记录真实工具 session，继承完整前一 handoff 全部模型状态后才释放原生工作。代码不编造工具 session、terminal chunk 或外部退出。

完整约 47,823 项历史源码图保留在认证 spec 中，不塞进环境变量。worker 使用严格小型子集：冻结 science 文件、已固定的实际 host 依赖、新 runtime 源码以及启动/DP 默认配置源码。每项必须是完整认证图的相同 pin；单项文本小于 4 MiB，单个 host identity 环境字符串不超过 64 KiB。完整 spec 有独立 64 MiB/100,000 项上限，小 receipt 仍为 8 MiB。

## 科学协议没有缩减

独立小型 native preflight 在全模型之前：实际四个 worker 类导入/pickle、42 个真实张量正向 PacketBuilder→Decoder roundtrip（8 dtype ×5 布局，加 scalar/masked）、真正单元素 stride0、一个非空 expanded stride0 的预期拒绝负控、序列化不改变 CPU RNG，以及冻结的完整 13 个命名 sampler 拒绝控制。此前文字中的“14 个”是计数错误，新报告按实际完整列表记录，没有增删原列表。

非空 expanded stride0 不声明受支持。源码审查没有发现当前实际 packet 字段的生产点会创建此类重叠布局；如果将来出现，必须失败并准备新 codec/leaf，不在这里偷偷把原 stride 元数据规范化。真实原生 preflight 尚未运行。

之后调用冻结 driver，全部两臂 state-copy、numeric32、cold84/672 zero checks/12 graph routes、long1541/87/slot reuse、203 原始向量 exact、capture/replay、原 13 控制、smoke、quality96/69正确且全部文本一致、initial120 和原 5×3 full HTTP 保持不变。preflight 使用外层 attempt UUID，冻结 driver 另生成且只绑定自己的唯一 UUID。

DP1 证据来自已固定的启动/parallel 源码、原 argv、实际 API 进程初始环境摘要，以及无 `VLLM_DP_*`/rank/旧 sampler 开关覆盖的干净环境；不声称直接读到了内存里的 parallel_config 对象。没有打开开发 API、添加 sitecustomize 或替换原 entry。首个 diagnostic HTTP 前检查只有实际 writer-owner、零个已写 packet、无 capture receipt；不把零已写包冒充 observer seen/call_id 为零。原实际 B1/B4 各 16 包门槛保持，发现原生/capture/context 异常必须中止。

## 资源与原子发布

根盘和 NVMe 可用空间底线仍为 28/30 GiB；新输出和临时目录均在专属 NVMe namespace。开始前检查 2 GiB 的剩余输出预留，运行中观察分配块、可用内存、磁盘底线和 unit 资源。

2 GiB 总输出预算分解为：模型序列化 832 MiB、两臂 capture 512 MiB、全部日志 512 MiB、metadata 128 MiB、临时文件和目录 64 MiB。模型单文件独立 512 MiB 上限，能容纳原 long-decode-validation.pt 的 287,083,673 B；两臂原序列化合计 804,674,198 B，不再用原始向量字节数代替实际 pt 存储。日志为六个 server×64 MiB、14 个 leaf×8 MiB、coordinator16 MiB。每类、单文件、条目数量、总量都有有限门槛。

保留原 server 26/25/12 GiB memory/high/swap 和 14,400 s 生命周期；model leaf 为 26/25/4 GiB，CPU leaf/coordinator 为 4/3/1 GiB，另有单阶段、启动 ACK、root ACK、清理和 180,000 s 总期限。测试没有实际配置这些资源，它们仍须未来安装时从 systemd properties 逐项验收。

原科学代码中存在 .json.tmp rename，冻结 worker 附件则是直接写。runtime 对仅已知 .json.publishing 同 inode hardlink→unlink、.json.tmp rename、声明 phase TMPDIR 下短命文件的消失，进行有限稳定重试；未知 hardlink/symlink/路径消失仍拒绝。/health 真正 ready 后才读取完成的 worker 附件，避免把正常半写 JSON 当故障。终态输出必须两个完整库存快照一致。

## 测试与剩余事项

最终只运行 `python3 -B research/test_qwen35_sampler_runtime_source_v1.py`：33/33 通过，实际工具 chunk `5d3b3e`，exit 0。覆盖 inert import/--help、完整历史与小 host 图分离、原 frozen 文件哈希、预检与 driver UUID、真实 live/terminal 区分、未 ACK 失败、已知 wrapper→Python exec、API intent→exec、原子发布/删除交错、未知链接拒绝、完整 scope 中 owner 已存在而 unit 消失、外层使用实际 drained coordinator 对象并在清理后释放锁。

首次错误使用全目录 unittest discover 时，旧 Dory 包额外导入 torch，在系统 Python 因模块不存在失败，整次 exit1；没有运行原生实验。后续新增测试的 fixture/缩进错误和修复后的真实结果都保留在测试 receipt，不覆盖为成功。测试中的所有服务、owner、锁、运行 handoff 都是注入 mock 或自动删除的临时目录；不能当实际 native/systemd 资格证据。

未来仍需 root：等待当前 Mini 及 recorder/audit 真正结束；完成最新完整 catalog/source closure；创建精确 future spec/review（含所有 runtime/启动/历史依赖）；独立复跑这些测试并审查源码；核对原 serial lease、完整双次 predecessor drain、资源和 frozen artifact；然后显式运行 CLI 并记录真实工具 session 的 running handoff。首次 native preflight 和实际 systemd 接线未执行，失败必须保留真实失败并停止。只有全部冻结 full gates、真实服务退出和独立最终审计通过后，才有资格讨论 sampler 提速；本次没有新倍率。
