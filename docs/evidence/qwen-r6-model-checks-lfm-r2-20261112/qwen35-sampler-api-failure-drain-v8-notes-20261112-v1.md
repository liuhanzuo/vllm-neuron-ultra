# Qwen API 异常日志收集 v8：源码交付

r6 的 baseline-capture API 日志保存为 15,008 字节，SHA256 `e00df54c92bbab721d30d53e93a75e88f5acda5ed25a3d5d4422d0fdc64a34be`。API traceback 只到 `async_llm.py:207` 的 `return cls(`，其后直接出现 supervisor 自身的通用 ValueError。独立 child-exit 记录 child returncode 为 -9；root-terminal 记录 supervisor exit 1。本次只读这些小日志/JSON，不做进程、服务、锁或 native 操作。

源码可确定的因果是：v6 supervisor 读一块管道输出，先写到日志，然后发现 `traceback (most recent call last)` 等任一原标记立即抛错；finally 在子进程仍存活时直接 SIGKILL，并关闭 selector，从未读取后续 traceback。保存证据与该路径一致。API 原始根异常尚未知；本修正只改善后续失败取证，不声称修复 API 启动原因。

v8 的六个文件在 v6 基础上只做版本接线、service 失败收集和 adapter 同步。contract、io、history、CLI 在归一化六文件版本名后与 v6 完全一致；原科学 driver、全部科学源码及 native leaf v2 均保持。v7 中间版已测试冻结，随后源码审阅发现 `Phase.finish` 的新等待路径若遇到原 guard 失败可能跳过本地 cleanup；v8 用 try/finally 保证仍执行原 owned-service cleanup，v7 保留，不再修改。

检测到原任一 API 错误标记后，supervisor 只发布一次 `api-failure.json`，绑定 phase、supervisor、原 child 身份、首个标记、时间与当时日志字节数。失败不可撤销；后续 child 即使退出 0，supervisor 也必须失败。原 capture quota/上下文错误不被吞掉。

失败收集的截止时间是 `min(首个标记时间 + 原 owned_cleanup_seconds=45, 原 phase deadline)`；无重复刷新。允许自然退出并继续转发日志至该期限前 25 秒，之后 TERM，最后 10 秒之前 KILL，期间持续读取剩余管道。原身份、资源、日志大小与期限 guard 仍运行；它们失败会立即进入原清理。最终 wait 也受原剩余失败收集时间限制。自然退出与 EOF 后保存实际 child-exit 及 `api-failure-complete.json`，明确记录 EOF、字节数、信号和实际退出；不声称 EOF 一定意味着完整 traceback。超时、资源不足或日志上限仍可能使尾部不可得，完整 cgroup 排空依旧由原生命周期流程强制完成。

协调进程在 API readiness 前后、上下文退出和执行 guard 检查失败锁存。已有失败时，它先等待同一绝对期限内的 supervisor 收集，不会抢先清理并再次截断日志。若等待本身失败，v8 仍以 finally 进入原 cleanup。原五个客户端操作仅增加调用前后的失败检查；原函数、输入和工作量不变，已失败的结果不会被接收，也不会开始下一客户端操作或下一阶段。不能把这一机制表述成瞬时取消已提交 HTTP 请求：原客户端操作内部正在进行的请求仍可能在有界终止前结束，但所有相关结果不具有资格。

新增两类 JSON 使用原通用 metadata/storage 预算及目录扫描；未增加任何 cap、时间、任务、模型、精度或采样额度。健康路径没有新增原生操作或 sleep，源码入口没有实际未来 spec/review/binding。

最终显式套件共 128 项检查通过：原 109 项源码/历史/命名空间/启动/leaf 安装检查，加 19 项失败收集模拟检查。工具 session 40069，terminal `e908fb`，exit 0；unittest 2.264 秒。测试以 fake Popen/selector/pipe/clock/signal 完整执行新 supervise，不启动子服务/进程；覆盖多块根异常尾部、child 已退出但 pipe 仍有数据、所有原错误标记、跨块与大小写、error 后 exit0、重复标记期限不续期、TERM/KILL、phase 剩余期限、stop 请求不能覆盖失败、log cap、guard 失败、client 前后拒绝与 cleanup 保证。显式套件同时安装 native import 拒绝器并确认 Torch/NumPy/ctypes 未加载。

保留的早期测试失败：首次专项 fixture 全局替换 os.read，意外影响原 JSON reader（`4ebfa2/1`）；测试 r2 改为仅替换 service 的 os 对象，执行源码未变（`042953/0`，17 项）。一次泛目录 unittest discovery 错误收录已冻结失败 fixture，并尝试导入不属于本任务的 Dory rotary 包而报错（session 68156，`228e76/1`）；没有将该轮视为通过，之后只运行显式文件名单。旧 API-intent mock 使用未经过构造函数的 Phase，缺少新增检查所需的 name，单项复现 `6acce8/1`，用 r2 fixture 补齐真实字段及 API-failure 空读取。v7 显式 126 项测试为 `817b52/0`，v8 最终为上述 128 项。请勿用目录 discover 重跑已保存失败草稿。

最终复核命令在 research 目录使用显式名字，或逐个执行这些文件：

- `test_qwen35_sampler_runtime_source_v8_r2.py`
- `test_qwen35_sampler_bootstrap_self_service_v8.py`
- `test_qwen35_sampler_virtual_namespace_v8.py`
- `test_qwen35_sampler_collected_terminal_v8.py`
- `test_qwen35_sampler_active_scope_v8.py`
- `test_qwen35_sampler_env_exec_window_v8.py`
- `test_qwen35_sampler_api_failure_drain_v8_r2.py`
- 原 `test_qwen35_sampler_split_install_v2.py`

仍需 root 独立源码审阅、未来真实 idle 历史绑定及全流程实测。原 r6 失败不可改写；本次没有运行准入、整模型合格或新提速结论。
