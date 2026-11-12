# MiniCPM5 映射观察器：有限 backend 源码交付

仅新增 `minicpm5_sdk_mapping_observer_backend_v1.py` 和纯 mock 检查；已接受的逻辑 v1 与原 SDK diagnostic v1 未修改。没有当前执行绑定，也没有真实 ctypes、ptrace、ELF/RECORD、模型、进程/服务/锁或网络操作。27 项 backend/mock 检查通过；这不证明 native ABI、映射 errno 或模型结果。

接口仍需要外部明确的 `authenticate(authority, guard)` 与 `verify_loaded_native(ctypes, libc, receipt, guard)`。前者必须在 ctypes 导入前完成实际来源、字节身份和依赖认证；后者必须在取得 libc 句柄后验证实际已加载来源并返回 True。源码只核验返回凭据的绑定和必需字段，不能自行签发凭据，也不能把 mock 的 True 视为实际认证。当前未提供可通过这一门槛的实际 observer authority。

已核对的现有材料与固定待补数据：

| 部件 | 保存材料中的事实 | 本次未完成部分 |
|---|---|---|
| 原九项 native | `/mnt/nvme/minicpm5-layer1-operator-native-r3-sourceclosure-20261111/source-authentication.json`，SHA `45c6c67a77d5ffb9bd57684c5a04339b2587f37c690769ff0e1f1a96d9cb6dda`；是原 D-Bus/query 九项完整认证 | 原路径继续执行，新增 observer 不能替换九项或将其外推到 ctypes |
| outer Python | `/home/ubuntu/venvs/model_reference_2026/bin/python` 解析到 `/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12` | 需要相同原解释器的完整字节摘要、稳定文件身份和已有方式可接受的来源凭据；不是要求重建 Python |
| `_ctypes` / ffi | `nemotron4b-bootstrap-native-manifest-candidate-20261110-v1.json` 保存该 Python 的 `_ctypes` 为 built-in；候选记录自己注明 `actual_interpreter_authenticated=false`、`complete_runtime_dependency_closure_proven=false` | `_ctypes` 应归属原 executable 的认证，不能误用 Python3.10 `_ctypes.so`；ffi 的实际内置/动态归属须补准确依赖观察，不能凭候选 maps 无 libffi 就宣布静态闭包完成 |
| ctypes 纯源码 | 已只读原 Python3.12 的 `ctypes/__init__.py`、`ctypes/_endian.py`；前者还导入 `os/sys/types/struct`、`_ctypes/_struct`，并在导入时执行 `PyDLL(None)` | 新增外部 authority 必须覆盖这些实际源码/内置来源；最小采用固定原路径、完整源摘要及现有依赖身份规则，不导入 `ctypes.util` 或调用 find_library |
| libc 与 loader | `dory-loader-candidate-elf-inputs-20261111-v1.json`（SHA `db254b660a0f29effc9e4bb1ee0784a7dbf66177a632a94b2e600ab713123fe6`）保存 libc6 `2.35-0ubuntu3.15` 已完成 signed-package member replay：libc SHA `b2cf6c33b74d2f22543b7a469a75b538911e690f769d0b238843a49465b83793`；loader SHA `41daf7836f7fce21a40ca63a00c5caf21b05a35a76ccc075d86f1716212cdf01` | 需沿该文件 `input_refs` 的 package_result/remaining 完整凭据链以及当前文件身份核验；这是已有来源证据可复用，不能表述成“来源无法证明”，也不能只信候选摘要 |

既有可复用认证原语是 `minicpm5_layer1_sdk_batch_layout_sources_v2.py:168` 的 complete-byte authentication 与 `:215` 的稳定 native 身份核验；其 `validate_native_receipt` 严格固定九项，保持原样。新增 observer 输入需由另一个明确调用层运用相同完整字节身份规则，不能扩展九项 schema 冒充原收据。Dory 的 `dory_serial_stage_contract_v2.py:123` 展示了已接受的 reference + current identity + signed-package 链，底层 `dory_native_inventory_finish_v2.file_bytes` 可作为现有稳定读取原语参考；不引入 Dory 的环境、预加载项或额外资源限额。`nemotron4b_interpreter_origin_policy_v1.running_executable/runtime_description` 提供候选原解释器身份描述，不能单凭描述替代内容认证。

ABI 来源是只读 `/usr/include/x86_64-linux-gnu/bits/ptrace-shared.h:142`（`long ptrace(enum request, ...)`）、`/usr/include/linux/ptrace.h` 的 request/options/syscall-info，以及 `bits/waitflags.h` 的 `__WALL | WNOHANG`。factory 明确要求 LP64 的 int=4、long/unsigned long/pointer=8；`ptrace` 返回有符号 long，设置 errno 副本但只在返回 -1 时读取。GET_SYSCALL_INFO 用 88 B 缓冲区，内核报告所需长度超过 88 就失败，不能将截断内容称完整；GETEVENTMSG 用 8 B unsigned long。不读取 tracee 内存或改写寄存器、调用参数、返回值。

未来原 outer 的唯一接入顺序：完整原认证 → 取得新增 observer authority → 原 SDK child 与完整 gate → lazy factory → 原逻辑 `ObservedSDK.run()` → 仅在真实 terminal 与所有已知任务/原组排空后 `finish_wait_handoff(result)`。factory 独占原 Popen 的 `_waitpid_lock`，保留强引用，stdin 只使用非阻塞 `os.write`，完整写入后关闭原 stdin。每个 `waitpid` 只针对已知 task，保留 ECHILD/EINTR 和原始 status；backend 另留实际 terminal 防止 handoff 接受虚构收据。信号仅用于已知且当前身份一致的 task。原串行锁不由本 backend 释放。

特别注意：取得 waitpid 所有权后不得调用原 `SingleSDKAdapter.finish/drain_owned`，它们使用 Popen.poll/wait。`cleanup_pending` 不能走这些旧 helper，也不能丢弃 ports/Popen 强引用或释放原串行锁；必须保留原外层的未排空处理路径。factory 在返回前失败时还没有 SEIZE，会归还 waitpid 锁，普通原清理才仍适用。原 outer 的 900 s 整体计时、CPU/AS/FSIZE 限制继续生效；不能用 factory 或逻辑开始时刻重置原计时。

Store 窄子类只允许原四个诊断文件和 `phase-1-memory-syscalls.jsonl`、`phase-1-observer-exit.json`。继续调用原 Store 的完整库存、排他创建、不可变身份、完整读回和实际 allocation 检查；trace 计入原 8 MiB metadata/32 MiB allocated，新 receipt 与原 runtime 文件共同计入 79,008 B envelope，summary/receipt 各不超过 16 KiB。它不修改原模块全局 RUNTIME，不增加绕过 sticky guard 的 writer。原 outer 最后“四文件”的检查需在未来明确接入新子类时适配六文件，不能靠 monkeypatch 修改冻结源码。

这份交付完成 lazy ports 与 Store 计账接口。独立 native authority 的实际内容/依赖观察、full-auth outer 的最终接线与准入、真实 ptrace 支持和实际诊断仍未完成，保持原模型、环境与限额后再分别审核。
