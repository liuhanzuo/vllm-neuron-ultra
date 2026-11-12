# MiniCPM5 实际 bootstrap 与原生认证源码交付

本次完成可以由未来受审 outer 调用的具体实现，不再把核心检查交给未实现的布尔 callback。仍为 source-only：未创建实际 authority/spec，未读取当前原生 payload，未调用真实 process/lock/signal/native 操作，也未接完 outer/worker。

## 当前冻结源码与检查

- `minicpm5_sdk_mapping_bootstrap_descriptor_v2.py`：实际原锁取得、完整 descriptor proof、原资源一次初始化、绝对 deadline sticky guard、向原 Lease/PVGuard/Metadata 交接。
- `minicpm5_sdk_mapping_native_closure_v1.py`：纯 ELF64/映射检查以及正向 FFI 定义证明。
- `minicpm5_sdk_mapping_native_authenticator_v2.py`：稳定完整字节读取、保存来源链重建、当前原生目录认证、实际 provider 和源码导入检查。
- `check_minicpm5_sdk_mapping_bootstrap_descriptor_v2.py`：19 项通过，actual tool `e0d970` / exit 0，0.039 s。
- `check_minicpm5_sdk_mapping_native_authenticator_v3.py`：26 项通过，actual tool `053652` / exit 0，0.138 s。

测试命令均为 run root 下 `python3 -B research/<上述检查文件>`。descriptor/process/resource/signal/lock 使用 fake；原生文件使用内存中的 synthetic ELF，ctypes/Torch/NumPy/Neuron 进口全程拒绝。来源链检查只读取已有 `.json` / `.json.gz`，没有打开 archive 或 ELF payload。旧执行版本不修改。

## bootstrap 的实际行为

复用原 `descriptor_serial_lock_proof_v1.prove` 与 `minicpm5_layer0_execution_protocol_v4.proc`，这两个 helper 的导入链不触发历史 ctypes。按原 policy 检查路径祖先、固定 inode、CLOEXEC、open 前后身份，执行一次非阻塞 FLOCK，并保留原 fd。没有第二把锁，也没有服务/子进程入口。

资源仍为 AS 2 GiB、CPU 600/605 s、CPU6、nice 至少19、FSIZE 2 MiB、wall 900 s。初始化记录绝对截止时间，安装一次原 alarm；每次 guard 验证同一 handler、`0 < remaining <= deadline - now <= 900`，并完整重验 descriptor。alarm handler 把首个 TimeoutError 永久存入 failed；捕获后不能继续执行。只在显式确认未启动子进程且已排空时才能关闭尚未交接的 fd；已经 adopt 的 fd 只能由原 outer cleanup 释放。

原 PVGuard→ComparisonGuard→AuditGuard 的构造源码仅初始化 lease/contract/resolved/last/storage，没有新 deadline 或 alarm。adopt 使用原 PVGuard 的窄子类，在每次原检查前后调用同一个 bootstrap guard；原 storage、历史、lease 检查全部保留，900 s 不重新开始。

## 现有来源与实际认证步骤

| 对象 | 已有独立来源 | 具体后继检查 |
|---|---|---|
| 原 uv CPython 3.12.16 executable | 官方 python-build-standalone archive `ce4f6d0a…` 的完整对照，executable 31,028,464 B、SHA `23bc148d…` | 对同一原路径重新完整读到 EOF、SHA、打开前/后/末尾身份；核对 /proc/self/exe，解析原位置 ELF |
| `_ctypes` / `_struct` | 保存 manifest 显示 built-in，归属原 executable，不能改用系统 Python3.10 扩展 | sys.builtin_module_names 正向记录；导入后检查真实 builtin ModuleSpec、BuiltinImporter 与无 module_file |
| FFI | 目前没有独立完成的当前符号归属记录 | 对已经完整认证的原 executable 解析符号表，必须在可执行 LOAD 中找到六项真实函数定义且没有未解决 ffi_* 符号；缺失或已 strip 掉必要证据就拒绝，绝不以缺少 libffi DT_NEEDED/maps 推定成功 |
| ctypes 源码 | 原 archive 中 __init__.py/_endian.py/struct.py/types.py 的 exact rows | 完整重读并认证源；后续 ctypes/history import 使用具体 SourceFileLoader.get_code，只编译已核验源码，拒绝旧 pyc |
| libc 与 loader 等七项 | libc6 2.35-0ubuntu3.15 signed Release/Packages → libc6.deb → 完整 package-member replay | 重建已有 JSON/JSON.gz 的 pin/签名记录/包摘要/成员摘要链，再完整认证当前七文件身份/bytes；当前 libc 的 hardlink count 被实际记录与重验，不擅改链接 |
| 实际 provider 闭包 | 保存 Dory copied-path ELF 只作线索，不足以证明原路径选择 | 当前 own /proc/maps 必须覆盖精确 executable+七库，每个 inode/device 与刚认证的文件一致；所有 NEEDED 有唯一实际映射 SONAME，PT_INTERP 从原路径解析到认证 loader；未知 native/anonymous executable 映射直接拒绝 |

完整原生候选是 8 文件，共 34,495,336 B。认证器不会将当前候选数量当成证明：每个文件必须有独立来源、完整当前 hash、稳定身份、可用 ELF、当前 mappings。正向 FFI 测试使用合成数据；实际原 executable 是否保留足够符号尚未观察，不能保证未来该门槛一定通过。

`verify_loaded_native` 还检查真实 ctypes 源 loader 的 owner，并取实际 `libc.ptrace` 函数指针，要求落在认证 libc 的 executable mapping 内；不执行 ptrace。backend 再验证、loaded-native/history 检查以及 source import 一旦失败，均永久 latch 第一项异常。

原九项 D-Bus/query native 收据与完整历史重建仍由冻结的原 auth.authenticate 路径执行。observer 的8文件来源不会替代原9项，输出 evidence 明示 `original_nine_native_authenticated_here=False`；history gate 只有原完整认证和完整 source map 精确相等后才开放 adapter 输入。

## 尚未接入的真实入口边界

SourceFinder 只保证它安装之后的 ctypes/history imports。当前 bootstrap 自身此前加载的 gate/backend/closure/helper 仍可能来自既有 pyc；`-B` 只禁止写入缓存，不能禁止读取旧缓存。已加载 ROOT 模块的磁盘 SHA 检查不等于其 executed code/pyc 来源证明。实际 outer successor 必须先采用已审阅的无缓存入口或经核验为空的 pycache prefix 等策略，并认证 bootstrap 自身执行来源；本交付没有实现或假定这个入口。

还需 root-reviewed concrete authority/source map、完整 outer admission/Store/observed adapter/finally 接线，以及 SDK Python3.10 worker 的独立入口来源/release 验证。outer 的 Python3.12 authority 不能外推给 worker。原实际队列/串行槽位未碰触，没有模型、精度、科学门槛、冷缓存/长解码/槽位复用测试的变更。

首轮 authenticator 检查 `510d64` / exit 1 的唯一失败来自异常类型预期：测试尝试默认入口时，SourceFinder 在系统 Python3.10 ctypes 的源码加载前就拒绝未认证路径，而测试预期后续 finder 的 AssertionError。结束断言证明 ctypes/_ctypes 均未加载。后继测试在 importlib 调用边界使用拒绝 stub，不尝试任何真实 native import 路径。更早的 ctypes 范围偏差记录也完整保留，不因本轮通过而撤销。
