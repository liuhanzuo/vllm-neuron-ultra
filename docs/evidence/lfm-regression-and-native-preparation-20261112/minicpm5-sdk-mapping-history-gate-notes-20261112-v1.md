# MiniCPM5 历史 import 边界：原源码保留的前置 gate

本版解决的是源码入口顺序，新增 `minicpm5_sdk_mapping_history_gate_v1.py`。它提供一次性的 `HistoricalImportGate`，把独立 bootstrap 认证放到第一次原历史导入之前；不修改、复制替换、AST 改写或伪装原历史模块。实际 bootstrap descriptor/limits/guard、原生认证 callbacks、最终 outer/worker 尚未实现，因此本版不是可直接启动的诊断。

## 最小真实导入链

旧 outer CLI `run_minicpm5_pv_sdk_import_diagnostic_v1.py:8` 顶层导入 `run`，旧 worker `minicpm5_pv_sdk_import_diagnostic_worker_v1.py:7` 顶层导入 `worker`，都先于各自函数中的认证逻辑。

```text
minicpm5_pv_sdk_import_diagnostic_v1.py:19
  → minicpm5_layer1_fixed_input_pv_adapter_v3.py:25
  → minicpm5_layer1_fixed_input_pv_prerequisites_v2.py:12
  → audit_minicpm5_layer1_saved_operators_v1.py:21
  → compare_minicpm5_layer1_saved_operators_v1.py:24
  → audit_minicpm5_layer1_operator_native_result_v4.py:20
  → minicpm5_layer1_operator_native_process_adapter_v7.py:29
  → minicpm5_python_trace_lifecycle_v4.py:7 import ctypes
```

这是本地源码 AST 广度优先得到的最短链。完整静态图保存在 `minicpm5-sdk-mapping-historical-ctypes-import-chain-20261112-v1.json`：233 个本地源码节点、697 条边、逐个完整 SHA256。可达图中只有这一处被列举 native/model 包的顶层 import。函数内部和 `__main__` 块不纳入这一静态边；该图不是动态/外部依赖闭包证明。

该 `ctypes` 导入本身会执行 Python 的 `PyDLL(None)` 初始化，认证必须发生在第一次导入原 diagnostic 之前。不能只放在 `run()`、`SingleSDKAdapter.authenticate_complete_prerequisites()` 或 observer backend 的 CDLL/ptrace 之前。原 `outer_support_v3` 的 acquire helper 自身也导入历史 runtime，因此直接提前调用旧 acquire 同样不构成纯 bootstrap。

## 后继接入顺序

1. 新 outer 用独立已审阅的纯 bootstrap helper 保留原描述符/owner/lease、原限额和早于认证的绝对截止时间；当前 gate 不实现、不取得、不关闭该描述符，也不设置 timer。bootstrap 不得通过原历史模块来取得这些 helper。
2. 构造 `HistoricalImportGate(bootstrap, authority, guard=..., authenticate_bootstrap=..., verify_loaded_native=..., verify_loaded_history=...)` 并持续保留对象。bootstrap 包含原完整 source map、原绑定、原 limits/caps 和相同 `outer_deadline`。
3. `load()` 在临时 import fence 下调用明确提供的 bootstrap authenticator。fence 拒绝 ctypes/_ctypes/Torch/NumPy/Neuron 及原 diagnostic/lifecycle 的新导入，并在成功/失败退出时恢复原 finder 列表；不会在 sys.modules 插入 fake ctypes。fence 前后还检查已加载模块，避免已经越过边界后假称完成前置认证。
4. 外部 authenticator 必须先验证真实 origin/current bytes/稳定身份/完整依赖、全部即将执行的 source pins，以及原描述符/限额。gate 只检查收据形状与绑定，不能凭布尔字段自行认证。验证完成后才允许 ctypes 的第一次实际导入、准确 libc 路径加载及实际 provider 核验。
5. provider 通过后按原 canonical 名称完整导入未改动的 `research.minicpm5_pv_sdk_import_diagnostic_v1`，并要求外部 loaded-history verifier 核验完整实际来源。原 233 个源不删改，原 lifecycle 也仍然导入真正的 ctypes，只是此时已经完成独立来源认证。
6. 将原已持有 descriptor 绑定到原 Lease/PVGuard/Metadata，然后调用 gate 的 `authenticate_original()`。它直接调用冻结的 `original.validate_spec`、`original.auth.authenticate` 和 `original.bind_authenticated`，保留所有旧历史/catalog/物理检查及原九项认证路径，不用 bootstrap 收据替代它们。完成后要求原重建 source map 与 pre-import map 完全相同，并检查原 witness 摘要。
7. 只有原完整认证通过，`adapter_inputs()` 才返回原 module/authenticated/context 对应输入及原 absolute deadline。接下来才能使用既有 observed adapter v3。不能调用旧 `run()` 再次取锁，不能调用 `impose_original_limits()` 重启 900 s；最终 outer 还需显式写出此前已接受的 admission/Store/adapter/cleanup 接线。

原 SDK worker 的 Python3.10 环境与 outer Python3.12 不同，本 gate 的 ctypes authority 固定原 outer Python3.12。不能把它直接用作 worker authority，也不能用 outer receipts 掩盖旧 worker 提前导入。worker 后继需要按原 release/gate 和其实际解释器来源单独接入；本版不声称该 worker 已修复。两处问题共用同一导入链事实，但权限与解释器认证必须分别成立。

## 检查与交付边界

`check_minicpm5_sdk_mapping_history_gate_v1.py` 实际 terminal `53e951` / exit 0，24 项，0.093 s。全程 rejecting finder 禁止真正 ctypes/_ctypes/Torch/NumPy/Neuron 导入。原历史仅作为源码读取，所有 native/provider/module/descriptor/guard 端口都是 fake。

检查覆盖认证前顺序、认证 callback 试图导入 ctypes/原历史时在加载前拒绝、原有预加载模块拒绝、失败后不重试/不丢弃 partial handles、错误来源/失败 provider 不进入历史、原完整认证必须真实调用相应接口、完整 source map 不允许少/改/多任何 pin、错误 witness/lease/shared guard 拒绝、原 deadline 原样传递、旧 run 与 limit reset 不调用。另完整复核 233 个已记录源的当前 SHA 与导入链行号。

这组 mock 不证明真实认证器正确、实际 native ABI 或模型表现。调用者须使用已审阅的 callback；缺少真正 callback 时 gate 没有默认授权路径。bootstrap source map 的完整性在原 full auth 后才能独立重建核对，所以 bootstrap 认证本身必须先依照外部 root 审阅的完整 map 验证可执行代码，不能将最后的等式校验误当成执行前认证。

保持所有先前 executed files 不变；早期 ctypes 测试范围偏差记录继续有效。当前仍未执行任何原历史/native/model 模块、SDK 子进程、真实锁、服务、下载、发布或运行绑定，没有新增模型资格或提速。
