# MiniCPM5 formal literal boundary successor：源交付 v1

本交付修复 formal import 的记录边界：37,178 B 的冻结 `-c` 源码在 admission、SDK owner、exit 及 stdout 重复展开，超出原 79,008 B runtime metadata / 16 KiB stdout 上限。只在明确列举的 argv[7] 位置引用该精确 literal；真实进程 argv、内存 owner、原鉴权和限制保持完整。

本次只完成源实现和 stdlib/mock 检查。未执行 native、未观察/获取锁、未创建当前运行绑定、未重跑模型，也没有新增性能结论。历史 r2 是 session 85296 / terminal 5fa419 / exit 1；它在 admission 写入预算处失败，SDK 尚未启动。r2 的完整动态 admission 未落盘，不能把任何下列 fixture 数字称为 r2 的真实失败总字节数。

## 固定边界与传输

- `minicpm5_sdk_mapping_nocache_entry_v6.py` 仍为原 37,178 B，SHA256 `bbf5affece28e3d6356ce27d48ccdb7f4d707fbf6717808a69ca1f61222763d0`。原早期 source fence、完整实际 -c argv 和唯一 900 秒开始点不变。
- admission 仍保留原顶层 schema 和完整 inline SDK plan。因此冻结 Entry.load_plan 可直接验证 wire 文件并取得原 plan；仅两处 outer owner 的 argv[7] 使用固定 literal 引用。
- 新 SDK successor 在构造 SDKEntryGuard/native owner 前完整 decode。outer 和 SDK 的 AdmissionMetadata 都先执行原 Metadata.read 的读取、hash、identity、guard，再仅对同一精确 admission_ref decode；原 Lease.metadata 对象保留。
- owner 一处、exit 两处 literal 位置同样编码；owner 未 ACK 时保留 owner=None 和单独的实际 argv。所有非 literal 参数、PID/PPID/cgroup/start_ticks、实际 returncode、parent wait/drain 字段均完整保留。
- 每份 wire 的文件 SHA 与完整展开对象的 SHA/字节数分别验证。固定字段、固定位置、固定源文件/hash/字节数、canonical JSON 都要匹配；不读取任意引用，不接受部分恢复。
- Store 沿用原创建、identity、只写一次、fsync、完整 readback 和 aggregate accounting。没有第七个输出文件。真实 start 在 Popen 前重新检查预算绑定与已落盘 admission 原字节。

## 实测字节与预算

以下是实际 serializer 与原 Store 对具体源测试输入产生的字节数。输入包含真实历史 r2 owner/plan/witness 引用，但未保存的动态 descriptor/types/time/SDK owner 明确使用 mock fixture；不是新运行证据。

| 项目 | 实际 wire 字节 | spawn 前预留字节 |
|---|---:|---:|
| admission.json | 34,989 | 34,989 |
| phase-1-owner.json | 902 | 921 |
| phase-1-exit.json | 1,773 | 1,800 |
| phase-1-observer-exit.json | 394 | 16,384 |
| runtime 合计 | 38,058 | 54,094 |
| 原 runtime 上限 | 79,008 | 79,008 |
| phase-1-summary.json | 210 | 原完整 16 KiB summary 额度 |
| phase-1-memory-syscalls.jsonl | 27 | 原 trace 额度 |

该 fixture 的展开 admission 为 110,132 B，wire 为 34,989 B。完整 prospective runtime 预留 54,094 B，距离原上限剩余 24,914 B。未来实际 admission 和实际 argv 必须在 Popen 前重新全量计算；这些测试数值不替代未来实际预算。owner/exit 的预算只用 Linux PID 和 canonical start_ticks 的最大数字宽度、实际 parent/cgroup/argv 计算，不生成虚假 owner receipt。原完整 metadata/allocation 额度及每文件 2 MiB 限制也继续检查。

| 整条输出 | 实测字节（含唯一换行） | spawn 前界限/容量 |
|---|---:|---:|
| 正常 stdout 六文件引用报告 | 1,394 | 1,419 |
| 缺 observer receipt 的失败 stdout | 2,870 | 3,525 |
| 真实 _persist_terminal 失败 stderr，OSError fixture | 1,962 | 原整条 16,384 |

正常与 unavailable stdout 两种完整 schema 均在 reserve_runtime 中按实际 root/argv、六个固定文件名、原每文件上限全量预序列化；输出时再次检查。失败 stderr 预算的完整基础 wire（异常名为空，最大 owner/exit 标量）为 1,982 B，异常名的序列化字节可用余量为 14,402 B；这不假设任意 Python 异常类名都有某个字符上界。实际异常名仍全量编码，超限不截断。二次编码/打印失败重抛原持久化异常，不声称成功产生了完整回退报告。

## 成功与失败证据的区别

正常 stdout 引用原六个已存在文件，保存完整原 result 的展开 hash/字节数，并在打印前通过真实 Store.read 完整重建相等。observer receipt 原先不声称自身写入成功；consumer 只在验证 durable 文件后恢复原内存 result 中的保存成功字段。

缺 durable observer/terminal 对时，报告明确为失败。terminal_wire 仍包含可无损恢复的完整实际 terminal/owner/argv/returncode，另有独立 wire SHA。observer payload 明确 unavailable，仅给原完整 payload hash/字节数，不声称整份 observer 已可恢复，也不输出不存在的文件引用。真实 _persist_terminal 写失败直接编码完整 terminal 事件，保留原 reporting_error_type，并重抛原错误；不尝试弱化 writer 或创建补写文件。

如果 inventory/reference/read 自身失败，renderer 沿用原异常退出，不声称 durable 对已成立。如果异常类名极长或 stderr 不能写，终端回退也可能无法输出；测试证明原异常对象、receipt_saved=False 和无伪造 exit 文件仍保留，未将这一极端路径算作成功恢复。

## 检查入口与覆盖

入口：`PYTHONDONTWRITEBYTECODE=1 python3 -B -m research.run_minicpm5_formal_literal_source_checks_v1 --output research/<新的唯一源检查文件名>.json`，cwd 为实验根目录。证据不能覆盖旧文件。

最终 25 项全部通过；记录完整 unittest 日志、源码 hash、实际 fixture 序列化计数。8 个旧源码与已保存 r2 bootstrap hash 完全相符；native 模块清单为空。完整 worker 主体在去掉两项明确 boundary 操作后 AST 相同，outer finally 和原 observed adapter 函数 AST 相同。

实际方法测试包括：outer.source_entry → 原 Store admission → 原继承 authenticate_complete_prerequisites → start 在 Popen 前检查；冻结 Entry.load_plan → SDK successor 在 guard/native 构造前 decode；原 Metadata 类的完整 hash/identity/guard；全预算超限在 Popen 前拒绝；实际 adapter.result → renderer → consumer 完整重建；实际 _persist_terminal 成功/失败只尝试一次；signal -15、未 ACK owner=None、cleanup、非零退出；错误 literal/hash、未知字段/引用、缺失/篡改文件；完整 stdout 预预算；超长异常名与 stderr 二次失败。仅 OS/native/auth 端口被明确 mock，真实 Store 写入均在可清理的临时目录。

## root 后续绑定要点

原 prepare_minicpm5_observed_formal_binding_v1.assemble 的 bootstrap 来自旧 manifest additions，不会自动把 additional_source_refs 的新文件加入 actual bootstrap。后续版本需同时绑定完整原 source map 和本交付八个 runtime 源码，更新 outer/SDK bootstrap、outer source_extension，以及两个角色的 successor entry path/module：outer_source_entry_v6、worker_source_entry_v6。launch 使用 v3；真正 -c payload 与 sdk_entry_source 仍是冻结 nocache_entry_v6。不要仅替换 final 引用后假设旧模板已包含这些源码。

原科学计划、1,284 行、完整 source map、九项 native 检查、冷缓存/长解码/槽位复用要求和原 serial/lease/wait/drain 语义仍由原流程负责。本交付不是当前 accepted review/runtime plan/policy/catalog；独立复核通过后由 root 决定下一次 formal 运行。
