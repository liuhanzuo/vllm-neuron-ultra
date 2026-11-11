# LFM3 NKI 0.5：显式整数控制 API 后继

本交付只修改新建的后继文件，读取源码与小型来源清单，执行纯标准库测试。没有导入真实 Torch/NKI/NumPy、读取模型或 capture 字节、运行模拟器、解析真实完整 history、检查进程/锁/服务或启动串行工作。真实 r4 原始失败、全部旧来源和安装 SDK 文件保留；没有创建当前运行 spec、review、catalog、handoff 或 terminal。

## 原失败与这次限定的修改

Root 已记录 r4 的真正非零退出：原实现与恢复原实现的 identity/writeback 失败保留，候选已完成 identity/copy/permutation，随后 selector 报 `AttributeError: 'bool' object has no attribute 'shape'`。原结果没有内层 traceback；下面是直接源码路径与纯控制测试支持的解释，不冒充那次已经保存的实际内层栈。

安装的 `_simulator.py` 直接调用 `kernel_fn`。`NkiTensor` 和其 `NKIObject` 基类没有 `__eq__`、`__sub__` 重载。因此原 selector 中 `mapping == destination` 走 Python 对象比较，得到 bool；`nl.where` 随即读取 `condition.shape`。只修复比较后，原 `selected - 1` 也缺少 tensor subtraction 重载。安装的公开 `nl.equal` 与 `nl.subtract` 已明确支持 tensor/scalar 输入。

三个新文件组成后继：

- `lfm3_nki05_transport_kernel_v2.py`：从冻结 transport v1 继承常量和纯 helper，只重定义原 `make_kernel` 与逐字节 `run_case`。整个 `make_kernel` 的唯一两项差异是 `mapping == destination` → `nl.equal(mapping, destination, dtype=nl.uint8)`，以及 `selected - 1` → `nl.subtract(selected, 1, dtype=nl.int32)`。`run_case` 的完整 AST 与旧版相同。
- `lfm3_nki05_transport_diagnostic_v3.py`：三组诊断接入新 kernel；科学 suite 与 fail-fast 条件不变，仅添加有界错误位置记录和新来源引用。所有三组使用相同显式控制 kernel，只保留原有 writeback precision 干预差别，没有给候选额外修改 payload 算法。
- `run_lfm3_nki05_transport_serial_v4.py`：仅替换 diagnostic import 和 source author。所有函数、类、资源限制、来源认证（含原 namespace helper）、完整历史、锁、写出、清理与非零退出逻辑均与 runner v3 的 AST 相同。

比较结果为 uint8 的 0/1 control tile；selected 和 source_row 保持 int32，last-writer 规则、inactive 范围、负数和 int32 极值处理不变。这些变化只作用于整数选择控制，payload 继续使用原 BF16/FP32 typed load/store、原 byte oracle、原 observer、原 dynamic gather 与 predicated copy，没有任何 payload cast、浮点运算或归一化。

## 失败位置与原输出上限

`bounded_error_trace` 读取异常自身 traceback，保留最多四个最内层文件名、函数名、行号；路径尾部最多 192 字符、函数和错误类型最多 64 字符，另保留总帧数和省略外层帧数。`OSError.filename` 若存在则保存最多 192 字符。它不抓取 frame locals、不读取源码行、不导入 traceback/linecache，也不读取张量。无 traceback 的异常明确保存空帧列表。

此记录加到 case error、suite error 和最外层 diagnostic error，便于下一次实际失败定位到内部 API。原异常 repr 字段仍保留；它不是新的实际退出记录。原 256 KiB 单文件与完整 JSON 约束不变，所有结果仍完整编码，超限失败。

保守的纯元数据序列化测试同时保留全部 32 个候选完整 rows、132 个 extraction events、40 个 substitutions，再额外加入三组最大长度的四帧 traceback、候选 4,096 字符错误、一个带 2,048 字符错误与完整 host-difference 的 case error。实际紧凑 writer 大小 **240,643 B**，diagnostic 默认 JSON 大小 **261,074 B**，均在 **262,144 B** 原上限内。这个刻意偏大的合成表示不是实际运行或性能数据。

## 完整门槛与失败语义保持

原实现 → 候选 → 恢复原实现三组仍分开执行并恢复 probe；原失败首先必须复现。候选仍须通过全部 **32 例 / 65,536 种 BF16 编码 / 132 次 extraction / 40 次精确 BF16 writeback substitutions**，全部输入不变、完整 observer coverage、arithmetic、round-to-nearest-even cast、负对照都保留。

完整 characterization 只说明该局部对照完成。原实现仍失败，runner 仍先保存完整诊断与独立 characterization status，再抛出真实 worker 失败以保留非零退出。它不把整模型、原 legacy 3,716 例、设备行为或提速标为通过。

## 源码闭包与测试

新闭包完整保留原 567 项路径/SHA，追加上述三个真实执行入口，合计 **570**；没有替换旧 SHA、忽略旧失败或缩减来源。新的本地 import edges 全部落在旧闭包或这三个新文件内。公开 equal/subtract 的安装源码及其依赖本已在旧闭包，不新增宽泛的 SDK 或 Torch 例外。配套 `lfm3-nki05-explicit-control-source-closure-20261111-v1.json` 提供准确原清单引用、完整 570 项 map 与三项 delta；原 1,024 来源上限保持。

12 项纯测试通过（tool exit 0 / chunk `ce05af`，报告 0.664 秒）。测试内容包括：

- 用没有算符重载的 1×1 合成 control tile，复现旧 bool.shape 和仅修比较后的 subtraction 错误；新代码执行 140 个容量 4/8/16 下的 inactive、重复写、最后有效写和 permutation 控制组合。它不是 NKI 模拟器。
- 精确 AST 比较确认两项 kernel 修改、全部 payload/observer 代码不变、原科学 suite 和三组 gate 不变、完整 runner 函数及限制不变。
- 真实纯 Python 异常栈和超长合成帧验证最内层位置、长度上限、无 locals；case 失败仍重新抛出且清除 probe label。
- 执行原三组消费者的隔离纯函数，验证少一个 case、编码、event、substitution、负对照、restoration 或 promotion 都拒绝；mock 写出后仍抛出非零失败。
- 完整表示大小与旧执行源码、probe、namespace helper 和相关安装 API 的 SHA 不变检查。

首次纯测试有一个测试 harness 缺少 json namespace 的 NameError（chunk `76b18d`，exit 1），只修复新测试 harness 后以上 12 项通过。没有把这次测试失败冒充真实诊断或模型失败，也没有更改生产代码来绕过验证。

Root 可独立复跑 `python3 -B /home/ubuntu/experiments/trn1-latest-models-20261031/check_lfm3_nki05_explicit_control_successor_v1.py`。未来 source review 的 author 为 `/root/mini_operator_saved_comparison_design_20261111`，runner 为 v4、diagnostic 为 v3。Root 必须在当前真实串行任务结束后，绑定届时完整 catalog 与 idle terminal，再做独立 source review 和实际运行。本交付没有自行接线、提前启动或声称候选已通过。
