# MiniCPM5 v7 完整启动接线：源码交付，尚未进行运行验证

本交付把已冻结的 38,218 B `nocache_entry_v7` 接入完整 outer → admission wire → SDK → observer 路径。它继承同次预算计算合并，并移除 SourceLoader 对同一份已验证源码字节的第二次 SHA 运算。源码、边界、完整参数和模拟进程身份检查已通过；实际 native 导入、完整模型、推理速度、600 秒 CPU 是否足够均没有在此验证。

此版本不替换旧 v6/v1 文件或首次 v7 交付。最终运行源码清单是 `minicpm5-formal-literal-v7-source-manifest-20261112-v2.json`。较早 v1 是未冻结的 10 文件草稿，不应当用它生成后续运行绑定。

## 完整接线

| 文件 | 相对已审实现的变化 |
|---|---|
| `minicpm5_sdk_mapping_nocache_entry_v7.py` | 已冻结的同次 verified SHA 重用；保留重复 SHA 原有 guard 检查序列、首次完整哈希、读取身份和 EOF 检查；没有跨调用缓存 |
| `minicpm5_formal_literal_codec_v2.py` | 仅切换精确 v7 literal 路径、SHA、长度和 digest 导入；原 wire 格式和全部边界不变 |
| `minicpm5_formal_literal_result_v2.py` | 仅切换 codec 导入 |
| `minicpm5_formal_literal_budget_v2.py` | 仅切换 codec/result 导入，继承已审 combined budget 完整函数体 |
| `minicpm5_sdk_mapping_nocache_launch_v4.py` | 仅切换 codec/文字身份；保留完整实际 argv |
| `minicpm5_sdk_mapping_observer_logic_v3.py` | 仅将 v2 的 codec 导入切到新 codec；使完整 v7 owner 可通过原有 2,048 B 编码身份检查 |
| `minicpm5_sdk_mapping_observer_backend_v6.py` | 原 Store 主体切到新 codec 和 combined budget；原 v4 native factory 函数体不变；原 v2 NativePorts 基类不变，identity 使用新 owner codec |
| `minicpm5_sdk_mapping_observed_adapter_v7.py` | 仅切换 backend 和 observer logic 导入 |
| `minicpm5_sdk_mapping_source_adapter_v5.py` | 仅切换 observed adapter、launch、codec 导入 |
| `minicpm5_sdk_mapping_worker_source_entry_v7.py` | 仅切换 launch/codec 导入；原 SDK/native/scientific 主体不变 |
| `minicpm5_sdk_mapping_outer_source_entry_v11.py` | 仅切换上述导入和精确 SDK worker 路径/module |

独立的 `prepare_minicpm5_literal_formal_binding_v2.py` 保留原 prepare/reconstruct/merged_plan 调用，新增完整源码 pins，并依赖顺序重算四个草稿：SDK template、outer spec、outer review、outer plan。SDK template 和两处 entry source 都切到精确 v7；引用 SHA 由完整实际 JSON 重算。原科学图、完整 source contract、descriptor-plan SHA、原 caller additions、原 catalog 全保留。它不生成 accepted native policy，不选择当前 serial catalog，不运行 History。

首次冻结的 v7 loader 交付仍是 `minicpm5-bootstrap-digest-reuse-source-delivery-20261112-v1.json`，SHA `d48fa80f93a2facf26536b64014ec78ae4ab3da742a8a0f8db51e41d478c3526`。其 11 项实际 SHA/mock 测试和 35 项继承测试不因此次接线改动而重跑。v7 自身 SHA 仍为 `25eb3657d8d490b28ae605a0b65d6f5a4e12db29ec8f5b422642f01225c5b512`。

## 检查中发现并修复的问题

旧组装器根据早期 manifest 恢复 154 个 bootstrap pins，r5 最终保存的成对 plan 实际含 172 项。只加入新文件会漏掉 18 个已保存校验项，其中 8 个是当时接线的传递依赖。新 preparer 明确读取并验证 r5 outer312/sdk310 两个完整 plan，要求角色正确、两份 172 项完整 map 相等，逐项核对源码 SHA，再加入 11 个运行文件。最终两端均为 183 项，原有校验域没有缩减。

另一个问题在初稿 Store 测试之后才被发现：observer logic_v2 和 native backend_v4.identity 都关闭在旧 codec_v1 上。即使 Store 能读写新 wire，真实完整 v7 owner 仍会被拒绝。最终版本增加 logic_v3，并将 native identity 的 codec 引用一并切换。此前“直接复用原 make_ports/NativePorts 对象”的草稿结论不适用于最终版本；准确保证是 **factory 函数体 AST 相等，默认 loader 对象相同，原 NativePorts 基类及 wait/lock/cleanup 继承方法相同，identity 仅切换 owner codec 绑定**。测试覆盖实际 observer constructor、模拟 kernel child 注册、实际 identity 方法的模拟 known-task 路径，并禁止调用旧 codec。

这些失败与修正分别保留在 `minicpm5-literal-v7-first-source-assembly-failure-20261112-v1.json`、`minicpm5-literal-v7-source-fixture-corrections-20261112-v1.json`、`minicpm5-literal-v7-final-review-corrections-20261112-v1.json`。首次完整检查的输出也保留；该次唯一测试错误是新增统计代码从错误模块取 TRACE_CAP，6 项 assembler 测试当次全部通过。

## 最新检查和预算

最终源码检查：30 项 boundary + 6 项 assembly，全通过。直接工具 session **62756**，terminal **01d363**，实际 exit **0**。记录为 `minicpm5-literal-v7-complete-source-checks-20261112-v2.json`，SHA `600099eef7be22e9f2bc4ee025108d08a910d6b7542055bcf4d6080dea56e57f`。

这些是保存的 r5 静态输入加显式模拟的动态 proof/time/owner，在临时目录调用实际 Store/codec/result/adapter 方法的结果。没有重建当前 descriptor/native 认证。Boundary suite 使用测试专用 hashlib SHA oracle，同时执行 v7 的原 guard 检查序列；生产 SHA 函数未改。Assembler suite 恢复并使用生产纯 SHA；首次 v7 实际 SHA 测试记录继续适用。正常/失败的完整 wire、展开对象及 hashes、错误身份、未知字段、截断/缺失引用、重复 JSON、超预算、原 argv、一次 exit 持久化与失败 fallback 均有控制。

| 指标 | 本次 fixture 值 | 原上限或说明 |
|---|---:|---|
| literal | 38,218 B | 原 entry byte cap 65,536 B；outer 原 reader cap 65,535 B |
| SDK 完整 argv JSON | 39,339 B | 保留真实 literal；仅落盘 wire 使用明确版本引用 |
| outer 完整 argv JSON | 39,839 B | 保留完整原参数 |
| 展开的 admission | 117,325 B | 原展开 admission cap 1 MiB |
| admission wire | 40,022 B | 完整 hash 和展开 hash 均校验 |
| 保守 owner / exit wire | 921 / 1,800 B | 实际 fixture 902 / 1,773 B |
| observer receipt 预留 | 16,384 B | 原预留 |
| 完整 runtime 预留 | 59,127 B | 79,008 B |
| 六文件逻辑 metadata 预留，包含完整 trace/summary cap | 2,172,663 B | 8,388,608 B |
| 六文件 allocation 预留，包含每文件原 65,536 B margin | 2,565,879 B | 33,554,432 B |
| 正常 / unavailable stdout 保守预算 | 1,419 / 3,525 B | 16,384 B |
| 完整 stdout 实际 fixture | 1,394 B | 16,384 B |
| fallback 实际 fixture | 1,962 B | 16,384 B；保守无 error name 为 1,982 B |
| 完整 source release | 8,433,940 B | 16,777,216 B |
| source release 加原 65,536 B margin | 8,499,476 B | 16,777,216 B |

六个实际文件的 fixture 大小：admission 40,022 B、owner 902 B、exit 1,773 B、trace 27 B、observer receipt 394 B、summary 210 B；总计 **43,328 B**。其中 runtime 四文件为 **43,091 B**。这里的短 trace/summary 是模拟输入，生产 pre-spawn 预算仍预留完整 2 MiB trace 和 16 KiB summary；每文件原 2 MiB cap 没有放宽。

完整 map 从 r5 的 48,223 项增至 48,235 项，新增 11 个运行文件加 1 个 preparer，原 source contract 不变。原 catalog 的 155 行和 descriptor-plan SHA `1838866c752ef628bda60cb63d2c9938a2009d0fb09fa77e5539802e36206294` 保留。对新运行入口递归做源码 import 检查覆盖 33 个源码模块；这是前期源码导入闭包，不替代保留的全部 172 项旧 bootstrap，也不替代完整 48,235 项历史认证。

上述大小与具体输入和路径相关。后续真实绑定仍须对那次完整实际 metadata/argv 重新执行原预算，不能把此 fixture 数值当作运行准入。

## 原约束和后续使用

CPU soft/hard 仍为 **600/605 秒**，wall **900 秒**，address space **2 GiB**；9 个 native/source 检查、1,284 个 scientific capture 检查、完整历史重建均保留且仍待真实执行。冷缓存、长解码、请求槽位复用、原权重与精度不在本源码交付中改变。本次未加入 CPU/wall stderr marker，以避免把额外运行输出和 guard 边界改动混入最小接线；root 可另做有界 marker 后继。

复现检查时指定一个不存在的输出路径，检查器会拒绝覆盖既有记录：

```sh
/home/ubuntu/venvs/vllm_neuron/bin/python -I -B   /home/ubuntu/experiments/trn1-latest-models-20261031/research/check_minicpm5_literal_v7_complete_source_20261112_v1.py   /home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-literal-v7-root-source-checks-20261112-v1.json
```

后续 root 应先审阅完整 diff/provenance 和上述源码/mock 证据，再按当时真实空闲串行状态生成新的 attempt request 与独立输出路径。向新 preparer 传入 v2 manifest 的精确 ref、preparer 自身精确 ref，以及原合法 read/text/guard/original 上下文；由原 prepare/reconstruct 路径重新生成完整 source/PV/native-policy 图。必须使用新产生的四个相互引用草稿，并独立完成实际 source/native/serial binding 审核。不能覆盖 r5 文件，不能复用旧 accepted 结果作为新运行认证。

本交付没有调用入口 main、History、native loader、ptrace、compiler、model、锁操作或真实 Popen，没有读取权重/科学 capture payload，没有改变当前 binding/handoff。它是可供后续准入检查的完整源码后继，**没有新增端到端提速倍数，也没有证明 r5 的 SIGXCPU 已解决**。
