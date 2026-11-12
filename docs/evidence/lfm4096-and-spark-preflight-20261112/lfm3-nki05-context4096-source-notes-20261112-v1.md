LFM3 实际 KV4096 的追加设备验证源码

2026-11-12。本次把 LFM3 的 `(capacity, 2, 4096, 128)` 缓存形状接入已审查的完整串行设备方案。原 91 个 child tasks、113 个 graphs、3491 个 checks 均原样保留在前面；新增 14 个 child tasks/graphs、813 个 checks，总计 105 tasks、127 graphs、4304 checks。没有运行原生 SDK、编译器或设备。4096 原语即使未来通过，也只是完整模型 4096 冷缓存、长解码、请求槽位复用和服务验证的前置条件。

| 阶段 | graphs | checks | 是否原样保留 |
|---|---:|---:|---|
| raw_device | 4 | 36 | 是 |
| regression_device | 9 | 43 | 是 |
| regression_alias | 12 | 60 | 是 |
| extended_device | 42 | 2520 | 是 |
| extended_alias | 24 | 192 | 是 |
| fp32_alias | 6 | 222 | 是 |
| wrapper_controls | 16 | 418 | 是 |
| context4096_device | 8 | 543 | 新增 |
| context4096_alias | 6 | 270 | 新增 |

新增容量范围直接沿用原模型合同：B4、B8、B16。standalone rows 为 B4 的 1/4、B8 的 1/4/8、B16 的 1/4/16，共 8 图；persistent alias 为每个容量的一对 reference/candidate，共 6 图。原 RNJ 的 `(4,4,4096,128)` 图继续保留，不能替代这里 LFM 的 2 KV heads/rank。

六个不可变新生产文件位于实验根目录：`lfm3_nki05_context4096_matrix_v1.py`、`lfm3_nki05_context4096_device_leaf_v1.py`、`lfm3_nki05_device_protocol_v2.py`、`lfm3_nki05_device_runtime_v2.py`、`lfm3_nki05_device_pipeline_v2.py`、`run_lfm3_nki05_device_serial_v2.py`。它们从首次测试后保持不变；所有旧文件均未改动。

standalone 复用已审查 extended leaf 的完整科学体：实际 cache/update 形状、特殊 BF16 位模式、独立 sequential Torch oracle、输出全字节比较、cache/update/slot 输入不变和完整 output hash。矩阵保留旧 B8/B16 mapping 的完整有序前缀，将同一 mapping 算法扩展到 B4，再补现有 signed-int64 controls，包括 `-2**63`、`2**63-1`、`-2**32`、`2**32`、`2**32+capacity-1`。slot 仍以 int64 传给已审查 typed wrapper，由其先判有效性再转 int32。

persistent alias 复用完整 K/V attention state 科学体、`input_output_aliases`、move-to-device、persistent state binding 身份检查、每一步 cache output 与设备持有 state 的全字节比较、input hash 和 reference/candidate 成对 fixture 核对。每个图有 45 步：先原来的 8 步 population/permutation/inactive/collision/reuse，随后 signed-int64 controls，末尾再全槽填入、全槽 inactive、反序复用。全部 reference/candidate 配对沿用相同 seed、初始状态和调用输入。

新增 alias 输入的 position 固定覆盖 0、127/128、255/256、511/512、1023/1024、2047/2048、4094/4095，每个 slot 在序列中都经过这些位置。这个 primitive 用于检查指定位置和状态复用，不能当作已经执行了模型连续 4096-token 推理。实际 full model 的长解码仍需独立完成。

为了让巨大的无效 slot 能进入 alias writer 检查，alias attention 的读取索引只对有效 `[0, capacity)` slot 使用原值，其他读取暂用 0；传给 writer 的 slot 保持原值。这样不会在 gather 阶段先因上界越界失败，reference/candidate 使用相同读取语义，writer 必须保持所有无效目标不写入。原有负 slot 的读取语义保持一致。它不是允许服务接受越界 slot 的行为修改。此 Torch/编译器路径尚未进行 native 验证。

新 protocol 对前 91 tasks 直接调用旧 validator；新阶段严格要求准确的 spec、2-head4096 shape、dtype、rows、artifact identity/source pins、完整有序 mappings、所有 byte/input/persistent-state/binding 条件和 4096 position 清单。pipeline 的 phase 汇总继续独立读取 child exit、ACK、PID/start/argv/cgroup、SDK receipt、artifact receipt；新增 alias 阶段也要求 reference/candidate 初始状态和全部输入 hash 相同。缺 task、缺最后一个 check、错误 shape、错误 position、无效 digest、非零退出或配对 fixture 不同均拒绝。

串行运行仍通过同一个具体 DeviceContext 与 owner 流程。新 owner 与原 owner 的源码差异仅为导入 pipeline v2；锁 descriptor、PID/start 绑定、env-to-Python transition、service 生命周期和排空后释放的代码未变。runtime 仅切换 protocol/pipeline/owner 的固定版本名。pipeline 追加两个 phase 的 dispatch 和成对校验，额外在 prerequisite 中读取固定 SHA 的旧 source closure，并强制其全部 616 个 pins 原样存在。原完整 CPU r2 的 595 个 pins、3716 cases × 3 passes 加 control、874 次真实退出检查全部继续保留。

新的 source closure 包含 622 个 pins，即完整旧 616 加 6 个新生产源码。原 595 与后续 616 的每个现有 `.py` 文件均重新核对 SHA。manifest 固定完整旧 protocol、完整原模型 gates、全部新 specs 和相同 budget，标记 `context4096_qualified=False`、`full_model_qualified=False`、`new_speedup_claim=False`、`automatic_successor_launch=False`。旧 protocol manifest 不能通过新 owner 的 spec 校验。

沿用原资源上限：MemoryMax 16 GiB、MemoryHigh 14 GiB、LimitAS 24 GiB、无 swap、CPU 6–7/nice19；单 artifact 256 MiB、单 compiler 文件 2 GiB、总输出 24 GiB；单图 child 1800 秒、每 phase 14400 秒、整个 worker 43200 秒。追加图不会自动扩大预算，超限应保留失败。新编译器真实峰值、artifact 大小和总耗时仍未知，不能根据源码容量计算声称能完成。

每个 child 仍完整核验 SDK，RECORD 文字估计的每次载荷 4,953,771,217 bytes 未改变；105 次约 520,145,977,785 bytes，比 91 次增加 69,352,797,038 bytes。这只是既有 RECORD 清单推算，没有重新读取二进制，也不是实测 I/O 或时间。未来若上限不足，应提交有证据的新预算方案，不能跳过原检查。

`check_lfm3_nki05_context4096_source_v1.py` 的 16 项 stdlib/mock 检查全部通过，实际工具 chunk `90b04f`，exit 0，0.316 秒。内容覆盖原 91 tasks 的 validator 兼容性、新 shape/rows/capacity、signed-int64 与 boundary coverage、新任务完整性、pair drift/缺 child/非零退出拒绝、旧 manifest 拒绝、616 pins 缺失拒绝和 owner/context 接线。所有 native 模块导入被拦截。该结果只证明源码/协议检查，不是设备数值或 MUST_ALIAS 成功。

审查入口：`lfm3-nki05-context4096-pretest-diff-20261112-v1.patch` 包含所有新代码及其与已审查前身的完整差异；`lfm3-nki05-context4096-pretest-freeze-20261112-v1.json` 是首次测试前固定的 SHA；`lfm3-nki05-context4096-additive-source-closure-20261112-v1.json` 是 622-pin 完整闭包；`lfm3-nki05-context4096-protocol-source-manifest-20261112-v1.json` 是可审查的完整 protocol/budget。

本次没有创建运行 spec、review、admission、lock、service 或 handoff，也没有改变当前 live r2 绑定。真实设备执行仍需 root 复核本方案，并等当前 r2 的真实结束、最终 exit 0/收据及完整排空、更新的串行记录和新的空闲确认。全模型 B4/B8/B16 的 numeric/cold/1529-step long/70-state lifecycle/slot reuse/quality96/匹配 HTTP 负载等原门槛全部仍需满足，4096 原语检查不能替代它们。
