# Dory Stage A 官方模型后继 worker / preparer 源码交付

本次完成三个后继源码模块和 33 项轻量检查，连接原始官方权重加载、六次独立空缓存采集、v2 归档事务和持久化 JSON manifest。没有执行 native 模型、读取 checkpoint/raw payload、产生 Stage A 数据、创建 production binding/spec/admission 或任何未来退出记录。已执行文件全部保留，原有源码没有改动。

## 新的具体连接点

`prepare_dory_stage_a_source_inputs_v1.py:124` 的 `prepare_source_inputs(guard=...)` 只读 17 份原已 pin JSON，在内存中生成未绑定的 source inputs。它使用原 bind_cases，保留六次的完整 tokens、选中行、原事件/重放引用和原控制 SHA。它从历史 runtime JSON 取得已 pin 的原 loader proof，但不会继承旧 checkpoint inode、output、admission 或任何运行准入。返回内容明确 runtime_bound=False/runtime_admitted=False；本轮仅测试内存返回值，没有保存成生产配置。

`dory_stage_a_worker_preparation_v1.py:99` 的 `_native_capture` 是官方-only 构造链：调用未修改的 `dory_checkpoint_loader_v1.load_checkpoint`，严格核对它真实返回的字段、507 个权重描述符和全 checkpoint hash；使用原官方 DoryForCausalLM，以 BF16 构造，确认 399 个物理参数及 3,975,229,440 个参数元素；按原 q/k/v、gate/up merged mapping 校验每个权重 slice 的实际字节 SHA。没有构造候选模型，没有安装 F.linear 路由。

随后保持原 `_prepare_for_inference(official)`，核对与旧实际 model.json 一致的 182 个 oneDNN backend 名字/代码引用、56 个独立 per-loop ReferenceAttention cache，以及两张 [262144,256] BF16 RoPE 表，base 分别为 10000 和 10000000。oneDNN 包装器只计数，再调用原算子；原 Attention 替换为原上游测试 ReferenceAttention，与此前官方 CPU baseline 一致，并不代表 Neuron attention 或生产 paged attention。

`capture_cases`（同文件第 49 行）直接构造原 NativeTensorAdapter / StageAObserver / ArchiveFileStore / CaptureArtifactBatch v2。每次使用完整原 input_ids 和从 0 开始的完整 int64 arange，保留 2060 / 2060 / 23 / 127 / 511 / 31 行。独占 56 个缓存，在每次前后清空 keys/values 并置 start_pos=0；要求恰好 6×281 个 oneDNN 调用。异常也会清空缓存、移除 hook、关闭归档资源。6179 仍记录原 slot 3 / epoch 2 身份，但这里是独立空历史诊断，不能冒充真实生产槽位复用通过。

`run_worker`（同文件第 222 行）只有在 observer.close、cache 释放、inference/config 上下文退出、default dtype 恢复、monkeypatch.undo 及原 dispatch 恢复核查后，才持久化四份证据 JSON，最后写 producer manifest。`batch.finish()` 的内存结果本身不等于 worker 完成，更不等于真实进程退出。

## 持久化元数据与真实退出边界

`dory_stage_a_durable_metadata_v1.py:68` 的 `write_json_exclusive` 在创建文件前完成树深度/节点数/字符串长度和字节上限检查，使用 allow_nan=False。保留字典插入顺序，避免改变 archive tensors 的物理顺序。文件以独占、无链接跟随的方式创建；全部内容写完后 flush、fsync、close、directory fsync，并再次校验真实文件身份。失败不覆盖或删除 partial，不返回完成凭据。真实文件系统行为仍待准入后实测；本轮仅用 BytesIO/fake filesystem。

持久化顺序为 stage-a-loader-proof.json、stage-a-loaded-parameters.json、stage-a-model.json、stage-a-artifacts.json，最后 stage-a-producer.json。前三份和最终 producer JSON 的各自限额为 2 MiB，artifacts 为 16 MiB；不会因超限自动放宽。所有写入均受同一实际 guard。producer manifest 里明确 producer_process_exit_observed=False、independent_saved_byte_review_complete=False、requires_actual_exit_and_separate_review=True。

未来实际调用形状：

```python
# 这是接口示意，不是已绑定生产脚本。
# 准备阶段：source-only，guard为当时允许的metadata观察守卫。
prepared = prepare_source_inputs(guard=source_guard)

# 执行阶段：必须另有实际串行准入、完整资源限额与已认证原native API。
# fresh_checkpoint_identity由当时实际准入提供，不能沿用历史inode或本轮fixture。
receipt = run_worker(api, prepared, fresh_checkpoint_identity,
                     fresh_archive_root, guard=producer_guard)
# receipt只是durable producer JSON身份，绝不证明此进程已退出。
```

调用方负责认证 api 的每个实际导入模块/函数与环境、完整 source closure 和输入 metadata SHA；prepared 内的 digest 只是内容完整性检查，不能认证任意调用方修改后自行重算的对象。原 tokenizer/chat template/特殊符号/生成策略与原始 tokens 的来源证明仍为外部契约，本 worker 不重跑 tokenizer，也不声明注入的 native API 已经验证。

真实 parent 在 producer 返回后必须观察实际进程退出、执行物理释放审计，认证 durable JSON 的 SHA/身份、实际 root 和当前 native source/weight/environment 证据，再以独立 review admission 启动 reviewer。reviewer 调用既有 `review_saved_batch(root, authenticated_cases, authenticated_artifact_manifest, guard=review_guard)`。这里没有实现或伪造 parent 的 wait/exit/release，也不会在 producer 中偷跑 reviewer 来宣称独立复核。

## 已完成的检查及其范围

本代理 33 项检查通过：session 99984，启动 chunk 68cad2，实际结束 chunk db8fa3，exit 0，1.875 秒。根代理另外实测同一套 33 项通过：session 79987，启动 d11596，结束 08cd36，exit 0，1.787 秒。两次不是 66 项不同覆盖。

所有测试经当前 observe 包装器，audit hook 禁止 Torch/NumPy 导入、磁盘 payload 读写和实际文件写入；拒绝记录为空，native 加载为空。检查分为：

- 8 项真实 durable writer 的 fake filesystem/BytesIO 检查，覆盖正确 JSON/顺序、NaN/超限/循环树拒绝、独占创建、write/flush/fsync/close/目录同步失败及关闭后变更。
- 8 项 source inputs 检查，实际读取原 pin JSON，覆盖 2 MiB cap、507 BF16 descriptor 字节等于 2×prod(shape)、重算 digest 后仍不得放宽 gate/shape/运行字段、错误 pin 和重复 JSON key。
- 12 项官方构造接口检查，真实走 `_native_capture` 和 `run_worker` 源码。507 个 VirtualWeight 只有形状和旧 metadata digest，没有 tensor payload；以此检查实际 merged mapping 能得到 399 / 3,975,229,440，182 backend/56 cache/RoPE 条件和恢复时序。checkpoint loader、权重 digest 和实际 capture 被测试替身替代，所以不代表真实模型构造、全量权重加载或 7.95GB 验证已经执行。另一个 tiny fixture 对真实 adapter 的非连续/offset tensor 字节流进行独立 SHA 核对。
- 4 项六次采集循环检查，以真实 NativeTensorAdapter 的 FakeTorch tokens/positions 和 batch 替身核对全部原 tokens、全部位置、六次顺序、56 cache 隔离、失败清理和 hook 移除。之前 v2 事务的实际 writer/emitter 小字节测试与 52 项已冻结测试继续作为依赖证据，本轮没有重复跑来扩大数字。
- 1 项静态边界检查：无 native import、无候选构造、无 F.linear 路由安装、无 reviewer 调用、CLI 入口拒绝启动。

## 实际 metadata cap 与资源边界

preparer 每份原 JSON 上限固定为 2,097,152 字节，顺序读取块最多 65,536 字节。实际 17 份总计 5,121,287 字节，最大为 selection_plan 的 1,871,774 字节；design 976,742 字节、旧 runtime 471,948 字节、loader proof 181,887 字节、model metadata 18,294 字节，12 份 event/replay JSON 为 131,532–135,136 字节。没有扩大 cap，也没有读取其中指向的原 raw 或 checkpoint。

未来真实 worker 会先由原 loader 完整读取 7,950,515,472 字节 checkpoint，再通过 NativeTensorAdapter 对全部 loaded parameter slices 额外遍历 7,950,458,880 字节。后者使用 ≤64 KiB chunk clone / uint8.tolist，可能有显著 CPU、Python对象与分配开销；这是精确字节校验的实现成本。本轮 fake tests 不提供这一链的 CPU 时间、RSS、存储适配或吞吐预算结论。准备的元数据每份上限不包含进程中 prepared/deepcopy/JSON 树等实际内存开销，未来 admission 必须单独计入。

Stage A payload 仍为 1,467,113,472 字节，三阶段总量仍为 2,046,031,872 字节，原计划预留 3 GiB，root 28 GiB / NVMe 30 GiB 额外余量不变。不借用 Nemotron 49 GiB，不创建预留或新的实际运行目录。

## 精确剩余工作

真实 Torch 2.14.0+cpu adapter ABI（尤其 singleton stride 的 clone 路径）、原官方模型构造、507 权重读取和额外完整参数遍历、文件系统 durable 行为均尚未运行。实际 source/API/tokenizer/environment 认证、资源估算与限额、当前 checkpoint/output 绑定、串行 admission、父进程真实退出与物理释放绑定、独立 reviewer 的执行仍须由未来 coordinator 完成。本轮明确没有写这些生产对象。

原 87 个标记点、7 个失败 3607/4105/4113/6160/6164/6176/6181、89 个调用和 153 个选中行保持原样；cosine≥0.995、max_abs≤2、KL≤0.02、冷缓存、长解码、请求槽位复用与 HTTP 门槛没有放宽。没有新增性能或全模型合格结论。当前 Nemotron 仍有活动会话约束，本交付不能绕过它自动启动。
