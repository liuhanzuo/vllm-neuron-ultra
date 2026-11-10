# Dory Stage A 归档事务源码交付（v2，未实际采集）

本次把现有 StageAObserver、NativeTensorAdapter、三路归档 writer 和独立 saved-file reader 接成一条调用链。没有启动 Dory，没有读取旧 tensor/raw/weights，没有创建运行准入、占用预留空间、替代串行队列或写入续跑交接；没有修改已执行源码。v1 的 46 项检查以及本次 v2 的 52 项检查均通过，v1 已冻结保留，v2 是当前候选。根代理独立复核尚未由本交付宣称通过。

## 可使用的实际接口

`dory_stage_a_artifact_transaction_v2.py` 提供以下接口：

- `ArchiveFileStore(root, guard=...)`：接受已存在的独占、规范路径目录。三份文件用 O_CREAT|O_EXCL|O_NOFOLLOW|O_NONBLOCK、0600 创建，不 mkdir、不覆盖、不删除。只在所有文件 flush、fsync、close，再 directory fsync 和最终身份检查成功后返回身份。一次失败保留全部 partial；standalone 使用者失败后必须调用 abort，batch 已负责这一点。
- `CaptureArtifactBatch(cases, observer, adapter, store, guard=...)`：cases 必须来自原 bind_cases 且已由调用方认证；observer.ops 必须就是这个 adapter。依次调用 `collect(tokens, positions)` 六次，消费真正的 `observer.collect(case,tokens,positions, TensorEmitter(..., adapter.iter_bytes))`，不是旁路或另造数据接口。
- `finish()`：仅在六次及 18 份归档全部完成、原 observer.finish 成功和 store 关闭后返回内存 manifest。它不持久化 JSON、不证明进程退出、不批准运行。
- `review_saved_batch(root,cases,manifest,guard=...)`：先核齐六个 owner/原证据引用/完整原始 source layout/全部 dense descriptors/18 个不同文件身份/固定预算，才按原独立 stage_a_requirements 逐文件调用冻结的 verify_file。严格核对其返回的完整 schema、字节数、tensor 数、旧控制数和目标选中行数。任一文件失败，不返回全量完成结果。

唯一数据流是：

```text
原完整 official forward → StageAObserver passive hooks
→ TensorEmitter.__call__(scope,label,tensor,rows)
→ NativeTensorAdapter.iter_bytes(tensor,rows)
→ ArchiveWriter.write(label,chunks) → 三个独占文件
→ flush/fsync/close → directory fsync → producer identities
→ 真正 producer 退出及来源认证（调用方）
→ review_saved_batch → verify_file → 独立 verify_archive
```

## 接到实际 runner 的位置

冻结的 `dory_residual_worker_preparation_v2.py:205` 已完成 `_prepare_for_inference(official)`；其 206–215 行有 182 个 oneDNN linear、56 个 ReferenceAttention、RoPE 的检查，119–124 行有实际 oneDNN 计数器，208–209 行可得到 caches 与 extract_layer_index。当前 266–267 行调的是旧 `dory_residual_replay_runtime_v2.run`。新的 Stage A worker 必须创建后继文件，在等价的官方模型准备及身份检查完成点接入下列接口，而非修改或重跑旧 worker。

Stage A 只需要官方 oneDNN 模型和原始上游 ReferenceAttention。下一份 worker 应保留原始官方权重加载及检查（507 个权重 tensor、399 个物理参数、3,975,229,440 参数），不构造/执行旧因果对照 candidate，也不安装候选 F.linear 路由。原始 RoPE、BF16 数学、实际 oneDNN counter 和完整词表保持不变。这里没有实现或批准该 worker。

以下是给未来已准入 runner 的具体调用形状，不是可独立启动的脚本。`env` 是已认证的原官方 loader 结果；`cases` 由新 preparer 对 design/selection_plan/六份 original 与 replay JSON 分别校验 SHA 后调用原 `bind_cases` 得到；`guard` 必须是当时实际有效的串行锁和资源守卫。

```python
from dory_stage_a_capture_core_v1 import StageAObserver
from dory_stage_a_tensor_adapter_v1 import NativeTensorAdapter
from dory_stage_a_artifact_transaction_v2 import ArchiveFileStore, CaptureArtifactBatch

torch = env['torch']                 # 已加载的原 CPU torch；这里不另行 import
cache_by_id = {env['extract_layer_index'](c.prefix): c for c in env['caches']}
adapter = NativeTensorAdapter(torch, guard=guard)
observer = StageAObserver(env['official'], cache_by_id, adapter,
                          lambda: env['counts'][0], guard)
store = ArchiveFileStore(archive_root, guard=guard)
try:
    batch = CaptureArtifactBatch(cases, observer, adapter, store, guard=guard)
    observer.install()
    with torch.inference_mode():
        for case in cases:
            guard()
            # Stage A 是六个原 start=0 宽输入的独立诊断重放。
            # runner 必须独占这些原 cache；不得有其他活请求或 bank 持有它们。
            for cache in cache_by_id.values():
                cache.keys = cache.values = None
                cache.start_pos = 0
            tokens = torch.tensor(case['input_ids'], dtype=torch.int64, device='cpu')
            positions = torch.arange(case['identity']['end'], dtype=torch.int64, device='cpu')
            case_receipt = batch.collect(tokens, positions)
    producer_manifest = batch.finish()
finally:
    try:
        observer.close()             # 移除所有148个 passive hook
    finally:
        store.abort()                # 成功时无open handle；失败时仅关闭、不删partial

# 后继runner在这里用独占+fsync的元数据写入器持久化producer_manifest，
# 连同实际source/weights/environment证据交给父进程，并真正退出。
# 旧preparer.save只是open('x')写JSON，不能作为durable元数据凭据直接复用。
```

事件 6179 的输入仍是原 fixture 1 / slot 3 / epoch 2 / 31 行，必须保存原身份；本独立空缓存重放不会证明生产请求槽位复用。事件 6181 是 slot 1 的继续解码，不能误写成 slot 1 被复用。上例清空缓存仅在调用方独占且无其他活请求时适用，不替代原完整请求生命周期测试。

独立 reviewer 的连接点必须在父进程拿到真实 producer exit、关闭/释放证据后，重新取得自己的串行 review admission；不能在 producer 内直接调用然后冒充独立复核。

```python
from dory_stage_a_artifact_transaction_v2 import review_saved_batch

# reviewer已独立认证cases、持久化manifest及来源/真实退出证据。
# review_guard是新的有效独立读取租约；不沿用producer的准入。
review = review_saved_batch(archive_root, cases, authenticated_manifest,
                           guard=review_guard)
# 再持久化review和此reviewer的真实退出记录；这不会令FP32质量或整模型合格。
```

## 检查结果与边界

v2 的 52 项检查在当前观察包装器内完成，进程 exit 0，耗时 13.567 秒。工具 session 15178、启动 chunk 063c5b、结束 chunk 44f29a。Python audit hook 禁止 Torch/NumPy 导入，禁止除源码/JSON以外的实际文件读取，禁止实际文件写入；拒绝记录为空，native 模块加载为空。

小型字节测试把真正的 NativeTensorAdapter 接到 FakeTorch 的非连续、有 storage offset 的 tensor，再走真正 TensorEmitter/ArchiveWriter，共 40 字节内存归档，以原 verify_archive 核对结果。文件系统全为假实现，验证独占打开标志、flush/fsync/close/目录同步顺序、失败不完成、不删除 partial、fdopen 所有权、文件身份/父目录/祖先链接变化，以及 close 后 size/mtime 改变。它没有实测真实 ext4/NVMe durability 或 native Torch ABI。

完整六 case 测试读取已 pin 的原 JSON，检查 18 文件、源完整形状、真实控制 SHA、预算和 reviewer 参数；只构造 metadata，不生成或验证 1.47GB 物理数据。verify_file 的完整返回 schema 通过与冻结源码对齐的测试 stub 验证；实际 saved-file bridge 的既有独立复核不是本轮全量字节复核。v1 的 46 项结果保留，不将两轮相同测试重复相加为独立覆盖。

预算保持：target 221,736,960 字节；selected 80,281,600 字节；original 1,165,094,912 字节；Stage A 1,467,113,472 字节；三阶段总计 2,046,031,872 字节，原计划 3 GiB。root 28 GiB / NVMe 30 GiB 的额外余量门槛不变，不借用 Nemotron 的 49 GiB。类里的固定字节限制不是磁盘预留，实际 guard/准入仍须检查资源。

下一步具体缺口是后继官方-only worker/preparer：固定 source closure 与环境权重证据，显式六次 cache 隔离与原输入张量准备，durable metadata writer，实际 producer/reviewer 退出绑定；以及真实 adapter ABI/文件系统验证。尤其 singleton stride 的 native dispatch 仍未实测。必须在当前 Nemotron 真实终止、审计、物理释放和新串行准入后才可执行，不能靠本源码交付自动启动。

原 87 标记点、7 个失败（3607、4105、4113、6160、6164、6176、6181）、89 个选中调用、153 个选中行，及 cosine≥0.995 / max_abs≤2 / KL≤0.02 全部保留。冷缓存、长解码、请求槽位复用、HTTP 和完整质量门槛均未取消。本交付不宣称 Stage A 已实际完成、根因已定位、原 FP32 验收通过、全模型合格或有新提速。
