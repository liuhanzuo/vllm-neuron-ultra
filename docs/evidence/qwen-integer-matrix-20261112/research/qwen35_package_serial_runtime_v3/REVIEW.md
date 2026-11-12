# Package serial runtime v3 — bounded compiler storage observation

本后继修复编译期间合法 cache 发布、解锁及 bucket 工作目录重建与存储扫描之间的竞态。v1、v2 共 40 个文件保持冻结。生产差异只有 `io.py` 的存储观察和 `run.py` 的 v3 导入命名空间；History、科学顺序、readiness、supervisor/direct-child ACK、wait/drain、锁、资源上限和持续 free-space reservation 均保留。

本交付只包含 SOURCE/STDLIB MOCK。没有运行原生导入、服务、编译、推理或锁，没有读取真实模型、tensor、packet 或 NEFF 内容，也没有选取当前 History/binding 或产生运行资格。

## 实际 SDK 行为与最小修复

固定 SDK `neuronxcc-2.23.6484.0+3b612583` 的本地 cache 使用 `MODULE_<20hex>+<8hex>` 子目录。`upload_file()` 将 `model.hlo_module.pb`、`model.neff` 或 `wrapped_neff.hlo` 先写到带当前 hostname 和正整数 PID 后缀的普通文件，然后 `os.replace()`；解锁则删除 `model.hlo_module.pb.lock`。v3 只对这些准确目录/名称提供新的消失重扫分支。最终 cache 文件、未知版本、未知 hostname、错误 PID、未知名字或嵌套位置消失仍拒绝。

实际 `get_work_dir(create_subdir=False)` 会删除并重建 bucket 工作目录。v3 仅承认两个固定 tag `context_encoding_model` / `token_generation_model` 的 `_tp0_bk0` 至 `_tp0_bk5` 以及 `layout_opt` 内部的工作树变动。闭源 compiler 的内部文件名不能静态穷举，因此此范围采用有限完整重扫；不是忽略所有 FileNotFoundError。其他 workdir 分支、六树以外路径以及 artifact/view 的异常消失没有新例外。发布 `build-immutable-identities.json` 后，新 compiler 消失例外关闭。

发现上述消失时，当前不完整扫描整体丢弃，不返回部分计费。继续原最多 8 次、间隔 0.01 秒的有限重试，并要求随后两次完整六树扫描收敛于相同路径集合和 inode/类型身份。相同普通文件的 size/allocated blocks 可以单调增长；目录大小和时间戳也无需停止变化。因此持续 service 日志或正常 compiler 文件写入不会仅因字节数不同而失败。每次完整扫描均重新核对每个文件、entry、class 和总量上限；返回最后一次完整扫描的实际用量，随后执行原 continuous remaining-reserve、root/NVMe statvfs 和 MemAvailable 检查。

两次完整扫描之间再次出现已识别的拓扑变化时，重新开始整对扫描，共用同一个 8 次界限。未知消失、未知 inode 替换以及正常 log/artifact 的非预期缩小仍立即失败。原 `terminal=True` 在完整 drain 之后要求两次完整计费和全 inventory 精确一致，未放宽。原 symlink/hardlink 拒绝、JSON 原子发布处理、最终完整计费和所有 caps 未跳过。

## SOURCE/MOCK 证据

- 新编译存储 14 项：工具 chunk `3d5226`，实际 exit 0。包括真实 stdlib 临时树中的 cache rename/unlink、bucket/layout 重建、错误路径/版本/hostname、封存后拒绝、symlink/hardlink、重扫后所有类别上限、持续日志/普通文件增长、第二次完整扫描超限、有限重试和 terminal 精确稳定规则。
- 原 Storage、LifecyclePreservation、FullSequenceMock 共 25 项：工具 chunk `ded488`，实际 exit 0。覆盖原资源 reservation、完整科学顺序、科学门禁、失败停机、wait/drain、History 和锁 AST 保留。
- 补充未知 inode 替换的单项控制：工具 chunk `f4d44d`，实际 exit 0。此前 14 项之后只新增此测试方法；生产实现未改变。合计 40 项针对性控制通过，并未声明是一次 40 项全量运行。
- `ast_preservation.json`：v1/v2 全部 40 文件 SHA 未变；11 个生产/契约文件与 v2 字节相同。已有 `io.py` AST 节点只改变 `storage_scan`、`storage`，新加四个纯 stdlib 辅助节点；`resources` 和所有原非存储观察函数 AST 不变。
- `source_diff.patch`、`predecessor_pins.json` 和 `input_pins.json` 提供可复核差异及完整文本来源。测试用 NativeBlocker 阻止 torch、Neuron、vLLM、Transformers、NumPy 导入。

## 预算审阅与限制

完整 SDK 调用链、准确源码行和 SHA 见 [预算审阅 JSON](/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-build-storage-source-review-20261112-v1.json) 和 [说明文档](/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-build-storage-source-review-20261112-v1.md)。原有限预算要求 root 29 GiB、NVMe 39.28125 GiB；准备回执时 root 仅余 144,257,024 字节超过要求，不能代替实时检查。

主要 bucket 编译实际传入新的 workdir/cache，wrapper 临时 HLO/NEFF 落入 `shared-build/tmp` 并计入 auxiliary。可选 layout compiler 路径仍继承实验 ROOT 作为 cwd；明确输出/logfile/TMPDIR 已路由，但不猜测闭源 compiler 其他写入。HF_HOME 同样在六树外。当前机制是有限 trial 与持续 floors/reservation 监控，不是文件系统硬配额，也不承诺已测得编译峰值或所有原生写入均位于六树。不修改 compiler cwd，不删除共享缓存，不代签 root 独立审阅或 runtime admission。
