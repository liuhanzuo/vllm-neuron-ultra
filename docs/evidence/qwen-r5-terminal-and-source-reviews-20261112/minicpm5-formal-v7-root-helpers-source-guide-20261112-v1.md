# MiniCPM5 v7 的 root 准备与接收辅助入口

本交付给已审 v7 完整源码增加实际可调用的准备与接收入口。所有未来 catalog、idle handoff、最后实际终止记录、准备进程实际 exit 和独立候选审查引用都由 root 在真实完成后提供。这里没有选择未来目录，没有生成或猜测任何未来 session、terminal chunk 或 digest，也没有运行准备、接收、History、锁或 native 操作。

三个新文件：

- `prepare_minicpm5_catalog_formal_binding_v4.py`：v3 的完整 `prepare` 函数体 AST 不变；literal preparer 切到 v2；保留旧 14 项 catalog helper 源码，新增自身 v4；外层 entry 选择 v11。原 `catalog_prepare_v3.prepare`、完整原始重建、原 strict merged_plan 和全部科学要求保持原路径。
- `prepare_minicpm5_formal_v7_metadata_root_v1.py`：从保存的 r5 请求继承完整原始输入与 additions，把显式传入的新 catalog/handoff/predecessor 和新目的路径接入上面的 binder。固定引用 root 已接受的 v7 source review，进一步核对精确 delivery、manifest v2 与 literal preparer v2。准备阶段只产生原来的 16 项草稿及 preparation receipt；reviewer/accepted/passed 不会在准备时改成接受状态。
- `accept_minicpm5_formal_v7_metadata_root_v1.py`：从保存的 r5 finalizer 派生；要求真实成功 metadata exit、完整物理清退、原锁重新核验和精确 preparation ref，以及独立候选内容与 finalizer 源码审查。核对全部 16 个规范 JSON、完整 r5 source map/contract/current catalog sources，核对原 r1 native policy 除 source-map SHA 之外逐字段完全相等；保留原 `compatible.validate`、`diagnostic.validate_spec`、`outer11.validate_inputs` 和独立解释器 authority 检查。最终生成 11 个相互引用的接受材料和完整 argv；本函数不会启动 argv。

没有 monkeypatch 原模块全局变量，没有替换原历史模块，也没有运行时 AST 装载替代原实现。既有 v7/literal/capture/native 源码及 r5 文件不变。

## root 后续调用方式

新 metadata helper 提供两个 CLI operation，实际调用必须放在串行工作真实结束之后：

```sh
/home/ubuntu/venvs/vllm_neuron/bin/python -I -B /home/ubuntu/experiments/trn1-latest-models-20261031/research/prepare_minicpm5_formal_v7_metadata_root_v1.py --operation prepare --inputs <实际输入JSON路径> --inputs-sha256 <完整实际SHA256>
```

随后 root 观察该 metadata tool 的真实最终退出，完成物理清退记录和独立候选审查，再调用同一 CLI 的 `--operation accept`，传入另一份精确输入 JSON。上述尖括号只是说明参数位置，不是已创建的输入文件。

`minicpm5-formal-v7-root-helpers-input-contract-20261112-v1.json` 列出了准确键集合、全部已冻结 helper/catalog 源码 refs、未来实际 exit 和独立 review 的字段要求。它只含静态源码输入，不含任何未来运行身份。

准备输入必须逐项提供：`catalog`、`handoff`、`predecessor`、`handoff_catalog_field`、`artifact_prefix`、`output_root`、`attempt_id`、`unit`、`catalog_source_refs`、`helper_source_refs`。所有 ref 严格使用 `{path, sha256}`。可直接复制 input contract 的两组静态 source refs；其他字段必须来自 root 当时的实际记录。

接收输入必须逐项提供：`catalog`、`handoff`、`predecessor`、`handoff_catalog_field`、`artifact_prefix`、`preparation`、`preparation_actual_exit`、`independent_review`。`preparation_actual_exit` 必须是原 `minicpm5_formal_metadata_root_actual_exit_v1` 语义，包含真实 tool session/chunk/exit 0、owners_gone、physically_drained、原锁核验，以及对应 preparation/catalog/handoff；本 CLI 不会生成这份外部实际退出证明。独立 review 必须精确绑定 preparation、它的实际 exit 和本次新 acceptance helper 的源码 ref，并审完全部 16 项候选内容，不能用本交付的源码测试代替。

CLI 先验证传入的完整目录／已结束 predecessor／保存的 idle 状态，再持原 RetainedLock 校验实际最新 handoff 和实际空闲状态。原 root 准备的 28 GiB 根盘、30 GiB NVMe 空间下限保留，完成前后均调用原 reader unchanged、lock check、active_heavy 与最新 handoff 检查。所有目的路径必须全新，文件使用原 O_EXCL/O_NOFOLLOW/fsync 发布方式。不会改 current handoff，不创建模型运行输出目录。

metadata CLI 的 CPU soft/hard 为原 **600/605 秒**，地址空间 **2 GiB**，metadata alarm 为原 **600 秒**。实际 formal runtime plan 的 wall 仍是 **900 秒**。9 项 native/source 和 1,284 项 capture、完整历史、原权重精度、冷缓存、长解码与请求槽位复用要求均保留。本交付没有添加 timing marker，没有证明之前 SIGXCPU 已解决。

CLI 在结束时发布 `*-metadata-terminal.json` 或 `*-acceptance-terminal.json`，其中 `actual_external_exit_pending=true`，这只是自身完成记录。root 仍须从实际 tool 结束结果建立独立 actual-exit，不能将自报 terminal 等同于外部实际退出。

## 检查结果与范围

7 项源码／mock 检查全部通过：实际工具 session **12891**，terminal **dc61f8**，exit **0**。记录：`minicpm5-formal-v7-root-helpers-source-checks-20261112-v1.json`，SHA `423cdb97bf5537543ddc8caf03e54501bc01b2994ec6c369d8e49816d19e6a2d`。

检查实际执行新完整组装器和 outer input validator；用明确 mock 替代历史 prepare、PV validator、diagnostic validator、实际 exit 与独立 review。测试中的 mock 只在内存里存在，没有落盘成当前接受材料。生产代码始终调用普通原 validator 和原 historical reconstruction，没有这些 mock。

覆盖完整 source/caller additions、原 contract、183 个 bootstrap、v7 精确 entry/argv、17 个准备输出、11 个接受输出及依赖 SHA；拒绝缺失/变化源码、运行中 handoff、错误目录/前驱、未完成或失败 exit、错误 preparation/finalizer ref、未接受独立 review 及改动资源限制。另检查全局模块没有动态 rebinding，主入口原锁／空闲／资源／exclusive write 条件保留。

保存的 r5 fixture 中，完整 source map 为 **48,238** 项（其中 183 项 bootstrap），source release **8,434,472 B**，加原 65,536 B margin 为 **8,500,008 / 16,777,216 B**。最大接受材料为 **82,107 B**，完整 argv JSON 为 **39,869 B**。这些是源码夹具大小，不是下一次真实输入大小；实际绑定必须再次计算完整预算。

可复现源码检查，输出路径必须不存在：

```sh
/home/ubuntu/venvs/vllm_neuron/bin/python -I -B /home/ubuntu/experiments/trn1-latest-models-20261031/research/check_minicpm5_formal_v7_root_helpers_v1.py /home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-formal-v7-root-helpers-root-checks-20261112-v1.json
```

本交付仅准备上述 root 操作的源码。实际 metadata 运行、其真实 completion、独立候选 review、实际 acceptance 和后续单次 launch 都尚未发生；现有 r5 launcher 仍固定旧 v6 参数，不能直接用于新 v7。新接受材料会提供准确的新 launch argv，root 应基于那份实际接受结果安排后续启动。
