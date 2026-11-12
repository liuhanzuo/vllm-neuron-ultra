# Mini literal formal 绑定薄接线 v1

本接线只补原 assembler 未将八个 literal-boundary runtime 源加入 actual bootstrap 的缺口。旧 `prepare_minicpm5_observed_formal_binding_v1.py`、旧 native policies builder、冻结 literal 源交付中的全部十个 .py 和旧源文件均未修改。本次没有选择当前 catalog，没有写 candidate spec/review/policy/handoff，没有锁或 native 操作。

新增 `prepare_minicpm5_literal_formal_binding_v1.py` 无 IO/CLI/自动发现。prepare() 接收 root 已选择的原 request 与 ordinary original preparer；调用原 prepare、reconstruct_audit_base、audited.merged_plan，确保完整 map 一致，再调用本次 assemble。只为 original_request.additional_source_refs 增加旧 additions、八个冻结 runtime 源和新 preparer 自身 pin；原 request 的 catalog/audit/recovery/output 等字段不变，旧 pin 冲突直接拒绝。

assemble() 先要求完整原 source map 和 PV source_extension 已包含上述九个 source pins，再调用原 assemble 和原 policies.build。随后仅重绑原四项 draft 的值、完整 bytes 和引用 SHA：sdk-entry-template、outer-spec、outer-review-draft、outer-entry-plan。两个 role 的 bootstrap 与 outer.source_extension 同时完整追加八个 runtime 源；SDK successor 改 worker_source_entry_v6，outer successor 改 outer_source_entry_v6；runtime 源的原 import 使用 launch_v3。原 nocache_entry_v6 literal/source ref、native policies 内容规则、PV/diagnostic 科学图、完整 source_contract、资源限制和未授权 draft 状态不变。新 preparer 自身进入完整 map/PV extension，不作为 runtime bootstrap 必需项。

使用步骤：root 在原有真实 source preparation 环境中准备并审阅显式 request；保留 request.source_manifest 指向原 formal source manifest，不把 literal 交付误当成旧 schema 的替代物。将原 assembler 调用替换为下列调用，其余原历史 import/reader/text/guard 与输出落盘流程沿用：

```python
from research import prepare_minicpm5_literal_formal_binding_v1 as wiring

candidate = wiring.prepare(
    request,
    literal_source_manifest={
        "path": "/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-formal-literal-source-manifest-20261112-v1.json",
        "sha256": "153c98e371692fc5ef65785c08f9428ef6394528f5e5ec6ef86d3e3cf7463100",
    },
    preparation_source=PREPARER_REF,
    read=read,
    text=text,
    guard=guard,
    original=original_preparer,
)
```

`PREPARER_REF` 的完整 path/hash 见本交付 JSON 的 preparation_source，不能换路径或省略 hash。ordinary original_preparer 仍必须是原 prepare_minicpm5_pv_sdk_import_diagnostic_v1 模块，不能用这次测试 mock 代替。candidate 仍只包含待审核 draft；接受后的引用重绑必须从 candidate 中实际生成的 SDK/outer plan 出发，不能又套回旧 v5 module/path。真正 argv[7] 仍是完整冻结 v6 字符串，外层 deadline 参数仍为 new，由原 entry 建立唯一开始点；原 native/auth/runtime 义务仍全部执行。

如果 root 已经通过原完整历史方法得到 candidate/full，也可直接调用 wiring.assemble(request, candidate, full, literal_source_manifest=..., preparation_source=..., read=read, text=text)。这要求 candidate/full 是本次真实 prepare 含九个新增 pin 的结果；本次 r2 shape fixture 不可用作真实运行输入。

测试入口：`PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v research.check_minicpm5_literal_formal_binding_v1`，cwd 实验根目录。

5 项检查全部通过，session93000 / terminal16b9e0 / exit0。使用真实保存 r2 的 request、完整源 map、完整 contract、candidate 形状与原来源策略的保存 JSON，仅在内存中增加本次九个源来验证 assemble 形状。original.prepare 的调度测试使用明确 mock，不能算新的完整历史重建。没有把当前 v97/146 行绑定进来。

r2 形状实测：完整源数 48,057 → 48,066；两个 bootstrap 均由 154 → 162，八个新源全部存在；从新 runtime 源递归检查到的 26 个本地 research import 源均被完整 bootstrap 覆盖且 hash 匹配，旧 map 每个 pin 未替换。完整 release packet 8,405,271 B，加原 65,536 B reservation 仍低于 16 MiB；SDK entry template 28,151 B；outer entry plan 28,668 B。原 r2 catalog 144 行仅作为保存形状参考，原 descriptor plan SHA1838866c752ef628bda60cb63d2c9938a2009d0fb09fa77e5539802e36206294 不变；九项 native 与 1,284 科学行仍 pending。

还覆盖缺失/改动完整 map pin、缺少 PV extension、错误 preparer、错误 literal、runtime 文件字节变化拒绝；原所有 draft 不被接受；完整图引用重算相符；只有四个既有 artifact 变化；旧 preparer 与上一冻结交付 hash 逐一不变。
