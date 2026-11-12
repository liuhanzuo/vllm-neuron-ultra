# Mini 已结束 Qwen package 历史源码兼容：最小源后继 v1

真实 Mini r3 preparation（session91690 / terminalf3f7a6 / exit1）停在原 extension_shape，candidate/artifacts/native 尚未产生。根因精确为70个 repo .py：`/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/...`。它们同时存在于已结束 Qwen package r1/r2 的真实行145/146、已审阅 spec 与 bundle；相对原 inherited47,839 不在原ROOT或四项COMPLETED_LFM_TEXT域。SDK20项、NumPy testing utils 均已继承，不是新增extension问题。v5新增18个PJRT helper .py也全部已继承且SHA一致；extensionless helper仍只在bundle，不获新域许可。

只有两处当前入口需要共享兼容检查：原prepare.extension_shape及真实runtime PV.validate。原audited_v4.merged_plan只有 disjoint/no replacement检查，没有ROOT域限制，保持不改。原历史audit/compare/native spec validators仍检查原已结束历史spec，不向它们灌入新当前extension；完整runtime classify/text/native/physical/history检查继续原流程。

新增5个源码，旧源码不可变：

- minicpm5_completed_qwen_package_text_v1.py：固定70个精确path/SHA，固定真实v98 catalog及两个terminal/tool/spec/review/bundle引用；完整原v98前147行必须保留。运行期通过原authenticated reader读取完整catalog、两个terminal、两个bundle，验证真实失败且已排空、owners/units/tool一致及全70pin；不获得目录许可，不把失败改成成功，不借LFM名义。未知路径、改hash、部分70项、缺退出证据拒绝。
- prepare_minicpm5_pv_sdk_import_catalog_v2.py：保留原完整prepare主体，仅依赖显式ordinary original preparer并把extension check换成共享函数。原disjoint merge、两条完整runtime重建路线、完整catalog147行、descriptor plan与资源上限不变。
- minicpm5_fixed_input_pv_catalog_auth_v1.py：保留原完整PV authenticate主体和所有九native/physical/1284行历史义务，仅调用共享validate；原gate载入真实原模块后，显式走此authenticate入口。无monkeypatch/模块替换/运行期AST。
- minicpm5_sdk_mapping_outer_source_entry_v7.py：从冻结outer_v6只增加catalog_auth import及显式调用新gate-authenticate函数，其余全部方法/cleanup相同。SDK worker_v6、literal codec、实际完整nocache_v6 -c字符串保留。
- prepare_minicpm5_catalog_formal_binding_v1.py：薄接线追加上述runtime三源及prepare两源到完整map/PV extension；runtime三源追加两role bootstrap和outer extension，outer successor改v7，SDK保持worker_v6。继续原native policies builder和literal wiring，重算原四项draft引用，不生成accepted review。

两个Qwen完整spec各约9.25MB，原普通runtime JSON只有8MiB。本次没有扩大限制。源冻结测试按精确hash读取完整spec/review，验证terminal→spec/review→bundle及70pin完全一致；运行期重新读取固定真实terminal携带的同一spec/review精确ref以及完整bundle，避免重复打开大spec。历史链其他验证仍照原History。测试明确限制共享函数runtime读取每个输入<=8MiB，并确认不会读取那四个大spec/review。

调用仍由root原source preparation环境负责：

```python
from research import prepare_minicpm5_catalog_formal_binding_v1 as wiring
candidate = wiring.prepare(
    request,
    literal_source_manifest=LITERAL_MANIFEST_REF,
    preparation_source=LITERAL_PREPARER_REF,
    catalog_source_refs=CATALOG_SOURCE_REFS,
    read=reader.read,
    text=reader.text,
    guard=combined_guard,
    original=ordinary_original_preparer,
)
```

`LITERAL_MANIFEST_REF` 仍为上一literal source manifest，SHA153c98e371692fc5ef65785c08f9428ef6394528f5e5ec6ef86d3e3cf7463100；`LITERAL_PREPARER_REF`仍为prepare_minicpm5_literal_formal_binding_v1.py，SHA9509c6fe9956f495b6833f669ce66acc335e1d77955f1b129bace99ac1cef334。`CATALOG_SOURCE_REFS`取本交付catalog_source_refs，恰好五项，均只含path/sha256。ordinary original preparer仍是原prepare_minicpm5_pv_sdk_import_diagnostic_v1。request本身仍原schema，旧source_manifest也仍原formal source manifest。运行前root必须按实际latest catalog独立准备和审核；本交付不会选择/写当前输入。

10项检查通过（session56618 / terminal10555c / exit0）。源方法测试使用真实保存v98与新接线完整主体；原audit reconstruction/native/physical通过显式mock，不能称新的原历史鉴权或模型执行。真实原serial_catalog方法读取全部147行；原merged_plan两次且重叠替换拒绝；实际runtime.validate/authenticate/gate直到原reconstruct边界；完整prepare/authenticate/outer AST除明确列举的依赖别名与边界调用外相同；旧六个历史源码和两次冻结源交付hash全部不变。

v98输入shape实测（只表示具体source/mock输入）：完整接线source map48,162项；bootstrap165项；release8,421,248B加原65,536B仍小于16MiB；SDK template28,676B；outer plan29,193B。没有生成current spec/review/catalog/policy/handoff、没有锁/native执行，没有新性能结果。未来完整admission运行时仍必须按literal方案再次检查79,008B和16KiB总预算。
