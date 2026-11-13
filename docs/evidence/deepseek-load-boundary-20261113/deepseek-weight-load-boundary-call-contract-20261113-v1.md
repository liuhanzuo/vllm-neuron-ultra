# DeepSeek: original off/on load-boundary wiring and a finite semantic-proof path

This is SOURCE ONLY. No actual artifact, model, weights, raw tensor, HLO/NEFF, current lifecycle JSON, History, process namespace or lock was read or run. Only Python source and in-memory synthetic fixtures were used. The original frozen entrypoints and accepted observer/descriptor remain unchanged. No actual service, compiler, model inference or recovery was started.

## Delivered boundary

Four new runtime source files provide the missing connection:

- `deepseek_weight_load_boundary_v1.py`: the shared exact archive-binding reader, accepted load observer, pure partial-report linker and off/on partial comparison.
- `deepseek_cold_control_device_entry_v2.py`: the complete original off DEVICE entry, with its one `model.load` enclosed by the shared boundary after original native imports.
- `deepseek_capture_graph_load_boundary_v1.py`: the complete original capture graph `load` function, using the frozen original authentication helpers, with only its original `model.load` enclosed. There is no new compile entry.
- `deepseek_cold_boundary_device_entry_v2.py`: the complete original on DEVICE entry, calling that new loader and forwarding the two explicit boundary inputs.

Both original DEVICE fresh-process checks execute before native imports. Capture-on also executes the original graph loader's separate `_begin` fresh check. The existing off `_native_model_class` or on `_native_dependencies` executes before the observer context is entered. There is no outer observer around a frozen `run`, no hidden pre-import and no weakened fresh check.

The original model classes, constructor/configuration, artifact authentication, original 39-file capture build authentication, `model.load(str(artifact), skip_warmup=True)` arguments and all original downstream driver bodies are preserved. Tests remove only the explicit new import/context/keyword forwarding from the AST and compare the complete original off run, on run and graph load ASTs. The scientific drivers are retained, but this partial-evidence successor deliberately cannot reach them until an independently implemented semantic release successor exists.

## Exact future call contract

Existing off/on positional arguments stay exactly the same. Both new device `run` functions additionally require keyword-only:

```python
weight_load_binding={
    "schema": "deepseek_weight_load_boundary_binding_v1",
    "arm": "off",  # "on" for the capture arm
    "archive": {
        "path": "<exact selected artifact>/model.pt",
        "sha256": "<independently authenticated full artifact digest>",
        "identity": {
            "device": 0, "inode": 0, "size": 0,
            "mtime_ns": 0, "ctime_ns": 0, "mode": 0, "nlink": 1
        }
    },
    "descriptor_limits": {
        "archive_bytes": 0, "pickle_bytes": 0,
        "members": 0, "code_bytes": 0
    },
    "checkpoint_provenance": {"<original authenticated complete provenance>": "<value>"}
},
retain_weight_partial=<authenticated owning worker's evidence sink>
```

The zeros/angle brackets above are documentation placeholders, not valid inputs or a runtime binding. Identity is the independently observed full seven-field archive stat record. All four caps are positive finite integers drawn from the already reviewed resource envelope; this component does not allocate a new budget. Checkpoint provenance is the externally authenticated complete original checkpoint/source-view provenance, and must agree between off/on when their reports are compared. A nonempty dictionary alone is not proof of original weights; the existing outer source/weight manifest authority and full serial execution guard remain mandatory.

Off requires the archive path/digest to equal the original `expected_artifact_sha256[artifact/model.pt]`. On requires them to equal the exact file entry of the original independently authenticated capture-build report. The accepted descriptor reader then independently authenticates and streams the whole same archive before the one original SDK load. It does not select an archive or derive a supposedly trusted expected hash from the file being read.

The accepted `observe_load` wraps the actual original SDK `shard_checkpoint` and `load_weights` only once. After real initialization returns, but before `load_weights` returns to the outer `load`, the callback checks the archive identity again and links the full descriptor to both ranks' pre-layout and loaded-CPU-view observations. All graph-declared weight keys must occur in every observed rank. Additional observed keys are listed explicitly; no key or tensor byte is dropped. Declared WLT presence must agree with the actual loaded object's observed WLT presence. Key coverage is explicitly not an ABI/layout semantic claim.

The sink receives a deep copy of `deepseek_bound_load_partial_evidence_v1`, containing the full original binding, full descriptor, full actual load observation, their hashes and exact linkage. The sink may persist that report under the outer authenticated worker's existing evidence budget. Its return value, including `True` or an object claiming admission, is ignored. The callback then raises `LayoutSemanticsUnresolved` **inside `load_weights`**. Original sharder/loader errors, guard revocation and sink errors propagate unchanged, and both observer method hooks restore through their original finally paths. No warmup, forward, driver call or model return is granted.

The outer caller must not turn this exception into model success, call the original loop manually, restart the load in the same process, or mark any of the original seven stages complete. The unchanged seven-stage pre-cost layout gate remains closed. Separately reviewed serial evidence production may collect these partial observations, but this delivery is not such a runtime request. Actual producer tool exit, ownership, drainage and immutable parent sealing must follow independently; every child partial report has `actual_producer_exit_observed=False`.

`compare_bound_evidence(off, on)` first regenerates and checks each entire report, including exact descriptor/observation digests. It calls the accepted complete load and serialized comparators without changing their interpretation. Even equal CPU views and equal saved descriptors do not release inference. Its `next_required_observation` identifies the concrete missing compiler/layout evidence below.

## What semantic evidence can close the gap

A finite strict producer is possible under the same fixed compiler/runtime trust already required by the original Neuron model. It need not demand an undocumented device allocator dump. It must connect **actual loaded values → the actual WLT program → actual graph parameter positions/layouts → preserved original output computations**. The current two partial tools provide the first observations and archive descriptors; they do not yet supply the compiled-HLO relationship.

The concrete unobserved quantity is the complete coordinate interpretation of every weight at the actual compiled entry parameter, including any transformation from its original checkpoint coordinate. This can be derived from the actual compiler-produced weight-layout stub and the actual HLO passed to compilation, with a witnessed binding to the returned NEFF. It cannot be derived from the SDK's potentially stale inference MetaNeff shape alone.

The strict producer should require all of the following actual inputs and checks:

1. **Fixed implementation and actual calls.** Bind the exact existing SDK Python closure, `libtorchneuron.so`, runtime and compiler binary/version/configuration hashes through the original environment authority. Keep the observed one original sharder, one original `load_weights` and original `NxDModel.initialize` call. The actual saved `NxDModel.initialize` and `initialize_spmd_models` code must be tied to its serialized class and checked to apply the same WLT and pass `self.weights` to every one of the actual 12 SPMD objects with the original rank. Merely finding a method name, or hashing only the currently installed source while ignoring saved code, is insufficient. Hash equality of all saved code is a useful strict case, but legitimate capture-output code differences require an exact scoped code comparison, not blanket acceptance or omission.
2. **Complete TP2 inputs and transformed values.** Compare all actual original shard keys, rank assignments, dtype and exact bytes before transformation, then retain the complete loaded CPU observations as a corroborating output check. A mismatch identifies a concrete rank/key; neither equal shapes nor equal `.cpu()` values by themselves establish the device-side argument interpretation.
3. **Actual WLT identity and coordinate program.** Bind the archive's complete WLT NEFF/metaneff/tag/ranks to the WLT compilation's exact input HLO and returned NEFF. The first strict route requires identical WLT programs and complete inputs across arms. Every weight must have either the exact original identity transform or a fully parsed, validated index transformation; no missing weight is implicitly treated as identity. Do not use the SDK CPU-WLO helper as an independent oracle: its own source states that CPU/device WLO may differ. An unsupported opcode or unequal program is an explicit failed strict-path check, not proof that model values differ.
4. **All 12 positional graph bindings.** Bind each graph's archive NEFF/metaneff to its actual compilation input HLO and returned NEFF. Use the original `weight_name_to_idx` mapping plus the actual parameter numbers and actual declared key positions, rather than sorting keys and losing the parameter order. Check all dimensions, element types, layout/index transformations, partitions/rank interpretation, aliases and relevant frontend/backend attributes. For non-priority graphs use the *post-layout-optimization* HLO actually submitted to compilation; for the priority graph use its compiler-produced layout stub as well, since its earlier saved input HLO alone predates the compiler's weight-layout choice.
5. **Preserved computation, with an exact capture-output projection.** For every bucket, compare the full original logits and all 72 KV-output dependency DAGs against the corresponding outputs of the capture graph. The only accepted additional outputs are the exact 468 reviewed capture outputs with their required names/geometry and no change to original output computations, state aliases, dtype or precision. Reconstruct the actual flattened output mapping from the authenticated serialization/metaneff; do not assume arbitrary tuple positions. A comparator may consistently rename purely syntactic instruction/computation IDs, but must retain opcode semantics, constants, shape/layout, parameter numbers, operand order, control dependencies, called computations, custom-call targets, backend configurations and semantic frontend attributes. Unknown fields, side effects or unsupported normalization fail closed. Output-count equality or pruning arbitrary changed subgraphs is not this proof.

Once an actual producer passes these checks, its report must be independently sealed after real exit and drainage and bound to both actual load reports and both exact archives. A separately reviewed release integration can then require that exact proof. It still grants only permission to attempt the original diagnostic run, **not model qualification or a speedup**. The original complete raw neutrality and later numerical/quality gates remain the deciding experimental tests. If the strict DAG comparison fails because compiler capture changes alter an original-output subgraph, report the precise changed bucket/output/operation and keep the original release gate closed; a broader equivalence argument would need an explicit new review.

This is an actionable producer design, not a claim that an HLO semantic comparator has already been implemented or that the required real files exist. The delivered boundary closes the actual-load wiring gap while keeping this remaining implementation and evidence gap explicit.

## Exact existing capture points in the fixed SDK source

Source-only inspection of the configured SDK found:

- `neuronx_distributed/trace/model_builder.py` defines `GRAPH_HLO_FILE='graph.hlo'`, `WRAPPED_NEFF_FILE='wrapped_neff.hlo'`, `NEFF_FILE='graph.neff'`.
- Each actual bucket compiler input is written at `<compiler_workdir>/<tag>/_tp0_bk<i>/model/graph.hlo`. The compiler call receives the same `hlo_module.SerializeToString()` bytes. Non-priority graphs reach this point after `_add_layout_optimization_to_remaining_hlo`; the priority graph additionally emits `wrapped_neff.hlo` when layout optimization occurs.
- `_get_hlo_stub` reads the priority `wrapped_neff.hlo`. `hlo_utils.get_layout_transform_map` uses the `TRANSPOSABLE_WEIGHT_IDX` frontend attribute and the original `weight_name_to_idx` to identify every changed weight's complete transform computation. Unchanged weights must be accounted for explicitly from the complete map.
- `_prepare_weight_layout_transform_model` constructs the WLT from that actual stub and map, writes `<compiler_workdir>/layout_opt/metaneff`, and calls `generate_neff` in `layout_opt`. Its retained compiler input is `layout_opt/model/graph.hlo`; its output is `layout_opt/graph.neff`. The full returned WLT NEFF must equal the bytes embedded in the authenticated archive. The complete WLT output coordinate program is retained in HLO even though inference MetaNeff weight shapes have a documented SDK TODO.
- The two temporary `<tag>_<bucket>.hlo` and `<tag>_<bucket>_optimized.hlo` files created during layout conversion are explicitly removed by the SDK. A producer must not require them to still exist or silently substitute them for the final per-bucket compiler input.

The immediate next read-only task for the owning root, after the serial recovery is settled, is to inventory and authenticate those **existing** compiler metadata files and the original compiler input/output evidence for the exact baseline and capture archives. This child did not inspect their existence. A neighboring HLO file whose name looks correct is not proof of having produced an old NEFF: bind it through the original completed compiler report/cache or add an observed input/returned-output witness at the original compiler call in a separately reviewed build successor. If the original baseline HLO or compilation witness was not retained, record that exact missing artifact. A newly compiled baseline is not automatically evidence for the old model.pt and cannot replace it silently; exact artifact identity must be established under the original model/configuration/precision/budget and separately reviewed serial execution.

## Original constraints and validation

Original BF16 TP2, B4, context 4096, all six CTE and six TKG buckets, original model/checkpoint, sampler and numerical tolerances remain unchanged. The complete original native42/device84 collection, 126 logits, 9,072 KV tensors, 144 state bindings, 14 original failed frames, cold-cache and capture-neutrality checks, 2,041-step long decode/93 checkpoints, slot reuse, quality96 and matched HTTP remain required. This source task runs none of them and substitutes no lower precision, smaller model, smaller denominator or reduced workload.

The archive reader uses external finite caps, original shared/private storage budgets and owning guard. Its full archive hash adds reads, and the accepted observer can copy one complete tensor at a time to CPU in addition to original SDK load peaks. These costs must fit the existing reviewed budget; this component does not increase it or assert a lower requirement. Capture neutrality is still unproved, including the possible observation overhead.

Synthetic tests execute both actual new DEVICE entry bodies, the real new capture loader, accepted observer and accepted full inert archive reader. Only native SDK/tensors, filesystem and artifact bytes are injected. They verify fresh ordering, one original load/shard, same actual selected archive, complete 12-graph descriptors, all declared keys in both ranks, archive mutation, malformed bindings, original exceptions, sink refusal, hook restoration, exact off/on partial comparison and full original entry/driver AST preservation.

The first draft test tool `cfa7c4 / exit 1` ran 13 tests and reported 17 errors from one fixture error: patching `Path.stat` with an already bound fixture method lost the path argument. This also caused secondary assertion-context/index errors. Only the fixture was corrected to a normal lambda forwarding the path; no runtime predicate was relaxed. The corrected 13-test tool was `11541 / abc605 / exit 0`, 1.498 seconds. The final suite adds a complete bound off/on comparison and is recorded in the adjacent final test log/delivery. No real payload, native library, process, lock or network operation is permitted by the test audit.
