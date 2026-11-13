# Additional KV256 S16 assembly control: source-only delivery

The fresh `qwen35_kv256_s16_concat_control_v1.py` is inert on import. It does not
modify the writer, model, geometry, saved diagnostics or any current binding.
No native library has been imported, no tensor allocated, no model/artifact read,
and no compilation or device execution has been performed for this delivery.

## Candidate operation and caller contract

The registered computation has exactly two BF16 operands:

```text
prefix [4,1,256,256]  -> BitcastConvert(s16) --+
                                             +-> ConcatInDim(axis=2)
tail   [4,1,3840,256] -> BitcastConvert(s16) --+          |
                                              BitcastConvert(bf16)
                                                       |
                                              [4,1,4096,256]
```

There is no numeric `Convert`, `.to(dtype)`, dtype-view overload, arithmetic on
payloads, widening, NaN normalization or special-value filtering in this source.
The outer tensor wrapper rejects other shapes/precisions and nonmatching XLA
devices. Both the lowering operands and its result are checked for exact shapes
and BF16 type. S16 is internal and equal-width.

`make_assembler()` explicitly imports the SDK only when invoked by a future
admitted caller. `_build_assembler(xla_call, xb)` is the same construction with
injected APIs for source mocks. Registration is performed once per factory call;
the returned SDK operation owns its normal compiled-computation cache.

`make_control(initial_cache, prefix_writer)` builds a module, without tracing it.
Its future caller must supply the exact original pinned NKI writer and initial
BF16 state; this module does not select a checkout or read saved inputs. It keeps:

- One BF16 full-size state, `cache`.
- `old = cache[slots.clamp_min(0)]` as physical output 0.
- The original writer on `cache[:,:,:256,:]`, updates and the same int32 slots.
- The untouched `cache[:,:,256:,:]` as the second assembly operand.
- The assembled BF16 full state as physical output 1.
- `alias_map(control) == {control.cache: 1}` for the future trace caller.

The caller must explicitly pass that map to `torch_neuronx.trace`; returning the
map does not itself establish an actual runtime alias. The current prototype
supports only KV256 and makes no claim about other buckets or real graph layout.

## Fixed installed API evidence

The source files were read as text from
`/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages`:

| Source/API | Observed signature or behavior |
| --- | --- |
| `torch_neuronx/xla_impl/base.py:15` | `xla_call(opfn, name=None)` returns `xla_op_registry.register(name, opfn)` |
| `torch_xla/core/xla_op_registry.py:46` | `register(name, opfn)` returns an `Op` callable |
| `torch_xla/core/xla_op_registry.py:23` | `Op.__call__(*args, **kwargs)` supplies tensor shapes and scalar kwargs to `create_computation`; one result returns a tensor |
| `torch_xla/core/xla_builder.py:148` | `Op.shape()` returns a `Shape`; `dtype` and `sizes` are properties |
| `torch_xla/core/xla_builder.py:434` | `Op.bitcast(self, to_type)` emits `BitcastConvert` |
| `torch_xla/core/xla_builder.py:717` | `Op.concat_in_dim(cls, ops, dimension, builder=None)` emits `ConcatInDim` |
| `torch_xla/core/xla_builder.py:817` | `create_computation(name, fn, shapes, **kwargs)` calls the lowering with one `Op` per tensor operand and builds the returned `Op` |

The adjacent `Op.cast()` emits numeric `Convert`; the candidate never calls it.
`Type.S16` and `Type.BF16` are the literal strings `s16` and `bf16`, with installed
Torch mappings to `torch.int16` and `torch.bfloat16`. These are front-end source
observations only, not evidence of a particular optimized HLO or device lowering.

## Pure source and mock checks

The following command actually exited 0, tool chunk `d1a5ed`, with 18 tests:

```text
python3 -I -B research/test_qwen35_kv256_s16_concat_control_v1.py
```

Tests parse the installed source AST and extract only the selected pure function
definitions of `xla_call`, `register`, `Op.bitcast`, `Op._extract_xla_ops` and
`Op.concat_in_dim`. These definitions execute with explicit fake globals and
synthetic shape-only objects. The SDK modules themselves are never imported.
The fake `mkop` rejects every operation except `BitcastConvert` and `ConcatInDim`.
Checks cover exact operand order, S16 types, axis 2, BF16 output shape, actual API
signatures, fixed alias/output order, the unchanged writer arguments, and rejection
of wrong dtypes, shapes, slots and devices. A meta-path blocker rejects native
imports; a test intentionally verifies that the future native factory is blocked.

These tests verify source wiring and error handling. They do not prove bitwise
device correctness, actual aliasing, compile support, memory usage or speed.
The evidence JSON records source digests and direct dependency roles. Native
transitive environment closure has not been resolved: a future admitted runtime
must include the original full environment/source closure and the newly added
control, plus the exact writer binding, before any native factory is invoked.

## Required future runtime evidence

After the independent r2 audit and the planned storage work, a separate admitted
diagnostic may add a matched BF16/S16 assembly pair. Preserve the same compiler
flags, sliced original prefix, writer, int32 slots, public BF16 state/output
shapes and alias 1. Save the actual input HLO and inspect the invoked computation
and dataflow: exact prefix/tail operands and slice bounds; two BF16-to-S16
bitcast-converts; S16 concatenate on axis 2; S16-to-BF16 bitcast-convert; and no
numeric Convert or arithmetic in that assembly path. Verify actual alias mapping.

Inspect the optimized lowering when available. An input HLO containing the
intended operations does not prove that later compiler optimization keeps the
integer assembly. Original raw, finite and row-tagged inputs must still undergo
complete old/new/host-input/tail/persistent-state byte checks. Keep full mismatch
positions and special-value bit patterns; do not relax comparisons.

This is an **additional** control. It does not replace missing alias-0, tail-copy
or full-copy outcomes, the original eight-layout/32-graph/40,000-call geometry
matrix, or any full-model cold-cache, long-decode, request-slot-reuse, quality and
matched HTTP check. It does not qualify a writer or establish a speedup.
