# Matched KV256 BF16/S16 assembly pair: concrete leaf and checker

This is a fresh source-only successor of the existing diagnostic leaf, with no
new admission framework. No frozen source was changed. No current input, raw
tensor, artifact, History, lock, native library, compiler or device was accessed.
Actual serial binding remains with root after the active MiniCPM5 work and its
storage/terminal checks.

## Future runnable entry

`qwen35_kv256_s16_pair_leaf_v1.run_diagnostic(directory, *, guard, save_report,
record_artifact, source_pins)` accepts the existing admitted owner callbacks and
a fresh output directory. It is inert on import and has no standalone launch CLI.
The future wrapper must retain the original complete source/environment binding,
serial owner/lock and process identity, SDKEnvironment restoration hook, child
ACK, descendant cleanup, disk floor plus remaining reservation, 6 GiB memory,
900-second leaf deadline and 1 GiB complete owned-output limit. The leaf retains
the original `check('before_compile:...')` post-import SDK hook boundary. It does
not establish any of those external ownership/resource conditions by itself.

The fixed two arms are `bf16_cat` and `s16_bitcast_cat`. Both use the already
delivered control module's same `_build_control`, the same pinned
`store_slots_nki_decode`, same finite BF16 initial state, same compiler flags,
same four int32 slots, full BF16 state `[4,1,4096,256]`, and `(old,new)` outputs
with `{cache:1}` aliases. Only the assembler callable changes. BF16 uses the
original `torch.cat((prefix,tail),dim=2)`; S16 uses the earlier equal-width bitcast
assembler. Both graphs compile freshly, are serialized, reloaded and run serially
with at most one live graph. Existing saved model binaries are not reused.

Each arm runs raw, finite and row-tagged raw with `[-1,-1,-1,-1]` slots, for six
calls total. Full cache reset bytes are checked before each call. It preserves
the full old/new/device-state captures, complete difference bitmaps and per-row
prefix/tail hashes and counts: 18 complete state comparisons. It additionally
saves both complete post-call host inputs for every call (12 files), so the
independent audit can verify unchanged-host booleans as well as mutations. The
started/returned counters are saved at their actual boundaries, before byte proofs,
so partial saved calls are distinguishable.

All six raw outcomes are retained, including failures; diagnostic completion
does not mean correctness. `s16_all_profiles_byte_exact` and
`both_finite_profiles_byte_exact` are independently derived from the complete
calls and cannot be set true over failures. All geometry/model/speedup flags stay
false. A failed finite or S16 byte comparison cannot qualify the intervention.

`validate_report()` requires the exact six-call matrix, both artifacts/HLO proofs,
same NKI function literal, full raw/bitmap sizes, all per-row prefix/tail regions,
both host captures, exact reset/output order and actual state evidence.
`audit_saved_bytes(report, *, guard)` is the separate future admitted post-exit
reader. It re-derives both HLO proofs and every saved raw/bitmap/input/host check;
no file is read by importing either function.

## Actual input HLO proof

The leaf saves the SDK's actual `graphs/<arm>/model/graph.hlo` reference with full
digest/size. The installed `trace.setup_compiler_dirs` writes this exact compiler
input protobuf; no synthetic substitute is used at runtime. The artifact and HLO
are retained even if the dataflow check rejects the graph. The HLO check precedes
loading the compiled graph for the six diagnostic calls.

`qwen35_kv256_s16_pair_hlo_v1.assembly_hlo_proof(payload, arm)` is a bounded
stdlib protobuf-wire reader tied to the installed descriptor fields. It follows
actual entry operands, called computations and tuple projections from output 1,
rather than counting strings or accepting disconnected bitcast markers. It proves:

- One full BF16 state parameter, one BF16 prefix update and one int32 slot input.
- Exactly two full BF16 outputs and output-1 `MUST_ALIAS` to that state parameter.
- BF16 arm: full BF16 concatenate on dimension 2.
- S16 arm: two equal-shape BF16-to-S16 bitcasts, S16 concatenate on dimension 2,
  and one equal-shape S16-to-BF16 bitcast to the full output.
- Tail comes from the exact `[0:4,0:1,256:4096,0:256]` slice of the original state.
- Prefix comes from the original NKI target/function, through only exact BF16
  reshapes/copies; its three inputs are the original prefix slice, actual updates
  and original int32 slots. The NKI function-literal hash must match both arms.
- No numeric Convert, arithmetic, transpose, different slice, different source,
  reordered concat operand or floating concat is accepted in the verified assembly
  path. Unrecognized lowering forms are rejected, not silently generalized.

The proof deliberately does not claim to interpret the NKI kernel IR or prove the
old-gather HLO semantics. Frozen writer-source binding and full raw old/new/state
tests are still necessary. Actual post-compiler machine instructions are also not
proven by input HLO; the checker reports that limitation explicitly.

## Source validation and evidence

The direct test command actually exited 0 with tool chunk `eec8ed`:

```text
python3 -I -B research/test_qwen35_kv256_s16_pair_v1.py
```

All 39 stdlib-only checks passed. They use tiny synthetic protobuf graphs and
metadata only, never actual model inputs or artifacts. Negative cases include
numeric converts on every bitcast edge, wrong concat dtype/axis/order, wrong
slice bounds/source, wrong writer target/function/inputs, wrong alias/kind/state,
wide slots, missing bitcasts, cycles, duplicate IDs, malformed protobuf, omitted
profiles/device-state/tails/host bytes, inconsistent success flags and false
qualification. Tests check the fixed SDK source descriptor field numbers and
the actual input-HLO write-path source. Native imports are actively denied.

An initial tiny test fixture mistakenly used BF16 `0x7f01` (normal) while expecting
a signaling NaN. The test failed at tool `8d0bd0` / exit 1; its expected input was
corrected to `0x7f81`, without changing production classification/comparison code.
The 28-test intermediate suite passed at `e85bbe`; the final expanded suite passed
at `49f577` and, after the exact HLO import-path assertion, at `eec8ed`.

Local dependencies are the unchanged raw-byte helper and unchanged S16 control
module, plus the two fresh pair modules. Native imports occur only inside the
future admitted leaf/factory; root must enroll their original complete transitive
environment/source closure. The manifest records these direct dependencies and
the original leaf/diff provenance; it is not a current runtime binding.

This pair is an additional intervention. It cannot replace the missing alias-0,
tail-copy or full-copy outcomes from the original eight-graph/24-call diagnostic,
the original eight-layout/32-graph/40,000-call writer matrix, or the model's
precision, cold-cache, long-decode, slot-reuse, quality and matched HTTP gates.
