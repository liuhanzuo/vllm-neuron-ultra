# KV256 bit-pattern repair proposal v2: saved-reference semantics

Status: **source analysis and frozen r3 metadata only; no repair implemented or
experiment launched**. This successor preserves v1. The current r2 diagnostic's
complete result, external exit, drain and independent raw-byte audit remain the
root coordinator's responsibility. No active output was read for this successor;
the r2 observations below were supplied by root.

## The saved `torch_where` result does not contradict r3

The original KV256 persistent `torch_where` graph completed 1,250 calls, but its
raw profile did **not** pass. Counting all rows in the frozen archived r3 report:

| Branch | Raw pass / fail | Finite pass / fail | Completed calls |
| --- | ---: | ---: | ---: |
| standalone `torch_where` | 0 / 625 | 625 / 0 | 1,250 |
| standalone `nki_predicated_decode_v1` | 625 / 0 | 625 / 0 | 1,250 |
| persistent `torch_where` | 8 / 617 | 625 / 0 | 1,250 |
| persistent `nki_predicated_decode_v1` | 0 / 1 | not reached | 1 |

`qwen35_writer_geometry_serial_v1/geometry.py:213` requires success when the
variant is NKI **or** the profile is finite. Original `torch_where/raw` failures
are retained as negative controls without stopping the loop. Line 216 checks
the count of completed calls; it does not promote those failures. Thus the root's
r2 observation that the saved persistent `torch_where` graph fails raw and passes
finite agrees with r3. It is not evidence of a previously passing graph regressing.

The saved graph identity also agrees exactly. The r3 report's third graph names:

```text
/mnt/nvme/neuron-work/qwen35-writer-geometry-writer-r3-20261113/kv256/artifacts/persistent-torch_where/model.pt
SHA256 9adedf8b7361a9b8999f51658bfb91126346c7fc096d8ef80fac0d697e53c45b
size   8514990 bytes
alias  output 1
```

These are precisely the `saved256_where_alias1` constants in
`qwen35_kv256_alias_diagnostic_leaf_v1.py:35`. `saved_path()` constructs the same
mode/variant path; the saved branch checks its digest and size before loading
at lines 201–204. This review compared saved metadata and source literals; it
did not reopen the model artifact or independently authenticate its binary.

The geometry compiler loop picks `text_core.store_slots` for this branch at
`geometry.py:144`, before creating its module and tracing with output-1 alias.
It is not an integer reference or the NKI writer under a misleading name.
`text_core.py:49` performs four successive `torch.where` selections. Indices
are 0–3, so slots `[-1,-1,-1,-1]` select the old prefix in all four rows.
The full graph separately concatenates that prefix with the untouched tail
(`geometry.py:124–125`). Source semantics require the full state to stay unchanged.

## What the first frozen r3 sample proves

For the persistent `torch_where/raw` first call, mapping 0, all four slots are -1:

```text
expected full-state SHA256
  585fbf90d9c212afe9be4e1620de1c21d0aaaee82186b012b926a80d34da71c4
actual new output and device state SHA256
  411e005a2720b9db8afbb7b6ef34d82bfb4fc9067cb408370ea48d49452e2961
expected tail SHA256
  f8ccf07f1fd1ce156c8fd983f93a411179a19f78ee4998ec63bcdc8e062340b2
actual tail SHA256
  161bb075a121f5dddac1b9777a373eaa54334cd7f2ae4426cbf7f66410d253b2
```

The old output is exact and both host inputs are unchanged. Across the 625 raw
calls the old output never fails; new output and device state each fail 617 times.
The tail fails only on the first call. That last count must be interpreted with
the source's state progression: after each persistent call, `expected` becomes
the observed device state (`geometry.py:204–206`). Later tail equality is equality
to the previous observed state, which may already contain the first call's
changes; it does not show that the original tail bytes were restored.

The eight passing raw mappings have indices 1, 6, 26, 31, 126, 131, 151 and 156.
They are results within this stateful sequence, not eight independent cold-state
proofs. No explanation of their exact lowered instructions is claimed.

For comparison, the first standalone `torch_where/raw` sample has a wrong full
new output but an exact tail. The first persistent NKI sample has wrong new/state
and tail, with new/state SHA256
`ed6bc91123d8ffc49e028b2685f894ed7f6ed131fab83fefe75f6f9a70ac059d`.
Root reports that r2 reproduces the latter exactly, and that all three saved
standalone-NKI profiles and all three saved KV128-NKI profiles are exact.

## Repair direction stays conditional on complete controls

The new metadata strengthens the distinction between an original floating
selection problem and the additional persistent tail-assembly problem. The NKI
writer receives only the prefix. The production model joins it with the original
tail at `vllm_neuron/models/qwen35/modeling.py:197`; geometry reproduces that
operation. A special-value-changing floating lowering is plausible, but neither
the source nor these hashes identify an individual compiler instruction or pass.

The narrowest first repair control remains **equal-width S16 bitcast assembly**:
keep BF16 state, prefix writer, alias and graph outputs; bitcast both BF16 prefix
and tail to S16, concatenate on axis 2, and bitcast back to BF16. The installed
`torch_xla/core/xla_builder.py:434` maps `Op.bitcast()` to `BitcastConvert`, while
`Op.concat_in_dim()` is available at line 717. This is source API evidence only;
it does not prove that a compiled graph retains the integer assembly path.

An NKI full-state assembler that explicitly load/stores the existing prefix and
untouched tail is the next alternative. A fused full-state slot writer is a
broader follow-up because it changes assembly and writer layout together. A
tail-copy control is informative but is not automatically a repair while outer
floating `cat` remains. Cat-only may simplify away; output-order alias controls
must preserve the model's eventual fixed output ABI.

Before implementing any candidate, finish the original eight-graph diagnostic
or retain its actual partial failure. If it cannot finish within the original
resource limits, resolve the measured execution overhead in a separately
reviewed successor; do not infer the unexecuted controls or enlarge the budget
from this proposal. A fresh matched BF16/S16 pair would need actual HLO operand,
slice, dtype and alias inspection plus complete raw/finite/row-tagged byte tests.
`.to(torch.int16)` is numerical conversion and is not this candidate.

After a repair, preserve the full eight-layout / 32-graph / 40,000-call writer
geometry matrix and all 625 slot mappings, then the original model precision,
cold-cache, long-decode, slot-reuse, quality and matched HTTP gates. The separate
46,368-call integer-reference matrix does not replace writer geometry. No model
qualification or performance claim follows from the current diagnosis.

## Frozen evidence used for this successor

Only Python source text, the root's supplied runtime scalars, and this existing
archived JSON metadata were used:

```text
docs/evidence/qwen-kv256-persistent-failure-20261113/kv256-report.json.gz
archive SHA256 23a13abda6d5082b18e63499a51ece3849c3fa651ad8e0eab8d698454d2dcda6
JSON    SHA256 2fbd42bcb95ea18ddfcd1c79509ecdb66f47cbea68c6f99410f8738c226ce209
```

The report binds `geometry.py` to
`ba22035fd25e52ec1cfe373b7fc0e844ffe83787f2f5ecd7e6e96a4cb277ad4d`
and `text_core.py` to
`3f0a966af288d87c55e72f5dd41b8f11f343b43d3c1f45385fa360d774aacdfb`.
No raw tensor, compiled artifact, native SDK import, History resolution, lock,
device execution, publication or current-state mutation was performed for v2.
