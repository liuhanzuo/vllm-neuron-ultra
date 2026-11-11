# Saved native/SDK boundary comparison: pure function only

Frozen source: `research/compare_minicpm5_layer1_native_sdk_v3.py`.
This adds no CLI, scheduler, filesystem implementation, lock acquisition,
process/service checks, model loading or operator replay. It must not be called
for the currently running capture or for partial/unaudited data. Root's existing
reviewed execution/audit path retains all actual terminal, serial, source,
physical, filesystem identity and resource responsibilities.

```text
compare(native_result, native_plan, native_audit,
        sdk_results, sdk_cases, sdk_plan, sdk_recovery,
        read_bytes, guard) -> complete diagnostic dictionary
```

`native_result` is the authenticated unwrapped original driver_v4; `native_plan`
is its original resolved plan; `native_audit` is the accepted successful
post-exit audit report. `sdk_results` are the two original authenticated SDK
rank0/rank1 results, `sdk_cases` are their original independently authenticated
AnchorStream cases in two lists of six, `sdk_plan` is the original SDK plan and
`sdk_recovery` is the completed original recovery audit report. The helper
requires complete native24frames/72calls/936raw and SDK2ranks/6calls/624raw audit
schemas/counts. Original SDK model exit1 is retained. These shape checks are not
authentication: the caller must independently authenticate reports/results and
their actual completed audit terminals before invoking the function.

`read_bytes(reference, absolute_path, exact_size)` must perform the existing
guarded, stable, owned, no-follow full byte read/EOF check. There is no default
reader. `guard()` runs before/after every read and at bounded scalar intervals;
it must retain the existing original lock/source/physical/storage conditions.
The comparator independently checks exact path, length, SHA256, unique reads,
finite numeric content, complete event manifests and preserved source layout
metadata. It returns no runtime admission and does not assert actual exits.

It reads **all native936 raw + SDK624 raw**, and all **72 native arm result
JSONs**, exactly once: **1632 input reads**. Native/SDK journals and donor data
are already covered by the required complete audits and are not reread here.
All SDK five endpoint controls, both ranks, every capture leaf, all four rows
and full prior padding are read and checked before selections. No whole-model,
cache or input payload is silently truncated to a sample.

## Exact alignment

The existing pure `frame_alignment_v1` validates each complete frame dictionary,
call, logical position, physical slot and official projection-shard identity.
Calls are **4,5,6,11,12,13**, with SDK buckets **2048,2048,2048,128,128,128**.
Each has all physical slots **0,1,2,3** and both ranks. Fixture order is not
assumed to equal slot order. There are 24 native frames and 48 rank/frame
comparisons, with complete `(rank,call,slot)` coverage checked explicitly.

For rank r, native query heads `[8r:8r+8]` align to SDK eight query heads;
native KV head r aligns to SDK one KV head. Projection/context columns are
`[1024r:1024r+1024]`; K/V projection columns are `[128r:128r+128]`. The four
original full BF16 projection matrix hashes must agree across native and both
SDK shard proofs. No matrix bytes are reread or multiplied by this function.

For native length L, p=L−1 is the absolute active position. SDK prior prefix
`[0:p]` is followed by the separately observed active tensor to align to native
`[0:L]`. SDK `[p:bucket]` is masked prior padding and has no native counterpart.
The comparator preserves each entire original tensor and reports every such
padding selection with hash, count, min/max/nonzero and original evidence.
Padding is never folded into valid keys or reported numerically equivalent.
Padding axes are selected by explicit event/path semantics, never by equal
shape dimensions: `effective_prior_kv` and `route_inputs_sdk_decode.past_key_value`
use axis2; prior score/probability/mask events and the route attention_mask use
axis3. Active scores, Q/K/V head_dim128, PV and returned_current_kv have no prior
padding axis, including bucket128.

Saved snapshots are packed logical byte sequences. Source stride/offset are
retained as provenance; they are **not reapplied** to already packed bytes.
Every comparison retains original references/metadata and the full ordered
logical selection chain, including physical slot and head/prefix selection.
The only transformations are byte slices, byte concatenation of separately
observed prior+active, and shape-only byte reinterpretation with equal element
counts. No matmul, scale, RoPE, softmax, PV sum or TP reduction is recomputed.

## Existing event boundaries used

| Native evidence | SDK evidence | Interpretation |
| --- | --- | --- |
| normalized_input; position_ids; rope_position_ids | same names | Full selected physical row |
| cache_before key/value | effective_prior_kv `[0]`/`[1]` | Native KV head r versus valid prior prefix |
| q_projection; k_projection; v_projection | same names | Official rank column slices |
| q_rope_input; k_rope_input; v_heads | same names | Q8/KV1 head slices |
| rope_cos; rope_sin | same names | Actual full row RoPE tables |
| q_post_rope; k_post_rope | same names | Actual observed rotation output |
| q_post_rope; k_post_rope; v_heads | q_after_prep; k_after_prep; v_after_prep | Adjacent actual SDK prep boundary |
| native_effective_key/value; cache_after | effective_prior_kv + k_after_prep/v_after_prep | Byte concatenation of observed valid prior and active only |
| cache_update_active_key | k_after_prep | Actual active key |
| qk0_unscaled | qk0_unscaled + qk1_unscaled | Full valid prior/active score alignment |
| qk0_scaled | qk0_scaled + qk1_scaled | Native multiply versus SDK divide provenance retained, neither recomputed |
| attention_mask; actual_eager_mask | attention_mask; prior_actual_mask | Native additive BF16 versus SDK Boolean semantic mask check |
| qk0_scaled prior prefix | prior_mask_input prefix | Adjacent pre-mask comparison |
| softmax_input prior prefix | prior_mask_output prefix | Adjacent post-mask comparison |
| softmax_input | softmax_input + softmax_active_input | Native BF16 versus SDK FP32 values; byte equality not claimed |
| probabilities_fp32 | probabilities_fp32 + probabilities_active_fp32 | Actual FP32 probabilities |
| probabilities_bf16 | probabilities_bf16 + probabilities_active_bf16 | Actual BF16 cast boundary |
| pv_context | context_before_o | SDK observed post-merge context reshaped to heads, no sum performed |
| context_before_o | context_before_o | Native query-head column slice |
| attention_output | o_proj_after_reduce | Actual full reduced SDK endpoint |

`pv_prior`, `pv_active` are **unknown** for cross-implementation split comparison:
native has no captured split. `o_proj_local_before_reduce` is **unknown** because
native has no captured rank-local output projection. The actual SDK tensors and
native available merged context remain in the evidence; they are not used to
fabricate missing native boundaries or an equivalent reference. Missing required
manifest events or partial captures reject the input rather than fabricating
an all-equal result. Mask encodings outside the explicitly recognized native
additive representation are unknown. A recognized nonzero native mask on a
valid position is a difference, not silently treated as equivalent.

## Statistics and first-difference meaning

Each comparable boundary reports complete selected tensor hashes, exact byte
equality when dtype matches, numeric equality, all-element mismatch count,
first numeric flat index, maximum absolute error and relative L2 using native
as denominator. BF16 words decode with stdlib struct by shifting into IEEE
FP32 bits; FP32, int64 and Boolean decode directly. No reference is replaced.
Different dtypes have `bytes_equal=null` even when values match. Zero native norm
with nonzero error has `relative_l2=null` plus an explicit undefined flag;
zero/zero yields 0. Nonfinite input rejects, and signed-zero byte differences
remain differences even if numeric error is zero.

The fixed boundary order follows input/cache → projection → RoPE/prep → QK →
scale → mask → probability → PV/context → output. Each rank/frame retains all
later declared boundaries even after a difference. `comparison_coverage` names
exactly those comparisons. `declared_comparison_scope_complete` applies to that
list, not every saved duplicate. `authenticated_sdk_leaves_not_individually_cross_compared`
explicitly lists all route_inputs_sdk_decode leaves, returned_current_kv leaves
and duplicate attention_output: their complete bytes are read/authenticated,
but there is no separate cross-implementation comparison claim for those
snapshots. q_after_prep/k_after_prep/v_after_prep are individually compared. `first_observed_difference` identifies
the earliest available differing boundary in that order, **not causal proof**.
Unknown split/local boundaries remain explicit even when every comparable
boundary matches. Report flags retain `arithmetic_reexecuted=false`,
`attention_causality_proven=false`, `full_model_qualified=false`, and
`new_speedup_claim=false`.

## Mock verification

Eleven v3 tests passed with no actual raw/native/process/service/lock operations:

1. Full 24-frame/48-rank-frame pass, all1632 reads exactly once, native936/SDK624,
   nonuniform query heads, fixture order different from physical slots, nonzero
   SDK padded suffix fully retained and excluded only from valid comparison.
2. Actual loaded comparison path with one changed scaled-QK value: reports
   qk0_scaled first, exact preceding QK and later softmax retained, all-element
   mismatch count/maxabs correct and no causal assertion.
3. Failed guard or incomplete native audit rejects before any byte read.
4. Wrong frame/slot/position/head mapping rejects instead of guessing.
5. BF16/FP32 cross-dtype equality, full maxabs/relative L2, zero denominator.
6. Packed logical slicing ignores original strides, concatenates only observed
   bytes, and retains nested slot/selection provenance.
7. Corrupt, truncated, duplicate or nonfinite byte evidence rejects.
8. Nonzero distinct KV heads and every prior position, selected physical slot2,
   observed active KV and distinctive padded suffix align correctly.
9. Wrong KV head, swapped prior positions and changed active V report the exact
   first available boundary.
10. Unsupported native mask encoding remains unknown; recognized blocking of a
    valid native position is different.
11. Bucket128 head_dim128 Q/K/V/PV/active snapshots never become padding; exact
    semantic padding set and explicit comparison/authentication-only coverage
    are checked.

The source-only static closure has five local files. Existing audited sources
and manifests remain unchanged. Actual collection and independent audit must
finish before any future reviewed caller executes this comparator.


## Measured output size and integration limit

The full synthetic 24-frame/48-rank-frame report measured 8,161,902 bytes as
compact JSON and 21,857,712 bytes with indent2, each including one final newline.
Its recursively counted, identity-deduplicated Python report object was about
15,476,677 bytes; comparator incremental tracemalloc peak was 21,981,977 bytes
above a 76,828,789-byte synthetic fixture. These are this synthetic process's
measurements, not a model-memory guarantee.

The plain full report **exceeds the original 2MiB individual output cap**.
This delivery remains pure/source-only and is not integrated into actual
execution. Keep the original limit; before publication use a separately
reviewed bounded lossless packed representation or an equivalent complete
reference-based representation. No new runner or automatic compression path
is supplied here, and no evidence may be silently omitted to fit the cap.

Root additionally reported six independent source/mock controls passing in
actual tool_call d5e8c6/exit0 (0.083s): 80 seeded full-coordinate selections,
four-axis concatenation oracle, nested physical-slot provenance, signed-zero
and cross-dtype FP32 drift, semantic axes at bucket128/2048, and final-valid-key
single-element divergence at slot2. They did not run model or actual payload IO.
