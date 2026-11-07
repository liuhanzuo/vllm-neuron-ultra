# Phi-4-mini: local projection improvement does not pass full-output gates

The decode-row intervention makes the first layer's gate/up projection closer
to the native single-request reference, but **neither Phi-4-mini variant passes
the original full-output numerical gates**. This candidate is not promoted.
No inference speedup was measured by this diagnostic.

The subsequent **p1 attention crossover has now completed**. With identical
captured inputs in a common layout, both attention paths produced identical
selected outputs in 576/576 cases. Changing earlier prefix K/V changed 174/576
selected outputs. These CPU boundary results narrow the next investigation;
they do not identify a global first cause or repair the failed full-output gates.

The experiment uses original BF16 weights for:

- `microsoft/Phi-4-mini-instruct`, revision `cfbefacb99257ffa30c83adab238a50856ac3083`.
- `microsoft/Phi-4-mini-reasoning`, revision `0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a`.

This is a CPU dense-port investigation supporting the Trn1 adaptation work.
It is not an upstream vLLM-Neuron throughput comparison or a new device
qualification. The candidate calls the original first-layer gate/up linear
projection separately on each of four decode rows; prefill and inactive calls
retain their original execution. An unchanged arm and a passthrough wrapper
provide controls.

## What changed

Counts below are selected prediction rows whose complete vector bytes differ
from the native single-request reference, out of 32 rows per variant. Byte
inequality is diagnostic; the original model acceptance uses numerical
tolerances, not universal bit equality.

| Boundary | Instruct: unchanged → candidate | Reasoning: unchanged → candidate |
|---|---:|---:|
| Layer 0 gate/up output | 23 → 4 | 24 → 0 |
| Layer 0 down-projection input | 23 → 4 | 24 → 0 |
| Layer 0 down-projection output | 19 → 20 | 22 → 13 |
| Layer 1 attention output before output projection | 22 → 19 | 31 → 28 |
| Final logits | 28 → 28 | 31 → 31 |

Both candidates still match the reference's highest-scoring token in 32/32
sampled rows. That alone does not satisfy the original gates. Maximum KL
divergence remains about **0.020647** for Instruct and **0.027953** for Reasoning,
above the unchanged **0.02** limit. All six comparisons against the original
single-request HF reference fail across the two variants and three arms.

The Instruct candidate also loses two previously passing comparisons: against
native batched execution and against the original dense port. Its maximum logit
error against the port is **3.5**, above the unchanged **2.0** limit. Reasoning
retains those two passes. None of these findings justify enabling the candidate
as a validated optimization.

## Evidence and scope

The saved-boundary analysis completed **5,184 comparisons**, including
**576 full-vocabulary logit comparisons**, covering both variants, three arms,
three reference anchors, 32 logical frame occurrences and nine boundaries.
The underlying schedule preserves six fixtures, two slot layouts, eight
prefills, six four-row decode calls and one inactive-slot call per port arm.
The analysis loaded 114 saved payload files, including 104 authenticated
activation payloads, and retained all 18 original gate outcomes.

Original criteria remain: finite outputs, cosine similarity at least 0.995,
KL at most 0.02, maximum absolute error at most 2, top-1 agreement at least 90%,
and exact top-1 agreement when the reference margin exceeds 0.5. The diagnostic
completed successfully; successful analysis is distinct from model acceptance.

The independent root audit verified the complete comparison matrix, unchanged
source closure, successful worker/controller exits and process-group drain.
The unchanged and passthrough-wrapper arms have identical saved raw hashes.

- [Raw comparison records](evidence/phi4-boundary-20261107/events.jsonl)
- [Scientific analysis and witnesses](evidence/phi4-boundary-20261107/scientific-analysis.json)
- [Independent root audit](evidence/phi4-boundary-20261107/root-audit.json)
- [Evidence manifest](evidence/phi4-boundary-20261107/manifest.json)

This evidence bundle includes analysis metadata and archival source, not model
weights, binary tensor payloads or a standalone inference runner. Absolute
paths in archived records refer to the original experiment machine.

## Completed p1 attention crossover

Matching one selected query row does not establish that the full attention
inputs match. Prefill attention already differs in 2/8 Instruct and 7/8
Reasoning rows despite matching selected-row QKV. Decode cache history therefore
cannot be asserted as the sole cause.

The completed p1 run captured complete Q/K/V, masks, positions and prefix-cache
versions, then crossed attention path and prefix state. It retained both the
original recipient layouts and a common-layout panel. All **10 source
self-replays passed** before interpreting the crossed results. The root audit
verified diagnostic completion, all 5,192 raw-file hashes, and the original
4,608-cell scope. This document update reads only saved JSON metadata; it does
not repeat that payload audit or run a model.

Each contrast in each panel contains **576 selected-output cases**:
2 model variants × 3 arms × 3 reference anchors × 32 ordered frame occurrences.
There are 1,152 two-by-two groups across two panels, each with four cells,
giving 4,608 cells. Those cells and repeated arm appearances are not independent
trials. The following counts describe exact selected-output bytes, not the
number of full-model numerical gates passed.

| Comparison | Exact cases | Largest absolute difference |
|---|---:|---:|
| Same anchor prefix state, common layout, two attention paths | **576/576** | 0 |
| Same anchor prefix state, original recipient layouts and paths | **567/576** | 0.0001220703125 |
| Anchor versus arm prefix state, fixed attention path | **402/576** | 0.0048828125 |

The last row holds separately for each path in each layout panel; it does not
pool four denominators. The intervention holds **all logical Q and the final
token's K/V fixed**, replacing only earlier valid K/V. During prefill, those
earlier tokens are generated within the same call. Thus this is an endpoint
substitution, not a consistent model trajectory or proof that prior decode
cache updates caused the original error.

Common-layout equality applies to these captured selected outputs. The
original-layout comparison also changes batch grouping, shape, padding,
strides, offsets or mask representation, so it cannot isolate implementation
alone. Its nine unequal appearances reduce to **three distinct ordered witness
occurrences**, each repeated across the unchanged, wrapper and candidate arms:

| Variant / anchor | Frame ordinal | Fixture / phase / step / slot |
|---|---:|---|
| Instruct / native single | 1 | 5 / prefill / — / 1 |
| Reasoning / native batch | 12 | 0 / decode / 3 / 3 |
| Reasoning / native batch | 30 | 0 / decode / 3 / 0 |

The two Reasoning occurrences also reuse one fixture. These are selected-output
witnesses, not nine independent discoveries or the earliest differing internal
operations.

The same-input down-projection control also completed. For B4 decode, all four
Instruct source streams had 0/6 byte-equal calls between batched execution and
concatenated row-view execution. Reasoning native-batch, unchanged and wrapper
streams had 1/6; its candidate had 0/6. Splitting the batch preserves the token
axis. B1 prefill is a no-op for this split, so this control does not test padded
versus unpadded prefill.

All **18 original gate outcomes remain unchanged**: all six original-HF
comparisons fail, and the Instruct candidate additionally fails against native
batch and original port. No local crossover upgrades these outcomes. p1 is a
CPU reference-boundary diagnostic; it supplies no Neuron throughput or full-model
qualification result.

- [Compact p1 analysis and all 18 gate outcomes](evidence/phi4-attention-cross-p1-20261107/analysis.json)
- [Root audit metadata](evidence/phi4-attention-cross-p1-20261107/root-audit.json)
- [Export hashes, original records and complete source/evidence references](evidence/phi4-attention-cross-p1-20261107/manifest.json)

The export removes repeated token-index arrays and stores the root audit's full
source/evidence pin map once in the manifest. Original record hashes identify
the unmodified local evidence. No weights or binary tensor payloads are bundled.

## Next experiment: prefix generation and local layout effects

The [next design](evidence/phi4-attention-cross-p1-20261107/next-design.md) is
prepared but has not run. It first reuses the saved p1 evidence to align every
prefix token with its generating call, then compares normalized hidden states,
pre-RoPE QKV, rotated K, V and effective cache contents. This distinguishes a
generator receiving different input from the same logical input executing
differently; a local execution difference can later become historical K/V.

Only an implicated generator gets a small same-input crossover with exact
source-reproduction controls. Its generated prefix is then substituted into a
fixed attention recipient, with sham and reciprocal controls. The three
original-layout witnesses receive a separate layout/path test with an equal
control. QKV weights are not assumed to be among the p1 saved tensors; resolving
them from the pinned checkpoint requires separate resource admission.

These witness tests cannot replace full coverage. Any justified candidate must
retain both variants, three arms, three anchors and all original 32 frames,
then undergo full-model validation from original tokens and empty caches with
its own generated history. Cold-cache, long-decode, request-slot reuse, quality
and fresh paired serving-performance checks remain required.
