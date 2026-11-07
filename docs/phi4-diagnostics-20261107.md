# Phi-4-mini: local projection improvement does not pass full-output gates

The decode-row intervention makes the first layer's gate/up projection closer
to the native single-request reference, but **neither Phi-4-mini variant passes
the original full-output numerical gates**. This candidate is not promoted.
No inference speedup was measured by this diagnostic.

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

## Next experiment

Matching one selected query row does not establish that the full attention
inputs match. Prefill attention already differs in 2/8 Instruct and 7/8
Reasoning rows despite matching selected-row QKV. Decode cache history therefore
cannot be asserted as the sole cause.

The next planned diagnostic captures complete Q/K/V, masks, positions and
prefix-cache versions, then crosses attention implementation and prefix state.
It separates original physical layouts from a common-layout control, and
requires each original path to reproduce its captured output before interpreting
crossed results. A same-input down-projection control tests the remaining
projection discrepancy. This experiment is not yet executed. Cold-cache,
long-decode, slot-reuse, quality and serving-performance requirements remain.
