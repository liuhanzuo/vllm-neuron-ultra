# Phi-4: frozen next scientific step (2026-11-07, v1)

Design only; queue after Mini. No runtime launch, raw/tensor reads, process probe, lock, installed/canonical change, or new qualification is implied. The accompanying JSON pins the authoritative p1 metadata analysis and root audit. Reuse immutable p1 evidence; do not create another full capture by default. The p1 results are CPU reference-boundary diagnostics, not new Neuron performance measurements.

## What p1 establishes

The full 4,608 cells contain 1,152 unique 2×2 groups across two layout panels. Per contrast/panel there are 576 selected-output cases (2 variants × 3 arms × 3 anchors × 32 ordered frames), not 4,608 independent trials.

| Contrast | Raw-equal cases | Maximum absolute difference |
|---|---:|---:|
| Same anchor state, canonical layout, native/port attention paths | 576/576 | 0 |
| Same anchor state, original recipient layouts/paths | 567/576 | 0.0001220703125 |
| Anchor/arm prefix state, fixed attention path (either path; either panel) | 402/576 | 0.0048828125 |

All ten source self-replays passed. Canonical path equality applies to the selected attention outputs under the captured inputs; it does not prove universal operator equivalence. Original-layout comparisons mix implementation with batch grouping, shape, padding, stride, offset and mask representation.

The state intervention fixes **all logical Q and the final/current token K/V** and replaces every earlier valid K/V. In prefill these earlier tokens are generated within that same prefill; they are not necessarily history from prior decode calls. It is a diagnostic endpoint substitution, not a consistent full-model trajectory.

| Canonical state contrast: raw-equal / 32 | Native single | Native batch | Port |
|---|---:|---:|---:|
| Instruct unchanged / wrapper (each) | 16 | 32 | 32 |
| Instruct candidate | 18 | 25 | 25 |
| Reasoning unchanged / wrapper (each) | 2 | 32 | 32 |
| Reasoning candidate | 4 | 19 | 19 |

First observed unequal state contrasts, in the frozen frame order:

- Against native single, all arms: instruct ordinal 1, fixture 5, prefill, slot 1; reasoning ordinal 0, fixture 0, prefill, slot 3. Both first differences have max_abs 0.00048828125.
- Candidate against native batch/port: instruct ordinal 10, fixture 2, decode step 2, slot 0; reasoning ordinal 8, fixture 0, decode step 2, slot 3. Both first differences have max_abs 0.00048828125. This is compatible with previous-step KV contributing, but does not identify its first differing generator.
- Original-layout path differences: instruct native-single anchor at ordinal 1, and reasoning native-batch anchor at ordinals 12 and 30 (fixture 0, decode step 3, slots 3 and 0). Each appears for all three arms: nine appearances, not nine independent witnesses. The two reasoning occurrences also reuse one fixture.

These are first observed selected-output contrasts, not the earliest differing token, operation, or global cause. Source order and frame order differ.

The same-input down-projection control already showed BF16 differences between batch-4 execution and concatenated row-view executions: instruct all four batch streams have 0/6 exact decode calls; reasoning native_batch/unchanged/wrapper have 1/6 and candidate 0/6. B1 prefill is a batch-split no-op. This control preserves the token axis and does not test padded versus unpadded prefill. Do not repeat it as a proposed discovery.

## Minimal intervention: trace generation, then fix its input

Keep the two mechanisms distinct: prefix values can differ because their generator received different hidden states, or because an operator executed differently on the same logical input. Cache insertion/readback is a third boundary, already constrained by p1 full-byte update proofs. A local execution effect can subsequently become historical KV; these explanations are not mutually exclusive.

1. **Reuse and align.** Under a future admitted replay, validate the pinned manifest/payload closure and source self-reproduction. Map every valid prefix token to its generating call and logical position, including the two slot-use groups. Compare complete layer-1 normalized hidden, packed pre-RoPE QKV, post-RoPE K, V and effective cache contents at aligned logical positions. Reuse p1 positions/cos/sin and before/after snapshots. Preserve physical padding/offset evidence separately; differently shaped raw hashes alone do not establish a logical difference. Identify the first observed differing generation boundary per witness and report missing boundaries explicitly. The saved values suffice for this pass; no model load is required.

2. **Only for the implicated generator, run a small same-input crossover.** Start with the two native-single prefill witnesses and two candidate step-2 witnesses above, plus a matched frame whose p1 contrast was equal. Keep identical valid hidden bytes, immutable weight/bias values, position and rotary coefficients. Compare the actual captured source execution geometries: unpadded versus padded prefill, or original B4 versus row views for decode. Preserve original padded tails and distinguish padding-domain changes from changes in logical data. Separately hold a complete physical input/shape/stride/offset/mask fixed while changing implementation where the actual source implementations differ. Never label a layout-and-path change an implementation-only effect. Exact source-geometry replay must reproduce the captured generator before crossing it.

   Full hidden/QKV/current-K/cos/sin are already captured. Only the layer-0 down weight was explicitly stored by the p1 capture; QKV weights must not be assumed present. Resolve the exact layer-1 QKV weight and any bias from the pinned original checkpoint only after admission, bounded to the required tensor(s), without another full model or persistent weight copy. Reuse source pins for actual linear/RoPE definitions. If pre-RoPE QKV agrees but rotated K differs, perform the corresponding same-input RoPE control. If hidden already differs and fixed-input generation agrees, the origin is upstream of this captured boundary; report that and request only the smallest missing upstream capture in a later admitted stage.

3. **Test whether the generated difference changes attention.** On the same canonical recipient, freeze all Q/current K/V and mask. Replace only the earlier prefix K/V generated in step 2, comparing baseline, sham same-source replacement and crossed replacement. Repeat in the reciprocal direction, keeping that recipient's Q/current K/V fixed within its own contrast. Verify every untouched byte and use raw equality plus descriptive error metrics. Require the generator self-control and the attention self-control first. A reproduced output shift demonstrates mediation at this selected attention boundary; it does not prove head-level recovery. K-only/V-only swaps are optional follow-up only if separating them changes the next implementation decision. Do not automatically launch another full factorial matrix.

4. **Separate the small original-layout residual.** Reuse only the three ordered witness occurrences plus an equal control. For identical complete physical operands, compare attention paths; for one fixed path, vary recipient geometry/padding/batch grouping while preserving logical values. Retain masked bytes and source strides/offsets. Record unsupported combinations instead of silently making a contiguous copy. This identifies local layout sensitivity independently of prefix generation. Canonical 576/576 is already the common-layout control.

The witness stage is discovery, not qualification. Any proposed single-target change must then cover all original 32 frames, both variants, all three arms and all three anchors. Retain fixtures 0–5, orders (0,5,2,4)/(1,3,0,2), slots (3,1,0,2), three teacher-forced decode steps, inactive slot and reuse checks. No new tolerance or replacement baseline.

## Decision and completion rules

- Different logical hidden, equal fixed-input QKV/RoPE: narrow the unexplained origin upstream; do not attribute it to attention or cache storage.
- Same logical hidden, different fixed-input generated KV, and reciprocal/sham-controlled prefix replay changes attention: an isolated generation/execution contribution is established for those cases. Report whether the varied factor was layout, padding, batching or implementation.
- Equal generated logical KV but unequal readback: investigate update/index/slot mapping, reconciling the existing p1 full-byte proofs before claiming corruption.
- Equal prefix and canonical path, original-layout-only difference: local layout/execution sensitivity remains. It can coexist with a separate generation effect.
- Null effects or incomplete saved boundaries: record the boundary limit; do not search arbitrarily until a favorable result appears.

Keep the original 18 gates verbatim in the JSON. All six original-HF comparisons remain false; instruct candidate additionally fails native_batch and original_port. No local control upgrades these gates. The original target remains layer-0 gate_up_proj, and a diagnostic change to another boundary is not silently substituted as its optimization.

After a justified candidate, the eventual goal is an actual full-model run from original tokens and empty caches, generating its own history under the unchanged numerical criteria; retain cold cache, long decode, request-slot reuse, output/quality checks and fresh paired performance. Teacher-forced three-step replay and transplanted endpoints cannot satisfy that goal or support a speedup claim.

## Resource and sequencing constraints

Admission follows Mini and must revalidate the live final predecessor catalog, released owners/groups including the p1 post-auditor, and serial ownership through the existing queue mechanism. The p1 audit explicitly had its auditor exit pending at write; this design makes no current liveness claim. Preserve current root/data floors and caps; no expansion, deletion, or floor reduction is authorized by this design. With tight disk margin, emit compact result metadata and only indispensable new witness tensors after calculating a bounded budget. Reference existing immutable raw evidence without copying it. If required evidence is unavailable, stop that diagnostic stage rather than replace it with weaker evidence. No large framework or runtime implementation is included here.
