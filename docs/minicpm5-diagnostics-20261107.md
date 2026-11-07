# MiniCPM5 observer and attention diagnostics — 2026-11-07

The replacement Python trace has completed the 88-context acquisition. Saved
data comparisons now cover the first five attention boundaries and three RoPE
boundaries. Inputs, K projection, RoPE coefficients and rotated K agree exactly
in the observed comparisons; Q and V have a small number of differing elements.
Saved decode masks also confirm an input-contract error in the diagnostic
harness: it admits the current cache address in the historical branch, while
the SDK separately processes the current token. A controlled four-frame CPU
replay now changes the numeric pass count from 1/4 to 4/4 after correcting only
that mask. A complete 32-frame replay on the compiled Neuron artifact improves
pass count from 17/32 to 30/32. One prefill and one decode frame still fail KL;
the model remains unqualified. Saved prefill comparisons now locate a further
numerical difference at the output projection path; causality for the remaining
final-model failures is unproven. Independent result reviews are below; the
earlier observer failure is retained as historical evidence.

The subsequent full-model CPU TP2 experiment has completed. Its strict-mask
baseline and exact repeat pass 29/32 frames; promoting all 42 attention output
projections to FP32 local products and reduction passes only 27/32. This
candidate is rejected. It adds two failing frames and changes a high-margin
top-1 prediction. The CPU result is separate from the earlier **30/32 Neuron
device result**. Neither establishes a qualified model or a new speedup.

The targeted follow-up also completed: changing only layer 0 during prefill,
or during both prefill and decode, passes 27/32 in each case against the exact
29/32 strict CPU control. Both candidates are rejected. Layer-0 prefill
promotion alone is sufficient to reproduce the high-margin frame-1 regression
in this workload; promotion of later layers is not necessary for that failure.

## Earlier dispatch observer failure

Five fresh CPU reference runs show that entering the current dispatch observer
changes MiniCPM5's saved prefill logits. The unobserved run, historical hooks and
noop wrappers reproduce all 32 original endpoints each. Dispatch-only and full
observation reproduce 24/32 each, and match each other in all 32 frames. This
diagnostic identifies an observation-method problem that must be resolved before
using the captured intermediates to diagnose the Neuron implementation.

**This is not a speedup result or a qualified MiniCPM5 Neuron baseline.**
No candidate from this experiment is enabled in the plugin.

## Setup and results

The reference uses the official MiniCPM5-2B weights with 2,516,756,480 parameters,
BF16, eager attention and two CPU threads on the Trn1 host. The pinned reference
implementation is `LlamaForCausalLM` from Transformers revision
`f97400052a43d642bbc6e9975e2397e3ae6a6b52`. Loading reported no missing,
unexpected or mismatched parameters. The run uses eight original fixtures,
one prefill and three teacher-forced decode steps per fixture. Each arm starts
in a fresh process and maintains its own cache history; every arm uses the
same original saved teacher tokens.

| Arm | Exact original endpoint matches | Prefill matches | Decode matches |
|---|---:|---:|---:|
| Unobserved | 32/32 | 8/8 | 24/24 |
| Historical 213 hooks | 32/32 | 8/8 | 24/24 |
| Noop wrappers | 32/32 | 8/8 | 24/24 |
| Dispatch mode only | 24/32 | 0/8 | 24/24 |
| Full observer | 24/32 | 0/8 | 24/24 |

All ten arm-pair comparisons were retained. Pairs within the first three arms
match in 32/32 frames; each of those arms versus either dispatch arm matches
in 24/32. Dispatch-only versus full observer matches in 32/32. The eight
different original frame indices are `0, 1, 2, 3, 16, 17, 18, 19`, all step 0.

Equality covers the complete 130,560-entry output vocabulary, not just the
selected token or decoded text. Each of the 160 frames retains four witnesses:
actual BF16, actual converted to FP32, original saved FP32, and its reversibly
derived BF16 representation. The original exact gate was not relaxed. Finite
differences did not truncate the matrix or remove failing prefills.

The independent terminal audit verified all 1,552 raw files, totaling
314,961,320 logical bytes, and all four witnesses for every frame. All five
workers completed; the controller and external launcher completed and their
groups drained. A separate root-observed exit receipt establishes the final
post-auditor's actual exit code 0; that auditor did not certify its own future
exit. Successful diagnostic execution does not turn its unequal outputs into
a correctness pass.

## What this establishes

For these inputs, the changed endpoints are associated with entering dispatch
instrumentation. The extra work of the full observer adds no further endpoint
difference relative to dispatch-only. Endpoint equality does not prove that
all intermediate states agree, identify a causal operator, explain the
framework's dispatch behavior, or establish universal observer invariance.

In particular, three teacher-forced decode steps do not establish long-decode
stability. This experiment neither compares Neuron throughput nor proves that
the port preserves native model quality.

## Completed Python trace validation

A fresh five-arm Python trace run now reproduces **160/160 original endpoints**.
Each arm matches all eight prefill and 24 decode frames, and all ten arm pairs
match in 32/32 frames. This resolves the observation-method blocker for these
native reference inputs; it does not establish SDK observer invariance or a
qualified Neuron model.

| Arm | Exact original endpoint matches |
|---|---:|
| Unobserved | 32/32 |
| Historical 213 hooks | 32/32 |
| Noop wrappers | 32/32 |
| Noop Python trace | 32/32 |
| Full Python trace capture | 32/32 |

The trace reads the actual scaled attention local immediately after its
assignment and before masking. It preserves the original function, returned
objects and complete-vocabulary endpoint gate. The source AST establishes
Python multiplication; no aten operator callback is claimed, and the scaled
tensor is not recomputed. The noop and full trace arms independently observed
32 authenticated scale boundaries each; only full trace copied the tensors.

An earlier attempt completed the first three arms (96 exact frames), then
stopped before the first noop-trace frame during Python code authentication.
Compiling an isolated function had omitted its module import context under
Python 3.12. The successor compiles the entire authenticated module without
executing it, selects the original function code, and retains the bytecode,
qualified-scope and line-table checks. The failed attempt remains separate;
its partial outputs are not included in the new run.

The independent post-audit re-read and hashed all **1,552 raw files**
(**314,961,320 logical bytes**), verified all four witnesses for all 160 frames,
and reproduced the complete arm and pairwise results. Root observed the actual
launcher and post-auditor exits as code 0, verified all six planned service
groups drained, and appended the run to the complete predecessor catalog.
No tolerance, model configuration, teacher tokens or final qualification gate
was changed. This is a CPU reference diagnostic and adds no speedup ratio.

- [Complete trace results and pairwise comparisons](evidence/minicpm5-python-trace-v3-20261107/completion.json)
- [Independent raw-file audit](evidence/minicpm5-python-trace-v3-20261107/root-audit.json)
- [Actual post-auditor exit](evidence/minicpm5-python-trace-v3-20261107/post-auditor-exit.json)
- [Root finalization summary](evidence/minicpm5-python-trace-v3-20261107/root-finalization-summary.json)
- [Export manifest and original record hashes](evidence/minicpm5-python-trace-v3-20261107/manifest.json)

The finalization summary omits the repeated predecessor-drain array; its full
local source hash is retained in the manifest. Publication copies only saved
metadata and does not repeat the raw-payload audit.

## Completed 88-context acquisition

All 88 contexts reproduced their respective original endpoints: 32 native
reference frames, 28 CPU controls using device-generated history, and 28 CPU
contexts using actual history generated by the preceding CPU calls. Both SDK
ranks, the original 213 taps and all 84 before/after state records were retained.
The capture observed 112 scale events. Independent raw audit verified all 3,000
payload files, totaling 522,660,008 bytes. The launcher, post-auditor and final
metadata finalizer have recorded actual exit code 0.

Reproducing each path's original endpoint validates this capture method on these
inputs; it does not mean the native and SDK model endpoints agree.

## Saved-data comparison results

The first five boundaries contain 960 unique comparison rows. The three RoPE
boundaries contain 1,152 rows across six component labels. Both cover all 32
frames, three arm pairs and two ranks, with no missing rows or nonfinite values.
Independent metadata reviews checked the CSVs, summaries, source references and
recorded actual exits. All positions in the selected input sequence were compared.

The table reports native versus **one** CPU arm. The other CPU arm produces the
same comparison results, so its repeated differences are not additional evidence.

| Component | Exact frame/rank rows | Different elements | Maximum absolute difference |
|---|---:|---:|---:|
| Positions | 64/64 | 0 | 0 |
| Normalized input | 64/64 | 0 | 0 |
| Q projection | 43/64 | 26 / 2,971,648 | 0.0009765625 |
| K projection | 64/64 | 0 | 0 |
| V projection | 63/64 | 1 / 371,456 | 0.00006103515625 |
| Q entering RoPE | 43/64 | 26 / 2,971,648 | 0.0009765625 |
| K entering RoPE | 64/64 | 0 | 0 |
| RoPE cosine | 64/64 | 0 | 0 |
| RoPE sine | 64/64 | 0 | 0 |
| Q after RoPE | 51/64 | 15 / 2,971,648 | 0.0009765625 |
| K after RoPE | 64/64 | 0 | 0 |

The two CPU arms agree exactly in all 320 first-five rows and all 384 RoPE rows.
Q projection differs in 19 frames; rotated Q differs in 12. The only V projection
difference is frame 7, rank 1, during decode. RoPE input Q/K results reproduce the
earlier projection comparison under the corresponding layout mapping. Shared
native cosine/sine tensors across rank comparisons are not independent samples.
Fewer differing Q elements after RoPE does not prove numerical improvement or
identify the cause of the full-model output discrepancy.

- [Acquisition finalization summary](evidence/minicpm5-original88-20261107/finalization-summary.json)
- [First-five comparisons](evidence/minicpm5-original88-20261107/first-five.csv)
- [First-five independent review](evidence/minicpm5-original88-20261107/first-five-independent-review.json)
- [RoPE comparisons](evidence/minicpm5-original88-20261107/rope.csv)
- [RoPE independent review](evidence/minicpm5-original88-20261107/rope-independent-review.json)
- [Export manifest, source hashes and summary omissions](evidence/minicpm5-original88-20261107/manifest.json)

These exports reuse audited metadata. They do not include weights or raw tensor
payloads, and publication did not rerun model inference or the raw audit.

## Actual decode-mask finding

A bounded saved-data inspection completed normally and retained all 120
observations: 24 native decode frames and 96 CPU frame/rank/arm mappings.
It reads only captured masks and positions, without recomputing attention.
The 96 unique mask/position payload files total 114,066 recorded bytes.

For every CPU mapping, the actual prior mask permits indices `0..p`, where `p`
is the current token position. The authenticated ordinary SDK route separately
processes one unmasked current K/V. The historical branch should permit
`0..p-1`. Thus all 96 CPU mappings fail the current-position-exactly-once gate;
all 144 cross-arm comparison mappings remain blocked. Native joined masks
permit `0..p` as expected in all 24 frames. No later allowed index, missing
historical index, unsupported mask value or position mismatch was observed.

For example, frame 4 uses position 24: the CPU prior branch permits 25 cache
positions rather than 24, then processes the active current token separately.
The extra cached value at address 24 may be stale or zero; this does **not**
establish that identical current K/V values were counted twice.

Source tracing places the inclusive `j <= p` construction in our offline
baseline's numeric, cold and long-decode tests. The matched/history captures
reuse those inputs. The installed ordinary text-serving loader instead passes
`attention_mask=None`; SDK inference constructs `j < p` for one-token decode.
The observed harness error therefore does not establish the same error in a
running vLLM-Neuron service or an upstream SDK defect. An earlier November 2
audit had already documented this contract issue in other validation harnesses;
the present work establishes its separate MiniCPM5 input lineage.

The original 32-frame result contains one prefill failure and 14 decode
failures. A decode-only change cannot resolve the retained prefill failure
(frame 1, fixture 7, KL 0.1016956 versus the unchanged 0.02 limit). The bounded
corrected CPU replay below measures four first-decode frames; no speed
measurement has been performed for this finding.

- [Decode-mask observation summary](evidence/minicpm5-decode-mask-20261107/summary.json)
- [Independent result review](evidence/minicpm5-decode-mask-20261107/independent-review.json)
- [Harness and installed service source comparison](evidence/minicpm5-decode-mask-20261107/source-scope.json)
- [Historical precedent](evidence/minicpm5-decode-mask-20261107/historical-precedent.json)
- [Export manifest and original source hashes](evidence/minicpm5-decode-mask-20261107/manifest.json)

## Completed same-width mask intervention

A fresh TP2 SDK CPU replay has now measured the effect of correcting this
harness input. Both ranks reproduced the original inclusive control exactly:
the complete output vocabulary, all 213 saved taps, all 84 post-state hashes
and unchanged input states. Only after both controls passed did the strict
arm run. Each arm executed one full 42-layer decode forward.

Both arms imported the same original device-before history and retained bucket
width 2048, official weights and teacher tokens. The strict arm changed exactly
four mask cells from 1 to 0 at the four physical current positions; all other
arguments and layouts were preserved. This is a CPU execution of the SDK model,
not a new execution on Neuron devices.

**Under the unchanged per-frame numeric criteria, pass count increased from
1/4 to 4/4.** The two ranks have identical full-output digests and metrics;
there are four unique frames, not eight independent examples. Thresholds remain
cosine ≥ 0.995, reference-to-output KL ≤ 0.02, maximum absolute logit error ≤ 2,
and matching top token when the reference margin exceeds 0.5.

| Frame / fixture | Cosine: control → strict | Max absolute error: control → strict | KL: control → strict | Per-frame result |
|---|---:|---:|---:|---|
| 4 / 0 | 0.970919 → 0.999642 | 3.031250 → 0.375000 | 2.43375744e-05 → 6.63042556e-07 | Fail → Pass |
| 5 / 7 | 0.994647 → 0.996950 | 1.187500 → 0.812500 | 1.4395539e-07 → 1.11796794e-07 | Fail → Pass |
| 6 / 2 | 0.968678 → 0.997953 | 4.117188 → 0.882812 | 0.00169280497 → 6.33683521e-05 | Fail → Pass |
| 7 / 5 | 0.998516 → 0.999686 | 0.609375 → 0.375000 | 0.00167035998 → 0.00167199771 | Pass → Pass |

All four top-token predictions match the reference in both arms. Frame 7's
KL increases slightly, while remaining below the threshold; not every metric
improves. The first changed recorded tap is layer 0 attention output. Each
rank has 211/213 changed taps and 82/84 changed post-state hashes (indices
2 through 83). These are intervention effects, not failed reproduction
requirements for the strict arm, and they do not identify a single causal layer.

The controlled intervention establishes that the extra prior-mask cell affects
these CPU outputs and explains the three local pass/fail changes. It does not
establish a corrected complete history trajectory, explain the original
prefill failure, or qualify MiniCPM5 on Neuron. Both arms still start from the
original device-generated history. No latency, throughput or speedup was measured.

The two rank processes, coordinator service and outer launcher exited with
code 0; root separately observed the live outer process and its actual exit,
and verified the service group drained. Independent review covers saved JSON
metadata and the source/exit chain, not an independent payload recomputation.

- [Four-frame result summary](evidence/minicpm5-call4-mask-20261107/root-result-summary.json)
- [Independent metadata review](evidence/minicpm5-call4-mask-20261107/independent-result-review.json)
- [Actual root-observed exit](evidence/minicpm5-call4-mask-20261107/root-exit.json)
- [Source review and export provenance](evidence/minicpm5-call4-mask-20261107/manifest.json)

## Completed compiled-device trajectory

The same-width mask correction has now been tested on the existing compiled
Neuron artifact, using the original official weights and two Neuron cores.
Each arm completes all 14 graph calls and 32 frames: eight prefill frames and
three teacher-forced decode steps for each fixture. Official model loading
and default warmup occur separately from these 28 replay calls.

The complete inclusive control first reproduces every original output and all
168 post-cache hashes at each of its 14 calls. Each arm imports the same
initial cache exactly once. Subsequent calls consume that arm's actual produced
cache, including the transition between request groups: bookkeeping resets
preserve all cache values. The strict trajectory changes 24 current-position
mask cells across six decode calls, with the same physical mask widths and
compiled graph routes. Prefill inputs are unchanged.

| Original criteria | Inclusive control | Strict prior mask |
|---|---:|---:|
| All numeric criteria passed | 17/32 | **30/32** |
| Prefill passed | 7/8 | 7/8 |
| Decode passed | 10/24 | **23/24** |
| Highest-logit token matches reference | 31/32 | **32/32** |

Thirteen formerly failing decode frames now pass, and no previously passing
frame becomes a failure. Only the six decode calls have changed output digests;
all eight prefill outputs remain exactly the same between the two arms.
These pass counts use the original cosine, KL and maximum-error thresholds,
plus the original high-margin top-token rule. The aggregate top-token match
floor also remains 0.9.

Two failures remain, both in fixture 7:

| Frame / phase | Control KL | Strict KL | Required maximum |
|---|---:|---:|---:|
| 1 / prefill | 0.101695620 | 0.101695620 | 0.02 |
| 13 / third decode step | 0.072239228 | 0.042208157 | 0.02 |

Frame 13 now passes cosine and maximum-error checks, but its KL remains above
the limit. Matching the highest-logit token does not imply that the complete
output probability distribution passes. Both arms therefore retain an overall
numeric failure. The shared fixture is a lead for further diagnosis, not proof
that its prefill and decode errors have the same cause.

This experiment establishes an improvement on the compiled device under the
explicit same-width mask contract and each trajectory's own cache history.
It does not validate ordinary serving's inferred-mask width, long generation,
quality or performance. **No new speedup ratio is reported.** The harness is
the source of the corrected mask; this result does not establish an equivalent
upstream SDK or ordinary vLLM-Neuron service defect.

An earlier attempt loaded and warmed the device model, then stopped during the
first control call's verification because its input list omitted the old
post-cache manifests. It produced no completed call report and never entered
the strict arm. The successor added the 14 manifests and their 2,352 recorded
cache identities, retained the model worker unchanged, and required the failed
process and service group to drain. That failure is preserved separately and
is not a numeric or post-cache mismatch result. The successful successor and
its device service both have root-observed actual exit code 0 and an empty group.

- [Complete 32-frame comparison and remaining failures](evidence/minicpm5-device32-mask-20261107/root-result-summary.json)
- [All per-frame metrics as CSV](evidence/minicpm5-device32-mask-20261107/results.csv)
- [Independent result metadata review](evidence/minicpm5-device32-mask-20261107/independent-result-review.json)
- [Successful actual exit](evidence/minicpm5-device32-mask-20261107/root-exit-r2.json)
- [Preserved failed attempt](evidence/minicpm5-device32-mask-20261107/root-exit-r1.json)
- [Export source hashes](evidence/minicpm5-device32-mask-20261107/manifest.json)

## Next validation

The immediate next diagnostic target is fixture 7: retain the unchanged
prefill KL failure and the remaining third-decode KL failure, and compare their
attention/cache boundaries against the original native reference. The earlier
four-frame CPU contrast and the full device trajectory are distinct experiments.
SDK-inferred mask width remains a separate follow-up because it may change
bucket routing. Ten later attention boundaries have not yet received the full
saved-data comparisons.

Active V values, prior cache, current writes and prefill masks still need
comparison. Expected cache manifests must
not be treated as captured numeric post-cache tensors. Decode must account for
logical prior positions and the current token exactly once; actual captured
masks must establish this before any canonical-key comparison.

Full-model numeric checks, all 84 cold-state frames, long decode, request-slot
reuse, inactive-slot behavior, the 96-case quality suite and paired serving
measurements remain required before model or performance qualification.

## Evidence

- [All arm and pairwise results](evidence/minicpm5-m2-20261107/metadata-analysis.json)
- [Independent terminal audit](evidence/minicpm5-m2-20261107/root-audit.json)
- [Actual post-auditor exit](evidence/minicpm5-m2-20261107/post-auditor-exit.json)
- [External launch and exit ledger](evidence/minicpm5-m2-20261107/external-invocations.json)
- [Publication manifest](evidence/minicpm5-m2-20261107/manifest.json)

These are unchanged copies of the local result metadata, with source hashes
in the manifest. Raw tensors and model weights remain in the experiment store;
publication does not perform another raw-data audit. The archived metadata
analyzer refers to that original experiment tree and is not a standalone
inference runner. Earlier failed attempts remain separate historical evidence.
No model family is declared fully optimized.

## Layer-0 prefill attention comparison

Saved-data analysis now covers all eight original prefill fixtures: 40 native/CPU
capture contexts, 576 numeric comparisons, 40 mask observations, 48 pair gates
and 200 physical score/probability observations. The serial analyzer completed
with exit code 0 and authenticated 560 saved payloads (124,556,034 bytes).
No model was rerun and no attention values were reconstructed.

Every real-query mask is causal and excludes padding. For the **captured final
prediction query only**, unscaled QK scores, scaled scores, masked scores and
both FP32 and BF16 probabilities agree bitwise for all eight fixtures and both
ranks. All saved padding probabilities are zero. These results do not cover
scores or probabilities for the earlier query rows, which were not captured.

The table compares native reference with one CPU arm. K/V, context and output
cover every real input token. The output after projection is replicated on both
ranks, so its element counts are deduplicated to eight complete output tensors.

| Saved layer-0 component | Exact mapped rows | Different elements | Maximum absolute difference |
|---|---:|---:|---:|
| Active V, effective K/V, returned current K/V (each) | 16/16 | 0 / 365,312 | 0 |
| Unscaled/scaled/masked score, FP32/BF16 probability (each, final query only) | 16/16 | 0 / 22,832 | 0 |
| Context entering output projection | 14/16 | 28 / 2,922,496 | 0.00048828125 |
| Complete attention output, deduplicated across ranks | 0/8 | 932,494 / 2,922,496 | 0.0078125 |

Both CPU history arms agree exactly in all 192 comparison rows. Their identical
findings are not independent new samples.

All 28 context differences occur in frame 1 / fixture 7, the 1,231-token prefill:
17 elements on rank 0 and 11 on rank 1. At its final query, rank 0 differs in only
one context element by 1.862645149230957e-9; rank 1 is exact. Its final 2,048-wide
attention output differs in 695 elements, maximum 0.00048828125. Replication on
rank 1 does not double that count.

For the other seven prefills, the complete real-token context agrees but the
output projection does not. This localizes an observed numerical difference to
the projection/collective path on those inputs. It **does not establish that this
difference causes the failing final-model KL**, and the failing long prefill has
small context differences as well. The current captures do not separate local
projection rounding from collective rounding.

Source inspection gives a concrete next experiment: HF uses one 2048-wide output
projection; SDK TP2 computes two 1024-wide partial projections and then sums them.
The configured SDK reduction dtype falls back to BF16. A controlled replay using
the same saved context and official weights can isolate the projection and
rounding choices. Changing only collective dtype cannot recover precision already
lost in a local BF16 projection and is not yet an accepted fix.

The saved prefill result neither clears the remaining device frame 1/13 KL
failures nor supplies a speedup ratio. Old decode captures still use the inclusive
mask and are unsuitable for that aligned comparison. Full CPU post-cache backing
was not captured. Cold-cache, long-decode, request-slot reuse, inactive-slot,
quality96 and paired-serving qualification remain required.

- [Per-comparison CSV](evidence/minicpm5-prefill-attention-20261107/comparisons.csv)
- [Root summary with deduplicated output counts](evidence/minicpm5-prefill-attention-20261107/root-summary.json)
- [Independent result review](evidence/minicpm5-prefill-attention-20261107/independent-result-review.json)
- [Source mapping and limits](evidence/minicpm5-prefill-attention-20261107/source-mapping.json)
- [Actual analyzer exit](evidence/minicpm5-prefill-attention-20261107/actual-exit.json)
- [Export manifest](evidence/minicpm5-prefill-attention-20261107/manifest.json)

## Controlled output-projection arithmetic replay

A saved-layer CPU replay completed with actual exit code 0. It reads only the
8 MiB official layer-0 `o_proj` matrix and existing context/output captures.
All **24 complete-shape controls** reproduce their saved output bits: eight
native contexts at length L and two CPU history arms at their original padded
bucket E. Native controls use two CPU threads; SDK controls use one. The replay
runs in the SDK Python environment and uses standalone addition; it does not
rerun the original native interpreter or a Gloo collective.

The six formulas then use the same final-query input within each case. This
reduces the input to one row, and that shape change is checked rather than
assumed harmless. **19/24 one-row baseline controls match; five do not.**

| One-row control mismatch | Different output elements | Maximum absolute difference |
|---|---:|---:|
| Frame 1 / fixture 7, native input | 2 / 2,048 | 0.000003814697265625 |
| Frame 16 / fixture 4, native input | 3 / 2,048 | 0.00000095367431640625 |
| Frame 16 / fixture 4, CPU device-history input | 1 / 2,048 | 0.00000762939453125 |
| Frame 16 / fixture 4, CPU own-history input | 1 / 2,048 | 0.00000762939453125 |
| Frame 18 / fixture 6, native input | 2 / 2,048 | 0.000030517578125 |

These five remain in the result matrix as local arithmetic trials and are
excluded from interpretation as exact reproductions of the original execution.
The two CPU history arms give identical results; 24 labeled cases represent
eight fixtures, not 24 independent prompts. There are only nine distinct input
byte digests. Equal bytes across source arms do not guarantee the same tensor
layout: native and CPU one-row views retain different leading strides in some
cases. Their results must not be treated as a controlled stride experiment.

| Controlled comparison | Same final BF16 output | What it establishes on these inputs |
|---|---:|---|
| Same BF16 partial outputs: BF16 sum versus FP32 sum then BF16 | 24/24 | Increasing only sum precision has no observed effect. |
| Same FP32 partial outputs: retain FP32 until sum versus round each partial to BF16 first | 0/24 | Intermediate BF16 quantization changes the result; 16,731 differing elements across labeled cases. |
| FP32 full projection versus FP32 split projection, final BF16 | 24/24 | These two FP32 arithmetic forms agree after final conversion in this trial. |
| SDK BF16 partials versus FP32 partials rounded individually to BF16, then summed | 21/24 | The underlying local kernels still differ slightly in three frame-16 cases; this is not a pure dtype comparison. |

For all **19 cases passing both control gates**, FP32 split projection with only
one final BF16 conversion matches the same-input HF BF16 one-row result exactly.
The BF16 split result differs in all 19; rounding FP32 partials individually back
to BF16 reproduces that BF16 split result in all 19. This supports investigating
the timing of local-output rounding, rather than treating a wider collective
alone as the candidate correction.

In the failing fixture-7 prefill, both CPU-input one-row controls reproduce the
saved SDK output exactly. Keeping local partials and their sum in FP32 changes
694 elements relative to that saved SDK output and matches the same-input HF
one-row projection. The native-input one-row control for this fixture fails by
two elements and is explicitly excluded. No conclusion here says the complete
native trajectory or final logits now match.

This is a layer-0 CPU arithmetic experiment, **not a full-model repair, Neuron
accumulator test or speed measurement**. The device trajectory remains 30/32
under the unchanged numeric gates. A next candidate needs complete original
shapes, a strict-mask CPU baseline, actual TP2 model execution, and the original
cold/long/reuse/inactive/quality96/paired-serving requirements.

The first metadata integration attempt exposed a preparation error: the worker
compared the original science plan with its finalized successor as whole files.
Only output namespace, source closure and capture contexts differ. Version 2
validates the exact previously accepted transformation and retains the real
finalized plan; all scientific fields remain unchanged. No weights or operators
were read/run in that failed metadata attempt. The mathematical and bounded
weight-reading functions are identical between worker versions.

- [All 144 formula results](evidence/minicpm5-oproj-intervention-20261107/formulas.csv)
- [Complete replay report](evidence/minicpm5-oproj-intervention-20261107/report.json)
- [Root summary and five preserved mismatches](evidence/minicpm5-oproj-intervention-20261107/root-summary.json)
- [Independent result review](evidence/minicpm5-oproj-intervention-20261107/independent-result-review.json)
- [Actual exit](evidence/minicpm5-oproj-intervention-20261107/actual-exit.json)
- [Export manifest](evidence/minicpm5-oproj-intervention-20261107/manifest.json)

## Full-shape output-projection intervention

All six formulas have now been evaluated on each complete original input shape,
including the CPU padding width, before extracting real-token or last-query
metrics. The serial CPU replay exited successfully. All 24 saved-output controls
and all 24 per-case controls reproduce exactly; the 144 formula outcomes cover
eight fixtures and three input labels. Both CPU history labels have identical
results and are counted once below. The five historical single-row control
failures remain recorded; this new trial does not retroactively clear them.

| Comparison with saved native layer-0 output, one CPU history arm | BF16 split projection | FP32 local products and sum, final BF16 |
|---|---:|---:|
| Different elements across all real tokens / 2,922,496 | 932,494 | 4,041 |
| Maximum absolute difference, all real tokens | 0.0078125 | 0.00390625 |
| Different elements at eight final queries / 16,384 | 5,580 | 5 |
| Maximum absolute difference, final queries | 0.0078125 | 0.000003814697265625 |
| Final queries exactly matching saved native | 0/8 | 4/8 |

All eight fixtures improve in different-element count, maximum error and mean
error, for both real-token and last-query domains. Fixture 7 retains most of
the difference: 3,868 real-token elements and one final-query element. Comparing
against a full BF16 projection of the **same CPU input**, rather than saved
native, reduces the real-token difference from 932,238 to 1,087 elements.
The saved-native comparison includes pre-existing context and geometry
differences and is not by itself a controlled operator comparison.

The controlled full-shape comparisons establish useful limits:

- Widening only the sum of the same BF16 partials still changes nothing: 24/24
  full outputs match.
- Inserting intermediate BF16 rounding into the same FP32 partials changes all
  24 full outputs.
- FP32 full projection and FP32 split projection **differ in all 24 full outputs**;
  only 18/24 final queries agree. The earlier single-row equality does not
  generalize to complete shapes.
- BF16 local kernels and FP32 local kernels followed by local BF16 rounding also
  differ in all 24 full outputs; 22/24 final queries agree. Local kernel/order
  effects remain separate from the controlled rounding contrast.

This motivated a full-model CPU experiment targeting only the 42 attention
output-projection instances: keep official BF16 parameters, compute local
products and actual TP2 reduction in FP32, then cast once to BF16. Its execution
order was original inclusive-mask trajectory reproduction, strict-mask baseline,
an independent exact baseline repeat, then a strict-mask candidate. Each arm
must carry its own actual caches through all 14 calls / 32 frames. The original
HF logits thresholds and every downstream qualification gate remain unchanged.

No full model, Gloo collective or Neuron kernel ran in this saved-layer trial.
Replay under the SDK interpreter is not proof of native-runtime equivalence.
The weight check covers the authenticated 8 MiB projection interval and inherited
checkpoint identity, not a fresh whole-checkpoint hash. The remaining device
KL failures and its 30/32 result are unchanged. There is no new speedup ratio.

- [All formula metrics](evidence/minicpm5-oproj-fullshape-20261107/formulas.csv)
- [Complete report](evidence/minicpm5-oproj-fullshape-20261107/report.json)
- [Deduplicated root summary](evidence/minicpm5-oproj-fullshape-20261107/root-summary.json)
- [Independent result review](evidence/minicpm5-oproj-fullshape-20261107/independent-result-review.json)
- [Actual exit record](evidence/minicpm5-oproj-fullshape-20261107/actual-exit.json)
- [Export manifest](evidence/minicpm5-oproj-fullshape-20261107/manifest.json)

## Full-model CPU output-projection trial: candidate rejected

The four-arm CPU TP2/Gloo run completed with exit code 0. Every arm executes
the complete original 14-call / 32-frame trajectory. It imports the original
call-0 cache seed once, then carries its own 84 cache tensors per rank across
prefill/decode transitions and the group reset at call 7. No baseline cache is
injected into the candidate midway through the run. The two replicated ranks
are counted once in the table.

| CPU arm | All per-frame numeric gates | Prefill | Decode | Top-1 matches |
|---|---:|---:|---:|---:|
| Original inclusive-mask control | 16/32 | 7/8 | 9/24 | 31/32 |
| Strict-mask baseline S0 | 29/32 | 7/8 | 22/24 | 32/32 |
| Independent strict-mask repeat S1 | 29/32 | 7/8 | 22/24 | 32/32 |
| Strict mask + all 42 FP32 output projections | **27/32** | **6/8** | **21/24** | **31/32** |

The original control reproduces all 14 saved complete outputs, the original
213 intermediate signatures, and all cache-state signatures on both ranks.
S1 exactly reproduces S0's complete output and intermediate/cache metadata
through all 14 calls. These are reproducibility checks; the HF-reference
numeric pass counts are a separate comparison.

Only the 42 exact `layers.N.self_attn.o_proj.o_proj` instance forwards change
in the candidate. Original BF16 weights and inputs are cast to FP32 for the
local projection; the actual TP2 reduction stays in FP32, followed by one
BF16 cast. All official output-projection weight column shards are checked,
and the original parameter objects and bytes are unchanged at cleanup.
No MLP, language-model head, global dtype or configuration change is included.

The candidate rescues **zero** previously failing frames and introduces
failures at frames **5 and 16**. All frame indices below are zero-based.

| Frame / fixture / phase | Strict S0 KL | Candidate KL | Candidate failure |
|---|---:|---:|---|
| 1 / 7 / prefill | 0.101270 | 0.197038 | KL; high-margin top-1 changes from reference 1062 to 15182 |
| 5 / 7 / decode step 1 | 0.0000000819 | 0.0000002351 | New cosine failure: 0.993273 |
| 9 / 7 / decode step 2 | 0.028122 | 0.021217 | KL remains above 0.02 |
| 13 / 7 / decode step 3 | 0.048726 | 0.038259 | KL remains above 0.02 |
| 16 / 4 / prefill | 0.018799 | 0.021150 | New cosine (0.994612) and KL failures |

The unchanged thresholds are cosine at least 0.995, KL at most 0.02,
maximum absolute error at most 2, aggregate top-1 agreement at least 90%,
and exact top-1 agreement whenever the reference margin exceeds 0.5.
Frame 1 has reference margin 1.125, so the candidate's aggregate 31/32
top-1 agreement does not excuse that failure. Strict S0 itself still fails
frames 1, 9 and 13. Correcting the inclusive mask improves the overall pass
count, but per-frame KL improves in 17 frames, ties in 8, and worsens in 7;
it does not improve every metric at every frame.

The saved layer-0 projection improvement therefore did **not** generalize to
this full-model, all-42-layer intervention. This does not invalidate the local
rounding observation or identify the cause of the remaining errors. The
candidate is not enabled in the plugin and will not be compiled for Neuron
on the strength of this result. New intermediate/cache payloads were not saved
in this run; their hashes establish equality or difference, not numerical
error magnitude or a causal operator.

This run exercised the full model on CPU, including actual Gloo collectives.
It did not compile or run a new Neuron candidate, and it measured no serving
speedup. Cold-cache, long-decode, slot reuse/inactive-slot, quality96 and paired
serving requirements remain outstanding. The earlier device result remains
30/32; the CPU strict baseline is 29/32. All 15 model families remain in scope.

- [Per-frame metrics](evidence/minicpm5-oproj-cpu-fullmodel-20261107/frames.csv)
- [Root summary](evidence/minicpm5-oproj-cpu-fullmodel-20261107/root-summary.json)
- [Independent metadata/source review](evidence/minicpm5-oproj-cpu-fullmodel-20261107/independent-result-review.json)
- [Actual exit record](evidence/minicpm5-oproj-cpu-fullmodel-20261107/actual-exit.json)
- [Worker source](evidence/minicpm5-oproj-cpu-fullmodel-20261107/worker.py)
- [Export manifest](evidence/minicpm5-oproj-cpu-fullmodel-20261107/manifest.json)

## Layer-0 and phase-restricted follow-up

The targeted three-arm CPU TP2 run completed with exit code 0. The strict
control exactly reproduces the previous S0 through all 14 calls / 32 frames
on both ranks. Each arm retains its own actual cache history. Official weight
checks cover all 42 layers; only the layer-0 attention output-projection
instance is changed when its selected phase is active.

| CPU arm | All per-frame numeric gates | Prefill | Decode | Top-1 matches |
|---|---:|---:|---:|---:|
| Exact strict S0 control | 29/32 | 7/8 | 22/24 | 32/32 |
| Layer-0 FP32 projection during prefill only | **27/32** | **5/8** | **22/24** | **30/32** |
| Layer-0 FP32 projection during both phases | **27/32** | **5/8** | **22/24** | **30/32** |

Both candidates pass the formerly failing frame 9, introduce failures at
frames 5, 16 and 17, and still fail frames 1 and 13. Equal pass counts do not
mean equal outputs. The two candidates exactly match through their first four
prefill calls and enter the first decode call with equal cache states; their
complete logits differ at that decode call. Subsequent differences include
the effects of their separately accumulated cache histories.

| Zero-based frame | Strict control | Prefill-only candidate | Both-phases candidate |
|---|---:|---:|---:|
| 1: KL | 0.101270 | 0.282340 | 0.282340 |
| 5: cosine | 0.998754 | 0.988312 | 0.986136 |
| 9: KL | 0.028122 | 0.018686 | 0.017490 |
| 13: KL | 0.048726 | 0.049063 | 0.068797 |
| 16: KL | 0.018799 | 0.025319 | 0.025319 |
| 17: KL | 0.010951 | 0.028214 | 0.028214 |

Frame 1 changes top-1 from reference 1062 to 15182 with reference margin
1.125 in both candidates. Frame 16 also changes top-1, with a smaller reference
margin of 0.125; its KL independently exceeds the unchanged 0.02 threshold.
The original high-margin, aggregate top-1, cosine, KL and max-error gates all
remain in force. Neither candidate is enabled in the plugin or admitted for
Neuron compilation.

The targeted run saves four **prediction-query vectors**, at layer-0 attention,
layer-0 output, layer-41 attention and layer-41 output. These are finite BF16
bundles totaling 3 MiB across all three arms and both ranks. Decode rows follow
physical slot order: the first decode bundle maps to frames `[6, 5, 7, 4]`,
while the metric report retains original frame order `[4, 5, 6, 7]`. Explicit
mapping prevents comparing different requests by accident. Full-sequence
intermediates and complete cache payloads are not exported.

### Saved numerical comparison

A separate serial analysis completed with exit code 0. It authenticated and
read all 84 selected bundles plus the 32 original native prediction archives:
58,078,560 bytes in total. Both ranks' selected bytes agree and are counted
once. The report contains all 768 comparisons: six reference/arm pairings,
32 frames and four boundaries. Native archives are bound both to the original
HF32 metadata and to the original88 native endpoint receipts.

| Mean absolute difference from native, all 32 prediction vectors | Strict control | Layer-0 prefill only | Layer-0 both phases |
|---|---:|---:|---:|
| Layer-0 attention output | 0.000035009 | 0.000028721 | 0.000025579 |
| Layer-0 layer output | 0.000145600 | 0.000126837 | 0.000125623 |
| Layer-41 attention output | 0.0250380 | 0.0271865 | 0.0262389 |
| Layer-41 layer output | 0.0885306 | 0.0946100 | 0.0931349 |

For the both-phases candidate, the layer-0 attention prediction error improves
in **all 32 frames** by mean absolute difference. At the final layer output,
the same measure improves in 12 frames and worsens in 20. The prefill-only
candidate improves the first two layer-0 boundaries in all eight prefills
and leaves their 24 decode vectors unchanged, while final-layer error worsens
in 21 of 32 frames. These per-frame observations are more informative than
an aggregate local improvement alone.

Two examples make the limitation concrete:

- Frame 1: layer-0 attention differs from native in 695 elements under the
  control and only **one** element under either candidate. Its mean error
  falls from 0.000020505 to approximately 0.000000001863. Nevertheless,
  final-layer mean error rises from 0.118588 to 0.185853 and final-logit KL
  rises from 0.101270 to 0.282340, with a high-margin top-1 mismatch.
- Frame 17: the candidate's layer-0 attention output is **byte-exact** to
  the native prediction vector. Final-layer mean error still rises from
  0.124774 to 0.152546 and final-logit KL from 0.010951 to 0.028214.

Thus local agreement at the measured attention boundary does not guarantee
agreement after the rest of the model. This does not identify a unique
downstream operator or justify adding per-layer errors together. Native
unsharded execution and padded CPU TP2 execution use different geometry and
cache histories; the comparison is between corresponding semantic prediction
boundaries, not an isolated same-input operator experiment. Intermediate
errors are not scored using the final-logit thresholds.

The numerical replay finished before an independent live `/proc` observation
could be obtained. Its actual tool exit code 0, emitted start identity,
authenticated result and subsequent process/lock cleanup are recorded; a
separate live-process receipt is **not** claimed. The independent result
review checks source, metadata and report arithmetic, without a second raw
tensor recomputation.

No accepted optimization or speedup results from these trials. Device,
cold-cache, long-decode, slot reuse/inactive-slot, quality96 and paired-serving
requirements remain unchanged.

- [Endpoint metrics](evidence/minicpm5-layer0-phase-20261107/frames.csv)
- [Endpoint summary](evidence/minicpm5-layer0-phase-20261107/root-summary.json)
- [Independent full-model result review](evidence/minicpm5-layer0-phase-20261107/independent-result-review.json)
- [Selected-vector numerical rows](evidence/minicpm5-layer0-phase-20261107/selected-taps.csv)
- [Complete numerical report](evidence/minicpm5-layer0-phase-20261107/selected-taps.json)
- [Interpretation and per-frame contrasts](evidence/minicpm5-layer0-phase-20261107/numeric-interpretation.json)
- [Independent numerical report review](evidence/minicpm5-layer0-phase-20261107/numeric-independent-review.json)
- [Actual model exit](evidence/minicpm5-layer0-phase-20261107/actual-exit.json)
- [Actual numerical replay exit](evidence/minicpm5-layer0-phase-20261107/numeric-actual-exit.json)
- [Export manifest](evidence/minicpm5-layer0-phase-20261107/manifest.json)


## One-fixture MLP capture and replay

The next diagnostic completed with exit code 0: native capture, real CPU TP2
capture, and serial numerical replay all exited successfully. It tests only
frame 17 / fixture 1: a 28-token prefill, processed in the SDK's original
128-token bucket. The CPU arm retains the previously rejected layer-0
prefill-only FP32 attention output projection so that its exact saved
intermediate observations can be reproduced; it is a diagnostic control,
not an accepted optimization.

Each capture executes embedding and layer 0, then stops before later layers.
The native and both CPU ranks reproduce all six original prediction anchors:
embedding, input normalization, attention, post-attention normalization, MLP,
and layer output. Five complete MLP tensors are saved per role (input, gate,
up, SiLU-times-up product, output), totaling 8,077,312 bytes across the three
roles. Official BF16 gate/up/down weights and actual TP2 shards are checked.
The ordinary prefill route consumes no prior CPU KV history; this is not a
replay of the complete model's cache trajectory.

Before the intervention, replay reproduces all **15 complete tensors and their
layouts exactly**, including the captured real TP2 outputs. It then compares
three calculations on the **same CPU input, original [1,128,2048] layout,
official BF16 weights, library and one-thread setting**:

- **S:** the original SDK split gate/up and split down projections.
- **F:** complete BF16 gate/up and complete BF16 down projections, a new
  counterfactual reference on the same input.
- **D:** reuse S's actual gate/up/activation product; concatenate its two
  3,072-wide shards in official order and use one complete BF16 down projection.

The gate, up and activation product are byte-exact between S and F across all
128 rows. The output comparison is:

| Output comparison | Different elements, all 128 rows | Different elements, real 28 rows | Different elements, prediction row |
|---|---:|---:|---:|
| Original split S versus full F | 88,577 / 262,144 | 18,367 / 57,344 | 664 / 2,048 |
| Down-only change D versus full F | **0 / 262,144** | **0 / 57,344** | **0 / 2,048** |

S versus F has maximum absolute error 0.015625 across all rows, and
0.00048828125 on the prediction row (mean 0.00003656495). D matches F byte for
byte. Thus, **the split down-projection path is sufficient to explain this
local fixed-input MLP difference**. This does not separate kernel accumulation,
local BF16 rounding and reduction order, and does not establish that this
change fixes the final model's KL failure. Replay matches the captured Gloo
result exactly, but does not itself reexecute Gloo.

The new full inputs also expose a limit of the earlier prediction-vector
observations: native and CPU post-normalization inputs differ in **2 of 57,344**
elements over the 28 real tokens, despite their prediction row being identical.
Therefore F must not be described as the original native observation; its
input and 128-row geometry are held fixed to the CPU capture.

The first attempt failed before any forward call because a metadata comparison
helper was passed raw bytes. The immutable successor uses direct byte equality
at both raw sites and direct equality for the role set; both original Python
environments passed the focused regression checks. The failed attempt and its
zero-payload result are retained.

The root observed the second attempt's outer and native processes live and
observed the CPU coordinator running. Separate complete CPU-rank and replay
live snapshots were not obtained before completion. Parent acknowledgments,
all three successful stage exits, actual tool exit 0, all six named processes
gone, empty service groups and lock release were verified. No missing live
snapshot is claimed to exist.

An independent review verified the source, metadata, 15 complete controls,
18 historical prediction anchors and terminal states. It did not reread raw
tensors or independently recompute their numerical values.

This result is a local diagnostic, with no new full-model candidate,
Neuron compilation or speedup. The strict CPU 29/32 and prior device 30/32
results, all rejected precision candidates, and all original cold-cache,
long-decode, slot reuse/inactive-slot, quality96 and paired-serving gates remain.

- [Root interpretation](evidence/minicpm5-frame17-mlp-20261107/root-summary.json)
- [Full replay metrics](evidence/minicpm5-frame17-mlp-20261107/replay-result.json)
- [Independent result review](evidence/minicpm5-frame17-mlp-20261107/independent-result-review.json)
- [Actual successful exit](evidence/minicpm5-frame17-mlp-20261107/actual-exit.json)
- [Retained first failure](evidence/minicpm5-frame17-mlp-20261107/first-attempt-exit.json)
- [Export manifest](evidence/minicpm5-frame17-mlp-20261107/manifest.json)
