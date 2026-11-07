# MiniCPM5 observer and attention diagnostics — 2026-11-07

The replacement Python trace has completed the 88-context acquisition. Saved
data comparisons now cover the first five attention boundaries and three RoPE
boundaries. Inputs, K projection, RoPE coefficients and rotated K agree exactly
in the observed comparisons; Q and V have a small number of differing elements.
The cause of the full-model discrepancy remains unresolved. Details and
independent result reviews are below; the earlier observer failure is retained
as historical evidence.

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

## Next validation

Ten later attention boundaries remain. Next are active V values, prior cache,
current cache writes and actual mask semantics. Expected cache manifests must
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
