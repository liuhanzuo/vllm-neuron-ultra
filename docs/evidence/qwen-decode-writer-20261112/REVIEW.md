# Qwen3.5 decode cache writer candidate — isolated source delivery

This is a concrete patch candidate. It has **not** been applied to the running
vLLM-Neuron checkout, imported with native libraries, compiled, or run on a device.
The current independent-package sampler experiment remains unchanged.

## Changes in the patch

`text_core.py` gives both attention classes the original `store_slots` callable
by default. Only the four cache-write calls use that field. The original writer,
old-state gathers, convolution, delta rule, attention, scatter and projections
remain unchanged. A CPU `Qwen35TextPort` instance therefore keeps the original
writer without consulting any new environment variable.

`modeling.py` resolves the writer once at the end of `init_model`, after tensor
parallel sharding. It checks the parent identity copied in the Neuron config,
the trace worker's selected mode and helper/kernel hashes, and the original
TP2/B4/ctxB1/context4096/18-linear+6-full layout. Every prefill instance binds the
original writer. The decode candidate has no runtime fallback and never selects
a backend from `cache.device`; trace inputs can legitimately be CPU tensors.
The original KV prefix slice and untouched-tail concatenation, cache allocation,
48 states per rank and FP32 preservation code remain byte-identical.

New `cache_writer.py` is stdlib-only on import. The switch
`NEURON_QWEN35_CACHE_WRITER` accepts only `torch_where` (default) and
`nki_predicated_decode_v1`. The candidate rejects unsupported cache/update
shapes or precision and requires **actual int32** four-row sequence slots.
It flattens without payload conversion or forced contiguity, calls the package
relative kernel, and restores the original shape. Slots are passed directly;
there is no hidden int64-to-int32 cast in this candidate. Dynamic slots keep the
original -1/inactive and 0..3 mapping, with last writer winning duplicates.

New `slot_cache_nki_predicated.py` is byte-for-byte the previously reviewed
kernel. Both new modules are top-level package Python files, so the existing
`source_identity()` glob includes them without recursively widening the source
scope. There is no runtime import from the experiments directory.

`identity.py` adds a `cache_writer` identity field carrying mode, decode/prefill
scope, version, int32 slots, payload geometry/precision, TP2 and 96-state counts,
all six buckets, 12 graphs and helper/kernel hashes. Existing `same_identity`,
saved-config comparison and artifact admission reject old artifacts lacking the
field, cross-mode artifacts, or changed sources. Both arms use the same patched
source tree; even the default arm needs its own fresh artifacts.

## Separate writer experiment, still pending

The current 25-stage sampler runner deliberately measures baseline sampler 0
against candidate sampler 1 with one shared artifact. **It must not be reused
unchanged for this writer experiment.** The small stdlib `pair_contract.py`
provides future metadata checks for the two named writer modes, equal explicit
sampler values, identical model/config/source/environment identities and six
separate nonnested artifact/cache/workdir paths. It also rejects incomplete or
duplicated 12-graph inventories. It performs no filesystem freshness admission,
launch or scientific qualification; those remain the original root protocol's
responsibility in a separately reviewed writer runner.

Required order:

1. Finish and independently audit the current sampler package run.
2. Review this patch and apply it to an isolated source successor. Qualify the
   actual int32 entry and sliced-KV graph behavior below before a model claim.
3. Keep the same explicit CPU sampler setting on both writer arms (prefer `1`
   only after independent sampler package qualification). Keep weights,
   tokenizer, BF16/FP32, compiler flags, CPU/thread settings and requests fixed.
4. Review budget for **both** fresh 12-graph artifacts, each with separate cache
   and workdir. Validate every graph route, compiled file, actual exit and
   original 96 state bindings; shared old graphs cannot establish a writer pair.
5. Preserve all original gates: in-place raw-zero state reset, original numeric
   thresholds, all six cold buckets, 1,541-step long decode, slot reuse,
   partial/inactive resume, complete 203 vectors, original sampling/capture
   controls, quality96 and original 69/96 baseline reproduction, observer-free
   HTTP smoke/initial120/matched workloads, actual process exits and independent
   final history/source/input/artifact/descendant audit.

### Device coverage still missing

The completed primitive matrix used an int64 entry and full tensors on one core.
This delivery's mocks verify dispatch and shape/type guards only. They establish
neither tensor results nor numerical equality. Still required:

- Actual int32 input for all eight layouts and all 625 four-row slot mappings,
  including inactive/duplicate destinations, raw and finite payloads, persistent
  aliases and pre-write old reads.
- Decode KV prefix slices of the maximum `(4,1,4096,256)` allocation for each
  bucket, keeping original gather/scatter dependencies and tail concatenation.
  Verify output, host input and full device state bytes, strides/materialization,
  and untouched tails; do not assume a reshape preserves alias behavior.
- TP2 integrated multi-state graph checks and all full-model gates above.
  Prefill 1-row behavior stays original and is not newly qualified by the
  four-row candidate matrix.

The integer-bitcast reference remains a separately named diagnostic, not a
replacement for the original model baseline. No speedup is claimed. Future
writer end-to-end ratios cannot be multiplied by sampler or primitive ratios.

## Source evidence

All 22 source pins in the accepted plan matched before preparation. Twenty
SOURCE/STDLIB/MOCK tests passed (tool `60718d`, exit 0), including parent/worker
mode/source rejection, old/cross-arm artifact identity rejection, exact int32
and eight geometries, pair-confounding checks, complete graph inventory negative
controls, and restoration of all original computation AST/source after only the
documented dispatch edits. Initial 17-test subset also passed (`3a5a76`, exit 0).
Native imports are blocked by the test driver. Neither test set touches devices,
locks, existing runtime processes, model weights or graph payloads.
