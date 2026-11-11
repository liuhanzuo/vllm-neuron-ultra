# LFM NKI 0.5 writeback followup: source-only proposal

## Actual r2 evidence

The authorized small `lfm3-nki05-transport-r2-20261111/diagnostic-result.json`
has SHA256 `5c252ec602649007e18f1f40519945fcb95b1dee5d884c849edf7beed06054e5`.
Its first identity check returned the exact expected 4,096 bytes/hash, with zero
different words. All ingress and pre-return view/backing observations passed,
coverage was complete, and `first_boundary_divergence` was null. Nevertheless,
`inputs_unchanged` was false. The original suite stopped at that failure, before
the remaining 31 movement cases. Root observed actual session 16107 exit 1,
terminal `fcf693`; this delivery does not change that outcome.

This matches the source-derived writeback hypothesis. It does not yet identify
which operand words changed or prove their specific conversion behavior. The
followup records those facts directly and separately tests the precision choice.

## New, inert sources

| File | SHA256 |
| --- | --- |
| lfm3_nki05_writeback_probe_v1.py | abdb58d5acc086ac11f177fa98d38b6424d5fa02d9612e5343c534331534e073 |
| lfm3_nki05_transport_diagnostic_v2.py | cf83c634b0f71f5d8a483704b7fed2bdad2588736866d81f28d19416ac11263a |
| check_lfm3_nki05_writeback_diagnostic_v1.py | 8e5ddb41fa8972dd80bc5e21c1b781bb6c503497c6977a7eeb7bd8afe0bb9047 |

There is no CLI, launcher, admission producer, SDK file modification or edit to
the executed v1 diagnostic/runner. Importing these sources imports no framework.
The original `diagnose` body is copied as `_suite` with an identical AST after
renaming that function. It keeps every original source/version/precision and
resource check, all fixture hashes, the original 32 cases, all 65,536 BF16
encodings, negative controls, arithmetic control and rounding cast control.
Its existing fail-fast assertions are preserved verbatim.

## Three distinct arms

1. **Original:** observe the unchanged extractor. Require reproduction of r2's
   first identity failure: output exact, observer coverage complete, no observer
   divergence, and host inputs changed. Preserve the failed suite object. It
   stops after that case exactly as the original implementation did.
2. **Candidate:** independently regenerate the same fixture and run the entire
   original suite with one narrow process-local interception described below.
   A candidate pass requires all original 32 cases and all auxiliary gates.
3. **Restored original:** restore the exact extractor object and rerun the
   original suite under observation. Require the same case results, including
   exact per-operand before/after hashes and bit differences, as the first arm.

The top-level schema is `lfm3_nki05_writeback_diagnostic_v2`. It explicitly
separates `original_suite_passed` and `candidate_suite_passed`. `complete` means
the characterization finished, not that the original implementation passed.
The top-level `passed` remains false when the original arm fails, even if the
candidate passes. The historical actual failure is never rewritten.

The unchanged serial runner's v1 success validator intentionally cannot accept
this new characterization schema. Root must independently review any future
binding/runner contract. No such binding or runtime file was created here.

## Narrow interception and restoration

Installation occurs lazily at the first original `run_case`, after `_suite` has
completed its original admission, source hashing, environment/version checks
and authorized framework imports. The caller supplies the already loaded NKI
simulator and NumPy modules; the probe performs no native import itself.

The probe authenticates `_simulator.py` against its frozen hash, compiles source
without executing it, and compares the actual `_tensor_to_numpy` and
`simulate_kernel` code objects and original default arguments with the source.
It derives the one omitted-precision call from the pinned AST: line 215.

The candidate substitutes `precise=True` only when all these conditions hold:

- The caller is the exact authenticated `simulate_kernel` code object.
- The call occurs at its unique omitted-precision writeback source line.
- The callee's tensor is the caller's current writeback tensor, with a current
  `orig_arr` host operand.
- The precision argument was omitted and the tensor dtype is exactly BF16.

Explicit `precise=False`, explicit `precise=True`, other callers, and other
dtypes retain their original behavior. Returned-output conversion therefore
keeps its original explicit precise path. No input is made read-only, restored
after the fact, normalized, replaced in the kernel, or excused from its gate.

Only the module's in-process extractor slot is temporarily replaced. Every arm
uses `finally` to restore the exact original function object, including ordinary
failure, cancellation and `KeyboardInterrupt`. A detected competing wrapper
still restores the original object before raising. Normal process termination
cannot persist this memory-only change to the installed SDK. No claim of
Python-level cleanup after an uncatchable process kill is needed or made.

## Evidence captured

For cache, updates and slots, each case stores the complete before/after SHA256,
dtype, shape, total word count, changed word count, total changed bits and a
counter for every bit position. Floating-point differences distinguish both-NaN
changes, NaN/non-NaN transitions, other changes, sign-bit changes and signed-zero
changes. All words are inspected; only the illustrative word sample is capped
at eight, so large mismatches do not imply truncated counts.

Every actual extractor call records caller/case, omitted versus explicit
precision, effective precision, tensor/storage/extracted dtypes, shape and
whether substitution occurred. At each BF16 input-writeback call, the probe
also applies the original extractor with explicit false and true to the same
unchanged simulator tensor, followed by the exact original `numpy.copyto`
primitive into two detached host-array clones. It records both complete bit
differences and verifies the real host array and simulator storage did not
change during these counterfactual probes. The clones never feed the kernel,
the real writeback, the byte oracle or an expected output.

These matched precision controls also cover the entire BF16 encoding operands
when the candidate reaches the unchanged exhaustive cases. The original arm
does not bypass its first failure merely to reach later cases. The candidate's
exhaustive inputs provide the same-original-extractor false/true comparison
without modifying the original suite's failure semantics.

A fully passing candidate requires exactly 132 extraction calls: four per
movement case (three input writebacks and one return), plus two each for the
arithmetic and cast controls. Exactly 40 substitutions are expected: two BF16
input operands in each of the original 20 BF16 movement cases. The 12 FP32
movement cases and the FP32-input arithmetic/cast controls are untouched.
Both original identity arms require four calls and zero substitutions.

`writeback_precision_explanation_supported` can become true only after the
candidate passes all original gates, the exact call/substitution counts hold,
the original false-precision clone copies lose bits while true-precision copies
preserve them, and the restored original reproduces its earlier failure.
This remains simulator characterization, not full legacy 3,716-case,
Trainium-device, model-quality or performance qualification.

## Bounds and validation

The original 10 GiB/8 GiB service memory policy, 12 GiB process address-space
limit, 1,800-second CPU/runtime bounds, CPU affinity, threads, output floor and
256 KiB report limit are not increased. The probe limits each captured operand
to 256 KiB and each arm to 160 extraction events. It stores no raw payloads in
the report. The complete three-arm JSON must fit 256 KiB at return; exceeding
that limit fails rather than truncating evidence. Actual three-arm report size
and native runtime behavior remain untested and are review/runtime checks.

Command: `/usr/bin/python3 -B check_lfm3_nki05_writeback_diagnostic_v1.py`.
Result: **9 source/mocked tests passed**, actual exit 0, terminal chunk
`c4ce93`, 0.243 seconds. Controls cover every word/bit across a synthetic 65,536
word input, finite/NaN/signed-zero classification, unchanged original behavior,
the exact-callsite-only candidate, explicit-false/other-caller noninterference,
restoration after tampering and cancellation, genuine pinned code/default
authentication without executing SDK functions, and unchanged original suite
AST/source hash. Synthetic conversion fixtures intentionally model a lossy
cast; they are not measurements of NumPy/ml_dtypes behavior.

Work stopped at this source and test delivery. No serial work was started; no
active Mini or Dory session, spec, handoff, installed file or runtime artifact
was changed.
