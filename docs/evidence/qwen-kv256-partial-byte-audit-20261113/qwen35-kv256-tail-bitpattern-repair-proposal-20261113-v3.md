# KV256 proposal v3: actual partial terminal and reference-state clarification

This addendum preserves v1 and the delivered v2 (SHA256
`005c20694338f781e985f63622097e0da62f4bc0e1cadc962f8d19b5bbe5c04b`).
It contains source analysis and final r2 observations supplied by root, not a
new runtime, artifact inspection or independent raw-byte audit.

## Actual r2 partial terminal, supplied by root

- Outer session **18826**, terminal chunk **ea9360**, actual exit **1**.
- **15 / 24 calls** returned. Six graph records exist: four saved graphs,
  fresh cat-only and fresh alias-0. Alias-0 has **no completed call record**.
- Cat-only returned exact bytes for raw, finite and row-tagged raw profiles.
  No outcome is established for alias-0, tail-copy or full-copy.
- The supervisor's `resources(spec)` rejected the NVMe free-space floor plus
  remaining reservation. Native child return code was **-15 / SIGTERM**.
  This was **not a 900-second deadline failure**.
- Root reports the whole owned scope drained, with **476,770,304 allocated
  output bytes**, below the 1 GiB output cap. Initial available NVMe was about
  **40.004 GiB**, too close to the reservation boundary.
- Independent post-exit audit **session 14191 is running**. Its final saved-byte
  verification is still pending; this note does not pre-claim audit success.

The maximum output size and the minimum remaining free space are different
conditions. Remaining below the output cap cannot override the reservation
failure. Preserve this failed run and its original constraints. Do not immediately
rerun it: finish the independent audit and prioritize the planned MiniCPM5
retained-data/storage work. Any later diagnostic requires actual sufficient disk
headroom under the unchanged floor, output reservation and other resource limits.

## What following the observed state does, and does not, imply

V2 established from frozen r3 metadata that persistent `torch_where/raw` had
617 failed calls and eight passing calls; finite passed all 625. Its tail failed
on the first call only. The source sets the next `expected` state to the current
device result so that original raw negative controls can continue. Later exact
tails therefore need not equal the original pre-failure tail.

The actual code order is precise:

1. `geometry.py:193–203` computes the old/new/tail/device-state proofs against
   the expected state and integer oracle established for this call.
2. `geometry.py:206` assigns the observed device state to `expected` for the
   next call. This cannot alter the proof booleans and hashes already computed.
3. `geometry.py:207–214` derives `passed` from those saved proofs and requires it
   for **every NKI candidate call and every finite-profile call**.

Thus the assertion occurs after the assignment syntactically, but checks evidence
computed before it. A candidate or finite mismatch still raises immediately; it
does not silently become a successful next-call reference. The reference-state
explanation is about the explicitly retained `torch_where/raw` negative-control
sequence. It is not a reason to disqualify all passing candidate/finite results.

## Repair hypotheses remain provisional

Cat-only passing does not prove that the problematic concatenation lowering is
correct: two adjacent slices of the same cache may fold back into that cache.
This is a possible optimization, not an observed HLO transformation. The actual
compiled graph would need later admitted inspection. A graph that joins a new
NKI prefix to an old tail need not take the same path.

V2's equal-width BF16-to-S16 bitcast concatenation and explicit NKI full-state
assembly remain candidate designs only. They preserve public BF16 state/output
types and require independent compiled-graph and full byte verification. No
specific compiler pass, instruction or generic alias defect has been confirmed.
No fix is implemented, no full writer matrix is qualified and no new speedup is
claimed. Original full geometry, precision, cold-cache, long-decode, slot-reuse,
quality and matched HTTP gates remain necessary.
