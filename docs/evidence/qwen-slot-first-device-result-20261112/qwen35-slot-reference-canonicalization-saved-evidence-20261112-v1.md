# Qwen conv slot writer r1: exact exceptional-value reconstruction

The first runtime correctness call failed in the existing compiled **reference**
writer. `conv` is the cache layout name here; this call does not run a convolution.
The candidate compiled successfully but had zero runtime correctness calls.

The saved call has BF16 shape `[4,3072,4]`, raw profile, round 0, step 0, cache
salt 0, update salt 17, and slots `[-1,-1,-1,-1]`. Frozen `store_slots` at
`qwen35_text_port.py:47` applies four `torch.where` selections. All four predicates
are false, so the required result is the original cache byte for byte. The leaf
computes that CPU expectation at `qwen35_predicated_slot_device_leaf_v1.py:173`.
Its full-byte comparison fails at line 201 before moving to the candidate.

The stdlib reconstruction follows `array_for` in
`probe_slot_cache_nki_read.py:81`. Both complete 98,304-byte reconstructed inputs
match the saved cache/update hashes. The source's little-endian int64 slots also
match their saved hash. Reported cache, updates and slots remain unchanged.

Applying just two transformations to the complete expected cache produces the
saved actual output hash exactly:

- Negative zero `0x8000` becomes positive zero `0x0000`.
- Every NaN becomes `0x7fc0`; all other 16-bit words remain unchanged.

The resulting SHA256 is
`7d4e13be97907a3c255b39b88eafdf33df817290cf363064ff96dd50a1e6c9bb`.
All 16 reported first difference offsets match too. The reconstructed difference
is 139 bytes/words: five negative zeros plus 134 changed NaNs. There are 135 NaN
words overall, one already equal to `0x7fc0`. There are **no negative NaNs** in
this exact input; it cannot distinguish retaining versus removing the NaN sign.

This is a deterministic full-output hash match, not a saved actual raw-output
capture. The report contains hashes and the first 16 offsets; it did not save
the actual values or all positions. The mechanism's location remains unknown:
compiled select lowering, runtime/transfer, and other device-path conversions
are not separated by this evidence. The saved HLO contains four `select`
operations, consistent with the reference source, but this does not identify
which later stage changes the exceptional bits. The `NKI - Kernel call` log
line occurs during candidate tracing and is not a candidate correctness result.

`qwen35_slot_reference_canonicalization_evidence_v1.py` reads saved files and
uses only stdlib byte operations. It checks all six relevant sources against
the actual r1 source map and binds root's physical-completion audit
`qwen35-predicated-slot-r1-root-actual-exit-20261112-v2.json`, SHA256
`6e07d96c6da69cfeb141d020e251cac8376ae007113a240d6d204db387318769`.
Reconstruction completed with tool chunk `82cace`, exit 0. No native code,
device call, current process/service scan or lock operation was performed here.

Next diagnostic: reuse the two saved reference/candidate artifacts and add one
explicit NKI load/store identity graph. Run each independently on the original
input, then a separately labeled input containing negative NaNs. Preserve every
actual output byte and every difference position. A reference mismatch must be
recorded without skipping the candidate. Identity must have an actual compiled
artifact and Neuron invocation; a host `return x` is not a transport control.

This independent diagnostic does not replace or pass any of the original eight
layout tests, 625 slot mappings, raw/finite profiles, alias tests or timing
requirements. R1 remains failed, with zero measurements and no speedup claim.
