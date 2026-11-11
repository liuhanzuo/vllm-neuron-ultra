# One exact empty second XZ stream: discovery v3

Source-only successor for root's reported Dory r3 failure (session 80896, terminal d0e616, exit 1). Root independently reported that the entire compressed index size/hash and primary decoded raw size/hash matched the signed Release, with first decoder EOF true and 32 unused bytes. Root then supplied this exact suffix and verified it in memory (tool 0d05e5, exit 0): output empty, EOF true, no remaining bytes.

`fd377a585a000004e6d6b446000000001cdf44211fb6f37d010000000004595a`

This is an empty second XZ stream, not zero padding. The initial padding hypothesis was withdrawn before any source was written. No actual payload, Release, signature, keyring, plan/catalog, lock or process was accessed for this work.

## Local format documentation and scope

The installed `/usr/share/man/man1/xz.1.gz`, decompressed in memory, states at lines 270–277 that XZ files may be concatenated and are decompressed together. Lines 279–282 separately describe stream padding as null bytes with length divisible by four. That padding rule does not describe the actual suffix here.

Discovery v3 intentionally accepts a narrower form than general XZ concatenation: the first stream may have no trailing bytes or exactly the root-observed 32-byte suffix above. It accepts no other empty-stream encoding, padding, arbitrary second stream, third stream, malformed/truncated stream or garbage. A generic concatenated decoder is unnecessary for this concrete compatibility correction and would broaden the accepted input form.

## Minimal implementation

Frozen discovery v2 is copied to a new v3 file. Only one main-stream predicate changes, from rejecting every `decoder.unused_data` to invoking the new pure `empty_xz_tail` helper. Main decoder EOF remains the first short-circuited condition. Full compressed bytes, including the suffix, must still match the signed compressed size and SHA256 before decoding. Main decoding retains `FORMAT_XZ`, 64 MiB memory limit and signed raw size + 1 output bound; the raw size and SHA256 must still match exactly afterward.

The helper requires exact bytes. Empty bytes pass without another decoder. Otherwise length must equal 32 and every byte must equal the constant; then a second `LZMADecompressor(FORMAT_XZ, memlimit=64 MiB)` decodes with `max_length=1` and must produce `b''`, reach EOF and leave no unused bytes. Thus the narrowly allowed suffix contributes zero bytes to the authenticated index. Both decoder objects are bounded; no additional file/network access or output artifact is introduced.

All Release-`@` compatibility, original package Filename restrictions, fixed suites/URLs/index selection, signature receipt binding, decompression limits, metadata caps, exact package-version rules and diagnostic flags remain unchanged. No existing discovery/adapter/binding/worker/root/auditor/recorder file is edited; root owns separately reviewed future wiring and additive source pins. The actual failed r3 stays failed.

## Small validation

Seven focused methods plus all ten original discovery methods run on synthetic memory metadata under the source-only audit hook. They check complete AST equivalence except the helper/constant/one predicate/message, no suffix and the exact observed suffix, rejection of zero padding (including 4/32 bytes), alternate empty encodings, nonempty second stream, third stream, truncation and garbage, unchanged signed compressed size/hash (including suffix), unchanged raw size/hash, both decoder parameter sets and required empty/EOF/no-remainder result, and coexistence with Release `@` while package Filename `@` still fails.

These tests do not perform real cryptographic verification, authenticate the actual index, establish a new completed run, authorize retry or qualify package content/native ABI/model behavior.
