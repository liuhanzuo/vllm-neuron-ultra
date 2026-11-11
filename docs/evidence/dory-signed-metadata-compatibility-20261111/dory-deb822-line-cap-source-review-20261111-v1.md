# Discovery v4: bounded long deb822 field compatibility

One-line source-only successor to discovery v3. Root reported r4 session 27281 / terminal 13b3aa / exit 1, independently audited by 48af4b / exit 0. Root's separate diagnostic 522153 / exit 0 authenticated the signed index hashes and identified one 70,840-character Provides line in the `librust-winapi-dev` universe stanza (stanza 71,832; complete raw index 64,332,414 bytes). This review did not read those actual files or perform live/locked/network work.

Only `stanzas`' per-line condition changes: `len(line) <= 65536` becomes `len(line) <= min(128 * 1024, stanza_cap)`. Line length retains the original decoded-character semantics after removing the LF terminator; the reported Provides line is ASCII. The 128 KiB ceiling and any smaller caller stanza cap both apply. No parser field is ignored, truncated or normalized.

The default 256 KiB stanza sum, Release's existing 4 MiB stanza bound, compressed/raw authenticated sizes and hashes, raw 128 MiB cap, field syntax, case-folded duplicate rejection, continuation semantics, stanza-count limit, exact empty-XZ-tail rule, Release-@ rule, fixed requests and package matching are unchanged. This is a bounded parser-field compatibility change; no CPU, process memory, disk, timeout, lock, trust or admission limit changes. All frozen predecessors remain unchanged.

Six focused memory-only tests check exact whole-module AST equivalence after the one expression substitution; 65,537 / 70,840 / 131,072-character lines pass and 131,073 fails; lower stanza caps remain stricter; default and custom stanza sums and raw aggregate caps still reject overflow; case-insensitive duplicate fields and orphan continuations fail; and Release's larger stanza allowance cannot bypass the 128 KiB line ceiling. No original payload/plan/catalog/keyring or runtime observation is used.

Source/mock success does not admit a retry or promote the failed r4. Root owns future additive source pins and successor wiring.
