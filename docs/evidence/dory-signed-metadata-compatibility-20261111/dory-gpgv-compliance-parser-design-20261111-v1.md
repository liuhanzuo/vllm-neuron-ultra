# One GnuPG compliance annotation: parser-only successor

Scope: the actual Dory signed-metadata r1 tool failed with exit 1 after the frozen parser rejected `VERIFICATION_COMPLIANCE_MODE`. Root supplied the saved six-line status shape, ending in `[GNUPG:] VERIFICATION_COMPLIANCE_MODE 23` immediately after `VALIDSIG`. This review did not read the saved payload/status, keyring, original plan/catalog, or any live process. It did not rerun gpgv or touch locks. The failed run remains failed and is handled by root's independent failure audit.

## Local documentation

The installed `/usr/share/doc/gnupg/DETAILS.gz`, decompressed in memory, defines:

- Lines 684–688: `VERIFICATION_COMPLIANCE_MODE <flags>` means the current signature-verification operation complies with the stated modes; flags are a space-separated list of numbers referring to field 18.
- Lines 244–251: the documented flags are 8 (RFC4880bis) and 23 (`de-vs`).
- Lines 421–425: `NEWSIG` is issued just before a signature verification starts, defining a current-signature context.
- Lines 427–435 and 481–512: GOODSIG and VALIDSIG provide the actual signature/key/fingerprint result. Expired or revoked signatures may still be cryptographically valid, which is why the separate failure/expiry statuses must continue to reject.

The local changelog begins with gnupg2 version `2.2.27-3ubuntu2.5`; this is documentation provenance, not a new executable or signature authentication claim. The docs do not promise an exact position relative to GOODSIG/VALIDSIG. The narrow position rule below is based on root's actual observed stream and intentionally rejects unreviewed alternative layouts. The docs' general recommendation to ignore future statuses is not adopted by this strict application parser.

## Pure helper contract

`research/dory_ubuntu_gpgv_status_compliance_v1.py` exposes the original four-argument `parse_status(stdout, returncode, trust, now)` interface and returns the exact old signature receipt structure.

It first enforces the original complete input byte/LF/no-CR/no-NUL and 1 MiB bounds, before filtering. Within a NEWSIG group, it removes at most one exact byte line `[GNUPG:] VERIFICATION_COMPLIANCE_MODE 23\n`, only after both GOODSIG and VALIDSIG have already appeared in that same group. It passes every other original byte and the original return code/trust/time to frozen adapter v2's `parse_status` exactly once. Merely seeing the two tag names cannot grant acceptance: the frozen parser subsequently validates their complete fields, signer binding, trusted primary fingerprint, strong digest, dates, expiry, duplicates, number of groups and successful exit.

The annotation is optional. It adds no signature success or independent trust; its absence does not fail a previously valid stream. The helper accepts only this observed one-flag form, despite the docs also defining 8. Empty/unknown/multiple/duplicate flags, extra fields, alternative whitespace, duplicate lines, an unscoped line, and a line before either success record are rejected. Every existing failure/expired/revoked/unknown status remains in the old parser input and fails. Nonzero gpgv exits continue to fail.

This is transient parser normalization only. The original complete stdout must still be saved, hashed and referenced as received. Do not rewrite the evidence file or replace its hash with the filtered parse view. No extra receipt fields are introduced, avoiding changes to saved replay equivalence.

## Future source wiring owned by root

No frozen adapter, binding, worker, parent launcher, saved auditor or source hash was edited. Root will create separately reviewed successor wiring for both actual verification and saved replay to call this helper. Worker and saved auditor must consume the same helper; updating only the live adapter would leave `saved_audit_v1.replay_saved` calling the rejecting v2 parser. Binding/source-review pins must add the helper and all actual successors without replacing historical frozen pins. Original transport, gpgv argv/process group/timeout, trust authority, full bytes/hash checks, output caps, locks and resource guards are unchanged.

## Small tests

Eight parser-only tests cover the exact root-supplied shape and unchanged old receipt, old streams without annotations, separate valid signer groups, byte-for-byte forwarding to the old parser, strict flags/whitespace, placement/duplicates, failure statuses and nonzero exits, original fingerprint/strong-digest/expiry/completeness/maximum-count rejection, and input cap/LF integrity before filtering. They perform no cryptographic verification, network access, payload reads, keyring reads, process observation, acquisition or real execution. Passing them does not qualify the failed run or authorize a retry.
