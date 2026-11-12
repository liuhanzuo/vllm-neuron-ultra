# Qwen package candidate preparer v7

This source-only successor preserves frozen preparer v6 and adds explicit
support for package runtime v8. It requires the v8 source delivery and root
review schemas, `source_text.py`, `compiler_assets.py`, and the literal-pinned
`compiler_assets.json` source manifest. Only the observed sixteen installed
PWP asset paths and the observed SDK version are accepted.

The caller's guarded stat-only reader checks every asset against its recorded
device, inode, size, mtime and ctime, then the existing final identity pass
checks it again. The authenticated manifest also fixes singleton regular-file
metadata, full hashes, bounds, and wheel RECORD provenance. This helper reads
no binary bytes. Runtime v8 separately performs full nofollow identity/hash
checks before lock acquisition and during storage observation.

All historical source/input merges, the six original typed binary receipts,
catalog prefix, original metadata limits, budget validation, and false returned
review/authorization are preserved. Only explicit runtime v8 recognition,
mandatory asset source closure, fixed-manifest/stat verification, and the
candidate version label change. No actual candidate preparation, current
binding selection, History, lock, native import, model or compiler was run.

The final suite passed 57 tests (actual chunk 3e98df, exit 0): the previous 49
plus eight v8 tests covering source closure, stat-only separation, missing or
changed modules, manifest rebinding, exact target membership, SDK version and
bounds, changed last-asset identity including final recheck, and review schema.
One initial test used an incorrect message regex; runtime behavior already
raised the intended identity error. Its initial failed output is retained.

The returned `catalog_source_extension` remains the aggregate overlay extension.
A later catalog append must derive genuinely new pins against recovered and
all historical rows, prove complete equality with the executed spec, preserve
every old row, and retain the existing 8 MiB catalog cap. This source delivery
does not authorize any such append or select a current catalog.
