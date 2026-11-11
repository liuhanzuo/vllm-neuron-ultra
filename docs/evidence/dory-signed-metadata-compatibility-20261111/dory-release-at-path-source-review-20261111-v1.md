# Release inventory paths containing @: minimal discovery v2

Source-only successor for the root-reported Dory r2 failure. Root reported actual session 41411 / terminal c87947 / exit 1 and, during its separate locked audit, 18 rejected entries among 919 signed Release SHA256 rows. All 18 were DEP-11 icon names with `@2`. This work did not read that Release, its signature, any saved payload, keyring, original plan/catalog or live process; the failed run remains failed.

`research/dory_ubuntu_package_sources_v2.py` copies frozen discovery v1 and changes only the Release path validator at one call site. The new `release_relative` uses the same original path rule plus the literal `@` character. The first character remains ASCII alphanumeric; every component must be nonempty and neither `.` nor `..`. Absolute paths, repeated/trailing slashes, traversal components, colon, backslash, percent encoding, query/fragment syntax, whitespace, control characters and non-ASCII names remain rejected.

The helper is used only for each SHA256-row path in `release_indexes`. It is not a URL parser and does not decode or normalize a path. All original digest, Release origin/suite/codename/architecture/component, date/expiry, duplicate selected index, exact size/hash, decompression, trust and input checks remain unchanged.

The wanted set remains exactly `main` and `universe` × `binary-amd64/Packages` and `Packages.xz`: four signed index entries per Release. The three fixed source suites and actual twelve metadata requests remain unchanged. The DEP-11 icon rows are syntactically accepted and ignored for selection; no icon, multiverse index or new URL is fetched.

Software-package `Filename` still calls the unmodified original `relative`. In particular, `pool/main/m/mock/libc6@2.deb` still fails. Package/component/version/architecture/member mapping and no-archive/no-qualification flags remain unchanged. Frozen discovery v1, adapters, bindings, workers, parents, auditors and recorders were not edited. Root owns their separately reviewed successor wiring and exact source-hash additions; old source pins must stay intact.

Validation uses an existing source-isolated synthetic fixture guarded against real network, subprocess, locks and non-module input reads. Six focused tests cover all 18 reported name shapes (main/multiverse/universe × 128x128/48x48/64x64 × `.tar`/`.tar.gz`), malicious path characters/components, four selected indexes, exact twelve request URLs without icons, package-Filename rejection, missing/duplicate/changed index paths, invalid digest and complete AST equivalence after removing the one helper/call-site change. The ten original discovery tests are also run unchanged against v2, with only synthetic input pins replacing actual fixture IO.

This source correction does not establish a real new Release/signature chain, admit a retry, promote r2 or qualify package content/native ABI/model behavior.
