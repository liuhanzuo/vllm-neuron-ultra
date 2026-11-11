# Six exact Ubuntu archives: minimal next byte-verification design

Design only. No endpoint, package/tool binary, keyring, archive, system library,
lock or process state was queried here. No concrete execution spec is created.

1. **Admit only audited exact addresses.** Consume the real signed-metadata
   producer's result and its independent actual-exit/recorder-exit refs once
   available. Require `exact_addresses_ready`, zero missing triples, exactly
   the fixed six package/version/amd64 entries and twelve target/member pairs,
   and unchanged original eleven observed payload refs. Missing versions stop
   all package downloads; preserve exact missing triples for a finite official
   historical-source follow-up. Never choose latest or infer a package hash
   from an unauthenticated pool listing.

2. **Download and authenticate six whole archives.** Select one deterministic
   already authenticated source URL per package; only that finite six-URL set
   is allowed. Reuse adapter's no-redirect TLS/HTTP-200/bounded transport rules
   and prior archive helper's chunked SHA/exclusive create/fsync/reread loops,
   with signed Size/SHA256 rather than HEAD-derived authority. Rehash each
   complete saved `.deb`, with exact EOF and stable identity, before parsing.
   Retain all archives (maximum 64 MiB total) and compact evidence under the
   existing 128 MiB stage cap; no expanded package trees. A failed fetch or hash
   stops immediately and retains partial evidence. No automatic alternate-URL
   retries or install step.

3. **Use existing Debian decompression, keep member verification small.** The
   inspected research archive helpers contain wheel/CPython tar parsers but no
   `.deb` parser. Preferred candidate is the fixed installed command
   `/usr/bin/dpkg-deb --fsys-tarfile EXACT_AUTHENTICATED_ARCHIVE`, reading stdout
   through a bounded pipe, stderr separately, zero exit and exact EOF required.
   This is a proposed system capability, not verified availability. Root must
   first establish its exact path/byte identity and trust basis. Like installed
   gpgv, this treats the byte-pinned OS tool as a base; it does not claim its own
   package/dependency closure is independently reconstructed. It executes no
   maintainer scripts and performs no extraction or installation. The subprocess
   and any decompressor children must inherit the worker process group; parent
   drain remains authoritative if worker dies. Retain full bounded stderr and
   actual command/exit. This avoids writing separate ar/XZ/gzip/Zstandard code;
   Ubuntu archives may use Zstandard, which the original Python may not support.

4. **Verify the complete tar stream and fixed members without extraction.**
   A small streaming USTAR reader can validate 512-byte header checksum, finite
   octal sizes, padding, total bytes/count and two zero end blocks followed only
   by zero padding through decompressor EOF. Use explicit member and stream
   bounds (proposed 10,000 headers, 64 MiB individual member, 256 MiB decompressed
   per package, 512 MiB aggregate; root should bind resource estimates before
   execution). Accept directory, ordinary file, symlink and hardlink records;
   reject devices/FIFO, sparse formats, base-256 numbers and unsupported PAX/GNU
   extensions until an observed need has a bounded reviewed successor. Keep
   normalized names unique: allow the conventional leading `./` and root `.`
   directory, reject absolute names, backslashes, `..` components and ambiguous
   repeated separators; collisions after normalization fail. Do not use a tar
   library's automatic extraction or unbounded hidden extension allocation.

5. **Treat links as archive relationships, not host resolution.** Hash all
   ordinary members in bounded chunks while scanning once; retain only compact
   name/type/size/hash/link descriptors in memory so forward hardlinks can be
   resolved later. For the twelve target members and saved alias names, resolve
   links entirely in that authenticated archive table (at most 32 hops, reject
   cycles/missing targets). Relative symlinks resolve from their containing
   archive directory; an absolute link is interpreted from the package root,
   never the host root. `..` in link text is allowed only when normalized within
   package root. Hardlink names are archive-root relative. Fixed canonical
   target mappings preserve `/lib` package names versus `/usr/lib` observed
   files explicitly; never follow an unknown host link or use live `realpath`
   to authenticate an archive link. Preserve loader `/lib64` and tcmalloc SONAME
   alias evidence. Missing/duplicate/escaping/cyclic target or required alias
   prevents a complete member result; do not claim an excluded member matched.

6. **Compare original and current bytes separately.** Compare target member
   full SHA256/size to each of the eleven saved observed refs. Then use the
   existing stable `O_NOFOLLOW`/before-after identity/full-SHA comparison pattern
   to compare exactly the twelve canonical installed files, if the next actual
   stage is admitted for those finite reads. Save current identities; an identity
   or byte change is a distinct mismatch, never overwrite the original refs.
   For system libgomp, no original full hash exists: derive its expected member
   hash from the authenticated archive, then make a new stable installed-file
   comparison and label it as a new observation, not a recovered r5 hash.

Non-target package content must still be consumed to validate the tar envelope,
EOF and duplicate names. It is **not** installed or compared to all local paths,
and its presence adds no system-inventory qualification. Compact non-target
counts/bytes and a deterministic descriptor-table digest suffice; retain the
whole `.deb` to reproduce the scan without storing full extracted content.

Minimum durable evidence: input/audit/source/trust refs; six exact archive URL/
size/SHA/identity records; exact decompressor argv/status/stderr and full tar
stream byte count/hash; complete fixed target/alias chains with terminal member
SHA/size; eleven original comparisons plus twelve current observations; bounded
non-target counts/table digest; ordered failure/completion events and actual
parent/worker exits. Independent audit rehashes archives and recomputes member
results using the same finite admitted capability under the original serial
contract. All payload/loader/cache/model flags remain false until their own
work is actually verified. Even all twelve byte matches do not choose the
13 vLLM loader edges, two libgomp load edges or four timestamp caches.
