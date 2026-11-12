MiniCPM5 SDK310 composite package origin source delivery, 2026-11-12

The frozen delivery is `minicpm5-sdk310-composite-origins-source-delivery-20261112-v1.json`, SHA256 `7560b44eb99a6b8447b4d0321eeb1cdbbf42dbe8ee545719183e5b7dbdcd6560`. This is source preparation, with no runtime admission or native/model qualification.

The original worker could only consume one package result and looked every provider up in the old signed result's six-package list. The newly completed eight-package result does not contain libc6, and its package names are intentionally absent from that immutable old list. The successor preserves both independent records and verifies each against its own original package source. It never constructs a merged signed result or changes the original input n.

Three runtime sources are added:

- `minicpm5_sdk310_composite_origins_v1.py`: explicitly reads the two real producer/replay chains and their separate recorder exits. It verifies the new eight-package archives against the unchanged original n, signed Release ancestry, complete immutable Packages plaintexts and all fourteen exact stanza occurrences. Old libc6 retains the already completed six-package signed archive/member chain. Every selected native or pure-source provider must be found in exactly these verified package records.
- `minicpm5_sdk_mapping_worker_native_v3.py`: changes the authority schema to `minicpm5_sdk310_native_source_review_v2` and dispatches package provenance to the new module, supplying the full original source map. All class/function AST remains identical after normalizing those two explicit changes. Current full-file hashes, ELF/SONAME/FFI checks, actual mappings, dynamic `_ctypes` selection and history checks are retained.
- `minicpm5_sdk_mapping_worker_source_entry_v4.py`: changes only its native-module import from v2 to v3. The entry body, original nine checks, full 1,284-row plan, zero capture reads, original 900-second deadline, 2 GiB AS limit and subsequent gates are unchanged.

The old worker v2 and source entry v3 are untouched. No existing entry plan, full source map, root review, admission or runtime authority is overwritten or generated here. The accepted outer312 entry and authority remain separate.

The authority successor contains an explicit `package_origins` object:

```python
package_origins = {
    'schema': 'minicpm5_sdk310_composite_package_origins_v1',
    'old_libc6': real_old_bundle,
    'new_eight': real_new_bundle,
}
```

Each bundle has exactly `producer_result`, `producer_exit`, `producer_recorder_exit`, `replay_result`, `replay_exit`, and `replay_recorder_exit`. The JSON-only real check contains all twelve actual references under `old_libc6` and `new_eight`. They bind the successful old download r3/replay r1 and the actual new download r4/replay r1. This parameter interface has no invented future success defaults. The old libc6 replay is additionally pinned to its existing exact path/hash. The new result still binds the original n SHA `e91bca576461d9d1d4b206e7f8b508c20a103a3231bef9b29535e97d3e8da266`.

Each stage's exact root-reviewed plan/source set must occur in the complete SDK source map. The future source-map builder must therefore retain those archive/snapshot source pins as well as add these three runtime sources. A union of stage source maps in the metadata check is explicitly only test input; it is not a reconstructed current runtime source map.

The origin reader uses the existing `StableReader` interface. It does not import the serial archive reader or depend on its current Python module/hash. The successful archive-stage plan and review remain the evidence for the explicit dpkg status observation. The origin function neither rereads stale dpkg status as current nor redirects old state references. Full current native bytes and loader selection remain the responsibility of the unchanged SDK native checks.

The full signed plaintext recheck is deliberately a future runtime operation. In this delivery its complete logic was tested against synthetic plaintext and negative controls. The real check reads only saved JSON, including both completed package results, all producer/recorder exits, root plan/review records, and snapshot-stage JSON. It does not reread actual compressed/uncompressed snapshot payloads, archives, binaries, weights or tensor captures.

Validation:

- Twelve bounded stdlib/source/mock tests passed at tool chunk `d32a48`, exit 0. They exercise the complete two-origin flow, full synthetic index bytes, exact stanza offsets/hashes, untrusted signatures, missing/failed completion, source/review mismatch, changed members, invented old hashes, real tool-call/session shapes, ref2/ref3 compatibility and unchanged worker/entry AST.
- Real JSON-only validation passed at `ef6d07`, exit 0, with the saved report `minicpm5-sdk310-composite-origin-real-metadata-check-20261112-v1.json`. After adding explicit signed-metadata recorder identity checks, the final source passed again at `6cd52d`, exit 0. It confirms old six packages and new eight packages/twelve targets, with zero payload reads.
- The first real check failed at `1dc891`, exit 1: the new replay's prior reference contains path/hash while the producer result also has bytes. The failed source is retained in `minicpm5-sdk310-composite-origin-refshape-failure-source-20261112-v1.py.txt`. The correction permits the two exact reference shapes while requiring equal path/hash and equal sizes whenever both carry a size. The result must still equal its exact reviewed plan's prior reference. A regression covers successful ref2/ref3 binding and conflicting-size rejection.

Both source check commands are bounded and perform no native/model execution:

```text
taskset -c 7 nice -n 19 prlimit --as=536870912 --cpu=60 -- python3 -S -B research/check_minicpm5_sdk310_composite_origins_v1.py -v
taskset -c 7 nice -n 19 prlimit --as=536870912 --cpu=60 -- python3 -S -B research/check_minicpm5_sdk310_composite_saved_metadata_v1.py
```

Root source review, real SDK310 native closure, real outer312 positive FFI/mapping authority and current full entry bindings remain required before the original import-only diagnostic. Cold-cache, long-decode, request-slot reuse, full model correctness and performance qualification remain pending and unchanged. No speedup is claimed by this source or dependency work.
