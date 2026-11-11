# Minimal next work after the six-package replay

Source-only design, 2026-11-11. No target interpreter, native package, cache,
binary, model, archive payload, compiler, or probe was opened/run for this note.
Only research source and explicitly bounded saved JSON metadata were read.

The package replay closes exactly the original 11 system byte-origin gaps.
Its system libgomp match is a new observation, not a historical selection.
The remaining set is 13 vLLM dependency edges, 2 libgomp selection edges, and
4 timestamp caches. Package provenance does not qualify native ABI or a model.

## Immediate operation: four cache payload comparisons

Implement one bounded diagnostic root driver, for example
`run_dory_timestamp_cache_verification_root_v1.py`, with an inert verification
function and an embedded minimal `-c` child program. This can be a single new
implementation file: no new queue, inventory clone, target worker script, or
modified frozen resolver is needed. A separate source-only checker is useful
for malformed headers, mismatched sources, truncated output, and child failure;
it must not import the target interpreter or native packages.

Suggested directly implementable interface:

```python
verify_timestamp_caches(
    *, remaining_ref, external_closure_ref, python_review_ref,
    sysconfig_confirmation_ref, package_result_ref,
    output_root, guard, emit, run_owned_child,
) -> result_reference
```

`run_owned_child` belongs to the root's existing serial admission, resource,
process-group drain, and actual-exit protocol. This API neither acquires a lock
nor claims admission. The root driver freezes its complete source closure,
binds latest actual exits/history, and supplies the guard before payload reads.

Inputs already present:

| Saved input | Bytes | SHA256 |
| --- | ---: | --- |
| research/dory-remaining-gaps-after-package-replay-20261111-v1.json | 37344 | fdea581b6127e48774ce235cc5a3969eb922c99ee6581a49a0cabc177dea8bb0 |
| /mnt/nvme/dory-serial-inventory-r5-20261111/inventory/external-closure.json | 137884 | 8e97c912e0cbcd75ee4eead45565a05d727c5bc043ef5bf0ea2eeb51002ffd6e |
| research/dory-torch-cpython-archive-r2-post-exit-root-review-20261111-v1.json | 5237 | 26456bac52441bd98c201e43ac8481076286cfd43f43e0ac0f47bd645b4a7e04 |
| research/dory-uv-sysconfig-root-confirmation-20261111-v1.json | 754 | 0868d594a99a83330fe7719561e0d869f36a5d5f2ae0ccad8f6d1f07bf8475c8 |
| /mnt/nvme/dory-serial-inventory-deb-replay-r1-20261111/packages/sources-result.json | 31097 | 594f9c3fa05579fa1da9af0a486ca252d988ff16e05041bd05b1b18237574a14 |

Here `research/` is
`/home/ubuntu/experiments/trn1-latest-models-20261031/research/`.
The actual r5 exit authenticates the external-closure reference; its own file is
`research/dory-serial-inventory-r5-actual-exit-20261111-v1.json`, 8157 bytes,
SHA256 `e6cff0181f6a1d95c7f816c6d906e9bf170dd4768a5cc52a56f94f26b170ec89`.

Exact interpreter reference:

```json
{"path":"/mnt/nvme/venvs/dory-official-cpu-20261108-v1/bin/python","bytes":31028464,"sha256":"23bc148d2b48964b621c5561ff2b81c1d9c7b717967a7491efbf12f4304882b5"}
```

The CPython archive review's `summary.cpython.venv_interpreter_copies` confirms
this exact copy against CPython 3.12.16, build 20261026. The external closure's
matching `files[]` row preserves its identity and ELF metadata. Its RPATH is
`$ORIGIN/../lib`, NEEDED is pthread/dl/util/m/rt/c, and interpreter is
`/lib64/ld-linux-x86-64.so.2`; their recorded canonical objects are among the
package-authenticated system files. Recheck the finite startup object bindings
and identities under the root guard, including the loader alias; do not infer
arbitrary future ELF selections from package hashes.

Let `PYLIB` be
`/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12`.
The four exact sources are:

| Source below PYLIB | Bytes | SHA256 | Existing cache bytes |
| --- | ---: | --- | ---: |
| encodings/__init__.py | 5884 | 78c4744d407690f321565488710b5aaf6486b5afa8d185637aa1e7633ab59cd8 | 5839 |
| encodings/aliases.py | 15677 | 6fdcc49ba23a0203ae6cf28e608f8e6297d7c4d77d52e651db3cb49b9564c6d2 | 12472 |
| encodings/utf_8.py | 1005 | ba0cac060269583523ca9506473a755203037c57d466a11aa89a30a5f6756f3d | 2211 |
| _sysconfigdata__linux_x86_64-linux-gnu.py | 51256 | cc3e58175aff8e0b034e55acbbf277613cfab33913cf904bb6c2daff2d985d5f | 63633 |

Cache paths, original identities, source identities, and original 16-byte headers
are exact fields in `external-closure.pyc[]`; remaining JSON reproduces the
four target paths and source references. The first three source references
come from exact CPython archive comparisons. The fourth is independently
verified UV relocation output: the confirmation binds upstream SHA256
`ea81f7710731753c3834d981e320bec5262f184f240cec4aa8e6658a68e02902`
to the installed source hash above. It does not authenticate the pyc by itself.

The original archive comparison is available by reference if root needs to
audit that derivation: `/mnt/nvme/dory-torch-cpython-archive-verification-r2-20261111/full-comparison.json.gz`,
1094180 compressed bytes, SHA256
`7e61203920e5b2e7f9e32d56900c42ebedf9f9a4eae6e5757b59a6edb35c2180`;
7939869 uncompressed bytes, SHA256
`5c4313be7f9831abef3e6dbb43f26f7fb83fd442f956bfd79f114770847fd965`.
Reading that archive is not needed to discover additional inputs for this
four-file operation; do not expand it into another full environment scan.

Procedure:

1. Authenticate the saved input JSON and the interpreter reference. Read only
   these four source files and four target caches, checking ordinary canonical
   files, exact expected lengths, stable descriptor/path identities, full
   source hashes, and the saved cache identities. Caps: 128 KiB per source or
   cache, 1 MiB combined source/cache input. Observed totals are 73822 source
   bytes plus 84155 cache bytes. Unexpected changes fail the fixed plan.
2. Before target-interpreter startup, authenticate the three encodings sources.
   Create a fresh owned empty cache-prefix directory within the diagnostic
   output. Launch the authenticated interpreter with
   `-I -S -B -X pycache_prefix=<that directory> -c <reviewed program>` under
   the root-owned child protocol. Use the independent verifier's reviewed
   minimal environment, with no inherited LD_PRELOAD/LD_LIBRARY_PATH or
   PYTHONPATH/PYTHONHOME. This environment is solely for the verifier.
3. The child imports only built-in `sys` and `marshal` and uses builtins for
   bounded framing/output. Check version 3.12.16, optimize=0, isolated/no-site,
   dont_write_bytecode, and exact `sys.pycache_prefix`. Return module origins;
   every non-built-in/non-frozen imported module must be one of the three
   preauthenticated encodings source paths. Startup necessarily executes those
   trusted encodings sources; it must not consume their original pycs.
4. Send source bytes and exact original absolute filenames through bounded
   input. For each, calculate
   `compile(source_bytes, original_absolute_filename, 'exec', dont_inherit=True, optimize=0)`
   and `marshal.dumps(code, 4)`; require the matching interpreter's marshal
   version 4. Never call marshal.loads on a target cache or exec the generated
   target code object. No json/hashlib/struct import is needed in this child;
   the system-Python parent can hash and serialize the result.
5. Reconstruct the timestamp header from the validated interpreter magic
   `cb0d0d0a`, flags=0, source mtime seconds modulo 2**32, and source byte length
   modulo 2**32. The saved four source timestamps are all 1793420276 seconds.
   Verify the original/current header equals this reconstruction; compare all
   16 header bytes plus the complete marshalled payload. Do not merely attach
   an unchecked input header. Each expected payload can be returned as bounded
   hex (total well below 1 MiB) for parent comparison and hashing.
6. Recheck all identities and the unchanged empty verifier cache prefix, record
   exact input/expected/actual hashes and sizes per file, and require actual
   child exit 0 plus process-group drain. Root separately records actual parent
   exit and saved evidence review. Mismatch preserves the gap; no cache deletion,
   rewrite, alternate co_filename search, or fallback interpreter is allowed.

An exact comparison establishes current cache payload derivation. The +43-byte
encodings difference is only a clue until this comparison succeeds; it must not
be predeclared to be co_filename relocation. Unchanged historical cache identity
supports continuity, but historical full-payload hashes did not exist, so do not
claim an independently recorded historical cache hash. The original target
interpreter flags and experimental environment remain unchanged.

## Reusable code and its limits

- `dory_native_inventory_finish_v4.py`: `file_bytes` (line 87), `load_json`
  (144), `pyc_decision` (369), and `write_json` (519) are usable guarded bounded
  helpers. `pyc_evidence` only reads a header and is insufficient for this task.
  Its `external_system_file_bytes` preserves the explicit two-hardlink exception.
- `dory_descriptor_inventory_runtime_v2.py`: acquire/Guard and exact descriptor
  evidence express the current lock policy. Its `validate` is intentionally
  inventory-only; do not relabel this target-interpreter diagnostic as inventory
  or change that frozen validation. Reuse approved lifecycle primitives in the
  new specific root driver with an independently reviewed bounded plan.
- `run_dory_serial_stage_root_v5.py`: history, drain, storage, final-result, and
  release-after-drain helpers remain applicable. `dory_serial_stage_journal_v2.py`
  supplies the durable event sink. Do not revert to an older lock proof.
- `dory_stage_a_native_resolver_v4.py`: `ENVIRONMENT`, `MODULES`, original_context,
  and resolve_native_api preserve the target route. The resolver is too broad
  for the cache diagnostic: it verifies all SITE files, imports packages, and
  checks tokenizer assets. Its audit hook is after interpreter startup.

## Then the finite loader operation

Once caches are reviewed, the next source extraction is limited to existing
authenticated Torch/vLLM wheel member evidence and these initialization routes:
Torch `torch/__init__.py`; vLLM `vllm/__init__.py`, `vllm/_custom_ops.py`,
`vllm/platforms/__init__.py`, `vllm/platforms/cpu.py`. These are proposed exact
source names, not yet newly pinned members; if an absent name or another helper
is required, return that finite dependency before expanding. Also extract the
saved member reference for `torch/lib/libgomp.so.1`. The current system candidate
already has authenticated 298776-byte payload SHA256
`d46f9225c1883039e8a6853e6d96ca1af11d034ce186a090952e4a7c8a7c2fdc`.

Reuse retained Torch wheel
`/mnt/nvme/dory-torch-cpython-archive-verification-r1-20261111/torch.whl`,
196253793 bytes, SHA256
`a09987c95ec4cffdb6df798d3d641558110a334cfccce22f2f046d83142bc260`,
and retained vLLM wheel
`/mnt/nvme/dory-official-cpu-20261108-v1/wheels/vllm-0.1.dev1+g75fc05a53.cpu-cp312-cp312-linux_x86_64.whl`,
132685835 bytes, SHA256
`37eae6e5e8f345b22d2c6147befa1c9f6b725c9fe88f83c336ae83f3184821f7`.
The exact-member evidence is in already-reviewed comparison records, so do not
download wheels or recompute all 48986 installation files to obtain it.

Produce a small 15-edge table with separate RPATH/RUNPATH/SONAME/NEEDED fields,
original Torch-before-vLLM order, actual CPU branch, original preload order
(tcmalloc then Intel libiomp5), candidate content refs, and selection evidence.
The existing `parse_elf`/`dependency_candidates` merge and sort search candidates;
they deliberately do not prove loader order and must not be used as the final
selection oracle. Intel libiomp5 is not a substitute for libgomp.

Only then design one original-environment import confirmation. Reuse the
resolver's exact original route and later the existing 23+3-case native adapter
probe; add bounded dependency-binding observation, not a second environment or
forced loading of every ISA variant. Mapped-path presence alone cannot close
ambiguous edges. A tracing environment may supply diagnostic evidence but does
not substitute for the original-environment confirmation. Unselected extensions
stay explicitly unexercised; do not turn them into passed ABI branches.

No loader implementation file is yet required before the finite route sources
are read. Once those are known, a small guarded loader-evidence helper and its
reviewed native root driver are sufficient; there is no reason to rewrite the
frozen inventory or resolver. The current design makes no ABI/full-model/speedup
claim. Original model, BF16, original runtime, actual-exit, full-case and full-model
gates all remain mandatory.
