# Qwen package runtime v8: exact installed compiler resources

This is a source-only successor to frozen runtime v7. It does not establish a
successful compilation, a completed scientific stage, or a model speedup.
Preparer v7 adds source-only candidate support for this exact runtime directory.

## Observed failure and provenance

The actual r5 run exited with code 1 (tool session 81683, terminal 0cdf4a).
Its recorded compiler unit contained the already-supported
`sg00/constant.21.npy -> ../constant.21.npy` link and sixteen absolute links to
the installed `neuronxcc/pwp/pwp_bin_trainium` directory. The sixteen names are
`act_info.json` plus `.json`, `_bkt.bin`, and `_ctrl.bin` for each of
`derivative_silu_and_others`, `natural_log_exp_and_others`,
`sigmoid_and_others`, `silu_and_others`, and `softplus_and_others`.

After the root agent observed actual exit and physical drain, it authorized
bounded stable nofollow reads of these sixteen files only. Observation
`qwen35-fixed-sdk-pwp-assets-observation-20261112-v1.json` records six JSON
assets and ten binary assets: 370,753 logical bytes and 393,216 allocated bytes.
All sixteen hashes and sizes agree with the installed wheel RECORD for
`neuronx-cc 2.23.6484.0+3b612583`. That audit read bytes without loading or
executing the assets. It is separate from the synthetic tests in this delivery.

The installed activation/link generator components are compiled `.so` files;
equivalent Python source was not found. Their bytes were neither read nor
loaded. The evidence establishes observed links, installed file identities,
content hashes, and wheel RECORD agreement. It does not prove the generator's
internal implementation.

## Exact allowance and storage behavior

`compiler_assets.json` is independently pinned by its literal SHA-256 in both
runtime and preparer. Only its sixteen exact absolute target paths are allowed.
Compiler links must use the existing model/bucket/compiler-unit/subgraph path
form and exactly the target basename. Relative aliases, other SDK files,
arbitrary absolute links, hardlinks, special files, and ancestor symlinks fail.

Directory file descriptors are opened from `/` with `O_NOFOLLOW`. Every
compiler directory from the declared work root through the link parent must
belong to the current user and remain on the same device. Directory edges,
link identity and target identity are rechecked before accepting an observation.
Each installed asset is a regular singleton file and must match its complete
recorded stat identity and bounded SHA-256 content read. Missing or changed
installed targets are permanent failures. The original limited retry mechanism
applies only to compiler-tree churn in its existing allowed window.

The existing relative `constant.N.npy` branch is unchanged and remains
metadata-only. Each new SDK link is charged normally. Each unique external
target is charged once per complete scan, including bytes and entries, to the
existing compiler-workdir limits. Shared targets are never charged once per
bucket. Link observations and every unique target's content are checked again
after complete inventory. All original class, single-file, entry and global
limits remain in force.

`allocated_bytes` includes the conservative existing-SDK charge.
`output_allocated_bytes` excludes it; `existing_sdk_allocated_bytes` reports
the difference. Remaining NVMe reservation subtracts only actual output bytes.
Thus preexisting files on the root disk cannot falsely reduce required NVMe
free space. Neither root nor NVMe headroom requirements are increased or relaxed.

## Preparation and preservation

Preparer v7 accepts the explicit runtime v8 directory and requires the new
module and manifest in its source closure. It authenticates the small authored
JSON manifest, checks all sixteen target identities with the injected stat-only
reader, and rechecks them before returning. It never reads SDK asset contents,
and never puts those asset paths into source or JSON-input maps. The existing
six binary inputs, model stat checks, full historical source/input preservation,
catalog prefix, finite metadata caps, and false candidate authorization remain.

Runtime prelock verifies all sixteen asset content pins before resources or
lock acquisition. Subsequent scans verify the exact linked targets. Frozen
runtime v7 and preparer v6 files remain unchanged. Existing science, terminal
failure preservation, lock, service stop, compiler drain, cleanup and stage-25
external-exit requirements are preserved. `source_diff.patch` and
`ast_preservation.json` identify the complete production delta.

## Validation and remaining work

Actual successful stdlib/mock test invocations:

| Test selection | Tests | Actual exit chunk | Exit |
| --- | ---: | --- | ---: |
| SDK assets and imported compiler-storage controls | 36 | 155714 | 0 |
| Existing compiler links/terminal, source, environment, host and source text | 98 | 331b7a | 0 |
| Updated predecessor preservation controls | 4 | c3101e | 0 |
| Preparer v7, including eight new v8 controls | 57 | 3e98df | 0 |

All SDK payloads used by tests are synthetic temporary files; the real small
authored manifest is the only new asset metadata read by preparation fixtures.
No current candidate, History resolver, actual prelock, original lock, native
import, compiler, model or device was invoked by this source delivery. An
initial preparer test run had one expected-error message regex mismatch;
the correctly raised failure was retained, its test was corrected, and all
57 tests then passed. The initial output remains recorded.

Independent root review and explicit current candidate/budget/history bindings
are still required before a future run. This delivery neither chooses a current
serial catalog nor starts an experiment or promotes an optimization result.
