# Explicit Qwen package candidate preparation v2

This delivery adds a source-only candidate preparer, not a launcher. No current
catalog/handoff was selected, no actual preparation was invoked, no model or
graph payload was opened, and no runtime/spec/review/continuation file was
changed by this delivery. Its tests use a synthetic metadata graph and bounded
temporary JSON fixtures only.

`prepare_qwen35_package_serial_candidate_v2.prepare(request, read_metadata,
read_source, identity_only, guard)` returns exact overlay/spec/review objects,
bytes and SHA-256 references. It never writes them. Both the returned candidate
and returned review have `passed=False` and `launch_authorized=False`.

## Request interface

All fields are mandatory; there are no implicit current-state lookups:

| Field | Value |
| --- | --- |
| `seed_spec`, `seed_review`, `seed_actual_exit` | Exact `{path,sha256}` refs to an executed, independently accepted sampler seed and actual successful exit. Seed spec/review are preparation-only envelopes, at most 64 MiB. |
| `current_catalog`, `idle_handoff` | Then-current root-selected refs. Every old catalog row remains an exact ordered prefix; SDK binding is unchanged, all later rows are retained, and the actual seed tool/owners/units row must agree. |
| `runtime_directory` | Exactly `Q/qwen35_package_serial_runtime_v2` or `Q/qwen35_package_serial_runtime_v3`. |
| `runtime_delivery`, `runtime_review` | Explicit exact refs. Review schema is `qwen35_package_serial_runtime_v{2|3}_root_source_review_v1`; root/source-only passed, runtime not admitted. Its `source_delivery` must equal the requested ref. |
| `resource_budget` | Root-reviewed exact six-tree resource-budget ref, retained as ordinary JSON in spec/overlay inputs. |
| `case` | Complete original package case, including unchanged Neuron configuration and four explicit package paths. |
| `binary_inputs` | Exact six runtime `BINARY_PATHS`, each `{kind:"binary_payload",identity:{realpath,device,inode,size,mtime_ns,ctime_ns}}`. These paths are statted only. |
| `preparation_sources` | `path -> sha256` for the preparer itself and every preparation/review helper used by root. These text sources are authenticated. |
| `required_helpers` | Nonempty explicit list of root helper paths; each must occur in `preparation_sources`. |
| `extra_metadata` | Explicit list of additional ordinary JSON refs; use `[]` when none. No large-envelope exception is granted here. |
| `future` | Exact fields `run_name`, `output_root`, `overlay_output_root`, `running_handoff_path`, `overlay_path`, `spec_path`, `review_candidate_path`. No output path may overwrite a retained input/source/old handoff. |

`output_root` is `/mnt/nvme/neuron-work/qwp-<run_name>`; the original control
root is `/mnt/nvme/neuron-work/qw-<run_name>`. All six RPC paths must fit the
original 107-byte limit. Overlay output remains an explicit fresh direct child
`/mnt/nvme/minicpm5-layer1-operator-native-<suffix>`. Output metadata paths are
distinct direct children of `Q`; the running handoff uses the existing
`continuation-handoff-*.json` namespace.

## Authenticated reader interface

`StableReaders(guard)` is optional root-side infrastructure. Its construction
does no IO. Use `readers.metadata(ref, cap)` and `readers.source(ref, cap)`;
`file_identity(path)` performs stat only. The preparer supplies every cap.

The metadata reader returns `{value,identity}` after full SHA authentication,
duplicate-key/nonfinite rejection and stable before/open/after identities.
The text source reader returns `{payload:bytes,identity}`. Previously observed
paths cannot be rebound to a changed hash or identity. The final preparation
pass rechecks all observed identities. Injected readers are a trust boundary:
production callers must authenticate complete bytes, as `StableReaders` does.

Only seed spec, seed review and seed-derived base plan receive 64 MiB
preparation-only envelopes. All ordinary receipts, catalogs, overlays, source
manifests and text-source reads retain the original 8 MiB cap. Generated
spec/review envelopes retain 64 MiB, while generated overlay retains 8 MiB.
No old binary/source base inputs are opened by this helper.

## Source and evidence preservation

The recovered source partition is `seed.source_sha256` minus the executed seed
overlay's `source_extension`. It includes recovery sources that are absent
from the original base plan. Deriving it from base sources alone is incorrect.
The preparer merges every seed source, every current catalog row's complete
source increment, runtime delivery files, runtime input pins, frozen science
delivery and both science input/capture manifests, plus preparation helpers.
Conflicting SHA pins fail; no source is replaced or silently discarded.

The base plan's entire original input map remains referenced verbatim,
including its raw/binary receipts. Existing seed/overlay JSON identities are
preserved exactly. A `.py` or other non-JSON extension in the selected seed's
`input_extension` is rejected, not dropped or migrated. The accepted r10 seed
route retains its complete 199/211 JSON extensions and all base inputs; later
slot helper evidence remains in the complete current catalog, historical refs
and all source additions.

The preparer imports the frozen v2 *pure contract* for structural checks even
when the selected runtime is v3. Preserve these import-source pins too:
`qwen35_package_serial_runtime_v2/{__init__,contract}.py`,
`qwen35_sampler_pair_protocol_v1.py`, and
`qwen35_package_full_validation_v2/{__init__,runtime}.py`.
They may already be in the merged history/delivery; otherwise add them to
`preparation_sources`. Version 3 admission is valid only with independent root
review of the source successor retaining that same contract/science.

## Returned artifacts and root integration

The return contains `{overlay,spec,review}`, their matching `*_bytes` and
`*_ref`, `catalog_source_extension`, seed/preparation references, observed
metadata/source identities and preservation counts. Bytes include real
newlines. Do not serialize the entire return as an ordinary receipt: it
contains bytes and multiple large full-source envelopes. Root should publish
only the explicit artifacts using its existing exclusive/no-overwrite writer
with the correct envelope limits, and separately record preparation evidence.

`catalog_source_extension` is the merged map minus the recovered base
partition, identical to the new overlay increment. Use that bounded increment
for a future catalog row, rather than duplicating the roughly 48,000-entry
complete spec map in every catalog row.

The only temporary true review is local in-memory structural validation of the
spec with the frozen contract. It is deleted and never returned or published.
The returned false review cannot pass runtime admission. This preparer does
not claim full History resolution, physical freshness, free resources, free
lock, actual child ACK, any of the 25 native/scientific stages, or stage-25
independent qualification. Root still performs its independent exact binding
review and the unchanged runtime performs all original gates.

## Validation

Run `python3 -B test_prepare_qwen35_package_serial_candidate_v2.py` in this
directory. Tests block native imports, real preparation IO, subprocess/system
calls and lock acquisition; only the separately named stable-reader tests use
small temporary JSON files. They cover v2/v3, variable catalog lengths,
original failed-row preservation, recovery-only source retention, immutable
metadata, exact caps, six stat-only binaries, original configuration, complete
helper/source closures, mutation failures and refusal of candidate admission.

## Frozen v1 successor correction

Version 1 remains frozen with its original failed-boundary evidence. Version 2
adds two checks without changing scientific gates, budget policy, source/input
preservation or false-review semantics:

1. Every parsed JSON float must be finite, including exponent overflow such as
   `1e999` and nested `-1e999`. The same check applies to injected metadata
   reader results. Existing NaN/Infinity token rejection is retained.
2. Future overlay/spec/review/handoff paths may not collide with any absolute
   `{path,sha256,...}` reference retained anywhere in the request or read
   metadata. This includes unread recovery/native-manifest refs, old/current
   completed-SDK refs and all catalog terminal refs. Sized/annotated refs and
   absolute lexical aliases also count; there is no filesystem resolution,
   reference rewrite, extra historical read or dropped evidence.

The updated author suite has 32 tests. The separate reviewer-authored
`check_qwen35_package_preparer_independent_v1.py` is unchanged and also passes
all 10 tests when explicitly targeting the v2 candidate module. These are
source/mock results, not actual History, native or root admission results.
