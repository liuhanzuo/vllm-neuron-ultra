KV256 diagnostic preparer v2: retain the existing typed historical plan
====================================================================

This is a source-only preparer successor. The frozen diagnostic runtime, leaf,
auditor, root helpers, workload and resource limits remain byte-for-byte v1.
No actual current request, catalog, handoff, runtime binding, real plan payload,
History, serial lock, native library, model artifact or device was accessed by
this task. Tests read source/old delivery metadata and use synthetic metadata.

Why the metadata preparation failed
----------------------------------

The root reported that v1 metadata preparation exited 1 at its final blanket
8 MiB input check, before publishing outputs or acquiring a runtime lock. An
already accepted seed retains this exact historical plan in input_extension:

- Path: research/minicpm5-layer1-operator-sdk-r2-20261111-plan.json
- SHA256: 2e5441f4b2c2ad2efd76192ae92c068c35d3ae9fc51c532062d4284692910632
- Bytes: 12803326
- Schema: minicpm5_layer0_runtime_inputs_v1

The path and digest were already fixed in the older
prepare_qwen35_sampler_v9_candidate_v3.py source. The byte count here comes from
the root's actual failed-preparation observation; v2 checks it against the
future authenticated reader and inherited identity. This source task did not
read the real plan or reinterpret its content.

The original runtime reader already permits this type
----------------------------------------------------

The frozen code chain is:

1. qwen35_package_preflight_history_v3.History.resolve checks each input's
   complete recorded identity and passes its exact path/SHA to Metadata.read.
2. qwen35_sampler_runtime_io_v12.Metadata.read uses the existing
   minicpm5_layer1_operator_sdk_sources_v1.load reader, with stable before/after
   identities. It deliberately preserves inherited typed large metadata.
3. operator_sdk_sources_v1.envelope_kind recognizes the operator-sdk research
   plan namespace. text requires an expected SHA, calls
   metadata.stable_bytes(path, metadata.ENVELOPE_LIMIT), verifies the complete
   SHA, parses JSON and calls envelope_shape.
4. nemotron9b_package_metadata_v27 defines ordinary LIMIT = 8 MiB and named
   ENVELOPE_LIMIT = 32 MiB. stable_bytes enforces size and stable dev/inode/size/
   mtime/ctime before returning the complete payload.
5. The plan branch requires its exact schema, complete bounded source map,
   source-contract reference, latest_completion dictionary, output_root string,
   and complete inputs dictionary with at most 100000 entries. The original
   source-pin checker also requires literal canonical absolute paths and SHA256.

The reader source proof pins every file in this chain. Tests execute only the
selected authenticated pure reader functions with synthetic callbacks; the
original modules and their transitive native/lifecycle imports are not loaded.
An in-memory JSON larger than 8 MiB demonstrates the original 32 MiB typed route,
while a wrong digest, arbitrary untyped path or malformed schema still fails.

The seven exact source edits
----------------------------

The derivation script authenticates the frozen v1 preparer, applies seven
literal edits, and proves that reversing them recreates every original byte.
The edits update SELF; declare the one fixed inherited-plan rule; add pure
shape/input checks and a CLI source-authentication check; require a separate
v2 helper delivery contract; replace the final blanket input cap check; record
the retained typed input; and verify the executing preparer's pin before prepare.

The exception is narrower than the existing general 32 MiB typed reader. V2
accepts only the exact fixed path, SHA and 12803326-byte size above, and only
when the complete input row is already present in the accepted seed and equals
the freshly authenticated base_plan row. The complete realpath, device, inode,
size, mtime_ns and ctime_ns must match; v2 never refreshes an inherited identity.
It rechecks the original typed plan branch without projecting or dropping any
plan data. It cannot newly promote another large preparation envelope into a
runtime input, even when that envelope has a valid typed filename.

Every other runtime input remains at most 8 MiB. Even a smaller forged size for
the fixed plan is rejected. Existing seed/review/base-plan preparation transport
bounds are unchanged; v2 adds no general 32 MiB or 64 MiB runtime allowance.

The plan remains in spec.input_extension and in all inherited source mappings.
Candidate preservation records its exact inherited row separately, so it is no
longer mislabeled as a preparation-only input. All original failed statuses,
catalog prefix, input identities, source maps, full History/LFM requirements,
unadmitted review draft and false qualification flags remain intact.

Separate delivery and review
----------------------------

Future preparation must invoke prepare_qwen35_kv256_alias_diagnostic_binding_v2.py
with an explicit future request. The runtime spec/workload schema stays v1.
The request must reference the independent v2 helper delivery and a root source
review of that exact delivery. No actual future request is supplied here.

The v2 helper delivery uses schema
qwen35_kv256_alias_diagnostic_binding_source_delivery_v2 and must include the
exact predecessor_preparer, retained_input_exception and source_sha256 fields.
The original v1 helper pin stays in the map, as required by the frozen runtime.
The actual v2 SELF SHA is checked from source before candidate preparation, and
the complete helper mapping is merged into the future spec before any launch.
The existing full runtime source review remains authoritative; accepting this
preparer source does not admit runtime work.

Validation
----------

V2 passes 46 source/synthetic tests: all original 27 binding controls run against
v2, plus 19 tests for the exact exception, full identity/sha preservation,
arbitrary oversized/changed input rejection, original reader behavior, exact
source reversal, helper pre-enrollment and inert imports. The unchanged v1
suite also passes its original 27 tests. A synthetic reconstruction reproduces
v1's size rejection and passes v2 without modifying the input records.
