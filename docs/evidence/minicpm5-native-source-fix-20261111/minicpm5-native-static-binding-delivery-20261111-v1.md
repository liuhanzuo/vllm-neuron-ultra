# MiniCPM5 lifecycle static binding: source-only delivery

The successor removes repeated JSON serialization and output-catalog rebuilding
from normal outer/coordinator/native lifecycle guards. It does not change model
execution, qualify a model result, or claim an inference speedup. No admission,
plan, terminal receipt, lock operation, service operation, raw payload read or
actual model run was performed for this delivery. Existing executed sources and
the active serial handoff were not modified.

## Integration

- `research/minicpm5_native_static_binding_v1.py`: authenticated immutable output
  catalog, fixed output root, and immutable records of every binding-consumed
  dict/list member and scalar. Shared JSON containers are recorded once. Normal
  mutation/rebinding/deletion of the binding object is rejected.
- `research/minicpm5_layer1_operator_native_lifecycle_core_v3.py`: original v2
  lifecycle and resource constants, with explicit `static_binding` consumption.
  The compatibility path without a static binding retains the old full check.
- `research/minicpm5_layer1_operator_native_process_adapter_v9.py`: v8 loaded
  source verifier and original v7 live methods, with `StaticPortsMixin`.
- `minicpm5_layer1_operator_native_runner_v12.py`: v11 package-coverage and early
  dependency gates, with the new adapter/core and native usage binding.
- `minicpm5_layer1_operator_native_outer_runner_v5.py`: v4 outer with adapter v9
  and explicit bound monitor usage. Release, lease, service and cleanup remain.
- `research/prepare_minicpm5_layer1_operator_native_v6.py`: v5 package inventory
  preparation, with adapter v9, runner v12 and outer v5. No preparation executed.

The full static facts check occurs in `usage` after each live walk, at the same
point where the original lifecycle serialized and hashed the plan. The walker
performs a cheap root-identity check and consumes the previously authenticated
immutable catalog and fixed output root. A mutation during the walk is rejected
before the guard returns. This avoids adding a second full facts traversal per
guard while preserving the original validation frequency.

Facts cover the entire contract and resolved plan, both complete catalogs,
digests, and the four flags used by the original lifecycle binding. Equal-value
container replacement is rejected. Scalar types and signed floating zero remain
distinct. Unsupported JSON values and cycles are rejected at construction.
Resolved historical branches outside the original binding are not copied or
traversed. The existing v3 source-only authentication phase retains its separate
immutable context and full entry/exit alias checks; it restores the normal guard
before model work. The new helper does not freeze the driver's plain dict API.

## Full binding boundaries and source gates

The full original lifecycle binding remains at these concrete call sites:

- StaticBinding constructor: helper line 94, before sealed catalog activation.
- Explicit full-boundary verifier: helper line 120.
- Coordinator entry: core v3 line 182; successful cleanup/exit boundary line 232.
- Native result wrapping: core v3 line 153, called by runner v12 line 111 after
  the final loaded-source verification and normal guard at runner line 109.
- Outer terminal entry: core v3 line 244; complete successful inventory at line
  267 invokes helper lines 129–134, including full binding before success.

The existing source-only authentication phase also retains its original full
entry/exit bindings. Failure paths retain the original cleanup behavior and do
not need to manufacture a successful terminal boundary.

Loaded-source early-check evidence is preserved from the sibling implementation:

- Runner v12 lines 101–103: actual loaded-source callback, complete selected
  package filename coverage, full original source authentication, loaded check.
- Driver v5 line 21: package coverage before native imports.
- Driver v5 lines 25–29: exact native Llama module import/path check, loaded-source
  authentication at line 28, then model allocation at line 29.
- Driver v5 lines 30–35: retained native module identity and loaded-source check
  at line 33 before the unchanged scientific capture.
- Runner v12 line 109: loaded-source check again after scientific driver return.

BF16, the original 88-frame gates, native 24 frames × 3 arms = 72 captures,
936 raw outputs, every source hash, metadata check, native identity, live writer,
PID/lease check, storage floor, time bound and resource limit remain unchanged.
The walker and inventory validator are the existing reviewed functions;
independent AST comparison with adapter v7/catalog v2 confirms the only omitted
operation is rebuilding the already authenticated catalog.

## Validation

`/usr/bin/python3 -B research/check_minicpm5_native_static_binding_v1.py`:
16 tests passed, actual tool exit 0, final chunk `bef0f0`, 3.308 seconds.
Tests cover nested mutation, equal-copy rebinding, exact scalar types, signed
zero, cycles/shared containers, immutable private state, mutation during walk,
fixed output root, existing source-phase compatibility, live writer identity,
resource caps, entry/exit full binding and original dynamic callback order.
The complete 936 raw / 144 metadata / 97 directory fixture gives the same
validation result as the original implementation; partial/missing/short/extra
output and pending-writer controls retain their outcomes.

AST checks preserve all lifecycle bodies except explicit static usage and the
additional coordinator exit full binding. Runner, outer and preparer differ
only in the listed imports/entry paths/static-binding argument. The original
driver is unchanged by this task. Four normal guards retain identical lease,
storage and walker events while reducing catalog builds from 12 to 0 and plan
digest calls to 0 within that loop. Full boundary checks still rebuild/hash.

The original bounded local-import collector, executed only as source definitions,
finds 593 local source files / 7,222,566 bytes for prep v6 + outer v5 + runner v12.
This stays within the unchanged 2,048 files / 64 MiB parser limits. No imported
framework or production orchestration package was executed for this check.

## Synthetic overhead measurement, not inference performance

The benchmark uses 44,503 synthetic source pins and 8,791 input records with
mocked live IO, alternating original/new guards for 9 iterations. Median wall
time for this in-memory harness was 0.0920241466 s before and 0.0395111674 s after,
an old/new ratio of 2.32906676. The final benchmark tool exit was 0, chunk
`bcc0f2`. This measures only repeated static binding/catalog work. It excludes
actual filesystem/process/metadata checking, model execution, token latency,
and end-to-end experiment duration. It must not enter a model speedup table.

## SHA256

| File | SHA256 |
| --- | --- |
| research/minicpm5_native_static_binding_v1.py | 53afd779f775948b95cc3fde8a2e45a8d473ca5a5c813b6262db7a5d680a8983 |
| research/minicpm5_layer1_operator_native_lifecycle_core_v3.py | d3048c6036ce715337a28eebb78a7a239caa387182fe34c0201565fd78f9a824 |
| research/minicpm5_layer1_operator_native_process_adapter_v9.py | e2c9c023da746db40a7a4037220b6b50318f400ab4962c24e6b7683eb53957d9 |
| minicpm5_layer1_operator_native_runner_v12.py | 7c16738daa36a94ea01fb0b723a3272f90f4da25a8a5807233fba6746a602b9a |
| minicpm5_layer1_operator_native_outer_runner_v5.py | 4f347763babe48489674883fd281c45f15b5858eb2808f32d73f806bb473ec05 |
| research/prepare_minicpm5_layer1_operator_native_v6.py | 369241b70d98908a94f35a71be96d87ca7f2c3cbd589e354c3f0b09434e646aa |
| research/check_minicpm5_native_static_binding_v1.py | 31d8f3700d2c91db5e454369b9f2830b6f6df06c0d6d8f74035d2938c751553a |

These are source delivery facts, not an accepted runtime source review or launch
authorization. Future integration still needs the original independently
authenticated source closure, concrete preparation/review, and serial admission.
