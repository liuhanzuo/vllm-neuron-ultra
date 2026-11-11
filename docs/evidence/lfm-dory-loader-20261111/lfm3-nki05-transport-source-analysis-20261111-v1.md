# LFM NKI 0.5 transport: source-only analysis before the actual result

The strongest source-level candidate is **input writeback after the simulated
kernel has finished**, even for the identity kernel. The input-writeback helper
explicitly selects non-precise dtype conversion, although the simulator context
uses `NKI_PRECISE_FP=1`. Returned outputs use a different, explicitly precise
path. This establishes an unnecessary BF16 → FP32 → BF16 conversion on writable
host inputs. Whether that conversion changes this fixture's NaN payloads is not
proven by Python-source reading alone.

No simulation, framework/native import, runtime output read, raw data read,
service/lock action, spec/handoff mutation, or executed-source edit was performed.
This report proposes no successor change; it awaits root's actual result.

## Authenticated source scope

Diagnostic source:
`/home/ubuntu/experiments/trn1-latest-models-20261031/lfm3_nki05_transport_diagnostic_v1.py`

SHA256: `62fcdb4faba6c26420f2a5d29123b704b297216f7f4ae6d3c916cca4d547fb55`.

NKI source root, abbreviated `NKI/` below:
`/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/nki/`

All eight installed Python files in the diagnostic's frozen `PINS` were read as
source and their current SHA256 values matched the pinned values:

| Source | SHA256 |
| --- | --- |
| _simulator.py | 3c28f10492ef8c6c5413acbd811c059586f7ff16d3ef795430e794c57ed8f119 |
| _backends/simulator/dtypes.py | 260a01643e5158647503d41a2012b5607523f522885b2a7c6bd4ba8cd3fedd4e |
| _backends/simulator/alloc.py | 40abe6512cc6884a8fec8c9ffa6d1a7bfdf088babe54a6873ca4bb622b9a8f8c |
| _backends/simulator/copy.py | 399389892965cea919dac81ffce900b3ee4820dc6347aa6d2d0550be7db58e87 |
| _backends/simulator/tensor_view.py | b378f0634c591ec8e48b1eecea2c2a77cde7f5ebdba4b19997098644c63cde81 |
| _backends/simulator/advanced.py | 8f7bec4e5712467e3c4b22ef323aae1ad688b178b34f0f6bec309f7bbb82d3b4 |
| framework/_simulate.py | 187dbb5779c8b57b5012ceb013ad9845ed18b864be9e7c813a92998481a40201 |
| language/tensor.py | 66e6c75be3a340475b1affa1b69b820e500fa2953159c13884ce0d4199926691 |

Public API forwarding files outside those eight pins were additionally read to
follow the call path. Their hashes are recorded at the end as observations;
they are not represented as additions to the frozen admission/source pins.

## Actual public API paths

**Simulator ingress and return.** `framework/_simulate.py:129` converts arguments
through `SimulatorFrameworkContext`, then calls `simulate_kernel` at line 134.
NumPy arguments pass through unchanged at lines 62–74; no Torch argument
conversion applies to this diagnostic. `_simulator.py:133` reads precise mode
and sets the context. `SimulationNkiTensor.from_numpy`, lines 96–119, allocates
the matching inferred dtype and uses `np.copyto` into storage. With the precise
context, `dtypes.py:64` maps BF16 to `ml_dtypes.bfloat16`; `alloc.py:58` consumes
that mapping. No obligatory FP32 ingress conversion is visible in this path.

**Load/store and assignment.** `nl.load` is exported from `language/_core.py`.
At lines 155–160 it requires an HBM source, defaults to the source dtype,
allocates an SBUF tile, and calls `dma_copy`. `nl.store` at line 172 likewise
calls `dma_copy`. For same-dtype data, simulator `copy.py:97–100` performs no
FP32 cast; its cast is conditional on source/destination NumPy dtype mismatch.
`NkiTensor.__setitem__`, `language/tensor.py:985`, routes HBM movement to DMA and
SBUF-to-SBUF assignment to `tensor_copy`. Simulator `copy.py:149–158` also casts
only when the destination dtype differs. Thus `prior[:, :] = loaded` and the
corresponding replacement assignment retain BF16 as written.

**Selection bookkeeping.** `nl.where` in `language/_math_funcs.py:268–303` is
implemented as floating-point masks and arithmetic: `(condition * x) +
((1-condition) * y)`. It is not a raw-bit selector. Here it receives only
candidate row numbers and the previous selected row, with explicit `nl.int32`
output. Those values lie in 0–16, so the FP32 intermediate exactly represents
them. BF16 cache/update payloads never pass through `nl.where` in this kernel.
Invalid slot values are compared against destinations 0–15; candidate selection
still uses only bounded row numbers. Iteration with candidate `row + 1` makes
the last matching row win; no match leaves zero and a false predicate.

**Dynamic update row.** The exact call at diagnostic lines 135–136 matches the
installed `NkiTensor.ap` signature at `language/tensor.py:1145`:
`pattern=[[width, 1], [1, 128]], offset=column, scalar_offset=source_row,
indirect_dim=0`. HBM permits scalar indirection along dimension 0; the explicit
rejection at lines 1199–1202 concerns on-chip partition indirection. The
indirect stride computed at lines 1223–1228 is `width` with unchanged dtype.
`_build_indirect_shape` at line 1441 preserves the effective `(rows, width)` HBM
shape. Simulator `tensor_view.py:91–111` therefore addresses
`source_row * width + column + element`, for elements 0–127. The no-match row is
clamped to zero, a valid row because the diagnostic requires at least one update.

This is scalar indirection: `isa/_copy.py:157–159` selects the separate
`dma_copy_indirect` route only for vector offsets. This scalar view takes the
normal backend DMA path at lines 207–217, which receives the scalar offset
through `tensor_ap_view`. The API name does not imply that scalar access must
use the separately named vector-indirect backend operation.

**Predicated payload copy.** `isa/_advanced.py:705` accepts the exact keyword
arguments used in the diagnostic and enforces matching source/destination
dtypes and an unsigned predicate. The diagnostic supplies BF16/BF16 (or
FP32/FP32), a `(1,128)` uint8 predicate, and SBUF tiles. Simulator
`advanced.py:397–407` reads the predicate as bool and uses
`np.where(pred, src_data, dst_data)` without an explicit payload FP32 cast.
NumPy/custom-dtype bit preservation still requires the actual experiment.

## Observer coverage and its limits

Diagnostic lines 168–171 obtain both
`tensor_ap_view(tensor).get_data()` and `tensor._storage.data`, immediately
snapshotting their bytes and dtypes. The observed ingress arrays, loaded SBUF
tiles, assigned tiles and full pre-return output are complete allocations.
The dynamic partial update view itself is not used as a whole-allocation
observer input.

`language/tensor.py:1480–1519` constructs a backend view from the same storage,
shape/stride pattern, dtype and offsets. Simulator `tensor_view.py:193–220`
uses `StorageHandle.data`, whose field is declared in `alloc.py:25`. The normal
matching-dtype `get_data` path at `tensor_view.py:398–415` performs integer-index
selection and reshaping, without an explicit floating-point conversion. View
bytes and direct backing bytes therefore distinguish view extraction from an
already-altered allocation, subject to actual NumPy behavior.

The observer snapshots stop at `pre_return`. They do **not** observe the later
host-input writeback. `first_boundary_divergence` is defined only over observer
events, so it can be null while `inputs_unchanged` is false.

## Strongest first-failure candidate: post-kernel input writeback

The exact source chain is:

1. `_simulator.py:24`: `_tensor_to_numpy(tensor, precise=False)`.
2. `_simulator.py:45`: it passes that explicit `False` to
   `to_numpy_dtype(tensor.dtype, precise=precise)`.
3. `dtypes.py:79–92`: only `precise is None` uses the current simulator context;
   explicit false selects the map where BF16 becomes NumPy FP32 (`dtypes.py:49`).
4. `_simulator.py:46–47`: BF16 storage is therefore converted with
   `data.astype(target_np_dtype)` when called with the default.
5. `_simulator.py:213–228`: every writable input is unconditionally extracted
   with `_tensor_to_numpy(tensor)` and copied into the original host array.
   Diagnostic operands are writable copies, so this includes unmodified cache
   and updates. Copying FP32 into those BF16 arrays casts them back.
6. `_simulator.py:252`: returned results separately use `precise=True`, avoiding
   that FP32 extraction path.

The docstring at `_simulator.py:30–31` says false respects the environment, but
the explicit bool supplied to `to_numpy_dtype` does not. The source establishes
this dtype-path mismatch independently of runtime behavior.

The first diagnostic operation is identity, before load/copy/gather/selection
(`diagnostic.py:288–289`). The fixture source
`lfm3_batch_slots_extended_v1.py:17–26` writes noncanonical NaN payload encodings
`0x7fc1`, `0x7fff`, and `0xffc3` into cache and updates. If the writeback round
trip changes any of those payloads, the expected signature is:

- Identity observer coverage is complete.
- View/backing observer failures and returned differences are zero.
- `first_boundary_divergence` is null.
- `inputs_unchanged` is false, making `passed` false at diagnostic line 208.

This signature would localize the problem outside kernel movement, but is a
prediction, not an observed result. The Python files do not establish the exact
NaN conversion behavior inside NumPy/ml_dtypes native implementations. No
particular failing word, count or payload replacement is claimed. If an earlier
ingress/view failure is observed, it takes precedence over this hypothesis.

## Additional observed API source hashes

| Source outside the eight frozen SDK pins | Observed SHA256 |
| --- | --- |
| language/__init__.py | 48691d3128bbad91e5d0110cdbec65242ddae7ff014516739ca2a424caf6c253 |
| language/_core.py | 09c72c612b9b9a06bacb9ecaacdbb56fd28b6362df788907c54f01214929bdc7 |
| language/_math_funcs.py | d3d985e6eab4d5144d42c49f46b1faa5c995b61d83fcddc4cd18e4f4067bb559 |
| language/_ops.py | 2f47b1b8b2ad2492c5f864d6215010277a2adf31ab7362855f0dd3d6ff94b5db |
| isa/__init__.py | 9083cd5ea3b7e6690bd08c10507ce6ebb5c0c386c5b7c97c3932f59cc4240e6e |
| isa/_copy.py | 973d2ff840a4e48b3c9af917d363d102cab8514b402a9e9e38c89b5faea0ac42 |
| isa/_advanced.py | f37173b2ff8b2e4fd66bea40871ea5d2bb9deaa68c0ead4f2e2ab337dedfb203 |

The static path review does not prove Trainium device behavior, full selector
qualification, all-encoding parity, model quality or any speedup.
