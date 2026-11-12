# Fixed special-value diagnostic implementation

The implementation is ready for root source review and a separately bound
serial run. No compiler, device, lock, service or current runtime binding was
used while preparing it. Actual identity compilation and six device calls are
still pending; source/mock success is not native correctness.

The source inventory is
`qwen35-slot-special-value-serial-source-delivery-20261112-v1.json`, 603,746 bytes,
SHA256 `74629180e32ada9ddb7ca449e7855251c5fd833a03f546554cd5baa15e2203bb`.
It records six new files, 623 local source files and 2,822 installed SDK Python
sources, plus the unchanged original primitive manifest and three required
original-failure JSON inputs. The previously frozen identity source and input
generator are included in its static source closure.

The fixed leaf executes reference/original_raw, reference/negative_nan_raw,
candidate/original_raw, candidate/negative_nan_raw, identity/original_raw and
identity/negative_nan_raw in that order. It loads the two existing immutable
model.pt files, compiles only the explicit three-tile NKI identity DMA, and
keeps at most one graph object live during the six calls. Ordinary byte
mismatches are persisted and do not skip later calls. A native/guard/storage
exception aborts into the original cleanup and preserves previous captures.

Each profile saves its complete cache, updates and int64 slots before native
imports. Each completed call saves all 98,304 output bytes and every changed
byte offset as little-endian uint32. The final validator rereads every saved
byte and reconstructs the entire offset list; a matching count or first-16
preview alone cannot pass. Artifact SHA256/size and filesystem identity are
checked across loading, calls and final validation. Loaded forward code must
contain the actual Neuron forward_v2 operation. Identity additionally retains
the whole HLO, verifies its custom-call markers and unique decoded NKI backend
config for `qwen35_conv_identity_dma`, and requires zero MACs/no collectives.
This HLO inspection is explicitly marker/config inspection, not a claim that
a compiler instruction stream was independently verified.

The new serial contract has the same original CPU, memory, storage, wall-time,
SDK environment and service properties. The old fields that described eight
maximum processes/graphs remain in that unchanged resource policy; the new
workload separately declares one diagnostic process and exactly six calls.
`capture_layout`, process/group checks, source guards, source-history transport,
Service/History/RetainedLock ownership, ACK and cleanup are reused. The inner
leaf guard and exception context restoration are identical to v3. The outer
finally differs only in its diagnostic receipt schema and an explicit false
qualification field. Frozen v3 and its eight-layout validator are unchanged.

Root binding uses the same spec field set with these fixed substitutions:

- schema: `qwen35_slot_special_value_serial_spec_v1`;
- runner: `research/qwen35_slot_special_value_serial_v1.py`;
- output root: `/mnt/nvme/neuron-work/qwen35-slot-special-value-<run_name>`;
- service: `trn1-qwen35-sampler-<run_name>-slot-special-values`;
- work child: `conv-special-values`;
- workload: `qwen35_slot_special_value_protocol_v1.workload()`;
- source_delivery: the two-field path/SHA reference to the inventory above;
- primitive_manifest: unchanged original contract.MANIFEST;
- limits: unchanged original contract.LIMITS.

Merge the complete then-current historical source map with the inventory's
local/SDK maps, rejecting conflicts and preserving all old pins. Include each
`diagnostic.INPUTS` JSON in input_extension with its full current ordinary-file
identity and fixed SHA. Existing binary model.pt files are immutable payload
inputs checked by the diagnostic; do not pass them to History's JSON metadata
reader. Root selects its then-current complete catalog, overlay, LFM scope,
idle handoff and fresh running-handoff path. None is selected by these sources.
The latest supplied starting point was the 151-row v102 catalog/v569 idle;
later root publications must be respected when actually binding.

Use the original `-I -B -X pycache_prefix=<fresh output>/bytecode-never-created`
entry convention and original --spec/--spec-sha256/--review/--review-sha256
arguments. Root's actual tool/owner/running ACK is still required before any
native leaf release. Exit zero means the six-call diagnostic completed and all
captures validated; it does not mean all three arms matched. Per-arm/profile
`bitwise_matches` records those results, and qualification_matrix_completed,
full_model_qualified and model_speedup_claim remain false. The original r1
failure, 46,368-call qualification schedule and timing requirements remain.

Verification completed:

- 12 focused source/mock tests, actual tool chunk `a10a2a`, exit 0. They execute
  the real leaf loop with explicit fake SDK objects, not native imports.
  Reference mismatch continues to both candidate and identity; runtime failure
  stops with previous output intact. Missing calls, false equality, changed
  artifact, a corrupted middle difference offset, resource changes and a
  qualification claim are rejected. Full mismatch saves all 98,304 offsets.
- Original lifecycle/resource/guard/finally AST comparisons pass. Native import
  blocking covers torch/numpy/Neuron and ctypes/_ctypes. The identity source is
  AST-checked for fresh HBM allocation and exactly its load/store loop; it is
  never imported by these checks.
- Inventory preparation: actual chunk `32d792`, exit 0. Full source-only
  verify_delivery: `dc238f`, exit 0. Inert --help: `a7109d`, exit 0.
- Independent reviewer reran the 12 tests: session 73821, terminal `36ce09`,
  exit 0, and reported no identified source blocker. This is reviewer-reported
  tool evidence, not an independently observed native/device result.
