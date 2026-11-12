# Qwen v10 bounded IPC path successor

The r7 v9 API failed because the installed vLLM constructed a 122-byte Unix
socket pathname, above the 107-byte limit. The observed failure was a natural
nonzero exit, not OOM. This delivery changes startup paths and their accounting;
it does not establish API/model success or a new speedup.

The actual installed selector is `vllm/utils/__init__.py:865`:

```python
base_rpc_path = envs.VLLM_RPC_BASE_PATH
return f"ipc://{base_rpc_path}/{uuid4()}"
```

`vllm/envs.py:456` uses the explicit `VLLM_RPC_BASE_PATH` environment value, or
`tempfile.gettempdir()` otherwise. Both actual files are already pinned by the
executed r7 spec. Their exact hashes are included in the v10 source manifest.
No installed vLLM file was changed or imported during this work.

The additive six-file runtime v10 makes these narrow changes:

- `control_root(spec)` is `/mnt/nvme/neuron-work/qw-<run_name>`. The model output
  namespace is unchanged. Distinct run names give distinct control roots. The
  original outer still rejects an existing control/output/handoff path, and
  original phase creation still uses exclusive directory creation.
- `ipc_paths(spec)` checks all six baseline/candidate capture/quality/HTTP
  paths, counting UTF-8 pathname bytes including the slash and UUID36, excluding
  the `ipc://` transport prefix. `validate_spec`, the preparation entry, and the
  server environment builder call this before model/API execution. A 107-byte
  path is accepted; 108 is rejected. For the illustrative name
  `r8-v10-20261112`, the maximum is 99 bytes. Merely using `r8` with the old
  `-runtime` directory would still produce a 111-byte candidate path.
- API child environments explicitly set `VLLM_RPC_BASE_PATH` to their existing
  phase `tmp`, equal to `TMPDIR`. No fallback `/tmp` tree or symlink is used.
  The pure environment function rejects inherited overrides and mismatched
  phase paths; all other scientific environment fields are retained.
- The original storage scanner rejected every socket. The successor allows
  only singleton UUID4-named sockets directly inside the six declared server
  `tmp` directories. Their allocated blocks, inventory entries, and size remain
  within the original temporary/class/whole-output limits. Other sockets,
  nested sockets, FIFOs, symlinks, hardlinks, oversized files, and cap violations
  still fail. Original bounded temporary-disappearance retries are unchanged.

History, service, and CLI v10 differ only in versioned references. The service
ownership/cleanup bodies, API failure collection, original 45-second cleanup,
root ACK deadlines, phases, native scientific environment, memory/CPU/storage
budgets, full-history checks, cold/long/reuse/203-vector/quality96/HTTP conditions
are unchanged. All temporary paths remain in the exact control root already
scanned by `storage_scan` and tied to the original owned services and cleanup.
This is source/mock verification, not an observation of successful live sockets
or a live cleanup.

Use the small preparation wrapper only after root selects the current complete
idle catalog and the fresh future namespace:

```python
import prepare_qwen35_sampler_v9_candidate_v3 as v9
import prepare_qwen35_sampler_v10_candidate_v1 as v10

reader = v9.JsonMetadata(metadata_guard)
candidate = v10.prepare(explicit_future, v10_manifest_ref,
                       reader.read, metadata_guard)
```

`explicit_future` retains the v9 fields, including `successor_sources` pointing
to the frozen **v9** manifest. The separate second argument points to
`qwen35-sampler-short-ipc-v10-sources-20261112-v1.json`, SHA-256
`7bc928e5f6d4b0450a593b0e0aa3265512ff04a17bd6413e8b67dcd3143c986c`.
The wrapper first performs all original v9 preparation/history/input checks,
then adds six source pins and the authenticated v10 manifest identity, rebuilds
overlay/spec/review hashes, and keeps the review unapproved. It writes nothing.
It does not select or hardcode catalog v85, or choose the next run. All actual
new historical rows and pins still come from the explicit complete catalog.

Fourteen focused new source/mock tests passed (`f0e2f9`, exit 0). They cover all
six paths, the 107/108 boundary, unique roots, environment overrides, socket
location/type/size/accounting/entry limits, and preparation pin preservation and
rejection. They create no sockets or services. A separate source check passed
(`164953`, exit 0): actual loaded Python closure, unchanged accepted v9 hashes,
identical science/resource/cleanup bodies, and retained actual vLLM selectors.
The executed r7's 48,002 pins plus the six additive runtime files form a 48,008
pin source union. This count is a source check, not a newly bound runtime plan.

All checks used system Python `-B -s`, CPU 7, nice 19, 512 MiB address-space and
60-second CPU limits, with torch/NumPy/ctypes/Transformers/vLLM/Neuron/NKI import
blocking. No old broad test suite was repeated. No frozen v9/predecessor source,
actual exit, prior result, or running file was changed. Independent root review,
the later current catalog binding, serial admission, and actual API execution
remain pending.
