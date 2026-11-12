Qwen3.5 packaged native preflight source, 2026-11-12

This supplies an executable, bounded preflight for a future root-owned serial slot. No native preflight, compiler, model, weight/tokenizer payload, artifact, device graph, process launch, lock inspection or cgroup inspection ran during preparation. Existing Qwen r10 remains outside this work. Frozen package v2 sources and the runtime checkout are unchanged.

Files are under `/home/ubuntu/experiments/trn1-latest-models-20261031/research/`:

- `qwen35_package_native_preflight_v1.py`: explicit worker; no arguments prints schema only.
- `prepare_qwen35_package_native_preflight_v1.py`: source-only bundle preparation, never a runtime plan/input.
- `check_qwen35_package_native_preflight_v1.py`: 18 passing standard-library/source/mock checks with all real native imports forbidden.
- `qwen35-package-native-preflight-source-bundle-20261112-v1.json`: 15,530 bytes; SHA256 `f07374b6e3d275a9f2476db5232f438f9c81a8a46472a8cecf4e56df1264963e`.

The bundle authenticates the unchanged package-v2 snapshot, all current `vllm_neuron` Python sources at source commit `e40fd753ba080a9a4640688f4d6beda9985b7099`, the worker/checker/preparer, the pinned SDK source hashes and runtime versions, and the original small official Qwen3.5 config fixture. It pins the exact original text-only transformation and Neuron root constructor kwargs. This bundle is source inventory, not permission or native success evidence. No runnable input, plan, review/approval receipt, or publication was produced.

The future worker uses only `/home/ubuntu/venvs/vllm_neuron/bin/python` (Python3.10). Root launches a fresh `-I -B` interpreter, applies the pure `environment(work, inherited)` helper, and places it under the existing original lock owner and bounded cgroup. The worker verifies owner PID/start ticks, original lock device/inode and actual existing kernel FLOCK; it never acquires, transfers or releases the lock. Root supplies exact references to its complete current history and physical drain metadata. The worker checks those exact bytes and records the binding; it does not reconstruct history, lower a predecessor floor, or certify that root's selected history references are complete. Root retains scheduling, timeout/failure cleanup, descendant/resource-tracker drain, terminal reconciliation and release responsibility.

Cgroup ceilings are MemoryMax ≤12 GiB, MemorySwapMax=0, TasksMax≤256, inherited affinity of at most8 CPUs. The worker adds 600s wall, 480s CPU, 16MiB file and zero-core limits. Both parent and actual spawn child re-observe binding and source pins. `multiprocessing.get_context('spawn')` starts the child, which repeats native checks; the parent joins it, checks actual PID/PPID/exit and compares configuration fingerprints and imported origins. On its own wait failure it terminates only that owned child. Root's enclosing cgroup lifecycle must still handle abrupt worker death and the multiprocessing resource tracker.

Native work, only in that future run:

1. Import actual torch, Neuron SDK, Transformers, vLLM and packaged adapter. Check pinned versions and package/SDK sources; record real native Python/binary module origins without hashing native binary contents. Every `vllm_neuron` module must originate in the exact checkout, and native dependencies in the original site-packages. A fresh `sys.pycache_prefix` plus disabled writes avoids loading preexisting dependency bytecode caches.
2. Explicitly register twice and verify real Transformers, vLLM and NxDI registries. Check that text_core holds the packaged corrected split function. Execute a CPU-only BF16 transpose/tensor_split coordinate check using6144 synthetic elements (12KiB BF16 storage). No model parameter, cache or weight is constructed.
3. Authenticate the original config JSON with the package's real `compatible_model_config`; write only the resulting config into a fresh shared config-only directory. The real AutoConfig → hf_adapter → Qwen35InferenceConfig path must normalize dtype aliases and pass package scope validation. Parent/spawn share this path because HF may serialize `_name_or_path`.
4. Save/load and reconstruct the actual complete SDK JSON, comparing every field via package canonicalization, including only the documented SDK +Infinity. Exercise real saved-configuration admission and a changed-context rejection, nine package scope negatives, and two full SDK JSON-constructor/package-scope invalid cases. The output includes full SDK JSON files and the complete type-tagged canonical body; it does not silently omit fields to force equality.
5. Construct two actual host Sampler instances. Standard opt-in must preserve its original operation; fast opt-in must wrap only the chosen instance, preserve its reference and remain idempotent. Package source hashes, full configuration and compiler environment must remain identical across host flags. Full artifact identity is explicitly not computed because its model payload admission would read weights. No sampling-history, numerical, device or performance result is claimed.

Runtime guard rejects model/compiled/archive payload opens, accelerator-device opens, arbitrary subprocess launch and network connects. It permits the required actual multiprocessing spawn. Native imports may expose a real unsupported initialization side effect or configuration mismatch; such failure is evidence, not permission to relax package pins or bypass the root slot.

Concrete future input schema (descriptive placeholders, not an issued runnable input):

```json
{
  "schema": "qwen35-package-native-preflight-v1",
  "bundle": {"path": "ABSOLUTE_SOURCE_BUNDLE_PATH", "bytes": 15530, "sha256": "f07374b6e3d275a9f2476db5232f438f9c81a8a46472a8cecf4e56df1264963e"},
  "work": "FRESH_ABSOLUTE_WORK_DIRECTORY",
  "binding": {
    "owner": {"pid": "CURRENT_ROOT_OWNER_PID_INTEGER", "start_ticks": "CURRENT_ROOT_START_TICKS_INTEGER"},
    "lock": {"path": "ORIGINAL_LOCK_ABSOLUTE_PATH", "device": "ACTUAL_ST_DEV_INTEGER", "inode": "ACTUAL_ST_INO_INTEGER"},
    "history": [{"path": "CURRENT_COMPLETE_HISTORY_OR_DRAIN_METADATA", "bytes": "ACTUAL_SIZE_INTEGER", "sha256": "ACTUAL_SHA256"}],
    "cgroup": "/EXACT_ROOT_ASSIGNED_CGROUP_V2_PATH"
  }
}
```

`history` is a nonempty list: root includes every necessary current history/drain reference, not just the one illustrative item. `work` must be absent with an ordinary existing parent. Root must provide exactly the environment returned by `environment(work, minimal_inherited_environment)` for relevant worker keys; historical TRN1 switches, other model selectors, compile opt-ins and artifact paths are rejected.

Future command under root's existing serial supervisor and cgroup, after root binds the actual input bytes:

```sh
/home/ubuntu/venvs/vllm_neuron/bin/python -I -B \
  /home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_package_native_preflight_v1.py \
  --execute-native --input ACTUAL_ROOT_INPUT_JSON --input-sha256 ACTUAL_ROOT_INPUT_SHA256
```

Source checks performed:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 research/check_qwen35_package_native_preflight_v1.py
PYTHONDONTWRITEBYTECODE=1 python3 research/prepare_qwen35_package_native_preflight_v1.py \
  research/qwen35-package-native-preflight-source-bundle-20261112-v1.json
```

18/18 tests pass. They include inert/no-argument behavior, hash-before-execution, no native action on stale root binding, mocked actual FLOCK/cgroup/affinity checks, source-only original config authentication, whole mocked SDK roundtrip with real package admission/scope helpers, invalid config rejection, bounded output/non-overwrite, spawn control-flow and no model/compiler/weight-loader calls. The preparation bundle was generated once; rerunning that command at the same destination intentionally refuses overwrite.

Remaining work is root source review, binding its then-current complete serial history/owner/cgroup and executing this preflight. Passing preflight would establish fresh native package imports/registration/configuration and tiny CPU split/host sampler integration only. Fresh graphs, model logits, cold/long/reuse state, actual sampling histories, HTTP quality/workloads and performance remain separate real qualification work; no r10 performance or frozen research graph success is inherited.
