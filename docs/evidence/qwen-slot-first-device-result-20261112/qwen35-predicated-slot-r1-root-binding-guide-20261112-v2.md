The next executable device experiment already has its accepted entry: `research/qwen35_predicated_slot_serial_v3.py`. No new runner is needed. The accepted root source review is `qwen35-predicated-slot-serial-root-source-review-20261112-v3.json`, SHA256 `8034727dfe0be10c5e4b3e2c486a2b133f9aa0453fe09f659dc9619db1a8dfcf`.

Package r4 has now actually passed: external session30240 / terminal035c00 / exit0. Root independently audited six owners, complete drain and the original lock. The post-r4 catalog is v101 with150 rows and its idle handoff is v567; their exact references appear in the invocation below. Its fixed seed has 48,194 source pins and a full History precheck already recorded in `qwen35-package-preflight-r4-20261112-root-binding-v5.json`. The accepted slot inventory adds nine distinct source paths without conflicts, reaching 48,203 before this candidate's new small metadata and preparer pin. The recovered 44,532-source partition and all 8,791 original base inputs remain unchanged. The large seed spec, seed review and base plan are preparation-only; they are never inserted into the ordinary 8 MiB runtime input table.

The r4 actual terminal, complete150-row catalog and idle handoff now exist and their supplied hashes were verified read-only. Still missing:

- The slot-specific overlay, spec and root-approved review. The thin recipe generates only the first two and a review draft.
- After launch, the root's running handoff using the actual new tool session. The original outer writes `running-ack.json` itself after checking that handoff.

The source-only script is `prepare_qwen35_predicated_slot_r1_root_binding_v2.py`. It fixes the exact validated r4 spec/review/overlay/root-binding and accepted slot-v3 delivery/review. It accepts explicit post-r4 catalog, idle handoff and terminal references. This is a finite immediate-successor recipe: if another serial row intervenes, it rejects the catalog rather than dropping that row. It does not discover or choose a current tip, call full History, inspect a lock, import native libraries, launch a process, or write a handoff. It has not been invoked to produce production binding files.

1. R4 terminal reconciliation is complete. The immutable inputs below were rehashed and their final row/tool/owner/unit and idle-handoff edges checked. If a new serial run or newer handoff intervenes, stop and rebind the complete current state; do not launch from stale v101/v567.

2. Root reviews this thin source recipe, then invokes it with the actual references. From `/home/ubuntu/experiments/trn1-latest-models-20261031`:

```sh
python3 -B research/prepare_qwen35_predicated_slot_r1_root_binding_v2.py \
  --catalog /home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v101.json \
  --catalog-sha256 3d6135fdf71318be12feb57dcfe402041bf27c7f3b921c68e095e291f6e57518 \
  --handoff /home/ubuntu/experiments/trn1-latest-models-20261031/research/continuation-handoff-20261112-v567-qwen-package-r4-ended.json \
  --handoff-sha256 3e4f8b0d89014c56bb636fb984e7f9beea1646086824461a263ff89acf367f99 \
  --terminal /home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-preflight-r4-root-actual-exit-20261112-v1.json \
  --terminal-sha256 9d8e3c7ff9d28d16402ee5a739c555a643585f8a2887df441389452634b9da1a \
  --run-name r1-v3-20261112 \
  --running-handoff /home/ubuntu/experiments/trn1-latest-models-20261031/research/continuation-handoff-20261112-v568-qwen35-predicated-slot-r1-running.json
```

These are the actual supplied completed refs, and root reserved v568 for the new running handoff. The recipe still takes them as explicit arguments and checks their complete contents. The original running-handoff validator permits lowercase letters, digits and hyphens only. The exact emitted files are:

| File under research/ | Meaning |
| --- | --- |
| `qwen35-predicated-slot-r1-v3-20261112-history-overlay.json` | Complete unchanged inherited plan plus additive current sources and inputs |
| `qwen35-predicated-slot-r1-v3-20261112-spec.json` | Exact slot-v3 spec with complete workload and limits |
| `qwen35-predicated-slot-r1-v3-20261112-review-draft.json` | `passed=false`, `launch_authorized=false` |
| `qwen35-predicated-slot-r1-v3-20261112-binding-candidate.json` | Exact refs, source count, preparation-only input identities and pending gates |

The future approved review path is `qwen35-predicated-slot-r1-v3-20261112-review.json`. Actual device outputs are under `/mnt/nvme/neuron-work/qwen35-predicated-slot-r1-v3-20261112`; the inherited metadata overlay names `/mnt/nvme/minicpm5-layer1-operator-native-slot-r1-v3-20261112` and does not create that directory.

The recipe calls the accepted `slot.verify_delivery(spec)` and `slot.resources(spec, reserve=True)` before exclusive metadata publication. Every old input identity must match; source merges reject replacements. Files are encoded using the original `objects.encode`, written exclusively, fsynced and checked against `objects.object_ref`. It never overwrites existing candidate files. A failed partial publication must be inspected, not silently replaced.

3. Root inspects the concrete complete spec and accepts its existing review envelope. Preserve the exact seven review keys, complete `source_sha256` and exact spec reference. Set `passed=true`, `blockers=[]`, `launch_authorized=true` only in the new approved review file after root review; retain the unapproved draft. Use `prepare_qwen35_sampler_v9_candidate_v3.encode` and exclusive publication, since the complete review exceeds the ordinary 8 MiB receipt limit. Then run the original complete History precheck before launching:

```python
from pathlib import Path
import sys
Q = Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
sys.path[:0] = [str(Q), str(Q.parent)]
import qwen35_predicated_slot_serial_v3 as stage
from qwen35_package_preflight_history_v2 import History
from qwen35_sampler_runtime_io_v12 import read, ref, loaded_local_sources
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest

candidate = read(ref(Q / 'qwen35-predicated-slot-r1-v3-20261112-binding-candidate.json'))
spec_ref = candidate['spec_ref']
review_ref = ref(candidate['paths']['review'], 64 * 2**20)
spec = stage.load_spec(spec_ref, review_ref)
tip = latest.latest_handoff()
catalog = read(spec['current_serial_catalog'])
assert tip['path'] == catalog['current_handoff']['path']
def guard():
    assert latest.latest_handoff() == tip, 'Serial tip changed during full History precheck'
bound = History(spec, spec_ref, review_ref, stage.names(spec), guard).resolve()
bound.unchanged()
fresh = bound.freshness()
assert fresh['latest_handoff'] == catalog['current_handoff']
assert len(bound.supplement['scope']['children']) == 874
loaded_local_sources(spec['source_sha256'])
stage.resources(spec, reserve=True)
```

This precheck replays the full original source/history/witness graph. It is deliberately pending in the recipe and its test results. Do not call `physical_pair` or acquire a lock here; the original outer owns the fresh physical pair and retained descriptor.

4. Launch only the accepted original outer, with the exact emitted spec hash and approved review hash. Its top-level check requires isolated mode and the exact nonexistent bytecode prefix before importing local code:

```sh
/usr/bin/env XDG_RUNTIME_DIR=/run/user/1000 \
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
  /home/ubuntu/venvs/vllm_neuron/bin/python -I -B \
  -X pycache_prefix=/mnt/nvme/neuron-work/qwen35-predicated-slot-r1-v3-20261112/bytecode-never-created \
  /home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_predicated_slot_serial_v3.py \
  --mode outer \
  --spec /home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-predicated-slot-r1-v3-20261112-spec.json \
  --spec-sha256 ACTUAL_SPEC_SHA \
  --review /home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-predicated-slot-r1-v3-20261112-review.json \
  --review-sha256 ACTUAL_APPROVED_REVIEW_SHA
```

The outer acquires the original retained lock, runs full History and the original physical pair, starts the single bounded service, captures its actual supervisor, then emits `qwen35_predicated_slot_running_handoff_required`. Its fields are `admission`, `coordinator_owner`, `coordinator`, and `running_handoff_path`. Native work is still held at that point. Use the actual external session returned by the tool launch, never a synthetic owner or guessed session.

5. Within the original 180-second ACK deadline, root writes only the requested running handoff using the existing pure function. With `event` equal to the exact emitted event and `actual_session_id` equal to the real tool session:

```python
from datetime import datetime, timezone
from qwen35_sampler_runtime_history_v12 import expected_running
from qwen35_sampler_runtime_io_v12 import read, once
admission = read(event['admission'])
coordinator = read(event['coordinator_owner'])
assert coordinator == event['coordinator']
previous = read(admission['freshness']['latest_handoff'])
running = expected_running(previous, datetime.now(timezone.utc).isoformat(),
    {'kind': 'tool_session', 'session_id': actual_session_id},
    event['admission'], admission, event['coordinator_owner'], coordinator)
once(event['running_handoff_path'], running)
```

Do not write `running-ack.json` manually. The original outer validates the complete handoff and actual identities, writes that ACK, and releases the supervisor. The supervisor reconstructs every history chunk and then runs eight fresh layout children sequentially. At completion root observes the actual external exit, rechecks all owners and the whole owned cgroup, and uses `stage.reconcile` with that same actual session; the existing scientific audit remains `audit_results` over all eight reports and 32 artifact receipts. A failed leaf is not retried with a smaller workload.

All limits stay fixed: 8 layouts, 32 graphs, 46,368 calls; NC0; CPUs6/7; nice19; 6 GiB MemoryMax/5 GiB MemoryHigh; zero swap; 900 seconds/layout, 7,200 seconds/service and 9,000 seconds/outer; 1 GiB/layout and 8 GiB total outputs. Initial NVMe is 40 GiB (30 GiB floor + 8 GiB reserved output + 2 GiB headroom); root floor28 GiB; separate LFM54 GiB requirement is unchanged. The read-only observation during preparation was NVMe40.413 GiB/root29.462 GiB. This is not a reservation or a launch-time guarantee; the original checks run again. No unfinished files were cleaned and no threshold was reduced.

Validation was entirely in memory: nine tests passed. They include a real v101/v567/r4-terminal candidate regression through the actual original JsonMetadata reader, plus synthetic adversarial cases. They checked exact prefixes, terminal drain/identity, source/input conflicts, full workload/limits, draft rejection by the actual contract and all 3,440 accepted local/SDK source hashes. The production preparation function, CLI publication, full History, native imports, compiler, locks, services and downloads were not run.

V2 correction: the actual r4 catalog row contains729 executed-source pins, not the full48194 seed map. The full map is preserved from the exact spec/review and all149 unchanged previous rows. Every shared increment pin must match; the only new current-row source must equal the actual terminal auditor_source. All729 current-row pins are merged, and the current row is neither replaced nor expanded. Both handoff.running=false and handoff.active_serial=null are required in addition to idle session/unit fields. V1 had incorrectly required its final row to contain the full map; the v2 actual-input test reproduces that v1 rejection. V1 source/checker files remain byte-for-byte unchanged as evidence.
