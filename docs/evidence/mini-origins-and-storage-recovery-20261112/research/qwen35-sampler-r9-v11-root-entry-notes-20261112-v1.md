# r9 v11 prepared metadata and root entry

Preparation completed through the unchanged v9 JSON reader plus the accepted
v10/v11 wrappers. Actual tool session 65417 ended at `7f9355`, exit 0. The
ctypes/model-stack import blocker stayed enabled. No lock was acquired, no
native/model API was called, and no binary input was read. The actual 140-row
catalog v91 and idle handoff v551 are bound; 172 JSON identities were checked.

The candidate contains 48,048 source pins and preserves every one of r8's
48,031 pins. There are zero source paths under `olmo3-compiler228/env`. Saved
retirement metadata remains part of complete history; its path references do
not imply reopening retired payloads. All six IPC paths remain 95–99 bytes.

Created artifacts:

- Overlay: `qwen35-sampler-r9-20261112-history-overlay-v11.json`, SHA-256
  `d30d73df1fb9fcbf2524d7d9e6d25aa214436d79a12f781043c63bf003af877d`.
- Spec: `qwen35-sampler-r9-20261112-spec-v11.json`, SHA-256
  `cb3b4673fae650f4ee2a36ecf9840f867881bf8967e32ba3dd5a756ee33af0ea`.
- Review candidate: `qwen35-sampler-r9-20261112-review-candidate-v11.json`,
  SHA-256 `4f89cbdb9fc846bb0b11e2703a4515602fc949268a76293edbe87a3b2be365be`.
  It has `passed=False`, `launch_authorized=False`; it cannot admit a run.
- Preparation evidence: `qwen35-sampler-r9-v11-metadata-preparation-20261112-v1.json`,
  SHA-256 `e315337a476d9611a6fb7a75cd4e663222cca38f66dcfb3a0bfb72fd1aa1a9eb`.

The root reviews these exact artifacts and independently produces a separate
accepted runtime review if appropriate, for example
`qwen35-sampler-r9-20261112-review-v11.json`. The contract interface is
`qwen35_sampler_runtime_contract_v11.validate_spec(spec, review, spec_ref)`;
there is no `validate_review` function. Candidate output and review hashes are
computed by the existing `prepare_qwen35_sampler_v9_candidate_v3.encode` /
`object_ref` encoding. Do not edit any prepared artifact in place.

The actual outer entry is:

```text
python3 -B -s /home/ubuntu/experiments/trn1-latest-models-20261031/research/run_qwen35_sampler_runtime_v11.py
  --mode outer
  --spec /home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-sampler-r9-20261112-spec-v11.json
  --spec-sha256 cb3b4673fae650f4ee2a36ecf9840f867881bf8967e32ba3dd5a756ee33af0ea
  --review /home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-sampler-r9-20261112-review-v11.json
  --review-sha256 ROOT_ACTUAL_REVIEW_SHA256
```

Use one actual tool session and record it. `outer` itself acquires and retains
the original lock, then applies actual resource reservation, `Bundle.resolve`,
full historical source/witness verification, two physical history checks,
latest-handoff freshness, immutable graph reuse, and all 21 fresh service
checks. None of these physical/native checks was performed by metadata
preparation. In particular, do not claim source-only `Bundle.resolve`: its
historical path may import ctypes. Let the unchanged outer execute it under
the original lock.

The control directory is `/mnt/nvme/neuron-work/qw-r9-v11-20261112` and the
model output directory is `/mnt/nvme/neuron-work/qwen35-sampler-pair-r9-v11-20261112`.
Neither directory was created by preparation. The metadata-only overlay output
namespace is `/mnt/nvme/minicpm5-layer1-operator-native-qwen35-r9-v11-20261112`.

After the outer prints `qwen35_sampler_root_running_handoff_required`, root
must independently observe the actual tool session and owner event, then
publish exactly the output of the existing helper:

```python
from datetime import datetime, timezone
from pathlib import Path
from qwen35_sampler_runtime_history_v11 import expected_running
from qwen35_sampler_runtime_io_v11 import ref, read, once

control = Path('/mnt/nvme/neuron-work/qw-r9-v11-20261112')
admission_ref = ref(control / 'admission.json')
admission = read(admission_ref)
coordinator_ref = ref(control / 'coordinator-owner.json')
coordinator = read(coordinator_ref)
previous = read(admission['freshness']['latest_handoff'])
tool = {'kind': 'tool_session', 'session_id': ACTUAL_LIVE_TOOL_SESSION_ID}
running = expected_running(previous, datetime.now(timezone.utc).isoformat(),
                           tool, admission_ref, admission,
                           coordinator_ref, coordinator)
once(Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research/'
          'continuation-handoff-20261112-v552-qwen35-r9-v11-running.json'), running)
```

The root should match the saved admission/coordinator objects to the actual
event/owners before this write. Use the complete `expected_running` result
without editing inherited fields. Do not reread the large spec through the
ordinary 8 MiB `read` limit just to construct ACK; it is not needed here.

The outer waits at most the unchanged 180 seconds for this handoff, validates
its exact contents, and writes `running-ack.json` itself. Root does not create
that ACK file. Confirm that resulting ACK and continue polling the same tool
session. No later handoff/catalog or successor launch is permitted while it is
active. Final qualification still requires actual external terminal, original
quality/scientific/HTTP checks and independent complete drain audit.

This delivery does not create the accepted root review, running handoff,
admission, ACK, service, model result or speedup.
