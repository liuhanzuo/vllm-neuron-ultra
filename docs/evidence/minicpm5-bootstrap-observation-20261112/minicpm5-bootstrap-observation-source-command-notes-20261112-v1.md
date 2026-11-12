# MiniCPM5 dual-Python bootstrap observation: source review package

This is a discovery implementation for independent root review. No actual plan,
admission, native execution, model execution, serial lock operation or authority
was created by this source delivery. The formal frozen predecessors are unchanged.

The source manifest to review is `minicpm5-bootstrap-observation-source-manifest-20261112-v2.json`,
SHA256 `05bbd33755eb64e37c16cb5b7c66ec0d890fea0217fe8f1af9287c93b65f7466`.
It retains every prior 106 source pin and adds the complete local AST import
closure: 133 files, 1,491,486 bytes, 447 import edges. It seeds both actual formal
entry modules explicitly because those entry names are selected dynamically.
The embedded helper + child literal is 42,142 bytes. The unreviewed manifest v1
is retained as failed metadata preparation evidence: its source hashes were
correct, but a generator variable shadow made its import-edge dictionary
incomplete. Use v2 only.

## Behavior and evidence boundaries

One parent acquires the original device/inode 66306/8294421 serial descriptor.
It authenticates only the saved finite candidate set, then executes SDK310 and
outer312 as separate children in that order. It waits for and drains the SDK
child before starting outer312. A failed SDK child prevents outer312 startup.
Both children inherit the same parent `started_monotonic + 900` deadline.

The actual parent command, child command, role interpreter, parent PID/start
identity, original descriptor, ACK pipe and limits are checked. The parent
stores the command with `admission:self` at argv index 13, hashes that admission,
then replaces only that slot with the resulting SHA for the real command.
The SDK child keeps `/usr/bin/python3.10`; outer keeps the original uv Python
3.12.16 executable. The original venv command paths are preserved for each.

Each child captures full raw `/proc/self/maps` at earliest entry, after importing
its real formal bootstrap module but before ctypes, and after ctypes plus the
explicit libc load. The real formal module is imported without calling
`source_entry`. The saved observation identifies actual selected `_ctypes`,
`_struct`, `_hashlib`, `_blake2` and zlib providers. A failed bootstrap saves
earliest and partial failed-bootstrap maps when the original deadline still
permits evidence writes; it never invents a completed pre-ctypes snapshot.

The parent full-hashes only the 18 distinct native candidates (46,364,152 bytes)
and six pure-source candidates already bound to completed saved package/archive
results. It reuses StableReader, the existing ELF parser, six positive FFI
definitions and mapping-closure code. Unknown mapped objects, including a Python
zlib extension absent from the finite table, retain their actual path/device/inode
as missing provenance; their binaries are not opened. A known candidate identity
mismatch or failed ELF/FFI parse is retained as a gap. Discovery completion can
therefore coexist with missing provenance.

FFI definitions describe known candidate content; they do not independently prove
actual loader selection. Pure-source receipts identify which saved source rows
really passed the pinned source loader; an already imported pure module has no
fabricated execution receipt. Native ABI, accepted native authority, model
qualification and speedup flags remain false throughout. Torch, NumPy and model
imports remain blocked by the unchanged entry fence and discovery subclass.

## Original history and resource policy

`prepare_minicpm5_sdk310_deb_plan_v4.build` is reused for complete saved-history
reconstruction, including its reviewed 121-row prefix, actual subsequent terminal
checks, all original owners/services and exact three retirement exceptions.
The discovery parent reconstructs that metadata again and compares the complete
history fields, then calls the original retirement preflight for every current
owner/unit. Old terminal fields are not normalized or rewritten.

The live original guard, disk/memory floors, precise retirement rules,
`life.drain` and `release_after_drain` are retained. Root must bind and review the
latest ended/drained catalog and idle handoff before launch. This implementation
does not independently search for a newer handoff filename; a running handoff
published after launch is not mistaken for a stale plan.

Limits stay AS 2 GiB, CPU soft/hard 600/605 seconds, FSIZE 2 MiB, CPU 6, nice 19,
one absolute 900-second deadline. Original aggregate output caps remain 32 MiB
allocated, 8 MiB metadata, 12,288,000 raw bytes and 4,392 files. Maps are complete
or fail at 1 MiB, never truncated. The original scientific 9 native checks and
1,284 descriptor rows are pending and unchanged.

The real saved replay-r1 history has 2,818 owners and 1,455 services, and its old
owner.json exceeded 2 MiB. This implementation persists identical complete
history rows as <=1 MiB shards plus a manifest; the owner record references that
manifest. A saved-JSON-only check preserved all 1,455 rows in four shards, totaling
3,277,299 bytes with a largest file of 1,046,071 bytes. The aggregate caps are not
raised and no historical row is removed.

## Interfaces for root review and a future admitted run

1. Independently review the v2 source manifest. The source acceptance JSON must
   contain `reviewer: /root`, `passed: true`, `blockers: []` and `manifest` bound
   to its exact path/SHA. This is source acceptance only.
2. After all intervening actual terminals/recorders and drains are in the latest
   catalog, compose an unreviewed plan using the callable API or CLI below.
   Preparation writes only a new plan; it does not launch or create a review.
3. Independently review that exact plan. Review schema is
   `minicpm5_bootstrap_observation_review_v1`; it binds `plan`, `source_sha256`,
   `candidate_inputs`, `limits`, `artifact_limits`, `budget`, `role_order`,
   `complete_current_serial_catalog`, `root_handoff`, `predecessor_exit` and
   `lock_policy`, with `reviewer: /root`, `passed: true`, `blockers: []`,
   `observation_only: true` and `accepted_native_authority_created: false`.
4. Root owns fresh launch admission and external observation of the actual
   parent's terminal. The runner deliberately does not claim its own process
   exit and does not implement a new standalone external exit recorder. Bind the
   existing root recorder/terminal workflow before promoting a completed stage
   into the successor catalog. No automatic successor is launched.

Callable preparation:

```python
build(read, catalog_ref=..., scope_ref=..., predecessor_ref=...,
      retirement_acceptance_ref=..., source_manifest_ref=...,
      source_acceptance_ref=..., output_root=...)
```

`read(ref)` must authenticate complete bounded JSON and stable path/content.
The scope and retirement acceptance are the existing complete LFM r2 scope and
98-source precise-retirement acceptance. The observation manifest additionally
requires all latest 106 pins unchanged. The current catalog and predecessor are
provided by root; this delivery creates neither a current plan nor an admission.

Preparation CLI argument names:

```text
python3 -S -B research/prepare_minicpm5_bootstrap_observation_v1.py
  --catalog PATH --catalog-sha256 SHA
  --scope PATH --scope-sha256 SHA
  --predecessor PATH --predecessor-sha256 SHA
  --retirement-acceptance PATH --retirement-acceptance-sha256 SHA
  --source-manifest PATH --source-manifest-sha256 SHA
  --source-acceptance PATH --source-acceptance-sha256 SHA
  --output-root /mnt/nvme/dory-serial-inventory-minicpm5-sdk310-bootstrap-UNIQUE
  --destination /home/ubuntu/experiments/trn1-latest-models-20261031/research/UNIQUE-plan.json
```

Future runtime command, only after root acceptance and physical admission:

```text
/usr/bin/python3 -B -s -X pycache_prefix=/dev/null
  /home/ubuntu/experiments/trn1-latest-models-20261031/research/run_minicpm5_bootstrap_observation_v1.py
  --plan PATH --plan-sha256 SHA --review PATH --review-sha256 SHA
```

## Source and mock verification

All five new files compiled without bytecode writes (`d9fb00`, exit 0).
Thirteen tests passed (`ce690a`, exit 0): original caps/order/authority rejection;
unknown or mismatched maps; missing FFI proof; exact embedded literal;
child argv/admission/flags/deadline tampering; child.main malformed ACK and formal
bootstrap failure with retained partial maps; complete preparation pins/history;
omitted owners/services; lossless bounded history shards; sequential parent
lifecycle; failed SDK preventing outer; timeout draining before release.
Only source, saved JSON and mock ports were used. The successful child ctypes
path has not been executed; actual native results remain unknown.

Reproducible bounded source/mock command:

```text
taskset -c 7 nice -n 19 prlimit --as=536870912 --cpu=60 -- python3 -S -B
  /home/ubuntu/experiments/trn1-latest-models-20261031/research/check_minicpm5_bootstrap_observation_v1.py
```
