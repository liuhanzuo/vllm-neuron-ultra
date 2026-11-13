# KV256 BF16/S16 pair: source-only serial integration

This delivery wires the accepted two-graph leaf to the existing serial lifecycle.
It does not select a current catalog, run, handoff, session or output directory.
No current runtime metadata, retained raw bytes, History, process, serial lock,
compiler, native library or device was inspected or executed in this task.
The root owns concrete admission and any subsequent execution/publication.

## Scientific boundary

One layout `kv256_s16_pair` contains `bf16_cat` and `s16_bitcast_cat`, each with
the accepted three input profiles: 2 graphs, 6 total calls, 18 complete old/new/
state raw-and-bitmap comparisons and 12 full post-call host-input captures.
The frozen leaf and HLO checker are unchanged. The actual compiler-input HLO
must establish BF16 public inputs/outputs, output 1 MUST_ALIAS, the original
prefix/tail and NKI writer inputs, and, for the S16 arm, integer concat with
three bitcasts and no numeric Convert. A complete diagnostic may retain
mismatches; completion never grants geometry/model qualification or speedup.

This extra pair does not replace the original missing alias0, tail-copy or
full-copy controls. It also does not complete full geometry/model qualification.
Actual compile/load support, the 900-second leaf deadline and bounded output
fit remain unproven until an admitted real run completes.

## Changes from the frozen predecessor

The new serial namespace retains the original environment, admission, ACK,
supervisor, child handling, deadline, cleanup, History and output scanner.
The adapter narrows the workload to the pair and accepts only the exact seven
artifact receipt fields: name, artifact, origin, bucket, alias, hlo, hlo_proof.
It authenticates full model/HLO path, size and SHA and independently recomputes
the complete HLO proof from the actual saved HLO bytes.

The leaf saves a graph after HLO capture but before the HLO gate and artifact
callback. The interrupted-prefix validator therefore permits exactly the final
graph to exist without a receipt, only with zero calls for that graph. The
independent auditor distinguishes compiled graphs, artifact receipts, started/
returned/completely proved calls, pending HLO proof and saved failure. A first
HLO rejection remains 1 graph / 0 receipts / 0 calls; a second-arm rejection
retains the first arm's three completed calls. Full success still requires two
complete receipts and all six calls through the full accepted leaf validator.

`render_qwen35_kv256_s16_pair_helpers_v1.render` authenticates three frozen
predecessor source strings and derives the preparer, root helper and auditor
byte-for-byte. Its changes are namespace substitution, 1/24/8 to 1/6/2 final
counters, and the explicit partial-HLO counters above. The preparer retains the
one exact already-inherited 12,803,326-byte plan identity and existing 32 MiB
typed reader; all ordinary metadata remains capped at 8 MiB. Neither that real
plan nor a current request is read by this source task.

Five root helper sources, including the independent physical History adapter
and existing post-exit source-delta helper, must be pinned before preparation.
Accept, ACK and finalize recheck those pins. Finalize cannot add source after
execution; it preserves every old catalog row and appends one actual outer
tool row, retaining the separate successful independent auditor tool receipt.

## Unchanged limits

- CPU affinity 6/7, nice 19, one visible Neuron core, one-thread native environment.
- 6 GiB MemoryMax, 5 GiB MemoryHigh, zero swap; original service/outer limits.
- 900 seconds per leaf and 1 GiB total output, all original per-file/class caps.
- Conservative initial NVMe admission remains 40 GiB; after the full 1 GiB
  output allocation, the corresponding free-space floor remains 39 GiB.
- Full source/SDK inventory, original History, original serial lock, complete
  declared services, owner drain and independent two-pass physical checks.

The smaller pair's output does not justify reducing the initial reservation.

## Future caller contract (placeholders, not a current request)

All references below have exact `{ "path": "<canonical absolute path>",
"sha256": "<full actual SHA256>" }` form. The root supplies then-current
references and fresh paths only after source acceptance and serial eligibility.
No placeholder is an accepted review or actual terminal observation.

1. Review `qwen35_kv256_s16_pair_serial_v1/source_delivery.json` and
   `qwen35-kv256-s16-pair-binding-source-delivery-20261113-v1.json` separately.
   The serial review must be written at the exact future path
   `research/qwen35-kv256-s16-pair-serial-root-source-review-20261113-v1.json`.
   Both reviews require reviewer `/root`, `source_accepted: true`, empty
   blockers and the exact delivery ref. Serial review requires
   `runtime_admitted: false`. Source review alone does not authorize launch.
2. Invoke `prepare_qwen35_kv256_s16_pair_binding_v1.py --request <future.json>
   --request-sha256 <actual digest>` using `python -B`. The request schema is
   `qwen35_kv256_s16_pair_future_binding_v1`. Its exact ref fields are
   `seed_spec`, `seed_review`, `seed_binding`, `catalog`, `idle_handoff`,
   `prerequisite_exit`, `diagnostic_source_review`, `preparer_source_review`,
   `preparer_delivery`; it also contains `prerequisite_tool`, `run_name` and
   `paths`. `prerequisite_tool` is the actual last catalog tool terminal.
   `paths` has exactly `overlay`, `spec`, `review_draft`, `review`, `candidate`,
   `output`, `overlay_output`, `running`. Output uses
   `/mnt/nvme/neuron-work/qwen35-kv256-s16-pair-<run_name>`; all paths are fresh.
   The preparer writes four candidate files with `passed: false` and
   `launch_authorized: false`, retaining full old source/input/catalog lineage.
3. Call root helper `--mode accept --request <accept.json> --request-sha256
   <digest>`. Schema: `qwen35_kv256_s16_pair_accept_request_v1`; exact other
   fields: `candidate` ref and `binding_output` path. This performs the existing
   full source/History/resource/idle admission under the original lock. It
   creates the concrete review and root binding, without starting runtime.
4. The root launches `qwen35_kv256_s16_pair_serial_v1/run.py --mode outer
   --spec <spec path> --spec-sha256 <sha> --review <review path>
   --review-sha256 <sha>` with the existing accepted outer invocation policy.
   Only the real tool result provides the actual outer session ID.
5. After real outer/supervisor admission, call root helper `--mode ack` with
   schema `qwen35_kv256_s16_pair_ack_request_v1` and exactly `binding` ref,
   `actual_tool: {kind: tool_session, session_id: <actual positive integer>}`,
   `live_output` path. ACK verifies live owners, original retained lock,
   service and exact idle handoff before publishing running. No invented owner
   or session may be supplied. A failure before admission/owner receipts keeps
   the original root no-admission failure path; this helper cannot invent them.
6. Observe the actual outer terminal. Independently invoke
   `qwen35_kv256_s16_pair_audit_v1.py --spec <path> --spec-sha256 <sha>
   --review <path> --review-sha256 <sha> --session-id <actual outer ID>
   --terminal-chunk <actual outer chunk> --exit-code <actual outer exit>`.
   The auditor preserves failed prefixes and performs the original independent
   complete two-pass History and current service drains. Only successful
   diagnostic reconciliation writes a diagnostic-only completion receipt.
7. Observe that distinct audit tool's actual successful terminal. Root helper
   `--mode finalize` accepts schema
   `qwen35_kv256_s16_pair_finalize_request_v1`, exact fields `binding`, `live`,
   `outer_tool`, `audit_tool`, `outputs`. The first two are refs; both tools
   contain kind, actual session_id, actual terminal_chunk and actual exit_code.
   `outputs` contains fresh `idle`, `catalog`, `receipt` paths. Finalize repeats
   the original drains under lock and appends one row without any source retrofit.

Entry imports and `--help` remain inert. Runtime modes intentionally retain the
original physical observations; they are not part of this source-only task.

## Validation and remaining review

32 serial and 40 helper source/mock tests pass. Coverage includes exact HLO
receipt validation, recomputed HLO proof, interrupted stages, all 27 inherited
binding controls, retained plan identities, no post-run source retrofit and
failed diagnostic publication. Native imports are blocked during tests. The
first helper run exposed a synthetic fixture selecting the predecessor's review
path; its fixture override was corrected and all 40 checks then passed. No
runtime implementation change was needed for that fixture failure.

The delivery records actual test terminal chunks and complete source hashes.
It awaits root source review, actual current serial binding, complete admission,
and any real compile/device execution. No new speedup is claimed here.
