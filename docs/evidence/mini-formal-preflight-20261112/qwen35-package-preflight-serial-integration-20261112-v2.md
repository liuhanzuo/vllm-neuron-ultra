Qwen3.5 packaged preflight serial integration v2, 2026-11-12

V2 fixes the independently reproduced v1 cleanup-composition blocker. When `systemd-run` failed before installing its unit, the original `Service.start` had already set `started=True`. Its original `cleanup` correctly returned a `never_loaded` proof with an empty cgroup, but v1's following unconditional `drained()` required a loaded unit and caused the retained-lock cleanup loop to retry indefinitely.

The only behavioral change is `owned_terminal(service)`, called from final verification. If the original service object has already produced `never_loaded=True`, verification calls the unchanged original `cleanup()` again for a second fresh no-install/empty-group observation. Every other started branch still uses the unchanged original `drained()`. The original no-install branch rejects any captured owner, installed invocation, pending process or unacknowledged terminal; disappearance of an observed unit is never treated as proof that it was not installed. A unit installed between observations enters the original authenticated cleanup path.

Frozen delivery: `research/qwen35-package-preflight-serial-source-delivery-20261112-v2.json`, 107,571 bytes, SHA256 `8927a4b1b177aafb65cad7ad36e3899c84a2dad1f1f33329ef6f6277c5765652`.

New source names:

- `qwen35_package_preflight_serial_v2.py`
- `qwen35_package_preflight_history_v2.py`
- `check_qwen35_package_preflight_serial_v2.py`
- `prepare_qwen35_package_preflight_serial_v2.py`

V1, the original Qwen v12 lifecycle sources, and accepted packaged native worker v3 remain immutable. The history helper is byte-for-byte identical to serial v1. Every previous adapter function except the final `outer` verification call is AST-identical after the explicit version-name substitutions. No history, source, resource, environment, worker-binding, output, running-handoff or actual-tool reconciliation checks were weakened. The existing root review envelope remains unchanged; the serial spec/admission/outer-terminal/reconciliation schemas use the v2 suffix.

Validation: `PYTHONDONTWRITEBYTECODE=1 python3 research/check_qwen35_package_preflight_serial_v2.py` passed18/18. The original14 tests remain. Four new checks authenticate frozen v1 bytes and unchanged functions, and exercise the actual original `Service` methods with only system ports replaced:

1. `start()` fails before unit installation. Original `cleanup()` followed by v2 verification performs two original no-install checks, returns complete cleanup proof, and does not enter the retained-lock wait.
2. A unit actually installs and exits7 before owner acknowledgment. Original `start`, `cleanup`, `observe`, property/command/duration validation and `drained` retain that exact failed status; v2 continues using the loaded-unit drain path.
3. A not-found unit with any captured owner, invocation, pending process or unacknowledged-terminal identity is rejected before its empty-group check. Such evidence cannot be relabelled `never_loaded`.

No real lock, service, process/cgroup query, native/tokenizer/model/archive execution or original payload read occurred. Tests used temporary metadata and mocked system ports. No current spec/review/runtime input, actual running directory or history selection was created. Complete lifecycle details and unchanged resource obligations are recorded in `qwen35-package-preflight-serial-integration-20261112-v1.md`; the corrected v2 successor is the candidate for independent source acceptance and later root-owned execution.
