Qwen3.5 packaged native preflight v3 source delivery, 2026-11-12

V3 fixes the prepared-tokenizer-alias write gap identified in `qwen35-package-native-preflight-independent-source-review-20261112-v2.json` (SHA256 `cabc284c8a442f1594267eb4456dd470234bedeb849be587ea875804609010de`). V2 remains immutable and is superseded for future execution. No native imports/tokenizer execution, original tokenizer/model payload reads, compile, accelerator work, runtime binding/input or experiment-state mutation occurred in this source-only correction.

The rejected v2 callback accepted `open('/work/shared-hf-config/tokenizer.json', 'w', O_WRONLY)` although that prepared alias points to the original tokenizer. V3 identifies read-only paths from the exact tokenizer allowlist, protecting original/historical config and tokenizer files plus all five prepared tokenizer aliases. It checks both normalized absolute names and resolved symlink equivalents. Writes are rejected when text mode contains `w`, `a`, `x` or `+`, or numeric flags contain `O_WRONLY`, `O_RDWR`, `O_CREAT`, `O_TRUNC` or `O_APPEND`. The fresh ordinary `shared-hf-config/config.json` remains writable; a config path resolving to a protected tokenizer target is rejected.

New immutable files under `/home/ubuntu/experiments/trn1-latest-models-20261031/research/`:

- `qwen35_package_native_preflight_v3.py`
- `qwen35_package_tokenizer_checks_v3.py`
- `prepare_qwen35_package_native_preflight_v3.py`
- `check_qwen35_package_native_preflight_v3.py`
- `qwen35-package-native-preflight-source-bundle-20261112-v3.json`

The bundle is 19,612 bytes, SHA256 `6c45ed5293eca2cd7ee48488ebd64f79e26c12136b2aec8b46af40c28238a080`. It pins all four sources and the same 70 package sources, 20 SDK sources and 11 tokenizer protocol/source/metadata references. The package-v2 snapshot, checkout/source commit and original resources are unchanged.

Validation: `PYTHONDONTWRITEBYTECODE=1 python3 research/check_qwen35_package_native_preflight_v3.py` passed 31/31 tests. All 28 v2 tests are retained. Three new checks cover all original/historical/prepared names with text and numeric write modes and normalized equivalents; resolved alternate symlinks, actual reads of only temporary synthetic files, and actual fresh config creation; and frozen v2 bytes plus unchanged worker functions outside the guard, allowing version-name substitutions. The tokenizer helper is byte-for-byte identical to v2. Guard callbacks are captured with a mock audit-hook installer; no global audit hook or real native runtime is installed in these tests.

All96 complete original-versus-view input-ID comparisons, saved historical request IDs, six original HF fixtures, two long sequences, exact template arguments, original-function/source-projection boundaries, resource ceilings, root binding requirements and pending native qualification remain exactly as documented in `qwen35-package-native-preflight-source-delivery-20261112-v2.md`. This change adds no prompts or qualification claims. Python audit hooks still do not claim to sandbox native library internals; the enclosing root service retains isolation and cleanup responsibilities.

The worker schema is `qwen35-package-native-preflight-v3`; its root-input shape remains `{schema,bundle,work,binding}`. Root must supply its then-current complete original owner/lock/history/cgroup binding. No actual runtime input or admission has been issued. Independent source acceptance and native execution remain separate pending evidence.
