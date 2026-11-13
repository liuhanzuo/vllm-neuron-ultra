# Qwen writer paired serial runtime v2 — source-only review

This successor fixes two confirmed source blockers using root-accepted package runtime v8: exact compiler constant / SDK links and exception-safe terminal storage. It is not a binder, ACK, observer, handoff, resource budget, launch admission or actual model result. Root source acceptance is pending.

## Scope

- `io.py` accepts only the observed five-component compiler layout. `constant.N.npy` may target exactly `../constant.N.npy` in the same owned arm/unit. Constant contents are never read. Directory/link/target identities are rechecked, and every constant target must appear in the complete ordinary-file inventory.
- `compiler_assets.py` and the exact unchanged v8 manifest bind only 16 installed Neuron 2.23.6484.0+3b612583 assets: 370,753 logical bytes / 393,216 allocated bytes. Binary assets remain typed assets; they are never inserted into the Python/text source map. Prelock verifies all 16 only during future explicit runtime admission. This delivery did not read real SDK asset contents.
- `compiler_workdir_owner` binds each link to exactly one writer arm. Compiler disappearance exceptions respect that arm's own immutable build seal. Both arms retain their own entry, class, and single-file caps. Shared SDK targets are physically counted once globally and charged to the first referencing arm in the fixed baseline-then-candidate root order. Candidate-only targets charge candidate. Existing SDK bytes on the root filesystem do not reduce the NVMe output reservation.
- `terminal_storage` is the exact accepted v8 helper. Failed storage proof cannot erase actual failure/drain evidence. A storage-only failure stays a failure; the original exception wins when both fail. The outer terminal is written before releasing the drained lock. No service/exit cleanup condition is relaxed.

## Preservation and evidence

The original writer runtime v1 and paired science v1 remain unchanged: all 23 + 29 = 52 delivery members were byte/SHA verified. All 28 scientific stages, 25 original requirements, paired independent builds, cold cache, long decode, slot reuse, numeric/quality/sampler/HTTP gates, r10 large metadata transport, resources/finite retries, service ownership, History, locks and final independent qualification remain intact. The final audit still does not self-promote `full_model_qualified`.

The first source-only test run passed **96/96** in 3.098 s (actual test session 35406, exit 0). Tests use native-import blockers, synthetic tempfile trees and mocked outer execution. They cover dual-arm links and shared-target accounting, exact path/owner/device/type/cap rules, racing identities, own-arm seals, all16 prelock checking, retained exit/failure evidence and exact source/AST preservation. `source_checks.json` records the actual test terminal and exact log pin. `predecessor_pins.json` binds prior runtime/science deliveries, accepted package v8/root review, accepted module hashes, fixed SDK manifest and the independent readiness note. `source_diff.patch` uses namespace-normalized v1 as its comparison.

## Remaining work

Independent root source review is required. No resource budget has been selected; the earlier 45.3125 GiB substitution had a 5.207 GiB gap and is not current admission evidence. Actual binder/ACK/observer, current History/source binding, independently qualified geometry, package r6 full validation plus actual storage facts, all 28 paired-writer stages, actual exit/drain and root final decision remain separate gates. No new throughput or speedup is claimed.
