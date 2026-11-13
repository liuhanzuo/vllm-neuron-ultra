# Qwen writer paired-model static readiness (source-only)

This is a static review record, not a handoff, binding, launch admission, geometry qualification or full-model result. No new speedup is claimed.

The frozen writer runtime v1 delivery has 23 members; paired science v1 has 29. All 52 source/artifact members were SHA-256 and byte-count verified unchanged. The earlier checkout review found clean writer commit `e44408483a6880baa775d236d393812dedb2a443`. The JSON companion contains exact delivery and accepted package v8 pins.

The 28-stage draft retains the original 25 scientific requirements: independent 12-graph builds for baseline `torch_where` and candidate `nki_predicated_decode_v1`, the same sampler, complete state/numeric/cold/long/vector checks, sampler controls, quality96, matched HTTP and independent audit. Only decode writer behavior differs; prefill remains original. Runtime v1 already includes r10 large JSON compatibility in `io.py` and `source_text.py`; this is not a missing feature. The service implementation matches package v8 after namespace normalization. API-only cleanup-stop enforcement remains intentional.

Two source blockers are confirmed: compiler symlink rejection, and outer terminal evidence loss when its final storage scan raises. A minimal v2 may port package v8's exact same-unit relative constants, fixed 16 SDK assets and exception-safe terminal storage to both arms. No scientific gate, service gate, exit rule, budget cap or final qualification may be relaxed.

Earlier observations, not refreshed here: root available 31,267,905,536 bytes and NVMe available 43,062,845,440 bytes (about 40.105 GiB). Doubling the old per-arm caps (view 32 MiB, artifact 1 GiB, compiler cache 2 GiB, workdir 3 GiB), adding shared build auxiliary 256 MiB, the original 30 GiB NVMe floor, 2 GiB science cap and 1 GiB headroom yields **45.3125 GiB**, a **5.207 GiB shortfall** against that earlier NVMe observation. This is arithmetic substitution only, not a reviewed peak budget or current admission observation.

Next actual order remains independently qualified geometry, package r6 full 25-stage validation and actual storage evidence, then reviewed paired-writer successor and all 28 stages. Writer binder, ACK and observer are still separate work; no native/compiler/model/History/lock/queue activity was performed for this note. Current parent-owned live audit was not polled.
