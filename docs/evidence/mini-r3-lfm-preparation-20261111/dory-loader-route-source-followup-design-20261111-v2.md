# Dory loader follow-up: finite source collection and exact Stage A prefix

The successor source and 34 pure synthetic checks are complete. No collector was launched, installed package source was not newly read, and no event/raw/native/archive payload, process, lock or queue was inspected. All 15 actual loader-selection gaps remain open.

## Original model constraints: 56 histories total

The original code preserves **2 loops, 56 total independent attention histories with per-loop separation, and TP1**. It does not mean 56 histories per loop.

- [CACHE_IDS and VISITS](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_stage_a_capture_core_v1.py:18) enumerate `8 + 20 + 20 + 8 = 56`: eight prefix, twenty first-loop, twenty second-loop and eight suffix attention histories. The repeated middle layers appear twice in VISITS.
- [Loop/count assertions](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_stage_a_capture_core_v1.py:137) require `num_loops == 2`, 56 unique cache object identities and `recurrent_kv_cache_mode == 'per_loop'`.
- [TP1 setup](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_stage_a_worker_preparation_v1.py:121) fixes world_size=1/rank=0; [cache construction check](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_stage_a_worker_preparation_v1.py:189) again requires both cache lists/maps to contain 56 entries.

## Actual source entrypoint

[Resolver MODULES](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_stage_a_native_resolver_v4.py:42) and [its import loop](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_stage_a_native_resolver_v4.py:353) specify this order:

`torch → pytest → transformers → tokenizers → vllm → vllm.config → vllm.distributed.parallel_state → vllm.model_executor.models.dory → vllm.model_executor.models.utils → vllm.transformers_utils.configs.dory → vllm._custom_ops`

The existing [original imports metadata](/mnt/nvme/dory-official-cpu-20261108-v1/qualify-r2/qualification/imports.json) is 2,470 bytes with SHA-256 `8daae2c286aeabd38dbcfc8c614c4d4039a124b2cb5ad573ae85f8be26ec50ea`. It records CpuPlatform, Intel Xeon Platinum 8375C, x86_64, AVX512=true, AVX512-BF16=false, and `LD_PRELOAD=tcmalloc:libiomp5`. Together with the saved source this predicts the `_C_AVX512` branch and no automatic libgomp append, provided earlier imports do not change the environment. It does not prove actual library selection. The new source collection checks env_override/envs precisely because those effects are not yet known.

A separately reviewed native diagnostic can use this exact ordered import prefix and stop before the resolver executes upstream test source, constructs tokenizers or creates a model. Running this prefix is a new diagnostic; it is not full resolver completion or reconstruction of a historical loader trace. Other AVX modules must not be force-imported as evidence of the original route.

## Bounded source selection

Five required files: torch/_utils_internal.py, vllm/env_override.py, vllm/envs.py, vllm/distributed/__init__.py and vllm/distributed/parallel_state.py.

Additional Python candidates are chosen only from authenticated event names under these prefixes: `vllm/distributed/device_communicators/`, `vllm/v1/kv_offload/`, `vllm/distributed/kv_transfer/kv_connector/v1/offloading/`; or vllm Python paths containing `fs_io`, `spinloop`, `filesystem`, `file_system`, or `offload`. The collector does not enumerate the installed filesystem. It must exhaust and authenticate the original event stream before reading selected source bytes; every read retains the original wheel-member hash and file-identity checks.

The limits are **192 files, 512 KiB per source, 8 MiB source bytes**, plus 2,048 findings and 2 MiB findings output. Exceeding a limit fails; it does not broaden or silently truncate selection. The original descriptor/lifecycle guards and resource floors are retained.

These paths are hypotheses for finding fs_io_C/spinloop call sites, not already proven callers. A caller in another directory/name or an assembled dynamic import can be missed. AST imports inside functions, branches or TYPE_CHECKING do not establish runtime activation. No match means **not located within this collection**, never whole-package absence.

## Reviewable source and checks

- [Collector successor](/home/ubuntu/experiments/trn1-latest-models-20261031/research/collect_dory_loader_route_sources_v2.py)
- [Pure source selector/parser](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_loader_route_selection_v2.py)
- [34 synthetic checks](/home/ubuntu/experiments/trn1-latest-models-20261031/research/check_dory_loader_route_sources_synthetic_v2.py)
- [Full design, hashes and exact source excerpts](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-loader-route-source-followup-design-20261111-v2.json)

The checks include EOF failures after the last match, provenance/identity mutations, limits, path escape, native-file exclusion, parser non-execution, and unchanged v1 guard/read/write/final-release AST. Parent review must bind the new source_selection_policy identically in a fresh v2 plan/review and retain the four original Stage A source pins. No concrete runtime plan, review, admission, terminal or handoff is produced by this source task.
