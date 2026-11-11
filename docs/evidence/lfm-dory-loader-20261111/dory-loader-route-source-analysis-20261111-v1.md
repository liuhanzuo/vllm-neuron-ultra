# Dory: source routes for the 15 remaining loader gaps

The five saved Python sources establish conditional loader routes. **All 15 actual-loader selection gaps remain open**: nine CPU-kernel dependency edges have conditional source routes, four fs_io_C/spinloop edges lack callers in this collection, and two libgomp edges have only a conditional preload policy. Prior closure of 11 system byte-origin gaps and four cache-derivation gaps is preserved. No ABI, full-model, or speedup qualification follows.

This report reads only saved Python sources and metadata; it does not import target packages, inspect live processes/services/locks, read native/model/archive payloads, or create a runtime admission.

## Source findings

- **package_lazy_entry**: vllm package initialization imports version and env_override; exported engine/model attributes are lazy. These five sources do not establish the original Dory caller or a complete package-to-native route.
  Evidence: [vllm/__init__.py:7](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/__init__.py:7) (through line 14), [vllm/__init__.py:16](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/__init__.py:16) (through line 27), [vllm/__init__.py:63](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/__init__.py:63) (through line 73).

- **cpu_selection_conditional**: An explicit CPU target bypasses broad plugin probing, but cpu_platform_plugin can still return ZenCpuPlatform when AMD Zen and zentorch import succeed; otherwise it returns CpuPlatform. The actual platform and environment were not observed here.
  Evidence: [vllm/platforms/__init__.py:174](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/__init__.py:174) (through line 184), [vllm/platforms/__init__.py:202](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/__init__.py:202) (through line 228), [vllm/platforms/__init__.py:240](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/__init__.py:240) (through line 250), [vllm/platforms/__init__.py:303](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/__init__.py:303) (through line 322).

- **torch_before_cpu_kernels**: Within the saved _custom_ops module body, import torch precedes current_platform resolution and import_kernels. CpuPlatform itself imports torch before defining its kernel loader. This proves source order on that route, not first-process load order, import success, platform choice, or the bytes selected by the ELF loader.
  Evidence: [vllm/_custom_ops.py:8](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/_custom_ops.py:8) (through line 21), [vllm/platforms/cpu.py:11](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/cpu.py:11) (through line 13), [vllm/platforms/cpu.py:656](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/cpu.py:656) (through line 693).

- **torch_native_routes**: Torch imports USE_GLOBAL_DEPS and USE_RTLD_GLOBAL_WITH_LIBTORCH from _utils_internal. A truthy TORCH_USE_RTLD_GLOBAL or flag chooses RTLD_GLOBAL|RTLD_LAZY around torch._C; otherwise the optional global-deps CDLL precedes torch._C. Global deps use a path beside __file__, with a distribution-location fallback. These flags and native transitive object selection are not established by the five sources. No direct Python selection of libc10.so/libtorch_cpu.so/libtorch.so/libgomp.so.1 occurs in this file.
  Evidence: [torch/__init__.py:59](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/torch/__init__.py:59) (through line 65), [torch/__init__.py:392](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/torch/__init__.py:392) (through line 424), [torch/__init__.py:450](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/torch/__init__.py:450) (through line 496).

- **variant_c**: CpuPlatform attempts vllm._C on x86 with both AVX512 and AVX512-BF16 support, or on non-x86. ImportError is caught and warned. No actual CPU capability or load attempt was observed.
  Evidence: [vllm/platforms/cpu.py:657](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/cpu.py:657) (through line 672), [vllm/platforms/cpu.py:689](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/cpu.py:689) (through line 693).

- **variant_avx512**: CpuPlatform attempts vllm._C_AVX512 on x86 with AVX512 but without AVX512-BF16 support. It suppresses the specific missing module export function ImportError and warns for other ImportErrors. The source comment that the library was imported is not a runtime observation or ABI qualification.
  Evidence: [vllm/platforms/cpu.py:657](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/cpu.py:657) (through line 680).

- **variant_avx2**: CpuPlatform attempts vllm._C_AVX2 on x86 without AVX512 support. It suppresses the same missing module export function ImportError and warns for other ImportErrors. A normal return cannot establish Python-module success, retained ELF mapping, registrations, or ABI.
  Evidence: [vllm/platforms/cpu.py:657](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/cpu.py:657) (through line 665), [vllm/platforms/cpu.py:681](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/cpu.py:681) (through line 688).

- **other_extension_callers_absent**: A complete textual scan of exactly the five authenticated Python sources finds zero fs_io_C and zero spinloop references. Their callers, activation conditions, and original reachability remain outside this source set. This is not proof the extensions are unused.

- **libgomp_preload_policy**: CpuPlatform.check_and_update_config sets the worker method to spawn and, on supported Linux architectures with no libomp/libgomp/libiomp substring in existing LD_PRELOAD, searches torch.libs before torch/lib, takes the first glob candidate, and appends it to LD_PRELOAD. Glob order within each directory is not explicitly sorted. Torch is already imported in this module. Updating os.environ does not retroactively apply startup LD_PRELOAD to the current process; future exec inheritance and actual binding still need observation. Reaching this method is unproven here.
  Evidence: [vllm/platforms/cpu.py:233](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/cpu.py:233) (through line 239), [vllm/platforms/cpu.py:426](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/cpu.py:426) (through line 426), [vllm/platforms/cpu.py:459](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/cpu.py:459) (through line 502).

## Complete edge mapping

| ID | Requesting object | Needed object | What source proves | Actual choice |
|---|---|---|---|---|
| V01 | `_C.abi3.so` | `libc10.so` | Conditional variant_c route after torch import | Unresolved |
| V02 | `_C.abi3.so` | `libtorch_cpu.so` | Conditional variant_c route after torch import | Unresolved |
| V03 | `_C.abi3.so` | `libtorch.so` | Conditional variant_c route after torch import | Unresolved |
| V04 | `_C_AVX2.abi3.so` | `libc10.so` | Conditional variant_avx2 route after torch import | Unresolved |
| V05 | `_C_AVX2.abi3.so` | `libtorch_cpu.so` | Conditional variant_avx2 route after torch import | Unresolved |
| V06 | `_C_AVX2.abi3.so` | `libtorch.so` | Conditional variant_avx2 route after torch import | Unresolved |
| V07 | `_C_AVX512.abi3.so` | `libc10.so` | Conditional variant_avx512 route after torch import | Unresolved |
| V08 | `_C_AVX512.abi3.so` | `libtorch_cpu.so` | Conditional variant_avx512 route after torch import | Unresolved |
| V09 | `_C_AVX512.abi3.so` | `libtorch.so` | Conditional variant_avx512 route after torch import | Unresolved |
| V10 | `fs_io_C.abi3.so` | `libtorch_cpu.so` | Caller absent from five-source set | Unresolved |
| V11 | `fs_io_C.abi3.so` | `libtorch.so` | Caller absent from five-source set | Unresolved |
| V12 | `spinloop.abi3.so` | `libtorch_cpu.so` | Caller absent from five-source set | Unresolved |
| V13 | `spinloop.abi3.so` | `libtorch.so` | Caller absent from five-source set | Unresolved |
| G01 | `libshm.so` | `libgomp.so.1` | Conditional future-worker LD_PRELOAD policy | Unresolved |
| G02 | `libtorch_cpu.so` | `libgomp.so.1` | Conditional future-worker LD_PRELOAD policy | Unresolved |

The JSON preserves each original row unchanged, including candidate references, RUNPATH metadata, and per-edge next observations. The PyTorch libgomp candidate is an authenticated original inventory/wheel-member reference; the system candidate is only a new current-byte observation, with no original full payload hash. Neither records which object was loaded.

## Finite next diagnostic proposal

1. Complete source-only flags/caller evidence: torch/_utils_internal.py, vllm/env_override.py, relevant vllm/envs.py definitions, original entrypoint imports, and exact fs_io_C/spinloop caller matches. Inspect ZenCpuPlatform only if the actual selection can reach it.
2. After separate parent review, use one fresh interpreter with the original import order and environment, a 120-second limit and 32 MiB loader-trace limit. Observe actual platform/CPU dispatch, requesting-object to loaded-provider identity, relevant startup variables, import outcomes, and any genuine worker exec. Load no model weights and send no generation requests.
3. Preserve failures and unobserved paths. Do not force alternative AVX modules or preselect libgomp. Mapped-library co-presence is insufficient to bind an edge. A new diagnostic cannot reconstruct historical loader selection from source order alone. Dependency selection and ABI/model correctness remain separate.

## Authenticated sources

| Saved source | Bytes | SHA-256 |
|---|---:|---|
| [torch/__init__.py](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/torch/__init__.py) | 130530 | `bd4a10ff16357ed78f45aaf4b46ce5b771fdc0e8549402206e324d0eb792c672` |
| [vllm/__init__.py](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/__init__.py) | 3662 | `087e15875e49ba50f4969c2229d450bb8397302ea9344b3a3bd06ee400263ebd` |
| [vllm/_custom_ops.py](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/_custom_ops.py) | 141872 | `6fdd0e901ab202121edcfd968d0d3d79628b67bde5e1bb0fcac30c91ec9f7d4f` |
| [vllm/platforms/__init__.py](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/__init__.py) | 12349 | `1cb18cae8c7ad58f9f8c052d5fad11ab0bec68f2fac01b6912235e9615635de5` |
| [vllm/platforms/cpu.py](/mnt/nvme/dory-loader-route-sources-r1-20261111/sources/vllm/platforms/cpu.py) | 33077 | `bfa74a5c08f4186b57a3bb83cf1c08fbe1bf35b5522c02020fc1d602dca12f50` |

All five files (321,490 bytes) were fully rehashed and AST-parsed without execution. Input metadata hashes:

- [remaining](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-remaining-loader-gaps-after-cache-20261111-v1.json): `41e6d4600ec218dfc56e2555333795c68fc3a5e4e4b279142356ae6877583681`
- [source_collection](/mnt/nvme/dory-loader-route-sources-r1-20261111/result.json): `5fae7953b883f17949b6c1438e78957558f7ba04604e5315339ed3580a733481`
- [source_collection_root_terminal](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-loader-route-sources-r1-root-actual-exit-20261111-v1.json): `5cc698cc618c69a6e90f3d40da9c484a4940f38f9620b4974afa502cb0d5f1d2`

Machine-readable mapping: [JSON](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-loader-route-source-analysis-20261111-v1.json).
