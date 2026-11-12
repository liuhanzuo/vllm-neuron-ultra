# Slot serial v3: exact SDK environment and context restoration

The frozen v2 serial adapter cannot accept successful SDK initialization: its periodic leaf guard compares the entire environment with the clean launch dictionary, while torch_xla and torch_neuronx add required defaults. A second installed SDK bug leaves PJRT_DEVICE=CPU and a literal `key=NEURON` variable after trace. v3 fixes both at the adapter boundary. The original primitive protocol, device leaf, compiler arguments, layouts, calls, comparisons and scientific interpretation remain unchanged.

The first failure is reproduced without SDK/native imports using actual pinned torch_xla AST functions `_setup_default_env`, `_set_missing_flags` and `_setup_xla_flags`. They add TF_CPP_MIN_LOG_LEVEL=1, GRPC_VERBOSITY=ERROR and the exact XLA_FLAGS string, then the actual v2 check rejects `Exact clean slot environment`. The initial independent reproduction was tool terminal `65a364`, exit 0; the reproducible cases are included in the v3 suite.

The second failure is in installed `torch_neuronx/contexts.py:35`: `os.environ.update(key=value)` writes the word `key` instead of the variable's name. Frozen leaf `run_case` calls `torch_neuronx.trace` with its unchanged default `cpu_backend=True`. Actual `generate_hlo` enters `mock_neuron_cores()` and `set_pjrt_device(True)`. Actual SDK context AST reproduces both normal and exception exits leaving PJRT_DEVICE=CPU and key=NEURON when the original value was NEURON. This is a third-party SDK context restoration bug, not permission to accept a CPU device or arbitrary environment drift.

`qwen35_predicated_slot_sdk_environment_v1.py` pins 18 full installed source files. The current source versions are torch 2.8.0+cu128, installed torch metadata 2.8.0 and torch_xla 2.8.1. The source audit inventories 85 environment-write sites in 23 relevant Python files and distinguishes active initialization defaults from conditional distributed, TPU, CUDA, debug and stochastic-rounding paths.

Each leaf starts with the unchanged exact clean dictionary. At the original `before_native_imports` callback, the adapter records the transition into SDK initialization. At the first original `before_compile` callback it requires the entire dictionary to equal the clean input plus these exact defaults:

| Variable | Required value |
| --- | --- |
| TF_CPP_MIN_LOG_LEVEL | 1 |
| GRPC_VERBOSITY | ERROR |
| XLA_FLAGS | ` --xla_cpu_enable_fast_math=false --xla_gpu_force_compilation_parallelism=8` (one leading space from the original SDK function) |
| XLA_DISABLE_FUNCTIONALIZATION | 1 |
| XLA_IR_SHAPE_CACHE_SIZE | 20480 |
| PJRT_DEVICE | NEURON |
| NEURON_LIBRARY_PATH | `/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/libneuronpjrt.so` |
| NEURON_RT_ROOT_COMM_ID | localhost:62182 |
| ALLREDUCE_GRADIENTS_BUCKET_SIZE_MB | 50 |
| JAX_LOGGING_LEVEL | NOTSET |

There is one finite optional branch: PJRT_DYNAMIC_PLUGINS may be absent or equal 1 according to torch_xla's native build flag. Its actual presence is fixed at the first post-import boundary. Later addition/removal/value changes fail. No prefix-wide NEURON/XLA allowance exists. Every original clean variable must remain unchanged, and every unknown additional key fails. Source-only tests execute the actual pinned initialization functions with only OS/helper/native ports mocked and verify all ten resulting defaults exactly.

After initialization, every leaf callback checks the complete fixed environment, independently of the original half-second expensive process/cgroup/ACK polling. Process flags, isolated bytecode prefix, CPU affinity and nice priority are checked through the same original predicates. The stdlib supervisor retains the original exact-clean check throughout. Each subsequent layout is still a fresh child with its own original clean environment.

The context correction is installed once in memory on the exact imported SDK module after source verification and initialization. It accepts only the original `mock_neuron_cores` and nested `set_pjrt_device` code objects with their exact dictionaries. Temporary mock-TPU variables and PJRT_DEVICE=CPU exist only inside those original synchronous lowering contexts. The corrected `finally` restores each real saved key, removes keys originally absent, and verifies the full environment before and after. It preserves the original tracing exception if another environment error occurs while unwinding, then retains the environment failure for subsequent checks. Unknown keys and mutations of fixed inputs are rejected, never silently normalized. The original SDK context function is restored when the leaf returns or fails. No installed package file is modified.

Only `layout` changes among serial execution functions after normalizing schema version strings; outer acquisition/history, actual ACK, Popen capture/wait, eight-layout schedule, resource monitoring, Service cleanup, retained drain and external reconciliation are unchanged. The successor schema names are `qwen35_predicated_slot_serial_spec_v3`, layout binding v3, admission v3, terminal v3 and reconciliation v3. The review envelope remains the original root source-review shape. Root must bind from the new generated inventory; no current spec or handoff is created by this delivery.

All original limits remain: 8 fresh sequential processes, 900 seconds/layout, 7,200 seconds/service, 9,000 seconds/outer, 6 GiB MemoryMax, 5 GiB MemoryHigh, zero swap, NC0, CPUs 6/7, 1 GiB/layout and 8 GiB total output. Initial NVMe free space is 40 GiB, root/NVMe floors 28/30 GiB, separate LFM requirement 54 GiB. Four graphs/layout, 32 graphs total and 46,368 explicit calls remain unchanged. Complete report/artifact and bytewise state validation are unchanged.

SDK inventory now also includes all 15 libneuronxla Python files, since its initialization code produces reviewed defaults. The complete delivery contains 618 local source pins and 2,822 SDK Python source pins. The five new files are the SDK environment helper, serial contract v3, serial runtime v3, preparer v3 and check suite v3. Frozen v2 and primitive files were rehashed unchanged.

Validation: 33 source/mock tests passed, session 34646 / terminal 9dfd5b / exit 0. They preserve the original 23 serial/resource/report tests and add actual installed SDK AST reproductions, exact initialization, negative environment mutations, repeated normal/exception context restoration, unknown callers, error precedence, and real `layout` integration with the actual environment owner and actual SDK context functions. No torch, torch_xla, torch_neuronx, numpy, libneuronxla, ctypes or _ctypes module was imported. No real process, lock, service, model or compiler ran.

Correction to the earlier v2 review scope: the old source suite blocked ML/native SDK roots but did not block ctypes. Its lifecycle test imported a historical module merely to mock `latest_handoff`; that import chain loads ctypes. Therefore the earlier broad statement that the v2 suite imported no native modules was inaccurate. No model/device/lock operation was performed. v3 blocks ctypes/_ctypes too and injects an explicit fake module for that already-mocked interface. The frozen v2 report is retained with this correction recorded separately.

The fixed environment is deliberately specific to the reviewed installed SDK sources. Native initialization and real compilation have not been rerun here. Any further native-created environment key will fail instead of being automatically allowed; such a failure requires its exact source/effect to be reviewed. These checks establish source correctness, not runtime budget fit, full-model qualification or speedup.
