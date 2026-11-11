LFM3 full CPU successor v1: static import compatibility review, 2026-11-11

Author: `/root/lfm_after_writeback_design_20261111`. Scope: installed Python source and distribution metadata only. This record is not a native import result, simulator result, source acceptance, runtime admission, or model qualification. No frozen implementation was changed and no native test, simulator, model, process/service/lock operation, history resolution, current spec/catalog binding, or MiniPV session inspection was performed for this review.

Conclusion: a NumPy simulation call definitely attempts to import Torch after the original audit hook is installed. The bounded source review did not establish an import-time route that necessarily invokes a prohibited backend, compiler subprocess, or temporary write in the current installation. Actual import compatibility and completion within the unchanged total resource/time budget remain unverified. The pure tests and earlier r5 execution under a different audit do not establish either property.

The confirmed call chain is:

1. [run_batch](/home/ubuntu/experiments/trn1-latest-models-20261031/lfm3_nki05_full_cpu_leaf_v1.py:328) installs `legacy.p.audit_child` at line 330 before `execute_batch_science` at line 332. The latter imports NumPy/ml_dtypes at lines 195/196, and NKI at line 217 for controls/candidate. The simulation calls are at lines 236 and 261.
2. [SimulateKernel.__call__](/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/nki/framework/_simulate.py:129) calls `ctx.convert_args` at line 132 before `simulate_kernel` at line 134. `convert_args` evaluates `is_torch_tensor(val)` at line 63 for every positional argument via line 72, including NumPy arrays.
3. [is_torch_tensor](/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/nki/framework/_torch_convert.py:192) calls `_get_torch`, whose line 56 is `import torch`. Its exception handler at line 60 catches only `ImportError`, `RuntimeError`, and `AttributeError`. A direct audit `AssertionError` is not caught by that handler. A successful import is cached only within the current child; this does not remove the fresh-child import cost.
4. [Original audit](/home/ubuntu/experiments/trn1-latest-models-20261031/lfm3_typed_writer_cpu_pilot_protocol_v3.py:400) forbids imports rooted at `torch_neuronx`, `torch_xla`, `transformers`, `vllm`, `neuronx_distributed`, and `neuronx_distributed_inference`, and forbids subprocess/spawn/network audit events. Writes and filesystem mutations are restricted to the exact receipt and its temporary receipt. It sets `tempfile.tempdir` before installing the hook, so ordinary `gettempdir()` discovery is not itself evidence of an allowed temporary write. `.pyc` reads raise `FileNotFoundError`; integer open handles are limited to 0/1/2.

Backend autoload is a conditional path, not a demonstrated failure here. [Torch backend loading](/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/torch/__init__.py:2817) scans `torch.backends` entry points and is enabled by default at lines 2857/2877. [The clean runner environment](/home/ubuntu/experiments/trn1-latest-models-20261031/run_lfm3_nki05_full_cpu_serial_v1.py:281) does not set `TORCH_DEVICE_BACKEND_AUTOLOAD`. However, the static inventory of all 57 `*.dist-info/entry_points.txt` files in this Python 3.12 site found **zero** `[torch.backends]` sections. The installed torch_neuronx entry points contain console scripts only; torch_xla contains console scripts and `[torch_xla.plugins]`, a different group. `pyvenv.cfg:5` sets `include-system-site-packages = false`; the three site `.pth` files register editable vllm/vllm_neuron finders and the setuptools shim. These observations concern the inspected installation and do not claim that dynamic entry-point discovery was executed or that all future paths/distributions have been excluded. Inventory was re-read using stdlib only in tool result `fd7806`, exit 0.

Exception detail: Torch's backend loader wraps `Exception` raised by entry-point load/call in `RuntimeError`. If an audit assertion occurs inside that particular wrapper, NKI's `_get_torch` can catch the resulting `RuntimeError`. This differs from a direct audit assertion during another Torch import path. The empty inspected backend group makes neither exception route an observed runtime outcome.

Two possible side-effect routes were examined without proving plain Torch-import activation:

| Route | Verified installed source | Boundary of the finding |
| --- | --- | --- |
| Compiler search / ISA dry compile | [cpp_compiler_search](/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/torch/_inductor/cpp_builder.py:77) invokes `subprocess.check_output([cxx, "--version"])` at line 98; `get_cpp_compiler` at line 334 calls it on the ordinary non-Windows/non-fbcode path. `CppOptions.__init__` selects `get_cpp_compiler()` at line 971. [cpu_vec_isa](/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/torch/_inductor/cpu_vec_isa.py:20) has compiler fingerprinting, `check_build` at line 92, `__bool__impl` at line 146, and ISA selection functions at lines 449/493. | These are runtime helpers. The review did not establish their invocation by the initial `import torch`. Their presence does not prove an import-time compiler subprocess or write. |
| Distributed RemoteModule template | [distributed.nn](/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/torch/distributed/nn/__init__.py:6) imports RemoteModule if RPC is available. [remote_module](/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/torch/distributed/nn/api/remote_module.py:15) imports the instantiator and instantiates a template at module scope. [instantiator](/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/torch/distributed/nn/jit/instantiator.py:21) constructs `tempfile.TemporaryDirectory()` at module scope. | If this module is reached under the audit, its directory creation conflicts with the allowed paths. The review did not establish that initial `import torch` reaches `torch.distributed.nn`, or evaluate native RPC availability. |

No successor implementation or additional preflight is proposed by this record. Per root direction, preserve the frozen execution order: `original_fixture` 3716 cases/291 children, `new_fixture` 3716/291, one separate three-arm controls child, and `candidate` 3716/291; **874 children total**. Keep AS 12 GiB, MemoryMax 10 GiB/MemoryHigh 8 GiB/no swap, TasksMax 32, CPU/nice and storage caps, the original audit, 1800-second batch / 14400-second phase / 30000-second whole-run limits, and all downstream scientific/device/alias/cold-cache/long-decode/slot-reuse/quality/HTTP gates. This review neither proves that repeated imports fit those limits nor relaxes them.

SHA-256 snapshots read for this review:

| Source | SHA-256 |
| --- | --- |
| `lfm3_nki05_full_cpu_leaf_v1.py` | `6489f9262fba72c223f2f4d96b1d95659fe98d2b6bc3d1f267ac417d565c08c9` |
| `run_lfm3_nki05_full_cpu_serial_v1.py` | `0d40699c4d70f0c8287730dcd1b8c56f7e550ba210e7a5147ba37c51b4eeed8d` |
| `check_lfm3_nki05_full_cpu_successor_v1.py` | `a1bfa6920235a7a39f08a088dd26e7ed93e8fb640e11b5b97bd7fb2b3e05cfd1` |
| `lfm3_typed_writer_cpu_pilot_protocol_v3.py` | `8cbf6d4f11c1abc753e65348c0b492527a14c63cae8f6ad7c8828ec103c45e87` |
| installed `nki/framework/_simulate.py` | `187dbb5779c8b57b5012ceb013ad9845ed18b864be9e7c813a92998481a40201` |
| installed `nki/framework/_torch_convert.py` | `2c510075ca5943ce655f3edd21553398ae6d83d05194ddaca21d8da3ff8ea729` |
| installed `torch/__init__.py` | `3caf7f40140ede2465bde40b9003af10cbbc8f7bcf436fa1de026471daa1b288` |
| installed `torch/distributed/nn/__init__.py` | `4dae25d09af680e2bb79047e7b7dacb5668328f3a557f2a28b4484c7275793f4` |
| installed `torch/distributed/nn/api/remote_module.py` | `d51779fdbafb1ca1d99d002e6f32cf17bc1fef730a6251e3ef0278d8078bd112` |
| installed `torch/distributed/nn/jit/instantiator.py` | `567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615` |
| installed `torch/_inductor/cpp_builder.py` | `82642e8f09e48e4fd9797a7d5e1a7f6b5f37425fdb0d47567866df45ec2d0ff2` |
| installed `torch/_inductor/cpu_vec_isa.py` | `bb42109a25611b13373859fdb91e8ac02f968610e41e3f1a0a1574cc786f7c49` |

Installed package paths above are under `/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages`. Project source paths are under `/home/ubuntu/experiments/trn1-latest-models-20261031`.
