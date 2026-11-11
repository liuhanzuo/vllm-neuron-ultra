"""Pure proposal for a NEW controlled Dory import observation environment.

No filesystem, process, environment mutation, target import, or launch API.
The result does not reconstruct unknown historical flags or grant admission.
Known settings and the eleven imports come from the pinned original resolver.
Actual CPU dispatch remains automatic. A future independently reviewed worker
must stop after these imports, before upstream tests/tokenizer/model execution.
"""

RESOLVER_SHA256 = "e872f26fafd78e72b77847353a782a526e4093d6575ee033b94c4d4298bd1eb9"
BASE = "/mnt/nvme/dory-official-cpu-20261108-v1"
ENV = "/mnt/nvme/venvs/dory-official-cpu-20261108-v1"
IMPORT_PREFIX = (
    "torch", "pytest", "transformers", "tokenizers", "vllm", "vllm.config",
    "vllm.distributed.parallel_state", "vllm.model_executor.models.dory",
    "vllm.model_executor.models.utils", "vllm.transformers_utils.configs.dory",
    "vllm._custom_ops",
)

# Absence is an explicit choice for THIS new diagnostic, not historical evidence.
# In particular, the string "0" is truthy for TORCH_USE_RTLD_GLOBAL.
EXPLICITLY_ABSENT = (
    "TORCH_USE_RTLD_GLOBAL", "LD_LIBRARY_PATH", "LD_AUDIT", "LD_BIND_NOW",
    "LD_BIND_NOT", "LD_DYNAMIC_WEAK", "LD_DEBUG_OUTPUT", "LD_PROFILE",
    "LD_PROFILE_OUTPUT", "LD_ORIGIN_PATH", "LD_USE_LOAD_BIAS", "LD_HWCAP_MASK",
    "GLIBC_TUNABLES", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
    "VLLM_CUDA_COMPATIBILITY_PATH",
)


def diagnostic_environment():
    """Return the entire proposed exec environment, with no inherited keys.

    LD_DEBUG changes observation only and must be declared in the future review.
    Its text is not, by itself, proof of DT_NEEDED requester/provider selection.
    """
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/ubuntu",
        "LANG": "C.UTF-8",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONOPTIMIZE": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "XDG_CACHE_HOME": BASE + "/cache",
        "HF_HOME": BASE + "/cache/huggingface",
        "TMPDIR": BASE + "/tmp",
        "TRITON_CACHE_DIR": BASE + "/cache/triton",
        "TORCH_EXTENSIONS_DIR": BASE + "/cache/torch",
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:"
        + ENV + "/lib/libiomp5.so",
        "VLLM_TARGET_DEVICE": "cpu",
        "VLLM_PLUGINS": "",
        "VLLM_LOGGING_LEVEL": "INFO",
        "VLLM_CPU_KVCACHE_SPACE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_USE_SPINLOOP_EXT": "0",
        "VLLM_ENABLE_CUDA_COMPATIBILITY": "0",
        "LD_DEBUG": "files,libs,versions,bindings",
    }


def validate_proposal(environment, import_prefix):
    """Check proposal consistency only. This never authenticates native files."""
    if type(environment) is not dict or any(
        type(k) is not str or type(v) is not str for k, v in environment.items()
    ):
        raise ValueError("Exact string environment dictionary required")
    if environment != diagnostic_environment():
        raise ValueError("The new controlled environment must be reviewed as a whole")
    if type(import_prefix) not in (list, tuple) or tuple(import_prefix) != IMPORT_PREFIX:
        raise ValueError("Only the unchanged original eleven-import prefix is proposed")
    return {
        "proposal_consistent": True,
        "environment_kind": "new_explicit_controlled_diagnostic",
        "historical_environment_reproduced": False,
        "historical_bindings_replayed": False,
        "native_content_authenticated_here": False,
        "loader_binding_established": False,
        "runtime_admitted": False,
        "native_ABI_qualified": False,
        "full_model_qualified": False,
        "new_speedup_claim": False,
    }
