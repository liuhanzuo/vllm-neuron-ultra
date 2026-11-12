"""Validate fresh API startup observations before comparing timed arms.

Pure JSON helpers: no process reads, launches, native imports, or historical
receipt repair. Callers supply real observations; mocks remain mocks.
"""
import copy
from pathlib import Path

from qwen35_sampler_pair_protocol_v1 import expected_case, host_identity, require
from qwen35_sampler_runtime_contract_v12 import (
    PYTHON, control_root, digest, native_environment, phase_environment,
)
from qwen35_sampler_worker_v1 import worker_name

# These values intentionally identify a particular arm or fresh output tree.
# Each is checked against its exact original binding BEFORE being omitted
# from the comparable settings. No other environment key is ignored.
BOUND_ENVIRONMENT_KEYS = (
    "QWEN35_SAMPLER_HOST_IDENTITY", "QWEN35_SAMPLER_HOST_RECEIPT",
    "TMPDIR", "VLLM_RPC_BASE_PATH", "QWEN35_SAMPLER_CAPTURE_ROOT",
)


def observed_settings(environment, argv, case, identity, directory, expected_argv):
    """Return comparable settings only for the exact clean original launch."""
    require(case == expected_case(), "Original shared model case required")
    require(type(identity) is dict and identity == host_identity(
        identity["arm"], identity["phase"], identity["source_sha256"]),
        "Exact unchanged worker host identity")
    directory = Path(directory)
    arm, host_phase = identity["arm"], identity["phase"]
    phase = directory.name
    require(phase in ("capture", "quality", "http") and directory.parent.name == arm
            and host_phase == ("correctness_capture" if phase == "capture" else "measured"),
            "Exact arm/phase directory binding")
    root = directory.parent.parent
    prefix = "qwen35-sampler-pair-"
    require(root.parent == Path("/mnt/nvme/neuron-work") and root.name.startswith(prefix)
            and directory.is_absolute() and ".." not in directory.parts,
            "Fresh original sampler output namespace")
    spec = dict(run_name=root.name[len(prefix):], output_root=str(root))
    control = control_root(spec) / (arm + "-" + phase)
    wanted = native_environment(case, host_identity=identity,
        host_receipt=directory / "worker-attachment.json",
        capture_root=directory / "packets" if phase == "capture" else None)
    wanted = phase_environment(spec, dict(arm=arm, phase=phase, control=str(control)), wanted)
    require(type(environment) is dict and all(type(k) is str and type(v) is str
            for k, v in environment.items()) and environment == wanted,
            "Actual API environment differs from the complete original clean launch")
    require(type(argv) is list and all(type(x) is str for x in argv)
            and argv == expected_argv and argv[0] == PYTHON
            and argv[-2:] == ["--worker-cls", worker_name(arm, host_phase)]
            and argv.count("--worker-cls") == 1,
            "Actual API argv differs from exact original worker launch")
    return dict(schema="qwen35_sampler_effective_host_settings_v1",
        environment={key: value for key, value in environment.items()
                     if key not in BOUND_ENVIRONMENT_KEYS},
        api_argv=copy.deepcopy(argv[:-2]),
        excluded_environment_bindings=list(BOUND_ENVIRONMENT_KEYS),
        excluded_argv_binding="--worker-cls", phase=host_phase)


def from_live_observation(live, case, identity, directory, expected_argv):
    """Recompute the producer's fields; absent, stale or invented data fails."""
    require(live["observation_kind"] == "live_ready" and live["ready"] is True
            and live["server_active"] is True and live["exit_observed"] is False,
            "Effective settings require the actual live API observation")
    proof = live["startup_config_proof"]
    environment, argv = proof["actual_api_initial_environment"], proof["actual_api_argv"]
    require(digest(environment) == proof["actual_api_initial_environment_sha256"]
            and argv == live["api_process_owner"]["argv"],
            "Observed environment hash and actual process argv must agree")
    result = observed_settings(environment, argv, case, identity, directory, expected_argv)
    require(proof["effective_host_settings"] == result,
            "Recorded effective settings disagree with complete observed startup values")
    return result
