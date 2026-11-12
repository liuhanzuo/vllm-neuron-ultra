"""Original observer-off native42/device84 leaves; no launcher or model load.

The admitted serial worker supplies an already validated original model, fresh
namespaces, complete source/weight/artifact bindings and a live execution guard.
Every arm uses its own fresh process; this module never compiles or loads a graph.
"""
import hashlib
from pathlib import Path


def require(value, message):
    if not value: raise ValueError(message)


def paths(destination, data_root, backend, execution_guard):
    require(callable(execution_guard), 'owning serial guard required'); execution_guard()
    require(backend in ('native', 'device'), 'original backend required')
    destination, data_root = map(Path, (destination, data_root))
    require(all(p.is_absolute() and p.resolve() == p and p.is_dir()
                and 'deepseek-cold-kv-diagnostic-v2' not in p.parts for p in (destination, data_root)),
            'independent ordinary existing roots required')
    require(destination != data_root and destination not in data_root.parents and data_root not in destination.parents,
            'disjoint metadata and payload roots required')
    targets = [destination/'cache-captures'/backend, data_root/backend]
    if backend == 'native': targets.append(destination/'cold-reference')
    else: targets += [destination/n for n in ('state-copy.json', 'state-lifecycle.json', 'lifecycle-states', 'cold-buckets.json', 'cold-buckets.pt')]
    require(all(not p.exists() and not p.is_symlink() and p.resolve() == p for p in targets),
            'fresh complete control outputs required')
    return destination, data_root


def reference(model, destination, data_root, execution_guard):
    destination, data_root = paths(destination, data_root, 'native', execution_guard)
    import torch
    from deepseek_lifecycle_full_protocol_v2 import build
    from qwen3_controls_reference import hf
    from validate_deepseek_cold_control_v1 import reference as original_loop
    torch.set_num_threads(2)
    protocol = build()
    digest = hashlib.sha256(Path(hf.__file__).read_bytes()).hexdigest()
    require(digest == protocol['native_source_sha256'] ==
            '64fec98db8cd0edaa740baf026be6adb1d5011e38bfa050ccdf6693b1da9f95d', 'original native source required')
    require(type(model) is hf.Qwen3ForCausalLM, 'already admitted original native model required')
    require(sum(p.numel() for p in model.parameters()) == 8190735360, 'full original parameter count')
    require(model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr(), 'untied original head')
    require(not model.training and str(model.dtype) == 'torch.bfloat16', 'original BF16 eval model required')
    execution_guard()
    directory = destination/'cold-reference'; directory.mkdir(exist_ok=False)
    original_loop(model, protocol, digest, directory, destination, data_root, execution_guard)
    execution_guard()
    return dict(native_frames=42, control_generated=True, original_raw_reproduction_pending=True,
                full_model_qualified=False, new_speedup_claim=False)


def device(model, case, destination, data_root, original_cold_metadata, execution_guard):
    destination, data_root = paths(destination, data_root, 'device', execution_guard)
    from deepseek_cold_kv_protocol_v2 import check_device_metadata, check_copy, check_lifecycle, expected_case, read
    from validate_deepseek_cold_control_v1 import main as original_loop
    require(case == expected_case(), 'original complete case required')
    require((destination/'cold-reference/cold-reference.json').is_file(), 'completed native reference required')
    execution_guard()
    try:
        original_loop(model, case, destination, destination/'cold-reference', data_root, execution_guard)
    except SystemExit as error:
        require(str(error) == 'Strict cold-bucket numerical/state gate failed', 'unexpected control termination')
    else:
        raise ValueError('original failed cold gate unexpectedly changed')
    execution_guard()
    # original_cold_metadata must be a caller-authenticated input. The exact
    # comparator retains every84 row/metric and all14 original failure indices.
    proof = check_device_metadata(read(destination/'cold-buckets.json'), original_cold_metadata)
    check_copy(read(destination/'state-copy.json'))
    lifecycle = read(destination/'state-lifecycle.json'); check_lifecycle(lifecycle)
    require(lifecycle['state_tensors_changed_by_warmup'] == 144, 'all original warmup transitions')
    execution_guard()
    return dict(device_frames=84, original_failed_gate_preserved=True, metadata_reproduction=proof,
                original_raw_reproduction_pending=True, full_model_qualified=False, new_speedup_claim=False)
