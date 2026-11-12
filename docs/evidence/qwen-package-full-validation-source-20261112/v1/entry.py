"""Root-callable package leaves. Every native stage belongs in its own process.

The existing root owns admission, full History, resource ceilings, process
creation and draining. This module has no lock, queue, subprocess or retry.
"""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .runtime import (ROOT, BASE, BUCKETS, configuration, validate_case, read,
                      require, require_execution, check_environment, write_new)

MODEL_FILES = {'state_copy': 'state-copy.json', 'numeric': 'real-logits-validation.json',
               'cold': 'cold-buckets.json', 'long': 'long-decode-validation.json'}
PAYLOADS = (('numeric', 'real-logits-validation.json.pt', 'real-logits-validation.json', 32),
            ('cold', 'cold-buckets.pt', 'cold-buckets.json', 84),
            ('long', 'long-decode-validation.pt', 'long-decode-validation.json', 87))


def prepare_view(case, execution_guard):
    require_execution(execution_guard)
    validate_case(case)
    from vllm_neuron.models.qwen35.prepare_view import prepare
    return prepare('/mnt/nvme/models/Qwen3.5-2B', case['model_path'])


def build_fresh(case, output, execution_guard):
    require_execution(execution_guard)
    check_environment(case, compile_allowed=True)
    require(not Path(output).exists(), 'Fresh build inventory required')
    for key in ('artifact', 'compiler_cache', 'compiler_workdir'):
        path = Path(case[key])
        require(not path.exists() and not path.is_symlink() and path.parent.is_dir(),
                'Fresh independent package compilation path required: ' + key)
    # Existing package entry constructs the real model, serially compiles and
    # loads it, then publishes its complete independent artifact manifest.
    from vllm_neuron.models.qwen35.build import build
    from vllm_neuron.models.qwen35.identity import ARTIFACT_MANIFEST, digest_file
    from vllm_neuron.models.qwen35.state import clear_states
    from .state_observation import graph_routes, raw_zero_states
    model = build(case['model_path'], configuration())
    routes = graph_routes(model)
    reset = clear_states(model.traced_model.nxd_model)
    layout = raw_zero_states(model)
    compiled = []
    for tag in ('context_encoding_model', 'token_generation_model'):
        for index, bucket in enumerate(BUCKETS):
            path = Path(case['compiler_workdir']) / tag / ('_tp0_bk' + str(index)) / 'graph.neff'
            require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                    'Missing fresh compiled bucket: ' + str(path))
            compiled.append(dict(tag=tag, index=index, bucket=bucket, path=str(path),
                                 bytes=path.stat().st_size, sha256=digest_file(path)))
    manifest = Path(case['artifact']) / ARTIFACT_MANIFEST
    result = dict(complete=True, case=case, artifact_absent_before=True, cache_absent_before=True,
        workdir_absent_before=True, artifact_manifest_sha256=digest_file(manifest),
        graph_routes=routes, compiled_graph_files=compiled, state_layout=layout, state_reset=reset,
        external_compiler_exit_audit_required=True, full_model_qualified=False, performance_measured=False)
    write_new(output, result)
    return result


def run_model_stage(phase, directory, build_inventory, execution_guard):
    require_execution(execution_guard)
    directory = Path(directory)
    case = validate_case(read(directory / 'case.json'))
    check_environment(case)
    require(phase in MODEL_FILES, 'Unknown full-model stage')
    output = directory / MODEL_FILES[phase]
    require(not output.exists() and not output.with_suffix('.pt').exists()
            and not Path(str(output) + '.pt').exists(), 'Fresh stage output required')
    from .runtime import read_build_inventory
    require(read_build_inventory(build_inventory)['case'] == case, 'New build case mismatch')
    if phase == 'numeric':
        from .numeric import neuron
        neuron(SimpleNamespace(model=case['model_path'], artifacts=case['artifact'], adapter='qwen35_text',
            reference=str(ROOT / 'qwen35-2b-hf-reference.pt'), output=str(output)))
    elif phase == 'cold':
        from .cold import main
        main(directory)
    elif phase == 'long':
        from .long import neuron
        neuron(directory / 'case.json', directory)
    else:
        from .state_copy import main
        main(directory, build_inventory)
    from .checks import check_model_metadata, original_validators
    return check_model_metadata(read(output), phase, original_validators(), case)


def compare_vectors(directory, execution_guard):
    require_execution(execution_guard)
    directory = Path(directory)
    case = validate_case(read(directory / 'case.json'))
    from .checks import check_model_metadata, original_validators
    from qwen35_sampler_full_checks_v1 import check_model_metadata as check_historical
    import torch
    torch.set_num_threads(1)
    validators = original_validators()
    phases = []
    for phase, payload, metadata, count in PAYLOADS:
        require_execution(execution_guard)
        check_historical(read(BASE / metadata), phase, validators)
        check_model_metadata(read(directory / metadata), phase, validators, case)
        old = torch.load(BASE / payload, weights_only=True, map_location='cpu')
        new = torch.load(directory / payload, weights_only=True, map_location='cpu')
        require(len(old) == len(new) == count, 'Raw frame count changed')
        for a, b in zip(old, new):
            require(set(a) == set(b) and {k: v for k, v in a.items() if k != 'logits'}
                    == {k: v for k, v in b.items() if k != 'logits'}, 'Raw frame identity changed')
            x, y = a['logits'], b['logits']
            require(x.shape == y.shape == (248320,) and x.dtype == y.dtype == torch.float32,
                    'Raw vector geometry changed')
            require(torch.isfinite(x).all().item() and torch.isfinite(y).all().item(), 'Nonfinite vector')
            require(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)),
                    'Fresh package vector does not reproduce historical bits')
        phases.append(dict(phase=phase, frames=count, identical_vectors=count))
        del old, new
    return dict(complete=True, frames=203, identical_vectors=203, phases=phases,
                full_model_qualified=False, fresh_execution_proven=False)


if __name__ == '__main__':
    print(json.dumps(dict(entry='Root imports this module and supplies its current execution_guard',
        calls=['prepare_view(case, guard)', 'build_fresh(case, output, guard)',
               'run_model_stage(phase, directory, build_inventory, guard)',
               'compare_vectors(directory, guard)'], native_execution=False), indent=2))
