"""Finite package configuration and admission calls; no scheduling or locking."""
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
BASE = ROOT / 'qwen35-state-reset-full-v1'
CHECKOUT = Path('/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107')
PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
BUCKETS = [128, 256, 512, 1024, 2048, 4096]
NEURON_KWARGS = dict(tp_degree=2, ctx_batch_size=1, batch_size=4,
    max_context_length=4096, seq_len=4096, enable_bucketing=True,
    is_continuous_batching=True, quantized=False, torch_dtype='bfloat16',
    padding_side='right', on_device_sampling_config=None, lora_config=None,
    pa_num_blocks=4, pa_block_size=4096,
    is_block_kv_layout=False, is_prefix_caching=False, apply_seq_ids_mask=True,
    context_encoding_buckets=BUCKETS, token_generation_buckets=None)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 2 * 1024**2, 'Small metadata required')
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def configuration():
    return copy.deepcopy(NEURON_KWARGS)


def validate_case(case):
    fixed = dict(name='qwen35-package-full-v2-20261112', served_name='qwen3.5-2b-text',
        batch=4, tp=2, max_model_len=4096, cpu=True, adapter='qwen35_text',
        package_schema='vllm-neuron-ultra-qwen35-split-v1',
        neuron_configuration=configuration())
    require(set(case) == set(fixed) | {'model_path', 'artifact', 'compiler_cache', 'compiler_workdir'},
            'Exact package case fields required; research provenance is not package admission')
    require(all(json.dumps(case[key], sort_keys=True, allow_nan=False)
                == json.dumps(value, sort_keys=True, allow_nan=False) for key, value in fixed.items()),
            'Package case or complete Neuron configuration changed')
    parents = dict(model_path='/mnt/nvme/models', artifact='/mnt/nvme/neuron-artifacts',
                   compiler_cache='/mnt/nvme/neuron-work', compiler_workdir='/mnt/nvme/neuron-work')
    paths = []
    for key, parent in parents.items():
        path = Path(case[key])
        require(path.is_absolute() and path.parent == Path(parent) and path.name.startswith('qwen35-package-full-v2-'),
                'Use a separate explicit package path: ' + key)
        require(not path.is_symlink(), 'Package path cannot alias a research directory')
        paths.append(path)
    require(len(set(paths)) == 4, 'Package view, artifact, cache and compiler workdir must differ')
    return case


def require_execution(execution_guard):
    require(callable(execution_guard), 'Existing root execution guard required')
    receipt = execution_guard()
    require(type(receipt) is dict and receipt.get('admitted') is True
            and receipt.get('exclusive') is True, 'Root has not admitted this stage')
    require(__debug__, 'Scientific assertions must remain enabled')
    return receipt


def check_environment(case, compile_allowed=False):
    validate_case(case)
    from vllm_neuron.models.qwen35.identity import selected_variant
    selected_variant()
    require(os.environ.get('NEURON_COMPILED_ARTIFACTS') == case['artifact'], 'Wrong package artifact')
    require(os.environ.get('NEURON_QWEN35_ALLOW_COMPILE', '0') == ('1' if compile_allowed else '0'),
            'Compile opt-in must be limited to the fresh build process')
    require(os.environ.get('NEURON_COMPILE_CACHE_URL') == case['compiler_cache'], 'Wrong compiler cache')
    require(os.environ.get('BASE_COMPILE_WORK_DIR') == case['compiler_workdir'], 'Wrong compiler workdir')
    require('cache' not in os.environ.get('NEURON_CC_FLAGS', '').lower(), 'No compiler cache redirect')


def load_admitted(model_path, artifact):
    """Build's existing-artifact branch performs all package admission checks."""
    require(os.environ.get('NEURON_QWEN35_ALLOW_COMPILE', '0') == '0', 'Validation cannot compile')
    require(os.environ.get('NEURON_COMPILED_ARTIFACTS') == str(artifact), 'Artifact environment differs')
    require(Path(artifact).is_dir() and not Path(artifact).is_symlink(), 'Admitted artifact must exist')
    from vllm_neuron.models.qwen35.build import build
    # The complete identity is freshly reconstructed from the package view,
    # original weights, SDK sources, Trn1 hardware, config and compile environment.
    return build(str(model_path), configuration())


def read_build_inventory(path):
    """Bind temporary copy states to actual new graph state layout, without loading graphs."""
    report = read(path)
    case = validate_case(report['case'])
    require(report['complete'] and report['artifact_absent_before'] and report['cache_absent_before']
            and report['workdir_absent_before'], 'Fresh package build evidence required')
    from vllm_neuron.models.qwen35.identity import ARTIFACT_MANIFEST
    manifest = Path(case['artifact']) / ARTIFACT_MANIFEST
    require(hashlib.sha256(manifest.read_bytes()).hexdigest() == report['artifact_manifest_sha256'],
            'Build inventory belongs to a different package artifact')
    from vllm_neuron.models.qwen35.validation import validate_states
    validate_states(report['state_layout'], zero=True)
    require(report['graph_routes']['complete'] and report['graph_routes']['buckets'] == BUCKETS
            and len(report['graph_routes']['graphs']) == len(report['compiled_graph_files']) == 12,
            'Actual twelve graph routes/compiled files required')
    return report


def server_arguments(case):
    validate_case(case)
    # Full override dictionary matches the standalone build constructor. In
    # this pinned platform, no prefix cache sets block_size=max_model_len;
    # loader defaults therefore give pa_num_blocks=4 and pa_block_size=4096.
    return [PYTHON, '-m', 'vllm_neuron.models.qwen35.serve',
        '--model', case['model_path'], '--served-model-name', case['served_name'],
        '--tensor-parallel-size', '2', '--max-model-len', '4096', '--max-num-seqs', '4',
        '--no-enable-prefix-caching', '--host', '127.0.0.1', '--port', '8010',
        '--additional-config', json.dumps({'override_neuron_config': configuration()})]
