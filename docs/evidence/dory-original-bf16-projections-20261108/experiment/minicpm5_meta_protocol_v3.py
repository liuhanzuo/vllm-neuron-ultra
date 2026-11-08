"""Independent MiniCPM5 meta admission; root supplies the serial queue plan."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import minicpm5_meta_alias_contract_v3 as alias_contract
import minicpm5_meta_padding_v3 as padding

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'minicpm5-meta-v3'
INPUT = ROOT / 'research/minicpm5-2b-readiness-20261104-v1'
SCRIPT = ROOT / 'run_minicpm5_meta_worker_v3.py'
REVISION = 'f97400052a43d642bbc6e9975e2397e3ae6a6b52'
MODEL = 'openbmb/MiniCPM5-2B'
PHASES = ('admit', 'native', 'tp1-rank0', 'tp2-rank0', 'tp2-rank1')
JOBS = {'tp1-rank0': (1, 0), 'tp2-rank0': (2, 0), 'tp2-rank1': (2, 1)}
BUCKETS = [128, 256, 512, 1024, 2048, 4096]
LOGICAL_CACHE_LENGTH = 4096
PHYSICAL_CACHE_LENGTH = 4224  # SDK adds 128 positions when apply_seq_ids_mask=True.
UNITS = {x: 'trn1-latest-verify-minicpm5-meta-v3-' + x for x in PHASES}
RESOURCES = {x: (4, 3, 0, 900) if x == 'admit' else (8, 6, 0, 3600) for x in PHASES}
SDK_PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
HF_PYTHON = '/home/ubuntu/venvs/model_reference_2026/bin/python'


def read(path):
    path = Path(path)
    assert path.stat().st_size < 1024 * 1024
    return json.loads(path.read_text())


def sha(path):
    path = Path(path)
    assert path.stat().st_size < 1024 * 1024
    assert path.suffix in {'.py', '.json', '.jinja', '.md', '.pth'}
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def process_identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def heavy_units():
    result = subprocess.check_output(['systemctl', '--user', 'list-units', '--type=service',
        '--state=running', '--plain', '--no-legend', '--no-pager'], text=True)
    names = [line.split()[0] for line in result.splitlines() if line.split()]
    return sorted(x for x in names if x.startswith(('trn1-bench-', 'trn1-latest-verify-', 'trn1-quant-', 'trn1-direct-'))
        or (x.startswith('trn1-') and x.endswith('-worker.service')))


def verify():
    assert __debug__
    plan = read(DEST / 'plan.json')
    assert plan['schema'] == 'minicpm5-meta-v3' and plan['checkpoint_revision'] == REVISION
    assert plan['model'] == MODEL and plan['phases'] == list(PHASES) and plan['buckets'] == BUCKETS
    assert plan['thinking'] is False and plan['weight_payload_bytes_allowed'] == 0
    assert isinstance(plan['predecessors'], list)
    assert plan['source_sha256']
    from prepare_minicpm5_meta_v3 import source_paths
    assert set(plan['source_sha256']) >= {str(path) for path in source_paths()} | {plan['queue_script']}
    for name, expected in plan['source_sha256'].items():
        assert sha(name) == expected, name
    return plan


def guard(phase):
    """Reject all unowned native imports, including direct root invocations."""
    assert __debug__ and phase in PHASES
    expected_python = '/usr/bin/python3' if phase == 'admit' else HF_PYTHON if phase == 'native' else SDK_PYTHON
    assert Path(sys.executable).resolve() == Path(expected_python).resolve()
    if phase != 'admit':
        assert Path(sys.prefix) == Path(expected_python).parent.parent
    assert os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1'
    assert os.environ.get('NXD_CPU_MODE') == '1' and os.environ.get('PYTHONOPTIMIZE') == '0'
    unit = UNITS[phase] + '.service'
    props = dict(line.split('=', 1) for line in subprocess.check_output(['systemctl', '--user', 'show', unit,
        '--property=MainPID,ActiveState,SubState,MemoryMax,MemoryHigh,MemorySwapMax,Nice'], text=True).splitlines())
    assert int(props['MainPID']) == os.getpid()
    assert props['ActiveState'] == 'active' and props['SubState'] == 'running'
    assert Path('/proc/self/cgroup').read_text().strip().endswith('/' + unit)
    args = [x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x]
    index = args.index(str(SCRIPT))
    assert args[index:] == [str(SCRIPT), phase]
    assert heavy_units() == [unit]
    maximum, high, swap, _ = RESOURCES[phase]
    assert [int(props[k]) for k in ('MemoryMax', 'MemoryHigh', 'MemorySwapMax')] == [x * 1024**3 for x in (maximum, high, swap)]
    assert int(props['Nice']) == 19 and os.sched_getaffinity(0) == {7}
    plan = verify()
    owner = read(DEST / 'queue-identity.json')
    assert owner['unit'] == plan['queue_unit'] and owner['script'] == plan['queue_script']
    assert process_identity(owner['pid']) == str(owner['start_ticks'])
    assert str(owner['script']).encode() in Path('/proc', str(owner['pid']), 'cmdline').read_bytes().split(b'\0')
    queue_pid = int(subprocess.check_output(['systemctl', '--user', 'show', owner['unit'], '--property=MainPID', '--value'], text=True))
    assert queue_pid == owner['pid']
    for prior in plan['predecessors']:
        assert process_identity(prior['pid']) != str(prior['start_ticks']), prior
    admission = read(DEST / 'serial-admission.json')
    assert admission['passed'] and admission['all_predecessors_drained']
    assert admission['plan_sha256'] == sha(DEST / 'plan.json')
    if phase != 'admit':
        receipt = read(DEST / 'admission.json')
        assert receipt['passed'] and receipt['plan_sha256'] == sha(DEST / 'plan.json')
    return plan


def command(phase):
    """Return only a command; never start/enroll a job here."""
    assert phase in PHASES
    maximum, high, swap, timeout = RESOURCES[phase]
    python = '/usr/bin/python3' if phase == 'admit' else HF_PYTHON if phase == 'native' else SDK_PYTHON
    return ['systemd-run', '--user', '--quiet', '--wait', '--collect', '--unit=' + UNITS[phase],
        '-p', f'MemoryMax={maximum}G', '-p', f'MemoryHigh={high}G', '-p', f'MemorySwapMax={swap}G',
        '-p', 'AllowedCPUs=7', '-p', 'CPUAffinity=7', '-p', 'Nice=19', '-p', f'RuntimeMaxSec={timeout}',
        '-p', 'TimeoutStopSec=180', '-p', 'WorkingDirectory=' + str(ROOT),
        '-p', 'StandardOutput=append:' + str(DEST / (phase + '.log')), '-p', 'StandardError=inherit',
        '--setenv=PYTHONOPTIMIZE=0', '--setenv=PYTHONNOUSERSITE=1', '--setenv=HF_HUB_OFFLINE=1',
        '--setenv=TRANSFORMERS_OFFLINE=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1',
        '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=TOKENIZERS_PARALLELISM=false', '--setenv=NXD_CPU_MODE=1',
        '--setenv=PATH=/home/ubuntu/venvs/vllm_neuron/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin',
        python, '-u', str(SCRIPT), phase]


def header():
    return {name: value for name, value in read(INPUT / 'safetensors-header.json').items() if name != '__metadata__'}


def expected_neuron_shapes(tp):
    assert tp in (1, 2)
    rows = {}
    for original, item in header().items():
        name = original.removeprefix('model.')
        axis = None
        if name == 'embed_tokens.weight':
            axis = 1
        elif name == 'lm_head.weight' or any(x in name for x in ('.q_proj.', '.k_proj.', '.v_proj.', '.gate_proj.', '.up_proj.')):
            axis = 0
        elif any(x in name for x in ('.o_proj.', '.down_proj.')):
            axis = 1
        shape = list(item['shape'])
        if axis is not None:
            assert shape[axis] % tp == 0
            shape[axis] //= tp
        for projection in ('q_proj', 'k_proj', 'v_proj'):
            name = name.replace('.self_attn.' + projection + '.', '.self_attn.qkv_proj.' + projection + '.')
        name = name.replace('.self_attn.o_proj.', '.self_attn.o_proj.o_proj.')
        rows[name] = shape
    assert len(rows) == 381
    return rows


def check_result(phase, result):
    assert phase in PHASES[1:]
    assert result['phase'] == phase and result['passed'] and result['checkpoint_revision'] == REVISION
    assert result['plan_sha256'] == sha(DEST / 'plan.json')
    for key in ('numerical_execution', 'device_execution', 'collectives_executed', 'full_model_qualified'):
        assert result[key] is False
    assert result['model_weight_payload_bytes_read'] == 0
    if phase == 'native':
        assert result['state_tensors'] == 381 and result['parameter_elements'] == 2516756480
        assert result['all_parameters_meta'] and result['independent_head'] and result['exact_header_shapes']
        assert result['state_shapes'] == {name: item['shape'] for name, item in header().items()}
        assert result['normalized_rope_parameters'] == {'rope_type': 'default', 'rope_theta': 5000000}
        assert result['raw_rope_theta'] == 5000000 and result['raw_rope_scaling'] is None
        assert result['runtime_weight_values_verified'] is False and result['native_config_fields_preserved'] is True
    elif phase in JOBS:
        assert [result['tp'], result['rank']] == list(JOBS[phase])
        assert result['mock_rank_topology'] is True
        padding.validate_controls(result['padding_controls'])
        assert len(result['stages']) == 2
        for stage, item in zip(('prefill', 'decode'), result['stages']):
            assert item['stage'] == stage and item['learned_tensors'] == 381
            assert item['all_parameters_meta'] and item['independent_head'] and item['actual_sdk_pipeline']
            assert item['file_payload_loader_substituted_with_meta'] is True
            assert item['cache_tensors'] == 84 and item['missing_keys_exact_cache_and_shared_rank_alias'] is True
            alias_contract.validate(item['load_keys'], expected_neuron_shapes(result['tp']), item['rank_buffer_keys'])
            padding.validate_stage(item['padding_context'], result['tp'], result['rank'], item['rank_buffer_keys'])
            assert item['logical_cache_length'] == LOGICAL_CACHE_LENGTH and item['physical_cache_length'] == PHYSICAL_CACHE_LENGTH
            assert item['cache_shapes'] == [[4, 2 // result['tp'], PHYSICAL_CACHE_LENGTH, 128]] * 84
            assert item['buckets'] == BUCKETS and item['alias_bindings'] == 84 * len(BUCKETS)
            assert item['all_shards_match_expected_shapes'] and item['all_shards_match_model_shapes']
            assert item['learned_shapes'] == expected_neuron_shapes(result['tp'])
            assert item['learned_storage_bytes'] == 2 * sum(math.prod(shape) for shape in expected_neuron_shapes(result['tp']).values())
            assert item['single_cache_pool_bytes'] == 2 * 84 * math.prod([4, 2 // result['tp'], PHYSICAL_CACHE_LENGTH, 128])
            assert item['rank_buffer_keys'] == sorted({'rank_util.rank'} | {f'layers.{i}.self_attn.rank_util.rank' for i in range(42)})


def final():
    verify()
    rows = []
    for phase in PHASES[1:]:
        path = DEST / (phase + '-result.json')
        check_result(phase, read(path))
        rows.append(dict(phase=phase, path=str(path), sha256=sha(path)))
    result = dict(passed=True, model=MODEL, checkpoint_revision=REVISION, phases=rows,
        sdk_rank_configurations=3, sdk_module_constructions=6, actual_sdk_sharder_invocations=6,
        learned_shape_checks=6 * 381, cache_alias_bindings=6 * 84 * len(BUCKETS),
        model_weight_payload_bytes_read=0, numerical_execution=False, device_execution=False,
        collectives_executed=False, full_model_qualified=False,
        scope='Complete HF/SDK meta shapes and aliases; original GQA helper under a harness-only meta allocation context. No model values or forward calls.')
    once(DEST / 'result.json', result)
    return result

