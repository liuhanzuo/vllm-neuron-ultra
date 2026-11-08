"""Exact torchrun/SDK environment transition, without native imports or mutation."""
from pathlib import Path
import os
import sys

SDK = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
SDK_SOURCES = ('torch/__init__.py', 'torch/version.py',
    'torch_neuronx/__init__.py', 'torch_neuronx/xla.py',
    'torch_neuronx/initialization/__init__.py', 'torch_neuronx/initialization/initializer.py',
    'torch_neuronx/utils/__init__.py', 'torch_neuronx/utils/get_visible_cores.py',
    'torch_xla/__init__.py', 'torch_xla/_internal/neuron.py', 'torch_xla/runtime.py')
RANK_KEYS = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE')
PJRT_KEYS = ('NEURON_PJRT_PROCESS_INDEX', 'PJRT_LOCAL_PROCESS_RANK',
    'NEURON_PJRT_WORLD_SIZE', 'PJRT_LOCAL_PROCESS_COUNT', 'NEURON_PT_MPMD_MODE')
FORBIDDEN = ('NEURON_PJRT_PROCESSES_NUM_DEVICES', 'NEURONCORE_NUM_DEVICES',
    'XLA_USE_SPMD', 'XLA_AUTO_SPMD')
_before = None
_after = None


def snapshot(environment):
    return {key: environment.get(key) for key in (*RANK_KEYS, *PJRT_KEYS,
        *FORBIDDEN, 'NEURON_RT_VISIBLE_CORES', 'NXD_CPU_MODE')}


def check_before(environment):
    """No SDK initialization has occurred; the admitted parent core set is intact."""
    row = snapshot(environment)
    assert row['RANK'] == row['LOCAL_RANK'] and row['RANK'] in ('0', '1')
    assert row['WORLD_SIZE'] == row['LOCAL_WORLD_SIZE'] == '2'
    assert row['NEURON_RT_VISIBLE_CORES'] == '0-1' and row['NXD_CPU_MODE'] == '1'
    assert all(row[key] is None for key in (*PJRT_KEYS, *FORBIDDEN))
    return row


def check_after(before, environment):
    """Validate the pinned SDK's non-SPMD LOCAL_RANK mapping, without changing it."""
    check_before(before)
    row = snapshot(environment)
    assert all(row[key] == before[key] for key in RANK_KEYS)
    rank = before['LOCAL_RANK']
    # The pinned Initializer selects LOCAL_RANK. The admitted physical map is
    # exactly [0, 1], so this also proves the selected core belongs to that map.
    assert row['NEURON_RT_VISIBLE_CORES'] == str((0, 1)[int(rank)])
    assert row['NEURON_PJRT_PROCESS_INDEX'] == before['RANK']
    assert row['PJRT_LOCAL_PROCESS_RANK'] == rank
    assert row['NEURON_PJRT_WORLD_SIZE'] == row['PJRT_LOCAL_PROCESS_COUNT'] == '2'
    assert row['NEURON_PT_MPMD_MODE'] == row['NXD_CPU_MODE'] == '1'
    assert all(row[key] is None for key in FORBIDDEN)
    return row


def guard(phase, child):
    global _before
    if not child:
        assert os.environ.get('NEURON_RT_VISIBLE_CORES') == '0-1'
        assert all(os.environ.get(key) is None for key in (*RANK_KEYS, *PJRT_KEYS, *FORBIDDEN))
        return
    assert phase == 'matched_cpu'
    if _before is None:
        assert 'torch_neuronx' not in sys.modules and 'torch' not in sys.modules
        _before = check_before(os.environ)
    elif _after is None:
        assert 'torch_neuronx' not in sys.modules
        assert check_before(os.environ) == _before
    else:
        assert check_after(_before, os.environ) == _after


def record_after_import():
    """Called exactly once after the real SDK import and before model construction."""
    global _after
    assert _before is not None and _after is None
    for name in ('torch_neuronx', 'torch_neuronx.xla',
                 'torch_neuronx.initialization', 'torch_neuronx.initialization.initializer'):
        module = sys.modules[name]
        relative = name.replace('.', '/') + ('/__init__.py' if hasattr(module, '__path__') else '.py')
        assert Path(module.__file__).resolve() == SDK / relative
    initializer = sys.modules['torch_neuronx.initialization']._initializer
    assert initializer is not None and initializer.configured_pjrt_env is True
    assert initializer.previous_pjrt_env_vars['NEURON_RT_VISIBLE_CORES'] == '0-1'
    _after = check_after(_before, os.environ)
    return dict(passed=True, before=dict(_before), after=dict(_after),
        physical_core_map=[0, 1], sdk_initializer_original_visible_cores='0-1',
        environment_modified_by_diagnostic=False)


def check_receipt(receipt, rank):
    assert receipt['passed'] and receipt['physical_core_map'] == [0, 1]
    assert receipt['sdk_initializer_original_visible_cores'] == '0-1'
    assert receipt['environment_modified_by_diagnostic'] is False
    assert check_before(receipt['before'])['RANK'] == str(rank)
    assert check_after(receipt['before'], receipt['after']) == receipt['after']
