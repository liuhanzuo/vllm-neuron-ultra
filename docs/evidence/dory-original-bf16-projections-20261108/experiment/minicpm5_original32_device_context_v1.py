"""Finite device admission; reuse only the frozen authenticated input readers."""
import copy
import os
from pathlib import Path
import re
import sys

from minicpm5_layer0_runtime_auth_v1 import RuntimeContext, clear_window, digest, require

PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
SDK = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
FORBIDDEN = ('NXD_CPU_MODE', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
             'TORCHELASTIC_RUN_ID', 'NEURON_PJRT_PROCESS_INDEX', 'PJRT_LOCAL_PROCESS_RANK',
             'NEURON_PJRT_WORLD_SIZE', 'PJRT_LOCAL_PROCESS_COUNT', 'NEURON_PT_MPMD_MODE',
             'NEURON_PJRT_PROCESSES_NUM_DEVICES', 'NEURONCORE_NUM_DEVICES',
             'XLA_USE_SPMD', 'XLA_AUTO_SPMD', 'PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD',
             'LD_LIBRARY_PATH', 'TRN1_CPU_SAMPLER_FAST', 'TRN1_SMOLLM3_MIN_DECODE_BUCKET',
             'TRN1_MINICPM5_NKI_CACHE', 'NXD_INFERENCE_CAPTURE_SNAPSHOT')
SDK_MODULES = ('torch_neuronx', 'torch_neuronx.xla', 'torch_neuronx.initialization',
               'torch_neuronx.initialization.initializer')


def environment_check(environment):
    require(environment.get('NEURON_RT_VISIBLE_CORES') == '0-1', 'Original device cores0-1')
    require(all(key not in environment for key in FORBIDDEN), 'Single device process, no CPU/rank/override environment')
    require(environment.get('PJRT_DEVICE') in (None, 'NEURON'), 'Only Neuron PJRT backend')
    return {key: environment.get(key) for key in (*FORBIDDEN, 'NEURON_RT_VISIBLE_CORES',
                                                 'PJRT_DEVICE', 'NEURON_LIBRARY_PATH')}


class DeviceContext(RuntimeContext):
    """Separate role/schema. No old permit or CPU context activation is reused."""
    def __init__(self, plan, plan_sha256, permit, external_guard):
        require(digest(plan) == plan_sha256, 'Exact device plan digest')
        require(plan['schema'] == 'minicpm5_original32_device_runtime_inputs_v1', 'Device input schema')
        require(not {'torch', 'torch_neuronx'} & set(sys.modules), 'Admit device before native imports')
        self.plan = copy.deepcopy(plan)
        self.plan_sha256 = plan_sha256
        self.permit = copy.deepcopy(permit)
        self.external_guard = external_guard
        self._verified = {}
        self._sources_checked = False
        self._device_after = None
        self._device_before = environment_check(os.environ)
        self.guard('device')

    def guard(self, role=None):
        clear_window()
        require(self.external_guard() == self.permit, 'External device admission changed')
        p = self.permit
        require(p['schema'] == 'minicpm5_original32_device_runtime_permit_v1' and
                p['plan_sha256'] == self.plan_sha256, 'Exact device permit')
        require(p['role'] == 'device' and role in (None, 'device') and
                p['rank'] == -1 and p['arm'] == 'device_trajectory', 'Only device trajectory role')
        require(re.fullmatch('[0-9a-f]{32}', p['attempt_id']) is not None, 'Device attempt')
        require(p['serial_exclusive'] is True and p['source_review_accepted'] is True and
                p['all_predecessors_drained'] is True, 'Device serial/source prerequisites')
        require(p['pid'] == os.getpid() and p['uid'] == os.getuid() == 1000, 'Device process identity')
        require(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19] == p['start_ticks'], 'Device process start')
        require(Path('/proc/self/cgroup').read_text().strip() == p['cgroup'], 'Device process group')
        require([x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x] == p['argv'], 'Device argv')
        require(sys.executable == p['python'] == self.plan['interpreters']['device'] == PYTHON and
                sys.dont_write_bytecode and __debug__, 'Fixed device interpreter')
        require(p['output_root'] == self.plan['output_root'], 'Fresh device output binding')
        require(all(os.environ.get(k) == v for k, v in p['environment'].items()), 'Stable device environment')
        environment_check(os.environ)
        if self._device_after is not None:
            require(sys.modules['torch_neuronx.initialization']._initializer.configured_pjrt_env is False,
                    'Device remains outside elastic PJRT rank mapping')
        return p

    def record_device_import(self):
        require(self._sources_checked and self._device_after is None, 'Single source-authenticated device import')
        self.guard('device')
        paths = {}
        for name in SDK_MODULES:
            module = sys.modules[name]
            relative = name.replace('.', '/') + ('/__init__.py' if hasattr(module, '__path__') else '.py')
            path = (SDK / relative).resolve()
            require(Path(module.__file__).resolve() == path and str(path) in self.plan['source_sha256'],
                    'Device SDK source path authenticated')
            paths[name] = str(path)
        initializer = sys.modules['torch_neuronx.initialization']._initializer
        require(initializer is not None and initializer.configured_pjrt_env is False and
                initializer.previous_pjrt_env_vars == {}, 'Non-elastic original SDK initialization')
        self._device_after = environment_check(os.environ)
        return dict(before=self._device_before, after=self._device_after, sdk_modules=paths,
                    configured_pjrt_env=False, environment_modified_by_diagnostic=False)

    def activate(self):
        raise RuntimeError('Device role cannot activate the frozen CPU/native context')

    def record_cpu_import(self):
        raise RuntimeError('Device role cannot use CPU rank initialization')
