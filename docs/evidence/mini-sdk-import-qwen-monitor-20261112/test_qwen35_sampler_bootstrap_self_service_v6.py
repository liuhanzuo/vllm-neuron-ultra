"""Actual bootstrap/Service/LeaseObserver code, mocked process and metadata IO.

No native imports, real services, locks, or runtime admission. The same missing
supervisor receipt reproduces v1's failure and exercises the v2 successor.
"""
import copy
import importlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]


def bootstrap(version, *, unknown=False, mutate=None):
    a = importlib.import_module('qwen35_sampler_runtime_adapter_' + version)
    s = importlib.import_module('qwen35_sampler_runtime_service_' + version)
    c = importlib.import_module('qwen35_sampler_runtime_contract_' + version)
    case = {'artifact': '/unchanged-artifact', 'split_correction_sha256': 'a' * 64}
    spec = {'run_name': 'bootstrap-regression', 'output_root': '/mnt/nvme/neuron-work/qwen35-sampler-pair-bootstrap-regression',
            'source_sha256': {}}
    admission_ref = {'path': '/admission.json', 'sha256': 'a' * 64}
    co_ref = {'path': str(c.control_root(spec) / 'coordinator-owner.json'), 'sha256': 'b' * 64}
    co_unit = c.units(spec['run_name'])['coordinator']
    group_prefix = '/fake/'
    co_proc = {'pid': 12, 'ppid': 1, 'start_ticks': '100', 'argv': s.argv('coordinator', admission_ref),
               'cgroup': '0::' + group_prefix + co_unit + '.service'}
    co_owner = {'process': co_proc, 'service_owner': {'unit': co_unit, 'pid': 12, 'start_ticks': 100, 'invocation_id': 'c' * 32}}
    directory = c.control_root(spec) / 'shared-native_preflight'
    phase = s.expected_phase(spec, admission_ref, co_owner, 'attempt', 'shared', 'native_preflight', directory)
    phase_ref = {'path': str(Path(phase['control']) / 'phase-spec.json'), 'sha256': 'd' * 64}
    phase_proc = {'pid': 13, 'ppid': 1, 'start_ticks': '200', 'argv': s.argv('supervisor', phase_ref),
                  'cgroup': '0::' + group_prefix + phase['unit'] + '.service'}
    processes = {12: copy.deepcopy(co_proc), 13: copy.deepcopy(phase_proc)}
    admission = {'outer_owner': {'pid': 11}, 'retained_fd': 8,
                 'deadline_monotonic': time.monotonic() + 60, 'host_source_sha256': {}}
    observer = a.LeaseObserver(admission, spec)
    bundle = SimpleNamespace(spec=spec)
    properties = {}
    checks = []

    for unit, process, props, log, parent, reference, invocation in (
        (co_unit, co_proc, c.COORDINATOR_PROPERTIES, c.control_root(spec) / 'coordinator.log', None, admission_ref, 'c' * 32),
        (phase['unit'], phase_proc, phase['properties'], Path(phase['log']), co_unit, phase_ref, 'e' * 32),
    ):
        cmd = s.command(unit, process['argv'], c.native_environment(case), props, log, parent)
        value = {k: v for k, v in props.items() if k not in ('RuntimeMaxSec', 'TimeoutStopSec')}
        value.update(Id=unit + '.service', LoadState='loaded', ActiveState='active', SubState='running',
                     MainPID=str(process['pid']), ExecMainPID=str(process['pid']), ControlPID='0', Job='',
                     InvocationID=invocation, ControlGroup=group_prefix + unit + '.service',
                     ExecStart='{ argv[]=' + ' '.join(cmd[cmd.index('/usr/bin/env'):]) + ' ; }',
                     ExecStopPost='', Environment='', StandardOutput='append', StandardError='inherit',
                     RuntimeMaxUSec=props['RuntimeMaxSec'] + 's', TimeoutStopUSec=props['TimeoutStopSec'] + 's',
                     BindsTo='' if parent is None else parent + '.service', After='' if parent is None else parent + '.service')
        properties[unit] = value

    def active_heavy(allowed):
        checks.append(list(allowed))
        running = {co_unit, phase['unit']}
        if unknown:
            running.add('trn1-unknown-heavy-service')
        if not running <= set(allowed):
            raise RuntimeError('Competing active heavy service')

    fake = SimpleNamespace(GROUP_PREFIX=group_prefix, proc=lambda pid: copy.deepcopy(processes.get(pid)),
                           invocation=lambda value: value in ('c' * 32, 'e' * 32), active_heavy=active_heavy,
                           terminal=lambda value: value['MainPID'] == '0')

    def running_reader(*args):
        # Real missing-receipt bootstrap has already called Service.live().
        assert observer.current.owner is not None and observer.current.started is False
        if mutate in ('argv', 'start_ticks', 'cgroup', 'pid'):
            processes[13][mutate] = {'argv': ['/unexpected'], 'start_ticks': '201',
                                    'cgroup': '0::/different', 'pid': 14}[mutate]
        elif mutate == 'invocation':
            properties[phase['unit']]['InvocationID'] = 'f' * 32
        # Metadata.read calls the exact LeaseObserver guard before and after IO.
        observer.check()
        return {'observation': {'path': '/unused'}}

    def read(reference, *args):
        if reference == phase_ref:
            return copy.deepcopy(phase)
        if reference == co_ref:
            return copy.deepcopy(co_owner)
        raise AssertionError('unexpected metadata read')

    with ExitStack() as stack:
        for name, value in (
            ('load_context', lambda _: (admission, bundle, observer)), ('read', read), ('ref', lambda _: co_ref),
            ('read_if_present', lambda _: None), ('expected_case', lambda: case),
            ('descriptor', lambda *args: None), ('resources', lambda *args: None),
            ('check_running', running_reader), ('loaded_local_sources', lambda *args: None),
            ('process_api', lambda: fake),
        ):
            stack.enter_context(patch.object(a, name, value))
        stack.enter_context(patch.object(s, 'process_api', lambda: fake))
        stack.enter_context(patch.object(s, 'properties', lambda unit: copy.deepcopy(properties[unit])))
        stack.enter_context(patch.object(a.os, 'getpid', return_value=13))
        value, context = a.worker_bootstrap(phase_ref, role='supervisor')
    return value, context, checks


class BootstrapRegression(unittest.TestCase):
    def test_original_missing_receipt_reproduces_self_competition(self):
        with self.assertRaisesRegex(RuntimeError, 'Competing active heavy service'):
            bootstrap('v1')

    def test_successor_accepts_exact_authenticated_self_before_receipt(self):
        phase, context, checks = bootstrap('v6')
        self.assertFalse(context['service'].started)
        self.assertEqual(context['service'].owner['process']['pid'], 13)
        self.assertEqual(checks, [[phase['coordinator']['service_owner']['unit'], phase['unit']]])

    def test_unknown_competing_unit_still_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Competing active heavy service'):
            bootstrap('v6', unknown=True)

    def test_changed_identity_or_invocation_cannot_bootstrap(self):
        for field in ('argv', 'start_ticks', 'pid', 'cgroup', 'invocation'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                bootstrap('v6', mutate=field)

    def test_no_native_packages_loaded(self):
        self.assertFalse({'torch', 'numpy', 'nki', 'transformers', 'vllm'} & set(sys.modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)
