"""Stdlib/source mocks only; no runtime import, History, locks or model IO."""
import copy
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


class NativeBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'transformers', 'vllm', 'vllm_neuron',
                'torch_neuronx', 'neuronx_distributed', 'neuronx_distributed_inference'}:
            raise AssertionError('native import forbidden: ' + fullname)


sys.meta_path.insert(0, NativeBlocker())
SOURCE = Path('/home/ubuntu/qwen35_package_full_r4_post_exit_20261112_v2.py')
module_spec = importlib.util.spec_from_file_location('post_exit_subject', SOURCE)
p = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(p)


def ref(path='/mock/input.json', sha='a'):
    return dict(path=path, sha256=sha * 64)


def observed(exit_code=0):
    return dict(schema='qwen35_package_external_tool_result_v1', reviewer='/root',
        actual_tool=dict(kind='tool_session', session_id=77, terminal_chunk='actual_mock_terminal', exit_code=exit_code),
        completed=True, observation_source='actual_external_tool_result', observed_unix_ns=123)


class Scope:
    def __init__(self):
        self.expected = {'coordinator', 'started', 'unstarted'}
        self.props = {name: dict(LoadState='loaded' if name != 'unstarted' else 'not-found',
                                InvocationID='known' if name != 'unstarted' else '') for name in self.expected}
        self.services = []
        for name in ('coordinator', 'started'):
            self.services.append(types.SimpleNamespace(unit=name, started=True,
                drained=Mock(return_value=dict(cleanup_complete=True, remaining_processes=[],
                    systemd_exit_status={'ExecMainStatus': 1})), historical_terminal_reuse=None))
        self.process = types.SimpleNamespace(GROUP_PREFIX='/mock/', active_heavy=Mock(), gone=Mock(),
                                            empty_group=Mock(return_value=[]))
        self.guard = Mock(); self.owners = [{'pid': 1}, {'pid': 2}, {'pid': 3}]
        self.declare = Mock(side_effect=self.declared)

    def declared(self, spec, admission, coordinator, *, retained, observations):
        assert spec == {'mock': 'runtime'} and admission == ref() and coordinator == {'owner': 'coordinator'}
        observations.extend(dict(unit=name, raw_properties=self.props[name]) for name in self.expected)
        return self.services

    def run(self):
        with patch('builtins.open', side_effect=AssertionError('no file IO')), \
             patch('os.open', side_effect=AssertionError('no descriptor IO')):
            return p.collect_scope(dict(units=self.expected, runtime_spec={'mock': 'runtime'}), ref(),
                {'owner': 'coordinator'}, self.owners, self.declare, lambda unit: copy.deepcopy(self.props[unit]),
                self.process, self.guard)


class ScopeChecks(unittest.TestCase):
    def test_success_and_failed_service_status_both_require_complete_two_pass_scope(self):
        f = Scope(); observations = f.run()
        self.assertEqual(f.declare.call_count, 2)
        self.assertEqual(set(observations[0]['current']), f.expected)
        self.assertEqual(observations[0]['current'], observations[1]['current'])
        self.assertEqual(f.process.gone.call_count, 6)
        self.assertEqual(f.process.empty_group.call_count, 6)
        for service in f.services: self.assertEqual(service.drained.call_count, 2)
        self.assertEqual(observations[0]['actual_terminals']['started']['systemd_exit_status']['ExecMainStatus'], 1)

    def test_missing_unstarted_unit_coverage_rejected(self):
        f = Scope()
        def missing(*args, **kwargs):
            kwargs['observations'].append(dict(unit='started'))
            return f.services
        f.declare.side_effect = missing
        with self.assertRaisesRegex(ValueError, 'complete declared'): f.run()

    def test_absent_scope_with_live_group_rejected(self):
        f = Scope(); f.process.empty_group.side_effect = RuntimeError('Live group descendants')
        with self.assertRaisesRegex(RuntimeError, 'Live group'): f.run()

    def test_missing_owner_gone_or_service_drain_fails(self):
        for mode in ('owner', 'drain', 'remaining'):
            f = Scope()
            if mode == 'owner': f.process.gone.side_effect = ValueError('owner still live')
            elif mode == 'drain': f.services[1].drained.side_effect = ValueError('service still live')
            else: f.services[1].drained.return_value['remaining_processes'] = [12]
            with self.subTest(mode=mode), self.assertRaises(ValueError): f.run()

    def test_omitted_or_foreign_declared_service_rejected(self):
        f = Scope(); f.services[0].unit = 'foreign'
        with self.assertRaisesRegex(ValueError, 'foreign'): f.run()

    def test_unstarted_unit_must_be_not_found(self):
        f = Scope(); f.props['unstarted']['LoadState'] = 'loaded'
        with self.assertRaisesRegex(ValueError, 'must remain absent'): f.run()

    def test_two_current_observations_must_match(self):
        f = Scope(); original = f.declared
        def changed(*args, **kwargs):
            if f.declare.call_count == 2: f.props['started']['InvocationID'] = 'changed'
            return original(*args, **kwargs)
        f.declare.side_effect = changed
        with self.assertRaisesRegex(ValueError, 'two stable'): f.run()

    def test_saved_historical_reuse_retained_as_historical(self):
        f = Scope(); f.services[1].historical_terminal_reuse = {'saved_exit_status_is_historical': True}
        out = f.run()
        self.assertEqual(out[0]['historical_terminal_reuse']['started'], {'saved_exit_status_is_historical': True})


class Outcome(unittest.TestCase):
    def test_success_calls_original_audit_with_exact_refs_and_guard(self):
        audit = Mock(return_value=ref('/mock/stage25.json')); guard = Mock()
        out = p.audit_outcome(observed()['actual_tool'], {'failed': False, 'result_present': True},
            ref('/mock/spec'), ref('/mock/review'), ref('/mock/external'), guard, audit)
        audit.assert_called_once_with(ref('/mock/spec'), ref('/mock/review'), ref('/mock/external'), guard)
        self.assertTrue(out['all25_stages_audited']); self.assertFalse(out['full_model_qualified'])
        self.assertFalse(out['new_speedup_claim'])

    def test_real_nonzero_and_terminal_inconsistency_never_call_success_audit(self):
        for code, failed, present in ((1, True, False), (1, False, True), (0, True, True), (0, False, False)):
            audit = Mock()
            out = p.audit_outcome(observed(code)['actual_tool'], {'failed': failed, 'result_present': present},
                ref(), ref(), ref(), Mock(), audit)
            audit.assert_not_called(); self.assertFalse(out['all25_stages_audited'])
            self.assertEqual(out['actual_outer_exit_code'], code); self.assertEqual(out['outer_reported_failed'], failed)

    def test_audit_exception_keeps_real_outer_success_and_records_separate_failure(self):
        audit = Mock(side_effect=RuntimeError('actual stage25 gate failed'))
        out = p.audit_outcome(observed()['actual_tool'], {'failed': False, 'result_present': True},
            ref(), ref(), ref(), Mock(), audit)
        self.assertEqual(out['actual_outer_exit_code'], 0); self.assertEqual(out['status'], 'independent_stage25_failed')
        self.assertEqual(out['stage25_error']['message'], 'actual stage25 gate failed')
        self.assertFalse(out['all25_stages_audited']); self.assertIsNone(out['stage25'])

    def test_external_receipt_requires_actual_terminal_and_exact_session(self):
        wanted = {'kind': 'tool_session', 'session_id': 77}
        self.assertEqual(p.actual_tool(observed(1), wanted)['exit_code'], 1)
        for mode in ('chunk', 'session', 'completed', 'source', 'code'):
            value = observed()
            if mode == 'chunk': del value['actual_tool']['terminal_chunk']
            elif mode == 'session': value['actual_tool']['session_id'] = 88
            elif mode == 'completed': value['completed'] = False
            elif mode == 'source': value['observation_source'] = 'inferred_child_exit'
            else: value['actual_tool']['exit_code'] = True
            with self.subTest(mode=mode), self.assertRaises(ValueError): p.actual_tool(value, wanted)


class Composition(unittest.TestCase):
    def fixture(self):
        rows = [dict(terminal=ref('/mock/terminal-' + str(i)), tool={}, owners=[], units=[],
                     source_sha256={'/mock/old.py': 'b' * 64} if i == 0 else {}) for i in range(161)]
        return {'runs': rows}, {'/mock/base.py': 'a' * 64, '/mock/old.py': 'b' * 64, '/mock/new.py': 'c' * 64}, {
            '/mock/old.py': 'b' * 64, '/mock/new.py': 'c' * 64}

    def test_only_true_new_sources_include_external_observer_and_ack(self):
        old, sources, extension = self.fixture()
        external = {'/mock/observer.py': 'd' * 64, '/mock/ack.py': 'e' * 64}
        delta, summary = p.source_delta(sources, extension, old, external)
        self.assertEqual(delta, {'/mock/new.py': 'c' * 64, **external})
        self.assertEqual(summary['post_audit_sources'], 5)
        self.assertNotIn('/mock/old.py', delta); self.assertNotIn('/mock/base.py', delta)

    def test_source_conflicts_and_missing_inherited_sources_rejected(self):
        for mode in ('prior', 'extension', 'external'):
            old, sources, extension = self.fixture(); external = {}
            if mode == 'prior': del sources['/mock/old.py']; del extension['/mock/old.py']
            elif mode == 'extension': extension['/mock/old.py'] = 'f' * 64
            else: external['/mock/base.py'] = 'f' * 64
            with self.subTest(mode=mode), self.assertRaises(ValueError): p.source_delta(sources, extension, old, external)

    def test_catalog_keeps_every_one_of_161_rows_and_appends_exact_five_field_row(self):
        old, _, _ = self.fixture(); before = copy.deepcopy(old); shape = Mock()
        new = p.append_catalog(old, ref('/mock/idle'), ref('/mock/terminal'), observed()['actual_tool'],
            [{'pid': 12}], ['b', 'a'], {'/mock/new.py': 'c' * 64}, shape)
        self.assertEqual(old, before); self.assertEqual(new['runs'][:-1], old['runs'])
        self.assertEqual(len(new['runs']), 162); self.assertEqual(shape.call_count, 2)
        self.assertEqual(set(new['runs'][-1]), {'terminal', 'tool', 'owners', 'units', 'source_sha256'})

    def test_missing_old_row_and_large_catalog_rejected(self):
        old, _, _ = self.fixture(); old['runs'].pop()
        with self.assertRaisesRegex(ValueError, '161 rows'):
            p.append_catalog(old, ref(), ref(), {}, [], [], {}, Mock())
        with self.assertRaisesRegex(ValueError, '8 MiB'):
            p.encode({'large': 'x' * p.CAP})


class StableSource(unittest.TestCase):
    def test_full_digest_same_size_change_and_identity_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'mock.py'; path.write_bytes(b'original')
            r = dict(path=str(path), sha256=hashlib.sha256(b'original').hexdigest())
            raw, identity = p.stable_source(r); self.assertEqual(raw, b'original')
            self.assertEqual(p.stable_source(r)[1], identity)
            path.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'digest'): p.stable_source(r)

    def test_symlink_hardlink_and_oversize_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'mock.py'; path.write_bytes(b'x')
            r = dict(path=str(path), sha256=hashlib.sha256(b'x').hexdigest())
            with self.assertRaisesRegex(ValueError, 'bounded'): p.stable_source(r, cap=0)
            other = path.with_name('other'); other.hardlink_to(path)
            with self.assertRaisesRegex(ValueError, 'singleton'): p.stable_source(r)
            other.unlink(); other.symlink_to(path)
            with self.assertRaisesRegex(ValueError, 'canonical'): p.stable_source(dict(r, path=str(other)))

    def test_import_does_not_load_runtime_or_native(self):
        self.assertFalse(any(name.startswith('qwen35_package_serial_runtime') for name in sys.modules))
        self.assertFalse(any(name in sys.modules for name in ('torch', 'numpy', 'vllm')))



class MissingTerminal(unittest.TestCase):
    def test_nonzero_exit_with_absent_terminal_never_calls_success_audit(self):
        for code in (1, 152):
            audit = Mock()
            out = p.audit_outcome(observed(code)['actual_tool'], None, ref(), ref(), ref(), Mock(), audit)
            self.assertEqual(out['actual_outer_exit_code'], code)
            self.assertIsNone(out['outer_reported_failed'])
            self.assertTrue(out['outer_terminal_missing'])
            self.assertEqual(out['status'], 'outer_failed_without_terminal')
            self.assertFalse(out['all25_stages_audited'])
            self.assertFalse(out['full_model_qualified'])
            audit.assert_not_called()

    def test_zero_exit_with_absent_terminal_cannot_be_recovered_as_failure_or_success(self):
        audit = Mock()
        with self.assertRaisesRegex(ValueError, 'actual failed outer exit'):
            p.audit_outcome(observed()['actual_tool'], None, ref(), ref(), ref(), Mock(), audit)
        audit.assert_not_called()

if __name__ == '__main__':
    unittest.main(verbosity=2)
