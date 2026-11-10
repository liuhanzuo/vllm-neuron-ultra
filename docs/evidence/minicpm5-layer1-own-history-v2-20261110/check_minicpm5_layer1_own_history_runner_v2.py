"""Synthetic own-history service lifecycle and source admission checks."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
import minicpm5_layer1_own_history_cpu_runner_v2 as runner
import minicpm5_layer1_own_history_cpu_worker_v2 as worker
from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained

def reference(name):return dict(path='/synthetic/'+name+'.json',sha256='a'*64)

class DrainHold(unittest.TestCase):

    def test_cleanup_failure_keeps_ownership_until_independent_drain(self):
        events = []
        attempt = [0]

        def cleanup():
            attempt[0] += 1
            events.append('cleanup')
            if attempt[0] == 1:
                raise RuntimeError('still running')

        def verify():
            events.append('verify')
            if attempt[0] < 2:
                raise RuntimeError('group populated')
            return dict(physically_drained=True, group='synthetic-empty')
        proof = retain_until_drained(cleanup, verify, lambda v: events.append('held'), lambda : events.append('retry'))
        self.assertTrue(proof['physically_drained'])
        self.assertEqual(events, ['cleanup', 'verify', 'held', 'retry', 'cleanup', 'verify'])

    def test_receipt_failure_and_interrupted_wait_cannot_unwind_lock(self):
        calls = [0]
        errors = []

        def cleanup():
            calls[0] += 1

        def verify():
            if calls[0] < 3:
                raise RuntimeError('pending')
            return dict(physically_drained=True)

        def receipt(value):
            errors.append('receipt')
            raise OSError('full')

        def wait():
            errors.append('wait')
            raise KeyboardInterrupt()
        self.assertTrue(retain_until_drained(cleanup, verify, receipt, wait)['physically_drained'])
        self.assertEqual(calls[0], 3)
        self.assertEqual(errors, ['receipt', 'wait', 'receipt', 'wait'])

    def test_successful_cleanup_without_actual_drain_is_insufficient(self):
        calls = [0]

        def verify():
            calls[0] += 1
            return dict(physically_drained=calls[0] > 1)
        proof = retain_until_drained(lambda : None, verify, lambda v: None, lambda : None)
        self.assertTrue(proof['physically_drained'])
        self.assertEqual(calls[0], 2)

    def test_independently_drained_service_can_release_despite_cleanup_receipt_error(self):

        def cleanup():
            raise OSError('receipt unavailable after stop')
        proof = retain_until_drained(cleanup, lambda : dict(physically_drained=True), lambda v: self.fail('already drained'), lambda : self.fail('already drained'))
        self.assertTrue(proof['physically_drained'])

class RunnerLifecycle(unittest.TestCase):

    def test_outer_early_failure_releases_only_after_actual_drain_and_preserves_primary_error(self):
        events = []
        checks = [0]
        with tempfile.TemporaryDirectory(prefix='mini-layer1-lifecycle-') as d:
            spec = dict(output_root=str(Path(d) / 'uncreated'), unit='synthetic')

            def acquire():
                events.append('acquire')
                return (999, dict(pid=123, start_ticks='456'), dict(inode=1))

            def prerequisites(*args):
                events.append('prerequisites')
                raise ValueError('original admission failure')

            def verify(*args):
                checks[0] += 1
                events.append('verify' + str(checks[0]))
                if checks[0] == 1:
                    raise RuntimeError('group not drained')
                return dict(physically_drained=True)
            with patch.object(runner, 'read_bundle', return_value=(spec, {})), patch.object(runner.p, 'user_bus'), patch.object(runner.post.external, 'acquire_lock', side_effect=acquire), patch.object(runner.post.external, 'release_lock', side_effect=lambda fd: events.append('unlock')), patch.object(runner.p, 'lock_proof'), patch.object(runner.p, 'clear_window'), patch.object(runner.p, 'active_heavy'), patch.object(runner, 'check_prerequisites', side_effect=prerequisites), patch.object(runner, 'cleanup_owned_service'), patch.object(runner, 'verify_owned_drain', side_effect=verify):
                with self.assertRaisesRegex(ValueError, 'original admission failure'):
                    runner.outer(reference('spec'), reference('review'))
            self.assertEqual(events, ['acquire', 'prerequisites', 'verify1', 'verify2', 'unlock'])

    def test_unowned_loaded_service_cannot_be_stopped_or_reported_drained(self):
        spec = dict(output_root='/synthetic/uncreated', unit='synthetic')
        with patch.object(runner.p, 'properties', return_value=dict(LoadState='loaded')), patch.object(runner.p, 'command') as command:
            with self.assertRaises(RuntimeError):
                runner.verify_owned_drain(spec, None, None, None, False)
            command.assert_not_called()

    def test_live_recorded_owner_or_untracked_group_member_prevents_drain(self):
        props = dict(LoadState='not-found', InvocationID='', MainPID='0', ControlPID='0', ExecMainPID='0')
        spec = dict(output_root='/synthetic/uncreated', unit='synthetic')
        owner = dict(pid=123, start_ticks='456')
        with patch.object(runner.p, 'properties', return_value=props), patch.object(runner.p, 'gone', side_effect=RuntimeError('owner live')):
            with self.assertRaisesRegex(RuntimeError, 'owner live'):
                runner.verify_owned_drain(spec, None, None, owner, True)
        with patch.object(runner.p, 'properties', return_value=props), patch.object(runner.p, 'gone'), patch.object(runner.p, 'empty_group', side_effect=RuntimeError('untracked child live')):
            with self.assertRaisesRegex(RuntimeError, 'untracked child live'):
                runner.verify_owned_drain(spec, None, None, owner, True)

    def test_changed_service_identity_prevents_stop(self):
        spec = dict(output_root='/synthetic/owned', unit='synthetic', master_port=29555)
        a = reference('admission')
        attempt = 'a' * 32
        command = runner.service_command(spec, a, attempt)
        with patch.object(runner.p, 'read', return_value=dict(command=command, admission=a)), patch.object(runner.p, 'properties', return_value=dict(LoadState='loaded')), patch.object(runner, 'validate_service_identity', side_effect=RuntimeError('replaced invocation')), patch.object(runner.p, 'command') as stop:
            with self.assertRaisesRegex(RuntimeError, 'replaced invocation'):
                runner.cleanup_owned_service(spec, a, attempt, None, True)
            stop.assert_not_called()

    def test_collection_worker_and_result_validator_are_new_but_original_resources_retained(self):
        self.assertEqual(runner.PROPS, runner.original_runner.PROPS)
        self.assertIs(runner.validate_rank_result, runner.layer1_results.validate_rank_result)
        self.assertIs(runner.validate_rank_pair, runner.layer1_results.validate_rank_pair)
        tree = ast.parse(Path(runner.__file__).read_text())
        rank = next((x for x in tree.body if getattr(x, 'name', None) == 'rank'))
        imports = [n.names[0].name for n in ast.walk(rank) if isinstance(n, ast.Import)]
        self.assertEqual(imports, ['minicpm5_layer1_own_history_cpu_worker_v2'])
        self.assertEqual((runner.CAP, runner.RANK_CAP), (256 * 2 ** 20, 128 * 2 ** 20))

    def test_complete_drain_receipt_keeps1255services_and_two_samples(self):
        sample = dict(properties=dict(Id='synthetic', LoadState='not-found', MainPID='0', ControlPID='0', ExecMainPID='0', InvocationID='', Unrelated='not-needed'), group=dict(absent=True))
        proof = dict(completed_layer1=dict(physically_drained=True), completed_layer1_owners=[{'synthetic': True}], legacy=dict(original=True), current=dict(processes=450, inherited=[dict(unit='old' + str(i), samples=[copy.deepcopy(sample), copy.deepcopy(sample)]) for i in range(1202)], current=[dict(stage='new' + str(i), samples=[copy.deepcopy(sample), copy.deepcopy(sample)]) for i in range(53)]))
        got = runner.compact_drain(proof)
        self.assertEqual(len(got['observations']), 1255)
        self.assertTrue(all((len(x['samples']) == 2 for x in got['observations'])))
        self.assertTrue(all(('Unrelated' not in x['samples'][0]['properties'] for x in got['observations'])))
        proof['current']['current'].pop()
        with self.assertRaises(RuntimeError):
            runner.compact_drain(proof)

    def test_source_import_does_not_load_native_runtime(self):
        self.assertFalse({'torch', 'numpy', 'torch_neuronx', 'transformers', 'vllm', 'neuronxcc'} & set(sys.modules))

class OwnHistoryAdmission(unittest.TestCase):
 def test_worker_guard_denial_precedes_native_imports_and_lineage_reads(self):
  class Denied:
   def guard(self,role):raise RuntimeError('busy')
  with patch.object(worker.history,'completed')as completed:
   with self.assertRaisesRegex(RuntimeError,'busy'):worker.run(Denied(),Path('/synthetic'))
  completed.assert_not_called()
 def test_runtime_candidate_requires_actual_successful_data_before_source_preparation(self):
  from research import prepare_minicpm5_layer1_own_history_runner_v2 as prepare
  with patch.object(prepare.history,'completed',side_effect=RuntimeError('no actual data exits')),patch.object(prepare,'source_candidate')as sources:
   with self.assertRaisesRegex(RuntimeError,'no actual data exits'):prepare.assemble_runtime_candidate({},'synthetic')
  sources.assert_not_called()
 def test_rank_context_guards_include_actual_data_processes_and_recursive_group(self):
  tree=ast.parse(Path(runner.__file__).read_text());rank=next(n for n in tree.body if getattr(n,'name',None)=='rank')
  text=ast.unparse(rank)
  self.assertIn('after_audit.completed',text);self.assertIn('p.gone(previous)',text);self.assertIn('p.empty_group',text)
  self.assertIn('RuntimeContext',text);self.assertIn('context.verify_sources()',text)
 def test_runner_keeps_original_service_resource_and_cleanup_logic(self):
  old=ast.parse(Path(runner.collector.__file__).read_text());new=ast.parse(Path(runner.__file__).read_text())
  unchanged=('usage','validate_service','validate_service_identity','secondary_error','guarded_receipt',
   'cleanup_children','wait_file','child_admission','verify_owned_drain','cleanup_owned_service')
  for name in unchanged:
   a=next(n for n in old.body if getattr(n,'name',None)==name);b=next(n for n in new.body if getattr(n,'name',None)==name)
   self.assertEqual(ast.dump(a),ast.dump(b),name)
  self.assertEqual(runner.PROPS,runner.collector.PROPS)
  self.assertEqual(runner.CAP,256*2**20);self.assertEqual(runner.RANK_CAP,128*2**20)
 def test_coordinator_requires_actual_rank_exits_and_independent_result_reads(self):
  tree=ast.parse(Path(runner.__file__).read_text());co=next(n for n in tree.body if getattr(n,'name',None)=='coordinator')
  text=ast.unparse(co)
  self.assertIn('validate_rank_pair(results, plan, guard)',text)
  self.assertIn('c.wait() == 0',text);self.assertIn('cleanup_children',text)


if __name__=='__main__':unittest.main(verbosity=2)
