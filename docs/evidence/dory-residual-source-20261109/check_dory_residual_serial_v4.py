"""Bounded source regressions: no original event traversal or native imports."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import dory_residual_serial_contract_v4 as c
import dory_after_nemotron_r9_v2 as history
import dory_residual_worker_v4 as worker
import run_dory_residual_serial_v4 as runner


def prepared_spec(mode='native'):
    old, proposal = c.load(c.ORIGINAL), c.load(c.PROPOSAL)
    spec = copy.deepcopy(old)
    names = ('dory_after_nemotron_r9_v2.py', 'dory_residual_serial_contract_v4.py',
        'run_dory_residual_serial_v4.py', 'dory_residual_worker_v4.py',
        'review_dory_layer0_serial_snapshot_v1.py', 'run_dory_layer0_serial_v2.py',
        'dory_residual_worker_preparation_v2.py', 'dory_residual_replay_runtime_v2.py',
        'dory_residual_visit_capture_v1.py', 'dory_residual_request_banks_v1.py',
        'dory_residual_replay_history_v1.py', 'dory_residual_event_stream_v1.py',
        'dory_residual_output_archive_v1.py', 'dory_long_delta_writer_v1.py',
        'dory_residual_saved_archives_v2.py', 'dory_residual_saved_observations_v1.py',
        'dory_residual_saved_payloads_v2.py', 'dory_residual_saved_complete_v2.py',
        'dory_checkpoint_loader_v1.py', 'dory_generation_weight_provenance_v1.py',
        'dory_generation_execution_provenance_v1.py', 'dory_generation_journal_replay_v1.py',
        'dory_generation_cache_replay_v1.py', 'review_dory_long_matrix_saved_v3.py',
        'explain_dory_long_cache_differences_v1.py')
    # Fixture pins are not a source review or admitted runtime specification.
    for name in names:
        spec['source_sha256'][str(Q/name)] = 'a'*64
    for ref in (c.PLAN, c.PROPOSAL):
        spec['source_sha256'][ref['path']] = ref['sha256']
    spec['source_sha256']['/synthetic/metadata-exit.json'] = 'b'*64
    spec['source_sha256'][c.FAILED_NATIVE['path']] = c.FAILED_NATIVE['sha256']
    spec.update(schema='dory_residual_complete_replay_diagnostic_v1', execution_mode=mode,
        diagnostic_only=True, runtime_admitted=True, review_only_spec=mode == 'review',
        selection_plan=c.PLAN, resource_proposal=c.PROPOSAL,
        metadata_output=str(Path(old['base'])/'residual-metadata-r1'),
        output=str(Path(old['base'])/'residual-replay-r2'),
        review_output=str(Path(old['base'])/'residual-replay-review-r2'),
        worker=str(Q/'dory_residual_worker_v4.py'), executor=str(Q/'run_dory_residual_serial_v4.py'),
        reviewer=str(Q/'dory_residual_saved_complete_v2.py'),
        expected_original_raw_bytes=c.RAW_BYTES, payload_review_budget=proposal['proposed_payload_review_budget'],
        budget=copy.deepcopy(c.METADATA_BUDGET if mode == 'metadata' else
            proposal['proposed_native_budget' if mode == 'native' else 'proposed_saved_review_budget']),
        metadata_preparation_exit=dict(path='/synthetic/metadata-exit.json', sha256='b'*64),
        failed_native_exit=c.FAILED_NATIVE,
        full_model_qualified=False, new_speedup_claim=False, automatic_launch=False)
    # Restore exact already-pinned original files if a dependency overlaps.
    spec['source_sha256'].update(old['source_sha256'])
    spec['essential_sources'] = list(spec['source_sha256'])
    return spec


def metadata_result():
    return dict(complete=True, mode='metadata', owner={'pid': 100}, actual_self_exit_claimed=False,
        native_started=False, tensor_payload_validation_performed=False,
        source_file_authentication=dict(source_and_binary_files_authenticated=True,
            file_count=2, total_bytes_read=26, bytes_by_suffix={'.py': 10, '.bin': 16}),
        full_model_qualified=False, new_speedup_claim=False,
        summary=dict(events=6200, native_successful_calls_required=6185, native_rejections_required=3,
            native_injections_required=1, selected_calls=89, original_pointwise_failures=7,
            tensor_labels=114, cache_tensor_labels=112, original_raw_archive_bytes=c.RAW_BYTES,
            original_selection_plan=c.PLAN, metadata_only=True, native_execution=False,
            raw_tensor_bytes_verified=False, raw_tensor_bytes_read=0))


class SourceContract(unittest.TestCase):
    def test_corrected_physical_memory_floor_is_only_budget_change(self):
        prior = c.load(dict(path=str(Q/'dory-residual-resource-proposal-20261109-v1.json'),
            sha256='6dfbd595c598745a10c30157ebb904b2755831fd38ab8ae127688aee9bf792ee'))
        current = c.load(c.PROPOSAL)
        expected = dict(prior['proposed_native_budget'], initial_memory_available_bytes=26*2**30)
        self.assertEqual(current['proposed_native_budget'], expected)
        self.assertLess(expected['initial_memory_available_bytes'], current['actual_memtotal_bytes'])
        self.assertEqual(current['proposed_saved_review_budget'], prior['proposed_saved_review_budget'])
        self.assertEqual(current['proposed_payload_review_budget'], prior['proposed_payload_review_budget'])

    def reject(self, field, value, mode='native'):
        spec = prepared_spec(mode)
        spec[field] = value
        with self.assertRaises((RuntimeError, KeyError)):
            c.validate(spec, mode)

    def test_three_explicit_modes(self):
        for mode in c.MODES:
            self.assertIs(c.validate(spec := prepared_spec(mode), mode), spec)

    def test_original_checkpoint(self):
        self.reject('checkpoint', {'path': '/replacement'})

    def test_original_tokenizer(self):
        self.reject('tokenizer_directory', '/replacement')

    def test_original_official_source(self):
        self.reject('official_source', '/replacement')

    def test_no_precision_or_input_pin_replacement(self):
        spec = prepared_spec()
        key = next(iter(c.load(c.ORIGINAL)['source_sha256']))
        spec['source_sha256'][key] = '0'*64
        with self.assertRaises(RuntimeError):
            c.validate(spec, 'native')

    def test_no_draft_admission(self):
        self.reject('runtime_admitted', False)

    def test_no_old_prefix_namespace(self):
        self.reject('output', str(Path(prepared_spec()['base'])/'first5-propagation-r1'))

    def test_no_shortened_plan(self):
        self.reject('selection_plan', dict(c.PLAN, sha256='c'*64))

    def test_no_unbounded_memory(self):
        spec = prepared_spec()
        spec['budget']['address_space_bytes'] *= 2
        with self.assertRaises(RuntimeError):
            c.validate(spec, 'native')

    def test_no_reduced_storage(self):
        spec = prepared_spec()
        spec['budget']['nvme_new_bytes'] //= 2
        with self.assertRaises(RuntimeError):
            c.validate(spec, 'native')

    def test_v2_archive_reader_mandatory(self):
        spec = prepared_spec()
        spec['essential_sources'].remove(str(Q/'dory_residual_saved_archives_v2.py'))
        with self.assertRaises(RuntimeError):
            c.validate(spec, 'native')

    def test_no_native_without_preflight_pin(self):
        spec = prepared_spec()
        del spec['source_sha256']['/synthetic/metadata-exit.json']
        spec['essential_sources'].remove('/synthetic/metadata-exit.json')
        with self.assertRaises(RuntimeError):
            c.validate(spec, 'native')

    def test_preflight_pin_cannot_be_rebound(self):
        spec = prepared_spec()
        spec['source_sha256']['/synthetic/metadata-exit.json'] = 'c'*64
        with self.assertRaises(RuntimeError):
            c.validate(spec, 'native')

    def test_unreviewed_receipt_rejected_before_read(self):
        with patch.object(c, 'load') as read:
            with self.assertRaises(RuntimeError):
                c.reviewed_ref({'source_sha256': {}}, {'path': '/missing', 'sha256': 'a'*64})
            read.assert_not_called()

    def test_no_mode_confusion(self):
        with self.assertRaises(RuntimeError):
            c.validate(prepared_spec('review'), 'native')

    def test_no_speedup_promotion(self):
        self.reject('new_speedup_claim', True)

    def test_no_auto_successor(self):
        self.reject('automatic_launch', True)


class MetadataScope(unittest.TestCase):
    def test_exact_complete_summary(self):
        self.assertIs(c.check_metadata_result(value := metadata_result()), value)

    def test_incomplete_histories_rejected(self):
        for field, value in (('events', 6199), ('native_successful_calls_required', 6184),
            ('native_rejections_required', 2), ('native_injections_required', 0), ('selected_calls', 88),
            ('original_pointwise_failures', 0), ('tensor_labels', 8), ('cache_tensor_labels', 6),
            ('original_raw_archive_bytes', c.RAW_BYTES-1)):
            with self.subTest(field=field):
                result = metadata_result()
                result['summary'][field] = value
                with self.assertRaises(RuntimeError):
                    c.check_metadata_result(result)

    def test_no_metadata_to_tensor_or_native_credit(self):
        for field in ('native_execution', 'raw_tensor_bytes_verified'):
            result = metadata_result()
            result['summary'][field] = True
            with self.assertRaises(RuntimeError):
                c.check_metadata_result(result)

    def test_no_self_reported_external_exit(self):
        result = metadata_result()
        result['actual_self_exit_claimed'] = True
        with self.assertRaises(RuntimeError):
            c.check_metadata_result(result)

    def test_no_partial_stream_native_import(self):
        fake_type = type('IncompleteStream', (), {'__iter__': lambda self: iter([]), 'completed_summary': None})
        with patch('dory_residual_event_stream_v1.OriginalEventStream', return_value=fake_type()):
            with self.assertRaises(RuntimeError):
                runner.metadata({'selection_plan': c.PLAN}, lambda: None, {}, {})


class RecordedOwners(unittest.TestCase):
    def owner(self, **kw):
        return dict(dict(pid=321, start_ticks='456', ppid=12, argv=['python', 'wrapper'], cgroup='0::/test'), **kw)

    def test_nested_complete_owner(self):
        owner = self.owner()
        self.assertEqual(list(history.captured_processes({'rows': [{'process': owner}]})), [owner])

    def test_no_inferred_start(self):
        self.assertEqual(list(history.captured_processes({'child': {'pid': 321, 'argv': ['python']}})), [])

    def test_invalid_recorded_start_rejected(self):
        with self.assertRaises(RuntimeError):
            list(history.captured_processes(self.owner(start_ticks='0')))

    def test_pre_and_post_exec_preserved(self):
        before, after = self.owner(), self.owner(argv=['python', 'server'])
        prior = [self.owner(pid=111)]
        processes, variants = history.extend_observed(prior, [('a', before), ('b', after)])
        self.assertEqual(len(processes), 2)
        self.assertEqual([v['owner'] for v in variants], [before, after])
        self.assertEqual(processes[0], prior[0])

    def test_pid_reuse_is_distinct(self):
        processes, variants = history.extend_observed([], [('a', self.owner()), ('b', self.owner(start_ticks='789'))])
        self.assertEqual(len(processes), 2)

    def test_repeated_identical_observation(self):
        processes, variants = history.extend_observed([], [('a', self.owner()), ('a', self.owner())])
        self.assertEqual((len(processes), len(variants)), (1, 1))

    def test_empty_exec_transition_retained_with_actual_later_command(self):
        processes, variants = history.extend_observed([], [('transition', self.owner(argv=[])),
                                                           ('executed', self.owner(argv=['python','server']))])
        self.assertEqual(processes[0]['argv'], ['python','server'])
        self.assertEqual(variants[0]['owner']['argv'], [])

    def test_no_command_invented_when_only_empty_argv_observed(self):
        processes, variants = history.extend_observed([], [('transition', self.owner(argv=[]))])
        self.assertEqual(processes[0]['argv'], [])
        self.assertEqual(variants[0]['owner']['argv'], [])


class ChildAdmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = dict(environment='/venv', worker='/worker.py', output=str(self.root), budget={'memory': 32},
                         metadata_preparation_exit={'path': 'exit', 'sha256': 'a'*64})
        self.spec_path = self.root/'spec.json'
        self.spec_path.write_text(json.dumps(self.spec))
        self.command = ['/venv/bin/python', '-B', '/worker.py', '--spec', str(self.spec_path),
                        '--admission', str(self.root/'admission.json')]
        self.owner = dict(pid=os.getpid(), ppid=77, start_ticks='900', argv=self.command)
        self.h = dict(rows=[{}]*1149, processes=[{}, {}, {}],
                      residual_metadata_preparation_exit=self.spec['metadata_preparation_exit'])
        self.drain = dict(complete=True, services=1149, processes=3)
        self.ad = dict(spec_sha256=hashlib.sha256(self.spec_path.read_bytes()).hexdigest(), mode='native',
            review=False, performance_measurement=False, budget=self.spec['budget'], command=self.command,
            owner={'pid': 77}, lock={'inode': 9}, predecessor_history={'path': 'history'},
            predecessor_drain_record={'path': 'drain'}, predecessor_drain=self.drain)

    def call(self):
        with patch.object(c, 'validate'), patch.object(c, 'load', side_effect=lambda r: self.h if r['path'] == 'history' else self.drain):
            return c.validate_child(self.spec_path, self.spec, self.ad, lambda pid: self.owner, lambda *args: None)

    def test_exact_dynamic_process_count(self):
        self.assertEqual(self.call(), self.owner)

    def test_pre_r9_service_total_rejected(self):
        self.drain['services'] = 1096
        with self.assertRaises(RuntimeError):
            self.call()

    def test_missing_actual_owner_rejected(self):
        self.drain['processes'] = 2
        with self.assertRaises(RuntimeError):
            self.call()

    def test_wrong_parent_rejected(self):
        self.owner['ppid'] = 78
        with self.assertRaises(RuntimeError):
            self.call()

    def test_changed_spec_rejected(self):
        self.spec_path.write_text('{}')
        with self.assertRaises(RuntimeError):
            self.call()

    def test_wrong_preflight_receipt_rejected(self):
        self.h['residual_metadata_preparation_exit'] = {'path': 'other'}
        with self.assertRaises(RuntimeError):
            self.call()

    def test_missing_parent_ack_prevents_original_setup(self):
        self.run_worker(b'X', self.owner, False)

    def test_changed_child_after_ack_prevents_original_setup(self):
        self.run_worker(b'R', dict(self.owner, start_ticks='901'), False)

    def test_original_setup_only_after_actual_ack(self):
        self.run_worker(b'R', self.owner, True)

    def run_worker(self, ack, live, succeeds):
        (self.root/'admission.json').write_text(json.dumps(self.ad))
        (self.root/'child-owner.json').write_text(json.dumps(self.owner))
        called = []
        fake = SimpleNamespace(main=lambda: called.append('native'))
        import minicpm5_layer0_execution_protocol_v4 as low
        with patch.object(sys, 'argv', ['worker', '--spec', str(self.spec_path), '--admission', str(self.root/'admission.json')]), \
             patch.dict(os.environ, {'DORY_RESIDUAL_READY_FD': '123'}), \
             patch.object(worker, 'validate_child', return_value=self.owner), \
             patch.object(worker.select, 'select', return_value=([123], [], [])), \
             patch.object(worker.os, 'read', return_value=ack), patch.object(worker.os, 'close'), \
             patch.object(low, 'proc', return_value=live), \
             patch.dict(sys.modules, {'dory_residual_worker_preparation_v2': fake}):
            if succeeds:
                worker.main()
            else:
                with self.assertRaises(RuntimeError):
                    worker.main()
        self.assertEqual(called, ['native'] if succeeds else [])


class NativeEnvironment(unittest.TestCase):
    def test_serial_modules_do_not_load_native_libraries(self):
        self.assertFalse({'torch', 'numpy', 'transformers', 'vllm'} & set(sys.modules))

    def test_original_offline_single_thread_setup(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/injected'}):
            env = runner.environment({'base': '/base', 'environment': '/venv'}, 17)
        self.assertNotIn('PYTHONPATH', env)
        self.assertEqual(env['DORY_RESIDUAL_READY_FD'], '17')
        self.assertEqual(env['OMP_NUM_THREADS'], '1')
        self.assertEqual(env['PYTHONOPTIMIZE'], '0')
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(env['LD_PRELOAD'], '/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/venv/lib/libiomp5.so')


class CompletedMetadata(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.prep = dict(metadata_output=str(self.base/'residual-metadata-r1'),
                         r9_actual_exit={'path': 'r9-exit'}, r9_additional_actual_exits=[],
                         r9_recovery_operations={'path': 'actual-operations'})
        self.owner = dict(pid=100, start_ticks='200')
        self.refs = {name: dict(path=str(self.base/name), sha256='a'*64)
                     for name in ('exit', 'spec', 'history')}
        for name in ('admission', 'result', 'source_files'):
            filename = 'source-files' if name == 'source_files' else name
            self.refs[name] = dict(path=str(Path(self.prep['metadata_output'])/(filename+'.json')), sha256='a'*64)
        self.h = dict(schema='dory_post_nemotron_r9_history_v1', rows=[{}]*1149,
            complete_count=1149, r9_stages=[{}]*53, r9_actual_exit=self.prep['r9_actual_exit'],
            r9_additional_actual_exits=[], r9_recovery_operations=self.prep['r9_recovery_operations'],
            canonical_sha256={'frozen': 'a'},
            predecessor_source_sha256={}, processes=[dict(pid=50, start_ticks='60')],
            known_admitted_directories=['previous'])
        self.result = metadata_result()
        self.result['owner'] = self.owner
        self.sources = dict(source_and_binary_files_authenticated=True, total_bytes_read=26,
                            files={'/synthetic/source.py': {'bytes': 10}, '/synthetic/original.bin': {'bytes': 16}})
        self.ad = dict(spec_sha256='a'*64, owner=self.owner, mode='metadata', performance_measurement=False,
            predecessor_history=self.refs['history'], canonical_sha256=self.h['canonical_sha256'])
        self.receipt = dict(reviewer='/root', actual_external_exit_observed=True, session_id=1,
            exit_code=0, tool_chunk='abc', owner=self.owner, spec=self.refs['spec'],
            admission=self.refs['admission'], result=self.refs['result'], predecessor_history=self.refs['history'])
        self.receipt['source_files'] = self.refs['source_files']
        self.values = dict(exit=self.receipt, spec=self.prep, history=self.h, admission=self.ad,
                           result=self.result, source_files=self.sources)
        self.by_path = {self.refs[k]['path']: v for k, v in self.values.items()}
        self.spec = dict(metadata_preparation_exit=self.refs['exit'], execution_mode='native', base=str(self.base),
                         source_sha256={r['path']: r['sha256'] for r in self.refs.values()})
        for name in ('previous', 'residual-metadata-r1'):
            (self.base/name).mkdir()
            (self.base/name/'admission.json').write_text('{}')
        self.failed_root = self.base/'residual-replay-r1'
        self.failed_root.mkdir()
        (self.failed_root/'admission.json').write_text('{}')
        self.spec['failed_native_exit'] = c.FAILED_NATIVE
        failure_refs = {name: dict(path='/synthetic/failed-'+name,sha256='f'*64)
                        for name in ('spec','admission','child','child_owner_record','child_exit','failure','history')}
        parent = dict(pid=300,start_ticks='400')
        child_owner = dict(pid=301,start_ticks='401',ppid=300,argv=['python','worker'])
        expected_history = copy.deepcopy(self.h)
        expected_history['processes'].append(self.owner)
        expected_history['known_admitted_directories'].append('residual-metadata-r1')
        expected_history['residual_metadata_preparation_exit'] = self.refs['exit']
        self.failed_history = expected_history
        self.failed_admission = dict(owner=parent,spec_sha256='f'*64,mode='native',
            command=['python','worker'],predecessor_history=failure_refs['history'])
        failed_values = dict(spec=dict(output=str(self.failed_root)),admission=self.failed_admission,
            child=dict(pid=301,command=['python','worker']),child_owner_record=child_owner,
            child_exit=dict(child_pid=301,child_owner=child_owner,returncode=1),failure={},history=expected_history)
        self.failed_receipt = dict(failure_refs,reviewer='/root',actual_external_exit_observed=True,
            session_id=34882,exit_code=1,tool_chunk='f9f680',successful_native_calls=0,model_initialized=True,
            checkpoint_loaded=True,owner=parent,child_owner=child_owner)
        self.by_path[c.FAILED_NATIVE['path']] = self.failed_receipt
        self.spec['source_sha256'][c.FAILED_NATIVE['path']] = c.FAILED_NATIVE['sha256']
        for name,value in failed_values.items():
            ref=failure_refs[name]
            self.by_path[ref['path']]=value
            self.spec['source_sha256'][ref['path']]=ref['sha256']

    def call(self):
        with patch.object(c, 'load', side_effect=lambda r: copy.deepcopy(self.by_path[r['path']])), \
             patch('dory_residual_serial_contract_v2.validate', side_effect=lambda spec, mode: spec), \
             patch('dory_residual_serial_contract_v3.validate', side_effect=lambda spec, mode: spec), \
             patch.object(c, 'validate', side_effect=lambda spec, mode: spec):
            return c.metadata_history(self.spec, lambda owner: None)

    def test_completed_preflight_owner_added_once(self):
        actual = self.call()
        self.assertEqual(actual['processes'], self.h['processes']+[self.owner,
            self.failed_receipt['owner'],self.failed_receipt['child_owner']])
        self.assertEqual(len(self.h['processes']), 1)

    def test_failed_preflight_cannot_admit_native(self):
        self.receipt['exit_code'] = 1
        with self.assertRaises(RuntimeError):
            self.call()

    def test_pre_r9_snapshot_cannot_admit_native(self):
        self.h['rows'] = self.h['rows'][:1096]
        self.h['complete_count'] = 1096
        with self.assertRaises(RuntimeError):
            self.call()

    def test_wrong_preflight_owner_rejected(self):
        self.ad['owner'] = dict(pid=101, start_ticks='201')
        with self.assertRaises(RuntimeError):
            self.call()

    def test_unknown_intervening_job_rejected(self):
        (self.base/'unknown').mkdir()
        (self.base/'unknown/admission.json').write_text('{}')
        with self.assertRaises(RuntimeError):
            self.call()

    def test_missing_completed_receipt_pin_rejected(self):
        del self.spec['source_sha256'][self.refs['result']['path']]
        with self.assertRaises(RuntimeError):
            self.call()

    def test_r9_source_rebinding_rejected(self):
        self.h['predecessor_source_sha256'] = {'/old-evidence': 'a'*64}
        self.spec['source_sha256']['/old-evidence'] = 'b'*64
        with self.assertRaises(RuntimeError):
            self.call()

    def test_wrong_source_byte_ledger_rejected(self):
        self.result['source_file_authentication']['bytes_by_suffix'] = {'.py': 26}
        with self.assertRaises(RuntimeError):
            self.call()

    def test_wrong_failed_native_child_rejected(self):
        self.failed_receipt['child_owner'] = dict(pid=999,start_ticks='888')
        with self.assertRaises(RuntimeError):self.call()

    def test_failed_native_predecessor_history_cannot_change(self):
        self.failed_history['complete_count'] = 1096
        with self.assertRaises(RuntimeError):self.call()


class SourceByteAccounting(unittest.TestCase):
    def test_binary_source_reads_are_retained(self):
        sources = dict(source_and_binary_files_authenticated=True, total_bytes_read=100,
            files={'/source.py': {'bytes': 20}, '/old.bin': {'bytes': 80}})
        self.assertEqual(c.source_accounting(sources)['bytes_by_suffix'], {'.py': 20, '.bin': 80})

    def test_wrong_total_rejected(self):
        with self.assertRaises(RuntimeError):
            c.source_accounting(dict(source_and_binary_files_authenticated=True,
                total_bytes_read=0, files={'/old.bin': {'bytes': 80}}))

    def test_blanket_zero_tensor_read_claim_rejected(self):
        result = metadata_result()
        result['raw_tensor_bytes_read'] = 0
        with self.assertRaises(RuntimeError):
            c.check_metadata_result(result)

    def test_hashing_does_not_become_tensor_validation(self):
        result = metadata_result()
        result['tensor_payload_validation_performed'] = True
        with self.assertRaises(RuntimeError):
            c.check_metadata_result(result)


class ParentCleanup(unittest.TestCase):
    def run_failed_child(self, code, descendants):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            spec = prepared_spec()
            spec.update(output=str(root/'native'), base=str(root), source_sha256={},
                        essential_sources=[], source_read_cap_bytes=1000)
            parent = dict(pid=100, start_ticks='10')
            command = [str(Path(spec['environment'])/'bin/python'), '-B', spec['worker'],
                       '--spec', '/synthetic/spec', '--admission', str(root/'native/admission.json')]
            child_owner = dict(pid=101, start_ticks='11', ppid=100, argv=command)
            child = SimpleNamespace(pid=101, returncode=code, poll=lambda: code)
            events = []
            review = dict(passed=True, reviewer='/root', blockers=[], spec_sha256='a'*64, source_sha256={})
            h = dict(rows=[{}]*1149, processes=[{}], predecessor_source_sha256={}, canonical_sha256={})
            for target, replacement in (
                ('load', lambda ref: spec if ref['path'] == '/synthetic/spec' else review),
                ('validate', lambda spec, mode: spec), ('acquire', lambda: (99, parent, {})),
                ('release', lambda fd: events.append('unlock')), ('free', lambda path: 100*2**30),
                ('memory_free', lambda: 100*2**30), ('review_source_pins', lambda *a, **kw: {}),
                ('metadata_history', lambda *a: h), ('group_members', lambda pid: descendants),
                ('terminate_owned_group', lambda *a: events.append('drain'))):
                stack.enter_context(patch.object(runner, target, replacement))
            stack.enter_context(patch('review_dory_layer0_serial_snapshot_v1.physical_drain',
                return_value=dict(complete=True, services=1149, processes=1)))
            stack.enter_context(patch.object(runner.p, 'canonical', return_value={}))
            for name in ('user_bus', 'lock_proof', 'clear_window', 'active_heavy', 'gone'):
                stack.enter_context(patch.object(runner.p.low, name))
            stack.enter_context(patch.object(runner.p.low, 'proc', return_value=child_owner))
            stack.enter_context(patch.object(runner.os, 'sched_setaffinity'))
            stack.enter_context(patch.object(runner.os, 'nice'))
            stack.enter_context(patch.object(runner.subprocess, 'Popen', return_value=child))
            # Fake child does not own the pipe; mock ACK write to avoid EPIPE.
            stack.enter_context(patch.object(runner.os, 'write', return_value=1))
            stack.enter_context(patch.object(sys, 'argv', ['runner', '--spec', '/synthetic/spec',
                '--spec-sha256', 'a'*64, '--review', '/synthetic/review', '--review-sha256', 'b'*64,
                '--mode', 'native']))
            with self.assertRaises(RuntimeError):
                runner.main()
            self.assertEqual(events, ['drain', 'unlock'])
            self.assertTrue((root/'native/failure.json').is_file())
            self.assertEqual(json.loads((root/'native/child-exit.json').read_text())['returncode'], code)
            self.assertFalse((root/'native/result.json').exists())

    def test_failed_child_drained_before_unlock(self):
        self.run_failed_child(1, [])

    def test_successful_exit_with_live_descendant_is_failure(self):
        self.run_failed_child(0, [dict(pid=102, start_ticks='12')])


if __name__ == '__main__':
    unittest.main(verbosity=2)
