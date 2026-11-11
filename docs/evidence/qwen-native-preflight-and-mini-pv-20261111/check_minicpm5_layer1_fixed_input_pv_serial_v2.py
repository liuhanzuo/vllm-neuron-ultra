"""Pure source/mocked checks; no native imports/captures/history/processes/locks.

Tests construct only synthetic descriptor/output bytes and explicitly mock
production process/FS calls. They do not call actual authenticate/run/worker.
"""
import ast
import copy
import importlib.abc
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'research'), str(ROOT)]


class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'transformers', 'nki', 'neuronxcc', 'torch_neuronx', 'neuronx_distributed_inference'}:
            raise AssertionError('Real native imports forbidden in source-only integration tests')


sys.meta_path.insert(0, DenyNative())
from research import minicpm5_layer1_fixed_input_pv_serial_v1 as serial
from research import minicpm5_layer1_fixed_input_pv_adapter_v2 as adapter
from research.check_minicpm5_layer1_fixed_input_pv_leaf_v1 import complete_plan
core, auth = serial.core, adapter.auth


def make_plan():
    plan = complete_plan()
    for call in plan['calls']:
        for side in ('native', 'sdk'):
            for rows in call[side].values():
                for row in rows: row['reference']['sha256'] = core.sha(bytes(row['spec']['bytes']))
    return plan


def owner(index):
    return dict(pid=100+index, start_ticks=str(200+index), ppid=9, cgroup='0::/synthetic', argv=['synthetic', str(index)])


def terminal(index):
    return dict(phase=index, owner=owner(index), argv=['synthetic', str(index)], returncode=0,
                parent_observed_wait=True, process_group_empty=True, exact_owner_gone=True, self_exit_claimed=False)


class FakePorts:
    def __init__(self): self.calls = []; self.bad_exit = None; self.bad_audit = None; self.gate = None
    def guard(self): self.calls.append('guard')
    def authenticate_complete_prerequisites(self): self.calls.append('authenticate')
    def start(self, index, gate):
        self.calls.append(('start', index, copy.deepcopy(gate)))
        if index >= 2: assert gate == self.gate
        return owner(index), ['synthetic', str(index)]
    def finish(self, index, actual, argv):
        self.calls.append(('finish', index)); result = terminal(index)
        if self.bad_exit is not None and index == 1: result['returncode'] = self.bad_exit
        return result
    def audit_phase(self, index, ended):
        self.calls.append(('audit', index)); env, stage = serial.PHASES[index]
        result = dict(phase=index, own_equalities=[24, 36, 24, 36, 0][index], flatten_equalities=24 if index in (0, 2) else 0,
            all_raw_bytes_reread=True, all_statistics_recomputed=True)
        if self.bad_audit == index: result['own_equalities'] = 23
        return result
    def publish_own_gate(self, completed):
        assert len(completed) == 2; self.calls.append('gate'); self.gate = dict(path='/synthetic/gate.json', sha256='a'*64); return self.gate
    def revalidate_own_gate(self, ref): assert ref == self.gate; self.calls.append('revalidate')
    def final_audit(self, completed, gate):
        assert len(completed) == 5 and gate == self.gate; self.calls.append('final')
        return dict(complete=True, raw_files=4392, raw_bytes=12288000, initial_own_equalities=60, final_own_equalities=60,
            all_five_actual_children_exited_and_drained=True, self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False)


def own_fixture(plan, index):
    env, stage = serial.PHASES[index]; events = []; objects = {}; reads = []
    def path_for(key): return '/synthetic/'+core.sha(key.encode())+'.bin'
    def output(raw, key, shape):
        reference = adapter.ref(raw, path_for(key)); objects[reference['path']] = raw
        stat = serial.comparison.full_stats(raw, 'torch.bfloat16', lambda: None)
        return dict(reference=reference, shape=shape, dtype='torch.bfloat16',
            stats={k: v for k, v in stat.items() if k not in ('bytes', 'sha256', 'dtype')},
            observed_tensor_layout=dict(stride=[core.math.prod(shape[i+1:]) for i in range(len(shape))], offset=0))
    for ident, targets, flatten in serial.own_cases(plan, env, 'own_before'):
        shape = [1, 16, 1, 128] if env == 'native' else [4, 8, 1, 128]
        outputs = {label: output(bytes(row['spec']['bytes']), serial.raw_key(ident, label), shape) for label, row in targets.items()}
        comparisons = {label: serial.comparison.compare_words(bytes(row['spec']['bytes']), bytes(row['spec']['bytes']), 'torch.bfloat16', lambda: None)
            for label, row in targets.items()}
        selected = core.selected_plan(plan, serial.comparison.digest(plan)); group = next(call for call in selected if call['call'] == ident['call'])[env][ident['frame'] if env == 'native' else ident['rank']]
        keys = (('probabilities_bf16', ()), ('native_effective_value', ())) if env == 'native' else (
            ('probabilities_bf16', ()), ('probabilities_active_bf16', ()), ('effective_prior_kv', (1,)), ('v_after_prep', ()))
        operands = [{k: copy.deepcopy(group[key]['metadata'][source]) for k, source in
            (('shape', 'source_shape'), ('stride', 'source_stride'), ('offset', 'source_offset'), ('dtype', 'source_dtype'))} for key in keys]
        operands.extend(copy.deepcopy(operands[-(1 if env == 'native' else 2):]))
        value = dict(identity=ident, outputs=outputs, comparisons_to_original_saved_outputs=comparisons,
            zero_tolerance_control=True, every_control_word_equal=True, metadata=dict(operands=operands),
            causal_operator_proven=False, full_model_qualified=False, new_speedup_claim=False)
        raw = serial.encoded(value); reference = adapter.ref(raw, '/synthetic/event-%d.json'%len(events)); objects[reference['path']] = raw
        events.append((value, reference))
    summary = dict(complete=True, environment=env, stage=stage, descriptor_plan_sha256=serial.comparison.digest(plan),
        actual_tool_exit_verified=False, self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False,
        causal_operator_proven=False, original_sdk_model_exit_code=1, raw_files=serial.leaf.EXPECTED[env, stage][0],
        raw_bytes=serial.leaf.EXPECTED[env, stage][1], evaluation_records=len(events), metadata_bytes=sum(ref['bytes'] for _, ref in events))
    selected = core.selected_plan(plan, serial.comparison.digest(plan))
    inputs = [row for call in selected for group in call[env].values() for row in group.values()]
    summary.update(original_capture_files_read=len(inputs), original_capture_bytes_read=sum(row['spec']['bytes'] for row in inputs),
        prerequisite_authentication_by_caller_required=True,
        runtime_fingerprint=dict(python_executable='/synthetic/python', python_version='synthetic', torch_version='synthetic',
            torch_threads=serial.design.ENVIRONMENTS[env]['torch_threads'], torch_interop_threads=1, default_dtype='torch.float32',
            float32_matmul_precision='highest', deterministic_algorithms=False, mkldnn_enabled=True, mkldnn_deterministic=False,
            mkl_available=True, cpu_capability='synthetic', torch_build_config='synthetic', torch_parallel_config='synthetic', environment=env))
    def read_input(reference, path, size): reads.append(path); return bytes(size)
    def verifier():
        return serial.IndependentOutputs(plan, index, read_input=read_input, read_output=lambda ref: objects[ref['path']],
            event_at=lambda n: events[n], guard=lambda: None, path_for=path_for)
    return NS(events=events, objects=objects, reads=reads, summary=summary, verifier=verifier)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.plan = make_plan()

    def test_01_fixed_five_original_interpreters_and_no_cross_bundle(self):
        ref = dict(path='/synthetic/admission.json', sha256='a'*64)
        for index in range(5):
            argv = serial.phase_argv('/synthetic/worker.py', ref, index)
            self.assertEqual(argv[0], serial.design.ENVIRONMENTS['sdk' if index in (1, 3) else 'native']['python'])
            self.assertEqual(argv[1], '-B'); self.assertEqual(argv[-1], str(index))
        for bad in (True, -1, 5, '1'):
            with self.assertRaises(ValueError): serial.python_for(bad)

    def test_02_all_five_exits_and_own60_gate_precede_variants(self):
        ports = FakePorts(); result = serial.FivePhases(ports).run()
        self.assertTrue(result['complete'])
        calls = [c for c in ports.calls if c != 'guard']
        self.assertEqual(calls[:8], ['authenticate', ('start', 0, None), ('finish', 0), ('audit', 0),
            ('start', 1, None), ('finish', 1), ('audit', 1), 'gate'])
        self.assertEqual(calls.count('revalidate'), 3); self.assertEqual(calls[-1], 'final')

    def test_03_nonzero_bool_missing_exit_cannot_advance(self):
        for code in (1, -9, False, True, '0'):
            ports = FakePorts(); ports.bad_exit = code
            with self.assertRaises(ValueError): serial.FivePhases(ports).run()
            self.assertFalse(any(type(row) is tuple and row[0] == 'start' and row[1] >= 2 for row in ports.calls))
        bad = terminal(0); del bad['returncode']
        with self.assertRaises(ValueError): serial.actual_exit(bad, owner(0), ['synthetic', '0'], 0)

    def test_04_wrong_owner_argv_or_incomplete_drain_rejected(self):
        for key, value in [('owner', owner(1)), ('argv', ['wrong']), ('parent_observed_wait', False),
            ('process_group_empty', False), ('exact_owner_gone', False), ('self_exit_claimed', True), ('phase', False)]:
            bad = terminal(0); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): serial.actual_exit(bad, owner(0), ['synthetic', '0'], 0)

    def test_05_59_equalities_never_authorize_variants(self):
        ports = FakePorts(); ports.bad_audit = 0
        with self.assertRaises(ValueError): serial.FivePhases(ports).run()
        self.assertNotIn('gate', ports.calls)

    def test_06_complete_ordered_domains_and_raw_budget(self):
        all_keys = []; total_events = 0
        for index in range(5):
            rows = serial.event_domain(self.plan, index); total_events += len(rows)
            self.assertEqual(len(rows), serial.EVENT_COUNTS[index])
            if index in (2, 3):
                last_nonown = max(i for i, row in enumerate(rows) if row[0] != 'own')
                self.assertTrue(all(row[0] == 'own' for row in rows[last_nonown+1:]))
                self.assertEqual(sum(row[0] == 'grid' for row in rows), 576)
                self.assertEqual(sum(row[0] == 'binding' for row in rows), 96)
                unequal = [row for row in rows if row[0] == 'grid' and row[1]['frame'] == 27 and row[1]['rank'] == 1]
                self.assertEqual({row[1]['origin'] for row in unequal}, {'native', 'sdk'})
            keys = [serial.encoded(row[1]) for row in rows if row[0] != 'reference']; self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(total_events, 1752)
        self.assertEqual(sum(value[0] for value in serial.leaf.EXPECTED.values()), 4392)
        self.assertEqual(sum(value[1] for value in serial.leaf.EXPECTED.values()), 12288000)
        self.assertEqual(sum((81408, 96624, 3243552, 4092624, 713472))+5*16384+adapter.RUNTIME_METADATA_CAP, 8*2**20)

    def test_07_native24_and_sdk36_full_raw_equalities_independent(self):
        for index, count, input_files in ((0, 24, 48), (1, 36, 36)):
            fixture = own_fixture(self.plan, index); result = fixture.verifier().verify(fixture.summary)
            self.assertEqual(result['own_equalities'], count); self.assertEqual(len(set(fixture.reads)), input_files)
            self.assertEqual(len(fixture.reads), input_files)
            self.assertTrue(result['all_statistics_recomputed'])

    def test_08_forged_flags_and_updated_hash_do_not_hide_own_mismatch(self):
        fixture = own_fixture(self.plan, 0); value, event_ref = fixture.events[0]; out = value['outputs']['context']
        wrong = b'\x01\x00'+bytes(out['reference']['bytes']-2)
        fixture.objects[out['reference']['path']] = wrong; out['reference']['sha256'] = core.sha(wrong)
        stats = serial.comparison.full_stats(wrong, 'torch.bfloat16', lambda: None)
        out['stats'] = {k: v for k, v in stats.items() if k not in ('bytes', 'sha256', 'dtype')}
        raw = serial.encoded(value); new_ref = adapter.ref(raw, event_ref['path']); fixture.events[0] = (value, new_ref); fixture.objects[event_ref['path']] = raw
        with self.assertRaisesRegex(ValueError, 'positive control raw'): fixture.verifier().verify(fixture.summary)

    def test_09_all_statistics_and_event_bytes_are_independently_verified(self):
        for fault in ('stats', 'comparison', 'raw_sha', 'event_bytes', 'identity', 'output_extra'):
            fixture = own_fixture(self.plan, 0); value, reference = fixture.events[0]
            if fault == 'stats': value['outputs']['context']['stats']['invented'] = 1
            elif fault == 'comparison': value['comparisons_to_original_saved_outputs']['context']['all_bits_equal'] = False
            elif fault == 'raw_sha': value['outputs']['context']['reference']['sha256'] = '0'*64
            elif fault == 'identity': value['identity']['frame'] = 9
            elif fault == 'output_extra': value['outputs']['extra'] = value['outputs']['context']
            if fault != 'event_bytes':
                raw = serial.encoded(value); new_ref = adapter.ref(raw, reference['path']); fixture.events[0] = (value, new_ref); fixture.objects[reference['path']] = raw
            else: fixture.objects[reference['path']] = b'{}\n'
            with self.subTest(fault=fault), self.assertRaises(ValueError): fixture.verifier().verify(fixture.summary)

    def test_10_restore_full_descriptor_plan_rejects_partial_original_reports(self):
        summary = dict(descriptor_plan_header={k: v for k, v in self.plan.items() if k != 'calls'})
        objects = {'summary': summary}; refs = []
        for call in self.plan['calls']:
            key = str(call['call']); refs.append(dict(path=key, sha256='a'*64))
            objects[key] = dict(call=call['call'], descriptor_plan_sha256=serial.comparison.digest(self.plan), complete_call_capture_descriptors=call)
        witness = dict(comparison_summary=dict(path='summary', sha256='b'*64), comparison_reports=refs, descriptor_plan_sha256=serial.comparison.digest(self.plan))
        restored = adapter.restore_descriptor_plan(witness, lambda ref: objects[ref['path']]); self.assertEqual(restored, self.plan)
        broken = copy.deepcopy(witness); broken['comparison_reports'].pop()
        with self.assertRaises(ValueError): adapter.restore_descriptor_plan(broken, lambda ref: objects[ref['path']])

    def test_11_finite_flat_namespace_and_source_extension_collisions(self):
        for name in ('0-r-'+'a'*64+'.bin', '4-e-0095.json', 'phase-2-summary.json', 'admission.json'): self.assertTrue(adapter.Store.legal(name))
        for name in ('../escape', '0-r-a.bin', '5-e-0000.json', '.tmp', 'extra.log', '0/foo.bin'): self.assertFalse(adapter.Store.legal(name))
        self.assertEqual(auth.merge_extensions({'a': 'a'}, {'b': 'b'}), {'a': 'a', 'b': 'b'})
        with self.assertRaises(RuntimeError): auth.merge_extensions({'a': 'a'}, {'a': 'b'})

    def test_12_process_finish_really_waits_drains_then_records(self):
        log = []; child = NS(pid=100, poll=lambda: 0, wait=lambda timeout: log.append('wait') or 0)
        ports = adapter.Adapter({}, {}, {}, NS(), lambda **kw: None, None, NS(json=lambda name, value: log.append(('write', name))))
        ports.child, ports.owner, ports.index = child, owner(0), 0
        with patch.object(adapter.r, 'drain_group', side_effect=lambda *a: log.append('drain')), \
             patch.object(adapter.r.p, 'gone', side_effect=lambda *a: log.append('gone')), \
             patch.object(adapter.r, 'group_members', return_value=[]):
            result = ports.finish(0, owner(0), ['synthetic', '0'])
        self.assertEqual(log[:3], ['wait', 'drain', 'gone']); self.assertEqual(result['returncode'], 0)
        self.assertIsNone(ports.child); self.assertEqual(log[-1], ('write', 'phase-0-exit.json'))

    def test_13_failed_physical_drain_never_clears_child_or_returns_release_proof(self):
        child = NS(pid=100); ports = adapter.Adapter({}, {}, {}, NS(), lambda **kw: None, None, None)
        ports.child, ports.owner, ports.index = child, owner(0), 0
        with patch.object(adapter.r, 'drain_group', side_effect=RuntimeError('synthetic surviving descendant')):
            with self.assertRaises(RuntimeError): ports.drain_owned()
        self.assertIs(ports.child, child); self.assertEqual(ports.index, 0)

    def test_14_no_native_import_or_actual_entry_in_tests(self):
        self.assertFalse(any(name.split('.')[0] in {'torch', 'numpy', 'nki', 'transformers'} for name in sys.modules))
        for filename in ('minicpm5_layer1_fixed_input_pv_serial_v1.py', 'minicpm5_layer1_fixed_input_pv_prerequisites_v2.py',
            'minicpm5_layer1_fixed_input_pv_adapter_v2.py', 'minicpm5_layer1_fixed_input_pv_worker_v2.py', 'run_minicpm5_layer1_fixed_input_pv_v2.py'):
            source = (ROOT/'research'/filename).read_text(); compile(source, filename, 'exec'); tree = ast.parse(source)
            self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) and
                (getattr(node, 'module', '') or '').split('.')[0] in {'torch', 'numpy', 'transformers'} for node in tree.body))


    def test_15_completed_independent_auditor_terminal_full_external_chain(self):
        def ref(name): return dict(path=str(ROOT/'research'/name), sha256='a'*64)
        spec_ref, review_ref = ref('synthetic-spec.json'), ref('synthetic-review.json')
        spec = dict(schema='minicpm5_saved_operator_post_exit_audit_spec_v1', comparison_exit=ref('synthetic-exit.json'),
            current_serial_catalog=ref('synthetic-catalog.json'), source_extension={str(ROOT/'research'/'synthetic.py'): 'b'*64},
            output=str(ROOT/'research'/'minicpm5-layer1-saved-operator-comparison-synthetic-post-exit-audit-synthetic.json'), limits=copy.deepcopy(auth.prior.LIMITS))
        review = dict(schema='minicpm5_saved_operator_post_exit_audit_source_review_v1', reviewer='/root', passed=True,
            blockers=[], spec=spec_ref, source_extension=spec['source_extension'])
        argv = ['python3', '-B', auth.prior.SELF, '--spec', spec_ref['path'], '--spec-sha256', spec_ref['sha256'],
            '--review', review_ref['path'], '--review-sha256', review_ref['sha256']]
        actual_owner = dict(owner(0), argv=argv)
        ended = dict(schema='minicpm5_saved_operator_independent_audit_actual_exit_v1', reviewer='/root', audit_spec=spec_ref,
            audit_review=review_ref, live=ref('synthetic-live.json'), result=dict(path=spec['output'], sha256='c'*64), owner=actual_owner,
            actual_tool=dict(kind='tool_session', session_id=123, terminal_chunk='abcdef', exit_code=0), actual_tool_exit_observed=True,
            exact_owner_gone=True, lock_identity=auth.r.ORIGINAL_LOCK, self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False)
        live = dict(schema='root_minicpm5_saved_operator_independent_audit_live_v1', reviewer='/root', observed_at='synthetic', spec=spec_ref,
            review=review_ref, actual_tool=dict(kind='tool_session', session_id=123), owner=actual_owner,
            actual_executable=str(Path(serial.python_for(0)).resolve()), actual_owner_observed=True, actual_tool_exit_observed=False,
            full_model_qualified=False, new_speedup_claim=False)
        result = dict(schema=auth.prior.RESULT_SCHEMA, complete=True, reviewer='/root', audit_spec=spec_ref, audit_review=review_ref, audit_owner=actual_owner)
        self.assertEqual(auth.audit_terminal(ended, live, result, spec, review)['session_id'], 123)
        for fault in ('exit', 'bool_exit', 'owner', 'python', 'live_session', 'self_exit', 'missing_ref'):
            e, l, v = copy.deepcopy(ended), copy.deepcopy(live), copy.deepcopy(result)
            if fault == 'exit': e['actual_tool']['exit_code'] = 1
            elif fault == 'bool_exit': e['actual_tool']['exit_code'] = False
            elif fault == 'owner': l['owner']['start_ticks'] = '999'
            elif fault == 'python': l['actual_executable'] = '/synthetic/wrong-python'
            elif fault == 'live_session': l['actual_tool']['session_id'] += 1
            elif fault == 'self_exit': e['self_exit_claimed'] = True
            else: del e['live']
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): auth.audit_terminal(e, l, v, spec, review)

    def test_16_aggregate_allocation_and_extra_files_rejected_without_actual_fs(self):
        def entry(name, size=1024, blocks=8, nlink=1, mode=0o100600):
            st = NS(st_mode=mode, st_uid=1000, st_nlink=nlink, st_size=size, st_blocks=blocks,
                st_dev=1, st_ino=2, st_mtime_ns=1, st_ctime_ns=1)
            return NS(name=name, suffix=Path(name).suffix, lstat=lambda: st)
        for fault in ('extra', 'symlink', 'hardlink', 'allocation', 'metadata'):
            store = adapter.Store('/synthetic', lambda: None); store._directory = lambda: NS(st_blocks=8)
            if fault == 'extra': files = [entry('extra.json')]
            elif fault == 'symlink': files = [entry('admission.json', mode=0o120777)]
            elif fault == 'hardlink': files = [entry('admission.json', nlink=2)]
            elif fault == 'allocation': files = [entry('admission.json', blocks=serial.CAPS['allocated_bytes']//512)]
            else: files = [entry('phase-%d-summary.json'%i, size=2*2**20) for i in range(5)]
            store.root = NS(iterdir=lambda: iter(files))
            with self.subTest(fault=fault), self.assertRaises(ValueError): store.inventory()

    def test_17_actual_spawn_retains_parent_fd_and_original_environment(self):
        for index in (0, 1):
            admission_ref = dict(path='/synthetic/admission.json', sha256='a'*64)
            path = '/synthetic/worker.py'; argv = serial.phase_argv(path, admission_ref, index)
            outer = dict(owner(0), pid=9); child_owner = dict(owner(index), argv=argv, ppid=9)
            chunks = []
            pipe = NS(write=lambda raw: chunks.append(raw) or len(raw), flush=lambda: None, close=lambda: None)
            child = NS(pid=child_owner['pid'], stdin=pipe, poll=lambda: None)
            lease = NS(admission=dict(outer_owner=outer, attempt_id='a'*32))
            authenticated = dict(spec=dict(worker=path), context=dict(pins_plan=dict(source_contract=dict(path='/synthetic/source.json', sha256='b'*64), source_sha256={path: 'c'*64})))
            store = NS(json=lambda *args: None)
            ports = adapter.Adapter(authenticated, admission_ref, {}, lease, lambda **kw: None, None, store)
            ports.last_exit = index-1 if index else None
            with patch.object(adapter.subprocess, 'Popen', return_value=child) as spawn, \
                 patch.object(adapter.r.sources, 'text', return_value=b'worker'), \
                 patch.object(adapter.r.p, 'proc', return_value=child_owner), \
                 patch.object(adapter.Path, 'resolve', return_value=Path('/synthetic/python')):
                actual, observed = ports.start(index, None)
            self.assertEqual(observed, argv); self.assertEqual(actual, child_owner)
            kwargs = spawn.call_args.kwargs
            self.assertTrue(kwargs['close_fds']); self.assertTrue(kwargs['start_new_session']); self.assertNotIn('pass_fds', kwargs)
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '2' if index == 0 else '1')
            packet = json.loads(b''.join(chunks)); self.assertEqual(packet['owner'], child_owner)
            self.assertEqual(packet['source_plan'], authenticated['context']['pins_plan'])



    def test_18_actual_repeat_callable_and_framework_pin_must_match(self):
        path, sha, module = adapter.REPEAT_SOURCES['native']; torch_path = '/synthetic/torch.py'
        def fake(): pass
        fake.__module__ = module; fake.__name__ = 'repeat_kv'
        fake.__code__ = fake.__code__.replace(co_filename=path, co_firstlineno=1)
        torch = NS(__file__=torch_path); pins = {path: sha, torch_path: 'a'*64}
        with patch.object(adapter.r.sources, 'text', return_value=b'def repeat_kv(): pass\n'):
            adapter.authenticate_framework(torch, fake, 'native', pins, lambda: None)
            for fault in ('module', 'source_pin', 'framework_pin', 'line'):
                bad = dict(pins); original_module, original_code = fake.__module__, fake.__code__
                if fault == 'module': fake.__module__ = 'invented'
                elif fault == 'source_pin': bad[path] = 'b'*64
                elif fault == 'framework_pin': del bad[torch_path]
                else: fake.__code__ = fake.__code__.replace(co_firstlineno=2)
                with self.subTest(fault=fault), self.assertRaises(ValueError): adapter.authenticate_framework(torch, fake, 'native', bad, lambda: None)
                fake.__module__, fake.__code__ = original_module, original_code


if __name__ == '__main__': unittest.main(verbosity=2)
