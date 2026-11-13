"""Source-only synthetic storage/terminal controls, adapted from accepted package v8.

Every actual asset read uses a tempfile fixture. No native imports, real SDK
binary reads, History operations, services, real lock, or resource observation.
"""
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_source import Q, cases
from qwen35_writer_pair_serial_runtime_v2 import contract as c, io, budget as b, adapter as a, compiler_assets as ca, qualification
budget = b
HERE = Path(__file__).resolve().parent


def spec_fixture():
    pin = {'path':'/synthetic/unbound.json','sha256':'a'*64}
    return dict(run_name='mock-r1', output_root='/synthetic/output', writer_cases=cases(),
        history_overlay=pin, current_serial_catalog=pin, source_sha256={'/synthetic/source.py':'a'*64},
        running_handoff_path='/synthetic/running.json')


class Storage:
    def fixture(self, temp):
        root=Path(temp); spec=spec_fixture(); spec['output_root']=str(root/'output')
        for arm in ('baseline','candidate'):
            for key in b.CASE_KEYS:spec['writer_cases'][arm][key]=str(root/(arm+'-'+key))
        reviewed={'classes':{key:dict(allocated_bytes=64*c.MIB,single_file_bytes=32*c.MIB,entries=1000) for key in b.CLASSES}}
        return spec, reviewed, root/'control'



class CompilerStorage(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            spec, reviewed, control = Storage().fixture(temp)
            with patch.object(io, 'control_root', return_value=control), \
                    patch.object(b, 'control_root', return_value=control), \
                    patch.object(b, 'reviewed_budget', return_value=reviewed), \
                    patch.object(b, 'unchanged_inputs', side_effect=lambda spec, value: value), \
                    patch.object(io.time, 'sleep') as sleep:
                yield spec, reviewed, control, sleep

    def cache(self, spec):
        directory = Path(spec['writer_cases']['baseline']['compiler_cache']) / 'neuronxcc-2.23.6484.0+3b612583' / (
            'MODULE_' + 'a' * 20 + '+' + 'b' * 8)
        directory.mkdir(parents=True)
        return directory

    def work(self, spec):
        directory = Path(spec['writer_cases']['baseline']['compiler_workdir']) / 'context_encoding_model' / '_tp0_bk0'
        directory.mkdir(parents=True)
        return directory

    def vanish_on_stat(self, victim, operation):
        original = io.observed_stat
        changed = []
        def observe(path):
            if path == victim and not changed:
                changed.append(True); operation()
            return original(path)
        return patch.object(io, 'observed_stat', side_effect=observe)

    def test_sdk_atomic_cache_publish_recounts_final_file(self):
        for name in ('model.hlo_module.pb', 'model.neff', 'wrapped_neff.hlo'):
            with self.subTest(name=name), self.fixture() as (spec, reviewed, control, sleep):
                directory = self.cache(spec)
                final = directory / name
                temporary = directory / (name + '-' + os.uname().nodename + '-12345')
                temporary.write_bytes(b'synthetic' * 700)
                with self.vanish_on_stat(temporary, lambda: temporary.replace(final)):
                    result = io.storage(spec)
                self.assertTrue(final.is_file()); self.assertFalse(temporary.exists())
                expected, inventory = io.storage_scan(spec)
                self.assertEqual(result, expected)
                self.assertEqual(result['allocated_bytes'], sum(row[6] * 512 for row in inventory))
                sleep.assert_called_once_with(.01)

    def test_sdk_cache_unlock_recounts_complete_tree(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            directory = self.cache(spec)
            lock = directory / 'model.hlo_module.pb.lock'; lock.touch()
            (directory / 'model.neff').write_bytes(b'synthetic' * 700)
            with self.vanish_on_stat(lock, lock.unlink): result = io.storage(spec)
            self.assertEqual(result, io.storage_scan(spec)[0])
            sleep.assert_called_once_with(.01)

    def test_bucket_replacement_does_not_reuse_partial_usage(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            directory = self.work(spec); victim = directory / 'model' / 'graph.hlo'
            victim.parent.mkdir(); victim.write_bytes(b'old synthetic')
            def reset():
                shutil.rmtree(directory); directory.mkdir()
                (directory / 'model.synthetic.neff').write_bytes(b'new synthetic' * 900)
            with self.vanish_on_stat(victim, reset): result = io.storage(spec)
            self.assertEqual(result, io.storage_scan(spec)[0])
            self.assertGreater(result['classes']['baseline_compiler_workdir'], 9000)
            sleep.assert_called_once_with(.01)

    def test_layout_directory_disappears_during_enumeration(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            directory = Path(spec['writer_cases']['baseline']['compiler_workdir']) / 'layout_opt'
            directory.mkdir(parents=True); original = Path.iterdir; changed = []
            def children(path):
                if path == directory and not changed:
                    changed.append(True); directory.rmdir(); directory.mkdir()
                    (directory / 'metaneff').write_bytes(b'synthetic')
                    raise FileNotFoundError('SDK directory reset during enumeration')
                return original(path)
            with patch.object(Path, 'iterdir', children): result = io.storage(spec)
            self.assertEqual(result, io.storage_scan(spec)[0]); sleep.assert_called_once_with(.01)

    def test_unknown_cache_missing_and_wrong_sdk_or_hostname_fail_immediately(self):
        for name in ('model.neff', 'arbitrary.lock', 'model.neff-wrong-host-12345',
                     'compile_flags.json', 'model.neff-' + os.uname().nodename + '-0'):
            with self.subTest(name=name), self.fixture() as (spec, reviewed, control, sleep):
                victim = self.cache(spec) / name; victim.write_bytes(b'synthetic')
                with self.vanish_on_stat(victim, victim.unlink):
                    with self.assertRaises(FileNotFoundError): io.storage(spec)
                sleep.assert_not_called()
        with self.fixture() as (spec, reviewed, control, sleep):
            wrong = Path(spec['writer_cases']['baseline']['compiler_cache']) / 'neuronxcc-other' / ('MODULE_' + 'a'*20 + '+' + 'b'*8)
            wrong.mkdir(parents=True); victim = wrong / 'model.hlo_module.pb.lock'; victim.touch()
            with self.vanish_on_stat(victim, victim.unlink):
                with self.assertRaises(FileNotFoundError): io.storage(spec)
            sleep.assert_not_called()

    def test_unknown_workdir_branch_and_other_tree_missing_fail(self):
        for key, relative in (('compiler_workdir', 'unexpected/file.bin'),
                              ('compiler_workdir', 'context_encoding_model/_tp0_bk6/graph.neff'),
                              ('artifact', 'model.neff'), ('model_path', 'model.neff')):
            with self.subTest(key=key, relative=relative), self.fixture() as (spec, reviewed, control, sleep):
                victim = Path(spec['writer_cases']['baseline'][key]) / relative
                victim.parent.mkdir(parents=True); victim.write_bytes(b'synthetic')
                with self.vanish_on_stat(victim, victim.unlink):
                    with self.assertRaises(FileNotFoundError): io.storage(spec)
                sleep.assert_not_called()

    def test_sealed_build_has_no_new_compiler_mutation_exception(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            root = Path(spec['output_root']) / 'baseline'; root.mkdir(parents=True)
            (root / 'build-immutable-identities.json').write_text('{}')
            lock = self.cache(spec) / 'model.hlo_module.pb.lock'; lock.touch()
            with self.vanish_on_stat(lock, lock.unlink):
                with self.assertRaises(FileNotFoundError): io.storage(spec)
            sleep.assert_not_called()

    def test_symlink_and_hardlink_in_compiler_trees_still_fail(self):
        for tree in ('compiler_cache', 'compiler_workdir'):
            with self.subTest(tree=tree), self.fixture() as (spec, reviewed, control, sleep):
                directory = self.cache(spec) if tree == 'compiler_cache' else self.work(spec)
                target = Path(spec['output_root']).parent / 'outside'; target.write_bytes(b'synthetic')
                entry = directory / 'model.neff'; entry.symlink_to(target)
                with self.assertRaises(ValueError): io.storage(spec)
                entry.unlink(); os.link(target, entry)
                with self.assertRaises(ValueError): io.storage(spec)
                sleep.assert_not_called()

    def test_each_cap_checked_again_after_topology_race(self):
        for field in ('single_file_bytes', 'allocated_bytes', 'entries'):
            with self.subTest(field=field), self.fixture() as (spec, reviewed, control, sleep):
                directory = self.work(spec); victim = directory / 'old'; victim.write_bytes(b'synthetic')
                def reset():
                    victim.unlink(); (directory / 'new').write_bytes(b'x' * 9000)
                    reviewed['classes']['baseline_compiler_workdir'][field] = 1
                with self.vanish_on_stat(victim, reset):
                    with self.assertRaisesRegex(ValueError, 'cap'): io.storage(spec)
                sleep.assert_called_once_with(.01)

    def test_live_log_and_regular_compiler_file_growth_converges(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            output = self.work(spec) / 'graph.neff'; output.write_bytes(b'synthetic')
            log = control / 'baseline-build' / 'phase.log'; log.parent.mkdir(parents=True)
            log.write_bytes(b'synthetic log')
            original = io.storage_scan; scans = []
            def scan(value):
                if not scans:
                    scans.append(None); raise io.CompilerTreeRace('injected prior SDK reset')
                for path in (output, log):
                    with path.open('ab') as stream: stream.write(b'x' * 5000)
                observed = original(value); scans.append(observed); return observed
            with patch.object(io, 'storage_scan', side_effect=scan): result = io.storage(spec)
            self.assertEqual(len(scans), 3); self.assertEqual(result, scans[-1][0])
            for kind in ('baseline_compiler_workdir', 'build_auxiliary'):
                self.assertGreater(scans[-1][0]['classes'][kind], scans[-2][0]['classes'][kind])
            sleep.assert_called_once_with(.01)

    def test_growth_past_cap_on_second_complete_scan_is_rejected(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            output = self.work(spec) / 'graph.neff'; output.write_bytes(b'synthetic')
            original = io.storage_scan; calls = []
            def scan(value):
                calls.append(True)
                if len(calls) == 1: raise io.CompilerTreeRace('injected prior SDK reset')
                if len(calls) == 3:
                    output.write_bytes(b'x' * 10000)
                    reviewed['classes']['baseline_compiler_workdir']['single_file_bytes'] = 8192
                return original(value)
            with patch.object(io, 'storage_scan', side_effect=scan):
                with self.assertRaisesRegex(ValueError, 'single.*cap'): io.storage(spec)
            self.assertEqual(len(calls), 3)

    def test_topology_churn_between_complete_scans_restarts_both(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            lock = self.cache(spec) / 'model.hlo_module.pb.lock'; lock.touch()
            original = io.storage_scan; calls = []
            def scan(value):
                calls.append(True)
                if len(calls) == 1: raise io.CompilerTreeRace('injected prior SDK reset')
                if len(calls) == 3: lock.unlink()
                return original(value)
            with patch.object(io, 'storage_scan', side_effect=scan): result = io.storage(spec)
            self.assertEqual(len(calls), 5); self.assertEqual(sleep.call_count, 2)
            self.assertEqual(result, original(spec)[0])

    def test_unknown_disappearance_between_complete_scans_is_not_swallowed(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            directory = Path(spec['writer_cases']['baseline']['artifact']); directory.mkdir()
            victim = directory / 'model.neff'; victim.write_bytes(b'synthetic')
            original = io.storage_scan; calls = []
            def scan(value):
                calls.append(True)
                if len(calls) == 1: raise io.CompilerTreeRace('injected prior SDK reset')
                if len(calls) == 3: victim.unlink()
                return original(value)
            with patch.object(io, 'storage_scan', side_effect=scan):
                with self.assertRaisesRegex(ValueError, 'unknown storage topology'): io.storage(spec)
            self.assertEqual(len(calls), 3); sleep.assert_called_once_with(.01)

    def test_unknown_inode_replacement_between_complete_scans_fails(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            directory = Path(spec['writer_cases']['baseline']['artifact']); directory.mkdir()
            victim = directory / 'model.neff'; victim.write_bytes(b'synthetic')
            original = io.storage_scan; calls = []
            def scan(value):
                calls.append(True)
                if len(calls) == 1: raise io.CompilerTreeRace('injected prior SDK reset')
                if len(calls) == 3:
                    replacement = directory / 'replacement'
                    replacement.write_bytes(b'synthetic'); replacement.replace(victim)
                return original(value)
            with patch.object(io, 'storage_scan', side_effect=scan):
                with self.assertRaisesRegex(ValueError, 'unknown storage topology'): io.storage(spec)
            self.assertEqual(len(calls), 3); sleep.assert_called_once_with(.01)

    def test_eight_attempt_bound_and_terminal_exact_inventory_preserved(self):
        with patch.object(io, 'storage_scan', side_effect=io.CompilerTreeRace('never settles')) as scan, \
                patch.object(io.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ValueError, 'finite storage retries'): io.storage({})
            self.assertEqual(scan.call_count, 8); self.assertEqual(sleep.call_count, 7)
        with patch.object(io, 'storage_scan', side_effect=[({'bytes': 1}, []), ({'bytes': 2}, [])]), \
                patch.object(io.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ValueError, 'terminal output inventory'): io.storage({}, terminal=True)
            sleep.assert_not_called()


class CompilerLinks(unittest.TestCase):
    def fixture(self):
        return CompilerStorage().fixture()

    def link(self, spec):
        unit = Path(spec['writer_cases']['baseline']['compiler_workdir']) / 'token_generation_model/_tp0_bk0/neuronxcc-fixture'
        path = unit / 'sg00/constant.21.npy'; path.parent.mkdir(parents=True)
        target = unit / path.name; target.write_bytes(b'x' * 208)
        path.symlink_to('../' + path.name)
        return path, target

    def test_exact_observed_form_counted_once_with_target_and_stable_terminal_scan(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            link, target = self.link(spec)
            with patch.object(io.os, 'read', side_effect=AssertionError('no target payload read')):
                result, inventory = io.storage_scan(spec)
                final = io.storage(spec, terminal=True)
            self.assertEqual(result, final); sleep.assert_not_called()
            self.assertEqual(result['allocated_bytes'], sum(row[6] * 512 for row in inventory))
            self.assertEqual(len([row for row in inventory if row[0] == str(target)]), 1)
            self.assertEqual(len([row for row in inventory if row[0] == str(link)]), 1)
            record = result['compiler_links'][0]
            self.assertEqual(record['target'], str(target)); self.assertEqual(record['payload_bytes_read'], 0)
            self.assertEqual(record['target_inventory'], next(row for row in inventory if row[0] == str(target)))

    def test_original_caps_apply_to_target_link_class_and_entries(self):
        for mode in ('target', 'link', 'class', 'entries'):
            with self.subTest(mode=mode), self.fixture() as (spec, reviewed, control, sleep):
                link, target = self.link(spec)
                if mode == 'target': reviewed['classes']['baseline_compiler_workdir']['single_file_bytes'] = 100
                elif mode == 'link':
                    target.write_bytes(b''); reviewed['classes']['baseline_compiler_workdir']['single_file_bytes'] = 10
                elif mode == 'class': reviewed['classes']['baseline_compiler_workdir']['allocated_bytes'] = 1
                else: reviewed['classes']['baseline_compiler_workdir']['entries'] = 1
                with self.assertRaises(ValueError): io.storage_scan(spec)

    def test_relative_target_must_be_unique_same_basename_parent(self):
        for mode in ('absolute', 'other_name', 'escape', 'same_directory', 'nested_parent', 'empty_component'):
            with self.subTest(mode=mode), self.fixture() as (spec, reviewed, control, sleep):
                link, target = self.link(spec); link.unlink()
                raw = {'absolute':str(target), 'other_name':'../constant.22.npy', 'escape':'../../other/constant.21.npy',
                       'same_directory':'constant.21.npy', 'nested_parent':'../sg00/../constant.21.npy',
                       'empty_component':'..//constant.21.npy'}[mode]
                link.symlink_to(raw)
                with self.assertRaisesRegex(ValueError, 'exact relative'): io.storage_scan(spec)

    def test_undeclared_scope_or_name_rejected(self):
        variants = [('token_generation_model','wrong_model'), ('_tp0_bk0','_tp1_bk0'),
                    ('_tp0_bk0','_tp0_bk6'), ('neuronxcc-fixture','unknown-fixture'),
                    ('sg00','sgLnk'), ('constant.21.npy','anything.npy')]
        for before, after in variants:
            with self.subTest(before=before,after=after), self.fixture() as (spec, reviewed, control, sleep):
                link, target = self.link(spec)
                changed = Path(str(link).replace(before,after)); link.unlink(); changed.parent.mkdir(parents=True,exist_ok=True)
                changed.symlink_to('../'+changed.name)
                with self.assertRaisesRegex(ValueError, 'observed bucket'): io.storage_scan(spec)

    def test_noncompiler_tree_symlink_remains_rejected(self):
        for key in ('artifact','compiler_cache'):
            with self.subTest(key=key), self.fixture() as (spec, reviewed, control, sleep):
                link, target = self.link(spec); link.unlink()
                bad = Path(spec['writer_cases']['baseline'][key]) / 'constant.21.npy'; bad.parent.mkdir(); bad.symlink_to(target)
                with self.assertRaisesRegex(ValueError, 'symlinks forbidden'): io.storage_scan(spec)

    def test_target_symlink_hardlink_directory_or_fifo_rejected(self):
        for mode in ('symlink', 'hardlink', 'directory', 'fifo'):
            with self.subTest(mode=mode), self.fixture() as (spec, reviewed, control, sleep):
                link, target = self.link(spec)
                if mode == 'hardlink': os.link(target,target.with_suffix('.other'))
                else:
                    target.unlink()
                    if mode == 'symlink': target.symlink_to(link)
                    elif mode == 'directory': target.mkdir()
                    else: os.mkfifo(target)
                with self.assertRaises(ValueError): io.compiler_constant_link(link,spec,2**20)

    def test_parent_directory_symlink_rejected_without_following(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            link,target=self.link(spec); parent=link.parent; moved=parent.with_name('moved')
            parent.rename(moved);parent.symlink_to(moved,target_is_directory=True)
            with self.assertRaises(OSError): io.compiler_constant_link(link,spec,2**20)

    def test_owner_and_device_negative_controls(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            link,target=self.link(spec)
            with patch.object(io.os,'getuid',return_value=os.getuid()+1):
                with self.assertRaisesRegex(ValueError,'owned ordinary'):io.compiler_constant_link(link,spec,2**20)
            real=io.os.stat
            def changed(path,*args,**kwargs):
                s=real(path,*args,**kwargs)
                if str(path)==target.name and kwargs.get('follow_symlinks') is False and s.st_mode & 0o170000 == 0o100000:
                    row={key:getattr(s,key) for key in dir(s) if key.startswith('st_')};row['st_dev']+=1
                    return types.SimpleNamespace(**row)
                return s
            with patch.object(io.os,'stat',side_effect=changed):
                with self.assertRaisesRegex(ValueError,'regular compiler target'):io.compiler_constant_link(link,spec,2**20)

    def test_link_changed_during_observation_never_yields_partial_usage(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            link,target=self.link(spec);original=io.os.readlink;changed=[]
            def swap(path,*args,**kwargs):
                raw=original(path,*args,**kwargs)
                if not changed:
                    changed.append(True);link.unlink();link.symlink_to('../constant.999.npy')
                return raw
            with patch.object(io.os,'readlink',side_effect=swap):
                with self.assertRaises(io.CompilerTreeRace):io.compiler_constant_link(link,spec,2**20)

    def test_target_omitted_from_full_inventory_rejected(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            link,target=self.link(spec);original=Path.iterdir
            def listing(path):return iter(p for p in original(path) if p!=target)
            with patch.object(Path,'iterdir',listing):
                with self.assertRaisesRegex(io.CompilerTreeRace,'complete accounted'):io.storage_scan(spec)

    def test_target_growth_between_target_inventory_and_link_recheck_retries_full_scan(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            link,target=self.link(spec);original=io.compiler_constant_link;count=[]
            def changed(*args,**kwargs):
                result=original(*args,**kwargs);count.append(True)
                if len(count)==1:target.write_bytes(b'y'*8192)
                return result
            with patch.object(io,'compiler_constant_link',side_effect=changed):result=io.storage(spec)
            self.assertEqual(result,io.storage_scan(spec)[0]);self.assertGreaterEqual(result['classes']['baseline_compiler_workdir'],8192)
            sleep.assert_called_once_with(.01)

    def test_dangling_target_cannot_settle_within_original_eight_retries(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            link,target=self.link(spec);target.unlink()
            with self.assertRaisesRegex(ValueError,'finite storage retries'):io.storage(spec)
            self.assertEqual(sleep.call_count,7)


class TerminalEvidence(unittest.TestCase):
    def test_original_failure_and_separate_storage_failure_preserved(self):
        original=RuntimeError('primary failure');scan=ValueError('storage failure')
        with patch.object(a,'storage',side_effect=scan):usage,error,failure=a.terminal_storage({},original)
        self.assertIsNone(usage);self.assertIs(failure,original)
        self.assertEqual(error,{'type':'ValueError','message':'storage failure'})

    def test_storage_failure_alone_changes_success_to_failure(self):
        scan=ValueError('storage proof failed')
        with patch.object(a,'storage',side_effect=scan):usage,error,failure=a.terminal_storage({},None)
        self.assertIs(failure,scan);self.assertIsNone(usage);self.assertIsNotNone(error)

    def test_successful_storage_retains_original_failure_and_original_usage(self):
        original=ValueError('other failure');usage={'entries':11}
        with patch.object(a,'storage',return_value=usage) as scan:
            result=a.terminal_storage({},original)
        self.assertEqual(result,(usage,None,original));scan.assert_called_once_with({},terminal=True)

    def test_mock_outer_publishes_terminal_before_release_and_reraises_original(self):
        with tempfile.TemporaryDirectory() as temp:
            spec=spec_fixture();spec['output_root']=str(Path(temp)/'output');spec['running_handoff_path']=str(Path(temp)/'running.json')
            control=Path(temp)/'control';names=c.units(spec['run_name']);failure=RuntimeError('original running failure')
            lock=types.SimpleNamespace(owner={'pid':7},fd=88,initial={},acquire=Mock(),check=Mock(),release=Mock())
            service=types.SimpleNamespace(unit=names['coordinator'],owner={'process':{'pid':9}},started=True,
                terminal={'cleanup_complete':True,'actual_exit_code':1},historical_terminal_reuse=None,
                start=Mock(side_effect=failure),cleanup=Mock())
            bundle=types.SimpleNamespace(spec=spec,names=names,resolve=Mock(),physical_pair=Mock(return_value={}),
                freshness=Mock(return_value={}),resolved={'plan_sha256':'a'*64},host={})
            process=types.SimpleNamespace(active_heavy=Mock(),empty_group=Mock(),GROUP_PREFIX='/mock/')
            writes=[];events=[]
            def once(path,value):
                writes.append((Path(path).name,copy.deepcopy(value)));events.append(Path(path).name)
                return {'path':str(path),'sha256':'b'*64}
            lock.release.side_effect=lambda drained:events.append(('release',drained))
            with patch.object(a,'prelock'),patch.object(a,'RetainedLock',return_value=lock),\
                 patch.object(a,'Bundle',return_value=bundle),patch.object(a,'control_root',return_value=control),\
                 patch.object(a,'resources'),patch.object(a,'process_api',return_value=process),\
                 patch.object(a,'fresh_package'),patch.object(a,'properties',return_value={'LoadState':'not-found','InvocationID':''}),\
                 patch.object(a,'once',side_effect=once),patch.object(a,'admit_shape'),\
                 patch.object(a,'coordinator_service',return_value=service),patch.object(a,'declared_services',return_value=[service]),\
                 patch.object(a,'storage',side_effect=ValueError('secondary terminal storage')),\
                 patch.object(a.signal,'signal'),patch('fcntl.flock',side_effect=AssertionError('real lock forbidden')):
                with self.assertRaises(RuntimeError) as error:a.outer({'path':'/mock/spec','sha256':'c'*64},{'path':'/mock/review','sha256':'d'*64})
            self.assertIs(error.exception,failure)
            terminal=dict(writes)['outer-terminal.json']
            self.assertEqual(terminal['failure'],{'type':'RuntimeError','message':'original running failure'})
            self.assertEqual(terminal['storage_error'],{'type':'ValueError','message':'secondary terminal storage'})
            self.assertIsNone(terminal['storage']);self.assertTrue(terminal['failed']);self.assertTrue(terminal['whole_owned_scope_drained'])
            self.assertEqual(terminal['complete_owned_unit_terminals'],[service.terminal]);service.cleanup.assert_called_once()
            self.assertLess(events.index('outer-terminal.json'),events.index(('release',True)))


class SDKAssets(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with CompilerStorage().fixture() as (spec, reviewed, control, sleep):
            sdk = control.parent/'sdk-fixture'; sdk.mkdir()
            assets = {}
            for name in ca.NAMES:
                path = sdk/name; raw = (name+'-synthetic-only').encode(); path.write_bytes(raw)
                assets[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                    identity=dict(zip(ca.ATTRIBUTES, ca.attrs(path.lstat()))))
            with patch.object(ca, 'ASSET_ROOT', sdk), patch.object(ca, 'manifest', return_value={'assets':assets}):
                yield spec, reviewed, control, sleep, sdk, assets

    def link(self, spec, sdk, name='act_info.json', bucket=0, sg='sg00'):
        path = Path(spec['writer_cases']['baseline']['compiler_workdir'])/f'token_generation_model/_tp0_bk{bucket}/neuronxcc-fixture/{sg}'/name
        path.parent.mkdir(parents=True, exist_ok=True); path.symlink_to(sdk/name)
        return path

    def test_all16_and_repeated_targets_fully_counted_once(self):
        with self.fixture() as (spec, reviewed, control, sleep, sdk, assets):
            for name in ca.NAMES:
                self.link(spec,sdk,name); self.link(spec,sdk,name,bucket=5)
            result, inventory = io.storage_scan(spec)
            self.assertEqual(len(result['compiler_links']),32)
            self.assertEqual(result['existing_sdk_targets'],sorted(assets))
            charged = sum((sdk/name).stat().st_blocks*512 for name in ca.NAMES)
            self.assertEqual(result['existing_sdk_allocated_bytes'],charged)
            self.assertEqual(result['allocated_bytes'],sum(row[6]*512 for row in inventory))
            self.assertEqual(result['output_allocated_bytes'],result['allocated_bytes']-charged)
            for target in assets:self.assertEqual(sum(row[0]==target for row in inventory),1)
            self.assertEqual(io.storage(spec,terminal=True),result);sleep.assert_not_called()

    def test_constant_branch_remains_metadata_only(self):
        with self.fixture() as (spec, reviewed, control, sleep, sdk, assets):
            CompilerLinks().link(spec)
            with patch.object(ca,'asset',side_effect=AssertionError('constant must not read SDK')),\
                 patch.object(io.os,'read',side_effect=AssertionError('no constant payload read')):
                result,_=io.storage_scan(spec)
            self.assertEqual(result['existing_sdk_allocated_bytes'],0)
            self.assertEqual(result['compiler_links'][0]['payload_bytes_read'],0)

    def test_exact_target_only_rejects_relative_traversal_other_asset_and_unlisted_name(self):
        with self.fixture() as (spec, reviewed, control, sleep, sdk, assets):
            path=self.link(spec,sdk)
            for target in ('../act_info.json',str(sdk/'silu_and_others.json'),str(sdk/'../sdk-fixture/act_info.json'),'/tmp/act_info.json'):
                path.unlink();path.symlink_to(target)
                with self.subTest(target=target),self.assertRaisesRegex(ValueError,'exact installed'):io.storage_scan(spec)
            path.unlink();unknown=self.link(spec,sdk,'unexpected.json')
            with self.assertRaisesRegex(ValueError,'observed bucket'):io.storage_scan(spec)

    def test_wrong_scope_and_external_links_in_other_trees_rejected(self):
        with self.fixture() as (spec, reviewed, control, sleep, sdk, assets):
            path=self.link(spec,sdk,bucket=6)
            with self.assertRaisesRegex(ValueError,'observed bucket'):io.storage_scan(spec)
            path.unlink()
            for kind in ('artifact','compiler_cache'):
                parent=Path(spec['writer_cases']['baseline'][kind]);parent.mkdir();bad=parent/'act_info.json';bad.symlink_to(sdk/'act_info.json')
                with self.assertRaisesRegex(ValueError,'symlinks forbidden'):io.storage_scan(spec)
                bad.unlink()

    def test_target_symlink_hardlink_directory_fifo_missing_rejected(self):
        for kind in ('symlink','hardlink','directory','fifo','missing'):
            with self.subTest(kind=kind),self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
                path=self.link(spec,sdk);target=sdk/path.name
                if kind=='hardlink':os.link(target,sdk/'another')
                else:
                    target.unlink()
                    if kind=='symlink':target.symlink_to(sdk/'silu_and_others.json')
                    elif kind=='directory':target.mkdir()
                    elif kind=='fifo':os.mkfifo(target)
                with self.assertRaises(ValueError):io.storage_scan(spec)
                sleep.assert_not_called()

    def test_any_target_ancestry_symlink_refused(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            self.link(spec,sdk);renamed=sdk.with_name('moved-sdk');sdk.rename(renamed);sdk.symlink_to(renamed)
            with self.assertRaises(OSError):io.storage_scan(spec)

    def test_link_ancestry_symlink_refused(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            path=self.link(spec,sdk);original=path.parent;new=original.with_name('sg01');original.rename(new);original.symlink_to(new)
            with self.assertRaises(OSError):ca.compiler_sdk_link(path,spec,2**20,assets)

    def test_every_compiler_ancestor_requires_owner_and_same_device(self):
        for kind in ('owner', 'device'):
            with self.subTest(kind=kind), self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
                path=self.link(spec,sdk); ancestor=path.parents[2]; selected=ancestor.stat().st_ino
                original=ca.os.fstat
                def changed(fd):
                    value=original(fd)
                    if value.st_ino != selected:return value
                    fields={key:getattr(value,key) for key in ca.ATTRIBUTES}
                    fields['st_uid' if kind=='owner' else 'st_dev'] += 1
                    return SimpleNamespace(**fields)
                with patch.object(ca.os,'fstat',side_effect=changed),self.assertRaisesRegex(ValueError,'owned compiler ancestry'):
                    ca.compiler_sdk_link(path,spec,2**20,assets)

    def test_directory_replaced_during_asset_read_rejected_before_success(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            path=self.link(spec,sdk); original=ca.asset; done=[]
            def racing(*args,**kwargs):
                result=original(*args,**kwargs)
                if not done:
                    done.append(True);path.parent.rename(path.parent.with_name('moved'));path.parent.mkdir()
                return result
            with patch.object(ca,'asset',side_effect=racing),self.assertRaisesRegex(ValueError,'directory edge changed'):
                ca.compiler_sdk_link(path,spec,2**20,assets)

    def test_wrong_hash_despite_matching_current_stat_rejected(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            self.link(spec,sdk);target=sdk/'act_info.json';target.write_bytes(b'y'*target.stat().st_size)
            assets[str(target)]['identity']=dict(zip(ca.ATTRIBUTES,ca.attrs(target.stat())))
            with self.assertRaisesRegex(ValueError,'content/hash'):io.storage_scan(spec)

    def test_same_content_replacement_identity_rejected(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            self.link(spec,sdk);target=sdk/'act_info.json';raw=target.read_bytes();other=sdk/'replacement';other.write_bytes(raw);other.replace(target)
            with self.assertRaisesRegex(ValueError,'identity changed'):io.storage_scan(spec)

    def test_target_changed_during_read_never_returns_usage(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            self.link(spec,sdk);target=sdk/'act_info.json';read=ca.os.read;done=[]
            def racing(*args):
                result=read(*args)
                if result and not done:done.append(True);target.write_bytes(b'z'*target.stat().st_size)
                return result
            with patch.object(ca.os,'read',side_effect=racing),self.assertRaisesRegex(ValueError,'content/hash'):io.storage_scan(spec)

    def test_link_retarget_during_content_read_never_returns_usage(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            path=self.link(spec,sdk);original=ca.asset;done=[]
            def racing(*args,**kwargs):
                result=original(*args,**kwargs)
                if not done:done.append(True);path.unlink();path.symlink_to(sdk/'silu_and_others.json')
                return result
            with patch.object(ca,'asset',side_effect=racing),self.assertRaises(io.CompilerTreeRace):io.storage_scan(spec)

    def test_last_target_is_rechecked_after_complete_inventory(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            target=sdk/ca.NAMES[-1];self.link(spec,sdk,target.name);original=ca.compiler_sdk_link;calls=[]
            def racing(*args,**kwargs):
                result=original(*args,**kwargs);calls.append(True)
                if len(calls)==2:target.write_bytes(b'z'*target.stat().st_size)
                return result
            with patch.object(ca,'compiler_sdk_link',side_effect=racing),self.assertRaisesRegex(ValueError,'identity changed'):io.storage_scan(spec)

    def test_original_class_entry_and_single_file_caps_apply(self):
        for kind in ('class','entries','single'):
            with self.subTest(kind=kind),self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
                self.link(spec,sdk)
                key={'class':'allocated_bytes','entries':'entries','single':'single_file_bytes'}[kind]
                reviewed['classes']['baseline_compiler_workdir'][key]=1
                with self.assertRaises(ValueError):io.storage_scan(spec)

    def test_existing_root_assets_do_not_reduce_nvme_reservation(self):
        with self.fixture() as (spec,reviewed,control,sleep,sdk,assets):
            self.link(spec,sdk);usage,_=io.storage_scan(spec)
            review=copy.deepcopy(reviewed);review['headroom_bytes']={'root':17,'nvme':19}
            result=budget.remaining_reserve(spec,usage,review)
            self.assertEqual(result['root'],17)
            self.assertEqual(result['nvme'],max(0,c.LIMITS['total_allocated_output_bytes']+sum(v['allocated_bytes'] for v in review['classes'].values())-usage['output_allocated_bytes'])+19)


class ManifestAndPreservation(unittest.TestCase):
    def test_real_manifest_is_metadata_only_fixed16(self):
        raw=(HERE/'compiler_assets.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),ca.MANIFEST_SHA256)
        spec={'source_sha256':{str(HERE/'compiler_assets.json'):ca.MANIFEST_SHA256}}
        with patch.object(io,'raw',return_value=raw):value=ca.manifest(spec)
        self.assertEqual(len(value['assets']),16)
        self.assertEqual(sum(v['bytes'] for v in value['assets'].values()),370753)
        self.assertEqual(sum(v['identity']['st_blocks']*512 for v in value['assets'].values()),393216)
        for bad in ({'source_sha256':{}},{'source_sha256':{str(HERE/'compiler_assets.json'):'0'*64}}):
            with self.assertRaises(ValueError),patch.object(io,'raw',side_effect=AssertionError('must fail before read')):ca.manifest(bad)

    def test_modified_manifest_refused(self):
        spec={'source_sha256':{str(HERE/'compiler_assets.json'):ca.MANIFEST_SHA256}}
        with patch.object(io,'raw',return_value=b'{}'),self.assertRaisesRegex(ValueError,'manifest changed'):ca.manifest(spec)

    def test_all16_checked_prelock_without_resources_or_lock_on_failure(self):
        spec={}
        with patch.object(qualification,'check_geometry_qualification'),patch.object(budget,'validate_spec'),patch.object(io,'read',return_value=spec),\
             patch.object(budget,'reviewed_budget'),patch.object(budget,'unchanged_inputs'),patch.object(budget,'fresh_paths'),\
             patch.object(ca,'verify_all',side_effect=ValueError('asset mismatch')) as check,\
             patch.object(io,'resources',side_effect=AssertionError('must not reach resources')):
            with self.assertRaisesRegex(ValueError,'asset mismatch'):budget.prelock({}, {})
            check.assert_called_once_with(spec)


class PairedStorage(unittest.TestCase):
    def link(self, spec, arm, name, target, sg='sg00'):
        path = Path(spec['writer_cases'][arm]['compiler_workdir']) / 'token_generation_model/_tp0_bk0/neuronxcc-fixture' / sg / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
        return path

    def constant(self, spec, arm):
        path = self.link(spec, arm, 'constant.21.npy', '../constant.21.npy')
        target = path.parent.parent / path.name
        target.write_bytes(b'synthetic constant')
        return path, target

    def test_all16_shared_targets_once_across_both_arms_and_repeated_links(self):
        with SDKAssets().fixture() as (spec, reviewed, control, sleep, sdk, assets):
            for arm in ('baseline', 'candidate'):
                for name in ca.NAMES:
                    for sg in ('sg00', 'sg01'):
                        self.link(spec, arm, name, sdk / name, sg)
            usage, inventory = io.storage_scan(spec)
            self.assertEqual(len(usage['compiler_links']), 64)
            self.assertEqual({row['writer_arm'] for row in usage['compiler_links']}, {'baseline', 'candidate'})
            self.assertEqual(usage, io.storage(spec, terminal=True))
            charged = sum((sdk / name).stat().st_blocks * 512 for name in ca.NAMES)
            self.assertEqual(usage['existing_sdk_allocated_bytes'], charged)
            self.assertEqual(usage['allocated_bytes'], sum(row[6] * 512 for row in inventory))
            for target in assets:
                self.assertEqual(sum(row[0] == target for row in inventory), 1)
            for arm in ('baseline', 'candidate'):
                root = Path(spec['writer_cases'][arm]['compiler_workdir'])
                own = sum(row[6] * 512 for row in inventory if Path(row[0]) == root or root in Path(row[0]).parents)
                self.assertEqual(usage['classes'][arm + '_compiler_workdir'], own + (charged if arm == 'baseline' else 0))
            sleep.assert_not_called()

    def test_candidate_only_sdk_target_is_charged_to_candidate(self):
        with SDKAssets().fixture() as (spec, reviewed, control, sleep, sdk, assets):
            self.link(spec, 'candidate', ca.NAMES[0], sdk / ca.NAMES[0])
            usage, inventory = io.storage_scan(spec)
            self.assertEqual(usage['classes']['baseline_compiler_workdir'], 0)
            self.assertEqual(usage['classes']['candidate_compiler_workdir'], usage['allocated_bytes'])
            self.assertGreater(usage['existing_sdk_allocated_bytes'], 0)

    def test_both_constants_have_distinct_targets_and_never_read_payloads(self):
        with CompilerStorage().fixture() as (spec, reviewed, control, sleep):
            targets = [self.constant(spec, arm)[1] for arm in ('baseline', 'candidate')]
            with patch.object(io.os, 'read', side_effect=AssertionError('no constant payload')):
                usage, inventory = io.storage_scan(spec)
                self.assertEqual(usage, io.storage(spec, terminal=True))
            self.assertEqual({row['writer_arm'] for row in usage['compiler_links']}, {'baseline', 'candidate'})
            for target in targets:
                self.assertEqual(sum(row[0] == str(target) for row in inventory), 1)
            self.assertEqual(usage['existing_sdk_allocated_bytes'], 0)

    def test_cross_arm_constant_targets_rejected_in_both_directions(self):
        for arm, other in (('baseline', 'candidate'), ('candidate', 'baseline')):
            with self.subTest(arm=arm), CompilerStorage().fixture() as (spec, reviewed, control, sleep):
                link, _ = self.constant(spec, arm)
                _, target = self.constant(spec, other)
                link.unlink(); link.symlink_to(os.path.relpath(target, link.parent))
                with self.assertRaisesRegex(ValueError, 'exact relative'):
                    io.storage_scan(spec)

    def test_each_arm_still_enforces_all_caps_for_shared_assets(self):
        for arm in ('baseline', 'candidate'):
            for cap in ('single_file_bytes', 'allocated_bytes', 'entries'):
                with self.subTest(arm=arm, cap=cap), SDKAssets().fixture() as (spec, reviewed, control, sleep, sdk, assets):
                    for selected in ('baseline', 'candidate'):
                        self.link(spec, selected, ca.NAMES[0], sdk / ca.NAMES[0])
                    reviewed['classes'][arm + '_compiler_workdir'][cap] = 1
                    with self.assertRaises(ValueError): io.storage_scan(spec)

    def test_each_arm_uses_only_its_own_build_seal(self):
        for arm, other in (('baseline', 'candidate'), ('candidate', 'baseline')):
            with self.subTest(arm=arm), CompilerStorage().fixture() as (spec, reviewed, control, sleep):
                link, target = self.constant(spec, arm)
                target.unlink()
                seal = Path(spec['output_root']) / other / 'build-immutable-identities.json'
                seal.parent.mkdir(parents=True); seal.write_text('{}')
                with self.assertRaises(io.CompilerTreeRace): io.compiler_constant_link(link, spec, 2**20)
                own_seal = seal.parent.parent / arm / seal.name
                own_seal.parent.mkdir(); own_seal.write_text('{}')
                with self.assertRaises(FileNotFoundError): io.compiler_constant_link(link, spec, 2**20)

    def test_unowned_or_ambiguous_arm_scope_rejected(self):
        with CompilerStorage().fixture() as (spec, reviewed, control, sleep):
            link, _ = self.constant(spec, 'baseline')
            with self.assertRaises(ValueError): io.compiler_workdir_owner(control / 'unexpected', spec)
            spec['writer_cases']['candidate']['compiler_workdir'] = spec['writer_cases']['baseline']['compiler_workdir']
            with self.assertRaisesRegex(ValueError, 'exactly one writer arm'):
                io.compiler_constant_link(link, spec, 2**20)

    def test_prelock_checks_all16_before_resources(self):
        with SDKAssets().fixture() as (spec, reviewed, control, sleep, sdk, assets):
            checked = []
            original = ca.asset
            def observe(path, row):
                checked.append(path)
                return original(path, row)
            def resources(*args, **kwargs):
                self.assertEqual(set(checked), set(assets))
            with patch.object(io, 'read', return_value=spec), patch.object(b, 'validate_spec'), \
                 patch.object(qualification, 'check_geometry_qualification'), patch.object(b, 'fresh_paths'), \
                 patch.object(ca, 'asset', side_effect=observe), patch.object(io, 'resources', side_effect=resources) as observation:
                self.assertIs(b.prelock({}, {}), spec)
                observation.assert_called_once_with(spec, reserve=True)


class SuccessorPreservation(unittest.TestCase):
    def source(self, name, directory='qwen35_writer_pair_serial_runtime_v1'):
        return (Q / directory / name).read_text().replace('qwen35_writer_pair_serial_runtime_v1', HERE.name)

    def nodes(self, text):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    def test_other_production_and_all_science_remain_exact(self):
        changed = {'io.py', 'adapter.py', 'budget.py', 'contract.py', 'compiler_assets.py', 'compiler_assets.json'}
        for name in c.RUNTIME_FILES:
            if name not in changed: self.assertEqual((HERE / name).read_text(), self.source(name), name)
        note = json.loads((Q / 'qwen35-writer-pair-static-readiness-20261113-v1.json').read_text())
        count = 0
        for delivery in note['frozen_source_verification']:
            for row in delivery['members']:
                raw = Path(row['path']).read_bytes()
                self.assertEqual(len(raw), row['bytes'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row['sha256'])
                count += 1
        self.assertEqual(count, 52)

    def test_every_unchanged_function_keeps_its_original_ast(self):
        for name, changed in (('io.py', {'storage_scan'}), ('budget.py', {'prelock', 'remaining_reserve'}),
                              ('adapter.py', {'outer'}), ('contract.py', set())):
            before, after = self.nodes(self.source(name)), self.nodes((HERE / name).read_text())
            for key in before:
                if key not in changed: self.assertEqual(before[key], after[key], name + ':' + key)

    def test_contract_only_adds_typed_asset_source_members(self):
        expected = self.source('contract.py').replace('"source_text.py", "qualification.py")',
            '"source_text.py", "qualification.py", "compiler_assets.py", "compiler_assets.json")')
        self.assertEqual((HERE / 'contract.py').read_text(), expected)
        self.assertEqual((HERE / 'compiler_assets.json').read_bytes(),
            (Q / 'qwen35_package_serial_runtime_v8/compiler_assets.json').read_bytes())

    def test_terminal_transplant_is_exact_and_outer_only_uses_it(self):
        accepted = self.nodes((Q / 'qwen35_package_serial_runtime_v8/adapter.py').read_text())
        current = self.nodes((HERE / 'adapter.py').read_text())
        self.assertEqual(current['terminal_storage'], accepted['terminal_storage'])
        text = (HERE / 'adapter.py').read_text().replace(
            '                terminal_usage, storage_error, failed = terminal_storage(bundle.spec, failed)\n', '').replace(
            '"storage": terminal_usage, "storage_error": storage_error,', '"storage": storage(bundle.spec, terminal=True),')
        self.assertEqual(self.nodes(text)['outer'], self.nodes(self.source('adapter.py'))['outer'])

    def test_28_stages_r10_transport_service_and_final_unqualified_preserved(self):
        from qwen35_writer_pair_serial_runtime_v2 import wiring
        self.assertEqual(len(wiring.STAGES), 28)
        for name in ('wiring.py', 'service.py', 'history.py', 'source_text.py', 'final_audit.py', 'qualification.py'):
            self.assertEqual((HERE / name).read_text(), self.source(name))
        self.assertIn('only explicitly root-requested API stop is accepted', (HERE / 'service.py').read_text())
        self.assertIn('HOST_SOURCE_ENVELOPE', (HERE / 'io.py').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)

