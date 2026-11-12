"""SOURCE/STDLIB MOCK only; synthetic compiler files and mocked outer flow."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from test_source import HERE, Q, c, io, a, Storage, spec_fixture
from test_compiler_storage import CompilerStorage


class CompilerLinks(unittest.TestCase):
    def fixture(self):
        return CompilerStorage().fixture()

    def link(self, spec):
        unit = Path(spec['case']['compiler_workdir']) / 'token_generation_model/_tp0_bk0/neuronxcc-fixture'
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
                if mode == 'target': reviewed['classes']['compiler_workdir']['single_file_bytes'] = 100
                elif mode == 'link':
                    target.write_bytes(b''); reviewed['classes']['compiler_workdir']['single_file_bytes'] = 10
                elif mode == 'class': reviewed['classes']['compiler_workdir']['allocated_bytes'] = 1
                else: reviewed['classes']['compiler_workdir']['entries'] = 1
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
                bad = Path(spec['case'][key]) / 'constant.21.npy'; bad.parent.mkdir(); bad.symlink_to(target)
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
            self.assertEqual(result,io.storage_scan(spec)[0]);self.assertGreaterEqual(result['classes']['compiler_workdir'],8192)
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


class Preservation(unittest.TestCase):
    def test_existing_case_contract_accepts_distinct_new_r5_paths_without_science_change(self):
        from qwen35_package_full_validation_v2.runtime import validate_case
        original=spec_fixture()['case'];case=copy.deepcopy(original)
        case.update(model_path='/mnt/nvme/models/qwen35-package-full-v2-view-20261112-r5',
            artifact='/mnt/nvme/neuron-artifacts/qwen35-package-full-v2-20261112-r5',
            compiler_cache='/mnt/nvme/neuron-work/qwen35-package-full-v2-cache-20261112-r5',
            compiler_workdir='/mnt/nvme/neuron-work/qwen35-package-full-v2-compiler-20261112-r5')
        with patch.object(Path,'is_symlink',return_value=False):self.assertEqual(validate_case(case),case)
        self.assertEqual({k:v for k,v in case.items() if k not in ('model_path','artifact','compiler_cache','compiler_workdir')},
                         {k:v for k,v in original.items() if k not in ('model_path','artifact','compiler_cache','compiler_workdir')})

    def test_service_and_lock_primitives_and_all_other_production_unchanged(self):
        old=Q/'qwen35_package_serial_runtime_v6'
        for name in c.RUNTIME_FILES:
            if name in ('io.py','adapter.py','run.py'):continue
            self.assertEqual((HERE/name).read_bytes(),(old/name).read_bytes(),name)
        def nodes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        old_io,new_io=nodes(old/'io.py'),nodes(HERE/'io.py')
        for name in old_io:
            if name!='storage_scan':self.assertEqual(old_io[name],new_io[name],name)
        old_a,new_a=nodes(old/'adapter.py'),nodes(HERE/'adapter.py')
        for name in old_a:
            if name!='outer':self.assertEqual(old_a[name],new_a[name],name)

    def test_original_non_api_stop_remains_failure_and_prior_failure_not_promoted(self):
        text=(HERE/'service.py').read_text()
        self.assertIn('only explicitly root-requested API stop is accepted',text)
        self.assertIn('require(api and request ==',text)
        self.assertEqual(text,(Q/'qwen35_package_serial_runtime_v6/service.py').read_text())

    def test_frozen_v1_through_v6_all_members_unchanged(self):
        pins=json.loads((HERE/'predecessor_pins.json').read_text())['files']
        for path,row in pins.items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),row['sha256'],path)


if __name__=='__main__':unittest.main(verbosity=2)
