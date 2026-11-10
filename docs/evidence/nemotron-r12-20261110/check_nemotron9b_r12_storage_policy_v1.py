"""Small stdlib controls for the successor; no native or real payload access."""
import ast
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'research')]
import nemotron9b_r12_storage_policy_v1 as space
import nemotron9b_r12_completion_v1 as complete


class Storage(unittest.TestCase):
    def test_original_full_nvme_reservation_cannot_be_reduced(self):
        values=dict(root=space.ROOT_FLOOR+space.ROOT_START_MARGIN,nvme=49*2**30)
        proof=space.check(values,0,0,True,True)
        self.assertEqual(proof['nvme_required_bytes'],49*2**30)
        with self.assertRaisesRegex(RuntimeError,'nvme'):
            space.check(dict(values,nvme=49*2**30-1),0,0,True,True)

    def test_root_launch_headroom_is_additional_and_runtime_floor_unchanged(self):
        values=dict(root=space.ROOT_FLOOR+space.ROOT_START_MARGIN-1,nvme=49*2**30)
        with self.assertRaisesRegex(RuntimeError,'fresh-admission'):
            space.check(values,0,0,True,True)
        self.assertEqual(space.check(values,0,0,True,False)['root_required_bytes'],space.ROOT_FLOOR)
        with self.assertRaisesRegex(RuntimeError,'runtime'):
            space.check(dict(values,root=space.ROOT_FLOOR-1),0,0,True,False)

    def test_failure_records_actual_bytes_and_deficit(self):
        with self.assertRaises(RuntimeError)as found:space.require_floor(123,456,'root','runtime')
        message=str(found.exception)
        for value in ('"available_bytes": 123','"required_bytes": 456','"deficit_bytes": 333'):
            self.assertIn(value,message)

    def test_output_growth_deducts_only_actual_admitted_remaining_reservation(self):
        proof=space.check(dict(root=space.ROOT_FLOOR,nvme=47*2**30),2**29,3*2**29,True,False)
        self.assertEqual(proof['nvme_required_bytes'],47*2**30)
        with self.assertRaisesRegex(RuntimeError,'caps'):
            space.check(dict(root=space.ROOT_FLOOR,nvme=49*2**30),2**30+1,0,True,False)

    def test_timed_check_uses_exact_original_floor_and_no_admission_margin(self):
        space.require_floor(space.ROOT_FLOOR,space.ROOT_FLOOR,'root','timed-runtime')
        with self.assertRaisesRegex(RuntimeError,'timed-runtime'):
            space.require_floor(space.ROOT_FLOOR-1,space.ROOT_FLOOR,'root','timed-runtime')


class Fixture:
    def __init__(self):
        self.rows={};self.lock=dict(device=1,inode=2)
        def ref(name,value):
            path='/metadata/'+name;self.rows[path]=value;return dict(path=path,sha256='a'*64)
        self.ref=ref
        def owner(pid):return dict(pid=pid,start_ticks=str(pid),argv=['python'],cgroup='0::/session',ppid=1)
        self.owner=owner
        result_ref=ref('storage-result',dict(complete=True,owner=owner(10)))
        plan_ref=ref('storage-plan',dict(serial_lock_identity=self.lock,canonical_sha256={'state':'b'*64},source_sha256={}))
        review_ref=ref('storage-review',dict(passed=True,blockers=[],plan=plan_ref,source_sha256={}))
        self.rows[result_ref['path']].update(plan=plan_ref,review=review_ref)
        audit_ref=ref('storage-audit',dict(complete=True,actual_exit=complete.STORAGE_EXIT,owner=owner(20),
            lock_identity=self.lock,all_original_bytes_reread_after_maintenance_exit=True,canonical_unchanged=True,
            original_bytes=805306368,compressed_bytes=95403896,
            files=[dict(full_decompression_verified=True,original_absent=True)for _ in range(6)]))
        for target,pid,result,flag in [(complete.STORAGE_EXIT,10,result_ref,'actual_exit_observed'),
                                    (complete.STORAGE_AUDIT_EXIT,20,audit_ref,'actual_tool_exit_observed')]:
            self.rows[target['path']]=dict(reviewer='/root',owner=owner(pid),result=result,lock_identity=self.lock,
                exact_owner_gone=True,serial_lock_reacquired=True,actual_exit_code=0,tool_chunk='abcdef',
                actual_execution_handle=dict(kind='tool_session',session_id=pid),**{flag:True})
        model_spec=ref('model-spec',dict(nemotron_completion=dict(actual_exit=complete.R11_EXIT,failed_release_exit=complete.R11_RELEASE)))
        for key in ('completed_layer1_data','completed_own_history','completed_cross_history','completed_factorial','completed_native_cache'):
            self.rows[model_spec['path']][key]={}
        admission=ref('admission',dict(predecessors_checked=True,complete_history_services=1255,complete_history_processes=806))
        drains=ref('drains',dict(complete=True,services=1255,processes=806))
        model=ref('model-exit',dict(spec=model_spec,outputs=dict(admission=admission,drains=drains)))
        self.rows[complete.HANDOFF['path']]=dict(live_tool_sessions=[],active_heavy_units=[],
            canonical_sha256={'state':'b'*64},storage_maintenance=dict(actual_exit=complete.STORAGE_EXIT,
            independent_audit_exit=complete.STORAGE_AUDIT_EXIT),official_history=dict(actual_exit=model),
            current_history_factorial=dict(actual_exit=complete.CURRENT_EXIT,independent_audit_exit=complete.CURRENT_AUDIT_EXIT))
        self.previous=dict(history=dict(rows=[{}for _ in range(1255)],processes=[{}for _ in range(806)],
            nemotron_r11_outcome=dict(failed_release=True,producer_qualified=False),canonical_sha256={'state':'b'*64}),source_sha256={})

    def run(self):
        from contextlib import ExitStack
        with ExitStack()as stack:
            stack.enter_context(mock.patch.object(complete.r11,'metadata_completed',return_value=self.previous))
            stack.enter_context(mock.patch.object(complete.current_history,'current',return_value=dict(owners=[])))
            for module in (complete.layer1,complete.own,complete.crossed,complete.factorial,complete.native):
                stack.enter_context(mock.patch.object(module,'completed',return_value=dict(lock_identity=self.lock)))
            return complete.metadata(lambda ref:copy.deepcopy(self.rows[ref['path']]))


class Completion(unittest.TestCase):
    def test_newer_handoff_blocks_fresh_preparation_and_launch(self):
        with mock.patch.object(complete,'Q')as q:
            old=mock.Mock();old.stat.return_value.st_mtime_ns=1
            newer=mock.Mock();newer.stat.return_value.st_mtime_ns=2
            q.glob.return_value=[old,newer]
            with self.assertRaisesRegex(RuntimeError,'newest completed'):complete.latest_terminal()

    def test_runtime_physical_enrollment_requires_actual_original_lock_before_metadata(self):
        low=mock.Mock();low.lock_proof.side_effect=RuntimeError('original lock not owned')
        with mock.patch.object(complete,'validate_spec'),mock.patch.object(complete,'metadata')as read_all:
            with self.assertRaisesRegex(RuntimeError,'original lock'):
                complete.physical(dict(units=dict(controller='trn1-current')),
                    lambda ref:dict(lock_identity=dict(device=1,inode=2)),low)
            read_all.assert_not_called()

    def test_valid_terminal_bindings_keep_full_r11_failed_history(self):
        f=Fixture();done=f.run();self.assertEqual(len(done['previous']['history']['processes']),806)
        self.assertEqual(done['lock_identity'],f.lock)

    def test_running_latest_handoff_rejected(self):
        f=Fixture();f.rows[complete.HANDOFF['path']]['live_tool_sessions']=[999]
        with self.assertRaisesRegex(RuntimeError,'terminal handoff'):f.run()

    def test_wrong_storage_terminal_or_auditor_exit_rejected(self):
        f=Fixture();f.rows[complete.HANDOFF['path']]['storage_maintenance']['actual_exit']=dict(path='/wrong',sha256='f'*64)
        with self.assertRaisesRegex(RuntimeError,'completed storage'):f.run()
        f=Fixture();f.rows[complete.STORAGE_AUDIT_EXIT['path']]['actual_exit_code']=1
        with self.assertRaisesRegex(RuntimeError,'successful tool exit'):f.run()

    def test_auditor_owner_or_lock_mismatch_rejected(self):
        for key,value in [('owner',dict(pid=999)),('lock_identity',dict(device=1,inode=999))]:
            f=Fixture();f.rows['/metadata/storage-audit'][key]=value
            with self.assertRaisesRegex(RuntimeError,'Independent complete'):f.run()

    def test_partial_archive_reread_and_missing_original_rejected(self):
        for key in ('full_decompression_verified','original_absent'):
            f=Fixture();f.rows['/metadata/storage-audit']['files'][0][key]=False
            with self.assertRaisesRegex(RuntimeError,'six exact originals'):f.run()

    def test_failed_r11_cannot_be_promoted_or_owner_list_shortened(self):
        f=Fixture();f.previous['history']['nemotron_r11_outcome']['producer_qualified']=True
        with self.assertRaisesRegex(RuntimeError,'Original r11 failure'):f.run()
        f=Fixture();f.previous['history']['processes'].pop()
        with self.assertRaisesRegex(RuntimeError,'Original r11 failure'):f.run()


class SourcePreservation(unittest.TestCase):
    def test_scientific_modules_unchanged_except_local_version_imports(self):
        for part in ('science','primitives','primitive_oracles','model_phases','quality','workloads','raw_verify','http_transport'):
            original=(ROOT/('nemotron9b_package_'+part+'_v26.py')).read_text()
            successor=(ROOT/('nemotron9b_package_'+part+'_v27.py')).read_text().replace('_v27','_v26').replace('-v27-','-v26-').replace(' v27 ',' v26 ')
            self.assertEqual(ast.dump(ast.parse(original)),ast.dump(ast.parse(successor)),part)

    def test_metadata_preparation_has_no_native_bundle_or_physical_calls(self):
        text=(ROOT/'research/nemotron9b_r12_r11_metadata_v1.py').read_text()
        calls=[ast.unparse(x.func)for x in ast.walk(ast.parse(text))if isinstance(x,ast.Call)]
        for prohibited in ('p.bundle','p.low.gone','p.predecessor_drain','p.low.active_heavy','p.low.empty_group'):
            self.assertNotIn(prohibited,calls)


if __name__=='__main__':unittest.main(verbosity=2)
