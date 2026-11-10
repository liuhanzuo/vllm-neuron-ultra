"""Synthetic completion boundary checks; never enroll the actual live r12."""
import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research import minicpm5_layer1_sdk_batch_layout_completion_v1 as c
import research


def ref(name):return dict(path='/synthetic/'+name+'.json',sha256='a'*64)
def owner(n,argv=None):return dict(pid=n,start_ticks=str(n+100),argv=argv or ['synthetic',str(n)])


class Completion(unittest.TestCase):
    def setUp(self):
        self.b=dict(schema='minicpm5_layer1_sdk_batch_layout_latest_completion_v1',status='complete',
            nemotron_r12=dict(schema='minicpm5_after_nemotron_r12_binding_v1',actual_exit=ref('future-exit'),failed_release_exit=None),
            current_history=copy.deepcopy(c.CURRENT),terminal_handoff=ref('future-handoff'))
        self.lock=dict(device=1,inode=2);canon={'f':'a'*64}
        self.terminal=dict(live_tool_sessions=[],active_heavy_units=[],nemotron_r12=copy.deepcopy(self.b['nemotron_r12']),
            current_history_factorial=dict(actual_exit=c.CURRENT['model_exit'],independent_audit_exit=c.CURRENT['audit_exit']),canonical_sha256=canon)
        self.n=dict(lock_identity=self.lock,history=dict(canonical_sha256=canon),spec=dict(post_r11_completion=dict(
            current_history_exit=c.CURRENT['model_exit'],current_history_audit_exit=c.CURRENT['audit_exit'])),
            source_sha256={'/synthetic/n.py':'b'*64},owners=[owner(1),owner(1,['exec','variant'])])
        self.current=dict(lock_identity=self.lock,spec=dict(completed_layer1_data={'collector':1},completed_native_cache={'native':1},source_sha256={}),owners=[owner(2)])
        self.collector=dict(lock_identity=self.lock,collector_spec=dict(canonical_sha256=canon,
            finalization=ref('final'),finalizer_exit=ref('final-exit'),catalog=ref('catalog')),
            saved_audit_spec=dict(source_sha256={}),collector_chain=dict(owners=[owner(4)]),saved_audit_owner=owner(5),additional_owners=[owner(6)])
        self.native=dict(lock_identity=self.lock,spec=dict(source_sha256={}),owners=[owner(3)])
        final=dict(passed=True,all_planned_launches_accounted_for=True,all_groups_drained=True,finalizer_owner=owner(7),references=dict(execution_spec=ref('bound')))
        self.data={ref('future-handoff')['path']:self.terminal,ref('final')['path']:final,
            ref('final-exit')['path']:dict(owner=owner(7),actual_exit_observed=True,actual_exit_code=0),
            ref('bound')['path']:dict(plan=ref('bound-plan'),control_root='/synthetic/control'),
            ref('catalog')['path']:{},ref('bound-plan')['path']:{}}
        self.read=Mock(side_effect=lambda r:copy.deepcopy(self.data[r['path']]))
        self.guard=Mock()
        self.fake=SimpleNamespace(completed=Mock(return_value=self.n),physical=Mock(return_value={'complete':True}),
            p=SimpleNamespace(low=SimpleNamespace(gone=Mock(),active_heavy=Mock())))
        self.patches=[patch.dict(sys.modules,{'research.minicpm5_after_nemotron_r12_v1':self.fake}),
            patch.object(research,'minicpm5_after_nemotron_r12_v1',self.fake,create=True),
            patch.object(c,'current_history',return_value=self.current),
            patch.object(c.collector_completion,'completed',return_value=self.collector),
            patch.object(c.native_completion,'completed',return_value=self.native)]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop()for p in reversed(self.patches)])

    def run_done(self):return c.completed(self.b,self.read,self.guard)
    def reject(self):
        with self.assertRaises((RuntimeError,KeyError)):self.run_done()

    def test_complete_metadata_is_not_physical(self):
        d=self.run_done();self.assertTrue(d['metadata_complete']);self.assertFalse(d['physical_checks_performed'])
        self.fake.physical.assert_not_called();self.fake.p.low.gone.assert_not_called()
        self.assertEqual(d['bound']['spec']['control_root'],'/synthetic/control')
        self.assertFalse(d['runtime_admitted']);self.assertFalse(d['full_model_qualified']);self.assertFalse(d['new_speedup_claim'])
    def test_pending_before_read_or_guard(self):
        self.b['status']='pending';self.reject();self.read.assert_not_called();self.guard.assert_not_called();self.fake.completed.assert_not_called()
    def test_exact_fixed_current(self):
        self.b['current_history']['model_exit']['sha256']='b'*64;self.reject();self.read.assert_not_called()
    def test_relative_terminal(self):
        self.b['terminal_handoff']['path']='relative.json';self.reject();self.read.assert_not_called()
    def test_nonjson_terminal(self):
        self.b['terminal_handoff']['path']='/synthetic/raw.bf16';self.reject();self.read.assert_not_called()
    def test_stale_prer12_handoff(self):
        self.b['terminal_handoff']=c.HANDOFF;self.reject();self.read.assert_not_called()
    def test_live_session_rejected_before_r12(self):
        self.terminal['live_tool_sessions']=[28398];self.reject();self.fake.completed.assert_not_called()
    def test_heavy_unit_rejected_before_r12(self):
        self.terminal['active_heavy_units']=['active'];self.reject();self.fake.completed.assert_not_called()
    def test_handoff_exit_mismatch(self):
        self.terminal['nemotron_r12']['actual_exit']=ref('wrong');self.reject();self.fake.completed.assert_not_called()
    def test_failed_release_mismatch(self):
        self.terminal['nemotron_r12']['failed_release_exit']=ref('release');self.reject();self.fake.completed.assert_not_called()
    def test_failed_release_forwarded_not_promoted(self):
        r=ref('release');self.b['nemotron_r12']['failed_release_exit']=r
        self.terminal['nemotron_r12']['failed_release_exit']=r;d=self.run_done()
        self.assertEqual(self.fake.completed.call_args.args[0]['failed_release_exit'],r);self.assertFalse(d['full_model_qualified'])
    def test_current_lineage_mismatch(self):
        self.terminal['current_history_factorial']['independent_audit_exit']=ref('wrong');self.reject()
    def test_lock_mismatch(self):
        self.native['lock_identity']={'device':1,'inode':3};self.reject()
    def test_canonical_mismatch(self):
        self.terminal['canonical_sha256']={'f':'b'*64};self.reject()
    def test_r12_must_follow_exact_mini(self):
        self.n['spec']['post_r11_completion']['current_history_exit']=ref('wrong');self.reject()
    def test_source_conflict_rejected(self):
        self.native['spec']['source_sha256']={'/synthetic/n.py':'c'*64};self.reject()
    def test_source_ref_conflict_rejected(self):
        self.n['source_sha256'][ref('future-handoff')['path']]='b'*64;self.reject()
    def test_no_finalizer_self_exit(self):
        self.data[ref('final-exit')['path']]['actual_exit_observed']=False;self.reject()
    def test_finalizer_owner_mismatch(self):
        self.data[ref('final')['path']]['finalizer_owner']=owner(8);self.reject()
    def test_finalizer_must_drain_all(self):
        self.data[ref('final')['path']]['all_groups_drained']=False;self.reject()
    def test_owner_variants_and_all_extra_preserved(self):
        d=self.run_done();self.assertEqual(len(d['owners']),8)
        self.assertEqual([o['argv']for o in d['owners']if o['pid']==1],[['synthetic','1'],['exec','variant']])
        for n in range(2,8):self.assertIn(owner(n),d['owners'])
    def test_owner_result_is_detached(self):
        d=self.run_done();d['owners'][0]['argv'].append('changed');self.assertEqual(self.n['owners'][0],owner(1))
    def test_physical_guard_and_all_owners(self):
        d=self.run_done();proof=c.physical(d,self.guard)
        self.fake.physical.assert_called_once_with(d['nemotron'],self.guard)
        self.assertEqual([x.args[0]for x in self.fake.p.low.gone.call_args_list],d['owners'])
        self.assertTrue(proof['physical_checks_performed']);self.fake.p.low.active_heavy.assert_called_once()
    def test_pending_physical_rejected(self):
        d=self.run_done();d['binding']['status']='pending'
        with self.assertRaises(RuntimeError):c.physical(d,self.guard)
        self.fake.physical.assert_not_called()
    def test_invalid_owner_identity(self):
        self.n['owners'][0]['pid']=False;self.reject()
    def test_old_pid_start_only_owner_retained(self):
        old=dict(pid=3389537,start_ticks='395641099',identity_source='historical',role='old')
        self.n['owners'].append(old);self.assertIn(old,self.run_done()['owners'])
    def test_empty_argv_variant_retained(self):
        old=owner(1);old['argv']=[];self.n['owners'].append(old)
        self.assertIn(old,self.run_done()['owners'])
    def test_zero_start_rejected(self):
        self.n['owners'][0]['start_ticks']='0';self.reject()
    def test_nonstring_argv_rejected(self):
        self.n['owners'][0]['argv']=[1];self.reject()
    def test_guard_failure_prevents_load(self):
        self.guard.side_effect=RuntimeError('lost lock');self.reject();self.read.assert_not_called()


if __name__=='__main__':
    unittest.main(verbosity=2)
