"""Synthetic rejection controls; never call real services or read payloads."""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q.parent),str(Q)]
import nemotron9b_r12_current_history_completion_v1 as c
from check_closed_journals_official_history_completion_v1 import Fixture as BaseFixture


class Fixture:
    def __init__(self):
        self.base=BaseFixture();b=self.base
        self.binding=b.make_model('current_history_factorial',300)
        self.end=b.load(self.binding['actual_exit']);self.spec=b.load(self.end['spec'])
        root=self.spec['output_root'];runner=str(c.ROOT/'minicpm5_layer1_current_history_factorial_cpu_runner_v1.py')
        self.spec['runner']=runner;self.end['owner']['argv'][2]=runner
        common=['--admission',self.end['outputs']['admission']['path'],'--admission-sha256',self.end['outputs']['admission']['sha256']]
        for owner,mode in [(self.end['coordinator'],'coordinator'),*((x,'rank')for x in self.end['ranks'])]:
            owner['argv']=['/home/ubuntu/venvs/vllm_neuron/bin/python','-B',runner,'--mode',mode,*common]
        intent=b.load(self.end['outputs']['intent']);intent['command']=['systemd-run','/usr/bin/env','X=1',*self.end['coordinator']['argv']]
        props=b.load(self.end['outputs']['service_exit'])['properties']
        props.update(Job='',SubState='dead',ExecStart='{ argv[]='+' '.join(intent['command'][1:])+' ; }')
        b.physical.properties=lambda unit:dict(LoadState='not-found',InvocationID='',MainPID='0',ControlPID='0',ExecMainPID='0',Job='',ActiveState='inactive',SubState='dead')
        self.official=dict(actual_exit=dict(path='/official-exit',sha256='a'*64),independent_audit_exit=dict(path='/official-audit',sha256='b'*64))
        self.storage=dict(actual_exit=dict(path='/storage-exit',sha256='c'*64),independent_audit_exit=dict(path='/storage-audit',sha256='d'*64))
        self.spec.update(completed_official_history=dict(schema='minicpm5_layer1_completed_official_history_v1',
            model_exit=self.official['actual_exit'],audit_exit=self.official['independent_audit_exit']),
            completed_storage=dict(maintenance_exit=self.storage['actual_exit'],audit_exit=self.storage['independent_audit_exit']))
        b.data[self.spec['plan']['path']]=dict(source_sha256={})
        audit_exit=b.load(self.binding['independent_audit_exit']);audit=b.load(audit_exit['result'])
        audit_exit['owner']['argv']=['python3','-B','research/audit_minicpm5_layer1_current_history_factorial_result_v1.py']
        self.binding.update(actual_live=self.end['live'],spec=self.end['spec'],review=self.end['review'],independent_audit=audit_exit['result'])
        self.binding['audit_live']=b.put('audit-live',dict(schema='minicpm5_current_history_factorial_audit_live_v1',
            reviewer='/root',actual_identity_verified=True,owner=copy.deepcopy(audit_exit['owner']),
            actual_execution_handle=copy.deepcopy(audit_exit['actual_execution_handle']),model_actual_exit=self.binding['actual_exit']))
        proof=dict(decode_calls=180,raw_bytes=2949120,unique_replicated_output_rows=360,
            previous_real_controls_full_bytes_exact=True,previous_control_calls=108)
        audit['result']=copy.deepcopy(proof);rank_results=[]
        for index,owner in enumerate(self.end['ranks']):
            path=root+'/rank%d-result.json'%index
            row=dict(complete=True,rank=index,process=copy.deepcopy(owner),admission=self.end['outputs']['admission'],self_exit_claimed=False)
            b.data[path]=row;self.end['outputs']['rank%d_result'%index]=dict(path=path,sha256='f'*64)
            rank_results.append(copy.deepcopy(row))
        b.load(self.end['outputs']['coordinator']).update(rank_owners=self.end['ranks'],results=rank_results,
            validated=copy.deepcopy(proof),self_exit_claimed=False)
        b.load(self.end['outputs']['cleanup'])['owners_checked']=[self.end['coordinator'],*self.end['ranks']]

    def run(self,physical=False):
        return c.current(self.binding,self.official,self.storage,self.base.lock,self.base.load,
            self.base.physical if physical else None)


class Checks(unittest.TestCase):
    def test_metadata_validates_all_five_without_physical_calls(self):
        f=Fixture();out=f.run();self.assertEqual(len(out['owners']),5)
        self.assertFalse(out['physical_checks_performed']);self.assertEqual(f.base.checked,[])

    def test_physical_mode_rechecks_all_five_and_group(self):
        f=Fixture();out=f.run(True);self.assertTrue(out['physical_checks_performed'])
        self.assertEqual(sorted(f.base.checked),[300,301,302,303,310])

    def test_missing_or_different_latest_live_spec_review_rejected(self):
        for key in ('actual_live','spec','review'):
            f=Fixture();del f.binding[key]
            with self.assertRaisesRegex(RuntimeError,'Complete latest'):f.run()
        f=Fixture();f.binding['spec']=dict(path='/other',sha256='e'*64)
        with self.assertRaisesRegex(RuntimeError,'handoff binds'):f.run()

    def test_auditor_live_owner_session_and_model_rejected(self):
        for key,value in [('owner',dict(pid=999)),('actual_execution_handle',dict(kind='tool_session',session_id=999)),
                          ('model_actual_exit',dict(path='/wrong',sha256='f'*64))]:
            f=Fixture();f.base.load(f.binding['audit_live'])[key]=value
            with self.assertRaisesRegex(RuntimeError,'live-to-exit'):f.run()

    def test_wrong_current_rank_command_rejected(self):
        f=Fixture();f.end['ranks'][1]['argv'][4]='unrelated'
        with self.assertRaisesRegex(RuntimeError,'both rank commands'):f.run()

    def test_failed_rank_or_stale_rank_result_rejected(self):
        f=Fixture();f.base.load(f.end['outputs']['children'])['exit_codes']=[0,1]
        with self.assertRaisesRegex(RuntimeError,'successful exits'):f.run()
        f=Fixture();f.base.load(f.end['outputs']['rank1_result'])['process']['start_ticks']='999'
        with self.assertRaisesRegex(RuntimeError,'rank result binds'):f.run()

    def test_service_identity_and_cleanup_owner_omission_rejected(self):
        f=Fixture();f.base.load(f.end['outputs']['service_exit'])['properties']['InvocationID']='b'*32
        with self.assertRaisesRegex(RuntimeError,'successful service exit'):f.run()
        f=Fixture();f.base.load(f.end['outputs']['cleanup'])['owners_checked'].pop()
        with self.assertRaisesRegex(RuntimeError,'cleanup includes'):f.run()

    def test_wrong_previous_official_or_storage_chain_rejected(self):
        for key in ('completed_official_history','completed_storage'):
            f=Fixture();f.spec[key]={}
            with self.assertRaises((RuntimeError,KeyError)):f.run()

    def test_changed_independent_manifest_rejected(self):
        f=Fixture();a=f.base.load(f.binding['independent_audit']);a['result']['decode_calls']=179
        with self.assertRaisesRegex(RuntimeError,'output manifest'):f.run()

    def test_live_latest_owner_or_group_still_blocks_physical_mode(self):
        f=Fixture();f.base.physical.gone=mock.Mock(side_effect=RuntimeError('owner still alive'))
        with self.assertRaisesRegex(RuntimeError,'owner still alive'):f.run(True)
        f=Fixture();f.base.physical.empty_group=mock.Mock(side_effect=RuntimeError('group nonempty'))
        with self.assertRaisesRegex(RuntimeError,'group nonempty'):f.run(True)


if __name__=='__main__':unittest.main(verbosity=2)
