"""Independent narrowly scoped current-authority controls; system IO is memory-only."""
import copy
from contextlib import ExitStack
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'research'),str(ROOT)]
from research import minicpm5_layer1_operator_native_process_adapter_v6 as r
from research import minicpm5_native_physical_authority_v1 as auth
from research import check_minicpm5_layer1_operator_native_current_execution_v1 as fixture

class IndependentAuthority(unittest.TestCase):
    def setUp(self):
        f=fixture.CurrentExecution('test_legitimate_running_handoff_accepted_with_nonempty_live_fields');f.setUp();self.f=f
        self.a=copy.deepcopy(f.admission);self.a.update(lock_policy=copy.deepcopy(r.LOCK_POLICY),attempt_id='f'*32,retained_fd=9)
        self.co=copy.deepcopy(f.co);self.co['cgroup']='0::'+r.p.GROUP_PREFIX+self.a['unit']+'.service'
        self.co_process={k:self.co[k]for k in r.life.PROCESS_KEYS}
        self.native=dict(pid=102,start_ticks='12',ppid=101,argv=r.life.command(self.a['contract'],'native',f.admission_ref),cgroup=self.co['cgroup'])
        self.procs={100:f.outer,101:self.co_process,102:self.native};self.active={self.a['unit']}
        self.lease=r.Lease(self.a);self.lease.processes=[self.co_process,self.native]
        self.lease.execution_observation=copy.deepcopy(f.observation)
        self.metadata=Mock();self.metadata.read.side_effect=lambda ref:self.a if ref==f.admission_ref else f.intent
        self.lease.metadata=self.metadata
        props=dict(LoadState='loaded',Id=self.a['unit']+'.service',InvocationID=self.co['invocation_id'],ExecMainPID='101',
            MainPID='101',ControlPID='0',Job='',ControlGroup=r.p.GROUP_PREFIX+self.a['unit']+'.service',ExecStopPost='')
        for key,value in r.life.PROPS.items():
            props[key.replace('Sec','USec')if key in ('RuntimeMaxSec','TimeoutStopSec')else key]=value
        props.update(RuntimeMaxUSec='2h',TimeoutStopUSec='15s')
        argv=['/usr/bin/env','-i',*[k+'='+v for k,v in r.environment(self.a).items()],*r.life.command(self.a['contract'],'coordinator',f.admission_ref)]
        props['ExecStart']='{ path=/usr/bin/env ; argv[]='+' '.join(argv)+' ; ignore_errors=no ; }'
        self.props=props;self.commands=[]
    def command(self,argv,**kw):
        self.commands.append(argv)
        if 'list-units'in argv:return '\n'.join(n+'.service loaded active running mock'for n in self.active)
        if 'show'in argv:return '\n'.join(k+'='+v for k,v in self.props.items())
        raise AssertionError(argv)
    def environment(self,stack,pid=101,owner=False):
        for item in [patch.object(r,'descriptor_observation',return_value={'global':{'target_row_count':1}}),
            patch.object(r.p,'clear_window'),patch.object(r.p,'proc',side_effect=lambda pid:self.procs.get(pid)),
            patch.object(auth.os,'getpid',return_value=pid),patch.object(r.p,'command',side_effect=self.command),
            patch.object(r.time,'monotonic',return_value=0),patch.object(r,'identity',return_value=self.f.observation['identity']),
            patch.object(r.completion.latest,'latest_handoff',return_value=self.f.observation),
            patch.object(r.p,'lock_proof',side_effect=AssertionError('Legacy self-lock guard invoked'))]:stack.enter_context(item)
        if owner:
            self.lease=r.Lease(self.a)
            return auth.Authority(r,self.lease)
        return auth.Authority(r,self.lease,admission_ref=self.f.admission_ref,admission=self.a,coordinator=self.co,metadata=self.metadata)
    def test_real_lease_and_real_service_contract_accept_coordinator_and_direct_native_only(self):
        for pid in (101,102):
            with self.subTest(pid=pid),ExitStack()as stack:
                authority=self.environment(stack,pid=pid)
                self.assertEqual(authority.active_heavy(),[self.a['unit']])
                authority.require_running_names({self.a['unit']+'.service'})
                for key,value in [('InvocationID','d'*32),('ExecMainPID','999'),('Nice','0')]:
                    with patch.dict(self.props,{key:value}),self.assertRaises(RuntimeError):authority.check()
                with patch.object(auth.os,'getpid',return_value=100),self.assertRaises(RuntimeError):authority.check()
                self.active.add('trn1-unrelated')
                with self.assertRaises(RuntimeError):authority.active_heavy()
                self.active.remove('trn1-unrelated')
        self.assertTrue(any('show'in argv for argv in self.commands))
    def test_outer_authority_has_no_exclusion_and_nested_lease_failure_is_not_bypassed(self):
        with ExitStack()as stack:
            authority=self.environment(stack,pid=100,owner=True)
            self.active.clear();self.assertEqual(authority.active_heavy(),[])
            self.active.add(self.a['unit'])
            with self.assertRaises(RuntimeError):authority.active_heavy()
            with self.assertRaises(RuntimeError):authority.require_running_names({self.a['unit']+'.service'})
        with ExitStack()as stack:
            authority=self.environment(stack,pid=102)
            failure=RuntimeError('actual admitted descriptor disappeared')
            with patch.object(r,'descriptor_observation',side_effect=failure):
                with self.assertRaises(RuntimeError)as caught:authority.check()
                self.assertIs(caught.exception,failure)
            with self.assertRaises(RuntimeError)as caught:authority.check()
            self.assertIs(caught.exception,failure)

if __name__=='__main__':unittest.main(verbosity=2)
