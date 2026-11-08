"""Synthetic terminal contract regressions; no service/model or future receipt."""
import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q.parent),str(Q)]
import observe_minicpm5_attention_terminal_v1 as r


def fixture():
    root=Q.parent/'minicpm5-frame1-attention-r99-20261108-v99'
    ref=lambda name:dict(path=str(root/name),sha256='a'*64)
    refs={name:ref(name)for name in r.names()}
    spec=dict(units={p:'trn1-mini-f1-attention-r99-20261108-v99-'+p for p in r.m.PHASES},master_port=29571,output_root=str(root),current_catalog=ref('predecessor.json'))
    outer=dict(pid=100,start_ticks='1000',ppid=99,argv=['python3','-B',r.m.SELF],cgroup='0::/synthetic.scope')
    exited=dict(schema='minicpm5_frame1_attention_actual_external_exit_v1',reviewer='/root',actual_tool_exit_observed=True,actual_exit=dict(session_id=88,exit_code=0,tool_chunk='abcdef'),spec=ref('spec.json'),review=ref('review.json'),outer_owner=outer,new_speedup_claim=False)
    live=dict(schema='minicpm5_frame1_attention_actual_live_v1',reviewer='/root',actual_identity_verified=True,terminal_exit_observed=False,actual_tool_session=88,**{k:copy.deepcopy(exited[k])for k in ('spec','review','outer_owner')})
    a=dict(spec=exited['spec'],review=exited['review'],outer_owner=outer,attempt_id='b'*32,lock_identity=dict(device=1,inode=2),current_catalog=spec['current_catalog'],current_history_drained=True)
    records={'admission.json':a}
    # Preserve actual successful property shape while giving fixtures explicit
    # new commands/owners. No real service is inspected or launched.
    previous=json.loads((Q/'minicpm5-frame1-upstream-r1-20261108-v1-root-terminal.json').read_text())
    for i,phase in enumerate(r.m.PHASES):
        owner=dict(pid=101+i,start_ticks=str(1001+i),ppid=1,argv=r.m.arguments('phase',refs['admission.json'],phase),cgroup='0::'+r.m.p.GROUP_PREFIX+spec['units'][phase]+'.service',unit=spec['units'][phase],invocation_id=str(i+1)*32)
        props=copy.deepcopy(previous['physical_drain_samples'][0][i]['properties'])
        for key,value in r.m.props(phase).items():
            if key in ('RuntimeMaxSec','TimeoutStopSec'):props[key.replace('Sec','USec')]=value+'s'
            else:props[key]=value
        command=r.m.service_command(spec,refs['admission.json'],a['attempt_id'],phase)
        props.update(Id=spec['units'][phase]+'.service',InvocationID=owner['invocation_id'],ExecMainPID=str(owner['pid']),ExecStart='{ argv[]='+' '.join(command[command.index('/usr/bin/env'):])+' ; }')
        records[phase+'-owner.json']=owner
        records[phase+'-exit.json']=dict(owner=owner,observed_by_outer=outer,successful=True,properties=props)
        records[phase+'-result.json']=dict(complete=True,process=r.process(owner))
        records[phase+'-launch-intent.json']=dict(unit=spec['units'][phase],command=command,admission=refs['admission.json'],outer_owner=outer,attempt_id=a['attempt_id'])
    ranks=[]
    for rank in (0,1):
        owner=dict(pid=104+rank,start_ticks=str(1004+rank),ppid=102,argv=r.m.arguments('rank',refs['admission.json'],'cpu'),cgroup=records['cpu-owner.json']['cgroup']);ranks.append(owner)
        records['rank'+str(rank)+'-owner.json']=owner
        records['cpu-rank'+str(rank)+'-result.json']=dict(complete=True,process=owner,capture=refs['cpu-rank'+str(rank)+'/capture.json'])
    captures={role:refs[role+'/capture.json']for role in r.ROLES}
    records['native-result.json']['capture']=captures['native']
    records['cpu-result.json'].update(ranks=ranks,actual_rank_exits=[0,0],captures=[captures[x]for x in r.ROLES[1:]])
    records['attention-review.json']=dict(schema='minicpm5_frame1_attention_complete_review_v1',complete=True,passed=True,all21_own_input_controls_exact=True,all96_full_payloads_authenticated=True,all21_original_upstream_payloads_exact=True,files=96,raw_bytes=353284922,full_model_qualified=False,new_speedup_claim=False,process_lifecycle_verified=False,unique_internal_or_final_KL_cause_proven=False,captures=captures)
    records['replay-result.json'].update(report=refs['attention-review.json'],scientific_passed=True)
    records['result.json']=dict(complete=True,self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False,owner=outer,admission=refs['admission.json'],report=refs['attention-review.json'])
    return root,spec,exited,live,records,refs


class Terminal(unittest.TestCase):
    def setUp(self):self.root,self.spec,self.exited,self.live,self.records,self.refs=fixture()
    def check(self):return r.terminal_metadata(self.root,self.spec,self.exited,self.records,self.refs)
    def reject(self):
        with self.assertRaises((RuntimeError,KeyError)):self.check()
    def test_complete_actual_shape_and_process_bindings(self):
        value=self.check();self.assertTrue(value['all_own_input_controls_exact']);self.assertEqual(len(value['rank_owners']),2)
    def test_missing_phase_fails(self):del self.records['native-exit.json'];self.reject()
    def test_phase_failure_fails_even_with_result(self):self.records['cpu-exit.json']['properties']['ExecMainStatus']='1';self.reject()
    def test_wrong_invocation_fails(self):self.records['native-exit.json']['properties']['InvocationID']='f'*32;self.reject()
    def test_nonterminal_phase_fails(self):self.records['replay-exit.json']['properties']['MainPID']='103';self.reject()
    def test_replaced_phase_start_tick_fails(self):self.records['native-result.json']['process']['start_ticks']='999';self.reject()
    def test_mismatched_admission_fails(self):self.records['admission.json']['spec']={'path':'wrong','sha256':'0'*64};self.reject()
    def test_changed_launch_command_fails(self):self.records['native-launch-intent.json']['command'].append('--changed');self.reject()
    def test_nonzero_rank_fails(self):self.records['cpu-result.json']['actual_rank_exits'][1]=1;self.reject()
    def test_boolean_rank_exit_fails(self):self.records['cpu-result.json']['actual_rank_exits'][1]=False;self.reject()
    def test_missing_rank_fails(self):del self.records['rank1-owner.json'];self.reject()
    def test_reparented_rank_fails(self):self.records['rank1-owner.json']['ppid']=999;self.reject()
    def test_wrong_capture_reference_fails(self):self.records['native-result.json']['capture']={'path':'wrong','sha256':'0'*64};self.reject()
    def test_forged_report_pass_fails(self):self.records['attention-review.json']['all21_own_input_controls_exact']=False;self.reject()
    def test_legitimate_failed_controls_preserved(self):
        self.records['attention-review.json'].update(passed=False,all21_own_input_controls_exact=False);self.records['replay-result.json']['scientific_passed']=False
        self.assertFalse(self.check()['all_own_input_controls_exact'])
    def test_payload_count_cannot_shrink(self):self.records['attention-review.json']['files']=95;self.reject()
    def test_self_exit_claim_rejected(self):self.records['result.json']['self_exit_claimed']=True;self.reject()
    def test_speedup_claim_rejected(self):self.records['result.json']['new_speedup_claim']=True;self.reject()
    def test_external_exit_bound_to_live_session(self):self.assertEqual(r.external_exit(self.exited,self.live),self.exited['outer_owner'])
    def test_wrong_session_rejected(self):
        self.exited['actual_exit']['session_id']+=1
        with self.assertRaises(RuntimeError):r.external_exit(self.exited,self.live)
    def test_failed_or_unobserved_external_exit_rejected(self):
        for key,value in [('exit_code',1),('exit_code',False),('session_id',True),('tool_chunk','invented')]:
            ex=copy.deepcopy(self.exited);ex['actual_exit'][key]=value
            with self.assertRaises(RuntimeError):r.external_exit(ex,self.live)
        self.exited['actual_tool_exit_observed']=False
        with self.assertRaises(RuntimeError):r.external_exit(self.exited,self.live)
    def test_historical_upstream_live_cannot_authorize_attention(self):
        live=json.loads((Q/'minicpm5-frame1-upstream-r1-20261108-v1-actual-live.json').read_text())
        with self.assertRaises(RuntimeError):r.external_exit(self.exited,live)
    def test_failed_exit_rejected_before_lock(self):
        self.exited['actual_exit']['exit_code']=1;self.exited['live']={}
        with patch.object(r.m.p,'read_ref',side_effect=[self.exited,self.live]),patch.object(r.m.old.post.external,'acquire_lock')as lock:
            with self.assertRaises(RuntimeError):r.observe({}, {})
            lock.assert_not_called()
    def test_original_runtime_pins_and_no_model_import(self):
        import hashlib
        for name,wanted in [('minicpm5_frame1_attention_runner_v1.py','022dc28be2c713340aea5c3df34676b848c2413d70c35be6fb6ed1f076744b42'),('minicpm5_frame1_attention_worker_v1.py','73cfe0de0fc38a922515f3d7da1ae4db56855d529d6cf9692a8c5a13d11634c7')]:self.assertEqual(hashlib.sha256((Q.parent/name).read_bytes()).hexdigest(),wanted)
        self.assertFalse({'torch','numpy','torch_neuronx','transformers'}&set(sys.modules))
        self.assertFalse(list(Q.glob('minicpm5-frame1-attention-r*-spec.json')))

class Orchestration(unittest.TestCase):
    def execute(self,fail_drain=False,scientific=False):
        from contextlib import ExitStack,redirect_stdout
        from io import StringIO
        root,spec,exited,live,records,refs=fixture()
        spec.update(schema='minicpm5_frame1_attention_spec_v1',runner=r.m.SELF,canonical_sha256=r.m.CANONICAL,new_sources={},root_floor_bytes=28*2**30)
        review=dict(spec=exited['spec'],reviewer='/root',passed=True,source_only=True,independent_review=True,blockers=[])
        exit_ref=dict(path='/synthetic/actual-exit.json',sha256='a'*64);approval_ref=dict(path='/synthetic/source-review.json',sha256='a'*64);exited['live']=dict(path='/synthetic/live.json',sha256='a'*64)
        approval=dict(schema='minicpm5_attention_terminal_source_review_v1',reviewer='/root',passed=True,source_only=True,independent_review=True,blockers=[],actual_exit=exit_ref,source_sha256={str(r.SELF):'a'*64})
        metadata=dict(raw_bytes=353284922)
        records['attention-review.json'].update(metadata=metadata,passed=scientific,all21_own_input_controls_exact=scientific)
        records['replay-result.json']['scientific_passed']=scientific
        for role in r.ROLES:records[role+'/capture.json']={}
        data={exit_ref['path']:exited,approval_ref['path']:approval,exited['live']['path']:live,exited['spec']['path']:spec,exited['review']['path']:review,r.m.BASE_PLAN['path']:dict(source_sha256={})}
        data.update({refs[name]['path']:value for name,value in records.items()})
        observer=dict(pid=200,start_ticks='2000',ppid=199,argv=['python3','-B',str(r.SELF)],cgroup='0::/synthetic.scope')
        def prop(unit):return records[next(p for p in r.m.PHASES if spec['units'][p]==unit)+'-exit.json']['properties']
        with ExitStack()as stack:
            for target,name,kwargs in [(r.m.p,'read_ref',dict(side_effect=lambda ref:data[ref['path']])),(r.m.p,'sha',dict(side_effect=lambda path:r.m.CANONICAL.get(Path(path).name,'a'*64))),(r,'loaded_sources',dict(return_value={str(r.SELF):'a'*64})),(r.m.old,'reference',dict(side_effect=lambda path:refs[str(Path(path).relative_to(root))])),(r.meta,'validate',dict(return_value=metadata)),(r.m,'usage',dict(return_value=dict(allocated_bytes=353284922,files=125))),(r.m.p,'properties',dict(side_effect=prop))]:stack.enter_context(patch.object(target,name,**kwargs))
            for name in ('user_bus','clear_window','active_heavy','gone','lock_proof'):stack.enter_context(patch.object(r.m.p,name))
            drain=stack.enter_context(patch.object(r.m.p,'empty_group',side_effect=RuntimeError('still occupied')if fail_drain else lambda group:dict(group=group,absent=True)))
            lock=stack.enter_context(patch.object(r.m.old.post.external,'acquire_lock',return_value=(10,observer,dict(device=1,inode=2))))
            release=stack.enter_context(patch.object(r.m.old.post.external,'release_lock'))
            write=stack.enter_context(patch.object(r.m.worker,'once',return_value=dict(path='/synthetic/terminal.json',sha256='a'*64)))
            with redirect_stdout(StringIO()):
                if fail_drain:
                    with self.assertRaisesRegex(RuntimeError,'still occupied'):r.observe(exit_ref,approval_ref)
                else:r.observe(exit_ref,approval_ref)
            lock.assert_called_once();release.assert_called_once_with(10)
            if fail_drain:write.assert_not_called()
            else:
                self.assertEqual(drain.call_count,6);write.assert_called_once();terminal=write.call_args.args[1]
                self.assertIs(terminal['all_own_input_controls_exact'],scientific);self.assertFalse(terminal['independent_saved_byte_review_complete']);self.assertFalse(terminal['full_model_qualified']);self.assertEqual(terminal['terminal_observer'],observer)
                r.post.validate_terminal(terminal)
    def test_complete_orchestration_preserves_scientific_failure(self):self.execute(scientific=False)
    def test_complete_orchestration_preserves_scientific_success(self):self.execute(scientific=True)
    def test_occupied_group_prevents_receipt_and_releases_lock(self):self.execute(fail_drain=True)


if __name__=='__main__':unittest.main(verbosity=2)
