"""Actual worker owner ACK precedes every payload/native action."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import nemotron9b_package_protocol_v11 as p


def prepare(spec,guard):
    for path,item in spec['payloads'].items():
        guard();p.require(p.file_stat(path)==item['stat'],'Original payload stat')
        p.require(p.stream_sha(path,guard)==item['sha256'],'Original payload content')
        p.require(p.file_stat(path)==item['stat'],'Payload changed during verification')
    from vllm_neuron.models.nemotron9b import identity,prepare_view
    software=identity.software_identity()
    p.require(software==spec['runtime_software_identity'],'Root-bound actual software identity')
    view=p.paths(spec)['view'];manifest=prepare_view.prepare(spec['original_model'],view)
    tok=prepare_view.load_tokenizer(view)
    text='Describe the patterns in these records and explain your reasoning.\n'+''.join(f'Record {i}: inventory {100+i}, sales {20+i%17}, returns {i%4}.\n'for i in range(400))+'\nGive a detailed explanation.'
    full=tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    inputs={str(n):full[:n-16]+full[-16:]for n in (32,128,512,1536)}
    p.require(all(len(ids)==int(n)for n,ids in inputs.items()),'Original workload lengths')
    fixture=p.read(p.PACKAGE/'vllm_neuron/models/nemotron9b/tokenizer_fixtures.json')
    suite=p.read(spec['quality_suite']['path'],spec['quality_suite']['sha256']);rows={x['id']:x for x in fixture['fixtures']}
    p.require(len(suite['cases'])==len(rows)==96,'Complete quality96')
    for item in suite['cases']:
        row=rows[item['id']];p.require(row['prompt']==item['prompt']and row['output_budget']==item['max_tokens'],'Original quality prompt/budget')
    e=Path(spec['evidence_root']);p.once(e/'workload-inputs.json',inputs)
    proof=dict(complete=True,passed=True,view_manifest=manifest,software=software,quality_inputs=96,raw_tokenizer_checks=22,
        original_payloads_verified=len(spec['payloads']),workload_inputs=p.ref(e/'workload-inputs.json'))
    p.once(e/'prepare.json',proof);return proof


def migration(spec,phase):
    import torch
    from nemotron9b_package_science_v11 import metrics,passed
    paths={'numeric':'numeric.json.pt','cold':'cold-buckets.pt','long':'long-decode-validation.pt'}
    fresh=torch.load(Path(spec['evidence_root'])/paths[phase],map_location='cpu',weights_only=True)
    old=torch.load(spec['references']['historical_'+phase],map_location='cpu',weights_only=True)
    counts={'numeric':32,'cold':84,'long':93};assert len(old)==len(fresh)==counts[phase]
    rows=[]
    for left,right in zip(old,fresh):
        assert {k:v for k,v in left.items()if k!='logits'}=={k:v for k,v in right.items()if k!='logits'}
        a,b=left['logits'].float().cpu().contiguous(),right['logits'].float().cpu().contiguous()
        assert a.shape==b.shape==(131072,)
        rows.append(dict(metrics=metrics(b,a),bitwise_equal=torch.equal(a.view(torch.uint8),b.view(torch.uint8))))
    proof=dict(complete=True,passed=passed([x['metrics']for x in rows]),frames=len(rows),rows=rows)
    p.once(Path(spec['evidence_root'])/(phase+'-migration.json'),proof)
    assert proof['passed'];return proof


def quality(spec):
    import nemotron9b_package_quality_v11 as old
    suite=p.read(spec['quality_suite']['path'],spec['quality_suite']['sha256'])
    fixture=p.read(p.PACKAGE/'vllm_neuron/models/nemotron9b/tokenizer_fixtures.json')
    rows={x['id']:x for x in fixture['fixtures']};assert len(rows)==len(suite['cases'])==96
    def prepared(case,actual):
        assert actual==suite
        return [(item,rows[item['id']]['input_ids'])for item in actual['cases']]
    old.prepare_inputs=prepared
    e=Path(spec['evidence_root']);report=old.evaluate(p.case(spec),e/'quality96.json')
    assert report['technical_complete']and report['total']==96
    assert all(row['usage']['prompt_tokens']==len(row['prompt_ids'])for row in report['rows'])
    pair=old.compare(spec['historical_quality']['path'],e/'quality96.json',e/'quality-comparison.json')
    assert pair['total']==pair['identical_texts']==96 and not pair['lost_correct_cases']
    return dict(complete=True,passed=True,quality=p.ref(e/'quality96.json'),comparison=p.ref(e/'quality-comparison.json'),
        quality_correct=report['correct'],quality_total=96,all96_identical=True)


def workloads(spec,stage,owner,guard):
    from nemotron9b_package_workloads_v11 import benchmark,verify,mixed
    e=Path(spec['evidence_root']);c=Path(spec['control_root'])
    prompts={int(k):v for k,v in p.read(e/'workload-inputs.json').items()}
    # No source/payload hashing or directory scans inside the timed window.
    guard();p.once(c/(stage+'-measurement-ready.json'),dict(owner=owner,pid=os.getpid(),prepared=True))
    deadline=time.monotonic()+60
    while not(c/(stage+'-measurement-begin.json')).exists():
        assert time.monotonic()<deadline;time.sleep(.05)
    begin=p.read(c/(stage+'-measurement-begin.json'))
    assert begin['owner']==owner and begin['allowed']is True
    barrier=p.read(begin['observer_barrier']['path'],begin['observer_barrier']['sha256'])
    assert barrier['worker']==owner and barrier['exclusive_observer_gate_held']is True
    from nemotron9b_package_window_v11 import registry
    assert registry()==barrier['registry']and barrier['registry']['performance_window']is True
    result=dict(case=p.case(spec),round=int(stage.rsplit('-',1)[1]),workloads={},inputs={str(k):v for k,v in prompts.items()})
    try:
        for name,n,m,sampling in [('short_decode',128,512,{}),('medium_prompt',512,128,{}),('long_prompt',1536,128,{}),
                ('short_stochastic',128,512,dict(temperature=.7,top_p=.9,top_k=40)),('mixed_lengths',None,None,dict(temperature=.7,top_p=.9,top_k=40))]:
            report=mixed(p.case(spec),prompts,sampling,rounds=1)if name=='mixed_lengths'else benchmark(8010,p.case(spec)['served_name'],[prompts[n]],4,output_tokens=m,rounds=1,sampling=sampling)
            if name!='mixed_lengths':verify(report)
            result['workloads'][name]=report;p.save(e/(stage+'.json'),result)
        result.update(complete=True,passed=True);p.save(e/(stage+'.json'),result)
    finally:p.once(c/(stage+'-measurement-end.json'),dict(owner=owner,complete=result.get('complete',False)))
    guard();return dict(complete=True,passed=True,report=p.ref(e/(stage+'.json')),workloads=5,rounds=1)


def http_stage(spec,stage,owner,guard,spec_ref,review_ref):
    """Actual post-exec identity and deliberate terminal status gate success."""
    from nemotron9b_package_server_lifecycle_v11 import observe_expected,shutdown,validate_chain
    c=Path(spec['control_root']);e=Path(spec['evidence_root']);child=None;child_owner=None;pidfd=None
    observation=None;workload_complete=False;original_error=None;result=None
    with socket.socket()as sock:assert sock.connect_ex(('127.0.0.1',8010))!=0,'Port already occupied'
    command=p.argv(spec,stage,spec_ref,review_ref)+['--server-child']
    p.once(c/(stage+'-server-intent.json'),dict(argv=command,parent=owner))
    log=c/(stage+'-server.log');start=time.monotonic()
    try:
        with log.open('xb')as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        # The handle is opened while this child is unreaped; all signals remain
        # bound to that kernel process even if a later observation detects exit.
        pidfd=os.pidfd_open(child.pid,0)
        actual=p.low.proc(child.pid);assert actual and actual['cgroup']=='0::'+p.group(spec,stage)
        deadline=time.monotonic()+15
        while actual['argv']!=command:
            assert time.monotonic()<deadline and child.poll()is None;time.sleep(.02);actual=p.low.proc(child.pid)
            assert actual is not None,'Child disappeared before ACK'
        child_owner=actual
        observe_expected(spec,stage,owner,child_owner,command)
        p.once(c/(stage+'-server-owner.json'),child_owner)
        p.once(c/(stage+'-server-ack.json'),dict(parent=owner,child=child_owner,execution_spec=spec_ref,source_review=review_ref))
        deadline=time.monotonic()+60
        from nemotron9b_exec_transition_v1 import await_exec
        def validate_exec_intent():
            executed=p.read(c/(stage+'-server-exec.json'))
            assert executed==dict(original_owner=child_owner,argv=p.server_argv(spec),ack=p.ref(c/(stage+'-server-ack.json')))
        observation=await_exec(child_owner=child_owner,target_argv=p.server_argv(spec),
            read_process=lambda:p.low.proc(child.pid),poll=child.poll,guard=guard,
            validate_intent=validate_exec_intent,
            observe_target=lambda:observe_expected(spec,stage,owner,child_owner,p.server_argv(spec)),
            save_diagnostic=lambda value:p.once(c/(stage+'-server-exec-diagnostic.json'),value),
            deadline=deadline)
        executed=p.read(c/(stage+'-server-exec.json'))
        assert executed==dict(original_owner=child_owner,argv=p.server_argv(spec),ack=p.ref(c/(stage+'-server-ack.json')))
        p.once(c/(stage+'-server-post-exec.json'),dict(parent=owner,observation=observation,
            exec_intent=p.ref(c/(stage+'-server-exec.json')),ack=p.ref(c/(stage+'-server-ack.json'))))
        while not(c/(stage+'-server-observed.json')).exists():
            guard();assert child.poll()is None and time.monotonic()<deadline,'Independent post-exec ACK timeout/exit'
            assert observe_expected(spec,stage,owner,child_owner,p.server_argv(spec))==observation
            time.sleep(.05)
        independent=p.read(c/(stage+'-server-observed.json'));admission=p.read(c/'admission.json')
        assert independent==dict(parent=owner,observation=observation,post_exec=p.ref(c/(stage+'-server-post-exec.json')),
            observer='serial-controller',controller=admission['controller'])
        deadline=time.monotonic()+(14000 if stage=='compile'else 900)
        while True:
            guard();assert child.poll()is None,'Server exited before readiness'
            assert observe_expected(spec,stage,owner,child_owner,p.server_argv(spec))==observation
            try:
                with urllib.request.urlopen('http://127.0.0.1:8010/health',timeout=2)as response:
                    if response.status==200:break
            except OSError:pass
            assert time.monotonic()<deadline,'Server readiness deadline';time.sleep(1)
        assert child.poll()is None,'Server exited after health response'
        assert observe_expected(spec,stage,owner,child_owner,p.server_argv(spec))==observation
        p.once(c/(stage+'-server-ready.json'),dict(observation=observation,post_exec=p.ref(c/(stage+'-server-post-exec.json')),
            controller_observation=p.ref(c/(stage+'-server-observed.json')),health_status=200))
        if stage=='compile':
            from nemotron9b_package_http_transport_v11 import request
            prompts=p.read(e/'workload-inputs.json');rows=[]
            for ids in (prompts['32'],prompts['128']):
                row=request(8010,p.case(spec)['served_name'],ids,16,stream=True)
                assert row['usage']['prompt_tokens']==len(ids)and row['usage']['completion_tokens']==16
                rows.append(row)
            result=dict(complete=True,passed=True,smoke=rows,artifact_manifest=p.ref(Path(spec['work_root'])/'artifact/nemotron9b-ultra-artifact.json'))
            p.once(e/'compile-smoke.json',result)
        elif stage=='quality':result=quality(spec)
        else:result=workloads(spec,stage,owner,guard)
        assert result['complete']is True and result['passed']is True
        assert child.poll()is None,'Server exited after final response'
        assert observe_expected(spec,stage,owner,child_owner,p.server_argv(spec))==observation
        p.once(c/(stage+'-server-request-complete.json'),dict(observation=observation,
            ready=p.ref(c/(stage+'-server-ready.json')),workload_complete=True))
        workload_complete=True
    except BaseException as error:
        original_error=repr(error)
        raise
    finally:
        try:
            if child is not None and child_owner is not None and pidfd is not None:
                shutdown(spec,stage,owner,child,child_owner,pidfd,observation,workload_complete,original_error,start)
        finally:
            if pidfd is not None:os.close(pidfd)
    refs={name:p.ref(c/(stage+'-server-'+name+'.json'))for name in
        ('intent','owner','ack','exec','post-exec','observed','ready','request-complete','stop-intent','exit')}
    result['server_lifecycle']=refs
    validate_chain(spec,stage,owner,result,p.read(c/'admission.json'))
    return result


def serve_child(stage,spec_ref,review_ref):
    p.require(not {'torch','numpy','transformers','vllm'}&set(sys.modules),'Child native imports wait for actual owner ACK')
    spec,_=p.bundle(spec_ref,review_ref);c=Path(spec['control_root']);deadline=time.monotonic()+60
    while not(c/(stage+'-server-ack.json')).exists():
        p.require(time.monotonic()<deadline,'Server child ACK timeout');time.sleep(.05)
    ack=p.read(c/(stage+'-server-ack.json'));actual=p.low.proc(os.getpid())
    p.require(ack['execution_spec']==spec_ref and ack['source_review']==review_ref and ack['child']==actual
        and actual['argv']==p.argv(spec,stage,spec_ref,review_ref)+['--server-child'],'Actual server child ACK')
    owner=p.read(c/(stage+'-owner.json'));p.require(ack['parent']==owner and actual['ppid']==owner['pid'],'Actual parent-child ownership')
    p.service_identity(spec,stage,spec_ref,review_ref,owner)
    p.healthy(spec,p.read(c/'admission.json'),stage)
    cmd=p.server_argv(spec)
    p.once(c/(stage+'-server-exec.json'),dict(original_owner=actual,argv=cmd,ack=p.ref(c/(stage+'-server-ack.json'))))
    # Preserve the package module as __main__ so spawn workers execute its
    # top-level registration. A nested runpy under this worker would not do so.
    os.execve(p.PYTHON,cmd,dict(os.environ))


def run(stage,spec_ref,review_ref):
    p.require(sys.dont_write_bytecode and __debug__ and sys.executable==p.PYTHON,'Exact -B native interpreter')
    p.require(not {'torch','numpy','transformers','vllm','neuronxcc'}&set(sys.modules),'ACK before native imports')
    spec,_=p.bundle(spec_ref,review_ref);p.low.user_bus();c=Path(spec['control_root']);leasepath=c/(stage+'-lease.json')
    deadline=time.monotonic()+60
    while not leasepath.exists():p.require(time.monotonic()<deadline,'Owner lease timeout');time.sleep(.05)
    lease_ref=p.ref(leasepath);lease=p.read(leasepath);owner=p.read(c/(stage+'-owner.json'));admission=p.read(c/'admission.json')
    p.require(lease['phase']==stage and lease['execution_spec']==spec_ref and lease['source_review']==review_ref
        and lease['runtime_admitted']is True and lease['admission']==p.ref(c/'admission.json'),'Exact runtime lease')
    p.require(lease['producer']==p.producer(owner)and owner['pid']==os.getpid()and lease['controller']==admission['controller'],'Actual worker/controller binding')
    controller,_=p.service_identity(spec,'controller',spec_ref,review_ref,admission['controller'])
    p.require(controller==admission['controller'],'Live actual controller')
    p.require(all(os.environ.get(k)==v for k,v in p.environment(spec,stage).items()),'Exact scrubbed worker environment')
    lease_stat=p.file_identity(leasepath);last=[0.]
    def guard():
        p.require(p.low.proc(owner['pid'])==p.proc_owner(owner),'Actual worker PID/start/argv/group')
        p.require(p.low.proc(admission['controller']['pid'])==p.proc_owner(admission['controller']),'Actual lock owner alive')
        p.require(p.file_identity(leasepath)==lease_stat,'Lease stat changed')
        p.require(not(c/'cancel.json').exists(),'Root cancellation')
        for path,floor in [(p.ROOT,p.ROOT_FLOOR),(Path('/mnt/nvme'),p.NVME_FLOOR)]:
            fs=os.statvfs(path);p.require(fs.f_bavail*fs.f_frsize>=floor,'Bounded worker disk floor')
        if time.monotonic()-last[0]<1:return
        actual,_=p.service_identity(spec,stage,spec_ref,review_ref,owner);p.require(actual==owner,'Live actual worker')
        p.require(p.sha(leasepath)==lease_ref['sha256'],'Immutable lease')
        p.healthy(spec,admission,stage)
        last[0]=time.monotonic()
    def force_guard():last[0]=0.;guard()
    force_guard();start=time.monotonic()
    try:
        k=p.kind(stage)
        if k=='prepare':result=prepare(spec,guard)
        elif k=='verify-raw':
            from nemotron9b_package_raw_verify_v11 import run as verify_raw
            result=verify_raw(spec,owner,lease_ref,guard)
        elif k in ('alias','split'):
            from nemotron9b_package_primitives_v11 import configure,alias,split
            configure(spec);result=(alias if k=='alias'else split)(stage)
        elif k in ('compile','quality','workloads'):result=http_stage(spec,stage,owner,guard,spec_ref,review_ref)
        else:
            import nemotron9b_package_model_phases_v11 as model
            model.configure(spec)
            getattr(model,{'state-copy':'state_copy','numeric':'numeric','cold':'cold','long':'long_decode'}[stage])()
            if stage!='state-copy':migration(spec,stage)
            output={'state-copy':'state-copy.json','numeric':'numeric.json','cold':'cold-buckets.json','long':'long-decode-validation.json'}[stage]
            result=dict(complete=True,passed=True,report=p.ref(Path(spec['evidence_root'])/output))
        force_guard();p.validate_result(spec,stage,result)
        if stage in p.SCIENTIFIC_STAGES:
            from nemotron9b_package_evidence_v11 import phase_manifest
            result['raw_manifest']=phase_manifest(spec,stage,owner,lease_ref,guard)
        force_guard()
        p.once(c/(stage+'-worker-result.json'),dict(complete=True,passed=True,phase=stage,producer=p.producer(owner),lease=lease_ref,result=result,
            seconds=time.monotonic()-start,new_speedup_claim=False))
    except BaseException as error:
        p.once(c/(stage+'-worker-failure.json'),dict(complete=False,passed=False,phase=stage,producer=p.producer(owner),lease=lease_ref,
            error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start,partial_evidence_retained=True))
        raise


def main():
    a=argparse.ArgumentParser();a.add_argument('--phase',choices=p.STAGES,required=True)
    a.add_argument('--server-child',action='store_true')
    for name in ('spec','spec-sha256','review','review-sha256'):a.add_argument('--'+name,required=True)
    x=a.parse_args();fn=serve_child if x.server_child else run
    fn(x.phase,dict(path=x.spec,sha256=x.spec_sha256),dict(path=x.review,sha256=x.review_sha256))


if __name__=='__main__':main()
