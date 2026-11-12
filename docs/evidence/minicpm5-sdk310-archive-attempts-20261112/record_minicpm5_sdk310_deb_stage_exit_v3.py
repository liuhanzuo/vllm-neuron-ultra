"""External actual-terminal recorder for Mini eight-archive producer/replay."""
import argparse,gzip,hashlib,json,os,sys,time,resource
from pathlib import Path
Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
import dory_serial_stage_contract_v2 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import minicpm5_sdk310_deb_input_binding_v3 as binding
import dory_deb_transport_v3 as io
import minicpm5_sdk310_deb_events_v1 as event_protocol
import run_dory_serial_stage_root_v5 as life
import minicpm5_sdk310_retired_lfm_r2_v2 as retirement

def main():
    cli=argparse.ArgumentParser()
    for k in ('plan','plan-sha256','review','review-sha256','chunk','destination'):cli.add_argument('--'+k,required=True)
    cli.add_argument('--session',required=True,type=int);cli.add_argument('--exit-code',required=True,type=int);a=cli.parse_args()
    c.require(a.exit_code in (0,1)and a.session>0 and sys.dont_write_bytecode,'Root supplies actual zero/one terminal and system -B')
    pr=dict(path=a.plan,sha256=a.plan_sha256);rr=dict(path=a.review,sha256=a.review_sha256);plan=binding.validate(c.load(pr),c.load(rr),pr);c.verify_sources(plan['source_sha256'])
    import minicpm5_layer0_execution_protocol_v4 as p
    p.user_bus();os.sched_setaffinity(0,{5});os.nice(19)
    for limit,key in ((resource.RLIMIT_AS,'address_space_bytes'),(resource.RLIMIT_CPU,'cpu_seconds'),(resource.RLIMIT_FSIZE,'single_file_bytes')):resource.setrlimit(limit,(c.INVENTORY[key],)*2)
    out=Path(plan['output_root']);owner=c.load(c.reference(out/'owner.json')[0]);child=c.load(c.reference(out/'child-owner.json')[0]);parent=owner['owner'];co=child['owner']
    p.gone(parent);p.gone(co);c.require(not life.group_members(co['pid']),'Actual complete group gone')
    fd,recorder,lock,first=runtime.acquire(p,pr,rr);started=time.monotonic();last=[0.]
    def guard():
        c.require(time.monotonic()-started<900,'Original recorder wall limit')
        if time.monotonic()-last[0]>=.5:
            runtime.observe(p,recorder,lock,fd);p.gone(parent);p.gone(co);p.active_heavy()
            c.require(life.free(Q.parent)>=28*c.GIB and life.free('/mnt/nvme')>=30*c.GIB,'Original disk floors');last[0]=time.monotonic()
    try:
        guard();authenticated=binding.authenticate(guard);evidence={};values={}
        names={'owner.json','initial-lock-proof.json','admission.json','child-owner.json','child-exit.json','worker.log','events.jsonl.gz','result.json','worker-result.json','metadata-result.json','failure.json','worker-failure.json','first-descriptor-duplicate-parent.json','first-descriptor-duplicate-worker.json','descriptor-lock-failure-parent.json','descriptor-lock-failure-worker.json'}
        n,_=binding.authenticated(guard);nested={'sources-result.json'}|{x['package']+'.deb'for x in n['packages']}
        for path in sorted(out.rglob('*')):
            guard();c.require(not path.is_symlink(),'No output links')
            if path.is_dir():c.require(path==out/'packages','Only owned archive directory');continue
            name=str(path.relative_to(out));c.require(name in names or path.parent==out/'packages'and path.name in nested,'Finite output inventory')
            if path.stat().st_size==0:
                evidence[name]=dict(path=str(path),bytes=0,sha256=hashlib.sha256(b'').hexdigest());continue
            ref,_=io.file_ref(path,guard,cap=64*c.MIB);evidence[name]=ref
            if path.suffix=='.json':values[name]=c.load(ref,8*c.MIB)
        c.require(binding.storage_usage(out)<=c.INVENTORY['output_bytes'],'Complete output size cap')
        admission=values['admission.json'];binding.validate_admission(admission,authenticated)
        c.require(owner['plan']==pr and owner['review']==rr and owner['lock']==lock and child['command']==co['argv']==admission['command']+['--admission-sha256',evidence['admission.json']['sha256']],'Original source/owner/command/admission binding')
        c.require(owner['owner']==admission['owner']==parent and values['child-exit.json']['owner']==co and values['child-exit.json']['entire_process_group_empty'],'Saved child actual exit/drain binding')
        retirement_verification=retirement.verify_recorded_history(plan,owner['history'],p)
        events=[];rawhash=hashlib.sha256();rawbytes=0
        with gzip.open(out/'events.jsonl.gz','rb')as f:
            while True:
                guard();line=f.readline(8*c.MIB+1)
                if not line:break
                rawbytes+=len(line);c.require(rawbytes<=64*c.MIB and len(line)<=8*c.MIB and line.endswith(b'\n'),'Complete bounded events')
                rawhash.update(line);events.append(json.loads(line));c.require(len(events)<=event_protocol.MAX_EVENTS,'Finite event count')
        if a.exit_code==0:
            c.require(not any(x in values for x in ('failure.json','worker-failure.json'))and values['child-exit.json']['returncode']==0,'Successful producer and child without failure')
            result=values['result.json'];worker=values['worker-result.json'];receipt=values['metadata-result.json'];pkg=values['packages/sources-result.json']
            c.require(result['complete']and result['plan']==pr and result['worker_result']==evidence['worker-result.json']and worker['metadata_result']==evidence['metadata-result.json']and receipt['sources_result']==evidence['packages/sources-result.json'],'Whole receipt chain')
            binding.validate_result(pkg,n,plan['archive_mode'],plan.get('replay_from'))
            event_protocol.validate_success(events,n['packages'],plan['archive_mode'],pkg,evidence['packages/sources-result.json'])
            ev=worker['events'];c.require(ev['raw_bytes']==rawbytes and ev['raw_sha256']==rawhash.hexdigest()and ev['events']==len(events)and all(ev[k]==evidence['events.jsonl.gz'][k]for k in ('path','bytes','sha256')),'Full durable journal matches receipt')
            for row,saved in zip(n['packages'],pkg['packages']):
                c.require(row['package']==saved['package']and row['version']==saved['version'],'Ordered exact package versions')
                if plan['archive_mode']=='download':c.require(saved['inspection']['archive']==evidence['packages/'+row['package']+'.deb'],'Full stored archive hash tied to result')
        else:c.require('failure.json'in values,'Actual failure retained')
        c.verify_sources(plan['source_sha256'],guard);guard()
        value=dict(schema='minicpm5_sdk310_deb_stage_actual_exit_v1',retirement_verification=retirement_verification,actual_tool=dict(kind='tool_session',session_id=a.session,terminal_chunk=a.chunk,exit_code=a.exit_code),plan=pr,review=rr,output_root=str(out),owner=parent,child_owner=co,owners_gone=True,child_group_empty=True,stage_success_observed=a.exit_code==0,mode=plan['archive_mode'],evidence=evidence,events=len(events),raw_event_bytes=rawbytes,raw_event_sha256=rawhash.hexdigest(),result=evidence.get('packages/sources-result.json'),recorder=recorder,lock_identity=lock,recorder_self_exit_claimed=False,native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False)
        print(json.dumps(c.save(Path(a.destination),value,8*c.MIB)),flush=True)
    finally:os.close(fd)
if __name__=='__main__':main()
