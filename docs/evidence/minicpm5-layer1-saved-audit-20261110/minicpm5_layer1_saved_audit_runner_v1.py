"""Separate serial saved-data audit, admitted only after an actual collector exit.

No model reexecution or subprocess launch of model/native code. Source tests may
exercise the pure terminal validator; run() is a direct exclusive entrypoint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
Q=Path(__file__).resolve().parent;ROOT=Q.parent
sys.path[:0]=[str(ROOT),str(Q)]
import minicpm5_original32_layer1_observation_cpu_runner_v1 as collector
from research import minicpm5_layer1_numeric_audit_v1 as numeric
from research import minicpm5_layer1_payload_io_v1 as payload
p=collector.p
require=payload.require
SELF=str(Path(__file__).resolve())
FILES=dict(admission='admission.json',intent='launch-intent.json',drains='predecessor-drain.json',
 coordinator_owner='coordinator-owner.json',rank0_owner='rank0-owner.json',rank1_owner='rank1-owner.json',
 rank0_result='rank0-result.json',rank1_result='rank1-result.json',children='children-exits.json',
 coordinator='coordinator-result.json',service_exit='service-exit.json',outer='outer-result.json',cleanup='cleanup-drained.json')
LIMITS=dict(address_space_bytes=2*2**30,cpu_seconds=600,cpu_hard_seconds=605,wall_seconds=900,
 file_bytes=2*2**20,allowed_cpus=[6],nice=19)


def validate_chain(ended,live,spec,review,plan,records,refs):
    """Validate independently observed completion; this never proves live drain."""
    require(ended['schema']=='minicpm5_layer1_collector_actual_exit_v1'
        and ended['reviewer']=='/root' and ended['actual_exit_observed']is True
        and type(ended['actual_exit_code'])is int and ended['actual_exit_code']==0,
        'Actual externally observed successful collector exit')
    require(live['schema']=='minicpm5_layer1_collector_actual_live_v1'
        and live['reviewer']=='/root' and live['actual_identity_verified']is True,
        'Independent original live collector identity')
    handle=ended['actual_execution_handle']
    require(set(handle)=={'kind','session_id'} and handle['kind']=='tool_session'
        and type(handle['session_id'])is int and handle['session_id']>0
        and handle==live['actual_execution_handle']
        and type(ended['tool_chunk'])is str and re.fullmatch('[a-f0-9]{6,64}',ended['tool_chunk']),
        'Actual external tool session and terminal chunk')
    require(ended['spec']==live['spec'] and ended['review']==live['review']
        and ended['owner']==live['owner'],'Original live/exit bindings unchanged')
    expected=[collector.PYTHON,'-B',collector.SELF,'--mode','outer']
    for key in ('spec','review'):
        expected+=['--'+key,ended[key]['path'],'--'+key+'-sha256',ended[key]['sha256']]
    require(ended['owner']['argv']==expected,'Actual direct collector argv')
    require(spec['schema']=='minicpm5_layer1_observation_cpu_execution_spec_v1'
        and spec['runner']==collector.SELF
        and spec['worker']==str(ROOT/'minicpm5_original32_layer1_observation_cpu_worker_v1.py'),
        'Exact original observation collector')
    require(review['schema']=='minicpm5_layer1_observation_runtime_source_review_v1'
        and review['reviewer']=='/root' and review['passed']is True and review['blockers']==[]
        and review['spec']==ended['spec'] and review['source_sha256']==spec['source_sha256'],
        'Exact concrete original source review')
    require(plan['source_sha256']==spec['source_sha256'] and plan['output_root']==spec['output_root']
        and spec['resources']==collector.PROPS and spec['output_cap_bytes']==collector.CAP
        and spec['root_floor_bytes']==28*2**30 and spec['output_floor_bytes']==30*2**30,
        'Original scientific plan and resource contract')
    require(re.fullmatch('trn1-mini-layer1-observation-[a-z0-9-]{1,80}',spec['unit'])is not None,
        'Original finite owned service namespace')
    require(spec['model_qualification_claim']is spec['automatic_full_gates_launch']is False,
        'Original collection remains diagnostic')
    root=Path(spec['output_root'])
    require(root.parent==Path('/mnt/nvme')
        and root.name.startswith('minicpm5-original32-layer0-full-bf16-cpu-layer1-'),
        'Exact collector output namespace')
    require(set(records)==set(refs)==set(FILES) and ended['outputs']==refs,
        'Complete externally bound collector record catalog')
    for key,name in FILES.items():
        require(refs[key]['path']==str(root/name),'Exact collector metadata path: '+key)
    a=records['admission'];outer=records['outer'];service=records['service_exit']
    c=records['coordinator_owner'];ranks=[records['rank'+str(i)+'_owner']for i in (0,1)]
    require(a['schema']=='minicpm5_layer1_observation_cpu_admission_v1'
        and a['spec']==ended['spec'] and a['review']==ended['review'] and a['plan']==spec['plan']
        and a['outer_owner']==ended['owner'] and a['lock_identity']==ended['lock_identity']
        and a['runtime_admitted']is a['predecessors_checked']is True,
        'Actual original admission and serial owner')
    require(a['observed_drains']==refs['drains'] and a['complete_history_services']==1255
        and records['drains']['complete']is True and records['drains']['services']==1255
        and len(records['drains']['observations'])==1255
        and all(len(row['samples'])==2 for row in records['drains']['observations']),
        'All original predecessor double observations retained')
    require(records['intent']['admission']==refs['admission']
        and records['intent']['command']==collector.service_command(spec,refs['admission'],a['attempt_id']),
        'Exact durable original service launch')
    require(len({(o['pid'],o['start_ticks'])for o in (ended['owner'],c,*ranks)})==4,
        'Four distinct actually observed collector processes')
    expected_group='0::'+p.GROUP_PREFIX+spec['unit']+'.service'
    require(c['unit']==spec['unit'] and c['cgroup']==expected_group
        and c['argv']==collector.argv('coordinator',refs['admission']),
        'Exact coordinator identity')
    collector.validate_service(spec,refs['admission'],a['attempt_id'],service['properties'],c)
    require(service['owner']==c and service['observed_by_outer']==ended['owner']
        and service['successful']is True and service['properties']['Result']=='success'
        and service['properties']['ExecMainCode']=='1' and service['properties']['ExecMainStatus']=='0'
        and p.terminal(service['properties']),'Actual successful terminal service')
    for i,owner in enumerate(ranks):
        require(owner['ppid']==c['pid'] and owner['cgroup']==expected_group
            and owner['argv']==collector.argv('rank',refs['admission']),
            'Original direct rank identity')
        result=records['rank'+str(i)+'_result']
        require(result['process']==owner and result['rank']==i and result['complete']is True
            and result['admission']==refs['admission'] and result['self_exit_claimed']is False,
            'Original completed rank output')
    children=records['children'];coordinator=records['coordinator']
    require(children['complete']is children['successful']is True and children['exit_codes']==[0,0]
        and children['owners']==ranks and coordinator['complete']is True
        and coordinator['rank_exits']==[0,0] and coordinator['rank_owners']==ranks
        and coordinator['results']==[records['rank0_result'],records['rank1_result']]
        and coordinator['self_exit_claimed']is False,'Actual children exits and exact coordinator results')
    require(outer['complete']is True and outer['owner']==ended['owner']
        and outer['service_exit']==refs['service_exit'] and outer['coordinator_result']==refs['coordinator']
        and outer['self_exit_claimed']is False and outer['raw_payload_reconstruction_pending']is True,
        'Outer completion does not self-certify exit or payload audit')
    cleanup=records['cleanup']
    require(cleanup['physically_drained']is True and cleanup['owner']==ended['owner']
        and cleanup['original_failure']is None and cleanup['owners_checked']==[c,*ranks]
        and ended['all_four_processes_gone']is ended['cgroup_empty']is True,
        'Completed cleanup plus independent external physical observations')
    for value in (a,outer):
        require(value['full_model_qualified']is value['new_speedup_claim']is False,
            'No performance or qualification promotion')
    return dict(owners=[ended['owner'],c,*ranks],coordinator=c,admission=a,
        admission_ref=refs['admission'],rank_results=[records['rank0_result'],records['rank1_result']])


def source_closure(spec):
    require(type(spec['source_sha256'])is dict and len(spec['source_sha256'])<4096,'Bounded complete source closure')
    for path,digest in spec['source_sha256'].items():
        file=payload.safe_path(path);info=file.lstat()
        require(file.suffix not in ('.pt','.safetensors','.bin','.neff','.so','.whl','.raw','.zlib','.pb')
            and info.st_size<8*2**20,'Original bounded source-only policy')
        require(p.sha(file)==digest,'Every original or audit source pin unchanged')
    for module in tuple(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and str(file).startswith(str(ROOT)+'/'):
            path=str(Path(file).resolve())
            require(spec['source_sha256'].get(path)==p.sha(path),'Every loaded local helper pinned')
    require(spec['source_sha256'].get(SELF)==p.sha(SELF),'Exact audit entrypoint pinned')
    require(not {'torch','numpy','torch_neuronx','neuronxcc','transformers','vllm'}&set(sys.modules),
        'Audit never imports native tensor/model runtime')


def run(spec_ref,review_ref):
    require(sys.dont_write_bytecode and __debug__ and os.getuid()==1000,'Original user and Python -B with assertions')
    p.user_bus();fd,owner,lock=collector.post.external.acquire_lock();started=time.monotonic()
    lease=lambda:p.lock_proof(owner,lock)
    handlers={}
    try:
        lease();p.clear_window();p.active_heavy()
        spec=p.read_ref(spec_ref);review=p.read_ref(review_ref)
        require(spec['schema']=='minicpm5_layer1_saved_audit_execution_spec_v1' and spec['runner']==SELF
            and spec['limits']==LIMITS,'Concrete finite saved-data audit')
        require(review['schema']=='minicpm5_layer1_saved_audit_runtime_review_v1'
            and review['reviewer']=='/root' and review['passed']is True and review['blockers']==[]
            and review['spec']==spec_ref and review['source_sha256']==spec['source_sha256'],
            'Exact independently reviewed saved-data runtime spec')
        source_closure(spec)
        target=payload.safe_path(spec['output'])
        require(target.parent==Q and re.fullmatch('minicpm5-layer1-saved-audit-[a-z0-9-]{1,80}\.json',target.name)
            and not target.exists(),'Fresh bounded audit output')
        ended=p.read_ref(spec['collector_exit']);live=p.read_ref(ended['live'])
        cs=p.read_ref(ended['spec']);cr=p.read_ref(ended['review']);plan=p.read_ref(cs['plan'])
        require(all(spec['source_sha256'].get(path)==sha for path,sha in cs['source_sha256'].items()),
            'All actual collector source pins retained')
        require(spec['canonical_sha256']==cs['canonical_sha256'],'Original canonical state')
        refs=ended['outputs'];records={key:p.read_ref(ref)for key,ref in refs.items()}
        chain=validate_chain(ended,live,cs,cr,plan,records,refs)
        require(lock==ended['lock_identity'],'Same inherited serial lock inode')
        def physical():
            lease();p.clear_window();p.active_heavy()
            for old in chain['owners']:p.gone(old)
            collector.verify_owned_drain(cs,chain['admission_ref'],chain['admission']['attempt_id'],chain['coordinator'],True)
            collector.usage(Path(cs['output_root']),cs['root_floor_bytes'])
        physical()
        bound,old_owners,priors=collector.check_prerequisites(cs,lease)
        p.gone(bound['finalizer_owner']);p.predecessor_drain(bound['catalog'])
        for old in old_owners:p.gone(old)
        history=collector.predecessor_drains(priors,lease)
        require({n:p.sha(ROOT/n)for n in spec['canonical_sha256']}==spec['canonical_sha256'],
            'Canonical files unchanged before real bytes')
        os.sched_setaffinity(0,set(LIMITS['allowed_cpus']))
        if os.getpriority(os.PRIO_PROCESS,0)<LIMITS['nice']:os.nice(LIMITS['nice']-os.getpriority(os.PRIO_PROCESS,0))
        for kind,pair in ((resource.RLIMIT_AS,(LIMITS['address_space_bytes'],)*2),
            (resource.RLIMIT_CPU,(LIMITS['cpu_seconds'],LIMITS['cpu_hard_seconds'])),
            (resource.RLIMIT_FSIZE,(LIMITS['file_bytes'],)*2)):
            resource.setrlimit(kind,pair)
        def interrupted(signum,frame):raise InterruptedError('Saved-data audit interrupted: '+str(signum))
        for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
            handlers[sig]=signal.signal(sig,interrupted)
        signal.alarm(LIMITS['wall_seconds']);last=[-float('inf')]
        def guard():
            lease();p.clear_window()
            for old in chain['owners']:p.gone(old)
            p.empty_group(p.GROUP_PREFIX+cs['unit']+'.service')
            if time.monotonic()-last[0]>=1:
                physical();last[0]=time.monotonic()
        def read_json(path,digest):
            guard();value=p.read_ref(dict(path=path,sha256=digest));guard();return value
        guard();result=numeric.audit_collection(plan,chain['rank_results'],read_json,guard);physical()
        source_closure(spec)
        require({n:p.sha(ROOT/n)for n in spec['canonical_sha256']}==spec['canonical_sha256'],
            'Canonical files unchanged after real audit')
        output=dict(schema='minicpm5_layer1_saved_audit_terminal_v1',complete=True,
            spec=spec_ref,review=review_ref,collector_exit=spec['collector_exit'],owner=owner,lock_identity=lock,
            result=result,predecessor_drain_sha256=collector.digest(history),self_exit_claimed=False,
            requires_separate_runtime_admission=False,full_model_qualified=False,new_speedup_claim=False,
            elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        p.once(target,output);return collector.reference(target)
    finally:
        signal.alarm(0)
        # This audit creates no native/model children. Exceptions stop reads and
        # release only this process's lock; failed audits never write complete.
        collector.post.external.release_lock(fd)
        for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    for name in ('spec','review'):
        cli.add_argument('--'+name,required=True);cli.add_argument('--'+name+'-sha256',required=True)
    args=cli.parse_args()
    print(json.dumps(run(*[dict(path=getattr(args,k),sha256=getattr(args,k+'_sha256'))for k in ('spec','review')]),indent=2))
