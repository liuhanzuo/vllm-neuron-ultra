"""Serial read-only timing of the exact released v24 pre-exec bundle.

No server/model launch, no frozen-source patch, no new qualification. A first
untraced call and a second line-traced call distinguish total time and location.
Trace overhead and filesystem cache are explicit limitations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

Q=Path(__file__).resolve().parent
ROOT=Q.parent
sys.path[:0]=[str(Q),str(ROOT)]
import nemotron9b_package_protocol_v24 as p
from nemotron9b_package_controller_v18 import acquire,release
from review_dory_layer0_serial_snapshot_v1 import physical_drain
from dory_generation_execution_provenance_v1 import review_source_pins
from run_dory_layer0_serial_v2 import free,memory_free
from nemotron9b_r4_terminal_metadata_v2 import load,ref,require

OUT=ROOT/'nemotron9b-preexec-timing-20261110-r1'
FORBIDDEN={'torch','numpy','transformers','vllm','neuronxcc','torch_neuronx'}
BUDGET=dict(wall_seconds=1800,cpu_seconds=1500,address_space_bytes=2*2**30,
            output_bytes=32*2**20,single_file_bytes=16*2**20,
            root_floor=28*2**30,nvme_floor=30*2**30,memory_floor=2*2**30,
            cpus=list(range(8)),nice=10)

def measure(fn,args,*,lines=False):
    require(sys.gettrace() is None,'No existing trace hook')
    timing=dict(line_trace_enabled=lines,lines={})
    last=[None,None]
    code=fn.__code__
    def trace(frame,event,arg):
        if frame.f_code is not code:return None
        now=time.monotonic()
        if event in ('line','return'):
            if last[0] is not None:
                row=timing['lines'].setdefault(str(last[0]),dict(visits=0,seconds=0.))
                row['visits']+=1;row['seconds']+=now-last[1]
            last[:]=[frame.f_lineno,now] if event=='line' else [None,None]
        return trace
    started,cpu=time.monotonic(),time.process_time()
    try:
        if lines:sys.settrace(trace)
        value=fn(*args)
        return value,timing
    finally:
        sys.settrace(None)
        timing.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)

def execute(spec,spec_ref,review_ref):
    require(sys.executable==p.PYTHON and sys.dont_write_bytecode and __debug__,'Original Python 3.10 -B')
    require(spec['schema']=='nemotron9b_preexec_timing_v1' and spec['output']==str(OUT)
            and spec['budget']==BUDGET and spec['target_function']=='nemotron9b_package_protocol_v24.bundle',
            'Exact read-only diagnostic scope')
    review=load(review_ref)
    require(review['reviewer']=='/root' and review['passed'] is True and review['blockers']==[]
            and review['spec_sha256']==spec_ref['sha256']
            and review['bootstrap_source_sha256']==spec['bootstrap_source_sha256'],'Exact root source review')
    require(str(Path(__file__).resolve()) in spec['bootstrap_source_sha256'], 'Timing entrypoint source is bound')
    require(not FORBIDDEN & set(sys.modules),'No model or SDK imports')
    p.low.user_bus()
    fd,owner,lock=acquire()
    started=time.monotonic();handlers={};created=False;written=0;checks=0
    def guard():
        nonlocal checks
        p.low.lock_proof(owner,lock);p.low.clear_window();p.low.active_heavy()
        require(time.monotonic()-started<BUDGET['wall_seconds']
                and free('/')>=BUDGET['root_floor'] and free('/mnt/nvme')>=BUDGET['nvme_floor']
                and memory_free()>=BUDGET['memory_floor'],'Original diagnostic resource bounds')
        require(not FORBIDDEN & set(sys.modules),'No model or SDK import during timing')
        checks+=1
    def save(name,value):
        nonlocal written
        require(name in {'admission.json','predecessor-drain.json','bootstrap-sources.json',
                         'bundle-untraced.json','bundle-traced.json','result.json','failure.json'},
                'Fixed timing outputs')
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        require(len(raw)<=BUDGET['single_file_bytes'] and written+len(raw)<=BUDGET['output_bytes'],
                'Finite output budget')
        with (OUT/name).open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        written+=len(raw)
        return ref(OUT/name)
    def interrupted(signum,frame):raise InterruptedError('Timing diagnostic interrupted')
    def tick(signum,frame):guard()
    try:
        guard()
        require(not OUT.exists() and free('/')>=BUDGET['root_floor']+BUDGET['output_bytes'],
                'Fresh namespace and complete output reservation')
        os.sched_setaffinity(0,set(BUDGET['cpus']));os.setpriority(os.PRIO_PROCESS,0,BUDGET['nice'])
        resource.setrlimit(resource.RLIMIT_AS,(BUDGET['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(BUDGET['cpu_seconds'],BUDGET['cpu_seconds']+5))
        resource.setrlimit(resource.RLIMIT_FSIZE,(BUDGET['single_file_bytes'],)*2)
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupted)
        handlers[signal.SIGALRM]=signal.signal(signal.SIGALRM,tick)
        signal.setitimer(signal.ITIMER_REAL,10,10)
        # Authenticate only the bounded executable bootstrap here. The exact
        # target bundle itself authenticates its complete inherited source map;
        # duplicating it before measurement would add an artificial warm-up.
        boot=review_source_pins(spec['bootstrap_source_sha256'],sorted(spec['bootstrap_source_sha256']),
                               max_total_bytes=spec['bootstrap_read_cap_bytes'],guard=guard)
        ended=load(spec['mini_actual_exit']);result=load(spec['mini_result']);ad=load(result['admission'])
        require(ended['actual_external_exit_observed'] is True and ended['exit_code']==0
                and ended['tool_chunk']=='1a4dcd' and ended['session_id']==37843
                and ended['result']['path']==spec['mini_result']['path']
                and ended['result']['sha256']==spec['mini_result']['sha256']
                and ended['owner']==result['owner']==ad['owner']
                and result['complete'] is True and result['canonical_unchanged'] is True,
                'Actual complete Mini r2 and exact exited owner')
        p.low.gone(ended['owner'])
        history=load(ad['predecessor_history'])
        require(len(history['rows'])==1202 and len(history['processes'])==397
                and history['canonical_sha256']==ad['canonical_sha256'],'Full completed 1202-service history')
        drain=physical_drain(history,[ended['owner']])
        require(drain['complete'] and drain['services']==1202 and drain['processes']==398,'Fresh complete physical drain')
        before=p.canonical();require(before==history['canonical_sha256'],'Frozen canonical files unchanged')
        target=load(spec['target_spec']);load(spec['target_review'])
        require(spec['environment']==p.environment(target,'compile')
                and all(os.environ.get(k)==v for k,v in spec['environment'].items()),'Original compile-child environment')
        guard();OUT.mkdir(mode=0o700);created=True
        dref=save('predecessor-drain.json',drain);bref=save('bootstrap-sources.json',boot)
        aref=save('admission.json',dict(owner=owner,lock=lock,execution_spec=spec_ref,source_review=review_ref,
            predecessor_drain=dref,bootstrap_sources=bref,canonical_sha256=before,
            inherited_target_source_verification_deferred_to_measured_bundle=True,model_or_compiler_execution=False,
            runtime_admitted=True,full_model_qualified=False,new_speedup_claim=False))
        print(json.dumps(dict(owner=owner,admission=aref)),flush=True)
        timings={}
        for label,traced in (('bundle-untraced',False),('bundle-traced',True)):
            guard();print(json.dumps(dict(stage=label,event='start')),flush=True)
            value,timing=measure(p.bundle,(spec['target_spec'],spec['target_review']),lines=traced)
            require(value[0]==target,'Measured original function returns exact original spec')
            guard()
            timing.update(target_spec=spec['target_spec'],target_review=spec['target_review'],
                          original_function_unchanged=True,original_checks_passed=True,
                          independent_server_start_proven=False,full_model_qualified=False,new_speedup_claim=False)
            timings[label]=save(label+'.json',timing)
            print(json.dumps(dict(stage=label,event='complete',wall_seconds=timing['wall_seconds'])),flush=True)
        require(p.canonical()==before,'Canonical unchanged after both timings')
        # Recheck the small bootstrap after the original bundle checked its
        # own large map, preserving both authentication scopes.
        for name,digest in spec['bootstrap_source_sha256'].items():
            require(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'Bootstrap changed during measurement')
        guard()
        result_ref=save('result.json',dict(complete=True,owner=owner,admission=aref,timings=timings,
            checks=checks,elapsed_seconds=time.monotonic()-started,canonical_unchanged=True,
            filesystem_cache_not_reset=True,parent_compile_guard_not_reproduced=True,
            second_call_line_trace_overhead=True,server_identity_and_healthy_not_executed=True,
            model_executed=False,compiler_executed=False,full_model_qualified=False,new_speedup_claim=False))
        print(json.dumps(dict(complete=True,result=result_ref)),flush=True)
    except BaseException as error:
        if created:save('failure.json',dict(owner=owner,error=repr(error),complete=False,full_model_qualified=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        release(fd)
        for sig,handler in handlers.items():signal.signal(sig,handler)

def main():
    parser=argparse.ArgumentParser()
    for n in ('spec','spec-sha256','review','review-sha256'):parser.add_argument('--'+n,required=True)
    args=parser.parse_args()
    sr=dict(path=args.spec,sha256=args.spec_sha256);rr=dict(path=args.review,sha256=args.review_sha256)
    execute(load(sr),sr,rr)

if __name__=='__main__':main()
