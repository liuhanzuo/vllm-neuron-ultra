"""Read-only source/metadata observation; invoke under ordinary observer gate."""
import argparse
import datetime
import json
from pathlib import Path
import sys
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
import nemotron9b_package_protocol_v26 as p
from research.minicpm5_after_nemotron_r11_v1 import SPEC,REVIEW,LIVE_OWNER


def observe(target):
    p.low.user_bus();spec=p.read(SPEC['path'],SPEC['sha256']);root=Path(spec['control_root']);a=p.read(root/'admission.json')
    p.low.lock_proof(p.proc_owner(a['controller']),a['lock_identity'])
    p.require(p.low.proc(LIVE_OWNER['pid'])==LIVE_OWNER,'Original external launcher remains live')
    rows=[];splits=[]
    for stage in spec['stages']:
        if p.kind(stage)not in ('prepare','alias','split'):break
        tail=root/(stage+'-retained-drain.json')
        if not tail.exists():break
        ended=p.read(root/(stage+'-retained-exit.json'));drain=p.read(tail)
        work=p.read(root/(stage+'-worker-result.json'));owner=ended['owner']
        p.require(ended['identity_verified']is ended['successful']is True and drain['physically_drained']is drain['successful_exit']is True,'Actual completed diagnostic')
        p.require(len(drain['samples'])==2 and drain['samples'][0]==drain['samples'][1],'Original double drain observations')
        p.require(work['complete']is work['passed']is True and work['producer']==p.producer(owner)and work['phase']==stage,'Original complete actual worker')
        p.validate_result(spec,stage,work['result'])
        seen=p.snapshot(spec,stage,SPEC,REVIEW,owner)
        p.require(seen['successful']and seen['identity_verified'],'Independent retained actual exit')
        p.low.gone(p.proc_owner(owner));group=p.low.empty_group(p.group(spec,stage))
        rows.append(dict(stage=stage,owner=owner,retained_exit=p.ref(root/(stage+'-retained-exit.json')),
            retained_drain=p.ref(tail),worker_result=p.ref(root/(stage+'-worker-result.json')),
            actual_exit_code=0,actual_owner_gone=True,actual_group=group))
        if p.kind(stage)=='split':
            result=work['result'];checks=[x for call in result['device_checks']for x in call['comparisons']]
            for x in checks:
                exact=x['actual_sha256']==x['expected_sha256']and x['different_bytes']==0 and x['shape']==x['expected_shape']and x['dtype']==x['expected_dtype']
                p.require(x['passed']is exact,'Recorded device verdict and exact byte metadata agree')
            splits.append(dict(stage=stage,job=result['job'],compile_passed=result['compile_passed'],
                primitive_passed=result['primitive_passed'],control_outcome_confirmed=result.get('control_outcome_confirmed'),
                device_exact=result['device_exact'],compiler_error_codes=result['compiler_error_codes'],
                device_comparisons=len(checks),passed_device_comparisons=sum(x['passed']for x in checks),
                device_check_metadata=result['device_checks'],cpu_passed=result['cpu_passed'],raw_manifest=result['raw_manifest']))
    status=p.read(root/'status.json');compile_owner=None
    owner_file=root/'compile-owner.json'
    if owner_file.exists():
        compile_owner=p.read(owner_file)
        p.snapshot(spec,'compile',SPEC,REVIEW,compile_owner,allow_live=True)
    out=dict(schema='nemotron9b_r11_progress_metadata_observation_v4',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        execution_spec=SPEC,source_review=REVIEW,observer_source=p.ref(Path(__file__).resolve()),
        admission=p.ref(root/'admission.json'),controller=a['controller'],controller_live=True,
        controller_holds_original_serial_lock=True,current_status=status,completed_stages=len(rows),
        completed_alias_stages=sum(p.kind(r['stage'])=='alias'for r in rows),completed_split_stages=len(splits),
        total_stages=52,rows=rows,split_results=splits,live_external_session=74703,
        compile_worker_owner=compile_owner,compile_lease_exists=(root/'compile-lease.json').exists(),
        model_startup_qualified=False,real_tensor_payloads_read=False,final_raw_audit_pending=True,
        full_model_qualified=False,new_speedup_claim=False)
    p.require(target.parent==Q and not target.exists(),'Fresh independent metadata observation')
    with target.open('x')as f:json.dump(out,f,indent=2);f.write(chr(10))
    summary={}
    for kind in ('nemotron_projection','nemotron_conv'):
        for variant in ('original','tensor_split'):
            selected=[x for x in splits if x['job']['case']['kind']==kind and x['job']['variant']==variant]
            summary[kind+'_'+variant]=dict(shapes=len(selected),passed=sum(x['passed_device_comparisons']for x in selected),
                comparisons=sum(x['device_comparisons']for x in selected))
    print(json.dumps(dict(observation=p.ref(target),time=out['observed_at'],completed_stages=len(rows),status=status,
        compile_worker=compile_owner,summary=summary)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    observe(Path(args.output))
