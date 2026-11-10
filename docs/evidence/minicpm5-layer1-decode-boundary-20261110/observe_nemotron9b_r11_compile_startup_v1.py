"""Read-only admitted startup/exec metadata observation; run under observer."""
from pathlib import Path
import argparse
import datetime
import json
import sys
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
import nemotron9b_package_protocol_v26 as p
from research.minicpm5_after_nemotron_r11_v1 import SPEC,REVIEW,LIVE_OWNER


def observe(out):
    p.low.user_bus();spec=p.read(SPEC['path'],SPEC['sha256']);root=Path(spec['control_root'])
    admission=p.read(root/'admission.json')
    p.low.lock_proof(p.proc_owner(admission['controller']),admission['lock_identity'])
    p.require(p.low.proc(LIVE_OWNER['pid'])==LIVE_OWNER,'Original launcher live')
    worker=p.read(root/'compile-owner.json');p.snapshot(spec,'compile',SPEC,REVIEW,worker,allow_live=True)
    server=p.read(root/'compile-server-owner.json');ack=p.read(root/'compile-server-ack.json')
    p.require(ack['child']==server and ack['parent']==worker and server['ppid']==worker['pid']
        and server['cgroup']==worker['cgroup'] and ack['execution_spec']==SPEC and ack['source_review']==REVIEW,
        'Original admitted parent/child startup acknowledgment')
    executed=p.read(root/'compile-server-exec.json')
    p.require(executed==dict(original_owner=server,argv=p.server_argv(spec),ack=p.ref(root/'compile-server-ack.json')),
        'Exact original exec intent to reviewed serving command')
    expected=dict(server,argv=p.server_argv(spec));actual=[]
    for _ in range(2):
        current=p.low.proc(server['pid']);p.require(current==expected,'Stable PID/start/parent/group and exact post-exec command')
        actual.append(current)
    p.low.lock_proof(p.proc_owner(admission['controller']),admission['lock_identity'])
    observation=dict(schema='nemotron9b_r11_compile_startup_observation_v1',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),observer_source=p.ref(Path(__file__).resolve()),
        execution_spec=SPEC,source_review=REVIEW,controller=admission['controller'],original_lock_held=True,
        external_session=74703,compile_worker=worker,server_original=server,server_actual=actual,
        ack=p.ref(root/'compile-server-ack.json'),exec_intent=p.ref(root/'compile-server-exec.json'),
        current_status=p.read(root/'status.json'),server_process_started=True,
        actual_preparation_to_serving_exec_verified=True,server_ready_not_proven=True,
        compile_complete_not_proven=True,full_model_qualified=False,new_speedup_claim=False)
    p.require(out.parent==Q and not out.exists(),'Fresh metadata output')
    with out.open('x') as f:json.dump(observation,f,indent=2);f.write(chr(10))
    print(json.dumps(dict(observation=p.ref(out),time=observation['observed_at'],server_pid=server['pid'],worker_pid=worker['pid'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    observe(Path(args.output))
