"""Bind an externally observed package session after independent live checks."""
import argparse
import datetime
import json
from pathlib import Path
import sys

R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031'); Q=R/'research'
sys.path[:0]=[str(Q),str(R)]
from qwen35_package_serial_runtime_v6 import io, contract, adapter, history
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest

def main():
    assert sys.dont_write_bytecode and __debug__
    parser=argparse.ArgumentParser()
    parser.add_argument('--session-id',type=int,required=True)
    parser.add_argument('--binding-sha256',required=True)
    args=parser.parse_args(); assert args.session_id>0
    binding_ref=dict(path=str(Q/'qwen35-package-full-r4-root-binding-20261112-v1.json'),sha256=args.binding_sha256)
    binding=io.read(binding_ref)
    assert binding['passed'] and binding['full_source_history_resolution_passed']
    spec=io.read(binding['spec'],64*2**20)
    root=contract.control_root(spec)
    admission_ref=io.ref(root/'admission.json'); admission=io.read(admission_ref)
    owner_ref=io.ref(root/'coordinator-owner.json'); owner=io.read(owner_ref)
    adapter.admit_shape(admission,spec,binding['spec'],binding['review'])
    assert io.process_api().proc(admission['outer_owner']['pid'])==admission['outer_owner']
    assert io.process_api().proc(owner['process']['pid'])==owner['process']
    descriptor=io.descriptor(admission['outer_owner'],admission['retained_fd'])
    service=adapter.coordinator_service(spec,admission_ref,owner)
    service.live(); io.process_api().active_heavy((service.unit,))
    previous_ref=admission['freshness']['latest_handoff']; previous=io.read(previous_ref)
    assert latest.latest_handoff()==dict(path=previous_ref['path'],identity=io.identity(previous_ref['path']))
    assert not previous['running'] and not previous['live_tool_sessions']
    tool=dict(kind='tool_session',session_id=args.session_id)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    running=history.expected_running(previous,now,tool,admission_ref,admission,owner_ref,owner)
    running_ref=io.once(spec['running_handoff_path'],running)
    live_ref=io.once(Q/'qwen35-package-full-r4-root-live-20261112-v1.json',dict(
        schema='qwen35_package_full_root_live_v1',reviewer='/root',observed_at=now,
        actual_tool=tool,root_binding=binding_ref,outer_owner=admission['outer_owner'],
        coordinator_owner=owner_ref,running_handoff=running_ref,descriptor_proof=descriptor,
        ack_source=io.ref(Path(__file__).resolve()),device_execution_not_yet_claimed=True,
        full_model_qualified=False,new_speedup_claim=False))
    print(json.dumps(dict(running=running_ref,root_live=live_ref,actual_tool=tool)))

if __name__=='__main__': main()
