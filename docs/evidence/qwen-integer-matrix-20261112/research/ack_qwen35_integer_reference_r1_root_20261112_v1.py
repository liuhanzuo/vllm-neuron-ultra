"""Root ACK only after observing the supplied actual tool session and live supervisor."""
import argparse
import datetime
import json
from pathlib import Path
import sys

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = R / 'research'
sys.path[:0] = [str(Q), str(R)]
import qwen35_integer_reference_serial_v1 as stage
from qwen35_sampler_runtime_io_v12 import read, ref, once, process_api, descriptor
from qwen35_sampler_runtime_history_v12 import expected_running
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest

assert sys.dont_write_bytecode and __debug__
parser=argparse.ArgumentParser();parser.add_argument('--session-id',type=int,required=True)
args=parser.parse_args();assert args.session_id>0
binding_ref = ref(Q / 'qwen35-integer-reference-r1-root-binding-20261112-v1.json')
binding = read(binding_ref)
spec = read(binding['spec'], 64 * 2**20)
root = Path(spec['output_root'])
admission_ref = ref(root / 'admission.json')
admission = read(admission_ref)
owner_ref = ref(root / 'supervisor-owner.json')
owner = read(owner_ref)
assert admission['spec'] == binding['spec'] and admission['review'] == binding['review']
assert process_api().proc(admission['outer_owner']['pid']) == admission['outer_owner']
assert process_api().proc(owner['process']['pid']) == owner['process']
descriptor(admission['outer_owner'], admission['retained_fd'])
unit = stage.names(spec)['coordinator']
service = stage.Service(unit, stage.supervisor_argv(admission_ref), stage.environment(root), stage.PROPERTIES, root / 'service.log', owner=owner)
service.live()
process_api().active_heavy((unit,))
previous = read(admission['freshness']['latest_handoff'])
assert latest.latest_handoff()['path'] == admission['freshness']['latest_handoff']['path']
tool = dict(kind='tool_session', session_id=args.session_id)
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
running = expected_running(previous, now, tool, admission_ref, admission, owner_ref, owner)
running_ref = once(spec['running_handoff_path'], running)
live_ref = once(Q / 'qwen35-integer-reference-r1-root-live-20261112-v1.json', dict(
    schema='qwen35_integer_reference_root_live_v1', reviewer='/root', observed_at=now,
    actual_tool=tool, root_binding=binding_ref, outer_owner=admission['outer_owner'],
    coordinator_owner=owner_ref, running_handoff=running_ref, ack_source=ref(Path(__file__).resolve()),
    device_execution_not_yet_claimed=True, full_model_qualified=False, new_speedup_claim=False))
print(json.dumps(dict(running=running_ref, root_live=live_ref, actual_tool=tool)))
