"""Bind an explicitly observed writer-pair tool session after live checks.

Only root supplies an actual tool session and an accepted concrete binding.
Import performs no IO, History, lock, process work or queue publication.
"""
import argparse
import datetime
import json
from pathlib import Path
import sys

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = R / 'research'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binding', 'binding-sha256', 'root-live'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--session-id', required=True, type=int)
    args = parser.parse_args(argv)
    if not (sys.dont_write_bytecode and __debug__ and args.session_id > 0):
        raise ValueError('python -B, assertions and actual positive tool session required')
    sys.path[:0] = [str(Q), str(R)]
    from qwen35_writer_pair_serial_runtime_v2 import io, contract, adapter, history
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
    require = contract.require
    binding_ref = contract.reference(dict(path=args.binding, sha256=args.binding_sha256))
    binding = io.read(binding_ref)
    require(binding['schema'] == 'qwen35_writer_pair_root_concrete_binding_v1'
            and binding['reviewer'] == '/root' and binding['passed'] is True
            and binding['full_source_history_resolution_passed'] is True,
            'exact independent complete root binding required')
    spec = io.read(binding['spec'], 64 * 2**20)
    contract.validate_spec(spec, io.read(binding['review'], 64 * 2**20), binding['spec'])
    self_ref = io.ref(Path(__file__).resolve())
    require(spec['source_sha256'].get(self_ref['path']) == self_ref['sha256'],
            'actual root ACK source must be pinned before launch')
    live_path = Path(args.root_live)
    require(live_path.parent == Q and str(live_path) == args.root_live
            and live_path.suffix == '.json' and not live_path.exists() and not live_path.is_symlink()
            and str(live_path) not in spec['source_sha256'] and str(live_path) not in spec['input_extension'],
            'fresh canonical root-live output required')
    root = contract.control_root(spec)
    admission_ref = io.ref(root / 'admission.json'); admission = io.read(admission_ref)
    owner_ref = io.ref(root / 'coordinator-owner.json'); owner = io.read(owner_ref)
    adapter.admit_shape(admission, spec, binding['spec'], binding['review'])
    require(io.process_api().proc(admission['outer_owner']['pid']) == admission['outer_owner']
            and io.process_api().proc(owner['process']['pid']) == owner['process'],
            'both exact outer/coordinator processes must be live')
    descriptor = io.descriptor(admission['outer_owner'], admission['retained_fd'])
    service = adapter.coordinator_service(spec, admission_ref, owner)
    service.live(); io.process_api().active_heavy((service.unit,))
    previous_ref = admission['freshness']['latest_handoff']; previous = io.read(previous_ref)
    require(latest.latest_handoff() == dict(path=previous_ref['path'], identity=io.identity(previous_ref['path']))
            and not previous['running'] and not previous['live_tool_sessions'],
            'exact idle predecessor is still current')
    tool = dict(kind='tool_session', session_id=args.session_id)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    running = history.expected_running(previous, now, tool, admission_ref, admission, owner_ref, owner)
    running_ref = io.once(spec['running_handoff_path'], running)
    live_ref = io.once(live_path, dict(schema='qwen35_writer_pair_root_live_v1', reviewer='/root',
        observed_at=now, actual_tool=tool, root_binding=binding_ref,
        outer_owner=admission['outer_owner'], coordinator_owner=owner_ref,
        running_handoff=running_ref, descriptor_proof=descriptor, ack_source=self_ref,
        device_execution_not_yet_claimed=True, full_model_qualified=False, new_speedup_claim=False))
    print(json.dumps(dict(running=running_ref, root_live=live_ref, actual_tool=tool)))


if __name__ == '__main__':
    main()
