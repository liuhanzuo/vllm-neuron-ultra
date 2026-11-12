"""Root audit for the observed r4 failure before SDK start, not a general runner."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = R / 'research'
sys.path[:0] = [str(Q), str(R)]
from qwen35_sampler_runtime_io_v12 import read, ref, once, process_api
from research import minicpm5_formal_literal_codec_v1 as codec

assert sys.dont_write_bytecode and __debug__
prefix = 'minicpm5-formal-import-r4-20261112-'
live_ref = ref(Q / (prefix + 'root-live.json'))
assert live_ref['sha256'] == '9c5d9e2c0b1426ed300daaefebc97432c645e1da95004d39f8430978f790f8ab'
live = read(live_ref)
wait_ref = ref(Q / (prefix + 'launcher-wait.json'))
waited = read(wait_ref)
actual = dict(kind='tool_session', session_id=95781, terminal_chunk='270de1', exit_code=1)
assert live['actual_tool'] == dict(kind='tool_session', session_id=95781)
assert waited['pid'] == live['owner']['pid'] == 855446
assert waited['actual_returncode'] == 1 and waited['popen_wait_observed'] is True
assert waited['self_exit_claimed'] is False
review = read(live['root_review'])
outer_spec = read(review['outer_spec'])
output = Q / 'minicpm5-fixed-input-pv-sdk-import-formal-r4-20261112'
assert sorted(x.name for x in output.iterdir()) == ['admission.json']
admission_ref = ref(output / 'admission.json')
wire = (output / 'admission.json').read_bytes()
literal = Path(codec.LITERAL['path']).read_text()
admission = codec.decode('admission', wire, literal, wire_sha256=admission_ref['sha256'])
assert len(wire) == 43273 < codec.RUNTIME_CAP
assert admission['lease']['outer_owner'] == live['owner']
assert admission['source_entry_binding']['outer_spec'] == review['outer_spec']
assert admission['source_entry_binding']['outer_review'] == review['outer_review']
assert admission['lease']['unit'] == outer_spec['unit'] == 'trn1-mini-formal-import-r4-20261112'
pins = read(review['complete_source_map'], 16 * 2**20)
assert len(pins) == 48169
for name in ('minicpm5_sdk_mapping_outer_source_entry_v7.py',
             'minicpm5_sdk_mapping_observed_adapter_v4.py',
             'minicpm5_pv_sdk_import_diagnostic_v1.py',
             'minicpm5_sdk_mapping_source_adapter_v2.py',
             'minicpm5_layer1_operator_native_outer_support_v3.py'):
    assert pins[str(Q / name)] == ref(Q / name)['sha256']
proof_ref = ref(Q / 'minicpm5-formal-r4-prerequisite-failure-source-review-20261112-v1.json')
assert proof_ref['sha256'] == '88e66efe7ba38f938286f185ecbd05811f112d0f0d42b7abc8d8586e6a29b0b3'
reproduction = json.loads(Path('/tmp/minicpm5-formal-r4-root-reproduction-20261112.json').read_text())
assert reproduction['original_first_conjunct_reproduced_false']
assert reproduction['original_three_constructors_and_auth_with_mocks_start_no_child']
log_ref = ref(Q / (prefix + 'tool.log'))
log = Path(log_ref['path']).read_text()
assert 'line 163, in source_entry\n    ports.authenticate_complete_prerequisites()' in log
assert log.endswith('ValueError: Actual original authenticated outer and exclusive admission\n')

p = process_api()
p.user_bus()
unit = admission['lease']['unit']
observations = []
fd = os.open(R / 'serial-enrollment.lock', os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
try:
    assert (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (66306, 8294421)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for _ in range(2):
        active = p.active_heavy()
        p.gone(live['owner'])
        props = p.properties(unit)
        assert p.terminal(props) and props['LoadState'] == 'not-found' and not props['InvocationID']
        group = p.empty_group(p.GROUP_PREFIX + unit + '.service')
        observations.append(dict(active_heavy=active, outer_pid_observation=p.proc(855446),
                                 properties=props, recursive_group=group))
    assert observations[0] == observations[1]
finally:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)

result = dict(schema='minicpm5_formal_r4_prestart_root_actual_exit_v1', reviewer='/root',
    observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), actual_tool=actual,
    actual_tool_exit_observed=True, actual_outer_exit_observed=True, root_live=live_ref,
    launcher_wait=wait_ref, root_review=live['root_review'], admission=admission_ref,
    admission_wire_bytes=len(wire), log=log_ref, failure_source_review=proof_ref,
    root_reproduction=reproduction, owners=[live['owner']], units=[unit],
    owners_gone=True, physically_drained=True, two_current_service_observations=observations,
    original_lock_reacquired_and_released=True, lock_identity=dict(device=66306,inode=8294421),
    failure='First complete admission equality rejects two tuple/list identity fields after JSON roundtrip.',
    sdk_child_started=False, observer_started=False, sdk_import_completed=False,
    child_exit_receipt_expected=False, remaining_conjuncts_evaluated=False,
    original_cleanup='No-child drain proof, then original descriptor close; independent current observations above.',
    model_executed=False, full_model_qualified=False, native_ABI_qualified=False,
    new_speedup_claim=False, auditor_source=ref(Path(__file__).resolve()))
receipt = once(Q / 'minicpm5-formal-r4-root-actual-exit-20261112-v1.json', result)
print(json.dumps(dict(result=receipt, actual_tool=actual, physically_drained=True, sdk_child_started=False)))
