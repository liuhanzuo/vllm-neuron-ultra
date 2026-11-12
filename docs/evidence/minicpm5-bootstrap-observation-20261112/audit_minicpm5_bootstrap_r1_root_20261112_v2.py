"""Root audit of the observed 88090/707c04/0 terminal; saved data and live drain."""
import datetime
import fcntl
import hashlib
import stat
import json
import os
from pathlib import Path
import sys
import time

Q = Path(__file__).resolve().parent
ROOT = Q.parent
sys.path[:0] = [str(Q), str(ROOT)]
import dory_serial_stage_contract_v2 as c
import minicpm5_layer0_execution_protocol_v4 as p
import minicpm5_bootstrap_observation_v1 as core
import minicpm5_sdk_mapping_native_closure_v1 as closure
import run_dory_serial_stage_root_v5 as life
import run_minicpm5_bootstrap_observation_v1 as runner

OUT = Path('/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-bootstrap-r1-20261112')


def ref(path):
    return c.reference(path, 8 * 2**20)[0]


def main():
    receipt = dict(kind='tool_session', session_id=88090, terminal_chunk='707c04', exit_code=0)
    result_ref = ref(OUT/'result.json')
    assert result_ref['sha256'] == 'f483526d13f00ae442db6cecc1114fb87da8b9b5315b54ecdc8d8e4dc5705902'
    result = c.load(result_ref)
    assert result['complete'] is True and result['role_order'] == ['sdk310', 'outer312']
    plan = c.load(result['plan'])
    review = c.load(result['review'])
    core.validate(plan, review, result['plan'])
    core.verify_sources(plan['source_sha256'], c.reference)
    saved_owner = c.load(ref(OUT/'owner.json'))
    live = c.load(ref(Q/'minicpm5-bootstrap-observation-r1-root-live-20261112-v1.json'))
    assert live['owner'] == result['owner'] == saved_owner['owner']
    assert saved_owner['deadline'] == result['deadline'] == saved_owner['started_monotonic'] + 900
    content = c.load(result['candidate_content'])
    table = core.candidates(c.load)
    owners = [result['owner']]
    summaries = {}
    child_exits = []
    for entry in result['children']:
        role = entry['role']
        value = c.load(entry['result'])
        admission = c.load(value['admission'])
        owner = c.load(ref(OUT/role/'owner.json'))['owner']
        ended_ref = ref(OUT/role/'exit.json')
        ended = c.load(ended_ref)
        assert ended['returncode'] == 0 and ended['parent_observed'] is True
        assert ended['entire_process_group_empty'] is True and ended['owner'] == owner
        assert value['complete'] is True and value['role'] == role
        assert value['parent_owner'] == result['owner'] and value['pid'] == owner['pid']
        assert value['argv'] == owner['argv']
        assert value['deadline'] == result['deadline'] == admission['deadline']
        assert value['plan'] == result['plan'] and admission['plan'] == result['plan']
        assert set(value['maps']) == {'earliest', 'pre_ctypes', 'post_ctypes'}
        for stage, mapping_ref in value['maps'].items():
            path = Path(mapping_ref['path'])
            assert path.parent == OUT/role and path.suffix == '.maps' and path.resolve() == path
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                before = os.fstat(fd)
                assert stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= 2**20
                raw = b''
                while len(raw) < before.st_size:
                    part = os.read(fd, min(65536, before.st_size-len(raw)))
                    assert part
                    raw += part
                assert os.read(fd, 1) == b''
                assert c.identity(before) == c.identity(os.fstat(fd)) == c.identity(path.lstat())
            finally:
                os.close(fd)
            assert hashlib.sha256(raw).hexdigest() == mapping_ref['sha256'] and len(raw) == mapping_ref['bytes']
            replay = core.summarize_maps(raw.decode(), role, table, content, closure,
                                         lambda path: str(Path(path).resolve()))
            assert replay == value['map_summaries'][stage], (role, stage)
        for path, source in value['executed_sources'].items():
            assert ref(path)['sha256'] == source['sha256']
        for key in ('accepted_native_authority_created', 'native_ABI_qualified', 'full_model_qualified', 'new_speedup_claim'):
            assert value[key] is False and result[key] is False
        owners.append(owner)
        child_exits.append(ended_ref)
        summaries[role] = {key: value[key] for key in ('selected_stdlib_providers', 'map_summaries',
                            'known_pure_source_observations', 'ffi_failure', 'missing_content_origins')}
    assert len(owners) == 3
    assert (OUT/'sdk310/exit.json').stat().st_mtime_ns < (OUT/'outer312/admission.json').stat().st_mtime_ns
    assert int(owners[1]['start_ticks']) < int(owners[2]['start_ticks'])
    hist = c.load(saved_owner['history'])
    rows = []
    for part in hist['parts']:
        data = c.load(part['reference'])
        assert part['first_row'] == data['first_row'] == len(rows)
        assert part['row_count'] == len(data['rows'])
        rows.extend(data['rows'])
    assert len(rows) == hist['row_count'] == len(plan['predecessor_services'])
    inventory = runner.inventory(OUT)
    assert inventory['files'] == result['output_inventory']['files'] + 1
    assert inventory['metadata_bytes'] == result['output_inventory']['metadata_bytes'] + result_ref['bytes']
    assert inventory['raw_bytes'] == result['output_inventory']['raw_bytes']
    assert not list(OUT.rglob('*failure*')) and not list(OUT.rglob('*pending*'))
    observations = []
    for _ in range(2):
        for owner in owners + plan['predecessor_owners']:
            p.gone(owner)
        groups = {str(owner['pid']): life.group_members(owner['pid']) for owner in owners[1:]}
        assert all(not members for members in groups.values())
        observations.append(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 owners_gone=len(owners), predecessor_owners_gone=len(plan['predecessor_owners']),
                                 child_groups=groups))
        time.sleep(.05)
    fd = os.open(ROOT/'serial-enrollment.lock', os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        st = os.fstat(fd)
        assert (st.st_dev, st.st_ino) == (66306, 8294421)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    value = dict(schema='minicpm5_bootstrap_observation_root_actual_exit_v1', reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), actual_tool=receipt,
        actual_exit_code=0, actual_tool_exit_observed=True, result=result_ref,
        plan=result['plan'], review=result['review'], owners=owners, units=[], child_exits=child_exits,
        stage_success_observed=True, owners_gone=True, child_group_empty=True, physically_drained=True,
        drain_observations=observations, lock_reacquired_and_released=True,
        lock_identity=dict(device=66306, inode=8294421), history_rows=len(rows), history_shards=len(hist['parts']),
        full_six_raw_maps_replayed=True, selected_providers_and_gaps=summaries, output_inventory=inventory,
        source_sha256=plan['source_sha256'], auditor_source=ref(Path(__file__)),
        accepted_native_authority_created=False, native_ABI_qualified=False,
        full_model_qualified=False, new_speedup_claim=False)
    print(json.dumps(c.save(Q/'minicpm5-bootstrap-observation-r1-root-actual-exit-20261112-v1.json', value)), flush=True)


if __name__ == '__main__':
    main()
