"""Root r2 terminal audit. Supply only an externally observed successful tool exit."""
import argparse
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
import minicpm5_bootstrap_observation_v2 as core
import minicpm5_sdk_mapping_native_closure_v2 as closure
import run_dory_serial_stage_root_v5 as life
import run_minicpm5_bootstrap_observation_v2 as runner

OUT = Path('/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-bootstrap-r2-20261112')


def ref(path):
    return c.reference(path, 8 * 2**20)[0]


def audit_types(role, value, parent_entry):
    """Recompute source gaps separately from the complete native map summaries."""
    observation = value['types_source_observation']
    sources = value['executed_sources']
    recomputed = core.source_execution_gaps(role, observation, sources)
    assert value['source_execution_provenance_gaps'] == recomputed
    assert parent_entry['source_execution_provenance_gaps'] == recomputed
    if role == 'sdk310':
        assert observation is None and recomputed == []
        return
    pin = dict(path='/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/types.py',
               bytes=10993, sha256='345474ef027a1273f353da9bdc1f7c18f65335e72e681bcc0376774cc51f2405')
    assert observation['current_source_reference'] == pin
    executed = observation['executed_through_pinned_loader']
    final = observation['final_provider']
    assert final['present'] is True and final['module_file'] == pin['path']
    assert final['spec_name'] == 'types' and final['spec_origin'] == pin['path']
    assert final['loader_is_spec_loader'] is True
    if executed:
        assert final['loader_type'] == 'TypesLoader'
        assert observation['first_checkpoint']['present'] is False
        assert observation['execution_receipt'] == sources[pin['path']]
        assert recomputed == []
    else:
        assert final['loader_type'] == 'SourceFileLoader'
        assert observation['first_checkpoint']['present'] is True
        assert observation['execution_receipt'] is None and pin['path'] not in sources
        assert recomputed == [dict(path=pin['path'], reason='preexisting_module_execution_source_unproved')]
    assert observation['preexisting_module_reloaded'] is False
    assert observation['interpreter_startup_cause_established'] is False


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--session-id', type=int, required=True)
    cli.add_argument('--terminal-chunk', required=True)
    cli.add_argument('--exit-code', type=int, choices=[0], required=True)
    cli.add_argument('--result-sha256', required=True)
    cli.add_argument('--live', required=True)
    cli.add_argument('--live-sha256', required=True)
    args = cli.parse_args()
    assert __debug__ and args.session_id > 0 and args.terminal_chunk
    receipt = dict(kind='tool_session', session_id=args.session_id,
                   terminal_chunk=args.terminal_chunk, exit_code=args.exit_code)
    result_ref = ref(OUT/'result.json')
    assert result_ref['sha256'] == args.result_sha256
    result = c.load(result_ref)
    assert result['schema'] == 'minicpm5_bootstrap_observation_result_v2'
    assert result['complete'] is True and result['role_order'] == ['sdk310', 'outer312']
    assert [entry['role'] for entry in result['children']] == result['role_order']
    plan = c.load(result['plan'])
    review = c.load(result['review'])
    core.validate(plan, review, result['plan'])
    core.verify_sources(plan['source_sha256'], c.reference)
    saved_owner = c.load(ref(OUT/'owner.json'))
    live_ref = dict(path=args.live, sha256=args.live_sha256)
    live = c.load(live_ref)
    assert live['session_id'] == args.session_id
    assert live['owner'] == result['owner'] == saved_owner['owner']
    assert saved_owner['deadline'] == result['deadline'] == saved_owner['started_monotonic'] + 900
    content = c.load(result['candidate_content'])
    table = core.candidates(c.load)
    literal = core.literal(*(Path(Q/name).read_text() for name in (core.HELPER, core.CHILD, core.TYPES)))
    assert len(literal.encode()) == 51479
    literal_sha = hashlib.sha256(literal.encode()).hexdigest()
    assert plan['output_root'] == str(OUT)
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
        assert value['schema'] == 'minicpm5_bootstrap_role_observation_v2'
        assert value['complete'] is True and value['role'] == role
        assert admission['literal_sha256'] == literal_sha
        assert value['argv'] == core.argv_for(role, table, literal, result['plan'],
                                             value['admission'], int(value['argv'][14]), result['deadline'])
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
            current = ref(path)
            assert current['sha256'] == source['sha256'] and current['bytes'] == source['bytes']
        audit_types(role, value, entry)
        missing = any(row['gaps'] for row in value['map_summaries'].values())
        assert value['missing_content_origins'] is missing and entry['missing_content_origins'] is missing
        assert entry['ffi_definitions_observed'] is (value['ffi_definitions'] is not None)
        for key in ('accepted_native_authority_created', 'native_ABI_qualified', 'full_model_qualified', 'new_speedup_claim'):
            assert value[key] is False and result[key] is False
        owners.append(owner)
        child_exits.append(ended_ref)
        summaries[role] = {key: value[key] for key in ('selected_stdlib_providers', 'map_summaries',
                            'known_pure_source_observations', 'ffi_failure', 'missing_content_origins',
                            'types_source_observation', 'source_execution_provenance_gaps')}
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
    value = dict(schema='minicpm5_bootstrap_observation_root_actual_exit_v2', reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), actual_tool=receipt,
        actual_exit_code=0, actual_tool_exit_observed=True, result=result_ref,
        plan=result['plan'], review=result['review'], owners=owners, units=[], child_exits=child_exits,
        stage_success_observed=True, owners_gone=True, child_group_empty=True, physically_drained=True,
        drain_observations=observations, lock_reacquired_and_released=True,
        lock_identity=dict(device=66306, inode=8294421), history_rows=len(rows), history_shards=len(hist['parts']),
        live=live_ref, literal_bytes=51479, literal_sha256=literal_sha,
        full_six_raw_maps_replayed=True, source_execution_gaps_recomputed=True, selected_providers_and_gaps=summaries, output_inventory=inventory,
        source_sha256=plan['source_sha256'], auditor_source=ref(Path(__file__)),
        accepted_native_authority_created=False, native_ABI_qualified=False,
        full_model_qualified=False, new_speedup_claim=False)
    print(json.dumps(c.save(Q/'minicpm5-bootstrap-observation-r2-root-actual-exit-20261112-v1.json', value)), flush=True)


if __name__ == '__main__':
    main()
