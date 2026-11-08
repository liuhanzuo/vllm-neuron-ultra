"""Pure root admission/ACK/terminal contracts for Phi CPU v6.

No runtime entry point, service operation, lock acquisition, file write, native
import or tensor read. A future root driver must gather current authenticated
evidence and enforce these contracts while holding the real serial lock.
"""
import hashlib
import json
from pathlib import PurePosixPath
import re

ROOT = PurePosixPath('/home/ubuntu/experiments/trn1-latest-models-20261031')
SCHEMA = 'phi4-saved-boundary-alignment-cpu-v7'
SCRIPT = str(ROOT / 'phi4_saved_boundary_cpu_v7.py')
CONTROLLER = 'trn1-latest-verify-' + SCHEMA + '-controller'
WORKER = 'trn1-latest-verify-' + SCHEMA + '-worker'
PREFIX = '/user.slice/user-1000.slice/user@1000.service/app.slice/'
SMOL = 'smollm3-cache-ablation-full-v8'
SMOL_QUEUE = 'trn1-' + SMOL + '-queue'
LFM_PREFIX = 'trn1-latest-verify-lfm3-package-paired-v2-'
FIXED_EXTRA = (
    'trn1-latest-verify-lfm3-package-native-core-v1',
    'trn1-latest-verify-lfm3-package-tokenizer-v1',
    'trn1-latest-verify-minicpm5-hf32-boundary-observer-v1',
    'trn1-latest-verify-minicpm5-hf32-boundary-observer-v2',
)
CANONICAL = ('continuation-state.json', 'roadmap.json', 'best-known-configurations.json')
BOUNDARIES = dict(gateup_output=16384, down_input=8192, down_output=3072,
    layer0_output=3072, layer1_norm=3072, layer1_qkv=5120,
    layer1_oproj_input=3072, final_norm=3072, logits=200064)
VARIANTS = ('instruct', 'reasoning')
ARMS = ('unchanged', 'wrapper', 'candidate')
ANCHORS = ('native_single', 'native_batch', 'port')
GATES = dict(native_single='original_hf', native_batch='native_batch', port='original_port')


def require(value, message):
    if not value: raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(value): return hashlib.sha256(encode(value)).hexdigest()


def nonzero_hex(value, width):
    return type(value) is str and re.fullmatch('[0-9a-f]{' + str(width) + '}', value) is not None and value != '0' * width


def owner_key(owner):
    require(type(owner) is dict and type(owner['pid']) is int and owner['pid'] > 0, 'Owner PID')
    require(type(owner['start_ticks']) is str and owner['start_ticks'].isdigit() and int(owner['start_ticks']) > 0, 'Owner start')
    require(type(owner['unit']) is str and re.fullmatch('[A-Za-z0-9_.@:-]+', owner['unit']) is not None
            and not owner['unit'].endswith('.service'), 'Owner unit')
    script = PurePosixPath(owner['script'])
    regular = script.is_absolute() and script.is_relative_to(ROOT) and '..' not in script.parts and script.suffix == '.py'
    cli = str(script) == '/home/ubuntu/venvs/vllm_neuron/bin/vllm'
    label = owner['unit'].removeprefix(LFM_PREFIX)
    servers = {role+'-'+phase for role in ('reference','candidate') for phase in ('compile','quality-server')}
    servers.update(role+'-workloads-server-'+str(trial) for role in ('reference','candidate') for trial in range(3))
    require(regular or cli and owner['unit'].startswith(LFM_PREFIX) and label in servers, 'Owner source/exact ten LFM CLI units')
    return {key: owner[key] for key in ('pid', 'start_ticks', 'unit', 'script')}


def lfm_pair_labels():
    names = ['reference-' + phase for phase in ('sampler', 'primitive-cpu', 'primitive-device', 'primitive-alias', 'inputs')]
    for role in ('reference', 'candidate'):
        names += [role + '-' + phase for phase in ('compile', 'numeric', 'cold', 'long', 'quality-server', 'quality')]
    names += ['reference-compare']
    for trial in range(3):
        for role in ('reference', 'candidate'):
            names += [role + '-workloads-server-' + str(trial), role + '-workloads-' + str(trial)]
    require(len(names) == len(set(names)) == 30, 'LFM exact planned unit catalog')
    return names


def expected_predecessors(smol_plan):
    require(len(smol_plan['predecessors']) == 170 and len(smol_plan['phases']) == 32, 'Smol exact plan geometry')
    require(smol_plan['queue_unit'] == SMOL_QUEUE, 'Smol queue')
    old = [owner_key(row)['unit'] for row in smol_plan['predecessors']]
    current = [SMOL_QUEUE]
    for phase in smol_plan['phases']:
        name = 'trn1-latest-verify-' + SMOL + '-' + phase['id']
        current.append(name)
        if phase['server']: current.append(name + '-server')
    require(len(current) == len(set(current)) == 45, 'Smol current unit catalog')
    all_units = old + current + list(FIXED_EXTRA) + [LFM_PREFIX + label for label in lfm_pair_labels()]
    require(len(all_units) == len(set(all_units)) == 249, 'Complete fixed predecessor catalog')
    return all_units


def terminal_state(props):
    require(props['MainPID'] == props['ControlPID'] == '0' and not props['Job'].strip(), 'Live predecessor/job')
    require(props['ActiveState'] in ('inactive', 'failed') or
            (props['ActiveState'], props['SubState']) == ('active', 'exited'), 'Nonterminal state pair')


def group_contract(unit, group):
    p = PurePosixPath(group)
    require(p.is_absolute() and p.is_relative_to('/sys/fs/cgroup') and '..' not in p.parts
            and p.name == unit + '.service', 'Group path identity')


def drain_contract(row, observation, allow_historical_missing_invocation=False):
    owner = owner_key(row['owner']); require(row['unit'] == owner['unit'], 'Drain owner unit')
    require(type(row['recorded_cgroups']) is list and row['recorded_cgroups'], 'Complete recorded cgroups required')
    for group in row['recorded_cgroups']: group_contract(row['unit'], group)
    require(observation['owner'] == owner and observation['old_pid_absent'] is True, 'Old PID still present or wrong owner')
    require(len(observation['samples']) == 2 and observation['samples'][0] == observation['samples'][1], 'Unstable double drain')
    sample = observation['samples'][0]; props = sample['properties']; terminal_state(props)
    require(props['Id'] == row['unit'] + '.service', 'Foreign terminal service')
    if nonzero_hex(observation['invocation_id'], 32):
        require(props['InvocationID'] == observation['invocation_id'] or
                (props['LoadState'] == 'not-found' and props['InvocationID'] == ''), 'Terminal invocation changed')
    else:
        require(allow_historical_missing_invocation and observation['invocation_id'] is None
                and observation.get('historical_invocation_unknown') is True
                and props['LoadState'] == 'not-found' and props['InvocationID'] == '',
                'Unrecorded invocation allowed only for explicitly retired legacy unit')
    paths = {str(PurePosixPath('/sys/fs/cgroup') / (PREFIX + row['unit'] + '.service').lstrip('/')),
             *row['recorded_cgroups']}
    if props['ControlGroup']:
        require(props['ControlGroup'] == PREFIX + row['unit'] + '.service', 'Actual cgroup mismatch')
        paths.add('/sys/fs/cgroup' + props['ControlGroup'])
    require(set(sample['groups']) == paths, 'Missing recursive cgroup drain')
    for path, proof in sample['groups'].items():
        group_contract(row['unit'], path)
        require(proof['empty'] is True and proof['recursive'] is True and proof['symlinks_rejected'] is True,
                'Cgroup descendants not proved empty')


def complete_predecessor_contract(plan, rows, drains, canonical_tail, root_review):
    expected = expected_predecessors(plan)
    require(root_review['passed'] is True and root_review['reviewer'] == '/root' and
            root_review['complete_predecessor_catalog'] == expected, 'Root complete predecessor review')
    require(len(rows) == 249 and len(drains) == 249 and [row['unit'] for row in rows] == expected, 'Missing/reordered predecessor')
    launched = [row for row in rows if row['owner'] is not None]
    require(len({(owner_key(row['owner'])['pid'], row['owner']['start_ticks']) for row in launched}) == len(launched), 'Duplicate process identity')
    for old, row in zip(plan['predecessors'], rows[:170]):
        require(owner_key(old) == owner_key(row['owner']), 'Frozen predecessor identity changed')
    require(owner_key(canonical_tail) == canonical_tail and canonical_tail in [owner_key(row['owner']) for row in launched], 'Current canonical tail omitted')
    require(root_review['canonical_tail'] == canonical_tail and root_review['predecessor_contract_sha256'] == digest(rows), 'Root tail/contract mismatch')
    for index, (row, observation) in enumerate(zip(rows, drains)):
        if row['owner'] is None:
            require(index >= 170 and row['never_launched'] is True, 'Historical owner cannot be omitted')
            require(row['absent_records'] and observation['absent_records']==row['absent_records'] and observation['records_still_absent'] is True, 'Never-launched record absence')
            require(observation['owner'] is None and len(observation['samples'])==2 and observation['samples'][0]==observation['samples'][1], 'Never-launched stable proof')
            sample=observation['samples'][0]; props=sample['properties'];terminal_state(props)
            require(props['Id']==row['unit']+'.service' and props['LoadState']=='not-found' and props['InvocationID']=='', 'Never-launched service not absent')
            require(set(sample['groups'])==set(row['recorded_cgroups']) and row['recorded_cgroups'], 'Never-launched groups')
            for path,proof in sample['groups'].items():
                group_contract(row['unit'],path)
                require(proof['empty'] is proof['recursive'] is proof['symlinks_rejected'] is True, 'Never-launched group not drained')
        else:
            require(row['never_launched'] is False, 'Launched owner mislabeled')
            drain_contract(row, observation, allow_historical_missing_invocation=index < 170)
    require(root_review['source_evidence_verified'] is True and root_review['all_planned_launches_accounted_for'] is True,
            'Raw launch/owner/terminal evidence not independently verified')
    return True


def admission_contract(admission, smol_plan, roots, rows, drains, root_predecessor_review):
    require(admission['schema'] == SCHEMA and admission['authorized_by'] == '/root', 'Wrong admission scope')
    require(admission['root_serial_reservation_active'] is True and admission['no_parallel_successor'] is True, 'No exclusive root reservation')
    require(admission['runtime_limits'] == dict(MemoryMax=2**31, MemoryHigh=1536 * 2**20, MemorySwapMax=0, TasksMax=32), 'Changed runtime limits')
    require(admission['core_limits'] == dict(input_read_bytes=4 * 2**30, output_bytes=32 * 2**20,
            single_payload_bytes=128 * 2**20, wall_seconds=1800), 'Changed scientific budget')
    require(admission['output_contract'] == dict(files=['events.jsonl', 'terminal.json'], total_bytes=32 * 2**20,
            per_file_bytes=24 * 2**20, terminal_bytes=8 * 2**20), 'Changed output contract')
    require(set(admission['canonical_sha256']) == set(CANONICAL) and
            all(admission['canonical_sha256'][name] == roots[name]['sha256'] for name in CANONICAL), 'Canonical bytes changed')
    require(roots[CANONICAL[0]]['tail'] == roots[CANONICAL[1]]['tail'], 'Canonical tail disagreement')
    tail = roots[CANONICAL[0]]['tail']
    require(admission['predecessors'] == rows and admission['predecessors_complete'] is True
            and admission['predecessor_count'] == 249 and admission['predecessor_contract_sha256'] == digest(rows)
            and admission['predecessors_complete_through_serial_tail'] == tail, 'Admission predecessor binding')
    require(nonzero_hex(admission['source_review_sha256'], 64), 'Source review binding')
    complete_predecessor_contract(smol_plan, rows, drains, tail, root_predecessor_review)
    return True


def controller_ack(owner, props, process, admission_sha, exact_command, lock):
    require(owner['unit'] == CONTROLLER and type(owner['pid']) is int and owner['pid'] > 0, 'Controller owner')
    require(nonzero_hex(owner['invocation_id'], 32) and props['InvocationID'] == owner['invocation_id'], 'Controller invocation')
    require(props['Id'] == CONTROLLER + '.service' and props['LoadState'] == 'loaded'
            and props['ActiveState'] == 'active' and props['SubState'] == 'running'
            and props['MainPID'] == str(owner['pid']) and props['ControlPID'] == '0', 'Controller live identity')
    require(owner['cgroup'] == props['ControlGroup'] == PREFIX + CONTROLLER + '.service', 'Controller cgroup')
    argv = ['/usr/bin/python3', '-B', '-u', SCRIPT, 'run', admission_sha]
    require(process == dict(pid=owner['pid'], start_ticks=owner['start_ticks'], argv=argv, cgroup='0::' + owner['cgroup']), 'Controller process identity')
    require(exact_command['argv'] == argv and exact_command['cleanup_argv'] == ['/usr/bin/python3', '-B', '-u', SCRIPT, 'cleanup', admission_sha], 'Installed command/cleanup')
    require(exact_command['actual_properties_validated'] is True and exact_command['before_equals_after'] is True, 'Resource identity ACK absent')
    require(lock['held_by_root'] is True and lock['whole_file_write'] is True and lock['actual_inode_matches'] is True
            and lock['controller_has_not_acquired'] is True, 'Root lost handoff lock')
    require(type(lock['elapsed_since_launch_s']) in (int, float) and 0 <= lock['elapsed_since_launch_s'] < 10, 'ACK too late for15s handoff')
    return True


def frame_keys():
    result = []
    for order in ([0, 5, 2, 4], [1, 3, 0, 2]):
        result += [dict(fixture=f, phase='prefill', slot=s) for f, s in zip(order, [3, 1, 0, 2])]
        result += [dict(fixture=f, phase='decode', step=step, slot=s) for step in (1, 2, 3)
                   for f, s in zip(order, [3, 1, 0, 2])]
    return result


def comparison_contract(events, original_gate_reports):
    rows = [row for row in events if row.get('event') == 'comparison']
    expected = [(variant, arm, anchor, ordinal, boundary) for variant in VARIANTS for arm in ARMS
                for anchor in ANCHORS for ordinal in range(32) for boundary in BOUNDARIES]
    require(len(rows) == len(expected) == 5184, 'Incomplete comparison matrix')
    keys = frame_keys(); heads = 0
    for row, key in zip(rows, expected):
        require(tuple(row[name] for name in ('variant', 'arm', 'anchor', 'ordinal', 'boundary')) == key, 'Duplicate/reordered comparison')
        variant, arm, anchor, ordinal, boundary = key
        require(row['key'] == keys[ordinal] and row['difference']['width'] == BOUNDARIES[boundary], 'Frame/feature geometry')
        require(row['selected_prediction_row_only'] is True and row['global_first_cause_proven'] is False, 'Unsupported causal scope')
        if boundary == 'logits':
            heads += 1
            require(row['original_logit_metrics'] == original_gate_reports[variant][arm]['comparisons'][GATES[anchor]][ordinal], 'Original metric row changed')
        else: require(row['original_logit_metrics'] is None, 'Nonhead claimed head metric')
    complete = [row for row in events if row.get('event') == 'complete']
    worker = [row for row in events if row.get('event') == 'worker_complete']
    require(len(complete) == len(worker) == 1 and worker[0]['summary'] == complete[0], 'Missing/duplicate completion')
    summary = complete[0]
    require(summary['comparisons'] == 5184 and summary['head_comparisons'] == heads == 576
            and summary['activation_payloads_verified'] == 104, 'Summary coverage mismatch')
    expected_gates = {variant + '/' + arm: original_gate_reports[variant][arm]['gates'] for variant in VARIANTS for arm in ARMS}
    require(summary['preserved_original_gates'] == expected_gates and sum(len(v) for v in expected_gates.values()) == 18, 'Original18 gates changed')
    require(summary['all_feature_vectors_compared'] is True and summary['duplicate_logical_occurrences_preserved'] is True, 'Incomplete vector/duplicate scope')
    for name in ('model_qualified', 'first_global_cause_proven', 'gateup_cause_proven', 'KV_history_restored', 'new_speedup_claim', 'original_gates_replaced'):
        require(summary[name] is False, 'Expanded scientific claim: ' + name)
    require(worker[0]['no_model_weights_or_device'] is True and worker[0]['full_model_qualified'] is False
            and worker[0]['new_speedup_claim'] is False, 'Expanded worker scope')
    return True


def terminal_contract(admission_sha, terminal, events, events_raw_sha, original_gate_reports, external):
    require(terminal['schema'] == SCHEMA and terminal['admission_sha256'] == admission_sha, 'Terminal admission binding')
    require(terminal['passed'] is True and terminal['cleanup_error'] is None and terminal['journal_error'] is None, 'Failed run')
    require(terminal['events_sha256'] == events_raw_sha and terminal['complete_comparisons'] == 5184
            and terminal['complete_heads'] == 576 and terminal['worker_failures'] == [], 'Terminal evidence mismatch')
    require(terminal['publication_lock_reacquired'] is True and terminal['controller_terminal_external_audit_still_required'] is True,
            'Managed cleanup does not replace external audit')
    require(terminal['service_exit'] == dict(SERVICE_RESULT='success', EXIT_CODE='exited', EXIT_STATUS='0'), 'Controller exit failure')
    require(all(terminal[name] is False for name in ('original_gates_replaced', 'full_model_qualified', 'new_speedup_claim')), 'Terminal scope expanded')
    require(not [row for row in events if row.get('event') in ('worker_failed', 'controller_prelaunch_failed')], 'Failure event retained')
    outcomes = [row for row in events if row.get('event') == 'supervisor_outcome']
    require(len(outcomes) == 1 and terminal['supervisor_outcomes'] == outcomes, 'Supervisor outcome mismatch')
    require(outcomes[0]['returncode'] == 0 and outcomes[0]['error'] is None and outcomes[0]['cleanup_error'] is None
            and outcomes[0]['launcher_reaped'] is True, 'Supervisor failed or unreaped')
    require(external['controller_drained'] is True and external['worker_drained'] is True
            and external['all_predecessors_drained'] is True and external['canonical_unchanged'] is True
            and external['sources_unchanged'] is True and external['root_holds_serial_lock'] is True, 'External terminal audit incomplete')
    require(external['actual_identity_and_recursive_drains_verified'] is True
            and external['oom_timeout_storage_errors'] == [] and external['other_heavy_units'] == [], 'External supervisor error or competing job')
    for key,unit in (('controller_terminal',CONTROLLER),('worker_terminal',WORKER)):
        proof=external[key];owner=proof['owner']
        require(proof['identity_verified'] is proof['physically_drained'] is proof['successful'] is True,'Successful retained service proof')
        require(owner['unit']==unit and nonzero_hex(owner['invocation_id'],32),'Terminal owner scope')
        require(len(proof['samples'])==2 and proof['samples'][0]==proof['samples'][1],'Retained terminal unstable')
        actual=proof['samples'][0]['properties'];terminal_state(actual)
        require(actual['Id']==unit+'.service' and actual['LoadState']=='loaded' and actual['InvocationID']==owner['invocation_id']
                and actual['ExecMainPID']==str(owner['pid']),'Retained actual terminal identity')
        require(actual['Result']=='success' and actual['ExecMainCode']=='1' and actual['ExecMainStatus']=='0','Actual retained service failed')
        require(proof['samples'][0]['cgroups'],'Missing retained recursive group proof')
    require(external['cleanup_observation']['physically_drained'] is True and external['storage_observation']['passed'] is True,'Actual cleanup/storage observations')
    comparison_contract(events, original_gate_reports)
    return True


def release_contract(release, audit):
    require(release['authorized_by'] == '/root' and release['source'] == 'external_terminal_audit', 'Explicit root release required')
    require(audit['passed'] is True and audit['terminal_contract_passed'] is True and audit['runtime_admission_sha256'] == release['runtime_admission_sha256'], 'Audit/admission release mismatch')
    require(release['audit_sha256'] == digest(audit) and release['all_groups_drained'] is True
            and release['canonical_unchanged'] is True and release['no_automatic_successor_launch'] is True, 'Release evidence incomplete')
    return True
