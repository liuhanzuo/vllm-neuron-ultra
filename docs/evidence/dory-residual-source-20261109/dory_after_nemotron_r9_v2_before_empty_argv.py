"""Post-r9 history including the actual failed-release extension and checks.

The r9 model remains failed. This consumes real completed metadata and does
not release, launch, rerun, reclassify or invent any missing process identity.
"""
import copy
from pathlib import Path
import dory_after_nemotron_r9_v1 as old

Q, SPEC, REVIEW, CONTROL = old.Q, old.SPEC, old.REVIEW, old.CONTROL
p, terminal, before, require = old.p, old.terminal, old.before, old.require
captured_processes, extend_observed = old.captured_processes, old.extend_observed


def release_extension(receipt, spec, *, load, read):
    owner = receipt['owner']
    command = owner['argv']
    require(command[2] == str(Q/'release_nemotron_r9_failed_v1.py')
            and '--audit' in command and '--release-failed' in command,
            'Exact actual failed-only release entry')
    # Validate options by exact occurrence, independent of the presentation order.
    for flag, value in (('--spec', SPEC['path']), ('--spec-sha256', SPEC['sha256']),
                        ('--review', REVIEW['path']), ('--review-sha256', REVIEW['sha256'])):
        require(command.count(flag) == 1 and command[command.index(flag)+1] == value,
                'Exact original release spec and review')
    for flag in ('--extension-review', '--extension-review-sha256'):
        require(command.count(flag) == 1, 'One concrete extension review')
    ref = dict(path=command[command.index('--extension-review')+1],
               sha256=command[command.index('--extension-review-sha256')+1])
    review = load(ref)
    require(review['reviewer'] == '/root' and review['passed'] is True
            and review['failure_release_only'] is True and review['full_model_qualified'] is False
            and review['original_spec'] == SPEC and review['original_review'] == REVIEW,
            'Original root review authorizes failure release only')
    start = read(CONTROL/'failed-release-extension-v1-intent.json')
    end = read(CONTROL/'failed-release-extension-v1-result.json')
    require(start['owner'] == end['owner'] == owner and start['extension_review'] == end['extension_review'] == ref
            and start['frozen_sources_unchanged'] is True and end['original_failure_preserved'] is True
            and end['release'] == terminal.ref(CONTROL/'root-release.json')
            and start['original_pre_stop'] == terminal.ref(CONTROL/'external-pre-stop.json')
            and start['original_failed_terminal'] == terminal.ref(CONTROL/'failed-terminal.json'),
            'Actual failed-only extension intent/result/release chain')
    checked = load(review['source_checks'])
    exited = load(review['source_check_exit'])
    require(checked['complete'] is True and checked['source_sha256'] == review['source_sha256']
            and checked['tests_run'] == 14 and checked['real_rejected_links'] == 32
            and exited['owner'] == checked['owner'] and exited['exit_code'] == 0
            and exited['actual_external_exit_observed'] is True,
            'Actual completed strict link checks')
    return review


def operation_owners(ledger, *, load):
    require(ledger['schema'] == 'nemotron_r9_recovery_operations_actual_exits_v1'
            and ledger['reviewer'] == '/root' and len(ledger['operations']) == 4,
            'All four actual serial recovery preparation owners')
    expected = ('inspect_nemotron_r9_failed_v1.py', 'inspect_nemotron_r9_recovery_sources_v1.py',
                'check_nemotron_r9_failed_release_v1.py', 'finalize_nemotron_r9_failed_release_review_v1.py')
    owners = []
    for row, name in zip(ledger['operations'], expected):
        require(row['actual_external_exit_observed'] is True and row['synchronous_tool_exit'] is True
                and row['session_id'] is None and row['exit_code'] == 0
                and isinstance(row['tool_chunk'], str) and row['tool_chunk']
                and row['owner']['argv'] == ['/usr/bin/python3', '-B', str(Q/name)],
                'Actual synchronous recovery exit, no invented session')
        intent = load(row['intent'])
        require(intent['owner'] == row['owner'] and intent['actual_self_exit_claimed'] is False,
                'Recovery exit binds exact saved owner intent')
        owners.append(row['owner'])
    require(len({(x['pid'], x['start_ticks']) for x in owners}) == 4, 'Distinct observed recovery owners')
    return owners


def reconstruct(actual_exit_ref, additional_actual_exits=(), recovery_operations=None, guard=lambda: None):
    require(recovery_operations is not None, 'Actual completed r9 recovery operations are mandatory')
    prior, pins = before.completed()
    require(len(prior['rows']) == 1096 and len(prior['processes']) == 291, 'Complete authenticated pre-r9 history')
    records = []
    def load(ref):
        guard()
        value = terminal.load(ref)
        require(ref['path'] not in pins or pins[ref['path']] == ref['sha256'], 'No historical pin rebinding')
        pins[ref['path']] = ref['sha256']
        # Worker result JSON contains large repeated source manifests. Preserve
        # actual owner observations without retaining all decoded manifests.
        records.append((ref, list(captured_processes(value))))
        return value
    def read(path):
        return load(terminal.ref(path))
    spec = load(SPEC)
    load(REVIEW)
    require(spec['stages'] == list(p.STAGES) and len(p.STAGES) == 52
            and set(spec['units']) == {'controller', *p.STAGES}
            and spec['control_root'] == str(CONTROL), 'Original complete r9 package')
    intent = read(CONTROL/'external-launch-intent.json')
    require(intent['execution_spec'] == SPEC and intent['source_review'] == REVIEW, 'Exact r9 launch')
    code = terminal.validate_actual_exit(load(actual_exit_ref), dict(owner=intent['external_owner']), SPEC, REVIEW, 69950)
    require(code == 1 and len(additional_actual_exits) == 1, 'Actual failed r9 and one reviewed release attempt')
    external = [intent['external_owner']]
    extension = None
    for ref in additional_actual_exits:
        receipt = load(ref)
        extra_code = terminal.validate_actual_exit(receipt, dict(owner=receipt['owner']), SPEC, REVIEW,
                                                   receipt['actual_exit']['session_id'])
        require(extra_code == 0, 'Actual completed failed-only release process')
        extension = release_extension(receipt, spec, load=load, read=read)
        external.append(receipt['owner'])
    pre = read(CONTROL/'external-pre-stop.json')
    require(set(pre['rows']) == set(spec['units']) and pre['auditor'] in external
            and pre['launch_intent'] == terminal.ref(CONTROL/'external-launch-intent.json'),
            'All53 original pre-stop service observations and exited auditor')
    release = read(CONTROL/'root-release.json')
    audit = load(release['audit'])
    require(release['audit']['path'] in (str(CONTROL/'external-audit.json'), str(CONTROL/'external-recovery-audit.json')),
            'Original audit namespace')
    if audit.get('recovery_only'):
        load(audit['original_audit'])
        require(audit['success_not_promoted'] is True and audit['passed'] is False, 'Recovery never promotes success')
    admission = read(CONTROL/'admission.json')
    failed = read(CONTROL/'failed-terminal.json')
    require(not (CONTROL/'terminal.json').exists()
            and admission['controller'] == pre['rows']['controller']['owner']
            and admission['execution_spec'] == SPEC and admission['source_review'] == REVIEW,
            'Original failed admitted controller')
    outcome = terminal.classify(code, release, audit, None, failed, terminal.ref(CONTROL/'external-pre-stop.json'),
                               release['audit'], terminal.ref(CONTROL/'admission.json'))
    require(outcome['failed_release'] is True and outcome['producer_qualified'] is False, 'Retain original failed outcome')
    rows, summaries = [], []
    for stage in ('controller', *p.STAGES):
        row, summary = old.retained_stage(spec, stage, SPEC, REVIEW, p, CONTROL, pre['rows'][stage], read=read)
        rows.append(row)
        summaries.append(summary)
    require(len({x['unit'] for x in rows}) == 53
            and not {x['unit'] for x in rows}.intersection(x['unit'] for x in prior['rows']), '53 unique new services')
    total = 0
    for path in sorted(CONTROL.glob('*.json')):
        total += path.stat().st_size
        require(total <= 2**30, 'Bounded complete terminal metadata')
        read(path)
    ledger = load(recovery_operations)
    operations = operation_owners(ledger, load=load)
    for ref in ledger['source_refs']:
        raw = Path(ref['path']).read_bytes()
        import hashlib
        require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Recorded recovery source unchanged')
        require(ref['path'] not in pins or pins[ref['path']] == ref['sha256'], 'No recovery source rebinding')
        pins[ref['path']] = ref['sha256']
    for path, digest in extension['source_sha256'].items():
        require(path not in pins or pins[path] == digest, 'No extension source rebinding')
        pins[path] = digest
    processes, variants = extend_observed(prior['processes'], records)
    require(all(any((x['pid'], x['start_ticks']) == (o['pid'], o['start_ticks']) for x in processes)
                for o in operations+external), 'Every actual external/recovery owner retained')
    result = copy.deepcopy(prior)
    result.update(schema='dory_post_nemotron_r9_history_v1', rows=prior['rows']+rows, processes=processes,
        complete_count=1149, predecessor_source_sha256=pins, r9_stages=summaries, r9_outcome=outcome,
        r9_actual_exit=actual_exit_ref, r9_additional_actual_exits=list(additional_actual_exits),
        r9_recovery_operations=recovery_operations, r9_observed_owner_variants=variants,
        r9_metadata_inventory_bytes=total, r9_metadata_inventory_cap=2**30,
        canonical_sha256=intent['canonical_sha256'],
        current_physical_release_verified=False, full_model_qualified=False, new_speedup_claim=False)
    return result
