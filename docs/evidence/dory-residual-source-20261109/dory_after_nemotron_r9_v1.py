"""Reconstruct actual r9 completion after its external tool exits.

Inert source preparation. Never release, restart or qualify an experiment here.
The final r9 files and root-observed exits do not exist until the live run ends.
"""
import copy
from pathlib import Path
import sys

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import nemotron9b_after_journal_compaction_v1 as before
import nemotron9b_package_protocol_v18 as p
import nemotron9b_r4_terminal_metadata_v2 as terminal
from dory_nemotron_r8_retained_failure_v1 import retained_stage

require = terminal.require
SPEC = dict(path=str(Q/'nemotron9b-package-runtime-v18-r9-20261109-spec.json'),
            sha256='9f55222ec65715e72502ebbc954d012454b38e1b17b16d568d4d90a91c25f3ec')
REVIEW = dict(path=str(Q/'nemotron9b-package-runtime-v18-r9-20261109-source-review.json'),
              sha256='77786f706bfb6710f819003b77c3256515e66938dbce240b70bfba6e491581ce')
CONTROL = Q.parent/'nemotron9b-package-r9-20261109-control'


def captured_processes(value):
    """Keep only complete recorded identities; never infer starts from PIDs."""
    if isinstance(value, dict):
        keys = ('pid', 'start_ticks', 'ppid', 'argv', 'cgroup')
        if all(k in value for k in keys):
            require(type(value['pid']) is int and value['pid'] > 0
                    and isinstance(value['start_ticks'], str) and value['start_ticks'].isdigit()
                    and int(value['start_ticks']) > 0 and type(value['ppid']) is int
                    and isinstance(value['argv'], list) and value['argv']
                    and all(isinstance(x, str) for x in value['argv'])
                    and isinstance(value['cgroup'], str), 'Actual complete recorded owner')
            yield {k: value[k] for k in keys}
        for child in value.values():
            yield from captured_processes(child)
    elif isinstance(value, list):
        for child in value:
            yield from captured_processes(child)


def extend_observed(prior, records):
    """Exec may change argv at one PID/start; preserve all observed variants."""
    processes = copy.deepcopy(prior)
    seen = {(x['pid'], x['start_ticks']) for x in processes}
    variants = []
    for source, value in records:
        for owner in captured_processes(value):
            key = owner['pid'], owner['start_ticks']
            if key not in seen:
                processes.append(dict(owner, role='nemotron_r9_recorded_owner'))
                seen.add(key)
            row = dict(source=source, owner=owner)
            if row not in variants:
                variants.append(row)
    return processes, variants


def reconstruct(actual_exit_ref, additional_actual_exits=(), guard=lambda: None):
    # This exact33-directory check is intentionally before any new Dory job.
    old, pins = before.completed()
    require(len(old['rows']) == 1096 and len(old['processes']) == 291,
            'Complete authenticated pre-r9 snapshot')
    records = []
    def load(ref):
        guard()
        value = terminal.load(ref)
        require(ref['path'] not in pins or pins[ref['path']] == ref['sha256'],
                'No historical metadata rebinding')
        pins[ref['path']] = ref['sha256']
        records.append((ref, value))
        return value
    def read(path):
        return load(terminal.ref(path))
    spec, review = load(SPEC), load(REVIEW)
    require(spec['stages'] == list(p.STAGES) and len(p.STAGES) == 52
            and set(spec['units']) == {'controller', *p.STAGES}
            and spec['control_root'] == str(CONTROL), 'Exact complete r9 package')
    intent = read(CONTROL/'external-launch-intent.json')
    require(intent['execution_spec'] == SPEC and intent['source_review'] == REVIEW,
            'Original immutable r9 launch')
    exited = load(actual_exit_ref)
    code = terminal.validate_actual_exit(exited, dict(owner=intent['external_owner']), SPEC, REVIEW, 69950)
    external = [intent['external_owner']]
    for ref in additional_actual_exits:
        receipt = load(ref)
        terminal.validate_actual_exit(receipt, dict(owner=receipt['owner']), SPEC, REVIEW,
                                      receipt['actual_exit']['session_id'])
        command = receipt['owner']['argv']
        require(command[2] == spec['controller'] and '--audit' in command,
                'Exact separately observed r9 auditor; unknown release wrappers require new source')
        for option, value in (('--spec', SPEC['path']), ('--spec-sha256', SPEC['sha256']),
                              ('--review', REVIEW['path']), ('--review-sha256', REVIEW['sha256'])):
            require(command.count(option) == 1 and command[command.index(option)+1] == value,
                    'Auditor exact source/spec binding')
        external.append(receipt['owner'])
    pre = read(CONTROL/'external-pre-stop.json')
    require(pre['launch_intent'] == terminal.ref(CONTROL/'external-launch-intent.json')
            and set(pre['rows']) == set(spec['units']) and pre['auditor'] in external,
            'Every original service and actual exited auditor')
    release = read(CONTROL/'root-release.json')
    require(release['audit']['path'] in (str(CONTROL/'external-audit.json'),
                                        str(CONTROL/'external-recovery-audit.json')),
            'Original r9 audit namespace')
    audit = load(release['audit'])
    if release['failed']:
        require(any('--release-failed' in x['argv'] for x in external[1:]),
                'Actual separately observed explicit failed release')
    if audit.get('recovery_only'):
        original = load(audit['original_audit'])
        require(audit['original_audit']['path'] == str(CONTROL/'external-audit.json')
                and audit['success_not_promoted'] is True and audit['passed'] is False,
                'First audit retained; recovery cannot promote success')
    admission = read(CONTROL/'admission.json')
    require(admission['execution_spec'] == SPEC and admission['source_review'] == REVIEW
            and admission['controller'] == pre['rows']['controller']['owner']
            and admission['canonical_sha256'] == intent['canonical_sha256'], 'Actual original admission')
    end = read(CONTROL/'terminal.json') if (CONTROL/'terminal.json').exists() else None
    failed = read(CONTROL/'failed-terminal.json') if (CONTROL/'failed-terminal.json').exists() else None
    outcome = terminal.classify(code, release, audit, end, failed,
        terminal.ref(CONTROL/'external-pre-stop.json'), release['audit'], terminal.ref(CONTROL/'admission.json'))
    rows, summaries = [], []
    for stage in ('controller', *p.STAGES):
        row, summary = retained_stage(spec, stage, SPEC, REVIEW, p, CONTROL, pre['rows'][stage], read=read)
        rows.append(row)
        summaries.append(summary)
    require(len({x['unit'] for x in rows}) == 53
            and not {x['unit'] for x in rows}.intersection(x['unit'] for x in old['rows']),
            '53 new services without overwriting old history')
    # Read bounded retained JSON only. This preserves recorded child/compiler
    # identities where present; an unrecorded descendant is never invented.
    total = 0
    for path in sorted(CONTROL.glob('*.json')):
        guard()
        total += path.stat().st_size
        require(total <= 128*2**20, 'Bounded complete r9 terminal metadata inventory')
        read(path)
    processes, variants = extend_observed(old['processes'], records)
    result = copy.deepcopy(old)
    result.update(schema='dory_post_nemotron_r9_history_v1', rows=old['rows']+rows,
        processes=processes, complete_count=1149, predecessor_source_sha256=pins,
        r9_stages=summaries, r9_outcome=outcome, r9_actual_exit=actual_exit_ref,
        r9_additional_actual_exits=list(additional_actual_exits), r9_observed_owner_variants=variants,
        r9_metadata_bytes_read=total, canonical_sha256=intent['canonical_sha256'],
        current_physical_release_verified=False, full_model_qualified=False, new_speedup_claim=False)
    return result
