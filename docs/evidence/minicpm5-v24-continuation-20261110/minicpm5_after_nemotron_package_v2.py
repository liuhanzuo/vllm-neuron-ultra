"""Future Mini saved-data predecessor history after completed Nemotron v22/v23/v24.

Metadata only. Requires an actual successful exit or an actual failed launch
plus a separately observed built-in failed release. No failed outcome becomes
a performance denominator. Custom recovery wrappers require a successor.
"""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

Q = Path(__file__).resolve().parent
ROOT = Q.parent
sys.path[:0] = [str(Q), str(ROOT)]
import nemotron9b_r4_terminal_metadata_v2 as terminal
from dory_after_nemotron_r9_v2 import captured_processes, extend_observed
from dory_nemotron_r8_retained_failure_v1 import retained_stage
from minicpm5_saved_tap_inputs_v1 import strict_json

require = terminal.require
PROFILES = {
    'direct-v22': (22, 'nemotron9b_completed_storage_completion_v1', 349, 37),
    'retry-v23': (23, 'nemotron9b_completed_storage_completion_after_retry_v1', 350, 38),
    'retry-v24': (24, 'nemotron9b_completed_storage_completion_after_retry_v2', 350, 38),
}


def merge(pins, additions):
    for path, digest in additions.items():
        require(Path(path).is_absolute() and type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest),
                'Exact absolute historical reference')
        require(path not in pins or pins[path] == digest, 'Historical references cannot be rebound')
        pins[path] = digest


class Reader:
    """Bounded stable JSON only; retain owner variants, not huge parsed records."""
    def __init__(self, guard):
        self.guard, self.pins, self.sizes, self.owners = guard, {}, {}, []

    def __call__(self, ref):
        self.guard()
        require(type(ref) is dict and set(ref) == {'path', 'sha256'}
                and Path(ref['path']).suffix == '.json', 'Explicit JSON-only reference')
        merge(self.pins, {ref['path']: ref['sha256']})
        from nemotron9b_package_metadata_v24 import stable_bytes
        raw = stable_bytes(ref['path'], 32*2**20)
        require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Exact complete JSON bytes')
        self.sizes[ref['path']] = len(raw)
        require(len(self.sizes) <= 4096 and sum(self.sizes.values()) <= 2**30,
                'Complete bounded terminal metadata inventory')
        value = strict_json(raw)
        self.owners.append((copy.deepcopy(ref), list(captured_processes(value))))
        self.guard()
        return value

    def path(self, path):
        self.guard()
        require(Path(path).suffix == '.json', 'Terminal inventory is JSON only')
        return self(terminal.ref(path))


def validate_scope(profile, spec, review, spec_ref, prior, p):
    require(profile in PROFILES, 'Only original direct/retry successor profiles')
    version, _, owners, directories = PROFILES[profile]
    require(spec['schema'] == 'nemotron9b_package_runtime_v'+str(version)
            and spec['root_finalized'] is True and spec['stages'] == list(p.STAGES)
            and len(p.STAGES) == 52 and p.STAGES[-1] == 'verify-raw',
            'All original 52 stages including final raw audit')
    require(spec['resources'] == {s:p.resources(s) for s in (*p.STAGES, 'controller')},
            'Original predecessor resource contracts')
    require(review['schema'] == 'nemotron9b_package_runtime_source_review_v'+str(version)
            and review['passed'] is review['source_only'] is review['independent_review'] is True
            and review['reviewer'] == '/root' and review['blockers'] == []
            and review['execution_spec_sha256'] == spec_ref['sha256']
            and review['source_sha256'] == spec['source_sha256'], 'Original final source review')
    require(spec['controller'] == str(ROOT/f'nemotron9b_package_controller_v{version}.py')
            and spec['worker'] == str(ROOT/f'nemotron9b_package_worker_v{version}.py')
            and spec['automatic_successor_launch'] is False and spec['speedup_denominator'] is None,
            'Exact supported predecessor sources and no failed performance denominator')
    require(type(spec['run_id']) is str and re.fullmatch('[a-z0-9-]{1,40}', spec['run_id']),
            'Exact reviewed predecessor run ID')
    require(spec['control_root'] == str(ROOT/('nemotron9b-package-'+spec['run_id']+'-control'))
            and spec['units'] == {s:'trn1-latest-verify-nemotron9b-package-'+spec['run_id']+'-'+s.replace('_','-')
                                 for s in (*p.STAGES, 'controller')}, 'Exact all-53-service namespace')
    require(len(prior['rows']) == 1149 and len(prior['processes']) == owners
            and len(prior['known_admitted_directories']) == directories
            and prior['completed_storage_maintenance'] == spec['required_completed_storage']
            and prior['completed_residual_review'] == spec['required_residual_completion'],
            'Exact complete Dory then maintenance history for selected branch')
    require(not set(spec['units'].values()) & {x['unit'] for x in prior['rows']},
            'No predecessor service overwritten')
    require((profile in ('retry-v23','retry-v24')) == ('failed_residual_saved_review' in prior),
            'Retain actual failed saved-review history only on retry path')


def validate_terminal(spec, spec_ref, review_ref, receipt, records, refs, failed_release_receipt=None):
    intent, pre, admission, release, audit, end = (records[k] for k in
        ('intent', 'pre', 'admission', 'release', 'audit', 'terminal'))
    owner = intent['external_owner']
    require(intent['execution_spec'] == admission['execution_spec'] == spec_ref
            and intent['source_review'] == admission['source_review'] == review_ref
            and intent['canonical_sha256'] == admission['canonical_sha256'], 'Exact original admission/intent')
    allowed_auditors = [owner] + ([failed_release_receipt['owner']] if failed_release_receipt else [])
    require(pre['launch_intent'] == refs['intent'] and pre['auditor'] in allowed_auditors
            and set(pre['rows']) == set(spec['units'])
            and admission['controller'] == pre['rows']['controller']['owner'],
            'Same actual launcher/auditor and every retained service owner')
    argv = owner['argv']
    require(argv[:3] == ['/usr/bin/python3', '-B', spec['controller']] and len(argv) == 12
            and argv.count('--launch') == 1, 'Exact supported external launch entrypoint')
    for flag, value in (('--spec',spec_ref['path']), ('--spec-sha256',spec_ref['sha256']),
                        ('--review',review_ref['path']), ('--review-sha256',review_ref['sha256'])):
        require(argv.count(flag) == 1 and argv[argv.index(flag)+1] == value, 'Exact original launch arguments')
    tool = receipt['actual_exit']
    require(type(tool['session_id']) is int and tool['session_id'] > 0, 'Actual external session captured')
    code = terminal.validate_actual_exit(receipt, dict(owner=owner), spec_ref, review_ref, tool['session_id'])
    require(release['audit'] == refs['audit'], 'Exact selected original or recovery audit')
    if release['successful']:
        require(code == 0 and failed_release_receipt is None and not audit.get('recovery_only',False),
                'Original successful launch, no fabricated recovery receipt')
    else:
        require(code != 0 and failed_release_receipt is not None, 'Failed launch needs actual separate failed release')
        released=failed_release_receipt
        args=released['owner']['argv']
        require(args[:3]==['/usr/bin/python3','-B',spec['controller']] and len(args)==13
                and args.count('--audit')==args.count('--release-failed')==1,
                'Only exact built-in failed release, not an unknown recovery wrapper')
        for flag,value in (('--spec',spec_ref['path']),('--spec-sha256',spec_ref['sha256']),
                           ('--review',review_ref['path']),('--review-sha256',review_ref['sha256'])):
            require(args.count(flag)==1 and args[args.index(flag)+1]==value, 'Failed release source/spec binding')
        session=released['actual_exit']['session_id']
        require(type(session)is int and session>0 and session!=tool['session_id']
                and (released['owner']['pid'],released['owner']['start_ticks'])!=(owner['pid'],owner['start_ticks']),
                'Distinct actually observed failed-release session and owner')
        require(terminal.validate_actual_exit(released,dict(owner=released['owner']),spec_ref,review_ref,session)==0,
                'Failed release itself actually exited successfully')
        if audit.get('recovery_only'):
            original=records['original_audit']
            require(audit['original_audit']==refs['original_audit']
                    and audit['success_not_promoted'] is True and audit['passed'] is False
                    and original['pre_stop']==refs['pre'], 'Original audit retained without recovery promotion')
    outcome = terminal.classify(code, release, audit, end, records.get('failed'), refs['pre'], refs['audit'], refs['admission'])
    require(outcome['producer_qualified'] is release['successful']
            and outcome['failed_release'] is release['failed'], 'Original scientific success/failure retained')
    return outcome


def extend(prior, spec, rows, summaries, owners, refs, outcome, pins):
    require(len(prior['rows']) == 1149 and outcome['producer_qualified'] is not outcome['failed_release'],
            'Only completed success or explicitly retained failure')
    require(len(rows) == len(summaries) == 53
            and [x['stage'] for x in summaries] == ['controller', *spec['stages']]
            and [x['unit'] for x in rows] == [spec['units'][s] for s in ('controller', *spec['stages'])]
            and len({x['unit'] for x in rows}) == 53, 'All exact 53 retained services')
    if outcome['producer_qualified']:
        require(all(not s['never_launched'] and s['original_exit_success'] is True for s in summaries),
                'Every successful stage really launched and exited successfully')
        require(all(s['worker_reported_complete'] is s['worker_reported_passed']
                    is s['normal_retained_exit_success'] is True for s in summaries[1:]),
                'All 52 complete original successful stage results')
    processes, variants = extend_observed(prior['processes'], owners)
    for process in processes[len(prior['processes']):]:
        process['role'] = 'nemotron_package_recorded_owner'
    wanted = {(s['owner']['pid'],s['owner']['start_ticks']) for s in summaries if s.get('owner') is not None}
    require(wanted <= {(o['pid'],o['start_ticks']) for o in processes}, 'Every recorded stage owner preserved')
    require(not {row['unit'] for row in rows} & {row['unit'] for row in prior['rows']},
            'New service rows never overwrite historical services')
    result = copy.deepcopy(prior)
    result.update(schema='minicpm5_after_nemotron_package_history_v2', rows=prior['rows']+copy.deepcopy(rows),
        processes=processes, complete_count=1202, predecessor_source_sha256=copy.deepcopy(pins),
        completed_nemotron_package=copy.deepcopy(refs), nemotron_outcome=copy.deepcopy(outcome),
        nemotron_stages=copy.deepcopy(summaries), nemotron_observed_owner_variants=variants,
        current_physical_release_verified=False, full_model_qualified=False, new_speedup_claim=False)
    return result


def completed(profile, actual_exit_ref, failed_release_exit_ref=None, guard=lambda: None):
    """Read concrete terminal evidence only after the actual future run finishes."""
    require(profile in PROFILES, 'Exact supported completion profile')
    version, maintenance_name, _, _ = PROFILES[profile]
    p = importlib.import_module('nemotron9b_package_protocol_v'+str(version))
    maintenance = importlib.import_module(maintenance_name).maintenance
    reader = Reader(guard)
    receipt = reader(actual_exit_ref)
    failed_release_receipt=reader(failed_release_exit_ref) if failed_release_exit_ref is not None else None
    spec_ref, review_ref = receipt['execution_spec'], receipt['source_review']
    spec, review = reader(spec_ref), reader(review_ref)
    before = maintenance.completed(spec['required_completed_storage'])
    prior = before['history']
    validate_scope(profile, spec, review, spec_ref, prior, p)
    require(all(spec['source_sha256'].get(k) == v for k,v in before['source_sha256'].items()),
            'Every original completed-history source retained in predecessor')
    control = Path(spec['control_root'])
    require(not list(control.glob('failed-release-extension*')),
            'Custom failure-release extensions require separately reviewed actual exits')
    names = dict(intent='external-launch-intent.json', pre='external-pre-stop.json', admission='admission.json',
                 release='root-release.json', audit='external-audit.json', terminal='terminal.json')
    release=reader.path(control/'root-release.json')
    require(release['audit']['path'] in (str(control/'external-audit.json'),str(control/'external-recovery-audit.json')),
            'Original built-in audit namespace')
    names['audit']=Path(release['audit']['path']).name
    if not (control/'terminal.json').exists():names.pop('terminal')
    if (control/'failed-terminal.json').exists():names['failed']='failed-terminal.json'
    if names['audit']=='external-recovery-audit.json':names['original_audit']='external-audit.json'
    refs = {key:terminal.ref(control/name) for key,name in names.items()}
    records = {key:reader(ref) for key,ref in refs.items()}
    records.setdefault('terminal',None)
    outcome = validate_terminal(spec, spec_ref, review_ref, receipt, records, refs,failed_release_receipt)
    require(records['admission']['canonical_sha256'] == prior['canonical_sha256'], 'Frozen canonical state retained')
    rows, summaries = [], []
    for stage in ('controller', *p.STAGES):
        row, summary = retained_stage(spec, stage, spec_ref, review_ref, p, control,
            records['pre']['rows'][stage], read=reader.path)
        rows.append(row); summaries.append(summary)
    for path in sorted(control.glob('*.json')): reader.path(path)
    pins = {}
    for additions in (before['source_sha256'], spec['source_sha256'], reader.pins): merge(pins, additions)
    endpoints = dict(profile=profile, actual_exit=actual_exit_ref, failed_release_exit=failed_release_exit_ref,execution_spec=spec_ref,
                     source_review=review_ref, control_root=str(control), **refs)
    history = extend(prior, spec, rows, summaries, reader.owners, endpoints, outcome, pins)
    for owner in history['processes']: p.low.gone(owner)
    guard()
    return dict(history=history, source_sha256=pins, metadata_files=len(reader.sizes),
        metadata_bytes=sum(reader.sizes.values()), fresh_physical_drain_verified=False,
        runtime_admitted=False, full_model_qualified=False, new_speedup_claim=False)


if __name__ == '__main__':
    raise SystemExit('Metadata library only; requires actual completed Nemotron and a new serial admission')
