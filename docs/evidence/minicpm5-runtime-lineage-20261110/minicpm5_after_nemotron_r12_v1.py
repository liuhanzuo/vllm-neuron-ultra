"""Future r12 terminal enrollment; metadata and physical drain are separate.

No entrypoint or model/service operations. The original protocol import reads
bounded historical JSON constants. The independently admitted caller supplies
a stable authenticated JSON read(ref) and its serial guard. Never run completed
or physical against the live r12 merely to prepare sources. Metadata may hash
bounded terminal JSON, never binaries/model payloads; physical is read-only.
"""
import copy
from pathlib import Path
import sys
Q = Path(__file__).resolve().parent
ROOT = Q.parent
sys.path[:0] = [str(ROOT), str(Q)]
import nemotron9b_package_protocol_v27 as p
import nemotron9b_r4_terminal_metadata_v2 as terminal
from minicpm5_after_nemotron_package_v2 import merge
from dory_after_nemotron_r9_v2 import captured_processes, extend_observed
from dory_nemotron_r8_retained_failure_v1 import retained_stage
require = p.require
SCHEMA = 'minicpm5_after_nemotron_r12_binding_v1'
SPEC = dict(path=str(Q/'nemotron9b-package-runtime-v27-r12-20261110-spec.json'),
    sha256='4186c9a29921b96fc3a55fb04040cf70977ca921130df288466c8d27e65feaaa')
REVIEW = dict(path=str(Q/'nemotron9b-package-runtime-v27-r12-20261110-source-review.json'),
    sha256='99b0cdb4ab842ca97b4b6b3283aacd595f8dabb60806e0823fae89458fcdef40')
LIVE = dict(path=str(Q/'nemotron9b-r12-actual-external-live-20261110-v1.json'),
    sha256='4a81f4c2eef387b6955f9da3dfcf42bc759833f460a00af4d654f80b4aa414ab')
LIVE_SESSION = 28398
LIVE_OWNER = dict(pid=723810, start_ticks='409772894', ppid=433847,
    argv=[p.PYTHON, '-B', str(ROOT/'nemotron9b_package_controller_v27.py'), '--launch',
        '--spec', SPEC['path'], '--spec-sha256', SPEC['sha256'],
        '--review', REVIEW['path'], '--review-sha256', REVIEW['sha256']],
    cgroup='0::/user.slice/user-1000.slice/session-56.scope')


def reference(ref):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'}
        and type(ref['path']) is str and Path(ref['path']).suffix == '.json',
        'Exact JSON path/SHA256 reference')
    merge({}, {ref['path']: ref['sha256']})
    return ref


def binding_check(binding):
    require(type(binding) is dict
        and set(binding) == {'schema', 'actual_exit', 'failed_release_exit'}
        and binding['schema'] == SCHEMA, 'Exact completed-r12 binding schema')
    reference(binding['actual_exit'])
    if binding['failed_release_exit'] is not None:
        reference(binding['failed_release_exit'])
        require(binding['failed_release_exit'] != binding['actual_exit'],
            'Actual launch and separate failed-release receipts differ')


class TrackedRead:
    """Adapt a plain authenticated JSON callback; preserve every owner variant.

    Callback owns stable full JSON digest/optional length validation. .path()
    freezes a bounded terminal JSON ref then sends it through the same callback.
    """
    def __init__(self, read, guard):
        require(callable(read) and callable(guard), 'Explicit JSON reader and serial guard')
        self.read, self.guard, self.pins, self.observations = read, guard, {}, []

    def __call__(self, ref):
        self.guard()
        require(type(ref) is dict and set(ref) in ({'path', 'sha256'}, {'path', 'sha256', 'bytes'}),
            'Exact authenticated JSON reference')
        reference({k: ref[k] for k in ('path', 'sha256')})
        if 'bytes' in ref:
            require(type(ref['bytes']) is int and ref['bytes'] >= 0, 'Optional exact metadata length')
        merge(self.pins, {ref['path']: ref['sha256']})
        value = self.read(ref)
        self.observations.append((copy.deepcopy(ref), list(captured_processes(value))))
        self.guard()
        return value

    def path(self, path):
        self.guard()
        require(Path(path).suffix == '.json', 'Terminal inventory is JSON only')
        return self(terminal.ref(path))


def scope(spec, review, previous):
    prior = previous['previous']['history']
    require(spec['schema'] == 'nemotron9b_package_runtime_v27' and spec['run_id'] == 'r12-20261110'
        and spec['root_finalized'] is True and spec['stages'] == list(p.STAGES)
        and len(spec['stages']) == 52, 'Exact full52-stage actual r12')
    require(spec['controller'] == str(ROOT/'nemotron9b_package_controller_v27.py')
        and spec['worker'] == str(ROOT/'nemotron9b_package_worker_v27.py')
        and all(spec[key] == '/mnt/nvme/nemotron9b-package-r12-20261110'+suffix for key, suffix in
            [('control_root', '-control'), ('evidence_root', '-evidence'), ('work_root', '')]),
        'Exact original source and three r12 namespaces')
    require(spec['units'] == {s: 'trn1-latest-verify-nemotron9b-package-r12-20261110-'+s.replace('_', '-')
            for s in (*p.STAGES, 'controller')}
        and spec['resources'] == {s: p.resources(s) for s in (*p.STAGES, 'controller')},
        'Every53 unit and original resource contract')
    require(review['schema'] == 'nemotron9b_package_runtime_source_review_v27'
        and review['execution_spec_sha256'] == SPEC['sha256']
        and review['source_sha256'] == spec['source_sha256'] and review['reviewer'] == '/root'
        and review['passed'] is review['source_only'] is review['independent_review'] is True
        and review['blockers'] == [], 'Exact finalized independent r12 review')
    require(spec['automatic_successor_launch'] is False and spec['speedup_denominator'] is None
        and spec['post_r11_completion'] == p.successor.BINDING, 'Unchanged predecessor binding and no promotion')
    require(len(prior['rows']) == 1255 and len(prior['processes']) == 806
        and prior['nemotron_r11_outcome']['failed_release'] is True
        and prior['nemotron_r11_outcome']['producer_qualified'] is False,
        'All original1255 services/806 owners and failed r11 retained')
    current = previous['current_history']
    require(current['decode_calls'] == 180 and current['previous_control_calls'] == 108
        and current['raw_bytes'] == 2949120 and len(current['owners']) == 5
        and current['complete_model_and_independent_auditor_authenticated'] is True,
        'Actual current/history180-call Mini model and independent auditor retained')
    require(previous['handoff']['canonical_sha256'] == prior['canonical_sha256']
        and not set(spec['units'].values()) & {x['unit'] for x in prior['rows']},
        'Original canonical history and disjoint r12 services')


def terminal_chain(spec, receipt, records, refs, failed_release=None):
    intent, pre, admission, release, audit = (records[k] for k in ('intent', 'pre', 'admission', 'release', 'audit'))
    require(intent['external_owner'] == receipt['owner'] == LIVE_OWNER, 'Exactly the actually observed live r12 owner')
    code = terminal.validate_actual_exit(receipt, dict(owner=LIVE_OWNER), SPEC, REVIEW, LIVE_SESSION)
    require(intent['execution_spec'] == admission['execution_spec'] == SPEC
        and intent['source_review'] == admission['source_review'] == REVIEW
        and intent['canonical_sha256'] == admission['canonical_sha256']
        and intent['lock_identity'] == admission['lock_identity'], 'Actual original admission and serial lock')
    require(pre['launch_intent'] == refs['intent'] and set(pre['rows']) == set(spec['units'])
        and pre['rows']['controller']['owner'] == admission['controller'], 'All53 retained owners, including controller')
    auditors = [LIVE_OWNER]
    if release['successful']:
        require(code == 0 and failed_release is None, 'Success has actual exit0 and no failed-release receipt')
    else:
        require(code != 0 and failed_release is not None, 'Failed launch requires separate actual failed release')
        owner, tool = failed_release['owner'], failed_release['actual_exit']
        command = [p.PYTHON, '-B', spec['controller'], '--audit', '--release-failed',
            '--spec', SPEC['path'], '--spec-sha256', SPEC['sha256'],
            '--review', REVIEW['path'], '--review-sha256', REVIEW['sha256']]
        require(owner['argv'] == command and type(tool['session_id']) is int and tool['session_id'] > 0
            and tool['session_id'] != LIVE_SESSION
            and (owner['pid'], owner['start_ticks']) != (LIVE_OWNER['pid'], LIVE_OWNER['start_ticks']),
            'Distinct actual supported v27 audit/release-failed owner and session')
        require(terminal.validate_actual_exit(failed_release, dict(owner=owner), SPEC, REVIEW, tool['session_id']) == 0,
            'Separate failed-release tool really exited0')
        auditors.append(owner)
        if audit.get('recovery_only'):
            require(audit['original_audit'] == refs['original_audit']
                and records['original_audit']['pre_stop'] == refs['pre']
                and audit['success_not_promoted'] is True, 'Original failed audit preserved across recovery')
    require(pre['auditor'] in auditors, 'Actual retained snapshot auditor')
    outcome = terminal.classify(code, release, audit, records.get('terminal'), records.get('failed'),
        refs['pre'], refs['audit'], refs['admission'])
    require(outcome['producer_qualified'] is release['successful']
        and outcome['failed_release'] is release['failed'], 'Failed outcomes cannot become successful')
    return outcome


def terminal_records(spec, reader):
    control = Path(spec['control_root'])
    require(not list(control.glob('failed-release-extension*')), 'Custom release requires a reviewed successor')
    release = reader.path(control/'root-release.json')
    require(release['audit']['path'] in (str(control/'external-audit.json'), str(control/'external-recovery-audit.json')),
        'Only original or supported recovery audit')
    names = dict(intent='external-launch-intent.json', pre='external-pre-stop.json', admission='admission.json',
        release='root-release.json', audit=Path(release['audit']['path']).name)
    for key, name in [('terminal', 'terminal.json'), ('failed', 'failed-terminal.json')]:
        if (control/name).exists(): names[key] = name
    if names['audit'] == 'external-recovery-audit.json': names['original_audit'] = 'external-audit.json'
    refs = {key: terminal.ref(control/name) for key, name in names.items()}
    return refs, {key: reader(ref) for key, ref in refs.items()}


def completed(binding, read, guard):
    """Metadata only. Actual missing/pending exit rejects before old history IO."""
    require(callable(guard), 'Explicit independent serial guard'); guard()
    binding_check(binding)
    reader = TrackedRead(read, guard)
    receipt = reader(binding['actual_exit'])
    terminal.validate_actual_exit(receipt, dict(owner=LIVE_OWNER), SPEC, REVIEW, LIVE_SESSION)
    live = reader(LIVE)
    require(live['schema'] == 'nemotron9b_r12_actual_external_live_v1'
        and live['actual_live_identity_verified'] is True and live['external_owner'] == LIVE_OWNER
        and live['actual_execution_handle']['session_id'] == LIVE_SESSION
        and live['execution_spec'] == SPEC and live['source_review'] == REVIEW, 'Original actually observed r12 lineage')
    failed_release = reader(binding['failed_release_exit']) if binding['failed_release_exit'] is not None else None
    spec = reader(SPEC); review = reader(REVIEW); catalog = reader(spec['final_predecessor_catalog'])
    previous = p.successor.metadata(reader, guard)
    scope(spec, review, previous)
    refs, records = terminal_records(spec, reader)
    outcome = terminal_chain(spec, receipt, records, refs, failed_release)
    prior = previous['previous']['history']
    require(records['admission']['canonical_sha256'] == prior['canonical_sha256']
        and records['admission']['lock_identity'] == previous['lock_identity'], 'Same complete original canonical and serial lock')
    rows, summaries = [], []
    for stage in ('controller', *p.STAGES):
        guard()
        row, summary = retained_stage(spec, stage, SPEC, REVIEW, p, Path(spec['control_root']),
            records['pre']['rows'][stage], read=reader.path)
        rows.append(row); summaries.append(summary)
    require(len(rows) == len(summaries) == 53
        and [x['stage'] for x in summaries] == ['controller', *p.STAGES]
        and [x['unit'] for x in rows] == [spec['units'][s] for s in ('controller', *p.STAGES)],
        'All53 original retained_stage validations in order')
    if outcome['producer_qualified']:
        require(all(not x['never_launched'] and x['original_exit_success'] is True for x in summaries)
            and all(x['worker_reported_complete'] is x['worker_reported_passed']
                is x['normal_retained_exit_success'] is True for x in summaries[1:]),
            'Every original successful stage completed; no absent stage promoted')
    require(all(x.get('owner') is not None for x in summaries if not x['never_launched']), 'Every launched stage has actual owner')
    for path in sorted(Path(spec['control_root']).glob('*.json')): reader.path(path)
    observations = [(dict(kind='authenticated-post-r11-metadata',
        bindings=copy.deepcopy(p.successor.BINDING)), previous), *reader.observations,
        (dict(kind='validated-r12-retained-stages', pre=refs['pre'],
            execution_spec=SPEC, source_review=REVIEW), summaries)]
    processes, variants = extend_observed(prior['processes'], observations)
    for process in processes[len(prior['processes']):]:
        process['role'] = 'post_r11_and_r12_recorded_owner'
    seen = {(x['pid'], x['start_ticks']) for x in processes}
    require(all((x['owner']['pid'], x['owner']['start_ticks']) in seen for x in summaries if x.get('owner') is not None),
        'Every retained r12 owner included')
    owners = copy.deepcopy(prior['processes'])
    for observation in variants:
        if observation['owner'] not in owners: owners.append(copy.deepcopy(observation['owner']))
    require(all(owner in owners for owner in current_owners(previous)), 'All five current/history Mini owners preserved')
    pins = {}
    for additions in (previous['source_sha256'], previous['previous']['source_sha256'], spec['source_sha256'], reader.pins):
        merge(pins, additions)
    endpoints = dict(actual_exit=copy.deepcopy(binding['actual_exit']), failed_release_exit=copy.deepcopy(binding['failed_release_exit']),
        execution_spec=SPEC, source_review=REVIEW, control_root=spec['control_root'], **refs)
    history = copy.deepcopy(prior)
    history.update(schema='minicpm5_after_nemotron_r12_history_v1', rows=prior['rows']+copy.deepcopy(rows),
        processes=processes, complete_count=1308, predecessor_source_sha256=copy.deepcopy(pins),
        completed_nemotron_r12=endpoints, nemotron_r12_outcome=outcome, nemotron_r12_stages=summaries,
        nemotron_r12_owner_variants=variants, current_physical_release_verified=False,
        full_model_qualified=False, new_speedup_claim=False)
    require(len(history['rows']) == len({x['unit'] for x in history['rows']}) == 1308,
        'All1308 historical services without duplicates')
    guard()
    return dict(schema='minicpm5_after_nemotron_r12_completed_v1', binding=copy.deepcopy(binding),
        source_sha256=pins, spec=spec, review=review, catalog=catalog, owners=owners,
        owner_variants=variants, history=history, outcome=outcome, pre_refs=endpoints,
        pre=records['pre'], lock_identity=copy.deepcopy(previous['lock_identity']), previous=previous,
        fresh_physical_drain_verified=False, runtime_admitted=False, full_model_qualified=False, new_speedup_claim=False)


def current_owners(previous):
    return list(captured_processes(previous['current_history']['owners']))


def physical(done, guard):
    """Future actual serial-admitted recheck; never called during preparation."""
    require(callable(guard), 'Explicit independent serial guard'); guard()
    require(done['schema'] == 'minicpm5_after_nemotron_r12_completed_v1', 'Actual metadata-completed result required')
    binding_check(done['binding']); scope(done['spec'], done['review'], done['previous'])
    spec, history, pre = done['spec'], done['history'], done['pre']
    require(len(history['rows']) == 1308 and history['completed_nemotron_r12'] == done['pre_refs']
        and set(pre['rows']) == set(spec['units']), 'Full1308 service history and exact r12 retained snapshot')
    for owner in done['owners']: guard(); p.low.gone(owner)
    # Original v27 drain includes all old services, post-r11 Mini/storage and
    # the authenticated180-call current/history model and independent auditor.
    inherited = p.predecessor_drain(done['catalog'], spec=spec); guard()
    current = []
    for stage in ('controller', *p.STAGES):
        saved, samples = pre['rows'][stage], []
        for _ in range(2):
            guard()
            props = p.validate_after(spec, stage, SPEC, REVIEW, saved, p.low.properties(spec['units'][stage]))
            require(p.low.terminal(props) and props['ActiveState'] in ('inactive', 'failed'), 'Every r12 unit actually released')
            if saved['owner'] is not None: p.low.gone(p.proc_owner(saved['owner']))
            samples.append(dict(properties=props, group=p.low.empty_group(p.group(spec, stage))))
        require(samples[0] == samples[1], 'Stable repeated original r12 service/group drain')
        current.append(dict(stage=stage, samples=samples))
    p.low.active_heavy(); guard()
    require(p.canonical() == history['canonical_sha256'], 'Canonical unchanged after complete physical drain')
    return dict(complete=True, services=1308, owner_observations=len(done['owners']),
        inherited=inherited, current=current, fresh_physical_drain_verified=True,
        runtime_admitted=False, full_model_qualified=False, new_speedup_claim=False)
