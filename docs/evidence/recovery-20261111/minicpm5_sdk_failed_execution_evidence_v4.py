"""Failure-only external evidence under an already caller-owned serial lock.

No lock acquire/release, launch, stop, kill, retry, model import, or payload read.
Rejected verification leaves the caller's original lock lifecycle intact.
This is not a release tool, successful-exit recorder, or successor admission.
"""
import copy
import datetime
import json
from pathlib import Path
import re

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT/'research'
RUNNER = ROOT/'minicpm5_layer1_sdk_batch_layout_cpu_runner_v4.py'
WORKER = ROOT/'minicpm5_layer1_sdk_batch_layout_cpu_worker_v4.py'
AUDITOR = Q/'audit_minicpm5_layer1_sdk_batch_layout_result_v4.py'
PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
GROUP = '/user.slice/user-1000.slice/user@1000.service/app.slice/'
SCHEMA = 'minicpm5_sdk_actual_failed_execution_evidence_v1'
PARTIAL_FILES = ('admission.json', 'launch-intent.json', 'coordinator-owner.json',
    'rank0-owner.json', 'rank1-owner.json', 'children-exits.json',
    'service-exit.json', 'outer-result.json', 'outer-failure.json',
    'cleanup-pending.json', 'cleanup-drained.json', 'source-authentication.json',
    'predecessor-drain.json', 'coordinator-result.json',
    'rank0-result.json', 'rank1-result.json', 'rank0-failure.json', 'rank1-failure.json')
MAX_PARTIAL_BYTES = 64*2**20


def require(ok, message):
    if not ok: raise RuntimeError(message)


def reference(ref):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'}
        and type(ref['path']) is str and Path(ref['path']).is_absolute()
        and str(Path(ref['path'])) == ref['path'] and '..' not in Path(ref['path']).parts
        and Path(ref['path']).suffix == '.json' and type(ref['sha256']) is str
        and re.fullmatch('[a-f0-9]{64}', ref['sha256']), 'Exact canonical JSON path/SHA256')


def owner(value):
    require(type(value) is dict and {'pid','start_ticks','ppid','argv','cgroup'} <= set(value)
        and type(value['pid']) is int and value['pid'] > 0
        and type(value['ppid']) is int and value['ppid'] > 0
        and type(value['start_ticks']) is str and value['start_ticks'].isdigit()
        and int(value['start_ticks']) > 0 and type(value['argv']) is list and value['argv']
        and all(type(x) is str for x in value['argv'])
        and type(value['cgroup']) is str and value['cgroup'].startswith('0::/'),
        'Exact recorded process identity')
    return value


def failed_tool(session, chunk, exit_code):
    require(type(session) is int and session > 0
        and type(chunk) is str and re.fullmatch('[a-f0-9]{6,64}', chunk)
        and type(exit_code) is int and -127 <= exit_code <= 255 and exit_code != 0,
        'Externally observed nonzero terminal session/chunk/code required')
    return dict(kind='tool_session', session_id=session)


class StableMetadata:
    """Original authenticated transports; bounded exact-path metadata only."""
    def __init__(self, runner, guard):
        self.runner, self.guard = runner, guard
        self.seen, self.missing, self.partial_bytes = {}, set(), 0

    def load(self, ref):
        reference(ref); self.guard()
        before = self.runner.identity(ref['path'])
        value = self.runner.source_auth.load(ref)
        require(self.runner.identity(ref['path']) == before, 'Stable authenticated metadata')
        require(ref['path'] not in self.seen or self.seen[ref['path']] == before, 'No metadata identity rebound')
        self.seen[ref['path']] = before; self.guard()
        return value

    def snapshot(self, path):
        path = str(path); self.guard()
        try: before = self.runner.identity(path)
        except FileNotFoundError:
            self.missing.add(path)
            return dict(path=path, status='missing'), None
        require(before['size'] <= 8*2**20, 'Original bounded partial JSON')
        self.partial_bytes += before['size']
        require(self.partial_bytes <= MAX_PARTIAL_BYTES, 'Bounded total partial metadata')
        ref = self.runner.reference(path); reference(ref)
        require(ref['path'] == path, 'Snapshot remains at requested path')
        try:
            value = self.runner.source_auth.load(ref)
            status, error = 'present', None
        except json.JSONDecodeError as exc:
            value, status, error = None, 'invalid_json', str(exc)[:256]
        require(self.runner.identity(path) == before, 'Stable partial metadata')
        require(path not in self.seen or self.seen[path] == before, 'No partial identity rebound')
        self.seen[path] = before; self.guard()
        row = dict(path=path, status=status, ref=ref, identity=before)
        if error is not None: row['parse_error'] = error
        return row, value

    def unchanged(self):
        self.guard()
        for path, old in self.seen.items():
            require(self.runner.identity(path) == old, 'Observed failure metadata changed')
        for path in self.missing:
            try: self.runner.identity(path)
            except FileNotFoundError: continue
            raise RuntimeError('A previously missing partial receipt appeared')
        self.guard()


def live_header(live, phase, handle):
    schema = ('minicpm5_layer1_sdk_batch_layout_actual_live_v1' if phase == 'model'
        else 'minicpm5_sdk_batch_layout_audit_live_v1')
    require(type(live) is dict and live.get('schema') == schema
        and live.get('reviewer') == '/root' and live.get('actual_identity_verified') is True
        and live.get('actual_execution_handle') == handle, 'Exact external live-to-terminal session')
    return owner(live['owner'])


def bundle(metadata, spec_ref, review_ref):
    reference(spec_ref); reference(review_ref)
    require(Path(spec_ref['path']).parent == Q and Path(review_ref['path']).parent == Q
        and Path(spec_ref['path']).name.startswith('minicpm5-layer1-sdk-batch-layout-')
        and Path(spec_ref['path']).name.endswith('-spec.json')
        and review_ref['path'] == spec_ref['path'][:-len('-spec.json')]+'-source-review.json',
        'Original SDK spec/review namespace')
    spec, review = metadata.load(spec_ref), metadata.load(review_ref)
    require(spec.get('schema') == 'minicpm5_layer1_sdk_batch_layout_cpu_execution_spec_v2'
        and spec.get('runner') == str(RUNNER) and spec.get('worker') == str(WORKER)
        and review.get('schema') == 'minicpm5_layer1_sdk_batch_layout_runtime_source_review_v2'
        and review.get('reviewer') == '/root' and review.get('passed') is True
        and review.get('blockers') == [] and review.get('spec') == spec_ref
        and review.get('source_sha256') == spec.get('source_sha256')
        and type(spec.get('source_sha256')) is dict and bool(spec['source_sha256']),
        'Original reviewed bundle metadata, not a new review')
    root = Path(spec['output_root'])
    require(root.parent == Path('/mnt/nvme')
        and re.fullmatch('minicpm5-layer1-sdk-batch-layout-[a-z0-9][a-z0-9-]{0,39}', root.name)
        and spec['unit'] == 'trn1-mini-layer1-sdk-batch-layout-'+root.name.removeprefix('minicpm5-layer1-sdk-batch-layout-'),
        'Original model output root and service')
    reference(spec['plan']); reference(spec['source_contract'])
    return spec, root


def record_failure(runner, phase, live_ref, session, chunk, exit_code, recorder, lock):
    """Return failed-execution evidence or reject while caller retains its lock.

    runner is the original pinned sdk cpu runner_v4 module. recorder and lock
    come from the caller's actual original serial-lock acquisition. This helper
    never acquires/releases that lock or terminates a process. It writes nothing.
    """
    require(phase in ('model', 'auditor'), 'Only model or independent auditor failure')
    handle = failed_tool(session, chunk, exit_code)
    reference(live_ref)
    require(Path(live_ref['path']).parent == Q, 'Original external live receipt in research')
    owner(recorder)
    require(type(lock) is dict and set(lock) == {'device','inode'}
        and all(type(v) is int and v > 0 for v in lock.values()), 'Original lock identity')
    p = runner.p
    def guard():
        p.lock_proof(recorder, lock); p.clear_window()
    guard(); metadata = StableMetadata(runner, guard)
    live = metadata.load(live_ref); failed_owner = live_header(live, phase, handle)
    require((failed_owner['pid'], failed_owner['start_ticks']) !=
        (recorder['pid'], recorder['start_ticks']), 'Recorder distinct from failed owner')
    p.gone(failed_owner); guard()
    model_exit_ref, model_exit, audit_output = None, None, None
    if phase == 'model':
        model_live = live
    else:
        model_exit_ref = live['model_actual_exit']; reference(model_exit_ref)
        model_exit = metadata.load(model_exit_ref)
        require(model_exit.get('schema') == 'minicpm5_layer1_sdk_batch_layout_actual_exit_v1'
            and model_exit.get('reviewer') == '/root' and model_exit.get('actual_exit_observed') is True
            and type(model_exit.get('actual_exit_code')) is int and model_exit['actual_exit_code'] == 0
            and model_exit.get('full_model_qualified') is False and model_exit.get('new_speedup_claim') is False,
            'Auditor failure refers to original model terminal metadata')
        model_live = metadata.load(model_exit['live']); model_handle = model_exit['actual_execution_handle']
        require(type(model_handle) is dict and set(model_handle) == {'kind','session_id'}
            and model_handle['kind'] == 'tool_session' and type(model_handle['session_id']) is int
            and model_handle['session_id'] > 0 and model_handle['session_id'] != session,
            'Separate original model session')
        require(live_header(model_live, 'model', model_handle) == model_exit['owner'], 'Original model live/exit owner')
        require(model_exit['spec'] == model_live['spec'] and model_exit['review'] == model_live['review']
            and model_exit['lock_identity'] == lock, 'Original model bundle and lock')
        args = failed_owner['argv']
        require(len(args) == 9 and args[:3] == ['python3','-B',str(AUDITOR)]
            and args[3:7] == ['--model-exit',model_exit_ref['path'],'--model-exit-sha256',model_exit_ref['sha256']]
            and args[7] == '--output', 'Exact failed independent auditor argv')
        audit_output = Path(args[8])
        require(audit_output.parent == Q and re.fullmatch(
            r'minicpm5-layer1-sdk-batch-layout-[a-z0-9-]+-post-exit-audit-[a-z0-9-]+\.json',audit_output.name),
            'Original auditor output namespace')
    model_owner = owner(model_live['owner'])
    spec_ref, review_ref = model_live['spec'], model_live['review']
    spec, root = bundle(metadata, spec_ref, review_ref)
    expected = [PYTHON,'-B',str(RUNNER),'--mode','outer','--spec',spec_ref['path'],
        '--spec-sha256',spec_ref['sha256'],'--review',review_ref['path'],'--review-sha256',review_ref['sha256']]
    require(model_owner['argv'] == expected, 'Exact original model outer argv')
    p.gone(model_owner); guard()
    snapshots, values = {}, {}
    for filename in PARTIAL_FILES:
        snapshots[filename], values[filename] = metadata.snapshot(root/filename)
    def critical(name):
        require(snapshots[name]['status'] != 'invalid_json', 'Malformed critical failure evidence: '+name)
        value = values[name]
        if snapshots[name]['status'] == 'missing':
            require(value is None, 'Missing evidence cannot have an invented value')
        else:
            require(type(value) is dict, 'Present critical evidence must be an object: '+name)
        return value
    admission, intent = critical('admission.json'), critical('launch-intent.json')
    coordinator = critical('coordinator-owner.json')
    ranks = [critical('rank%d-owner.json'%rank) for rank in (0,1)]
    admission_ref = snapshots['admission.json'].get('ref'); attempt = None
    if admission is not None:
        require(admission.get('schema') == 'minicpm5_layer1_sdk_batch_layout_cpu_admission_v2'
            and admission.get('spec') == spec_ref and admission.get('review') == review_ref
            and admission.get('plan') == spec['plan'] and admission.get('outer_owner') == model_owner
            and admission.get('lock_identity') == lock and admission.get('source_contract') == spec['source_contract']
            and admission.get('latest_completion') == spec['latest_completion']
            and type(admission.get('attempt_id')) is str and re.fullmatch('[a-f0-9]{32}',admission['attempt_id']),
            'Available admission belongs to actual original model/lock')
        attempt = admission['attempt_id']
    launch_evidence_present = intent is not None
    if launch_evidence_present:
        require(admission is not None and intent.get('admission') == admission_ref
            and intent.get('command') == runner.service_command(spec,admission_ref,attempt),
            'Exact durable owned launch intent')
    require(not(coordinator is not None or any(x is not None for x in ranks)) or launch_evidence_present,
        'Service/rank records require original admission and launch intent')
    checked_owners = [model_owner]
    if phase == 'auditor': checked_owners.insert(0,failed_owner)
    if coordinator is not None:
        owner(coordinator)
        require(coordinator.get('unit') == spec['unit'] and type(coordinator.get('invocation_id')) is str
            and re.fullmatch('[a-f0-9]{32}',coordinator['invocation_id']) and coordinator['invocation_id'] != '0'*32
            and coordinator['argv'] == runner.argv('coordinator',admission_ref)
            and coordinator['cgroup'] == '0::'+GROUP+spec['unit']+'.service',
            'Coordinator belongs to original service/invocation')
        checked_owners.append(coordinator)
    for rank in ranks:
        if rank is None: continue
        owner(rank)
        require(coordinator is not None and rank['ppid'] == coordinator['pid']
            and rank['argv'] == runner.argv('rank',admission_ref) and rank['cgroup'] == coordinator['cgroup'],
            'Rank belongs to original coordinator')
        checked_owners.append(rank)
    require(len({(x['pid'],x['start_ticks'])for x in checked_owners}) == len(checked_owners),
        'Distinct failed and service process identities')
    if phase == 'auditor':
        require(coordinator == model_exit['coordinator'] and ranks == model_exit['ranks'],
            'Exact previously observed model service processes')
    for name in ('outer-failure.json','cleanup-pending.json','cleanup-drained.json'):
        value = critical(name)
        if value is not None: require(value.get('owner') == model_owner,'Failure/cleanup receipt belongs to actual outer')
    value = critical('children-exits.json')
    if value is not None:
        require(type(value.get('owners')) is list, 'Explicit recorded child owner list')
        available_ranks = [rank for rank in ranks if rank is not None]
        for child in value['owners']:
            owner(child)
            require(child in available_ranks, 'No unrelated recorded child owner')
    for index in (0,1):
        for suffix in ('result','failure'):
            value = values['rank%d-%s.json'%(index,suffix)]
            if type(value) is dict and 'process' in value:
                require(ranks[index] is not None and value['process'] == ranks[index],
                    'Partial rank record has original owner')
    for value in checked_owners: p.gone(value); guard()
    if audit_output is not None:
        row,_ = metadata.snapshot(audit_output); snapshots['auditor-output'] = row
    # Only the original read-only drain verification; no cleanup or signal API.
    proof = runner.verify_owned_drain(spec,admission_ref,attempt,coordinator,launch_evidence_present)
    require(type(proof) is dict and proof.get('physically_drained') is True, 'Owned service actually drained')
    require(p.terminal(proof['properties']) and proof['properties'].get('Job') == '', 'No live service or pending Job')
    expected_service_owners = ([coordinator]if coordinator is not None else[])+[x for x in ranks if x is not None]
    require(proof.get('owners_checked') == expected_service_owners,'Verifier checked all available service owners')
    for value in checked_owners: p.gone(value)
    p.active_heavy(); metadata.unchanged(); guard()
    return dict(schema=SCHEMA,reviewer='/root',phase=phase,failed=True,successful=False,
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        live=copy.deepcopy(live_ref),model_actual_exit=copy.deepcopy(model_exit_ref),
        spec=copy.deepcopy(spec_ref),review=copy.deepcopy(review_ref),
        actual_tool_exit_observed=True,actual_execution_handle=handle,actual_exit_code=exit_code,tool_chunk=chunk,
        failed_owner=copy.deepcopy(failed_owner),recorder=copy.deepcopy(recorder),lock_identity=copy.deepcopy(lock),
        exact_recorded_processes_gone=copy.deepcopy(checked_owners),owned_model_drain=proof,
        available_partial_metadata=copy.deepcopy(snapshots),full_source_verification_performed=False,
        raw_tensor_bytes_read=False,source_authentication_not_requalified=True,historical_release_not_requalified=True,
        original_cleanup_receipts_not_invented=True,absent_outputs_not_invented=True,no_stop_or_kill_performed=True,
        lock_acquired_or_released_by_helper=False,automatic_successor_release=False,runtime_admitted=False,
        full_model_qualified=False,new_speedup_claim=False)
