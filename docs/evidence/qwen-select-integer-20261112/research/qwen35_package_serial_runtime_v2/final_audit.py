"""Stage25 callable only by the independently admitted root after actual tool exit.

The external tool API is not observable from a child Python program. Root must
supply its hash-bound actual tool observation, including the session ID and
successful exit. This function independently checks all other available source,
input, stage, saved actual-child and current systemd/cgroup observations. It does
not choose a current binding, acquire a lock, load tensors or promote a result.
"""
import hashlib
import os
from pathlib import Path

from .contract import require, control_root, PHASES, LIMITS
from .io import read, ref, raw, once, process_api, storage, loaded_local_sources
from .history import Bundle, Authority
from .adapter import declared_services, admit_shape
from .budget import unchanged_inputs
from .wiring import STAGES, ARMS, manifest_sha, stage_receipt, check_build_seal


def audit(spec_ref, review_ref, external_tool_exit_ref, execution_guard):
    require(callable(execution_guard), 'existing independent root guard required')
    def guard():
        value = execution_guard()
        require(type(value) is dict and value.get('admitted') is True and value.get('exclusive') is True,
                'root must retain original serial audit authority')
        return value
    guard()
    bundle = Bundle(spec_ref, review_ref, guard).resolve()
    spec = bundle.spec; root = Path(spec['output_root']); control = control_root(spec)
    admission_ref = ref(control / 'admission.json'); admission = read(admission_ref)
    admit_shape(admission, spec, spec_ref, review_ref)
    coordinator_ref = ref(control / 'coordinator-owner.json'); coordinator = read(coordinator_ref)
    auditor = process_api().proc(os.getpid())
    require(auditor is not None and auditor != admission['outer_owner'] and auditor != coordinator['process'],
            'stage25 must be a genuinely separate root observer after original outer exit')
    process_api().gone(admission['outer_owner'])
    tool = read(external_tool_exit_ref)
    running = read(ref(spec['running_handoff_path']))
    wanted_tool = running['qwen35_sampler_current_execution']['tool']
    require(set(tool) == {'schema', 'reviewer', 'spec', 'admission', 'outer_owner', 'tool', 'exit_code', 'completed',
                         'observation_source', 'observed_unix_ns'}
            and tool['schema'] == 'qwen35_package_root_actual_tool_exit_v1' and tool['reviewer'] == '/root'
            and tool['spec'] == spec_ref and tool['admission'] == admission_ref and tool['outer_owner'] == admission['outer_owner']
            and tool['tool'] == wanted_tool and tool['observation_source'] == 'actual_external_tool_result'
            and type(tool['exit_code']) is int and tool['exit_code'] == 0 and tool['completed'] is True
            and type(tool['observed_unix_ns']) is int
            and tool['observed_unix_ns'] > 0, 'independently observed exact outer tool session successful exit required')
    terminal_ref = ref(control / 'outer-terminal.json'); terminal = read(terminal_ref)
    require(terminal['admission'] == admission_ref and terminal['outer_owner'] == admission['outer_owner']
            and terminal['failed'] is False and terminal['whole_owned_scope_drained'] is True
            and terminal['result_present'] is True, 'successful entire outer scope terminal required')
    services = declared_services(spec, admission_ref, coordinator)
    require(len(services) == 1 + len(PHASES), 'all declared finite services required at final audit')
    ends = {}
    for service in services:
        guard(); actual = service.drained()
        require(actual['cleanup_complete'] is True and actual['remaining_processes'] == []
                and actual['systemd_exit_status'] == {'ExecMainCode': 1, 'ExecMainStatus': 0, 'Result': 'success'},
                'independent current actual systemd terminal plus two stable recursive empty-group checks required')
        ends[service.unit] = actual
    for arm, phase in PHASES:
        stage = control / (arm + '-' + phase)
        phase_ref = ref(stage / 'phase-spec.json'); phase_spec = read(phase_ref)
        code = read(ref(stage / 'child-exit.json')); supervisor = read(ref(stage / 'supervisor-owner.json'))
        child = read(ref(stage / 'child-bootstrap-owner.json'))
        require(code['phase_spec'] == phase_ref and code['supervisor'] == supervisor
                and code['child_bootstrap_owner'] == child and code['actual_popen_wait_observed'] is True,
                'every phase must retain its real direct-child owner and bounded Popen.wait result')
        api = phase in ('capture', 'quality', 'http')
        require(code['stop_requested'] is api and code['child_returncode'] in ((0, -15) if api else (0,)), 'actual phase child exit failed')
        process_api().gone(child)
        require(phase_spec['unit'] in ends, 'phase terminal outside complete owned scope')
    receipts = []
    for index, (arm, stage) in enumerate(STAGES[:-1], 1):
        receipt = ref(stage_receipt(spec, index)); value = read(receipt)
        require(value['index'] == index and (value['arm'], value['stage']) == (arm, stage)
                and value['case'] == spec['case'] and value['completed'] is True, 'all 24 exact completed predecessor stages required')
        read(value['result']); receipts.append(receipt)
    co_result = read(ref(control / 'coordinator-result.json'))
    require(co_result['admission'] == admission_ref and co_result['coordinator'] == coordinator
            and co_result['driver_result']['stage_receipts'] == receipts
            and co_result['driver_result']['completed_stages'] == 24
            and co_result['self_exit_claimed'] is False, 'actual coordinator result must bind the same complete 24-stage receipts')
    # Re-run complete saved scientific metadata gates. Binary vectors/packets were
    # checked in their real separately owned phases; stage25 does not reread them.
    from qwen35_package_full_validation_v2.checks import check_model_metadata, original_validators
    from qwen35_package_full_validation_v2 import sampler, http
    validators = original_validators()
    names = {'state_copy': 'state-copy.json', 'numeric': 'real-logits-validation.json',
             'cold': 'cold-buckets.json', 'long': 'long-decode-validation.json'}
    for arm in ARMS:
        for phase, name in names.items():
            check_model_metadata(read(ref(root / arm / 'model' / name)), phase, validators, spec['case'])
        vectors = read(ref(root / arm / 'model/vectors203.json'))
        require(vectors['frames'] == vectors['identical_vectors'] == 203 and vectors['complete'] is True, 'complete original raw-vector gates required')
    sampler_gate = sampler.check_saved_sampler_gates(spec['case'], root)
    hosts = {arm: read(ref(root / arm / 'http/root-live.json')) for arm in ARMS}
    scientific = http.compare_saved_pair(spec['case'], root / 'baseline', root / 'candidate', hosts,
                                        {arm: str(root / arm / 'http') for arm in ARMS})
    unchanged_inputs(spec); check_build_seal(spec); manifest = manifest_sha(spec)
    for path, sha in spec['source_sha256'].items():
        # Inherited complete source partitions are verified by Bundle.resolve;
        # this exact smaller host closure is additionally rehashed here.
        if path in bundle.host:
            require(hashlib.sha256(raw(path, 4 * 1024**2)).hexdigest() == sha, 'executed package/source pin changed')
    pair = bundle.physical_pair(Authority(bundle, guard, lambda: []))
    stable_storage = storage(spec, terminal=True)
    bundle.unchanged(); loaded_local_sources(spec['source_sha256']); guard()
    result = dict(schema='qwen35_package_independent_final_audit_v1', reviewer='/root', actual_observer=auditor,
        spec=spec_ref, admission=admission_ref, external_tool_exit=external_tool_exit_ref,
        outer_terminal=terminal_ref, stage_receipts=receipts, actual_service_terminals=ends,
        full_history_pair=pair, storage=stable_storage, artifact_manifest_sha256=manifest,
        sampler=sampler_gate, science=scientific, all25_stages_audited=True,
        full_model_qualified=False, independent_root_qualification_decision_required=True,
        automatic_best_promotion=False, binary_contents_rehashed_here=False)
    result_ref = once(root / 'independent-final-audit.json', result)
    receipt = once(stage_receipt(spec, 25), dict(index=25, arm='external', stage=STAGES[-1][1], case=spec['case'],
        result=result_ref, completed=True, full_model_qualified=False, independent_root_qualification_decision_required=True))
    storage(spec, terminal=True)
    return receipt
