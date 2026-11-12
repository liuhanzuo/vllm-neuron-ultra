"""Fixed three-pass full CPU successor, called only by retained LFM owner.

No lifecycle entrypoint, native import, history read or execution on import.
The existing retained-descriptor outer owns the whole 30,000-second window.
The original pilot's child identity/termination and write/import audit are
reused; this module only adds the fixed scientific stages and their receipts.
"""
from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import lfm3_full_cpu_known_warnings_v1 as known_warnings
import run_lfm3_typed_writer_cpu_pilot_v3 as legacy
import lfm3_typed_fixture_ledger_leaf_v1 as fixtures
from lfm3_nki05_compact_writeback_v1 import CompactWriteback

ROOT = Path(__file__).resolve().parent
OLD_PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
NEW_PYTHON = '/home/ubuntu/venvs/vllm_neuron_053_py312/bin/python'
STAGES = ('original_fixture', 'new_fixture', 'controls', 'candidate')
PHASES = ('regression_cpu', 'extended_cpu')
BATCH_SECONDS, PHASE_SECONDS, WORKER_SECONDS = 1800, 14400, 30000


def require(value, message):
    if not value:
        raise RuntimeError(message)


def matrices():
    values = fixtures.validate_matrices({phase: legacy.p.cases(phase) for phase in PHASES})
    require(all(fixtures.original_batches(values[phase]) == legacy.p.batches(phase) for phase in PHASES),
            'Original 3716 cases and every original batch boundary retained')
    return values


def schedule():
    result = []
    for stage in STAGES:
        for phase in (('extended_cpu',) if stage == 'controls' else PHASES):
            bounds = [[0, 1]] if stage == 'controls' else legacy.p.batches(phase)
            result.append(dict(stage=stage, phase=phase, bounds=bounds))
    require(sum(len(row['bounds']) for row in result) == 874, 'Three full291 passes plus one three-arm control child')
    return result


def protocol_manifest():
    return dict(stages=list(STAGES),phases=list(PHASES),full_cases=3716,full_batches=291,
                full_passes=3,control_children=1,total_children=874,
                case_sha256={phase:value[2] for phase,value in fixtures.MATRICES.items()},
                batch_seconds=BATCH_SECONDS,phase_seconds=PHASE_SECONDS,worker_seconds=WORKER_SECONDS,
                old_python=OLD_PYTHON,new_python=NEW_PYTHON,
                legacy_runtime_plan=dict(path=str(legacy.p.RUNTIME_PLAN),sha256=legacy.p.RUNTIME_PLAN_SHA),
                actual_controls_before_candidate=True,stop_first_failure=True,
                no_device_trace_model_performance=True)


def command(host, admission_ref, stage, phase, begin, end):
    require(stage in STAGES and phase in PHASES, 'Fixed finite CPU stage')
    python = OLD_PYTHON if stage == 'original_fixture' else NEW_PYTHON
    return [python, '-B', host.SELF, '--mode', 'batch', '--admission', admission_ref['path'],
            '--admission-sha256', admission_ref['sha256'], '--stage', stage, '--phase', phase,
            '--begin', str(begin), '--end', str(end)]


def child_environment(host, root, attempt, stage):
    result = host.environment(root, attempt)
    prefix = Path(OLD_PYTHON if stage == 'original_fixture' else NEW_PYTHON).parent
    result.update(PATH=str(prefix) + ':/usr/bin:/bin', TMPDIR=str(root / stage / 'tmp'))
    return result


def result_path(root, stage, phase, begin, end):
    return root / stage / phase / f'{begin:05d}-{end:05d}.json'


def launch_path(root, stage, phase, begin, end):
    return root / stage / phase / f'{begin:05d}-{end:05d}-launch-owner.json'


@contextmanager
def legacy_child_binding(host, root, spec, admission_ref, admission, owner):
    """Bind only existing original child identity/termination helpers, no new lock."""
    parent = dict(owner, cgroup=owner['cgroup'].removeprefix('0::'))
    def read(path):
        require(Path(path) == root / 'worker-owner.json', 'Original helper reads only acknowledged parent')
        require(host.p.read(path) == owner, 'Current actual parent owner unchanged')
        return parent
    def service():
        return host.service_identity(spec, admission_ref, admission['attempt_id'], host.properties(spec['unit']), owner)
    with patch.multiple(legacy.p, DEST=root, read=read, exact_service=service):
        yield


def scientific_record(case, fixture, operands, expected, actual):
    """Same strict gates as original check_case; no numerical normalization."""
    before = [row['sha256'] for row in fixture['inputs']]
    after = [hashlib.sha256(value.tobytes()).hexdigest() for value in operands]
    actual_bytes, expected_bytes = actual.tobytes(), expected.tobytes()
    same_shape = actual.shape == operands[0].shape
    same_dtype = actual.dtype == operands[0].dtype
    equal, unchanged = actual_bytes == expected_bytes, before == after
    width = 2 if case['dtype'] == 'bfloat16' else 4
    mismatch = []
    if not equal:
        for offset in range(0, min(len(actual_bytes), len(expected_bytes)), width):
            if actual_bytes[offset:offset+width] != expected_bytes[offset:offset+width]:
                mismatch.append(dict(word_offset=offset//width, actual_hex=actual_bytes[offset:offset+width].hex(),
                                     expected_hex=expected_bytes[offset:offset+width].hex()))
                if len(mismatch) == 16:
                    break
    return dict(id=case['id'], case=case, fixture=fixture, passed=same_shape and same_dtype and equal and unchanged,
                dtype_equal=same_dtype, shape_equal=same_shape, bitwise_equal=equal, inputs_unchanged=unchanged,
                actual_dtype=str(actual.dtype), expected_dtype=str(operands[0].dtype),
                actual_shape=list(actual.shape), expected_shape=list(operands[0].shape),
                actual_sha256=hashlib.sha256(actual_bytes).hexdigest(),
                expected_sha256=hashlib.sha256(expected_bytes).hexdigest(), input_sha256=before,
                input_after_sha256=after, actual_bytes=len(actual_bytes), expected_bytes=len(expected_bytes),
                first16_mismatches=mismatch)


def validate_report(report, stage, phase, begin, end, expected_cases, admission_ref):
    require(report['complete'] is report['passed'] is True and report['admission'] == admission_ref
            and report['stage'] == stage and report['phase'] == phase
            and report['begin'] == begin and report['end'] == end,
            'Complete exact finite child result')
    require(report['full_model_qualified'] is report['new_speedup_claim'] is False, 'CPU-only result scope')
    known_warnings.validate_receipt(report['warning_log'], 'batch')
    if stage == 'controls':
        require(report['controls_validation'] == fixtures.validate_small_controls(report['arms']), 'Actual three-arm control receipts')
        return
    checks = report['checks']
    require([row['case'] for row in checks] == expected_cases, 'Every original case actually reported in order')
    require(all(row['phase'] == phase and row['id'] == row['case']['id'] for row in checks), 'Original case identities')
    for row,case in zip(checks,expected_cases):
        require(set(row)=={'phase','id','case','inputs','expected'} and len(row['inputs'])==3,
                'Complete original fixture metadata only')
        shapes=([case['capacity'],case['width']],[len(case['slots']),case['width']],
                [len(case['slots'])],[case['capacity'],case['width']])
        for descriptor,dtype,shape in zip(row['inputs']+[row['expected']],
                (case['dtype'],case['dtype'],'int32',case['dtype']),shapes):
            count=1
            for dimension in shape: count*=dimension
            require(set(descriptor)=={'dtype','shape','bytes','sha256'} and descriptor['dtype']==dtype
                    and descriptor['shape']==shape and descriptor['bytes']==count*(2 if dtype=='bfloat16' else 4)
                    and type(descriptor['sha256']) is str and len(descriptor['sha256'])==64
                    and set(descriptor['sha256'])<=set('0123456789abcdef'), 'Every complete typed fixture hash descriptor')
    if stage in ('new_fixture', 'candidate'):
        require(report['original_fixture_match'] is True, 'All four complete original hashes matched')
    if stage == 'candidate':
        require(len(report['scientific']) == end-begin and len(report['extractions']) == end-begin,
                'Every case has numerical and extraction results')
        for scientific, fixture, extraction in zip(report['scientific'], checks, report['extractions']):
            require(scientific['case'] == fixture['case'] and scientific['fixture'] == fixture
                    and all(scientific[key] is True for key in ('passed','dtype_equal','shape_equal','bitwise_equal','inputs_unchanged'))
                    and scientific['actual_sha256'] == scientific['expected_sha256'] == fixture['expected']['sha256']
                    and scientific['input_sha256'] == scientific['input_after_sha256'] == [x['sha256'] for x in fixture['inputs']]
                    and scientific['first16_mismatches'] == [], 'Original strict shape/dtype/all-byte/all-input acceptance')
            require(extraction['phase'] == phase and extraction['id'] == fixture['id']
                    and extraction['mode'] == 'precise_bf16_writeback' and extraction['extraction_calls'] == 4
                    and extraction['substitutions'] == (2 if fixture['case']['dtype'] == 'bfloat16' else 0),
                    'Every simulator call has bounded complete extraction inventory')
        probe = report['probe']
        require(probe['installed'] is probe['restored'] is True and probe['failed'] is False
                and probe['completed_cases'] == end-begin and probe['extraction_calls'] == 4*(end-begin)
                and probe['substitutions'] == sum(row['substitutions'] for row in report['extractions']),
                'Restored compact simulator adapter covers the entire original batch')


def stage_complete(host, root, stage, admission_ref):
    result = host.p.read(root / (stage + '-complete.json'))
    require(result['complete'] is result['passed'] is True and result['stage'] == stage
            and result['admission'] == admission_ref, 'Actual prior whole stage completed')
    if stage == 'controls':
        require(result['cases'] == 1 and result['batches'] == 1, 'One extra control child, outside original suite')
    else:
        require(result['cases'] == 3716 and result['batches'] == 291, 'Every original case/batch in prior stage')
    for ref in result['phases']:
        phase = host.p.read(ref['path'], ref['sha256'])
        require(phase['complete'] is phase['passed'] is True and phase['admission'] == admission_ref
                and phase['stage'] == stage and all(row['returncode'] == 0 and row['passed'] is True for row in phase['outcomes']),
                'Actual parent-observed exit zero for every prior child')
    return result


def execute_batch_science(host, root, stage, phase, begin, end, admission_ref, admit, result):
    """Called only after exact owner/source/environment gates and original IO audit."""
    admit()
    host.bytecode_guard(root)
    import numpy as np
    import ml_dtypes
    wanted = {'numpy': '1.26.4', 'ml_dtypes': '0.5.4'} if stage == 'original_fixture' else {
        'numpy': '2.5.4', 'ml_dtypes': '0.6.0'}
    versions = {'numpy': np.__version__, 'ml_dtypes': ml_dtypes.__version__}
    require(versions == wanted, 'Actual original/migrated native versions, no fallback')
    result['versions'] = versions
    values = matrices()
    bound = fixtures.FixtureLedger(admit, legacy.numpy_inputs, values, versions)
    original = None
    if stage != 'original_fixture':
        stage_complete(host, root, 'original_fixture', admission_ref)
        bounds = (0, 16) if stage == 'controls' else (begin, end)
        original = host.p.read(result_path(root, 'original_fixture', phase, *bounds))
        validate_report(original, 'original_fixture', phase, *bounds, values[phase][bounds[0]:bounds[1]], admission_ref)
    if stage in ('controls', 'candidate'):
        stage_complete(host, root, 'new_fixture', admission_ref)
        from importlib.metadata import version
        require(version('nki') == '0.5.0+28631259367.ga768afa6'
                and version('neuronx-cc') == '2.26.6360.0+6f180f47', 'Original r5 SDK/compiler identities')
        for path, wanted_sha in host.diagnostic.PINS.items():
            require(host.p.sha(path) == wanted_sha, 'All original r5 diagnostic input/source pins retained')
        import nki
        from slot_cache_nki05_typed_selector_v1 import make_kernel
        kernel = make_kernel()
        simulator = sys.modules['nki._simulator']
    if stage == 'candidate':
        stage_complete(host, root, 'controls', admission_ref)
    if stage == 'controls':
        result['arms'] = []
        case = values['extended_cpu'][0]
        for name, mode in (('original','original'), ('candidate','precise_bf16_writeback'), ('restored_original','original')):
            operands = legacy.numpy_inputs(copy.deepcopy(case), phase)
            fixture, expected = fixtures.fixture_record(case, phase, operands)
            fixtures.verify_same_fixture(original['checks'][0], fixture)
            probe = CompactWriteback(mode)
            arm = dict(name=name, fixture=fixture)
            result['arms'].append(arm)
            try:
                probe.install(simulator, admit)
                probe.begin_case(phase, 0, operands)
                actual = nki.simulate(kernel)(*operands)
                arm['extraction'] = probe.end_case()
                row = scientific_record(case, fixture, operands, expected, actual)
                arm.update({key: row[key] for key in ('shape_equal','dtype_equal','bitwise_equal',
                           'actual_sha256','inputs_unchanged','input_after_sha256')})
            finally:
                if probe.installed and not probe.restored:
                    probe.restore()
                arm['probe'] = probe.report()
            del operands, expected, actual
        result['controls_validation'] = fixtures.validate_small_controls(result['arms'])
        return
    probe = CompactWriteback('precise_bf16_writeback') if stage == 'candidate' else None
    if probe is not None:
        result.update(scientific=[], extractions=[])
    try:
        if probe is not None:
            probe.install(simulator, admit)
        for fixture, operands, expected in bound.batch(phase, begin, end):
            admit()
            result['checks'].append(fixture)
            if original is not None:
                fixtures.verify_same_fixture(original['checks'][fixture['id']-begin], fixture)
            if probe is not None:
                probe.begin_case(phase, fixture['id'], operands)
                actual = nki.simulate(kernel)(*operands)
                result['extractions'].append(probe.end_case())
                scientific = scientific_record(fixture['case'], fixture, operands, expected, actual)
                result['scientific'].append(scientific)
                require(scientific['passed'], 'Original strict scientific check failed; stop first')
                del actual
            del operands, expected
        result['original_fixture_match'] = original is not None
    finally:
        if probe is not None:
            if probe.installed and not probe.restored:
                probe.restore()
            result['probe'] = probe.report()


def run_batch(host, admission_ref, stage, phase, begin, end):
    require(stage in STAGES and phase in PHASES and type(begin) is type(end) is int, 'Finite batch coordinates')
    values = matrices()
    allowed = [[0,1]] if stage == 'controls' and phase == 'extended_cpu' else legacy.p.batches(phase)
    require([begin,end] in allowed and (stage != 'controls' or phase == 'extended_cpu'), 'One original batch or separate control')
    a = host.history.serial.sources.load(admission_ref)
    spec = host.read_bundle(a['spec'], a['review']); root = Path(spec['output_root'])
    require(admission_ref['path'] == str(root/'admission.json') and a['schema'] == 'lfm3_nki05_full_cpu_admission_v1'
            and a['lock_identity'] == host.LOCK and a['runtime_admitted'] is a['predecessors_checked'] is True,
            'Actual outer-produced admission only')
    require(sys.executable == (OLD_PYTHON if stage == 'original_fixture' else NEW_PYTHON)
            and sys.dont_write_bytecode and __debug__, 'Exact admitted Python with -B')
    require(all(os.environ.get(key) == value for key, value in child_environment(host,root,a['attempt_id'],stage).items()),
            'Exact clean child environment')
    require(os.sched_getaffinity(0) == {6,7} and os.getpriority(os.PRIO_PROCESS,0) == 19, 'Retained CPU/nice limits')
    for limit, cap in ((resource.RLIMIT_AS,12*2**30),(resource.RLIMIT_CPU,1800),(resource.RLIMIT_FSIZE,host.FILE_CAP)):
        resource.setrlimit(limit,(cap,cap))
    process = host.p.proc(os.getpid())
    require(process['argv'] == command(host,admission_ref,stage,phase,begin,end), 'Actual exact child argv')
    ack_path = launch_path(root,stage,phase,begin,end)
    deadline = time.monotonic()+60
    while not ack_path.exists():
        host.descriptor_proof(a['outer_owner'],a['retained_fd'])
        require(time.monotonic()<deadline and os.getppid()>1,'Actual parent ACK deadline')
        time.sleep(.05)
    ack = host.p.read(ack_path)
    parent = host.p.read(root/'worker-owner.json')
    require(ack['pid']==process['pid'] and ack['start_ticks']==process['start_ticks']
            and ack['argv']==process['argv'] and '0::'+ack['cgroup']==process['cgroup']
            and ack['parent']==dict(parent,cgroup=parent['cgroup'].removeprefix('0::'))
            and process['ppid']==parent['pid'], 'Parent acknowledged actual original-helper child identity')
    host.service_identity(spec,admission_ref,a['attempt_id'],host.properties(spec['unit']),parent)
    runtime = host.p.read(root/'runtime-auth.json')
    require(runtime['complete'] is True and runtime['admission']==admission_ref
            and runtime['legacy_distributions']==7, 'Original native RECORD authentication completed by owned coordinator')
    path = result_path(root,stage,phase,begin,end)
    result = dict(schema='lfm3_nki05_full_cpu_batch_v1',stage=stage,phase=phase,begin=begin,end=end,
                  admission=admission_ref,pid=os.getpid(),start_ticks=process['start_ticks'],
                  complete=False,passed=False,checks=[],full_model_qualified=False,new_speedup_claim=False)
    began = time.monotonic()
    def admit():
        host.descriptor_proof(a['outer_owner'],a['retained_fd'])
        require(host.p.proc(os.getpid())==process and host.p.proc(parent['pid'])=={key:parent[key] for key in ('pid','start_ticks','ppid','argv','cgroup')},
                'Actual child and owning coordinator remain exact')
        host.check_latest(a['latest_handoff_identity'])
        require(time.monotonic()-began < BATCH_SECONDS, 'Original per-batch wall deadline')
        require({path:host.p.sha(path) for path in spec['canonical_sha256']}==spec['canonical_sha256'], 'Canonical snapshots unchanged')
        host.verify_sources(spec)
        return True
    failure = None
    try:
        admit()
        with patch.object(legacy.p,'DEST',root/stage):
            # Retained original no-spawn/no-network/no-device/output-only audit.
            legacy.p.audit_child(phase,(begin,end))
            try:
                execute_batch_science(host,root,stage,phase,begin,end,admission_ref,admit,result)
                result.update(complete=True,passed=True,warning_log=host.warning_receipt('batch'))
                validate_report(result,stage,phase,begin,end,values[phase][begin:end],admission_ref)
            except BaseException as error:
                failure=error
                result.update(complete=False,passed=False,error_type=type(error).__name__,error=str(error)[:2048],original_failure_preserved=True)
            finally:
                result['seconds']=time.monotonic()-began
                result['warning_log']=host.warning_receipt('batch')
                host.once(path,result)
    except BaseException:
        raise
    if failure is not None:
        raise failure


def run_one_batch(host, spec, admission_ref, admission, owner, root, stage, phase, begin, end, admit, timeout):
    child=None; child_owner=None; code=None; failure=None; secondary=[]
    argv=command(host,admission_ref,stage,phase,begin,end)
    path=result_path(root,stage,phase,begin,end)
    started=time.monotonic()
    with legacy_child_binding(host,root,spec,admission_ref,admission,owner):
        try:
            admit()
            child=subprocess.Popen(argv,cwd=root,env=child_environment(host,root,admission['attempt_id'],stage),start_new_session=True)
            child_owner=legacy.child_identity(child,argv)
            host.once(launch_path(root,stage,phase,begin,end),child_owner)
            deadline=time.monotonic()+min(BATCH_SECONDS,timeout)
            while True:
                admit()
                try:
                    code=child.wait(timeout=min(2,max(.01,deadline-time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    require(time.monotonic()<deadline,'Original finite batch timeout')
            require(code==0,'Actual finite child nonzero exit preserved')
            report=host.p.read(path)
            validate_report(report,stage,phase,begin,end,legacy.p.cases(phase)[begin:end],admission_ref)
            require(report['pid']==child.pid and report['start_ticks']==child_owner['start_ticks'], 'Report belongs to actual exited child')
        except BaseException as error:
            failure=error
        finally:
            if child is not None:
                try:
                    require(child_owner is not None,'Missing launch identity: original systemd owner must drain all children')
                    legacy.stop_child(child,child_owner)
                    if code is None:
                        code=child.returncode
                except BaseException as error:
                    secondary.append(str(error)[:1024]); failure=failure or error
    outcome=dict(stage=stage,phase=phase,begin=begin,end=end,pid=child.pid if child else None,
                 start_ticks=child_owner['start_ticks'] if child_owner else None,returncode=code,
                 passed=failure is None,seconds=time.monotonic()-started,
                 report=host.reference(path) if path.exists() else None,
                 error=str(failure)[:2048] if failure else None,secondary_errors=secondary)
    return outcome, failure


def require_full_success(result):
    known_warnings.validate_receipt(result['warning_log'], 'worker')
    require(result['schema']=='lfm3_nki05_full_cpu_result_v1' and result['complete'] is result['passed'] is True,
            'All full CPU passes and actual controls completed')
    require(result['stage_counts']=={stage:dict(cases=1,batches=1) if stage=='controls' else dict(cases=3716,batches=291) for stage in STAGES},
            'Three complete original matrices and one separate actual control child')
    require(result['cpu_full_gate_passed'] is result['simulator_writeback_workaround_required'] is True
            and result['original_simulator_failure_preserved'] is True, 'Explicit modified simulator, original failure not rewritten')
    require(all(result[key] is False for key in ('device_behavior_proven','alias_gate_passed','full_model_qualified','new_speedup_claim','automatic_successor_launch')),
            'No device/alias/model or performance promotion')


def audit_full_success(host,result):
    """Independent outer reread of all original batches, gates and actual exits."""
    require_full_success(result)
    admission_ref=result['admission'];root=Path(admission_ref['path']).parent
    require([row['path'] for row in result['stages']]==[str(root/(stage+'-complete.json')) for stage in STAGES],
            'All original stages in exact order, bound to this output tree')
    values=matrices();parent=host.p.read(root/'worker-owner.json')
    for stage,ref in zip(STAGES,result['stages']):
        summary=host.p.read(ref['path'],ref['sha256'])
        require(summary==stage_complete(host,root,stage,admission_ref),'Immutable exact stage summary')
        planned=[item for item in schedule() if item['stage']==stage]
        require([item['path'] for item in summary['phases']]==[str(root/stage/(item['phase']+'-outcomes.json')) for item in planned],
                'Complete original phase order and exact outcome paths')
        for plan,phase_ref in zip(planned,summary['phases']):
            phase=plan['phase'];phase_result=host.p.read(phase_ref['path'],phase_ref['sha256'])
            outcomes=phase_result['outcomes']
            require([(row['begin'],row['end']) for row in outcomes]==[tuple(pair) for pair in plan['bounds']],
                    'Every parent-observed original batch exit retained in order')
            for row in outcomes:
                begin,end=row['begin'],row['end'];path=result_path(root,stage,phase,begin,end)
                require(row['stage']==stage and row['phase']==phase and row['passed'] is True
                        and row['returncode']==0 and row['error'] is None and row['secondary_errors']==[]
                        and row['report']==host.reference(path), 'Exact real successful child exit and full report pin')
                report=host.p.read(path,row['report']['sha256'])
                validate_report(report,stage,phase,begin,end,values[phase][begin:end],admission_ref)
                ack=host.p.read(launch_path(root,stage,phase,begin,end))
                require(ack['pid']==report['pid']==row['pid'] and ack['start_ticks']==report['start_ticks']==row['start_ticks']
                        and ack['parent']==dict(parent,cgroup=parent['cgroup'].removeprefix('0::'))
                        and ack['argv']==command(host,admission_ref,stage,phase,begin,end), 'Actual acknowledged child identity')
                if stage in ('new_fixture','candidate'):
                    old=host.p.read(result_path(root,'original_fixture',phase,begin,end))
                    require(report['checks']==old['checks'],'Every original input/oracle byte hash matches in all full passes')
                if stage=='controls':
                    old=host.p.read(result_path(root,'original_fixture','extended_cpu',0,16))
                    require(all(arm['fixture']==old['checks'][0] for arm in report['arms']), 'Actual controls use exact original generated fixture')
    return True


def execute_full_cpu(host,admission_ref,owner,root,admit):
    a=host.history.serial.sources.load(admission_ref)
    spec=host.read_bundle(a['spec'],a['review'])
    result=dict(schema='lfm3_nki05_full_cpu_result_v1',complete=False,passed=False,admission=admission_ref,
                stages=[],stage_counts={},cpu_full_gate_passed=False,simulator_writeback_workaround_required=True,
                original_simulator_failure_preserved=True,device_behavior_proven=False,alias_gate_passed=False,
                full_model_qualified=False,new_speedup_claim=False,automatic_successor_launch=False)
    try:
        admit()
        host.bytecode_guard(root)
        # Existing original seven-distribution RECORD verifier, unchanged. It
        # reads native runtime files only here, after actual owned admission.
        import lfm3_movement_device_protocol_v3 as runtime
        require(host.p.sha(legacy.p.RUNTIME_PLAN)==legacy.p.RUNTIME_PLAN_SHA, 'Frozen original runtime plan')
        old_plan=host.p.read(legacy.p.RUNTIME_PLAN)
        with patch.object(runtime,'WORKER_PYCACHE',root/'tmp'/'empty-pycache'):
            proofs=runtime.verify_runtime_distributions(old_plan['runtime'])
        require(len(proofs)==7,'All original seven runtime distributions authenticated')
        host.once(root/'runtime-auth.json',dict(complete=True,admission=admission_ref,owner=owner,
                    legacy_distributions=7,proofs=proofs,warning_log=host.warning_receipt('worker'),model_payloads_read=False,new_speedup_claim=False))
        for stage in STAGES:
            (root/stage).mkdir();(root/stage/'tmp').mkdir()
            phase_refs=[];cases_done=batches_done=0
            for item in [row for row in schedule() if row['stage']==stage]:
                phase=item['phase'];(root/stage/phase).mkdir()
                phase_started=time.monotonic();outcomes=[];failure=None
                try:
                    for begin,end in item['bounds']:
                        remaining=PHASE_SECONDS-(time.monotonic()-phase_started)
                        require(remaining>0,'Original full phase deadline')
                        outcome,error=run_one_batch(host,spec,admission_ref,a,owner,root,stage,phase,begin,end,admit,remaining)
                        outcomes.append(outcome)
                        if error is not None:
                            raise error
                    require([(row['begin'],row['end']) for row in outcomes]==[tuple(pair) for pair in item['bounds']],
                            'All original finite batch exits in exact order')
                except BaseException as error:
                    failure=error
                finally:
                    path=root/stage/(phase+'-outcomes.json')
                    host.once(path,dict(schema='lfm3_nki05_full_cpu_phase_v1',stage=stage,phase=phase,
                         admission=admission_ref,complete=failure is None,passed=failure is None,
                         outcomes=outcomes,seconds=time.monotonic()-phase_started,
                         original_failure_preserved=failure is not None))
                if failure is not None:
                    raise failure
                phase_refs.append(host.reference(path));batches_done+=len(outcomes)
                cases_done+=sum(row['end']-row['begin'] for row in outcomes)
            summary=dict(stage=stage,admission=admission_ref,complete=True,passed=True,
                         cases=cases_done,batches=batches_done,phases=phase_refs)
            host.once(root/(stage+'-complete.json'),summary)
            stage_complete(host,root,stage,admission_ref)
            result['stages'].append(host.reference(root/(stage+'-complete.json')))
            result['stage_counts'][stage]=dict(cases=cases_done,batches=batches_done)
        result.update(complete=True,passed=True,cpu_full_gate_passed=True,warning_log=host.warning_receipt('worker'))
        require_full_success(result);admit()
        result['warning_log']=host.warning_receipt('worker')
        host.once(root/'diagnostic-result.json',result)
        host.once(root/'worker-result.json',dict(complete=True,passed=True,owner=owner,admission=admission_ref,
                    diagnostic=host.reference(root/'diagnostic-result.json'),self_exit_claimed=False,
                    full_model_qualified=False,new_speedup_claim=False))
    except BaseException as error:
        if not (root/'diagnostic-result.json').exists():
            result.update(complete=False,passed=False,cpu_full_gate_passed=False,error_type=type(error).__name__,error=str(error)[:2048],
                          warning_log=host.warning_receipt('worker'))
            host.once(root/'diagnostic-result.json',result)
        host.report_failure(root/'worker-failure.json',error,owner=owner,admission=admission_ref)
        raise
