"""JSON-only verifier of the future complete actual LFM r2 CPU predecessor.

Copied pure scientific validators are ordinary reviewed source, no AST extraction,
native libraries, simulator adapters, weights or runtime operations on import.
"""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import lfm3_full_cpu_known_warnings_v1 as known_warnings
from lfm3_nki05_device_matrix_v1 import cpu_cases

ROOT=Path(__file__).resolve().parent
CPU_ROOT=Path('/mnt/nvme/lfm3-nki05-full-cpu-r2-20261112')
CPU_SELF=str(ROOT/'run_lfm3_nki05_full_cpu_serial_v3.py')
CPU_SPEC=dict(path=str(ROOT/'research/lfm3-full-cpu-r2-20261112-spec.json'),sha256='955c2267d1ac27bb1af5f680b4e0acefa4355c05bdfa874cf4f46f4381116630')
CPU_ADMISSION=dict(path=str(CPU_ROOT/'admission.json'),sha256='bcc175f2aeb21fdfb0fd4c0fa585336ef6b7996c3666b6596ffc4a78f7b026c5')
CPU_SESSION=42892
STAGES=('original_fixture','new_fixture','controls','candidate')
PHASES=('regression_cpu','extended_cpu')
OLD_PYTHON='/home/ubuntu/venvs/vllm_neuron/bin/python'
NEW_PYTHON='/home/ubuntu/venvs/vllm_neuron_053_py312/bin/python'

MATRICES = {
    'regression_cpu': (1648, 123, '68e6ead99c215943ecb8123cefb7d2d2eb37b4f616156929276e2958c64090dc'),
    'extended_cpu': (2068, 168, 'b08c2282d952d4ce2c33e7c969faac98ee45d7f190a225e0c5798fc21eee328c'),
}
OLD_FIRST_HASHES = (
    '5781e3d8081dc2fb9a8d4c3593c858dab77b09ed734bed5662d3d7d2a5dcb98d',
    '94555ec4d65509f60933cab1609a615e0d8e0b73a2c9a8a09c5657e5e6fe3f39',
    'df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119',
    'fd8dea2bf754093760877b1a8cdd6b455b320b268e09bff488826fef4482b4c5',
)
def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def original_batches(values):
    """Same complete original batching; callers compare with original plan too."""
    result, begin = [], 0
    while begin < len(values):
        end = begin + 1
        if values[begin]['width'] < 524288:
            while end < len(values) and end-begin < 16 and values[end]['width'] < 524288:
                end += 1
        result.append([begin, end])
        begin = end
    return result


def validate_matrices(matrices):
    require(type(matrices) is dict and set(matrices) == set(MATRICES), 'Both original complete matrices')
    checked = copy.deepcopy(matrices)
    for phase, (count, batches, wanted) in MATRICES.items():
        values = checked[phase]
        require(type(values) is list and len(values) == count and digest(values) == wanted,
                'Every original case, order, seed, dtype, mapping and width is unchanged')
        require([value['id'] for value in values] == list(range(count))
                and len(original_batches(values)) == batches, 'Original identifiers and batch count')
    return checked


def regression_protocol():
    cases = []
    for dtype_index, dtype in enumerate(['bfloat16', 'float32']):
        for rows in range(1, 5):
            for mapping in itertools.product([-1, 0, 1, 2, 3], repeat=rows):
                cases.append({'kind': 'exhaustive', 'dtype': dtype, 'width': 256,
                              'slots': list(mapping), 'seed': 20261031 + dtype_index * 10 + rows})
    assert len(cases) == 1560
    for dtype in ['bfloat16', 'float32']:
        for mapping in [[4, 0, -1, 2], [0, 9, 0, 0], [-9, -9, -9, -9], [4, 5, 6, 7]]:
            cases.append({'kind': 'extra', 'dtype': dtype, 'width': 384,
                          'slots': mapping, 'seed': 20261031 + len(cases)})
        for width in [128, 7168, 32768, 65536, 131072, 262144, 524288, 2097152]:
            for mapping in [[3, 1, 0, 2], [2, 2, 0, 3], [-1, -1, -1, -1], [3], [-1]]:
                cases.append({'kind': 'extra', 'dtype': dtype, 'width': width,
                              'slots': mapping, 'seed': 20261031 + len(cases)})
    assert len(cases) == 1648
    return [{'id': index, **row} for index, row in enumerate(cases)]


def validate_small_controls(arms):
    """Required receipt gate before any full CPU candidate batch is admitted.

    Caller must generate real records with fixture_record and actual strict
    output/input checks, close each CompactWriteback case, then restore it.
    These control receipts are additional to, never counted in, the 3,716.
    """
    require(type(arms) is list and len(arms) == 3, 'Original, candidate, restored-original controls')
    require([row['name'] for row in arms] == ['original', 'candidate', 'restored_original'], 'Exact control order')
    for index, row in enumerate(arms):
        fixture = row['fixture']
        require(fixture['phase'] == 'extended_cpu' and fixture['id'] == 0 and fixture['case'] == dict(id=0, seed=20261103,
                kind='selector_coverage', capacity=8, dtype='bfloat16', width=256, slots=[0]),
                'Exact original v5 first selector case')
        require(tuple(x['sha256'] for x in fixture['inputs']) + (fixture['expected']['sha256'],) == OLD_FIRST_HASHES,
                'Exact original first-case input and oracle bytes')
        descriptors = [dict(dtype=dtype, shape=shape, bytes=size, sha256=sha)
                       for dtype, shape, size, sha in zip(
                           ('bfloat16', 'bfloat16', 'int32', 'bfloat16'),
                           ([8, 256], [1, 256], [1], [8, 256]),
                           (4096, 512, 4, 4096), OLD_FIRST_HASHES)]
        require(fixture['inputs'] + [fixture['expected']] == descriptors,
                'Exact first-case metadata and full hashes, no substituted fixture descriptions')
        require(row['shape_equal'] is row['dtype_equal'] is row['bitwise_equal'] is True
                and row['actual_sha256'] == OLD_FIRST_HASHES[3], 'Selector output remains exactly correct')
        require(row['inputs_unchanged'] is (index == 1), 'Original/restored fail; only candidate inputs stay unchanged')
        after = row['input_after_sha256']
        require(type(after) is list and len(after) == 3
                and all(type(x) is str and len(x) == 64 and set(x) <= set('0123456789abcdef') for x in after),
                'Three complete post-call input hashes')
        require(after == list(OLD_FIRST_HASHES[:3]) if index == 1 else
                after[0] != OLD_FIRST_HASHES[0] and after[1] != OLD_FIRST_HASHES[1] and after[2] == OLD_FIRST_HASHES[2],
                'Both original BF16 inputs change; int32 slots never change')
        mode = 'precise_bf16_writeback' if index == 1 else 'original'
        probe, event = row['probe'], row['extraction']
        require(probe == dict(mode=mode, installed=True, restored=True, failed=False, completed_cases=1,
                             extraction_calls=4, substitutions=2 if index == 1 else 0,
                             installed_sdk_source_modified=False), 'Actual completed extraction and restoration receipt')
        require(event['phase'] == 'extended_cpu' and event['id'] == 0 and event['mode'] == mode
                and event['extraction_calls'] == 4 and event['substitutions'] == (2 if index == 1 else 0),
                'Exactly four bounded extractions per control')
        expected_events = []
        for position, (name, descriptor) in enumerate(zip(('cache', 'updates', 'slots', 'output'), descriptors)):
            changed = index == 1 and position < 2
            out_dtype = 'float32' if index != 1 and position < 2 else descriptor['dtype']
            expected_events.append(dict(operand=name, tensor_dtype=descriptor['dtype'], extracted_dtype=out_dtype,
                shape=descriptor['shape'], bytes=descriptor['bytes'] * (2 if index != 1 and position < 2 else 1),
                requested_precise=True if position == 3 else 'omitted',
                effective_precise=changed or position == 3, substitution=changed))
        require(event['events'] == expected_events, 'Full exact ordered extraction inventory')
    require(arms[0]['fixture'] == arms[1]['fixture'] == arms[2]['fixture'], 'Identical fixtures across all three controls')
    require(arms[0]['input_after_sha256'] == arms[2]['input_after_sha256']
            and arms[0]['extraction'] == arms[2]['extraction'], 'Removal reproduces original failure exactly')
    return dict(complete=True, original_failure_preserved=True, restored_original_reproduced=True,
                candidate_selector_control_passed=True, controls_sha256=digest(arms),
                full3716_passed=False, device_passed=False, new_speedup_claim=False)


def validate_report(report, stage, phase, begin, end, expected_cases, admission_ref):
    require(report['complete'] is report['passed'] is True and report['admission'] == admission_ref
            and report['stage'] == stage and report['phase'] == phase
            and report['begin'] == begin and report['end'] == end,
            'Complete exact finite child result')
    require(report['full_model_qualified'] is report['new_speedup_claim'] is False, 'CPU-only result scope')
    known_warnings.validate_receipt(report['warning_log'], 'batch')
    if stage == 'controls':
        require(report['controls_validation'] == validate_small_controls(report['arms']), 'Actual three-arm control receipts')
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



def matrices():
    return validate_matrices(dict(regression_cpu=[dict(row,capacity=4) for row in regression_protocol()],extended_cpu=cpu_cases()))


def schedule():
    values=matrices()
    return [dict(stage=stage,phase=phase,bounds=[[0,1]] if stage=='controls' else original_batches(values[phase]))
            for stage in STAGES for phase in (('extended_cpu',) if stage=='controls' else PHASES)]


def result_path(root,stage,phase,begin,end):return root/stage/phase/f'{begin:05d}-{end:05d}.json'
def launch_path(root,stage,phase,begin,end):return root/stage/phase/f'{begin:05d}-{end:05d}-launch-owner.json'
def command(admission_ref,stage,phase,begin,end):
    return [OLD_PYTHON if stage=='original_fixture' else NEW_PYTHON,'-B',CPU_SELF,'--mode','batch',
            '--admission',admission_ref['path'],'--admission-sha256',admission_ref['sha256'],
            '--stage',stage,'--phase',phase,'--begin',str(begin),'--end',str(end)]


def validate_binding(binding):
    require(type(binding) is dict and set(binding)=={'spec','admission','result','worker_result','service_exit','outer_result','cleanup','actual_exit'},'Complete fixed r2 predecessor receipts')
    require(binding['spec']==CPU_SPEC and binding['admission']==CPU_ADMISSION,'Actual original r2 spec/admission, never a partial or different run')
    filenames=dict(result='diagnostic-result.json',worker_result='worker-result.json',service_exit='service-exit.json',outer_result='outer-result.json',cleanup='cleanup-drained.json')
    for key,name in filenames.items():require(binding[key]['path']==str(CPU_ROOT/name),'Exact r2 output path')
    require(Path(binding['actual_exit']['path']).parent==ROOT/'research','External actual tool exit recorded by root')
    for ref in binding.values():
        require(set(ref)=={'path','sha256'} and type(ref['sha256']) is str and len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef'),'Pinned exact predecessor metadata')
    return binding


def audit(host,binding):
    """host.p.read/reference are bounded JSON-only operations with stable identities.
    Re-read all 874 real outcomes, ACKs, reports and complete scientific gates.
    Physical gone/cgroup/lock checks remain separate fresh outer/child operations.
    """
    validate_binding(binding)
    load=lambda key:host.p.read(binding[key]['path'],binding[key]['sha256'])
    old_spec=load('spec');admission=load('admission');result=load('result')
    require(admission['spec']==CPU_SPEC and result['admission']==CPU_ADMISSION,'Complete result tied to actual r2 admission')
    require_full_success(result)
    require(old_spec['output_root']==str(CPU_ROOT) and old_spec['runner']==CPU_SELF,'Fixed original complete CPU owner')
    root=CPU_ROOT;admission_ref=CPU_ADMISSION;values=matrices();parent=host.p.read(root/'worker-owner.json')
    require([row['path'] for row in result['stages']]==[str(root/(stage+'-complete.json')) for stage in STAGES],'Exact full stage order')
    children=[]
    for stage,ref in zip(STAGES,result['stages']):
        summary=host.p.read(ref['path'],ref['sha256'])
        require(summary==stage_complete(host,root,stage,admission_ref),'Exact complete stage summary')
        planned=[item for item in schedule() if item['stage']==stage]
        require([item['path'] for item in summary['phases']]==[str(root/stage/(item['phase']+'-outcomes.json')) for item in planned],'Complete original phase order')
        for plan,phase_ref in zip(planned,summary['phases']):
            phase=plan['phase'];outcomes=host.p.read(phase_ref['path'],phase_ref['sha256'])['outcomes']
            require([(row['begin'],row['end']) for row in outcomes]==[tuple(pair) for pair in plan['bounds']],'Every actual original batch exit')
            for row in outcomes:
                begin,end=row['begin'],row['end'];path=result_path(root,stage,phase,begin,end)
                require(row['stage']==stage and row['phase']==phase and row['passed'] is True and row['returncode']==0
                        and row['error'] is None and row['secondary_errors']==[] and row['report']==host.reference(path),'Exact actual successful child exit')
                report=host.p.read(path,row['report']['sha256']);validate_report(report,stage,phase,begin,end,values[phase][begin:end],admission_ref)
                ack=host.p.read(launch_path(root,stage,phase,begin,end))
                require(ack['pid']==report['pid']==row['pid'] and ack['start_ticks']==report['start_ticks']==row['start_ticks']
                        and ack['parent']==dict(parent,cgroup=parent['cgroup'].removeprefix('0::'))
                        and ack['argv']==command(admission_ref,stage,phase,begin,end),'Actual acknowledged parent/PID/start/argv')
                children.append(dict(pid=ack['pid'],start_ticks=ack['start_ticks'],ppid=parent['pid'],argv=ack['argv'],cgroup='0::'+ack['cgroup']))
                if stage in ('new_fixture','candidate'):
                    original=host.p.read(result_path(root,'original_fixture',phase,begin,end))
                    require(report['checks']==original['checks'],'All original fixture/oracle byte hashes identical')
                if stage=='controls':
                    original=host.p.read(result_path(root,'original_fixture','extended_cpu',0,16))
                    require(all(arm['fixture']==original['checks'][0] for arm in report['arms']),'Controls use the same original fixture')
    require(len(children)==874 and len({(x['pid'],x['start_ticks']) for x in children})==874,'All distinct actual child owners')
    worker=load('worker_result');ended=load('service_exit');outer=load('outer_result');cleanup=load('cleanup');external=load('actual_exit')
    require(worker['complete'] is worker['passed'] is True and worker['owner']==parent and worker['admission']==CPU_ADMISSION and worker['diagnostic']==binding['result'],'Exact actual successful worker receipt')
    require(ended['owner']==parent and ended['observed_by_outer']==admission['outer_owner'] and ended['successful'] is True
            and ended['properties']['Result']=='success' and ended['properties']['ExecMainCode']=='1' and ended['properties']['ExecMainStatus']=='0'
            and ended['drain']['physically_drained'] is True,'Actual observed service exit zero and drain')
    require(outer['complete'] is outer['passed'] is True and outer['owner']==admission['outer_owner']
            and outer['service_exit']==binding['service_exit'] and outer['worker_result']==binding['worker_result'],'Successful exact outer result')
    require(cleanup['physically_drained'] is True and cleanup['owner']==parent and cleanup['self_exit_claimed'] is False,'Actual cleanup before release')
    require(external['reviewer']=='/root' and external['actual_tool_exit_observed'] is True and external['spec']==CPU_SPEC
            and external['actual_tool']['kind']=='tool_session' and external['actual_tool']['session_id']==CPU_SESSION
            and external['actual_tool']['exit_code']==0 and type(external['actual_tool']['terminal_chunk']) is str
            and external['actual_tool']['terminal_chunk'],'External actual original outer exit zero, never a live self-claim')
    for owner in (admission['outer_owner'],parent):
        require(any(all(candidate.get(k)==owner[k] for k in ('pid','start_ticks','ppid','argv','cgroup')) for candidate in external['owners']),'External exit retains exact actual owners')
    for value in (worker,outer):require(value['self_exit_claimed'] is value['full_model_qualified'] is value['new_speedup_claim'] is False,'No CPU promotion or self-exit claim')
    return dict(complete=True,passed=True,binding=binding,full_cpu_result=result,children=children,
                owners=[admission['outer_owner'],parent],actual_tool=external['actual_tool'],unit=old_spec['unit'],
                actual_outer_exit_verified=True,device_behavior_proven=False,full_model_qualified=False,new_speedup_claim=False)
