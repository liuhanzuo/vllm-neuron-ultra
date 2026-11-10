"""Freeze current-input/history factorial after actual official-history and storage exits."""
import argparse
import copy
import json
from pathlib import Path
import sys
Q = Path(__file__).resolve().parent
R = Q.parent
sys.path[:0] = [str(R), str(Q)]
from research import prepare_minicpm5_layer1_own_history_runner_v3 as previous
from research import minicpm5_layer1_after_official_history_v1 as official
import minicpm5_layer1_current_history_factorial_cpu_runner_v1 as runner
import minicpm5_layer1_current_history_factorial_cpu_worker_v1 as worker
from prepare_dory_residual_spec_v3 import local_imports
require = runner.p.require


def build():
    parser=argparse.ArgumentParser()
    parser.add_argument('--handoff',required=True);parser.add_argument('--handoff-sha256',required=True)
    args=parser.parse_args(); handoff_ref=dict(path=args.handoff,sha256=args.handoff_sha256)
    require(handoff_ref==official.HANDOFF,'Exact root v420 terminal handoff')
    reader=previous.Reader(); handoff=reader(handoff_ref); h=handoff['official_history']
    binding=dict(schema='minicpm5_layer1_completed_official_history_v1',model_exit=h['actual_exit'],
                 audit_exit=h['independent_audit_exit'])
    completed=official.completed(binding,reader)
    storage_binding=dict(schema='minicpm5_completed_storage_maintenance_v1',handoff=handoff_ref,
        maintenance_exit=handoff['storage_maintenance']['actual_exit'],
        audit_exit=handoff['storage_maintenance']['independent_audit_exit'])
    official.storage_completed(storage_binding,completed,reader)
    spec=copy.deepcopy(completed['spec']);plan=copy.deepcopy(completed['plan'])
    for rank in (0,1):
        for row in completed['records']['rank%d_result'%rank]['records']:
            desc=row['output'];value=dict(sha256=desc['sha256'],identity=desc['identity'])
            require(runner.identity(desc['path'])==value['identity'],'Unchanged previous full output identity')
            require(desc['path'] not in plan['inputs'] or plan['inputs'][desc['path']]==value,'Retain original input descriptor')
            plan['inputs'][desc['path']]=value
    checks = [Q/'minicpm5-layer1-current-history-factorial-checks-20261110-v1.txt',
              Q/'minicpm5-layer1-current-history-factorial-integration-checks-20261110-v2.txt',
              Q/'minicpm5-layer1-current-history-factorial-analysis-checks-20261110-v2.txt']
    for path in checks:
        text = path.read_text(); require('\nOK\n' in text and '\nFAILED' not in text, 'Actual targeted checks passed')
    entries = (Path(runner.__file__), Path(worker.__file__), Path(__file__).resolve(),
        Q/'check_minicpm5_layer1_current_history_factorial_v1.py', Q/'check_minicpm5_layer1_current_history_factorial_integration_v1.py',
        Q/'minicpm5_layer1_current_history_factorial_analysis_v1.py',
        Q/'check_minicpm5_layer1_current_history_factorial_analysis_v1.py',
        Q/'audit_minicpm5_layer1_current_history_factorial_result_v1.py',
        Q/'record_minicpm5_layer1_current_history_factorial_exit_v1.py')
    graph = local_imports(entries=entries)
    for path, sha in graph['source_sha256'].items():
        require(path not in plan['source_sha256'] or plan['source_sha256'][path] == sha, 'Retain every previously executed source')
        plan['source_sha256'][path] = sha
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and str(path).startswith(str(R)+'/'):
            path = str(Path(path).resolve()); sha = previous.ref(path)['sha256']
            require(path not in plan['source_sha256'] or plan['source_sha256'][path] == sha, 'Actual loaded sources unchanged')
            plan['source_sha256'][path] = sha
    for path, value in reader.inputs.items():
        require(path not in plan['inputs'] or plan['inputs'][path] == value, 'Actual predecessor metadata unchanged')
        plan['inputs'][path] = value
    output = '/mnt/nvme/minicpm5-layer1-current-history-factorial-r1-20261110'
    require(not Path(output).exists(), 'Fresh original-model intervention output')
    plan.update(output_root=output, layer1_current_history_factorial=binding,completed_storage_maintenance=storage_binding)
    spec.update(schema='minicpm5_layer1_current_history_factorial_cpu_execution_spec_v1', runner=runner.SELF,
        worker=str(Path(worker.__file__).resolve()), output_root=output, source_sha256=plan['source_sha256'],
        unit='trn1-mini-layer1-current-history-factorial-r1-20261110', master_port=29629, completed_official_history=binding,completed_storage=storage_binding)
    base = Q/'minicpm5-layer1-current-history-factorial-r1-20261110'
    planpath = Path(str(base)+'-plan.json'); specpath = Path(str(base)+'-spec.json'); reviewpath = Path(str(base)+'-review.json')
    runner.p.once(planpath, plan); spec['plan'] = previous.ref(planpath); runner.p.once(specpath, spec)
    review = dict(schema='minicpm5_layer1_current_history_factorial_runtime_source_review_v1', reviewer='/root',
        passed=True, blockers=[], spec=previous.ref(specpath), source_sha256=spec['source_sha256'], runtime_admitted=False,
        changes='Original full4-row attention with own/current/history/both/restored; current normalized input changes Q/K/V jointly; previous-frame valid K/V only',
        attention_invocation_ast_identical=True, original_precision_scientific_inputs_and_gates_preserved=True,
        completed_official_history=binding,completed_storage=storage_binding,additional_model_audit_storage_drains_required=True,
        tests=[previous.ref(path) for path in checks], limits=runner.PROPS,
        expected_decode_calls=180, expected_original_controls=36, expected_restored_controls=36,
        expected_official_interventions=108, expected_raw_output_bytes=2949120,
        previous_real_controls_full_bytes_required=True,expected_previous_control_calls=108)
    runner.p.once(reviewpath, review); runner.read_bundle(previous.ref(specpath), previous.ref(reviewpath))
    print(json.dumps(dict(spec=previous.ref(specpath), review=previous.ref(reviewpath),
        source_files=len(plan['source_sha256']), runtime_admitted=False)))


if __name__ == '__main__': build()
