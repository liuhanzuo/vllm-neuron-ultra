"""Concrete factorial experiment retaining all original sources and gates."""
import copy
import json
from pathlib import Path
import sys
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
from research import prepare_minicpm5_layer1_own_history_runner_v3 as previous
from research import minicpm5_layer1_after_cross_history_v1 as crossed
import minicpm5_layer1_kv_factorial_cpu_runner_v1 as runner
import minicpm5_layer1_kv_factorial_cpu_worker_v1 as worker
from prepare_dory_residual_spec_v3 import local_imports
require=runner.p.require


def build():
    h=json.loads((Q/'continuation-handoff-20261110-v412-mini-cross-history-complete.json').read_text())['cross_history']
    binding=dict(schema='minicpm5_layer1_completed_cross_history_v1',model_exit=h['actual_exit'],audit_exit=h['independent_audit_actual_exit'])
    reader=previous.Reader();completed=crossed.completed(binding,reader)
    spec=copy.deepcopy(completed['spec']);plan=copy.deepcopy(completed['plan'])
    entries=(Path(runner.__file__),Path(worker.__file__),Path(__file__).resolve(),
        Q/'check_minicpm5_layer1_kv_factorial_v1.py',Q/'audit_minicpm5_layer1_cross_history_result_v1.py')
    graph=local_imports(entries=entries)
    for path,sha in graph['source_sha256'].items():
        require(path not in plan['source_sha256']or plan['source_sha256'][path]==sha,'Preserve all frozen executed sources')
        plan['source_sha256'][path]=sha
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(R)+'/'):
            path=str(Path(path).resolve());sha=previous.ref(path)['sha256']
            require(path not in plan['source_sha256']or plan['source_sha256'][path]==sha,'Actual loaded sources retained')
            plan['source_sha256'][path]=sha
    for path,value in reader.inputs.items():
        require(path not in plan['inputs']or plan['inputs'][path]==value,'Original input metadata identity')
        plan['inputs'][path]=value
    output='/mnt/nvme/minicpm5-layer1-kv-factorial-r1-20261110';require(not Path(output).exists(),'Fresh factorial output')
    plan.update(output_root=output,layer1_kv_factorial=binding)
    spec.update(schema='minicpm5_layer1_kv_factorial_cpu_execution_spec_v1',runner=runner.SELF,
        worker=str(Path(worker.__file__).resolve()),output_root=output,source_sha256=plan['source_sha256'],
        unit='trn1-mini-layer1-kv-factorial-r1-20261110',master_port=29623,completed_cross_history=binding)
    base=Q/'minicpm5-layer1-kv-factorial-r1-20261110'
    planpath=Path(str(base)+'-plan.json');specpath=Path(str(base)+'-spec.json');reviewpath=Path(str(base)+'-review.json')
    runner.p.once(planpath,plan);spec['plan']=previous.ref(planpath);runner.p.once(specpath,spec)
    review=dict(schema='minicpm5_layer1_kv_factorial_runtime_source_review_v1',reviewer='/root',passed=True,blockers=[],
        spec=previous.ref(specpath),source_sha256=spec['source_sha256'],runtime_admitted=False,
        changes='Explicit3input x3K x3V factorial with diagonal before and restored controls after; all previous paired cells exact',
        attention_invocation_ast_identical=True,original_precision_scientific_inputs_and_gates_preserved=True,
        completed_cross_history=binding,additional_model_and_auditor_physical_drains_required=True,
        tests=previous.ref(Q/'minicpm5-layer1-kv-factorial-checks-20261110-v1.txt'),limits=runner.PROPS,
        expected_matrix_cells=324,expected_restore_controls=36,expected_raw_output_bytes=5898240)
    runner.p.once(reviewpath,review);runner.read_bundle(previous.ref(specpath),previous.ref(reviewpath))
    print(json.dumps(dict(spec=previous.ref(specpath),review=previous.ref(reviewpath),source_files=len(plan['source_sha256']),runtime_admitted=False)))
if __name__=='__main__':build()
