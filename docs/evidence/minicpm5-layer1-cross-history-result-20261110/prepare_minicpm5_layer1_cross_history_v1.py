"""Build a concrete new cross-history plan preserving all executed source pins."""
import copy
import hashlib
import json
from pathlib import Path
import sys
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
from research import prepare_minicpm5_layer1_own_history_runner_v3 as previous
from research import minicpm5_layer1_after_own_history_v1 as own
import minicpm5_layer1_cross_history_cpu_runner_v1 as runner
import minicpm5_layer1_cross_history_cpu_worker_v1 as worker
from prepare_dory_residual_spec_v3 import local_imports
require=runner.p.require


def build():
    hand=json.loads((Q/'continuation-handoff-20261110-v410-mini-own-history-reproduced.json').read_text())
    h=hand['own_history'];binding=dict(schema='minicpm5_layer1_completed_own_history_v1',
        model_exit=h['actual_exit'],audit_exit=h['independent_audit_actual_exit'])
    reader=previous.Reader();completed=own.completed(binding,reader)
    spec,plan,checks=previous.assemble_runtime_candidate(completed['plan']['layer1_own_history_replay'],'cross-preparation-20261110')
    # Preserve every actual own-history source and input, then add current sources.
    for table in ('source_sha256','inputs'):
        for path,value in completed['plan'][table].items():
            require(path not in plan[table]or plan[table][path]==value,'Preserved actual predecessor '+table)
            plan[table][path]=value
    entries=[Path(runner.__file__),Path(worker.__file__),Path(__file__).resolve(),
        Q/'check_minicpm5_layer1_cross_history_v1.py',Q/'audit_minicpm5_layer1_own_history_result_v1.py']
    graph=local_imports(entries=tuple(entries))
    for path,sha in graph['source_sha256'].items():
        require(path not in plan['source_sha256']or plan['source_sha256'][path]==sha,'Never overwrite executed source pins')
        plan['source_sha256'][path]=sha
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(R)+'/'):
            path=str(Path(path).resolve());sha=previous.ref(path)['sha256']
            require(path not in plan['source_sha256']or plan['source_sha256'][path]==sha,'Loaded source closure preserved')
            plan['source_sha256'][path]=sha
    for path,value in reader.inputs.items():
        require(path not in plan['inputs']or plan['inputs'][path]==value,'Preserved predecessor metadata identity')
        plan['inputs'][path]=value
    output='/mnt/nvme/minicpm5-layer1-cross-history-r1-20261110'
    require(not Path(output).exists(),'Fresh cross-history output')
    plan.update(output_root=output,layer1_cross_history=binding)
    spec.update(schema='minicpm5_layer1_cross_history_cpu_execution_spec_v1',runner=runner.SELF,
        worker=str(Path(worker.__file__).resolve()),output_root=output,source_sha256=plan['source_sha256'],
        unit='trn1-mini-layer1-cross-history-r1-20261110',master_port=29621,completed_own_history=binding)
    base=Q/'minicpm5-layer1-cross-history-r1-20261110'
    planpath=Path(str(base)+'-plan.json');specpath=Path(str(base)+'-spec.json');reviewpath=Path(str(base)+'-review.json')
    runner.p.once(planpath,plan);spec['plan']=previous.ref(planpath);runner.p.once(specpath,spec)
    review=dict(schema='minicpm5_layer1_cross_history_runtime_source_review_v1',reviewer='/root',passed=True,blockers=[],
        spec=previous.ref(specpath),source_sha256=spec['source_sha256'],runtime_admitted=False,
        changes='Paired K/V history swap with fixed query input; diagonal before off-diagonal, restored control after; full4row outputs',
        original_attention_invocation_ast_identical=True,original_official_weights_precision_and_scientific_gates_preserved=True,
        completed_own_history=binding,previous1255_services_plus_actual_replay_and_independent_auditor_drains_required=True,
        tests=previous.ref(Q/'minicpm5-layer1-cross-history-checks-20261110-v2.txt'),
        limits=runner.PROPS,expected_matrix_cells=108,expected_restore_controls=36,expected_raw_output_bytes=2359296)
    runner.p.once(reviewpath,review)
    runner.read_bundle(previous.ref(specpath),previous.ref(reviewpath))
    print(json.dumps(dict(spec=previous.ref(specpath),review=previous.ref(reviewpath),source_files=len(plan['source_sha256']),
        original_inputs=len(plan['inputs']),runtime_admitted=False)))

if __name__=='__main__':build()
