"""Build an official native32 collection preserving actual factorial sources."""
import copy
import json
from pathlib import Path
import sys
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
from research import prepare_minicpm5_layer1_own_history_runner_v3 as previous
from research import minicpm5_layer1_after_kv_factorial_v1 as factored
import minicpm5_layer1_native_cache_runner_v1 as runner
import minicpm5_layer1_native_cache_worker_v1 as worker
from prepare_dory_residual_spec_v3 import local_imports
require=runner.p.require


def build():
    h=json.loads((Q/'continuation-handoff-20261110-v414-mini-kv-factorial-complete.json').read_text())['factorial']
    binding=dict(schema='minicpm5_layer1_completed_kv_factorial_v1',model_exit=h['actual_exit'],audit_exit=h['independent_audit_actual_exit'])
    reader=previous.Reader();completed=factored.completed(binding,reader)
    spec=copy.deepcopy(completed['spec']);plan=copy.deepcopy(completed['plan'])
    entries=(Path(runner.__file__),Path(worker.__file__),Path(__file__).resolve(),Q/'check_minicpm5_layer1_native_cache_v1.py',
        Q/'audit_minicpm5_layer1_kv_factorial_result_v1.py')
    graph=local_imports(entries=entries)
    for path,sha in graph['source_sha256'].items():
        require(path not in plan['source_sha256']or plan['source_sha256'][path]==sha,'Every frozen executed source retained')
        plan['source_sha256'][path]=sha
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(R)+'/'):
            path=str(Path(path).resolve());sha=previous.ref(path)['sha256']
            require(path not in plan['source_sha256']or plan['source_sha256'][path]==sha,'Actual loaded sources pinned')
            plan['source_sha256'][path]=sha
    for path,value in reader.inputs.items():
        require(path not in plan['inputs']or plan['inputs'][path]==value,'Actual predecessor metadata unchanged')
        plan['inputs'][path]=value
    output='/mnt/nvme/minicpm5-layer1-native-cache-r1-20261110';require(not Path(output).exists(),'Fresh native cache output')
    plan.update(output_root=output,layer1_native_cache=binding)
    spec.update(schema='minicpm5_layer1_native_cache_execution_spec_v1',runner=runner.SELF,
        worker=str(Path(worker.__file__).resolve()),output_root=output,source_sha256=plan['source_sha256'],
        unit='trn1-mini-layer1-native-cache-r1-20261110',master_port=29625,completed_factorial=binding)
    require(runner.PYTHON==plan['interpreters']['native'],'Exact original native interpreter')
    base=Q/'minicpm5-layer1-native-cache-r1-20261110'
    planpath=Path(str(base)+'-plan.json');specpath=Path(str(base)+'-spec.json');reviewpath=Path(str(base)+'-review.json')
    runner.p.once(planpath,plan);spec['plan']=previous.ref(planpath);runner.p.once(specpath,spec)
    review=dict(schema='minicpm5_layer1_native_cache_runtime_source_review_v1',reviewer='/root',passed=True,blockers=[],
        spec=previous.ref(specpath),source_sha256=spec['source_sha256'],runtime_admitted=False,
        changes='Read-only post-forward layer1 K/V, full FP32 logits and all213selectedBF16taps; exact official32 full trajectory',
        original_loader='minicpm5_layer0_native_adapter_v2.load',original_observer='minicpm5_hf32_boundary_observer_v2.Hooks',
        precision_native_source_model_weights_inputs_and_all_scientific_gates_preserved=True,
        completed_factorial=binding,additional_model_and_auditor_drains_required=True,
        tests=previous.ref(Q/'minicpm5-layer1-native-cache-checks-20261110-v1.txt'),limits=runner.PROPS,
        expected_native_forwards=32,expected_kv_bytes=5894144,expected_logits_bytes=16711680,expected_selected_taps=6816)
    runner.p.once(reviewpath,review);runner.read_bundle(previous.ref(specpath),previous.ref(reviewpath))
    print(json.dumps(dict(spec=previous.ref(specpath),review=previous.ref(reviewpath),source_files=len(plan['source_sha256']),runtime_admitted=False)))
if __name__=='__main__':build()
