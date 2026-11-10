"""Prepare the combined audit source; actual runtime needs external collection exit."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_saved_audit_runner_v2 as runner
from research.prepare_minicpm5_layer1_observation_source_v1 import Reader
from prepare_dory_residual_spec_v3 import local_imports
require=runner.require
PRIOR=dict(path=str(Q/'minicpm5-layer1-runner-source-plan-20261110-v1.json'),
 sha256='122cf50cac06022356a0681712ff762f343b80c74ac53520964e02b9451537de')
NAMES=('minicpm5_layer1_logits_archive_v1.py','minicpm5_layer1_numeric_audit_v1.py',
 'check_minicpm5_layer1_numeric_audit_v1.py','minicpm5_layer1_saved_audit_runner_v2.py',
 'check_minicpm5_layer1_saved_audit_runner_v2.py','prepare_minicpm5_layer1_saved_audit_v2.py')


def ref(path):
    path=Path(path);return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def prepare():
    reader=Reader();original=reader.read(PRIOR['path'],PRIOR['sha256']);plan=copy.deepcopy(original)
    graph=local_imports(entries=tuple(Q/name for name in NAMES))
    for path,digest in graph['source_sha256'].items():
        require(path not in plan['source_sha256']or plan['source_sha256'][path]==digest,'Frozen source retained')
        plan['source_sha256'][path]=digest
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(R)+'/'):
            path=str(Path(path).resolve());digest=ref(path)['sha256']
            require(path not in plan['source_sha256']or plan['source_sha256'][path]==digest,'Loaded source pin retained')
            plan['source_sha256'][path]=digest
    runner.source_closure(plan)
    require(not Path(plan['output_root']).exists(),'Source-only predecessor output placeholder')
    require({k:v for k,v in plan.items()if k!='source_sha256'}==
        {k:v for k,v in original.items()if k!='source_sha256'},'Original scientific plan unchanged')
    checks=dict(schema='minicpm5_layer1_saved_audit_source_preparation_v1',source_only=True,
        inherited_source_plan=PRIOR,new_sources=[ref(Q/name)for name in NAMES],
        source_files=len(plan['source_sha256']),source_bytes=sum(Path(p).stat().st_size for p in plan['source_sha256']),
        original_scientific_plan_unchanged=True,original_source_pins_retained=True,
        native_imports=False,real_tensor_payloads_read=False,runtime_spec_created=False,runtime_admitted=False,
        full_model_qualified=False,new_speedup_claim=False)
    return plan,checks


def assemble_runtime_candidate(collector_exit_ref,attempt):
    require(type(attempt)is str and re.fullmatch('[a-z0-9-]{1,40}',attempt),'Fresh bounded audit attempt')
    reader=Reader();read=lambda reference:reader.read(reference['path'],reference['sha256'])
    ended=read(collector_exit_ref);live=read(ended['live']);cs=read(ended['spec']);cr=read(ended['review']);cp=read(cs['plan'])
    refs=ended['outputs'];records={key:read(value)for key,value in refs.items()}
    runner.validate_chain(ended,live,cs,cr,cp,records,refs)
    plan,checks=prepare();sources=dict(plan['source_sha256'])
    for path,digest in cs['source_sha256'].items():
        require(path not in sources or sources[path]==digest,'Actual collector source retained')
        sources[path]=digest
    target=Q/('minicpm5-layer1-saved-audit-'+attempt+'.json')
    require(not target.exists(),'Actual fresh saved audit output')
    candidate=dict(schema='minicpm5_layer1_saved_audit_execution_spec_v1',runner=runner.SELF,
        collector_exit=collector_exit_ref,source_sha256=sources,canonical_sha256=cs['canonical_sha256'],
        limits=runner.LIMITS,output=str(target))
    checks.update(actual_external_collector_exit_metadata_authenticated=True,
        actual_physical_drain_still_required=True,concrete_runtime_review_still_required=True)
    return candidate,checks


if __name__=='__main__':
    require(sys.dont_write_bytecode and __debug__,'Python -B with assertions')
    plan,checks=prepare()
    target=Q/'minicpm5-layer1-saved-audit-source-plan-20261110-v2.json'
    proof=Q/'minicpm5-layer1-saved-audit-source-checks-20261110-v2.json'
    require(not target.exists()and not proof.exists(),'Fresh immutable source preparation')
    with target.open('x')as f:json.dump(plan,f,indent=2);f.write(chr(10))
    checks['plan']=ref(target)
    with proof.open('x')as f:json.dump(checks,f,indent=2);f.write(chr(10))
    print(json.dumps(dict(plan=ref(target),checks=ref(proof),source_files=checks['source_files'],
        source_bytes=checks['source_bytes'],runtime_admitted=False)))
