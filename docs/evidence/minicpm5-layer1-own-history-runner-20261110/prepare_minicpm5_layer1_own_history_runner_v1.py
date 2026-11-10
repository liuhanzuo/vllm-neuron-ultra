"""Source preparation; actual candidate requires both completed data exits."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
import minicpm5_layer1_own_history_cpu_runner_v1 as runner
import minicpm5_layer1_own_history_cpu_worker_v1 as worker
from research import minicpm5_layer1_after_saved_audit_v1 as history
from research.prepare_minicpm5_layer1_observation_source_v1 import Reader as OriginalReader
from prepare_dory_residual_spec_v3 import local_imports
require=history.require
SOURCE_OUTPUT=Path('/mnt/nvme/minicpm5-layer1-own-history-source-20261110')
ENTRIES=(Path(runner.__file__),Path(worker.__file__),Path(history.__file__),
 Q/'minicpm5_layer1_own_history_results_v1.py',Q/'minicpm5_layer1_official_shards_v2.py',
 Q/'minicpm5_layer1_own_history_cpu_driver_v2.py',Path(__file__).resolve(),
 Q/'check_minicpm5_layer1_after_saved_audit_v1.py',Q/'check_minicpm5_layer1_own_history_runner_v1.py',
 Q/'check_minicpm5_layer1_own_history_results_v1.py',Q/'check_minicpm5_layer1_official_shards_v2.py')


def ref(path):
    path=Path(path);return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class Reader:
    def __init__(self):self.original=OriginalReader();self.inputs={}
    def __call__(self,reference):
        value=self.original.read(reference['path'],reference['sha256'])
        self.inputs[reference['path']]=dict(sha256=reference['sha256'],identity=runner.identity(reference['path']))
        return value


def source_candidate():
    reader=Reader();original=reader(runner.RESULTS_SOURCE_PLAN);plan=copy.deepcopy(original)
    plan['output_root']=str(SOURCE_OUTPUT)
    require(not SOURCE_OUTPUT.exists(),'Uncreated source-only output placeholder')
    graph=local_imports(entries=ENTRIES)
    for path,digest in graph['source_sha256'].items():
        require(path not in plan['source_sha256']or plan['source_sha256'][path]==digest,'Preserve every frozen source')
        plan['source_sha256'][path]=digest
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(R)+'/'):
            path=str(Path(path).resolve());digest=ref(path)['sha256']
            require(path not in plan['source_sha256']or plan['source_sha256'][path]==digest,'Actual loaded source closure')
            plan['source_sha256'][path]=digest
    history.audit.source_closure(plan)
    require({k:v for k,v in plan.items()if k not in ('source_sha256','output_root')}==
        {k:v for k,v in original.items()if k not in ('source_sha256','output_root')},
        'Original complete precision, scientific inputs and acceptance gates')
    checks=dict(schema='minicpm5_layer1_own_history_runner_source_preparation_v1',source_only=True,
        original_source_plan=runner.RESULTS_SOURCE_PLAN,entries=[ref(p)for p in ENTRIES],
        source_files=len(plan['source_sha256']),source_bytes=sum(Path(p).stat().st_size for p in plan['source_sha256']),
        all_original_scientific_fields_preserved=True,all_original_source_pins_preserved=True,
        source_placeholder=str(SOURCE_OUTPUT),runtime_spec_created=False,runtime_admitted=False,
        actual_collector_and_saved_audit_exits_required=True,physical_drain_still_required=True,
        original_full_acceptance_gates=plan['remaining_gates'],real_tensor_payloads_read=False,
        native_imported=False,full_model_qualified=False,new_speedup_claim=False)
    return plan,checks


def assemble_runtime_candidate(binding,attempt):
    require(type(attempt)is str and re.fullmatch('[a-z0-9-]{1,40}',attempt),'Finite fresh own-history attempt')
    reader=Reader();bundle=history.completed(binding,reader)
    prepared,checks=source_candidate();plan=copy.deepcopy(bundle['collector_plan'])
    for table in (prepared['source_sha256'],bundle['saved_audit_spec']['source_sha256']):
        for path,digest in table.items():
            require(path not in plan['source_sha256']or plan['source_sha256'][path]==digest,'All previous scientific sources retained')
            plan['source_sha256'][path]=digest
    # The complete collected metadata graph is finite and contains no raw payload
    # reads. Payload bytes stay in their original owned tree for guarded rereads.
    for rank,result in enumerate(bundle['collector_chain']['rank_results']):
        require(result['rank']==rank,'Both actual completed rank metadata')
        for arm,entry in zip(history.stream.raw.ARMS,result['arm_reports']):
            report=reader(dict(path=entry['report'],sha256=entry['sha256']))
            require(report['rank']==rank and report['arm']==arm and report['complete']is True
                and len(report['reports'])==14,'Complete actual collected arm')
            for ci,item in enumerate(report['reports']):
                value=reader(dict(path=item['report'],sha256=item['sha256']))
                require(value['complete']is True and [value['rank'],value['arm'],value['call']['index']]==[rank,arm,ci],
                    'Every complete original collected call JSON')
    for path,record in reader.inputs.items():
        require(path not in plan['inputs']or plan['inputs'][path]==record,'Every actual metadata identity retained')
        plan['inputs'][path]=record
    output=Path('/mnt/nvme')/('minicpm5-layer1-own-history-'+attempt)
    require(not output.exists(),'Fresh actual own-history output')
    plan.update(output_root=str(output),layer1_own_history_replay=copy.deepcopy(binding))
    spec=copy.deepcopy(bundle['collector_spec'])
    spec.update(schema='minicpm5_layer1_own_history_cpu_execution_spec_v1',runner=runner.SELF,
        worker=str(Path(worker.__file__).resolve()),output_root=str(output),source_sha256=plan['source_sha256'],
        unit='trn1-mini-layer1-own-history-'+attempt,master_port=29619,
        prepared_source_plan=runner.RESULTS_SOURCE_PLAN,completed_layer1_data=copy.deepcopy(binding),plan=None)
    checks.update(actual_collector_and_saved_audit_metadata_authenticated=True,
        all84_collected_call_metadata_bound=True,actual_metadata_files=len(reader.inputs),
        concrete_runtime_plan_write_and_review_required=True,actual_physical_drain_still_required=True)
    return spec,plan,checks


if __name__=='__main__':
    require(sys.dont_write_bytecode and __debug__,'Original Python -B with assertions')
    plan,checks=source_candidate()
    target=Q/'minicpm5-layer1-own-history-runner-source-plan-20261110-v1.json'
    proof=Q/'minicpm5-layer1-own-history-runner-source-checks-20261110-v1.json'
    require(not target.exists()and not proof.exists(),'Fresh immutable source preparation')
    with target.open('x')as f:json.dump(plan,f,indent=2);f.write(chr(10))
    checks['plan']=ref(target)
    with proof.open('x')as f:json.dump(checks,f,indent=2);f.write(chr(10))
    print(json.dumps(dict(plan=ref(target),checks=ref(proof),source_files=checks['source_files'],source_bytes=checks['source_bytes'],runtime_admitted=False)))
