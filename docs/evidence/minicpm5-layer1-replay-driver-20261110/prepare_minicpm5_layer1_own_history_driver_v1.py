"""Source closure preparation only; no actual replay spec, admission or launch."""
import copy
import hashlib
import json
from pathlib import Path
import sys
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_own_history_cpu_driver_v1 as driver
from research import minicpm5_layer1_saved_audit_runner_v1 as audit_runner
from research.prepare_minicpm5_layer1_observation_source_v1 import Reader
from prepare_dory_residual_spec_v3 import local_imports
require=driver.require
PRIOR=dict(path=str(Q/'minicpm5-layer1-saved-audit-source-plan-20261110-v1.json'),
 sha256='2af99c21df8f01c60aa82591cb761105ba4448b1fb864fcab826134b7728f177')
BOUNDARY=dict(path=str(Q/'minicpm5-layer1-decode-boundary-source-plan-20261110-v1.json'),
 sha256='03d018e787ce6242b6e727e0a791d9d0852f5d554d87cf6832354e098d2f1a50')
NAMES=('minicpm5_layer1_decode_stream_v1.py','check_minicpm5_layer1_decode_stream_v1.py',
 'minicpm5_layer1_own_history_cpu_driver_v1.py','check_minicpm5_layer1_own_history_cpu_driver_v1.py',
 'minicpm5_layer1_decode_boundary_v1.py','minicpm5_layer1_own_history_callable_v1.py',
 'minicpm5_layer1_official_shards_v1.py','prepare_minicpm5_layer1_own_history_driver_v1.py')


def ref(path):
    path=Path(path);return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def prepare():
    reader=Reader();original=reader.read(PRIOR['path'],PRIOR['sha256']);plan=copy.deepcopy(original)
    boundaries=reader.read(BOUNDARY['path'],BOUNDARY['sha256'])
    require(boundaries['total_rank_decode_calls']==36 and boundaries['physical_prediction_rows']==144
        and boundaries['own_history_replay_not_yet_executed']is True,
        'Original36 complete decode boundaries remain unexecuted')
    for entry in boundaries['source_refs']:
        require(ref(entry['path'])==entry,'Frozen boundary source retained')
    for row in boundaries['sdk_functions']:
        require(ref(row['source']['path'])==row['source'],'All pinned original SDK/capture source functions retained')
    graph=local_imports(entries=tuple(Q/name for name in NAMES))
    for path,digest in graph['source_sha256'].items():
        require(path not in plan['source_sha256']or plan['source_sha256'][path]==digest,'Preserve every original source pin')
        plan['source_sha256'][path]=digest
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(R)+'/'):
            path=str(Path(path).resolve());digest=ref(path)['sha256']
            require(path not in plan['source_sha256']or plan['source_sha256'][path]==digest,'Loaded source pin unchanged')
            plan['source_sha256'][path]=digest
    audit_runner.source_closure(plan)
    require({k:v for k,v in plan.items()if k!='source_sha256'}==
        {k:v for k,v in original.items()if k!='source_sha256'},'Original scientific inputs and complete acceptance gates unchanged')
    require(not Path(plan['output_root']).exists(),'Source placeholder is not an actual runtime collection or replay')
    require(not {'torch','numpy','torch_neuronx','neuronxcc','transformers','vllm'}&set(sys.modules),
        'No native model, SDK or compiler imports')
    checks=dict(schema='minicpm5_layer1_own_history_driver_source_preparation_v1',
        source_only=True,inherited_source_plan=PRIOR,decode_boundaries=BOUNDARY,
        entries=[ref(Q/name)for name in NAMES],source_files=len(plan['source_sha256']),
        source_bytes=sum(Path(path).stat().st_size for path in plan['source_sha256']),
        all_original_scientific_inputs_preserved=True,all_original_source_pins_preserved=True,
        all84history_calls_required=True,all36complete_decode_outputs_required=True,
        official_layer1_full_byte_comparison_required=True,
        outputs_per_rank=dict(calls=18,rows=72,raw_bytes=294912),
        real_tensor_payloads_read=False,native_imported=False,runtime_spec_created=False,runtime_admitted=False,
        missing_runtime_work=[
            'Actual r11 exit/release, then complete84call collection with observed external exit and cleanup.',
            'Independent full-vocabulary/official32reference/full168cache audit with actual external exit and physical drain.',
            'Separate concrete replay execution plan and review bind both completed actual lineages, exact source and input closure.',
            'Outer serial runner and TP2 rank entrypoint must enforce physical predecessors drained, original CPU SDK import transition, source guards, full resource/output/storage limits and cleanup before unlock.',
            'Actually execute official8shard byte checks and all36 original attention calls; source/synthetic tests do not prove native success.',
            'Only after all four rows of all36 own-history outputs are exact can paired-history causal interventions be admitted.',
            'Original precision and cold-cache4096/long-decode/slot-reuse/generation/RNG/stop/text/quality/realHTTP-C4/paired-performance gates remain.'
        ],full_model_qualified=False,new_speedup_claim=False)
    return plan,checks


if __name__=='__main__':
    require(sys.dont_write_bytecode and __debug__,'Python -B and assertions')
    plan,checks=prepare();target=Q/'minicpm5-layer1-own-history-driver-source-plan-20261110-v1.json'
    proof=Q/'minicpm5-layer1-own-history-driver-source-checks-20261110-v1.json'
    require(not target.exists()and not proof.exists(),'Fresh immutable source preparation')
    with target.open('x')as f:json.dump(plan,f,indent=2);f.write(chr(10))
    checks['plan']=ref(target)
    with proof.open('x')as f:json.dump(checks,f,indent=2);f.write(chr(10))
    print(json.dumps(dict(plan=ref(target),checks=ref(proof),source_files=checks['source_files'],
        source_bytes=checks['source_bytes'],runtime_admitted=False)))
