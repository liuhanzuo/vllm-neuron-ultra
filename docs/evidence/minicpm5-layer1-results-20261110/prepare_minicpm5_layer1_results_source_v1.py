"""Prepare an immutable source successor only; no runtime spec or native reads."""
import hashlib
import json
from pathlib import Path
import sys
Q=Path(__file__).resolve().parent;R=Q.parent
sys.path[:0]=[str(R),str(Q)]
from research.prepare_minicpm5_layer1_observation_source_v1 import assemble,Reader,require
from prepare_dory_residual_spec_v3 import local_imports
import minicpm5_original32_layer1_observation_cpu_worker_v1 as worker

def ref(path):
    path=Path(path)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def prepare():
    plan,checks=assemble()
    prior=Q/'minicpm5-layer1-observation-source-plan-20261110-v1.json'
    prior_plan=json.loads(prior.read_text())
    require(plan==prior_plan,'Original source plan reproduced exactly')
    entries=[Q/name for name in ('minicpm5_layer1_runtime_results_v1.py',
        'minicpm5_layer1_raw_verifier_v1.py','minicpm5_layer1_payload_io_v1.py',
        'check_minicpm5_layer1_results_v1.py','prepare_minicpm5_layer1_results_source_v1.py')]
    graph=local_imports(entries=tuple(entries))
    for path,digest in graph['source_sha256'].items():
        require(path not in plan['source_sha256'] or plan['source_sha256'][path]==digest,
                'No pinned predecessor source overwritten')
        plan['source_sha256'][path]=digest
    reader=Reader()
    def read(path,digest):
        entry=plan['inputs'][path]
        require(entry['sha256']==digest and entry['identity']==worker.identity(path),'Original admitted metadata')
        return reader.read(path,digest)
    worker.validate_plan(plan,read)
    require(not Path(plan['output_root']).exists(),'Source plan does not authorize any output job')
    require(not {'torch','numpy','neuronxcc','torch_neuronx'}&set(sys.modules),'No native runtime imported')
    dest=Q/'minicpm5-layer1-results-source-plan-20261110-v1.json'
    with dest.open('x')as stream:json.dump(plan,stream,indent=2);stream.write('\n')
    checks.update(schema='minicpm5_layer1_results_source_preparation_v1',previous_source_plan=ref(prior),
        plan=ref(dest),source_files=len(plan['source_sha256']),new_entrypoints=[ref(path)for path in entries],
        original_scientific_plan_unchanged=True,original_source_pins_retained=True,
        result_metadata_validation_ready=True,independent_byte_verifier_ready=True,
        guarded_payload_reader_ready=True,raw_original_tensor_bytes_read=0,
        runtime_admitted=False,runtime_spec_created=False,full_model_qualified=False,new_speedup_claim=False,
        remaining=['Source-reviewed collection runner with drain-before-release cleanup and current serial history',
         'Actual native complete84-call repeat/export under original output caps',
         'Separately admitted real payload audit plus original complete-logits/HF checks',
         'Same-current-input own-history attention replay before paired-history causal comparison'])
    proof=Q/'minicpm5-layer1-results-source-checks-20261110-v1.json'
    with proof.open('x')as stream:json.dump(checks,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(plan=ref(dest),checks=ref(proof),source_files=checks['source_files'],
        original_scientific_plan_unchanged=True,runtime_admitted=False)))
if __name__=='__main__':
    require(sys.dont_write_bytecode and __debug__,'Python -B without optimized assertions')
    prepare()
