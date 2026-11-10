"""Build a metadata-only source candidate; never creates an execution spec."""
import copy
import hashlib
import json
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1];Q=R/'research'
sys.path[:0]=[str(R),str(Q)]
import minicpm5_original32_layer1_observation_cpu_worker_v1 as worker
from prepare_dory_residual_spec_v3 import local_imports

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Reader:
    def __init__(self):self.refs={};self.bytes=0;self.cache={}
    def read(self,path,digest):
        path=Path(path);key=str(path)
        require(path.suffix=='.json' and path.is_absolute() and path.resolve()==path
            and (path.is_relative_to(R) or path.is_relative_to('/mnt/nvme')),
            'Only original bounded JSON metadata')
        require(key not in self.refs or self.refs[key]==digest,'Metadata identity cannot be rebound')
        if key in self.cache:return copy.deepcopy(self.cache[key])
        before=worker.identity(path)
        require(0<path.stat().st_size<=8*2**20,'Original8MiB metadata cap')
        raw=path.read_bytes()
        require(len(raw)<=8*2**20 and hashlib.sha256(raw).hexdigest()==digest
            and worker.identity(path)==before,'Stable complete original metadata')
        self.bytes+=len(raw);require(self.bytes<=128*2**20,'Finite full metadata collection')
        self.refs[key]=digest;self.cache[key]=json.loads(raw)
        return copy.deepcopy(self.cache[key])

def assemble():
    reader=Reader()
    def load(ref):return reader.read(ref['path'],ref['sha256'])
    ended=load(worker.OBSERVED_EXIT);spec=load(ended['spec']);original=load(spec['plan'])
    plan=copy.deepcopy(original)
    plan['output_root']='/mnt/nvme/minicpm5-original32-layer0-full-bf16-cpu-layer1-source-candidate-20261110'
    require(not Path(plan['output_root']).exists(),'No existing runtime output namespace')
    calls=[]
    for rank in(0,1):
        row=load(ended['outputs']['rank'+str(rank)+'_result'])
        for arm,item in zip(worker.ARMS,row['arm_reports']):
            require(item['arm']==arm,'Original complete arm ordering')
            report=load(dict(path=item['report'],sha256=item['sha256']))
            for call in report['reports']:
                ref=dict(path=call['report'],sha256=call['sha256']);value=load(ref)
                require(value['rank']==rank and value['arm']==arm and value['call']['index']==call['call'],
                    'Actual original call reference')
                calls.append(dict(rank=rank,arm=arm,call=call['call'],**ref))
    require(len(calls)==84,'Complete original three arms and both ranks')
    plan['layer1_observation']=dict(schema='minicpm5_layer1_observation_plan_v1',
        original_exit=worker.OBSERVED_EXIT,original_plan=spec['plan'],original_calls=calls,
        selected_taps=list(worker.layer1.TAPS),cache_names=list(worker.layer1.CACHES),
        cache_shape=list(worker.layer1.codec.SHAPE),
        output_caps=dict(per_rank=128*2**20,total=256*2**20,single_file=16*2**20),
        collection_mode='after_forward_exact_frozen_repeat_then_readonly_export')
    for path,digest in reader.refs.items():
        entry=dict(sha256=digest,identity=worker.identity(path))
        if path in plan['inputs']:
            require(plan['inputs'][path]==entry,'Original admitted metadata unchanged')
        else:plan['inputs'][path]=entry
    entries=[Path(worker.__file__),Q/'minicpm5_layer1_observation_v1.py',
        Q/'minicpm5_layer1_cache_codec_v1.py',Path(__file__).resolve(),
        Q/'check_minicpm5_layer1_observation_v1.py']
    graph=local_imports(entries=tuple(entries))
    for path,digest in graph['source_sha256'].items():
        require(path not in plan['source_sha256'] or plan['source_sha256'][path]==digest,
            'Frozen source cannot be changed: '+path)
        plan['source_sha256'][path]=digest
    def admitted(path,digest):
        require(str(path)in plan['inputs'] and plan['inputs'][str(path)]['sha256']==digest
            and plan['inputs'][str(path)]['identity']==worker.identity(path),
            'Every metadata read admitted')
        return reader.read(path,digest)
    worker.validate_plan(plan,admitted)
    return plan,dict(metadata_files=len(reader.refs),metadata_bytes_read=reader.bytes,
        source_files=len(plan['source_sha256']),original_calls=84,frames_per_arm=32,
        ranks=2,arms=3,raw_cache_bytes_without_compression=84*2*worker.layer1.codec.CACHE_BYTES,
        original_output_cap_retained=256*2**20,model_or_tensor_payloads_read=False,
        native_started=False,runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False,
        remaining=['Actual runner and complete current predecessor admission',
            'Native14-call/32-frame repeat against every frozen original call',
            'Independent full payload reconstruction and whole-cache byte comparison',
            'Own-history attention replay must reproduce captured decode outputs before paired-history causal comparison'])

def write():
    plan,checks=assemble()
    dest=Q/'minicpm5-layer1-observation-source-plan-20261110-v1.json'
    proof=Q/'minicpm5-layer1-observation-source-checks-20261110-v1.json'
    require(not dest.exists() and not proof.exists(),'Fresh immutable source candidate')
    with dest.open('x')as stream:json.dump(plan,stream,indent=2);stream.write('\n')
    checks.update(source_only=True,plan=dict(path=str(dest),sha256=sha(dest)))
    with proof.open('x')as stream:json.dump(checks,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(plan=checks['plan'],checks=dict(path=str(proof),sha256=sha(proof)),
        metadata_files=checks['metadata_files'],metadata_bytes=checks['metadata_bytes_read'],
        runtime_admitted=False)))
if __name__=='__main__':
    require(sys.dont_write_bytecode and __debug__,'-B and original assertions')
    write()
