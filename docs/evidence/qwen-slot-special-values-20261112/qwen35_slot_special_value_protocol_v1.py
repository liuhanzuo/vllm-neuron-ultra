"""Fixed six-call diagnostic data and whole-file validation; inert on import."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import struct

from research import qwen35_slot_reference_canonicalization_evidence_v1 as original
from qwen35_sampler_runtime_contract_v12 import require
from qwen35_sampler_runtime_io_v12 import identity, raw

ROOT=Path(__file__).resolve().parents[1]
Q=ROOT/'research'
NAME='conv-special-values'
ARMS=('reference','candidate','identity')
PROFILES=('original_raw','negative_nan_raw')
SHAPE=[4,3072,4]
BYTES=98304
ARTIFACT_CAP=128*2**20
PAYLOAD_CAP=3342400
ARTIFACTS={
    'reference':dict(path=str(original.RUN/'conv/artifacts/standalone-reference/model.pt'),bytes=51005,
        sha256='38595fac55af2e83b7134d8df3bc1057490c7e87fa510916d6351343aeb59b92'),
    'candidate':dict(path=str(original.RUN/'conv/artifacts/standalone-candidate/model.pt'),bytes=173683,
        sha256='938e0aaef59a60ed2cbeca60c5a0d07be985670f95952de25642708e2a712f84')}
INPUTS=(dict(path=str(Q/'qwen35-slot-reference-canonicalization-saved-evidence-20261112-v1.json'),
             sha256='423b3b3bdb0170199eee2d667a368ee6551039e11128c494a6983772d3160a78'),
        dict(path=str(Q/'qwen35-predicated-slot-r1-root-actual-exit-20261112-v2.json'),
             sha256='6e07d96c6da69cfeb141d020e251cac8376ae007113a240d6d204db387318769'),
        dict(path=str(original.RUN/'conv/report.json'),
             sha256='74645c6ee057d423654bbb88a22cbc6c097c612aa0bd4149a2e1901f85967351'))

def sha(payload):return hashlib.sha256(payload).hexdigest()
def file_ref(path,cap=ARTIFACT_CAP):
    payload=raw(path,cap)
    return dict(path=str(path),sha256=sha(payload),bytes=len(payload))
def checked(reference,cap=ARTIFACT_CAP):
    require(set(reference)=={'path','sha256','bytes'},'Exact complete binary reference')
    payload=raw(reference['path'],cap)
    require(len(payload)==reference['bytes'] and sha(payload)==reference['sha256'],'Stable full artifact bytes and SHA256')
    return payload
def inputs(profile):
    require(profile in PROFILES,'Only two fixed profiles')
    negative=profile=='negative_nan_raw'
    return dict(cache=original.input_bytes(0,negative_nan=negative),
        updates=original.input_bytes(17,negative_nan=negative),slots=struct.pack('<4q',-1,-1,-1,-1))
def workload():
    return dict(schema='qwen35_slot_special_value_workload_v1',name=NAME,shape=SHAPE,dtype='bfloat16',
        arms=list(ARMS),profiles=list(PROFILES),calls=6,existing_artifacts=ARTIFACTS,
        original_failure_inputs=list(INPUTS),raw_payload_cap=PAYLOAD_CAP,max_live_graphs=1,
        original_eight_layout_acceptance_unchanged=True,qualification_matrix_completed=False,
        full_model_qualified=False,model_speedup_claim=False)
def verify_inputs():
    for reference in INPUTS:
        require(sha(raw(reference['path'],16*2**20))==reference['sha256'],'Frozen original failure evidence')
    for reference in ARTIFACTS.values():checked(reference)

def write_raw(path,payload,cap):
    """Exclusive complete fsynced raw file; only caller-owned fresh output paths."""
    path=Path(path)
    require(type(payload) is bytes and len(payload)<=cap and path.is_absolute()
            and path.parent.resolve()==path.parent and not path.exists(),'Fresh bounded diagnostic file')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    with os.fdopen(fd,'wb') as handle:
        require(handle.write(payload)==len(payload),'Complete raw write')
        handle.flush();os.fsync(handle.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:os.fsync(fd)
    finally:os.close(fd)
    result=file_ref(path,cap)
    require(result['sha256']==sha(payload) and result['bytes']==len(payload),'Complete persisted raw readback')
    return result

def differences(actual,expected):
    require(type(actual) is bytes and type(expected) is bytes and len(actual)==len(expected)==BYTES,
            'Whole exact-shaped BF16 output before byte comparison')
    positions=[i for i,(a,b) in enumerate(zip(actual,expected)) if a!=b]
    return positions,struct.pack('<%dI'%len(positions),*positions)

def identity_hlo_proof(payload):
    """Inspect the whole saved HLO's fixed NKI markers/config, without importing protobuf."""
    require(b'custom-call' in payload and b'AwsNeuronCustomNativeKernel' in payload,'Actual NKI custom-call HLO markers')
    configs=[]
    for token in re.findall(rb'[A-Za-z0-9+/=]{200,}',payload):
        try:value=json.loads(base64.b64decode(token,validate=True))
        except (ValueError,UnicodeError):continue
        if type(value) is dict and 'func_name' in value:configs.append(value)
    require(len(configs)==1,'One complete NKI backend config in identity HLO')
    value=configs[0]
    require(value['func_name']=='qwen35_conv_identity_dma' and value['platform_target']=='trn1'
            and value['mac_count']==0 and value['has_collectives'] is False and value['grid']==[],
            'Exact identity DMA NKI function, no arithmetic or collectives')
    return dict(inspection='whole HLO markers and unique decoded NKI backend config',
        func_name=value['func_name'],platform_target=value['platform_target'],mac_count=0,
        has_collectives=False,func_literal_sha256=sha(value['func_literal'].encode()))

def record_call(root,arm,profile,actual,shape,dtype,after,guard):
    require(arm in ARMS and profile in PROFILES and shape==SHAPE and dtype=='torch.bfloat16','Exact diagnostic output shape/dtype')
    expected=inputs(profile);positions,offsets=differences(actual,expected['cache'])
    require(set(after)==({'cache'} if arm=='identity' else set(expected)),'All actual inputs checked')
    unchanged={name:after[name]==expected[name] for name in after}
    guard('before_capture:'+arm+':'+profile)
    prefix=Path(root)/(profile+'-'+arm)
    output=write_raw(str(prefix)+'-output.raw',actual,BYTES)
    mismatch=write_raw(str(prefix)+'-differences.u32',offsets,4*BYTES)
    row=dict(arm=arm,profile=profile,shape=shape,dtype=dtype,actual_call_completed=True,
        output=output,differences=mismatch,different_bytes=len(positions),first16_different_bytes=positions[:16],
        bitwise_equal=not positions,input_unchanged=unchanged,input_after_sha256={key:sha(value) for key,value in after.items()})
    guard('after_capture:'+arm+':'+profile)
    return row

def validate_report(report,root):
    root=Path(root)
    require(report['schema']=='qwen35_slot_special_value_result_v1' and report['complete'] is True
        and report['workload']==workload() and report['runtime_calls']==6 and report['error'] is None
        and all(report[key] is False for key in ('qualification_matrix_completed','full_model_qualified','model_speedup_claim')),
        'Complete diagnostic only; never original model/primitive qualification')
    require(list(report['inputs'])==list(PROFILES) or set(report['inputs'])==set(PROFILES),'Both full input profiles')
    total=0
    for profile in PROFILES:
        require(set(report['inputs'][profile])==set(inputs(profile)),'Whole fixed input tuple')
        for key,payload in inputs(profile).items():
            reference=report['inputs'][profile][key]
            require(reference['path']==str(root/(profile+'-'+key+'.raw')) and checked(reference,BYTES)==payload,'Exact saved full input bytes')
            total+=reference['bytes']
    graphs=report['graphs'];require([x['arm'] for x in graphs]==list(ARMS),'All three exact graph arms')
    for graph in graphs:
        arm=graph['arm'];artifact=graph['artifact'];checked(artifact)
        if arm in ARTIFACTS:require(artifact==ARTIFACTS[arm],'Unchanged original compiled graph')
        else:
            require(artifact['path']==str(root/'artifacts/identity/model.pt'),'Fresh identity artifact namespace')
            require(graph['hlo']['path']==str(root/'artifacts/identity/model/graph.hlo')
                and graph['hlo_proof']==identity_hlo_proof(checked(graph['hlo'],16*2**20)),'Full identity HLO evidence')
        require(graph['load_before']==graph['load_after']==graph['post_calls']==artifact,'Stable artifact before/after loading and calls')
        require(identity(artifact['path'])==graph['load_identity'],'Same loaded artifact identity retained through completion audit')
        require(graph['device_core']==0 and graph['moved_to_device'] is True,'Actual fixed core move completed')
        code=checked(graph['forward_code'],65536).decode()
        require(graph['forward_code']['path']==str(root/(arm+'-forward.txt'))
            and ('ops.neuron.forward_v2' in code or 'neuron::forward_v2' in code),'Actual loaded Neuron invocation path')
    require([(r['arm'],r['profile']) for r in report['calls']]==[(arm,profile) for arm in ARMS for profile in PROFILES],
            'Six independent ordered diagnostic calls; mismatch must not skip an arm')
    for row in report['calls']:
        arm,profile=row['arm'],row['profile'];expected=inputs(profile)
        require(row['actual_call_completed'] is True and row['shape']==SHAPE and row['dtype']=='torch.bfloat16','Actual whole output')
        require(row['output']['path']==str(root/(profile+'-'+arm+'-output.raw'))
            and row['differences']['path']==str(root/(profile+'-'+arm+'-differences.u32')),'Exact immutable output/difference paths')
        actual=checked(row['output'],BYTES);positions,offsets=differences(actual,expected['cache'])
        require(checked(row['differences'],4*BYTES)==offsets and row['different_bytes']==len(positions)
            and row['first16_different_bytes']==positions[:16] and row['bitwise_equal']==(not positions),'Every differing position independently rederived')
        keys={'cache'} if arm=='identity' else set(expected)
        require(row['input_unchanged']==dict.fromkeys(keys,True)
            and row['input_after_sha256']=={key:sha(expected[key]) for key in keys},'All inputs remain byte-identical')
        total+=row['output']['bytes']+row['differences']['bytes']
    require(total<=PAYLOAD_CAP,'Whole raw+position payload budget')
    return dict(layout=NAME,diagnostic_complete=True,calls=6,graphs=3,
        bitwise_matches={arm:{row['profile']:row['bitwise_equal'] for row in report['calls'] if row['arm']==arm} for arm in ARMS},
        qualification_matrix_completed=False,full_model_qualified=False,model_speedup_claim=False)
