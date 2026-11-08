"""Independent CPU-only raw verifier, invoked solely after actual worker lease."""
import hashlib
import io
import math
import os
from pathlib import Path
import nemotron9b_package_protocol_v14 as p
from nemotron9b_package_evidence_v14 import RAW,REPORT,COUNTS,digest,identity,verify_phase,tree,snapshot_inputs
from nemotron9b_package_science_v14 import metrics,passed,expected_long_frames,PREFIXES


def same_metrics(actual,claimed):
    p.require(set(actual)<=set(claimed),'Complete per-row reported metrics')
    for key,value in actual.items():
        old=claimed[key]
        if type(value)in (bool,int):p.require(type(old)is type(value)and old==value,'Exact discrete metric '+key)
        else:p.require(type(old)in (float,int)and math.isfinite(old)and math.isclose(old,value,rel_tol=1e-6,abs_tol=1e-6),
            'Recomputed saved-output metric '+key)


def expected_numeric(reference):
    p.require(len(reference)==6 and len({row['id']for row in reference})==6,'Six original native numeric fixtures')
    order=[0,5,2,4,1,3,0,2];slots=[3,1,0,2];rows=[]
    for start in (0,4):
        group=[reference[i]for i in order[start:start+4]]
        for step in range(4):
            for item,slot in zip(group,slots):
                p.require(len(item['logits'])==4 and len(item['continuation_ids'])==3,'Original four numeric checkpoints')
                rows.append(dict(key=dict(fixture=item['id'],phase='prefill'if step==0 else'mixed_length_decode',step=step),
                    slot=slot,reference=item['logits'][step]))
    return rows


def expected_cold(reference,metadata):
    p.require([r['job']for r in reference]==metadata['jobs'],'Exact14 cached cold jobs')
    mapping={(r['job']['prefix'],r['job']['fixture']):r for r in reference}
    p.require(len(mapping)==14,'Distinct cold jobs')
    return [dict(key=dict(prefix=prefix,step=step,slot=slot,fixture=slot%2,position=prefix-1+step),
        reference=mapping[prefix,slot%2]['logits'][step])for prefix in PREFIXES for step in range(3)for slot in range(4)]


def expected_long(reference,metadata):
    p.require(len(reference)==len(metadata['fixtures'])==5,'Five original cached long trajectories')
    for actual,meta in zip(reference,metadata['fixtures']):
        p.require(actual['fixture']==meta['fixture']and actual['positions']==meta['positions'],'Original long reference metadata')
    rows=[]
    for key in expected_long_frames(metadata):
        source=reference[key['trajectory']]
        rows.append(dict(key=key,reference=source['logits'][source['positions'].index(key['position'])]))
    return rows


def run(spec,owner,lease,guard):
    """Independent process: no models, compiler, device work or new inference."""
    import torch
    torch.set_num_threads(1)
    c=Path(spec['control_root']);e=Path(spec['evidence_root']);phases={};manifests={}
    for stage in p.SCIENTIFIC_STAGES:
        guard();record=p.read(c/(stage+'-worker-result.json'));producer=p.read(c/(stage+'-owner.json'))
        p.require(record['producer']==p.producer(producer)and record['phase']==stage and record['passed']is True,'Original scientific producer')
        p.validate_result(spec,stage,record['result'])
        manifest=verify_phase(spec,stage,record['result'],producer,record['lease'],guard)
        phases[stage]=p.ref(c/(stage+'-worker-result.json'));manifests.update(manifest['files'])
    def load(path,expected):
        guard();actual=digest(path,guard)
        p.require(actual==expected,'Exact pre-load tensor identity')
        p.require(actual['stat']['size']<=512*p.MIB,'Bounded independent CPU tensor input')
        # Parse the same bounded in-memory bytes whose digest is checked. A
        # separate path-based torch.load could race a same-stat replacement.
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            with os.fdopen(fd,'rb')as stream:data=stream.read(actual['stat']['size']+1)
        except BaseException:raise
        p.require(len(data)==actual['stat']['size']and hashlib.sha256(data).hexdigest()==actual['sha256']
            and identity(path)==actual['stat'],'Exact tensor bytes before independent deserialization')
        rows=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
        guard();return rows
    proofs={}
    for phase in ('numeric','cold','long'):
        fresh_path=str(e/RAW[phase]);fresh=load(fresh_path,manifests[fresh_path])
        ref_path=spec['references'][phase];old_path=spec['references']['historical_'+phase]
        reference=load(ref_path,dict(path=ref_path,**spec['payloads'][ref_path]))
        historical=load(old_path,dict(path=old_path,**spec['payloads'][old_path]))
        if phase=='numeric':expected=expected_numeric(reference)
        else:
            ref=spec['reference_metadata'][phase];metadata=p.read(ref['path'],ref['sha256'])
            expected=(expected_cold if phase=='cold'else expected_long)(reference,metadata)
        p.require(len(fresh)==len(expected)==len(historical)==COUNTS[phase],'All independent raw frames')
        report=p.read(e/REPORT[phase]);claims=report['rows']if phase!='cold'else [r for ex in report['experiments']for r in ex['rows']]
        migration=p.read(e/(phase+'-migration.json'))
        p.require(len(claims)==len(migration['rows'])==COUNTS[phase],'All saved metric and migration rows')
        rows=[]
        for item,old,wanted,claim,prior in zip(fresh,historical,expected,claims,migration['rows']):
            guard();key={k:v for k,v in item.items()if k!='logits'}
            p.require(key==wanted['key']=={k:v for k,v in old.items()if k!='logits'},'Exact saved/original/historical frame identity')
            p.require(all(claim[k]==v for k,v in key.items()),'Saved report frame identity')
            if phase=='numeric':p.require(claim['slot']==wanted['slot'],'Original permuted request slot')
            actual=item['logits'];baseline=old['logits'];oracle=wanted['reference']
            p.require(isinstance(actual,torch.Tensor)and isinstance(baseline,torch.Tensor)and isinstance(oracle,torch.Tensor),'Raw tensor types')
            p.require(actual.dtype==baseline.dtype==torch.float32 and actual.shape==baseline.shape==(131072,)and oracle.numel()==131072,
                'Exact complete-vocabulary raw shape/dtype')
            p.require(bool(torch.isfinite(actual).all())and bool(torch.isfinite(baseline).all())and bool(torch.isfinite(oracle).all()),'Finite complete vocabularies')
            checked=metrics(actual,oracle);migrated=metrics(actual,baseline)
            same_metrics(checked,claim['metrics']if phase=='cold'else claim);same_metrics(migrated,prior['metrics'])
            exact=torch.equal(actual.contiguous().view(torch.uint8),baseline.float().contiguous().view(torch.uint8))
            p.require(prior['bitwise_equal']is exact,'Recomputed historical bitwise outcome')
            rows.append(dict(key=key,metrics=checked,migration=migrated,bitwise_equal=exact,
                shape=list(actual.shape),dtype=str(actual.dtype)))
        p.require(passed([row['metrics']for row in rows])and passed([row['migration']for row in rows]),'Unchanged native and historical numeric criteria')
        if phase=='cold':
            keyed={(r['prefix'],r['step'],r['slot']):r['logits']for r in fresh}
            for prefix in PREFIXES:
                for step in range(3):
                    for a,b in ((0,2),(1,3)):
                        p.require(torch.equal(keyed[prefix,step,a].contiguous().view(torch.uint8),keyed[prefix,step,b].contiguous().view(torch.uint8)),
                            'Saved cold duplicate-slot raw-byte equality')
        proofs[phase]=dict(complete=True,passed=True,frames=len(rows),rows=rows)
        del fresh,reference,historical,expected
    guard();excluded=[str(e/'raw-verification.json'),str(e/'final-raw-manifest.json')]
    closure=dict(evidence=tree(e,guard,True,spec,exclude=excluded),work=tree(Path(spec['work_root']),guard,spec=spec),inputs=snapshot_inputs(spec,guard))
    admission=p.read(c/'admission.json')
    final=dict(complete=True,producer=p.producer(owner),lease=lease,execution_spec=admission['execution_spec'],
        source_review=admission['source_review'],source_sha256=spec['source_sha256'],scientific_results=phases,closure=closure)
    p.once(e/'final-raw-manifest.json',final)
    report=dict(complete=True,passed=True,phases=phases,raw_replay=proofs,final_manifest=p.ref(e/'final-raw-manifest.json'),
        independent_process=True,inference_performed=False,models_loaded=0,original_scientific_stages=51)
    p.once(e/'raw-verification.json',report)
    return dict(complete=True,passed=True,report=p.ref(e/'raw-verification.json'))
