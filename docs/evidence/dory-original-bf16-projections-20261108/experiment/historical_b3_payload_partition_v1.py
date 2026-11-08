"""Pure metadata partition for the exact completed Phi4 b3 evidence set.

No native/process/service/payload IO. Frozen catalog objects remain unchanged.
A custom reader is for source tests; production uses SHA-authenticated JSON.
"""
import hashlib
import json
from pathlib import Path
import re
import stat

ROOT=Path(__file__).resolve().parent
RESEARCH=ROOT/'research'
MIB=2**20
FORBIDDEN=frozenset(('.pt','.raw','.safetensors','.neff','.so','.bin','.zlib','.whl','.pb'))
CATALOG=dict(path=str(RESEARCH/'root-post-phi4-qkv-b3-complete-predecessors-20261107-v1.json'),sha256='617e4680da3476aef5c2f40892307b120e84dcfbe462b225e292cb120cf78a76')
AUDIT=dict(path=str(RESEARCH/'phi4-same-input-qkv-b3-root-terminal-audit-20261107-v1.json'),sha256='f2859ded8955cacfd40d7436eecbc385f353b7b8cc4dfe6cce0621635c38f507')
PREVIOUS=dict(path=str(RESEARCH/'root-post-phi4-qkv-b2-failed-complete-predecessors-20261107-v1.json'),sha256='6eb2b8e31a7516a441e4fe975c7ba2c83dc9281dd7063ca6906664b74b450c05')
WHITELIST = [{'bytes': 40960, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-21-source-b4.raw', 'sha256': '3f785f1089442bd27a63e2971d183a47de1f73761a856b217c9b525aa0aeee55'}, {'bytes': 10240, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-21-source-b1-witness.raw', 'sha256': '52d7b8a40ba708c90b3595cd3efcb9403e5caf1287b66be6553ee964854a44fe'}, {'bytes': 10240, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-21-source-b1-control.raw', 'sha256': 'f5cb458d249768acf42eccfd6d4027505126fab1737acb7a3770f325e7f10fb2'}, {'bytes': 40960, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-21-row-views.raw', 'sha256': '936b5052c6761ce40fb22b46d62eea9fbbea7fcad19bb836cd11198e14751ecd'}, {'bytes': 10240, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-21-isolated-witness.raw', 'sha256': '52d7b8a40ba708c90b3595cd3efcb9403e5caf1287b66be6553ee964854a44fe'}, {'bytes': 10240, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-21-isolated-control.raw', 'sha256': 'f5cb458d249768acf42eccfd6d4027505126fab1737acb7a3770f325e7f10fb2'}, {'bytes': 40960, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-25-source-b4.raw', 'sha256': '6dfce528e87c31e8b03c9f46f8ad20770c285f8ff39572da413ec22185ad24bc'}, {'bytes': 10240, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-25-source-b1-witness.raw', 'sha256': '9233976d2e8713d0851db142e773a9c0e7690b4da11640c8957ff26fb94ccd30'}, {'bytes': 40960, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-25-row-views.raw', 'sha256': 'f82dc764bb81e31f7960392848b93dc3150af272471e97728e5a6708925de55d'}, {'bytes': 10240, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-25-isolated-witness.raw', 'sha256': '9233976d2e8713d0851db142e773a9c0e7690b4da11640c8957ff26fb94ccd30'}, {'bytes': 40960, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-29-source-b4.raw', 'sha256': '56bea1d39fbca01d60c6c01346a9a0968a4ab2cddcd056b93b4fb99ed564386d'}, {'bytes': 10240, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-29-source-b1-witness.raw', 'sha256': 'dd265a22e2ed9f39ee2653388841be0c0bcea7a71ef2ed3a298f662616a090f0'}, {'bytes': 40960, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-29-row-views.raw', 'sha256': '8289cd7d126cf2ddca56745a7c37c809dfefea252f1c19fc08c621bd2ad4a2f9'}, {'bytes': 10240, 'durable': True, 'exclusive': True, 'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/phi4-same-input-qkv-control-b3/frame-29-isolated-witness.raw', 'sha256': 'dd265a22e2ed9f39ee2653388841be0c0bcea7a71ef2ed3a298f662616a090f0'}]

def require(value,reason):
    if not value:raise RuntimeError(reason)


def read_ref(ref):
    require(type(ref)is dict and set(ref)=={'path','sha256'},'Exact historical metadata reference')
    require(type(ref['path'])is str and type(ref['sha256'])is str and re.fullmatch('[0-9a-f]{64}',ref['sha256']),'Literal reference fields')
    path=Path(ref['path'])
    require(path.parent==RESEARCH and path.suffix=='.json'and path.is_absolute()and str(path)==ref['path']and path.resolve()==path,'Bounded research JSON only')
    require(not any(x.is_symlink()for x in(path,*path.parents)),'No source symlinks')
    before=path.stat();require(stat.S_ISREG(before.st_mode)and before.st_size<=8*MIB,'Bounded regular metadata')
    raw=path.read_bytes();after=path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)and len(raw)<=8*MIB,'Metadata read stable')
    require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Historical metadata digest')
    return json.loads(raw)


def pin_shape(mapping):
    require(type(mapping)is dict,'Pin mapping')
    for name,digest in mapping.items():
        require(type(name)is str and Path(name).is_absolute()and str(Path(name))==name and '..'not in Path(name).parts
                and type(digest)is str and re.fullmatch('[0-9a-f]{64}',digest),'Literal absolute pin and SHA')


def partition(final_catalog,read_ref=read_ref):
    """Return exactly three maps; input remains untouched, raw is never opened."""
    baseline=read_ref(dict(CATALOG));audit=read_ref(dict(AUDIT))
    require(baseline['schema']=='root_post_phi4_qkv_b3_complete_predecessors_v1'
            and len(baseline['rows'])==372 and len(baseline['processes'])==27,'Exact completed b3 baseline')
    require(audit['schema']=='phi4_same_input_qkv_root_terminal_audit_v1'and audit['reviewer']=='/root'
            and audit['passed']is audit['diagnostic_complete']is audit['all_groups_drained']is True
            and type(audit['actual_exit_code'])is int and audit['actual_exit_code']==0
            and audit['previous_catalog']==PREVIOUS and audit['added_standalone_process']==baseline['processes'][-1]
            and audit['complete_rows']==372 and audit['complete_processes']==27,'Authenticated completed b3 provenance')
    result=audit['result']
    require(result['schema']=='phi4_same_input_qkv_replay_v1'and result['diagnostic_complete']is True
            and all(row[key]is False for row in(audit,result)for key in('full_model_qualified','new_speedup_claim','global_first_cause_proven')),'Original b3 diagnostic limitations')
    outputs=result['outputs'];require(type(outputs)is list and len(outputs)==14,'Exact14 b3 outputs')
    expected={row['path']:row for row in WHITELIST};observed={}
    for item in outputs:
        require(type(item)is dict and set(item)=={'reference','metadata'},'Exact original output metadata')
        ref=item['reference'];meta=item['metadata']
        require(type(ref)is dict and set(ref)=={'path','sha256','bytes','durable','exclusive'}and ref.get('path')in expected,'Exact reviewed output reference')
        require(type(ref['bytes'])is int and ref['bytes']>0 and ref['durable']is ref['exclusive']is True and ref==expected[ref['path']],'Literal original five-field output reference')
        require(ref['path']not in observed,'No duplicate historical outputs');observed[ref['path']]=ref['sha256']
        require(type(meta)is dict and set(meta)=={'bytes','dtype','label','ordinal','sha256','shape'}and type(meta['bytes'])is int
                and meta['bytes']==ref['bytes']and meta['sha256']==ref['sha256']and meta['dtype']=='bfloat16'
                and type(meta['ordinal'])is int and meta['ordinal']in(21,25,29)and type(meta['label'])is str
                and Path(ref['path']).name=='frame-'+str(meta['ordinal'])+'-'+meta['label']+'.raw','Matching historical metadata')
        expected_shape=[4,1,5120]if meta['label']in('source-b4','row-views')else[1,1,5120]
        require(type(meta['shape'])is list and all(type(x)is int for x in meta['shape'])and meta['shape']==expected_shape
                and meta['bytes']==2*expected_shape[0]*expected_shape[1]*expected_shape[2],'Exact original bfloat16 shape/bytes')
    require(observed=={path:row['sha256']for path,row in expected.items()}and sum(row['bytes']for row in WHITELIST)==327680
            and type(audit['new_raw_bytes'])is type(result['new_raw_bytes'])is int
            and audit['new_raw_bytes']==result['new_raw_bytes']==327680,'Exact14/327680 metadata budget')
    pin_shape(audit['evidence_sha256']);pin_shape(baseline['source_sha256'])
    for mapping in(audit['evidence_sha256'],baseline['source_sha256']):
        require({k:v for k,v in mapping.items()if Path(k).suffix in FORBIDDEN}==observed,'Only exact reviewed historical payload subset')
    require(type(final_catalog)is dict and final_catalog['reviewer']=='/root'
            and final_catalog['root_finalized']is final_catalog['all_planned_launches_accounted_for']is True
            and type(final_catalog['complete_count'])is int and final_catalog['complete_count']==len(final_catalog['rows'])
            and len(final_catalog['rows'])>=372 and len(final_catalog['processes'])>=27,'Complete current root catalog')
    require(final_catalog['rows'][:372]==baseline['rows']and final_catalog['processes'][:27]==baseline['processes'],'All historical rows and process objects retained')
    dynamic={'schema','rows','processes','complete_count','source_sha256','root_finalization_observed_at','appended_from','appended_terminal_audit'}
    require(all(final_catalog.get(k)==v for k,v in baseline.items()if k not in dynamic),'All historical extra fields retained')
    pins=final_catalog['source_sha256'];pin_shape(pins)
    require(all(pins.get(k)==v for k,v in baseline['source_sha256'].items()),'Every historical source and payload fingerprint retained')
    historical={k:v for k,v in pins.items()if Path(k).suffix in FORBIDDEN}
    require(historical==observed,'No unknown or missing payload fingerprint')
    source={k:v for k,v in pins.items()if k not in historical}
    require(not set(source)&set(historical)and dict(source,**historical)==pins,'Lossless disjoint pin partition')
    provenance={row['path']:row['sha256']for row in(CATALOG,AUDIT)}
    require(all(k not in source or source[k]==v for k,v in provenance.items()),'No conflicting historical metadata pin')
    return dict(source_sha256=source,historical_payload_sha256=historical,provenance_sha256=provenance)
