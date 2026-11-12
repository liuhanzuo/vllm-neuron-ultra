"""Reconstruct the actual v2 candidate with the frozen runtime parser only.

Does not import either preparation generator or requirement parser. Reads
candidate JSON, accepted provider source, and original installed RECORD text.
No native payload or model weight read; no runtime admission is created.
"""
import builtins
import hashlib
import json
from pathlib import Path
import sys

BLOCKED={'torch','numpy','safetensors','transformers','ctypes','torch_xla','torch_neuronx',
         'neuronx_distributed','neuronx_distributed_inference','vllm','vllm_neuron',
         'tokenizers','ml_dtypes','huggingface_hub','packaging'}
original_import=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0]in BLOCKED:raise AssertionError('Forbidden import: '+name)
    return original_import(name,*args,**kwargs)
builtins.__import__=guarded

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
CANDIDATE=ROOT/'research/spark26-native-manifest-candidate-20261112-v2'
EXPECTED={
    'manifest.json':'ee81ffa1a4465b972ebf6cf73dc472789946ff244818dd0a4375276e4d603524',
    'source-map.json':'58a4d127ea86b6b80bb665f47b9a7a3cb513817439abc7488ea7b302e590e863',
    'preparation.json':'25b9b086e504f828e099ba685630dc9d9d2cc6912cac1bd9332ccdb99d9a949e',
    'text-inputs.json':'5ac96241d27890170c01d8e4ead1789c09ce8f26d1f3c7edb732b286ac55f013',
    'selected-distributions.json':'68e4ba209687e8a3592a97adc4b78fc370bfde8da896c5cc45455cee8f7b00f7',
}


def exact_text(path,wanted):
    path=Path(path)
    assert path.resolve()==path and not path.is_symlink()
    assert path.name=='RECORD' or path.suffix in ('.json','.py')
    assert '.so' not in path.name and path.stat().st_size<64*2**20
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==wanted
    return raw.decode()


def main():
    exact_text(ROOT/'spark26_native_provider_v2.py','12f9d6e63dbe112359d3acd32984d69d993b7fddfee80ef670d4fd901aed93e6')
    import spark26_native_provider_v2 as runtime
    artifacts={name:json.loads(exact_text(CANDIDATE/name,sha))for name,sha in EXPECTED.items()}
    manifest=artifacts['manifest.json'];sources=artifacts['source-map.json'];preparation=artifacts['preparation.json']
    reads=[]
    def record(ref):
        assert Path(ref['path']).name=='RECORD'
        assert ref==artifacts['text-inputs.json'][ref['path']]
        text=exact_text(ref['path'],ref['sha256']);reads.append(ref['path']);return text
    payloads=runtime.merge_manifest(manifest,sources,record)
    assert len(payloads)==41173 and sum(x['bytes']for x in payloads.values())==13247146338
    assert len(reads)==178 and len(set(reads))==178
    assert not manifest['python_overrides'] and len(manifest['additional_files'])==2619
    assert {k:manifest[k]for k in ('payload_count','payload_bytes','payload_sha256')}=={
        'payload_count':len(payloads),'payload_bytes':sum(x['bytes']for x in payloads.values()),
        'payload_sha256':runtime.contract.digest(payloads)}
    assert preparation['graph']['unresolved']==[dict(kind='missing_installed_distribution',name='accelerate')]
    assert len(preparation['elf']['unresolved'])==1271 and preparation['elf']['exact_saved_objects']==0
    conflicts=[i for i in preparation['issues']if i['kind']=='ambient_conflicting_RECORD_owner']
    assert len(conflicts)==15 and {i['unselected_distribution']for i in conflicts}=={'opencv_python'}
    assert any(i['path'].endswith('/cv2/cv2.abi3.so')for i in conflicts)
    assert any(i['kind']=='dynamic_import_closure_not_proven_by_declared_metadata'for i in preparation['issues'])
    assert preparation['runtime_admitted'] is False and preparation['complete_runtime_closure'] is False
    assert preparation['actual_native_hash_authentication'] is False and preparation['actual_native_imports'] is False
    assert preparation['weights_read'] is False and preparation['native_payload_bytes_read']==0
    assert not BLOCKED&set(sys.modules)
    assert 'spark26_native_manifest_generator_v2'not in sys.modules and 'spark26_manifest_requirements_v2'not in sys.modules
    result=dict(schema='spark26_native_manifest_candidate_reconstruction_v1',
        candidate=str(CANDIDATE),artifacts_sha256=EXPECTED,frozen_runtime_parser_reconstruction_passed=True,
        separate_validator_does_not_import_generator=True,metadata_RECORDs_rehashed=len(reads),
        payload_count=len(payloads),payload_bytes=manifest['payload_bytes'],payload_sha256=manifest['payload_sha256'],
        runtime_authentication_budget_bytes=runtime.MAX_RUNTIME_AUTHENTICATION_READ,
        within_authentication_hash_budget=True,original_16_root_versions_preserved=True,
        missing_selected_distributions=['accelerate'],ambient_conflicting_owners=15,
        missing_original_ELF_evidence=1271,dynamic_import_closure_proven=False,
        candidate_runtime_admitted=False,actual_native_imports=False,actual_native_payload_reads=False,
        weights_read=False,packages_modified=False,serial_or_process_operations=False,new_speedup_claim=False)
    path=ROOT/'research/spark26-native-manifest-candidate-validation-20261112-v1.json'
    with path.open('x')as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
