"""Lossless inherited source partition; explicit once-only outer native hashing.

No model import, launch, completion call or implicit native/payload read. Every
original pin remains in the scientific plan. Source-only verification opens
only reviewed text/three original Python-byte fixtures and exact metadata aliases.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import nemotron9b_package_metadata_v27 as metadata
import nemotron9b_package_metadata_alias_v1 as aliases
from minicpm5_layer0_runtime_auth_v1 import digest,identity
from minicpm5_original88_decode_mask_analysis_v1 import p

ROOT=Path(__file__).resolve().parent.parent
Q=ROOT/'research'
R12_SPEC=dict(path=str(Q/'nemotron9b-package-runtime-v27-r12-20261110-spec.json'),
    sha256='4186c9a29921b96fc3a55fb04040cf70977ca921130df288466c8d27e65feaaa')
CONTRACT_SCHEMA='minicpm5_layer1_sdk_batch_layout_source_contract_v2'
PARTITION_SCHEMA='minicpm5_layer1_sdk_batch_layout_source_partition_v2'
AUTH_SCHEMA='minicpm5_layer1_sdk_batch_layout_source_authentication_v2'
CATEGORIES=('text_sha256','historical_payload_sha256','python_bytes_sha256','query_native_binary_sha256')
ENVELOPES={
    'spec':'minicpm5_layer1_sdk_batch_layout_cpu_execution_spec_v2',
    'source-review':'minicpm5_layer1_sdk_batch_layout_runtime_source_review_v2',
    'plan':'minicpm5_layer0_runtime_inputs_v1',
}
require=metadata.require


def reference(ref):
    require(type(ref)is dict and set(ref)in ({'path','sha256'},{'path','sha256','bytes'})
        and type(ref['path'])is str and Path(ref['path']).is_absolute()
        and type(ref['sha256'])is str and re.fullmatch('[a-f0-9]{64}',ref['sha256']),
        'Exact absolute authenticated metadata reference')
    require('bytes'not in ref or type(ref['bytes'])is int and ref['bytes']>=0,'Optional exact metadata bytes')


def envelope_kind(path):
    path=Path(path)
    if path.parent!=Q:return None
    match=re.fullmatch(r'minicpm5-layer1-sdk-batch-layout-([a-z0-9-]{1,80})-(source-review|spec|plan)\.json',path.name)
    return match[2]if match else None


def envelope_shape(path,value):
    role=envelope_kind(path)
    require(role is not None and type(value)is dict and value.get('schema')==ENVELOPES[role],
        'Exact new Mini path and schema envelope')
    metadata.pin_shape(value['source_sha256'])
    if role=='source-review':
        reference(value['spec'])
        require(type(value['passed'])is bool and value['reviewer']=='/root'and type(value['blockers'])is list,
            'Exact independent source review envelope')
    else:
        reference(value['source_contract'])
        require(type(value['latest_completion'])is dict and type(value['output_root'])is str,
            'Exact new Mini lineage and source contract envelope')
        if role=='plan':require(type(value['inputs'])is dict and len(value['inputs'])<=100000,'Complete bounded original input map')
        else:reference(value['plan'])
    return role


def text(path,expected=None):
    # Existing historical envelopes/aliases stay with their reviewed transports.
    if envelope_kind(path)is None:return metadata.text(path,expected)
    require(type(expected)is str and re.fullmatch('[a-f0-9]{64}',expected),
        'New large Mini envelopes require a frozen digest before reading')
    raw=metadata.stable_bytes(path,metadata.ENVELOPE_LIMIT)
    require(hashlib.sha256(raw).hexdigest()==expected,'Exact new Mini envelope digest')
    value=json.loads(raw.decode('utf8'));envelope_shape(path,value);return raw


def load(ref):
    reference(ref);require(Path(ref['path']).suffix=='.json','Metadata JSON only')
    raw=text(ref['path'],ref['sha256'])
    require('bytes'not in ref or len(raw)==ref['bytes'],'Exact metadata reference length')
    return json.loads(raw)


def classify(pins,contract_ref,read=load):
    """Only authenticated metadata classification; never opens a binary pin."""
    reference(contract_ref)
    path=Path(contract_ref['path'])
    require(path.parent==Q and re.fullmatch(r'minicpm5-layer1-sdk-batch-layout-source-contract-[a-z0-9-]{1,80}\.json',path.name),
        'Exact Mini source contract namespace')
    metadata.pin_shape(pins)
    require(pins.get(contract_ref['path'])==contract_ref['sha256'],'Source contract retained in complete source map')
    contract=read(contract_ref)
    require(type(contract)is dict and set(contract)=={'schema','r12_spec','metadata_limits'}
        and contract['schema']==CONTRACT_SCHEMA and contract['r12_spec']==R12_SPEC
        and contract['metadata_limits']==metadata.LIMITS,'Exact original r12 source classification contract')
    anchor=read(R12_SPEC)
    require(anchor['schema']=='nemotron9b_package_runtime_v27'and anchor['run_id']=='r12-20261110'
        and anchor['metadata_limits']==metadata.LIMITS and pins.get(R12_SPEC['path'])==R12_SPEC['sha256'],
        'Exact fixed original r12 anchor retained')
    require(all(pins.get(path)==sha for path,sha in anchor['source_sha256'].items()),
        'All inherited pins retained unchanged before classification')
    catalog=read(anchor['final_predecessor_catalog'])
    # A read-only execution view extends the original classifier's pin map. No
    # archived spec or scientific plan object is changed or projected downward.
    view=copy.deepcopy(anchor);view['source_sha256']=copy.deepcopy(pins)
    aliases.validate_spec(view)
    categories=metadata.classify_sources(view,catalog)
    require(set(categories)==set(CATEGORIES),'Only original four reviewed source categories')
    combined={}
    for name in CATEGORIES:
        values=categories[name];require(not set(combined)&set(values),'Disjoint source categories')
        combined.update(values)
    require(combined==pins,'Lossless complete original and successor source map')
    observation=read(metadata.query_sources.OBSERVATION)
    require(categories['query_native_binary_sha256']==metadata.query_sources.native_pins(observation),
        'Exact observed nine native dependencies')
    require(len(categories['historical_payload_sha256'])==20 and len(categories['python_bytes_sha256'])==3,
        'Original20 historical payload and3 non-UTF8 categories')
    require(not any('.so.'in Path(path).name or Path(path).suffix in metadata.history.FORBIDDEN
        for path in categories['text_sha256']),'Unknown native/payload cannot masquerade as text')
    return dict(schema=PARTITION_SCHEMA,source_contract=copy.deepcopy(contract_ref),r12_spec=R12_SPEC,
        source_map_sha256=digest(pins),partition_sha256=digest(categories),categories=categories,
        metadata_aliases=copy.deepcopy(view['metadata_aliases']),
        query_observation_ref=copy.deepcopy(metadata.query_sources.OBSERVATION),query_observation=observation)


def check_partition(partition):
    require(partition['schema']==PARTITION_SCHEMA and set(partition['categories'])==set(CATEGORIES),
        'Exact immutable source partition')
    combined={}
    for values in partition['categories'].values():
        require(not set(combined)&set(values),'No overlapping source classes');combined.update(values)
    require(digest(combined)==partition['source_map_sha256']
        and digest(partition['categories'])==partition['partition_sha256'],'Complete source and partition digest')
    require(partition['r12_spec']==R12_SPEC
        and partition['query_observation_ref']==metadata.query_sources.OBSERVATION
        and partition['categories']['query_native_binary_sha256']==metadata.query_sources.native_pins(partition['query_observation']),
        'Fixed original r12 and native observation binding')
    aliases.validate_spec(dict(source_sha256=combined,metadata_aliases=partition['metadata_aliases']))
    return combined


def verify_text_sources(partition,guard):
    """Source-only path: no historical raw bytes or native libraries opened."""
    require(callable(guard),'Explicit observer/runtime guard');guard();check_partition(partition)
    for path,sha in partition['categories']['text_sha256'].items():
        guard()
        actual=(aliases.source_sha(path,partition['metadata_aliases'],text)
            if path in partition['metadata_aliases']else hashlib.sha256(text(path,sha)).hexdigest())
        require(actual==sha,'Unchanged exact text source')
    for path,sha in partition['categories']['python_bytes_sha256'].items():
        guard();metadata.python_source_bytes(path,sha)
    guard()
    return dict(complete_text_verification=True,source_map_sha256=partition['source_map_sha256'],
        partition_sha256=partition['partition_sha256'],historical_payload_bytes_read=0,native_bytes_read=0)


def native_identities(partition):
    expected=partition['query_observation']['source_and_binary_pins'];result={}
    for path in sorted(partition['categories']['query_native_binary_sha256']):
        row=identity(path);pin=expected[path]
        require((row['device'],row['inode'],row['size'])==(pin['device'],pin['inode'],pin['bytes']),
            'Exact original native identity and byte count')
        result[path]=row
    return result


def authenticate_native_sources(partition,outer_owner,lock_identity,attempt_id,guard):
    """Outer only, under actual original lock; call once before service launch."""
    check_partition(partition)
    require(outer_owner['pid']==os.getpid() and re.fullmatch('[a-f0-9]{32}',attempt_id),
        'Only actual outer process authenticates native source bytes')
    def lease():
        guard();p.lock_proof(outer_owner,lock_identity)
    lease();before=native_identities(partition)
    report=metadata.query_sources.authenticate_native_sources(
        partition['categories']['query_native_binary_sha256'],partition['query_observation'],lease)
    lease();require(native_identities(partition)==before,'Native file identities unchanged across complete authentication')
    receipt=dict(schema=AUTH_SCHEMA,source_contract=copy.deepcopy(partition['source_contract']),
        source_map_sha256=partition['source_map_sha256'],partition_sha256=partition['partition_sha256'],
        query_observation=partition['query_observation_ref'],native_sha256=copy.deepcopy(partition['categories']['query_native_binary_sha256']),
        native_identities=before,native_authentication=report,outer_owner=copy.deepcopy(outer_owner),
        lock_identity=copy.deepcopy(lock_identity),attempt_id=attempt_id,
        historical_payload_bytes_read=0,native_libraries_loaded_here=False,
        full_model_qualified=False,new_speedup_claim=False)
    validate_native_receipt(receipt,partition,outer_owner,lock_identity,attempt_id)
    return receipt


def validate_native_receipt(receipt,partition,outer_owner,lock_identity,attempt_id):
    """Pure binding validation for worker or post-exit independent auditor."""
    check_partition(partition)
    require(receipt['schema']==AUTH_SCHEMA and receipt['source_contract']==partition['source_contract']
        and receipt['source_map_sha256']==partition['source_map_sha256']
        and receipt['partition_sha256']==partition['partition_sha256']
        and receipt['query_observation']==partition['query_observation_ref']
        and receipt['native_sha256']==partition['categories']['query_native_binary_sha256']
        and receipt['outer_owner']==outer_owner and receipt['lock_identity']==lock_identity
        and receipt['attempt_id']==attempt_id,'Exact source authentication and externally observed original outer')
    require(receipt['native_authentication']==dict(complete=True,files=9,bytes=metadata.query_sources.NATIVE_BYTES,
        native_libraries_loaded_here=False,model_or_tensor_bytes_read=0,full_runtime_environment_verified=False),
        'Complete original bounded nine-library hash authentication')
    require(receipt['historical_payload_bytes_read']==0 and receipt['native_libraries_loaded_here']is False
        and receipt['full_model_qualified']is receipt['new_speedup_claim']is False,'No invented model/runtime qualification')
    rows=receipt['native_identities'];expected=partition['query_observation']['source_and_binary_pins']
    require(set(rows)==set(receipt['native_sha256']),'Every native identity retained')
    for path,row in rows.items():
        require(set(row)=={'realpath','device','inode','size','mtime_ns','ctime_ns'}and row['realpath']==path
            and all(type(row[k])is int and row[k]>=0 for k in ('device','inode','size','mtime_ns','ctime_ns'))
            and (row['device'],row['inode'],row['size'])==(expected[path]['device'],expected[path]['inode'],expected[path]['bytes']),
            'Exact recorded native identity tied to original dependency observation')
    return dict(native_hash_authentication_inherited=True,native_bytes_read_here=0,historical_payload_bytes_read_here=0)


def check_native_identities(receipt,partition,guard):
    """Stat only. The guard belongs to worker admission or independent audit."""
    guard();require(native_identities(partition)==receipt['native_identities'],'Authenticated native identities still unchanged');guard()
    return dict(all_nine_native_identities_unchanged=True,native_bytes_read_here=0)
