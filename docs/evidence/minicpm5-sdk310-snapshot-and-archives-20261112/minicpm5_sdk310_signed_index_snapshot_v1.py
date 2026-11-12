"""Bounded old signed XZ to immutable plaintext; no payload IO at import.

Only run_snapshot reads compressed payloads, and its caller must already hold
reviewed serial admission. Pure decode_exact accepts an injected decoder for tests.
"""
import copy
import datetime as dt
import hashlib
import os
from pathlib import Path
import stat
import dory_serial_stage_contract_v2 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import minicpm5_sdk310_retired_lfm_r2_v2 as retirement

WORK_KIND='minicpm5_sdk310_signed_index_snapshot_v1'
ASSESSMENT=dict(path=str(c.Q/'minicpm5-sdk310-signed-index-snapshot-source-assessment-20261112-v1.json'),sha256='c1b98284c6486b8e870a9446412bcf42782bfe605747a79699cc94a7e2b1d7d0')
OLD_MANIFEST=dict(path=str(c.Q/'minicpm5-sdk310-deb-source-manifest-20261112-v3.json'),sha256='2fe3541566bcd56932df09b808f2f5fcee6ee2d8a28235c308f92b2f02a184de')
CATALOG=dict(path=str(c.Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v79.json'),sha256='383e3565f30c12c454049274f9c3f083c79dc3feeb62a78d20491a29dd797918')
NEW_NAMES=('minicpm5_sdk310_signed_index_snapshot_v1.py','run_minicpm5_sdk310_signed_index_snapshot_v1.py')
SCHEMA='minicpm5_sdk310_signed_index_snapshot_v1'
EMPTY_TAIL=bytes.fromhex('fd377a585a000004e6d6b446000000001cdf44211fb6f37d010000000004595a')
MEMLIMIT=64*c.MIB
RAW_TOTAL=50319218


def metadata(read=c.load):
    assessment=read(ASSESSMENT);n=read(assessment['frozen_mini_inputs']);signed=read(assessment['signed_metadata'])
    producer=read(assessment['producer_actual_exit']);recorder=read(assessment['recorder_actual_exit']);authority=read(assessment['trust_authority'])
    c.require(producer['actual_tool']==assessment['producer_actual_tool'] and producer['actual_tool']['exit_code']==0
              and producer['stage_success_observed'] is True and producer['complete_events_audit']['recomputed']['entire_discovery_reproduced'] is True
              and recorder['actual_tool']==assessment['recorder_actual_tool'] and recorder['actual_tool']['exit_code']==0
              and recorder['owner_gone'] is True and recorder['result']==assessment['producer_actual_exit'], 'Original completed signed-metadata producer and actual recorder')
    c.require(producer['evidence']['signed-metadata/sources-result.json']==assessment['signed_metadata'], 'Original actual producer binds this exact complete signed result')
    trust=signed['trust_anchor']
    c.require(authority['schema']=='dory_ubuntu_archive_key_authority_v1' and authority['independent_authority_verified'] is True
              and authority['keyring']['sha256']==trust['keyring_sha256'] and authority['primary_fingerprints']==trust['primary_fingerprints']
              and all(trust['authority_ref'][k]==assessment['trust_authority'][k]for k in ('path','sha256')), 'Original independently reviewed full trust anchor binding')
    for key in ('signed_metadata','producer_actual_exit','recorder_actual_exit','trust_authority'):
        c.require(n[key]==assessment[key],'Frozen Mini input ancestry unchanged')
    c.require(len(assessment['rows'])==3 and [r['suite']for r in assessment['rows']]==['jammy','jammy-updates','jammy-security']
              and sum(r['expected_uncompressed']['bytes']for r in assessment['rows'])==RAW_TOTAL, 'Exactly three old main indexes')
    metadata_refs={r['path']:r for r in signed['adapter']['metadata_files']}
    evidence={r['path']:r for r in producer['evidence'].values() if isinstance(r,dict) and 'path'in r}
    for row,chain in zip(assessment['rows'],n['signed_index_reuse']):
        c.require(row['suite']==chain['suite'] and row['original_logical_index_ref']==chain['current_uncompressed_index']
                  and row['release']==chain['release'] and row['status_text']==chain['status_text']
                  and row['saved_signature_proof']==chain['saved_signature_proof'], 'Exact unchanged logical old input and Release proof')
        releases=[r for r in signed['releases'] if r['release_sha256']==row['release']['sha256']]
        c.require(len(releases)==1,'One original signed suite Release');release=releases[0]
        indexes=[r for r in release['indexes'] if r['url']==chain['index_url']]
        c.require(len(indexes)==1 and indexes[0]['compressed']=={k:row['compressed'][k]for k in ('bytes','sha256')}
                  and indexes[0]['uncompressed']==row['expected_uncompressed'], 'Both old compressed and raw full hashes signed')
        proof=row['saved_signature_proof']
        c.require(proof==release['signature_proof'] and proof['verified'] is True and proof['process_returncode']==0
                  and proof['primary_fingerprint'] in authority['primary_fingerprints'] and proof['keyring_sha256']==trust['keyring_sha256']
                  and proof['status_reference']==row['status_text'] and metadata_refs[row['compressed']['path']]==evidence[row['compressed']['path']]==row['compressed'],
                  'Complete saved signature proof and original compressed file receipt')
    return assessment


def validate(plan,review,plan_ref):
    runtime.validate(plan,review,plan_ref)
    c.require(plan['work_kind']==review['work_kind']==WORK_KIND and plan['snapshot_assessment']==review['snapshot_assessment']==ASSESSMENT,
              'Exact independently reviewed immutable snapshot work')
    c.require(plan['archive_mode']=='snapshot' and plan['external_authorities'] is None
              and Path(plan['output_root']).name.startswith('dory-serial-inventory-minicpm5-sdk310-index-snapshot-'), 'Separate snapshot namespace; no native/archive task')
    old=c.load(OLD_MANIFEST)
    c.require(len(plan['source_sha256'])==100 and all(plan['source_sha256'].get(p)==h for p,h in old['source_sha256'].items())
              and all(str(c.Q/name) in plan['source_sha256']for name in NEW_NAMES), 'Original98 pins plus exactly two snapshot sources')
    c.require(plan['mini_scientific_constraints']==review['mini_scientific_constraints']==old['scientific_constraints'], 'Separate original Mini science constraints unchanged')
    c.require(plan['complete_current_serial_catalog']==CATALOG and len(c.load(CATALOG)['runs'])==124, 'Exact complete124-row prior history')
    retirement.validate(plan,review);return plan


def authenticate(guard):
    guard();a=metadata()
    import dory_ubuntu_package_sources_v4 as discovery
    for row in a['rows']:
        raw=read_verified(row['release'],guard,c.MIB);read_verified(row['status_text'],guard,4096)
        indexes,_=discovery.release_indexes(raw,row['suite'],dt.datetime.now(dt.timezone.utc))
        c.require(indexes['main/binary-amd64/Packages']==row['expected_uncompressed']
                  and indexes['main/binary-amd64/Packages.xz']=={k:row['compressed'][k]for k in ('bytes','sha256')},
                  'Actual authenticated Release text rebinds compressed and raw tuples')
    return dict(schema='minicpm5_sdk310_snapshot_inputs_authenticated_v1',work_kind=WORK_KIND,
                assessment=ASSESSMENT,original_inputs=a['frozen_mini_inputs'],indexes=3,raw_total_bytes=RAW_TOTAL,
                compressed_payloads_read=False,system_APT_indices_read=False)


def validate_admission(admission,inputs):
    c.require(admission['work_kind']==WORK_KIND and admission['metadata_input_binding']==inputs, 'Exact snapshot metadata admission binding')


def read_verified(ref,guard,cap):
    path=Path(ref['path']);c.require(path.is_absolute() and path.resolve()==path and type(ref['bytes'])is int and 0<ref['bytes']<=cap,'Bounded canonical pinned input')
    guard();fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd);c.require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size==ref['bytes'],'Exact complete ordinary input size')
        parts=[];size=0;digest=hashlib.sha256()
        while size<before.st_size:
            guard();part=os.read(fd,min(65536,before.st_size-size));c.require(part,'Complete input bytes');parts.append(part);size+=len(part);digest.update(part)
        c.require(os.read(fd,1)==b'' and c.identity(before)==c.identity(os.fstat(fd))==c.identity(path.lstat()) and path.resolve()==path,'Stable pinned input EOF')
        c.require(digest.hexdigest()==ref['sha256'],'Full input SHA before use');return b''.join(parts)
    finally:os.close(fd)


def decode_exact(compressed,row,decoder_factory,guard=lambda:None):
    expected=row['compressed'];raw=row['expected_uncompressed']
    c.require(type(compressed)is bytes and len(compressed)==expected['bytes'] and hashlib.sha256(compressed).hexdigest()==expected['sha256'],'Full original XZ SHA before decoder creation')
    guard();decoder=decoder_factory(MEMLIMIT)
    plain=decoder.decompress(compressed,max_length=raw['bytes']+1);guard()
    c.require(decoder.eof and len(plain)==raw['bytes'] and hashlib.sha256(plain).hexdigest()==raw['sha256'],'Complete exact old signed raw bytes and full SHA')
    tail=decoder.unused_data
    if tail:
        c.require(tail==EMPTY_TAIL,'Only original exact32-byte empty trailing XZ stream')
        guard();second=decoder_factory(MEMLIMIT);empty=second.decompress(tail,max_length=1);guard()
        c.require(empty==b'' and second.eof and second.unused_data==b'','Exact empty second stream with no third stream')
    return plain


def save_plain(path,plain,expected,guard):
    c.require(len(plain)==expected['bytes'] and hashlib.sha256(plain).hexdigest()==expected['sha256'],'Full signed bytes before snapshot write')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        offset=0
        while offset<len(plain):
            guard();count=os.write(fd,plain[offset:offset+65536]);c.require(count>0,'Complete exclusive snapshot write');offset+=count
        os.fchmod(fd,0o400);os.fsync(fd)
    finally:os.close(fd)
    directory=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(directory)
    finally:os.close(directory)
    ref=dict(path=str(path),bytes=expected['bytes'],sha256=expected['sha256'])
    read_verified(ref,guard,64*c.MIB);return ref


def run_snapshot(plan,guard,emit,decoder_factory=None):
    a=metadata();out=Path(plan['output_root']);directory=out/'indexes'
    guard();directory.mkdir(mode=0o700)
    parent_fd=os.open(out,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(parent_fd)
    finally:os.close(parent_fd)
    if decoder_factory is None:
        import lzma
        decoder_factory=lambda limit:lzma.LZMADecompressor(format=lzma.FORMAT_XZ,memlimit=limit)
    rows=[]
    for row in a['rows']:
        compressed=read_verified(row['compressed'],guard,32*c.MIB)
        emit(dict(event='snapshot_xz_verified',suite=row['suite'],compressed_ref=row['compressed']))
        plain=decode_exact(compressed,row,decoder_factory,guard)
        snapshot_ref=save_plain(directory/row['proposed_snapshot_basename'],plain,row['expected_uncompressed'],guard)
        rows.append(dict(suite=row['suite'],logical_ref=row['original_logical_index_ref'],snapshot_ref=snapshot_ref,
                         compressed_ref=row['compressed'],release=row['release'],status_text=row['status_text'],saved_signature_proof=row['saved_signature_proof']))
        emit(dict(event='snapshot_plaintext_written',suite=row['suite'],snapshot_ref=snapshot_ref))
        del compressed,plain
        c.require(storage_usage(out)<=plan['budget']['output_bytes'],'Complete allocated snapshot output cap')
    result=dict(schema=SCHEMA,assessment=ASSESSMENT,original_inputs=a['frozen_mini_inputs'],signed_metadata=a['signed_metadata'],
                producer_actual_exit=a['producer_actual_exit'],recorder_actual_exit=a['recorder_actual_exit'],
                rows=rows,raw_total_bytes=RAW_TOTAL,complete=True,actual_process_exit_claimed=False,
                system_APT_files_changed=False,native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False)
    validate_snapshot_result(result,plan,a)
    ref=c.save(out/'snapshot-manifest.json',result)
    emit(dict(event='snapshot_complete',snapshot_manifest=ref,raw_total_bytes=RAW_TOTAL,indexes=3));guard();return ref


def validate_snapshot_result(value,plan,assessment=None):
    a=assessment if assessment is not None else metadata()
    c.require(value['schema']==SCHEMA and value['assessment']==ASSESSMENT and value['original_inputs']==a['frozen_mini_inputs']
              and all(value[k]==a[k]for k in ('signed_metadata','producer_actual_exit','recorder_actual_exit'))
              and value['raw_total_bytes']==RAW_TOTAL and value['complete'] is True,'Exact complete snapshot and original signed provenance')
    for key in ('actual_process_exit_claimed','system_APT_files_changed','native_ABI_qualified','full_model_qualified','new_speedup_claim'):
        c.require(value[key] is False,'No stage/model/system-state promotion')
    c.require(len(value['rows'])==3,'All three snapshots retained')
    for actual,want in zip(value['rows'],a['rows']):
        expected=dict(suite=want['suite'],logical_ref=want['original_logical_index_ref'],
            snapshot_ref=dict(path=str(Path(plan['output_root'])/'indexes'/want['proposed_snapshot_basename']),**want['expected_uncompressed']),
            compressed_ref=want['compressed'],release=want['release'],status_text=want['status_text'],saved_signature_proof=want['saved_signature_proof'])
        c.require(actual==expected,'Exact ordered unchanged logical ref to owned immutable plaintext snapshot')
    return value


def validate_events(events,value,manifest_ref):
    expected=[]
    for row in value['rows']:
        expected.extend([dict(event='snapshot_xz_verified',suite=row['suite'],compressed_ref=row['compressed_ref']),
                         dict(event='snapshot_plaintext_written',suite=row['suite'],snapshot_ref=row['snapshot_ref'])])
    expected.append(dict(event='snapshot_complete',snapshot_manifest=manifest_ref,raw_total_bytes=RAW_TOTAL,indexes=3))
    c.require(events==expected,'Exact complete seven-event snapshot order')


def storage_usage(out):
    pending=[Path(out)];total=count=0
    while pending:
        path=pending.pop();info=path.lstat();count+=1
        c.require(count<=2000 and not stat.S_ISLNK(info.st_mode) and (stat.S_ISREG(info.st_mode)or stat.S_ISDIR(info.st_mode)),'Original bounded regular output tree')
        total+=max(info.st_size,info.st_blocks*512)
        if stat.S_ISDIR(info.st_mode):pending.extend(path.iterdir())
    return total


def save_final_result(out,result,budget):
    ref=c.save(out/'result.json',result);c.require(storage_usage(out)<=budget['output_bytes'],'Final receipt included in output cap');return ref
