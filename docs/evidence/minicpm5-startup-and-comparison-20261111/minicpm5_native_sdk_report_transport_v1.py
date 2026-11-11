"""Pure bounded lossless JSON transport for the reviewed native/SDK report.

Two existing pure codec definitions are copied verbatim, without importing their
historical modules. Original files/limits/format remain unchanged. No default IO.
"""
import base64
import hashlib
import json
import re
import zlib

FILE_CAP=2*2**20
REPORT_SCHEMA='minicpm5_layer1_native_sdk_saved_comparison_v3'


def require(ok,message):
    if not ok:raise RuntimeError(message)


def packed_physical(proof):
    raw=json.dumps(proof,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    require(len(raw)<=32*2**20,'Bounded complete historical physical observation')
    compressed=zlib.compress(raw,9)
    require(len(compressed)<2*2**20,'Compressed complete pair within original file bound')
    encoded=base64.b64encode(compressed).decode('ascii')
    require(len(encoded)<2*2**20,'Base64 complete pair within original file bound')
    require(zlib.decompress(compressed)==raw,'Lossless complete physical evidence retention')
    return dict(encoding='zlib-base64-json-v1',uncompressed_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
        compressed_bytes=len(compressed),data=encoded,all_samples_retained=True)


def unpack(proof):
    require(type(proof)is dict and set(proof)=={'encoding','uncompressed_bytes','sha256','compressed_bytes','data','all_samples_retained'}
        and proof['encoding']=='zlib-base64-json-v1' and proof['all_samples_retained']is True,
        'Exact complete physical evidence envelope')
    require(type(proof['uncompressed_bytes'])is int and 0<proof['uncompressed_bytes']<=32*2**20
        and type(proof['compressed_bytes'])is int and 0<proof['compressed_bytes']<2*2**20
        and type(proof['data'])is str and len(proof['data'])<2*2**20
        and type(proof['sha256'])is str and re.fullmatch('[a-f0-9]{64}',proof['sha256']),
        'Original bounded full physical metadata')
    encoded=base64.b64decode(proof['data'],validate=True)
    require(len(encoded)==proof['compressed_bytes'],'Exact stored compressed bytes')
    decoder=zlib.decompressobj();raw=decoder.decompress(encoded,proof['uncompressed_bytes']+1)
    require(len(raw)==proof['uncompressed_bytes']and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
        'One complete stream, exact original length, no trailing or excess data')
    require(hashlib.sha256(raw).hexdigest()==proof['sha256'],'Complete original physical JSON digest')
    return json.loads(raw)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def report_shape(value):
    """Transport completeness shape only; original caller audits remain mandatory."""
    require(type(value)is dict and value['schema']==REPORT_SCHEMA and value['complete']is True
        and value['frames']==24 and value['rank_frame_comparisons']==48
        and value['raw_files']==dict(native=936,sdk=624)and value['metadata_files']==72,
        'Only complete reviewed v3 comparison schema')
    sources=value['source_files'];require(type(sources)is list and len(sources)==1632
        and len({x['reference']['path']for x in sources})==1632
        and {kind:sum(x['kind']==kind for x in sources)for kind in ('native','sdk','metadata')}
            ==dict(native=936,sdk=624,metadata=72),'All1632 original complete source references retained')
    rows=value['comparisons'];require(type(rows)is list and len(rows)==48
        and {(x['rank'],x['call'],x['slot'])for x in rows}=={
            (r,c,s)for r in (0,1)for c in (4,5,6,11,12,13)for s in range(4)},
        'All original rank/call/physical-slot comparison rows retained')
    for row in rows:
        require(row['declared_comparison_scope_complete']is True
            and row['first_difference_is_causal_localization']is False
            and row['comparison_coverage']==[dict(boundary=x['boundary'],status=x['status'])for x in row['boundaries']]
            and row['unknown_boundaries']==[x['boundary']for x in row['boundaries']if x['status']=='unknown']
            and {'pv_prior','pv_active','o_proj_local_before_reduce'}<=set(row['unknown_boundaries'])
            and type(row['sdk_padding'])is list and type(row['authenticated_sdk_leaves_not_individually_cross_compared'])is list,
            'All boundary/unknown/padding/explicit coverage fields retained; no evidence projection')
    require(value['complete_source_tensors_read_before_selection']is value['caller_must_authenticate_audit_terminals']is True
        and type(value['original_sdk_model_exit_code'])is int and value['original_sdk_model_exit_code']==1,
        'Original complete source scope and failed SDK exit preserved')
    for key in ('arithmetic_reexecuted','actual_exits_validated_here','attention_causality_proven','full_model_qualified','new_speedup_claim'):
        require(value[key]is False,'Transport never promotes diagnostic '+key)
    return value


def unpack_report(envelope):
    value=unpack(envelope);report_shape(value)
    raw=canonical(value)
    require(len(raw)==envelope['uncompressed_bytes']and hashlib.sha256(raw).hexdigest()==envelope['sha256'],
        'Entire decoded report reserializes to original canonical length/SHA; no duplicate or noncanonical JSON loss')
    return value


def pack_report(value):
    report_shape(value);envelope=packed_physical(value)
    require(unpack_report(envelope)==value,'All original source references/boundaries/unknowns/coverage losslessly round trip')
    return envelope


def encode_report(value):
    envelope=pack_report(value);raw=(json.dumps(envelope,indent=2,allow_nan=False)+'\n').encode()
    require(len(raw)<FILE_CAP,'Entire serialized packed report including JSON overhead/newline below original2MiB')
    return raw


def decode_report(raw):
    require(type(raw)is bytes and 0<len(raw)<FILE_CAP,'Complete stored envelope bytes below original2MiB')
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,'No duplicate stored envelope keys');value[key]=item
        return value
    envelope=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda token:(_ for _ in ()).throw(RuntimeError('Nonfinite JSON envelope')))
    return unpack_report(envelope)
