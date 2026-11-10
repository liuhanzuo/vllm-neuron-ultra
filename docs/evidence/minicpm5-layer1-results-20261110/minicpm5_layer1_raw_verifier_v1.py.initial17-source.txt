"""Independent byte review of layer1 observations, without native imports or I/O.

The runtime must separately admit and authenticate metadata/payload reads before
passing bytes here. No code from the writer, codec or runtime validator is
imported. This checks saved bytes and history, not attention math or speed.
"""
import hashlib
import re
import zlib

ARMS=('strict_anchor_s0','strict_layer0_full_bf16_prefill_only',
      'strict_layer0_full_bf16_both_phases')
TAPS=('layers.1.input_layernorm','layers.1.self_attn','layers.1')
CACHES=('kv_mgr.past_key_values.2','kv_mgr.past_key_values.3')
SHAPE=[4,1,4224,128]
CACHE_BYTES=4*4224*128*2
PAGE_BYTES=65536
PAGES=66
MAX_PACKED=CACHE_BYTES+PAGES*128
MAX_PAYLOAD=3*4*2048*2+2*MAX_PACKED
ORIGINAL_FIELDS=('call','bucket','input_metadata','execution_input_metadata',
 'integer_widening','changed_mask_cells','output_metadata','stage_names',
 'stage_metadata','stage_sha256','before_metadata','input_after_metadata',
 'post_metadata','route_records','physical_rows','HF_metrics','preservation',
 'candidate_layer0_full_bf16_enabled','candidate_targets','oproj_dtype_records')

def require(value,message):
    if not value:raise ValueError(message)

def sha(data):return hashlib.sha256(data).hexdigest()

def hash_string(value):
    return type(value)is str and re.fullmatch('[0-9a-f]{64}',value)is not None

def states(rows):
    require(type(rows)is list and len(rows)==84,'All84 state records')
    out=[]
    for index,row in enumerate(rows):
        require(row['name']=='kv_mgr.past_key_values.'+str(index)
            and row['shape']==SHAPE and row['dtype']=='torch.bfloat16'
            and hash_string(row['sha256']),'Original complete state geometry')
        out.append({k:row[k]for k in ('name','shape','dtype','sha256')})
    return out

def unpack_cache(entry,data,source,base):
    """Decode page records independently of the writer implementation."""
    header=entry['codec']
    require(set(header)=={'schema','shape','dtype','encoding','page_bytes','raw_bytes',
        'sha256','base_sha256','packed_bytes','packed_sha256','pages'},'Exact cache header')
    require(header['schema']=='minicpm5_layer1_cache_pages_v1'
        and header['shape']==SHAPE and header['dtype']=='torch.bfloat16'
        and header['encoding']=='absolute_changed_pages_zlib'
        and type(header['page_bytes'])is int and header['page_bytes']==PAGE_BYTES
        and type(header['raw_bytes'])is int and header['raw_bytes']==CACHE_BYTES,
        'Entire original BF16 cache')
    require(entry['source_metadata']==source and entry['name']==source['name']
        and source['shape']==SHAPE and source['dtype']=='torch.bfloat16'
        and header['sha256']==source['sha256'],'Reconstruction binds original input cache')
    require(type(data)is bytes and len(data)<=MAX_PACKED
        and type(header['packed_bytes'])is int and header['packed_bytes']==len(data)
        and header['packed_sha256']==sha(data),'Exact encoded cache bytes')
    if base is None:
        require(header['base_sha256']is None,'First input cache cannot inherit history')
        output=bytearray(CACHE_BYTES)
    else:
        require(type(base)is bytes and len(base)==CACHE_BYTES
            and header['base_sha256']==sha(base),'Previous captured input cache bytes')
        output=bytearray(base)
    pages=header['pages']
    require(type(pages)is list and len(pages)<=PAGES,'Finite original page count')
    indices=[];offset=0
    for page in pages:
        require(type(page)is dict and set(page)=={'index','offset','packed_bytes','raw_bytes','sha256'},
                'Exact page fields')
        index=page['index'];size=page['packed_bytes']
        require(type(index)is int and 0<=index<PAGES
            and (not indices or index>indices[-1]),'Strict distinct physical page order')
        expected=min(PAGE_BYTES,CACHE_BYTES-index*PAGE_BYTES)
        require(type(size)is int and 0<size<=expected+128
            and type(page['offset'])is int and page['offset']==offset
            and type(page['raw_bytes'])is int and page['raw_bytes']==expected
            and offset+size<=len(data),'Exact bounded compressed page')
        decoder=zlib.decompressobj()
        piece=decoder.decompress(data[offset:offset+size],expected+1)
        require(len(piece)==expected and decoder.eof and not decoder.unused_data
            and not decoder.unconsumed_tail,'Complete single compressed page stream')
        require(sha(piece)==page['sha256'],'Every original page digest')
        start=index*PAGE_BYTES
        output[start:start+expected]=piece
        offset+=size;indices.append(index)
    require(offset==len(data),'Every encoded byte consumed')
    if base is None:require(indices==list(range(PAGES)),'First record contains every physical page')
    raw=bytes(output)
    require(len(raw)==CACHE_BYTES and sha(raw)==source['sha256'],
            'Every restored cache byte agrees with original input')
    return raw

def verify_call(record,original,original_ref,payload,previous=None):
    require(type(record['rank'])is int and record['rank']in (0,1)
        and record['rank']==original['rank'] and record['arm']in ARMS
        and record['arm']==original['arm'],'Original rank and arm')
    for field in ORIGINAL_FIELDS:
        require(record[field]==original[field],'Original repeat changed '+field)
    require(record['full_model_qualified']is record['new_speedup_claim']is False,
            'Observation cannot qualify the model')
    before=states(record['before_metadata']);after=states(record['post_metadata'])
    require(record['input_after_metadata']==record['before_metadata'],'Original input states unchanged')
    ci=record['call']['index'];rank=record['rank'];arm=record['arm']
    require(type(ci)is int and 0<=ci<14,'Original call index')
    if ci==0:require(previous is None,'No history before first call')
    else:
        require(type(previous)is dict
            and [previous['rank'],previous['arm'],previous['call']]==[rank,arm,ci-1],
            'Complete same-arm/rank call chain')
        require(before==previous['post_metadata'],'Own history preserves previous returned states')
    rows=sorted(record['physical_rows'],key=lambda row:row['physical_row'])
    require([x['physical_row']for x in rows]==list(range(len(rows))) and len(rows)in (1,4),
            'Every physical prediction row')
    order=[row['frame']['index']for row in rows]
    require(len(set(order))==len(order) and all(type(x)is int and 0<=x<32 for x in order),
            'Finite distinct original prediction frames')
    require(len(record['stage_names'])==213,'All original stage records retained')
    descriptor=record['layer1_observation']
    require(descriptor['schema']=='minicpm5_layer1_observation_v1'
        and [descriptor[k]for k in ('rank','arm','call')]==[rank,arm,ci]
        and descriptor['original_call']==original_ref
        and descriptor['physical_frame_order']==order,'Exact original observation identity')
    require(all(descriptor[k]is True for k in ('all_original_fields_exact',
        'observation_occurs_after_forward','full_cache_reconstruction_required',
        'caches_are_original_input_not_post_state'))
        and descriptor['full_model_qualified']is descriptor['new_speedup_claim']is False,
        'Original observation contract')
    require(type(payload)is bytes and 0<len(payload)<=MAX_PAYLOAD<16*2**20
        and type(descriptor['payload_bytes'])is int and len(payload)==descriptor['payload_bytes']
        and sha(payload)==descriptor['payload_sha256'],'Exact whole observation payload')
    entries=descriptor['entries']
    require(type(entries)is list and [e['name']for e in entries]==list(TAPS+CACHES),
            'All3 prediction taps and2 complete input caches')
    offset=0;taps={};caches={}
    for i,entry in enumerate(entries):
        size=entry['bytes']
        require(type(size)is int and size>=0 and type(entry['offset'])is int
            and entry['offset']==offset and offset+size<=len(payload),'Contiguous bounded payload spans')
        data=payload[offset:offset+size];offset+=size;name=entry['name']
        if i<3:
            require(entry['kind']=='tap' and entry['shape']==[len(order),1,2048]
                and entry['dtype']=='torch.bfloat16' and entry['frame_indices']==order
                and size==len(order)*4096
                and entry['source_metadata']==record['stage_metadata'][name],
                'Exact original prediction tap geometry')
            require(sha(data)==entry['sha256']==record['stage_sha256'][name],
                    'Every saved tap byte agrees with original prediction')
            taps[name]=data
        else:
            require(entry['kind']=='cache','Original input cache record')
            base=None if previous is None else previous['caches'][name]
            caches[name]=unpack_cache(entry,data,record['before_metadata'][i-1],base)
    require(offset==len(payload),'No unreferenced payload bytes')
    state=dict(rank=rank,arm=arm,call=ci,caches=caches,post_metadata=after)
    report=dict(rank=rank,arm=arm,call=ci,physical_frame_order=order,payload_bytes=len(payload),
        payload_sha256=sha(payload),tap_sha256={k:sha(v)for k,v in taps.items()},
        cache_sha256={k:sha(v)for k,v in caches.items()},complete_cache_bytes=2*CACHE_BYTES)
    return report,state

class Trajectory:
    """Stream exactly 2 ranks x3 arms x14 calls, retaining one complete cache pair."""
    def __init__(self):
        self.reports=[];self.previous=None;self.frames=[];self.payload_bytes=[0,0]
        self.complete=False

    def accept(self,record,original,original_ref,payload):
        index=len(self.reports)
        require(not self.complete and index<84,'Finite complete trajectory')
        rank=index//42;arm=ARMS[(index%42)//14];ci=index%14
        require([record['rank'],record['arm'],record['call']['index']]==[rank,arm,ci],
                'Every call of both ranks and all3 arms in original order')
        previous=None if ci==0 else self.previous
        report,state=verify_call(record,original,original_ref,payload,previous)
        frames=[] if ci==0 else self.frames
        frames=frames+report['physical_frame_order']
        require(len(frames)==len(set(frames)),'No repeated physical frame in one arm')
        if ci==13:require(sorted(frames)==list(range(32)),'Every original frame in completed arm')
        total=self.payload_bytes[rank]+len(payload)
        require(total<=128*2**20,'Original per-rank payload cap')
        self.reports.append(report);self.previous=state;self.frames=frames
        self.payload_bytes[rank]=total
        return report

    def finish(self):
        require(not self.complete and len(self.reports)==84,'All84 calls must be reconstructed')
        self.complete=True;self.previous=None
        return dict(schema='minicpm5_layer1_independent_byte_verification_v1',complete=True,
            ranks=2,arms=3,calls=84,frames_per_arm=32,full_cache_records=168,
            complete_cache_bytes=168*CACHE_BYTES,payload_bytes_by_rank=self.payload_bytes,
            every_tap_and_full_cache_hash_exact=True,all_own_history_links_exact=True,
            raw_records=self.reports,independent_from_writer=True,model_reexecuted=False,
            attention_causality_proven=False,full_model_qualified=False,new_speedup_claim=False)
