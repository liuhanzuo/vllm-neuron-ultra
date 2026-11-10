"""Lossless full layer1 cache records under the original output caps.

Pure bytes only: no Torch, NumPy, file access or admission. Every 4224-position
cache is reconstructed, including inactive slots and physical padding. Changed
64KiB pages are stored as absolute replacements; this is never tensor cropping.
"""
import hashlib
import math
import zlib

SHAPE = (4, 1, 4224, 128)
CACHE_BYTES = math.prod(SHAPE)*2
PAGE_BYTES = 65536
PAGES = (CACHE_BYTES+PAGE_BYTES-1)//PAGE_BYTES
MAX_PACKED_BYTES = CACHE_BYTES+PAGES*128
SCHEMA = 'minicpm5_layer1_cache_pages_v1'

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(raw): return hashlib.sha256(raw).hexdigest()

def validate_bytes(raw):
    require(type(raw) is bytes and len(raw)==CACHE_BYTES, 'Complete original BF16 cache bytes')

def encode(raw, previous=None):
    validate_bytes(raw)
    if previous is not None: validate_bytes(previous)
    pages=[];parts=[];offset=0
    for index in range(PAGES):
        start=index*PAGE_BYTES;piece=raw[start:start+PAGE_BYTES]
        if previous is not None and piece==previous[start:start+PAGE_BYTES]:continue
        packed=zlib.compress(piece,level=1)
        require(len(packed)<=len(piece)+128,'Bounded compressed page')
        pages.append(dict(index=index,offset=offset,packed_bytes=len(packed),
                          raw_bytes=len(piece),sha256=sha(piece)))
        parts.append(packed);offset+=len(packed)
    packed=b''.join(parts)
    require(len(packed)<=MAX_PACKED_BYTES,'Complete bounded encoded cache')
    return dict(schema=SCHEMA,shape=list(SHAPE),dtype='torch.bfloat16',
                encoding='absolute_changed_pages_zlib',page_bytes=PAGE_BYTES,
                raw_bytes=CACHE_BYTES,sha256=sha(raw),
                base_sha256=sha(previous) if previous is not None else None,
                packed_bytes=len(packed),packed_sha256=sha(packed),pages=pages),packed

def decode(record, packed, previous=None):
    require(type(record)is dict and set(record)=={
        'schema','shape','dtype','encoding','page_bytes','raw_bytes','sha256',
        'base_sha256','packed_bytes','packed_sha256','pages'},'Exact cache record fields')
    require(record['schema']==SCHEMA and record['shape']==list(SHAPE)
        and record['dtype']=='torch.bfloat16'
        and record['encoding']=='absolute_changed_pages_zlib'
        and record['page_bytes']==PAGE_BYTES and record['raw_bytes']==CACHE_BYTES,
        'Original whole-cache geometry and encoding')
    require(type(packed)is bytes and len(packed)<=MAX_PACKED_BYTES
        and record['packed_bytes']==len(packed) and sha(packed)==record['packed_sha256'],
        'Exact bounded encoded bytes')
    if record['base_sha256'] is None:
        require(previous is None,'First cache cannot inherit history')
        output=bytearray(CACHE_BYTES)
    else:
        validate_bytes(previous)
        require(sha(previous)==record['base_sha256'],'Exact prior full cache')
        output=bytearray(previous)
    pages=record['pages']
    require(type(pages)is list and len(pages)<=PAGES,'Bounded page table')
    last=-1;offset=0
    for item in pages:
        require(type(item)is dict and set(item)=={
            'index','offset','packed_bytes','raw_bytes','sha256'},'Exact page fields')
        index=item['index'];size=item['packed_bytes']
        require(type(index)is int and last<index<PAGES,'Strict ordered distinct pages')
        start=index*PAGE_BYTES;wanted=min(PAGE_BYTES,CACHE_BYTES-start)
        require(type(size)is int and 0<size<=wanted+128 and item['offset']==offset
            and item['raw_bytes']==wanted and offset+size<=len(packed),'Exact page boundaries')
        decoder=zlib.decompressobj()
        raw=decoder.decompress(packed[offset:offset+size],wanted+1)
        require(len(raw)==wanted and decoder.eof and not decoder.unused_data
            and not decoder.unconsumed_tail,'One complete bounded compressed page')
        require(sha(raw)==item['sha256'],'Original page hash')
        output[start:start+wanted]=raw
        last=index;offset+=size
    require(offset==len(packed),'No unaccounted encoded bytes')
    if previous is None:require([x['index']for x in pages]==list(range(PAGES)),'First record contains every page')
    raw=bytes(output)
    require(sha(raw)==record['sha256'],'Complete reconstructed cache hash')
    return raw
