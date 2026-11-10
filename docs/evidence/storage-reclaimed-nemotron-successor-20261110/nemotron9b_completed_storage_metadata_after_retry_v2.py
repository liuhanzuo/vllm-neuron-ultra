"""Exact six text pins and two named maintenance JSON schemas.

For a future successor's metadata reader. No wildcard suffix approval and no
payload read. The fixed plan uses32MiB; its review and six texts retain8MiB,
matching the actual maintenance executor. Other rules remain with the reader.
"""
import hashlib
import json
from pathlib import Path
import re

import nemotron9b_completed_storage_metadata_after_retry_v1 as previous

Q = Path(__file__).resolve().parent
PLAN = Q/'completed-qwen-storage-after-retry-plan-20261110-v3.json'
REVIEW = Q/'completed-qwen-storage-after-retry-source-review-20261110-v3.json'
MODEL = Path('/mnt/nvme/models/Qwen3-8B')
LIMIT, ENVELOPE = 8*2**20, 32*2**20
EXACT_TEXT = {
    str(MODEL/'.cache/huggingface/download/model-00001-of-00005.safetensors.metadata'):
        'c34d74fb8c6654d9df6ad80defdfd3135c98d6f751e67976da6a3b6f51a083bf',
    str(MODEL/'.cache/huggingface/download/model-00002-of-00005.safetensors.metadata'):
        '2b912b535453199c1912743186849775d5c017a890199a10efcb3592b32c32f4',
    str(MODEL/'.cache/huggingface/download/model-00003-of-00005.safetensors.metadata'):
        'c266fcc0580ba782a1f588d711dc741805b0fb13c832cba9699264bd927a16b0',
    str(MODEL/'.cache/huggingface/download/model-00004-of-00005.safetensors.metadata'):
        'c962ac2fe6827f223541d51d3da6fdeecb9a2a1b73e9337e11733d1128e85f2d',
    str(MODEL/'.cache/huggingface/download/model-00005-of-00005.safetensors.metadata'):
        '27b9aabc6f8e88c45d5702e19c4acad5c75a56335ad00e9daa2b893b58cf52e2',
    '/home/ubuntu/serve_int8_cpu_bs4_fast.sh':
        '8f540ba7cbb3edfc6716a332319a56e178a57ee1fc39b29e0d34f7467b5ca358',
}


def require(ok,message):
    if not ok:raise RuntimeError(message)


def accepts(path):return previous.accepts(path) or Path(path) in (PLAN,REVIEW)


def shape(path,value):
    if previous.accepts(path):return previous.shape(path,value)
    require(isinstance(value,dict) and isinstance(value.get('source_sha256'),dict)
            and 6 <= len(value['source_sha256']) <= 100000, 'Bounded explicit maintenance source map')
    pins=value['source_sha256']
    require(all(isinstance(k,str) and Path(k).is_absolute() and isinstance(v,str)
                and re.fullmatch('[0-9a-f]{64}',v) for k,v in pins.items()), 'Literal source map references')
    require(all(pins.get(k)==v for k,v in EXACT_TEXT.items()), 'Six exact preserved text pins in named envelopes')
    if Path(path)==PLAN:
        require(value['schema']=='completed_qwen_storage_after_retry_plan_v3'
                and isinstance(value['budget'],dict) and isinstance(value['completed_residual_review'],dict)
                and isinstance(value['weights'],list) and len(value['weights'])==5
                and isinstance(value['journals'],list) and len(value['journals'])==8
                and value['runtime_admitted'] is False and value['scientific_constraints_changed'] is False,
                'Only exact five-weight/eight-journal maintenance plan envelope')
    elif Path(path)==REVIEW:
        require(value['schema']=='completed_qwen_storage_after_retry_source_review_v3'
                and value['reviewer']=='/root' and value['passed'] is True and value['blockers']==[]
                and value['plan']['path']==str(PLAN)
                and re.fullmatch('[0-9a-f]{64}',value['plan']['sha256']),
                'Only exact final maintenance source-review envelope')
    else:raise RuntimeError('No other maintenance metadata envelope')
    return True


def text(path,expected,stable_bytes):
    if previous.accepts(path):return previous.text(path,expected,stable_bytes)
    require(accepts(path) and isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected),
            'Exact path and literal hash required before opening')
    if str(path) in EXACT_TEXT:
        require(expected==EXACT_TEXT[str(path)], 'No rebinding original preserved text')
        cap=LIMIT
    else:cap=ENVELOPE if Path(path)==PLAN else LIMIT
    raw=stable_bytes(path,cap)
    require(len(raw)<=cap and hashlib.sha256(raw).hexdigest()==expected, 'Exact bounded maintenance text bytes')
    raw.decode('utf8')
    if str(path) not in EXACT_TEXT:shape(path,json.loads(raw))
    return raw
