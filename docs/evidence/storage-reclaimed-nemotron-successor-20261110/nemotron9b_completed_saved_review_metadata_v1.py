"""One completed Dory r3 JSON envelope; exact receipt and hash, no payload reads."""
import hashlib
import json
from pathlib import Path
from research import nemotron9b_after_dory_saved_retry_v2 as residual

Q=Path(__file__).resolve().parent
EXIT=dict(path=str(Q/'dory-residual-saved-review-r3-actual-exit-20261110-v1.json'),
          sha256='90ff8fb033bcaa5de600a049c961b6699ef14bf2cc5d3560f3cf579487ac8cfa')
SAVED=dict(path=str(residual.OUT/'saved-residual-review.json'),
           sha256='946a5fc4b28e46e434e161b34d70f6a2076bf9b2b92358bd8ea68b1fed31c08a')
BYTES=16594498
require=residual.require

def accepts(path):
    return str(path)==SAVED['path']

def text(path,expected,stable_bytes):
    require(accepts(path) and expected==SAVED['sha256'],
            'Only exact completed r3 saved review envelope')
    raw_exit=stable_bytes(EXIT['path'],8*2**20)
    require(hashlib.sha256(raw_exit).hexdigest()==EXIT['sha256'],'Exact actual external completion receipt')
    receipt=json.loads(raw_exit)
    residual.external(receipt,'saved-review-retry')
    require(receipt['saved_review']==SAVED,'Completed receipt must bind exact saved review')
    raw=stable_bytes(path,32*2**20)
    require(len(raw)==BYTES and hashlib.sha256(raw).hexdigest()==expected,'Exact complete saved review bytes')
    residual.old.saved_scope(json.loads(raw))
    return raw
