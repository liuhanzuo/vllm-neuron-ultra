"""Three exact JSON envelopes from the actually failed Mini saved-data r1.

Keep its failure and original32MiB bound. Ordinary metadata stays8MiB.
No broad suffix/directory exception and no tensor or model reads.
"""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
Q=ROOT/'research'
OUT=ROOT/'minicpm5-saved-tap-analysis-20261110-r1'
FAILED=dict(path=str(Q/'minicpm5-saved-tap-r1-failed-actual-exit-20261110-v1.json'),sha256='376e5ec5febfba598452a81d61ad45215f2c0018809367fdbc96cac29ed038ac')
ROWS={
 str(OUT/'predecessor-history.json'):dict(sha256='4f4d55a97aefc652a397dd8ccc2a9872d708c0c4cd48585f1b8d1a2a56a81ef0',bytes=11986746,role='history'),
 str(OUT/'source-files.json'):dict(sha256='2b8b09c0d0e8cc27a189079b7ac981d38ab36d4fa4fdbc2b538e1389064f62f7',bytes=20546986,role='sources'),
 str(Q/'minicpm5-saved-tap-runtime-v3-20261110-r1-spec.json'):dict(sha256='818b73f0c7112f45134240fa2b8ab097a65f898db61c086b3d71ab3271a53dbd',bytes=12411650,role='spec')}
LIMIT=32*2**20

def require(ok,message):
    if not ok:raise RuntimeError(message)
def accepts(path):return str(path) in ROWS

def shape(role,value):
    require(type(value)is dict,'Original JSON object')
    if role=='history':
        require(value['schema']=='minicpm5_after_nemotron_package_history_v2'
                and len(value['rows'])==1202 and len(value['processes'])==396
                and value['nemotron_outcome']['producer_qualified'] is False
                and value['nemotron_outcome']['failed_release'] is True,'Original failed-r1 history')
    elif role=='sources':
        require(value['source_and_binary_files_authenticated'] is True and len(value['files'])==43016
                and value['total_bytes_read']==6721984131
                and value['native_runtime_identity_verified'] is value['model_forward_provenance_verified']
                is value['full_model_qualified'] is value['new_speedup_claim'] is False,'Source authentication not model qualification')
    else:
        require(role=='spec' and value['schema']=='minicpm5_saved_tap_runtime_v3'
                and value['output']==str(OUT) and value['budget']['single_file_bytes']==LIMIT
                and type(value['source_sha256'])is dict and 0<len(value['source_sha256'])<=100000
                and value['runtime_admitted'] is value['full_model_qualified'] is False,'Original failed-r1 scope and limit')

def text(path,expected,stable_bytes):
    require(accepts(path) and expected==ROWS[str(path)]['sha256'],'Exact failed Mini path/hash before open')
    proof_raw=stable_bytes(FAILED['path'],8*2**20)
    require(hashlib.sha256(proof_raw).hexdigest()==FAILED['sha256'],'Original failed-r1 receipt')
    proof=json.loads(proof_raw)
    require(proof['actual_external_exit_observed'] is True
            and (proof['session_id'],proof['exit_code'],proof['tool_chunk'])==(22560,1,'ebd522'),
            'Original failure remains failed')
    row=ROWS[str(path)]
    ref=proof['execution_spec']if row['role']=='spec'else proof['files'][Path(path).name]
    require(ref==dict(path=str(path),sha256=expected),'Original failed exit binds exact envelope')
    raw=stable_bytes(path,LIMIT)
    require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==expected,'Complete exact bounded envelope bytes')
    shape(row['role'],json.loads(raw.decode('utf8')))
    return raw
