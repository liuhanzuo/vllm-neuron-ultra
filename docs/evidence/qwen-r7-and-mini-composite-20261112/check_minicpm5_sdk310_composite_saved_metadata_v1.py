"""Actual completed JSON chains only; deliberately cannot read any payload."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research import minicpm5_sdk310_composite_origins_v1 as m
Q=ROOT/'research'
SUMMARY=dict(path=str(Q/'minicpm5-sdk310-eight-archive-completed-root-summary-20261112-v1.json'),
             sha256='6395f1f392dcd2e253fbbc9c91b2cc42f974ac1a4b3b1f2a5bfaa6946af3b8a3')


def raw_json(path):
    p=Path(path);assert p.suffix=='.json' and p.resolve()==p
    before=p.stat();assert 0<before.st_size<=8*2**20
    raw=p.read_bytes();after=p.stat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(
        after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
    assert len(raw)==before.st_size
    return raw


def reference(path):
    raw=raw_json(path)
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


class JSONOnly:
    def __init__(self):self.inputs={}
    def json(self,ref):
        raw=raw_json(ref['path']);assert hashlib.sha256(raw).hexdigest()==ref['sha256']
        if 'bytes' in ref:assert len(raw)==ref['bytes']
        self.inputs[ref['path']]=dict(ref)
        return json.loads(raw)
    def read(self,*args,**kwargs):raise AssertionError('Payload reads forbidden in this metadata check')


def check():
    reader=JSONOnly();summary=reader.json(SUMMARY)
    def bound(producer,replay,producer_recorder,replay_recorder):
        pe=reference(Q/producer);re=reference(Q/replay)
        return dict(producer_result=reader.json(pe)['result'],producer_exit=pe,
            producer_recorder_exit=reference(Q/producer_recorder),replay_result=reader.json(re)['result'],
            replay_exit=re,replay_recorder_exit=reference(Q/replay_recorder))
    old=bound('dory-six-deb-r3-actual-exit-20261111-v1.json','dory-six-deb-replay-r1-actual-exit-20261111-v1.json',
              'dory-six-deb-r3-recorder-actual-exit-20261111-v1.json','dory-six-deb-replay-r1-recorder-actual-exit-20261111-v1.json')
    new=bound('minicpm5-sdk310-deb-download-r4-actual-exit-20261112-v1.json','minicpm5-sdk310-deb-replay-r1-actual-exit-20261112-v1.json',
              'minicpm5-sdk310-deb-download-r4-recorder-actual-exit-20261112-v1.json','minicpm5-sdk310-deb-replay-r1-recorder-actual-exit-20261112-v1.json')
    for key,other in [('producer_exit','download_exit'),('replay_exit','replay_exit'),
                      ('producer_result','download_result'),('replay_result','replay_result'),
                      ('replay_recorder_exit','replay_recorder_exit')]:assert m.same(new[key],summary[other])
    assert m.same(old['replay_result'],m.OLD_REPLAY)
    pins={}
    for b in (old,new):
        for role in ('producer','replay'):
            plan=reader.json(reader.json(b[role+'_exit'])['plan'])
            for path,sha in plan['source_sha256'].items():
                assert path not in pins or pins[path]==sha
                pins[path]=sha
    old_result,_=m.bundle(reader,old,'dory_deb',pins)
    new_result,plans=m.bundle(reader,new,'minicpm5_sdk310_deb',pins)
    n=reader.json(m.NEXT);m.signed_metadata(reader,n);m.new_packages(new_result,n)
    snapshot=plans[0]['signed_index_snapshot'];assert snapshot==plans[1]['signed_index_snapshot']
    m.stage(reader,snapshot['manifest'],snapshot['stage_exit'],snapshot['recorder_exit'],
            'minicpm5_sdk310_snapshot',pins)
    return dict(schema='minicpm5_sdk310_composite_origin_real_metadata_check_v1',source_metadata_only=True,
        passed=True,old_libc6=old,new_eight=new,metadata_inputs=reader.inputs,stage_source_union=pins,
        original_checks=m.CONSTRAINTS,old_packages=len(old_result['packages']),new_packages=len(new_result['packages']),
        new_targets=new_result['target_count'],payload_reads=0,current_native_hashes_observed_here=False,
        full_snapshot_plaintext_replayed_here=False,full_source_map_reconstructed_here=False,runtime_admitted=False,
        full_model_qualified=False,new_speedup_claim=False)


if __name__=='__main__':
    result=check()
    if len(sys.argv)==2:
        with Path(sys.argv[1]).open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    elif len(sys.argv)!=1:raise SystemExit('Optional sole argument: new metadata report path')
    print(json.dumps({k:result[k] for k in ('passed','old_packages','new_packages','new_targets','payload_reads',
        'full_snapshot_plaintext_replayed_here','full_source_map_reconstructed_here')},sort_keys=True))
