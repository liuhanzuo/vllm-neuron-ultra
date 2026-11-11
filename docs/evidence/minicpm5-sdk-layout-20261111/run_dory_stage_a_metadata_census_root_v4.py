"""Root-only bounded metadata census after actual completed Mini SDK audit."""
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research';sys.path.insert(0,str(ROOT))
import minicpm5_layer1_sdk_batch_layout_cpu_runner_v4 as runner
from research import dory_stage_a_native_metadata_census_v2 as census
SOURCE_SHA='94f97e7019889bfd25ebe6330e86c1ddf342dac70717dd7b1fb2a298a43780b9'
MODEL_EXIT=Q/'minicpm5-layer1-sdk-batch-layout-r2-actual-exit-20261111-v1.json'
FINAL_EXIT=Q/'minicpm5-layer1-sdk-batch-layout-r2-audit-actual-exit-repaired-20261111-v1.json'
COMPLETE=Q/'minicpm5-sdk-completed-root-observation-20261111-v1.json'
PINS={str(MODEL_EXIT):'aa4c60e6eff2b0800eb470f1d3dbdb8180abfc1727da9c3393b861c6f0b26a43',
 str(FINAL_EXIT):'d7a46dd447c15a43c99035f43916225b3d50cd6aa83ba08ab90951e8e6a9d255',
 str(COMPLETE):'52374b5fb1800c5a83cad065e35c66a619108a168c4cf374730d55a4f3409990'}
OUT=Path('/mnt/nvme/dory-stage-a-metadata-census-r1-20261111')

class BoundedWriter:
    def __init__(self, stream, guard, cap=16*2**20):
        self.stream,self.guard,self.cap,self.count=stream,guard,cap,0
    def write(self, data):
        self.guard()
        runner.p.require(self.count+len(data)<=self.cap,'Complete compressed metadata cap')
        written=self.stream.write(data)
        runner.p.require(written==len(data),'Complete compressed metadata write')
        self.count+=written
        return written
    def flush(self):self.stream.flush()
    def tell(self):return self.stream.tell()


def main():
    require=runner.p.require
    require(sys.dont_write_bytecode and __debug__,'Original python -B')
    require(hashlib.sha256(Path(census.__file__).read_bytes()).hexdigest()==SOURCE_SHA,'Reviewed exact census source')
    records={path:runner.source_auth.load(dict(path=path,sha256=sha))for path,sha in PINS.items()}
    model,ended,completed=[records[str(p)]for p in (MODEL_EXIT,FINAL_EXIT,COMPLETE)]
    require(model['actual_exit_code']==ended['actual_exit_code']==0 and ended['complete_audit_metadata_independently_verified'] is True
        and ended['decoded_historical_unit_count']==1316 and completed['independent_raw_audit_complete'] is True,
        'Actual finished Mini model and independent complete audit')
    require(completed['audit_exit_recorder']==dict(session=42578,terminal_chunk='7c806d',exit_code=0),
        'External actual recorder terminal retained')
    require(not OUT.exists(),'Fresh metadata census namespace')
    runner.p.user_bus();fd,owner,lock=runner.post.external.acquire_lock()
    last=[0.0]
    def guard():
        runner.p.lock_proof(owner,lock);runner.p.clear_window()
        now=time.monotonic()
        if now-last[0]>=1:
            for old in [model['owner'],model['coordinator'],*model['ranks'],ended['owner']]:runner.p.gone(old)
            require(os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>=28*2**30,'Original root floor')
            require(os.statvfs('/mnt/nvme').f_bavail*os.statvfs('/mnt/nvme').f_frsize>=30*2**30,'Original NVMe floor')
            last[0]=now
    try:
        guard();runner.p.active_heavy();OUT.mkdir(mode=0o700)
        try:
            os.sched_setaffinity(0,{5});os.nice(19)
            resource.setrlimit(resource.RLIMIT_AS,(2*2**30,)*2)
            resource.setrlimit(resource.RLIMIT_CPU,(360,365))
            resource.setrlimit(resource.RLIMIT_FSIZE,(16*2**20,)*2)
            runner.p.once(OUT/'owner.json',dict(owner=owner,lock=lock,completed_mini=PINS,source_sha256=SOURCE_SHA))
            result=census.collect(guard=guard);guard()
            require(result['complete'] is True and result['native_or_model_payload_bytes_read']==0,'Metadata-only complete census')
            path=OUT/'complete-census.json.gz';count=0;digest=hashlib.sha256()
            with path.open('xb') as f:
                with gzip.GzipFile(fileobj=BoundedWriter(f,guard),mode='wb',mtime=0) as stream:
                    pending=bytearray()
                    for part in json.JSONEncoder(ensure_ascii=True,separators=(',',':'),allow_nan=False).iterencode(result):
                        raw=part.encode();count+=len(raw)
                        require(count<=256*2**20,'Bounded complete metadata serialization')
                        digest.update(raw)
                        for offset in range(0,len(raw),65536):
                            piece=raw[offset:offset+65536]
                            if len(pending)+len(piece)>65536:
                                guard();stream.write(pending);pending.clear()
                            pending.extend(piece)
                            if len(pending)==65536:
                                guard();stream.write(pending);pending.clear()
                    if pending:guard();stream.write(pending)
                f.flush();os.fsync(f.fileno())
            guard();readhash=hashlib.sha256();readcount=0
            with gzip.open(path,'rb') as f:
                while True:
                    guard();raw=f.read(1024*1024)
                    if not raw:break
                    readcount+=len(raw);require(readcount<=count,'Bounded complete metadata roundtrip');readhash.update(raw)
            require(readcount==count and readhash.digest()==digest.digest(),'Lossless complete metadata persisted')
            guard();runner.p.active_heavy()
            report=dict(schema='dory_stage_a_actual_metadata_census_completed_v1',complete=True,owner=owner,lock=lock,
                source_sha256=SOURCE_SHA,completed_mini=PINS,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                output=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    uncompressed_bytes=count,uncompressed_sha256=digest.hexdigest()),
                counts={k:result['summary'][k]for k in ('entry_kinds','observed_regular_files','observed_regular_bytes',
                    'inventory_files','inventory_bytes','by_scope','symlinks','hardlinked_regular_files','empty_files')},
                commands_returncodes=[x['returncode']for x in result['commands']],usage=result['usage'],
                actual_process_exit_claimed=False,native_or_model_payload_bytes_read=0,complete_authenticated_inventory=False,
                native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False)
            runner.p.once(OUT/'result.json',report);print(json.dumps(runner.reference(OUT/'result.json')))
        except BaseException as e:
            runner.p.once(OUT/'failure.json',dict(error_type=type(e).__name__,error=str(e),owner=owner,
                actual_process_exit_claimed=False,native_or_model_qualification_claimed=False));raise
    finally:runner.post.external.release_lock(fd)

if __name__=='__main__':main()
