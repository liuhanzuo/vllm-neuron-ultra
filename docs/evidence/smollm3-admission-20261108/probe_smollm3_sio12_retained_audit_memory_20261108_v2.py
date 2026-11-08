"""Bounded retained-data audit-memory probe; repeated fixtures are not new runs."""
import copy,datetime,hashlib,json,os,resource,sys,time
from pathlib import Path
MIB=2**20
resource.setrlimit(resource.RLIMIT_AS,(512*MIB,512*MIB));resource.setrlimit(resource.RLIMIT_FSIZE,(16*MIB,16*MIB))
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path.insert(0,str(R))
import smollm3_observer_invariance_audit_v21 as audit
import smollm3_observer_invariance_lifecycle_v20 as life
import smollm3_observer_invariance_protocol_v16 as p
import smollm3_observer_metadata_codec_v2 as codec
import historical_b3_payload_partition_v1 as history
from research.smollm3_sio12_prospective_budget_v2 import merge,transform,EXTRAS
OUT=Q/'smollm3-sio12-retained-audit-memory-probe-20261108-v2.json'
CAT=Q/'nemotron9b-package-r3-maintenance-20261108-complete-catalog.json'
def ref(f):return dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
def memory():
 status=dict(line.split(':',1)for line in Path('/proc/self/status').read_text().splitlines()if ':'in line)
 return dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,vmsize_bytes=int(status['VmSize'].split()[0])*1024,rss_bytes=int(status['VmRSS'].split()[0])*1024)
def main():
 assert not OUT.exists()and not p.DEST.exists();samples=[];phase='initial';error=None;verified=[];start=time.monotonic()
 try:
  assert ref(CAT)['sha256']=='d0572d983c400dc87713858ac1522d83da33366f8ed7fd7b1afb63c44f98da8f'
  catalog=json.loads(CAT.read_text());plan=json.loads((Q/'smollm3-observer-invariance-source-template-20261108-v17.json').read_text());handoff=json.loads((Q/'smollm3-sio12-metadata-source-handoff-20261108-v1.json').read_text());base=p.load_base()
  partition=history.partition(catalog);pins={}
  for values in [partition['source_sha256'],partition['provenance_sha256'],handoff['source_sha256'],plan['source_sha256']]:merge(pins,values)
  for f in [CAT,*[Q/n for n in EXTRAS],Path(__file__)]:item=ref(f);merge(pins,{item['path']:item['sha256']})
  stats=life.source_closure(plan,pins,handoff['original_metadata_symlink_resolutions'],hash_bytes=False)
  # Match distinct JSON-reader objects retained by execute: spec, review,
  # admission/source maps plus a second final-source stat map. No future owner
  # or successful result is invented and no execution spec is emitted.
  spec_pins=json.loads(json.dumps(pins));review_pins=json.loads(json.dumps(pins))
  admission_stats=json.loads(json.dumps(stats));extra_future_binding_allowance=bytearray(16*MIB)
  samples.append(dict(phase='current_history_and_distinct_source_bindings',**memory()))
  captures=[];proofs=[];exports=[];summaries=[]
  original_dest=p.DEST;p.DEST=R/'sio10'
  try:
   for index in range(12):
    phase='fixture_slot_'+str(index);fixture='r00'if index%4 in (0,3)else 'r01';root=p.DEST/fixture
    capture=json.loads((root/'capture-result.json').read_text());audit.raw_capture(base,fixture,capture)
    proof=json.loads((root/'root-mode-lifecycle.json').read_text());captures.append(capture);proofs.append(proof)
    if fixture!='r00':
     logs=[];engine=[]
     for producer in proof['authenticated_processes']:
      path=root/('observer-'+str(producer['pid'])+'.jsonl');rows,digest=audit.read_observer(path,producer,capture['role'])
      logs.append(dict(path=str(path),sha256=digest,rows=len(rows)))
      if any(row['kind']=='engine_begin'for row in rows):engine.append((path,rows,digest,producer))
     assert len(engine)==1;path,rows,digest,producer=engine[0]
     exported=audit.exporter.export(rows,audit.worker.export_capture_view(base,fixture,capture),retain_decoded=False)
     for key in ('frames','engine_steps'):
      values=exported.pop(key);exported[key+'_chunks']=audit.chunks(root,key,values,False);exported[key+'_count']=len(values)
     saved=json.loads((root/'replay-export.json').read_text())
     for key in ('run','capture_sha256','observer_file','observer_sha256','observer_producer','all_observer_logs'):exported[key]=saved[key]
     assert exported==saved,'Actual exporter/chunk output changed'
     exports.append(exported);summaries.append(audit.analysis.on_history_summary(exported))
     del rows,engine,values,saved
    samples.append(dict(phase=phase,fixture=fixture,**memory()));verified.append(dict(slot=index,fixture=fixture,original_capture=ref(root/'capture-result.json'),original_proof=ref(root/'root-mode-lifecycle.json'),raw_wire_reparse=True,export_bytes_recomputed=fixture!='r00'))
  finally:p.DEST=original_dest
  phase='final_source_stats_and_catalog_encoding'
  final_stats=life.source_closure(plan,pins,handoff['original_metadata_symlink_resolutions'],hash_bytes=False)
  successor=copy.deepcopy(catalog);merge(successor['source_sha256'],pins)
  wire,blobs=codec.pack(successor);decoded=codec.unpack(wire,blobs.__getitem__);assert decoded==successor
  # Exercise remaining source-map transports while all12captures,6exports,
  # graph-history summaries and actual completed-history objects remain alive.
  for key,value in [('source-admission',dict(source_stats=stats,source_pins_verified=True)),('final-identity-rehash',dict(source_stats=final_stats,full_source_rehash=True))]:
   w,b=codec.pack(value);assert codec.unpack(w,b.__getitem__)==value
  samples.append(dict(phase=phase,**memory()))
  assert len(captures)==12 and len(proofs)==12 and len(exports)==6 and len(summaries)==6
  assert not {'torch','numpy','transformers','vllm','neuronxcc'}&set(sys.modules)
 except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc)[:4096],phase=phase)
 report=dict(schema='smollm3_retained_data_audit_memory_probe_v1',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_only=True,source=ref(Path(__file__)),catalog=ref(CAT),actual_owner=life.process(os.getpid()),passed=error is None,error=error,samples=samples,fixture_slots=verified,source_entries=len(pins)if'pins'in locals()else None,address_space_limit_bytes=512*MIB,extra_future_binding_allowance_bytes=16*MIB,elapsed_seconds=time.monotonic()-start,final_memory=memory(),new_model_runs=0,only_two_original_completed_fixture_runs=True,twelve_fixture_slots_not_twelve_actual_runs=True,actual_raw_capture_reader=True,actual_observer_reader=True,actual_exporter_and_chunk_comparison=True,complete_physical_lifecycle_audit_executed=False,final_payload_hashing_not_executed=True,source_bytes_rehashed=False,remaining_runtime_checks='Actual final spec/review and distinct future run outputs must still satisfy unchanged readers, monitors and terminal audit. This empirical memory test is not a maximum-size proof or scientific qualification.',runtime_namespace_created=False,runtime_admitted=False,new_speedup_claim=False)
 with OUT.open('x')as out:json.dump(report,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
 print(json.dumps(dict(report=ref(OUT),passed=report['passed'],error=error,peak_rss=report['final_memory']['peak_rss_bytes'],vmsize=report['final_memory']['vmsize_bytes'],seconds=report['elapsed_seconds'])))
 if error is not None:raise SystemExit(1)
if __name__=='__main__':main()
