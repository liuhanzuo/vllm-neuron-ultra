"""Source-only complete-category forecast; never grants runtime admission.

Re-run against the actual final predecessor catalog. Existing completed JSON
shapes are measured; future output allowances are estimates, not new limits.
No model/raw/observer bytes are read and no runtime namespace is created.
"""
import argparse,copy,datetime,hashlib,json,math,os,resource,sys
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R))
import smollm3_observer_invariance_protocol_v17 as p
import smollm3_observer_invariance_lifecycle_v21 as life
import smollm3_observer_metadata_codec_v2 as codec
import historical_b3_payload_partition_v1 as history
MIB=2**20;GIB=2**30
TEMPLATE=Q/'smollm3-observer-invariance-source-template-20261108-v18.json'
HANDOFF=Q/'smollm3-sio12-metadata-source-handoff-20261108-v2.json'
EXTRAS=['finalize_smollm3_sio12_root_20261108_v2.py','smollm3_sio12_catalog_transport_review_v2.py',
        'check_smollm3_sio12_finalizer_source_20261108_v2.py','check_smollm3_sio12_catalog_transport_review_v3.py',
        'smollm3-sio12-finalizer-source-checks-20261108-v2.json']

def reference(path):
 raw=p.bounded_source(Path(path));return dict(path=str(path),sha256=codec.sha(raw)),raw

def allocation(size):
 assert type(size)is int and size>=0
 return ((size+4095)//4096)*4096

def merge(target,values):
 for name,digest in values.items():
  assert name not in target or target[name]==digest
  target[name]=digest

def transform(value,pins,stats,fingerprints):
 # Only metadata-map transport changes. Old owner/round data remain old fixture
 # values; nothing here is emitted as an execution specification or evidence.
 if isinstance(value,dict):
  result={}
  for key,item in value.items():
   if key=='source_stats':result[key]=stats
   elif key in ('runtime_source_sha256','root_runtime_source_sha256'):result[key]=pins
   elif key in ('source_sha256','new_source_and_evidence_pins'):
    combined=dict(item);merge(combined,fingerprints);result[key]=combined
   else:result[key]=transform(item,pins,stats,fingerprints)
  return result
 if isinstance(value,list):return [transform(x,pins,stats,fingerprints)for x in value]
 return value

def encoded_record(value):
 # Match metadata_store.record: records without nonempty source chunks retain
 # the original indented JSON bytes, not the compact codec envelope.
 wire,blobs=codec.pack(value)
 if blobs:
  assert codec.unpack(wire,blobs.__getitem__)==value
  kind='compact_source_record'
 else:
  wire=life.json_bytes(value);assert json.loads(wire)==value
  kind='original_indented_json'
 assert len(wire)<=codec.RECORD_CAP
 return wire,blobs,kind

def ledger(known_control,ordinary_parent_peak,export_peak,observer_peak,raw_peak,source_count,history_current):
 assert all(type(x)is int and x>=0 for x in (known_control,ordinary_parent_peak,export_peak,observer_peak,raw_peak))
 assert type(source_count)is int and source_count>0
 # These allowances forecast the original fixed successful workload; original
 # runtime caps and fail-and-retain behavior remain completely unchanged.
 common=dict(metadata=max(8*MIB,allocation(ordinary_parent_peak*3)),
             two_service_logs=32*MIB,
             raw_requests_responses=max(4*MIB,allocation(raw_peak*8)),
             directories_and_ipc=2*MIB,
             concurrent_pending_and_failure_bytes=16*MIB)
 on_extra=dict(active_engine_observer=max(64*MIB,allocation(observer_peak)),
               other_startup_observers=2*MIB,
               exported_graph_and_engine_records=max(64*MIB,allocation(export_peak*4)))
 control=dict(measured_shared_chunks_catalog_and_parents=known_control,
              supervisor_and_external_logs=32*MIB,
              future_terminal_and_catalog_growth=4*MIB,
              control_directories=2*MIB,
              interrupted_publication=8*MIB,
              original_terminal_reserve=8*MIB)
 rounds=[]
 for row in p.matrix():
  components=dict(common)
  if row['mode']=='on':components.update(on_extra)
  total=sum(components.values());rounds.append(dict(run_id=row['run_id'],role=row['role'],mode=row['mode'],requests=row['requests'],output_tokens=row['output_tokens'],components=components,forecast_allocated_bytes=total,original_run_cap_bytes=row['run_cap_bytes'],headroom_bytes=row['run_cap_bytes']-total,within_cap=total<=row['run_cap_bytes']))
 total=sum(control.values())+sum(x['forecast_allocated_bytes']for x in rounds)
 return dict(source_entries=source_count,history_current=history_current,rounds=rounds,control_components=control,control_forecast_allocated_bytes=sum(control.values()),original_control_cap_bytes=p.CONTROL_CAP,control_headroom_bytes=p.CONTROL_CAP-sum(control.values()),total_forecast_allocated_bytes=total,original_total_cap_bytes=p.OUTPUT_CAP,total_headroom_bytes=p.OUTPUT_CAP-total,outside_runtime_root_finalization_allowance_bytes=32*MIB,full_scientific_request_count=sum(x['requests']for x in rounds),full_scientific_output_tokens=sum(x['output_tokens']for x in rounds),forecast_fits_original_caps=all(x['within_cap']for x in rounds)and sum(control.values())<=p.CONTROL_CAP and total<=p.OUTPUT_CAP,
             category_coverage_complete=True,runtime_admitted=False,
             assumptions=['Metadata reserves permit three times the observed ordinary parent-record bytes (original process-count limit remains8).','Raw response allowance is an empirical forecast, at least8x measured allocated bytes and4MiB; it is not a replacement for the original4MiB PER-RESPONSE reader bound.','Only one request engine is permitted by the original audit. Its observer reserves its full64MiB file cap; startup-only observer totals use a2MiB forecast.','Export allowance is at least64MiB and4x measured complete on-run export allocation; logs reserve both full16MiB file caps.','Concurrent pending/failure allowance is16MiB per run. This forecasts the original fail-and-retain path; unexpected output still fails under the unchanged inclusive monitor.','Future owners/identities and source bindings are not fabricated; actual final history and fresh free-space/resource checks remain mandatory.'])

def forecast(catalog_path,catalog_sha,verify_source_bytes=False):
 assert sys.dont_write_bytecode and __debug__
 catalog_ref,raw=reference(catalog_path);assert catalog_ref['sha256']==catalog_sha
 catalog=json.loads(raw);del raw
 template_ref,raw=reference(TEMPLATE);template=json.loads(raw);del raw
 handoff_ref,raw=reference(HANDOFF);handoff=json.loads(raw);del raw
 assert handoff['source_frozen']is True and handoff['source_template']['sha256']==template_ref['sha256']
 partition=history.partition(catalog);pins={};merge(pins,partition['source_sha256']);merge(pins,partition['provenance_sha256']);merge(pins,handoff['source_sha256']);merge(pins,template['source_sha256'])
 for ref in [catalog_ref,template_ref,handoff_ref]+[reference(Q/name)[0]for name in EXTRAS]+[reference(Path(__file__).resolve())[0]]:merge(pins,{ref['path']:ref['sha256']})
 stats=life.source_closure(template,pins,handoff['original_metadata_symlink_resolutions'],hash_bytes=verify_source_bytes)
 assert set(stats)==set(pins)
 fingerprints=dict(pins);merge(fingerprints,partition['historical_payload_sha256'])
 chunks={'source-maps':{},'catalog-maps':{}};known=defaultdict(int);export=defaultdict(int);ordinary=defaultdict(int);refs=[];raw_stats=defaultdict(int);observer_stats=defaultdict(int);other_stats=defaultdict(int)
 files=sorted((R/'sio10').rglob('*.json'));assert len(files)==433
 for path in files:
  relative=path.relative_to(R/'sio10');bucket=relative.parts[0]
  ref,raw=reference(path);value=json.loads(raw);del raw
  candidate=transform(value,pins,stats,fingerprints)
  wire,blobs,wire_kind=encoded_record(candidate)
  namespace='catalog-maps'if relative.as_posix()=='control/successor-catalog.json'else'source-maps'
  for digest,payload in blobs.items():
   assert digest not in chunks[namespace]or chunks[namespace][digest]==len(payload)
   chunks[namespace][digest]=len(payload)
  size=allocation(len(wire));known[bucket]+=size
  is_export=path.name.startswith(('frames-','engine_steps-'))or path.name=='replay-export.json'
  (export if is_export else ordinary)[bucket]+=size
  refs.append(dict(source=ref,projected_parent_bytes=len(wire),projected_parent_allocated_bytes=size,chunk_namespace=namespace,chunk_count=len(blobs),wire_kind=wire_kind,export=is_export))
  del value,candidate,wire,blobs
 # Measure the complete current history too; use the separate catalog namespace.
 wire,blobs=codec.pack(catalog);assert codec.unpack(wire,blobs.__getitem__)==catalog
 for digest,payload in blobs.items():chunks['catalog-maps'][digest]=len(payload)
 current_catalog_parent=allocation(len(wire));del wire,blobs
 # Stat only: no raw HTTP/SSE, log, observer, model or tokenizer bytes opened.
 for path in (R/'sio10').rglob('*'):
  if not path.is_file()or path.suffix=='.json':continue
  v=path.stat();bucket=path.relative_to(R/'sio10').parts[0];amount=max(v.st_size,v.st_blocks*512)
  if path.suffix=='.raw':raw_stats[bucket]+=amount
  elif path.suffix=='.jsonl':observer_stats[bucket]+=amount
  else:other_stats[bucket]+=amount
 # Existing drain fixture covers530 rows. Include a proportional allowance
 # for every additional actually completed service, without inventing states.
 prior_drain=json.loads((R/'sio10/control/predecessor-drain.json').read_text())
 old_count=prior_drain['count'];assert old_count==len(prior_drain['observations'])==530
 expanded_drain=max(0,math.ceil((len(catalog['rows'])-old_count)/old_count*(R/'sio10/control/predecessor-drain.json').stat().st_size))
 chunk_allocations={name:sum(allocation(size)for size in sizes.values())for name,sizes in chunks.items()}
 control=known['control']+current_catalog_parent+sum(chunk_allocations.values())+allocation(expanded_drain)
 forecast_value=ledger(control,max(ordinary.get(k,0)for k in ('r00','r01','r02')),max(export.get(k,0)for k in ('r00','r01','r02')),max(observer_stats.values()),max(raw_stats.values()),len(pins),catalog_ref)
 v=os.statvfs(R);root_free=v.f_bavail*v.f_frsize;v=os.statvfs('/mnt/nvme');nvme_free=v.f_bavail*v.f_frsize
 return dict(schema='smollm3_sio12_complete_category_budget_forecast_v2',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_only=True,source=reference(Path(__file__).resolve())[0],source_template=template_ref,source_handoff=handoff_ref,current_completed_catalog=catalog_ref,catalog_counts=dict(services=len(catalog['rows']),processes=len(catalog['processes'])),actual_future_terminal_catalog_not_guessed=True,full_source_bytes_rehashed=verify_source_bytes,source_entry_count=len(pins),source_stats_sha256=p.digest(stats),historical_payload_fingerprints=len(partition['historical_payload_sha256']),input_json_records=len(files),every_transformed_json_roundtrip_equal=True,ordinary_records_use_original_indented_wire=True,projection_records=refs,chunk_namespace_counts={name:len(v)for name,v in chunks.items()},chunk_namespace_allocated_bytes=chunk_allocations,projected_known_parent_json_allocated_bytes=dict(known),projected_export_parent_allocated_bytes=dict(export),projected_ordinary_parent_allocated_bytes=dict(ordinary),stat_only_raw_allocated_bytes=dict(raw_stats),stat_only_observer_allocated_bytes=dict(observer_stats),stat_only_other_allocated_bytes=dict(other_stats),current_catalog_parent_allocated_bytes=current_catalog_parent,additional_current_history_drain_allowance_bytes=allocation(expanded_drain),forecast=forecast_value,root_free_bytes=root_free,nvme_free_bytes=nvme_free,original_root_admission_floor_bytes=30*GIB,root_free_floor_met=root_free>=30*GIB,original_nvme_floor_bytes=5*GIB,nvme_floor_met=nvme_free>=5*GIB,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],model_or_native_started=False,raw_observer_log_model_or_tokenizer_payload_bytes_read=False,runtime_namespace_created=False,runtime_admitted=False,new_speedup_claim=False,remaining=['Rebind to actual final Nemotron r3 catalog after actual release/exit.','Recompute using final source/spec/review closure, and measure actual external audit memory path; this forecast is not end-to-end512MiB proof.','Recheck live disk floors and dispose only confirmed finished/reproducible unpinned data as authorized.','Root review all forecast assumptions before creating source acceptance and fresh runtime spec.'])

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--catalog',required=True);parser.add_argument('--catalog-sha256',required=True);parser.add_argument('--output',required=True);parser.add_argument('--verify-source-bytes',action='store_true');args=parser.parse_args()
 out=Path(args.output);assert out.parent==Q and not out.exists()
 resource.setrlimit(resource.RLIMIT_AS,(512*MIB,512*MIB));resource.setrlimit(resource.RLIMIT_FSIZE,(16*MIB,16*MIB))
 result=forecast(Path(args.catalog),args.catalog_sha256,args.verify_source_bytes)
 raw=life.json_bytes(result);assert len(raw)<16*MIB
 fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 print(json.dumps(dict(output=dict(path=str(out),sha256=codec.sha(raw)),forecast_fits=result['forecast']['forecast_fits_original_caps'],total_forecast_bytes=result['forecast']['total_forecast_allocated_bytes'],control_forecast_bytes=result['forecast']['control_forecast_allocated_bytes'],peak_rss_bytes=result['peak_rss_bytes'],root_floor_met=result['root_free_floor_met'],runtime_admitted=False)))
if __name__=='__main__':main()
