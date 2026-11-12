"""Root-owned actual r5 metadata preparation after allocation session1710 ended."""
import datetime,json,resource,signal,sys,uuid
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path[:0]=[str(Q),str(R)]
from qwen35_sampler_runtime_io_v12 import RetainedLock,process_api,ref,read,once
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
from research import prepare_minicpm5_formal_r5_metadata_root_v1 as prep
assert sys.dont_write_bytecode and sys.flags.isolated and __debug__
resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));resource.setrlimit(resource.RLIMIT_CPU,(600,605));signal.alarm(600)
prefix='minicpm5-formal-import-r5-20261112'
catalog_ref=ref(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v106.json');assert catalog_ref['sha256']=='f972c54ed0cc40f51d1615f95f8f0ab1b13a4f0fb16d95abcd81317f4d3d1d6d'
catalog=read(catalog_ref);handoff_ref=catalog['current_handoff'];handoff=read(handoff_ref);predecessor_ref=catalog['runs'][-1]['terminal'];ended=read(predecessor_ref)
assert len(catalog['runs'])==155 and ended['actual_tool']==dict(kind='tool_session',session_id=1710,terminal_chunk='4b463a',exit_code=0) and ended['owners_gone'] and ended['physically_drained']
assert handoff['running'] is False and handoff['live_tool_sessions']==handoff['active_heavy_units']==[] and handoff['active_serial'] is None
assert handoff['complete_current_serial_catalog_next']==catalog_ref['path']
review_ref=ref(Q/'minicpm5-completed-slot-catalog-root-source-review-20261112-v1.json');review=read(review_ref);assert review['source_accepted'] and review['blockers']==[]
delivery_ref=review['delivery'];assert delivery_ref['sha256']=='ba81c4d505dc817fc47fb1b527b4f484192dd9e5474fc5213754410531b443e2'
args=[]
for name,value in [('catalog',catalog_ref),('handoff',handoff_ref),('predecessor',predecessor_ref),('source-delivery',delivery_ref)]:args.extend(['--'+name,value['path'],'--'+name+'-sha256',value['sha256']])
args.extend(['--handoff-catalog-field','complete_current_serial_catalog_next','--artifact-prefix',str(Q/prefix),'--output-root',str(Q/'minicpm5-fixed-input-pv-sdk-import-formal-r5-20261112'),'--attempt-id',uuid.uuid4().hex,'--unit','trn1-mini-formal-import-r5-20261112'])
lock=RetainedLock();lock.acquire();complete=False
try:
 tip=latest.latest_handoff();assert tip['path']==handoff_ref['path'];process_api().active_heavy()
 for path,floor in [(R,28*2**30),(Path('/mnt/nvme'),30*2**30)]:
  import os
  st=os.statvfs(path);assert st.f_bavail*st.f_frsize>=floor
 print(json.dumps(dict(event='minicpm5_r5_metadata_preparation_started',owner=lock.owner,catalog=catalog_ref,catalog_rows=155,source_review=review_ref)),flush=True)
 assert prep.main(args)==0
 lock.check();process_api().active_heavy();assert latest.latest_handoff()==tip;complete=True
finally:
 lock.check();owner=lock.owner;lock.release(True)
 terminal=once(Q/(prefix+'-metadata-terminal.json'),dict(schema='minicpm5_formal_metadata_preparation_terminal_v1',owner=owner,complete=complete,catalog=catalog_ref,handoff=handoff_ref,source=ref(Path(__file__).resolve()),source_review=review_ref,original_lock_released=True,no_children_or_units_created=True,native_executed=False,actual_external_exit_pending=True,preparation_receipt=str(Q/(prefix+'-preparation-receipt.json')) if complete else None))
 print(json.dumps(dict(metadata_terminal=terminal,complete=complete)),flush=True)
