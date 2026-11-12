"""One exact approved literal launch, durable parent wait; never retries."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path[:0]=[str(Q),str(R)]
from qwen35_sampler_runtime_io_v12 import ref,read,once,process_api
from research import minicpm5_sdk_mapping_nocache_launch_v3 as launch
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
assert sys.dont_write_bytecode and __debug__
prefix='minicpm5-formal-import-r5-20261112'
review_ref=ref(Q/(prefix+'-root-concrete-review.json'));review=read(review_ref)
assert review['reviewer']=='/root' and review['passed'] and review['blockers']==[] and review['catalog_rows']==155
prep=read(review['preparation']);assert latest.latest_handoff()['path']==prep['handoff']['path']
assert prep['catalog']==ref(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v106.json')
plan=read(review['outer_entry_plan']);literal=Path(review['entry_source']['path']).read_bytes()
assert len(literal)==37178 and hashlib.sha256(literal).hexdigest()==review['entry_source']['sha256']
argv_ref=ref(Q/(prefix+'-launch-argv.json'));argv=read(argv_ref)
assert argv==launch.launch(literal,review['entry_source'],plan,review['outer_entry_plan'])
process_api().active_heavy()
assert not (Q/'minicpm5-fixed-input-pv-sdk-import-formal-r5-20261112').exists()
env=dict(os.environ)
for name in ('LD_PRELOAD','LD_AUDIT','LD_LIBRARY_PATH'):env.pop(name,None)
with (Q/(prefix+'-tool.log')).open('xb',buffering=0) as log:
 child=subprocess.Popen(argv,cwd=str(R),env=env,stdout=log,stderr=subprocess.STDOUT)
 start_ref=once(Q/(prefix+'-launcher-start.json'),dict(schema='minicpm5_formal_root_launcher_start_v1',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=child.pid,launcher_pid=os.getpid(),root_review=review_ref,launch_argv=argv_ref,source=ref(Path(__file__).resolve()),removed_environment=['LD_PRELOAD','LD_AUDIT','LD_LIBRARY_PATH']))
 print(json.dumps(dict(event='minicpm5_formal_r5_literal_started',pid=child.pid,launcher_start=start_ref)),flush=True)
 code=child.wait()
 waited=once(Q/(prefix+'-launcher-wait.json'),dict(schema='minicpm5_formal_root_launcher_wait_v1',actual_returncode=code,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=child.pid,popen_wait_observed=True,self_exit_claimed=False))
 print(json.dumps(dict(actual_child_returncode=code,launcher_wait=waited)),flush=True)
sys.exit(code if code>=0 else 128-code)
