"""Independent targeted lifecycle checks; all metadata is synthetic in memory."""
import ast,copy,hashlib,json,os,re,sys
from pathlib import Path
from unittest.mock import patch
Q=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
ROOT=Q.parent
blocked=[]
def audit(event,args):
    if event=='open':
        path=args[0]
        if isinstance(path,(str,bytes)):
            p=os.fsdecode(path);mode=args[1];flags=args[2]
            if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
                blocked.append((event,p));raise AssertionError('No filesystem writes: '+p)
            if p.startswith((str(ROOT),'/proc/','/sys/','/mnt/')) and not p.endswith(('.py','.pyc')):
                blocked.append((event,p));raise AssertionError('No actual metadata/payload/process read: '+p)
    if event.startswith(('subprocess.','socket.','fcntl.flock')):
        blocked.append((event,repr(args)));raise AssertionError('No actual runtime/lock/process/network: '+event)
    if event=='import' and args[0].split('.')[0] in {'torch','numpy','transformers','torch_neuronx','neuronxcc','vllm'}:
        blocked.append((event,args[0]));raise AssertionError('No native/model imports')
sys.addaudithook(audit)
sys.path[:0]=[str(Q),str(ROOT)]
import test_minicpm5_retained_replay_lifecycle_v1 as fixture
c=fixture.c;composer=fixture.composer;consumer=fixture.consumer
checks={}
def fail_closed(name,f):
    try:f()
    except (ValueError,KeyError,AssertionError):checks[name]=True
    else:raise AssertionError(name+' unexpectedly accepted')
def compose(f):return f.compose()

f=fixture.Fixture();out=compose(f);objects=out['objects'];new=objects[-2][1];idle=objects[-1][1]
assert new['runs'][:-2]==f.oldcat['runs']
prior={p:h for row in f.oldcat['runs']for p,h in row['source_sha256'].items()}
added=new['runs'][-2]['source_sha256']
assert not set(added)&set(prior) and not new['runs'][-1]['source_sha256']
assert dict(prior,**added)==out['source_delta']
checks['existing_catalog_sources_deduplicated_and_complete']=True
assert idle['complete_current_serial_catalog']==f.spec['current_serial_catalog']
assert idle['complete_current_serial_catalog_next']==out['catalog']['path']
checks['idle_explicit_old_catalog_and_successor_path']=True
for ref,value in objects:f.values[ref['path']]=value
assert f.freshness(out['catalog'],out['idle_handoff'])['complete']
checks['unchanged_original_freshness_consumer_accepts_new_catalog']=True
# Whole source-map reconstruction retains recovered-only pins beyond base plan.
f=fixture.Fixture();extra=str(Q/'synthetic-recovered-extra-independent.py');f.spec['source_sha256'][extra]='e'*64
f.resolved['plan']['source_sha256'][extra]='e'*64
out=compose(f)
assert extra not in out['source_delta']
assert out['objects'][0][1]['recovered_sources']>len(f.read(f.read(f.spec['history_overlay'])['base_plan'])['source_sha256'])
checks['recovered_extra_outside_base_not_misclassified_as_delta']=True
# A failed observed pair alone still records both declared units and no fake replay owner.
f=fixture.Fixture(failed=True);f.running=f.pair;f.running_ref=f.pair_ref
f.observation.update(running=f.pair_ref)
f.final.update(running=f.pair_ref)
f.terminal['phase_states']=f.terminal['phase_states'][:1]
f.observation['actual_outer_exit']=fixture.observer.reconcile(f.terminal,f.running,f.actual)
out=compose(f);row=out['objects'][-2][1]['runs'][-2]
assert [x['pid']for x in row['owners']]==[101,201]
assert row['units']==sorted(f.spec['units'].values()) and not out['retained_replay_passed']
checks['failed_pair_prefix_retains_declared_replay_unit_without_invented_owner']=True
# Failed replay install has no observed owner or exit: preserve, not fabricate.
f=fixture.Fixture(failed=True);f.running=f.pair;f.running_ref=f.pair_ref
f.observation.update(running=f.pair_ref);f.final.update(running=f.pair_ref)
f.terminal['phase_states'][1].update(owner=None,exit=None)
f.observation['actual_outer_exit']=fixture.observer.reconcile(f.terminal,f.running,f.actual)
out=compose(f);assert [x['pid']for x in out['objects'][0][1]['owners']]==[101,201]
checks['unobserved_failed_replay_install_retains_no_fabricated_owner_or_exit']=True
# Source-only mutations into completed science and actual exits must reject.
for field,value in [('full_frames32',31),('cache_pairs9408',9407),('cpu_hook_pairs5964',5963),
                    ('stage_vectors13632',13631),('all_original_failures_preserved',False)]:
    f=fixture.Fixture();f.equality[field]=value
    fail_closed('science_'+field+'_mutation_rejected',lambda:compose(f))
f=fixture.Fixture();f.terminal['phase_states'][1]['exit']['properties']['ExecMainStatus']='1'
fail_closed('phase_nonzero_exit_cannot_be_promoted_by_outer_zero',lambda:compose(f))
f=fixture.Fixture();f.terminal['failed']=True
fail_closed('outer_zero_cannot_hide_failed_terminal',lambda:compose(f))
f=fixture.Fixture(failed=True);f.observation['full_replay_equality']=f.equality_ref
fail_closed('failed_outer_cannot_attach_successful_equality',lambda:compose(f))
f=fixture.Fixture();f.observer_owner.update(pid=101,start_ticks='1010')
fail_closed('observer_cannot_reuse_outer_identity',lambda:compose(f))
f=fixture.Fixture();f.observation['final_owned']['all_declared_units']=[]
fail_closed('observation_omitted_declared_units_rejected',lambda:compose(f))
# Check partial prefix visibility with the original catalog consumer.
for count in (1,2,3):
    f=fixture.Fixture();out=compose(f)
    for ref,value in out['objects'][:count]:f.values[ref['path']]=value
    fail_closed('partial_publication_'+str(count)+'_not_usable_catalog',lambda:consumer.serial_catalog(
        dict(serial_predecessors=out['catalog'],source_sha256=f.spec['source_sha256']),f.read,lambda:None))
# The reviewed fix must preserve all acknowledged identities on failure.
for index,phase in enumerate(('pair','replay')):
    f=fixture.Fixture(failed=True);f.terminal['phase_states'][index].update(owner=None,exit=None)
    fail_closed('failed_terminal_cannot_omit_acknowledged_'+phase+'_owner',lambda:compose(f))
f=fixture.Fixture(failed=True)
f.terminal['phase_states'][1]['owner']['owner'].update(pid=901,start_ticks='9010')
f.terminal['phase_states'][1]['owner']['process'].update(pid=901,start_ticks='9010')
fail_closed('self_consistent_replacement_owner_cannot_replace_running_ack',lambda:compose(f))
f=fixture.Fixture();f.values[f.pair_ref['path']]['previous']=f.ref('foreign-idle')
fail_closed('foreign_pair_admission_predecessor_rejected',lambda:compose(f))
# Actual comparator produces a three-field metadata ref: this remains accepted.
f=fixture.Fixture();f.equality_ref['bytes']=123
out=compose(f);assert out['retained_replay_passed']
checks['size_bearing_comparator_metadata_reference_accepted']=True
# Explicit finalizer self pin: valid bytes pass, absent/wrong pins fail.
finalizer=fixture.finalizer
source_data=(Q/'minicpm5_retained_replay_finalizer_v3.py').read_bytes()
self_spec={'source_sha256':{finalizer.SELF:hashlib.sha256(source_data).hexdigest()}}
fake_io=fixture.SimpleNamespace(raw=lambda path,cap:source_data)
with patch.object(fixture.caller,'api',return_value=(fake_io,None)):
    finalizer.verify_self(self_spec);checks['finalizer_current_self_source_pin_accepts']=True
    fail_closed('finalizer_missing_self_pin_rejected',lambda:finalizer.verify_self({'source_sha256':{}}))
    fail_closed('finalizer_wrong_self_pin_rejected',lambda:finalizer.verify_self({'source_sha256':{finalizer.SELF:'0'*64}}))
# Pin allowed exact metadata source shape; none of the synthetic state hit disk.
assert blocked==[],blocked
paths=['minicpm5_retained_replay_catalog_v1.py','minicpm5_retained_replay_contract_v2.py',
 'minicpm5_retained_replay_caller_v2.py','minicpm5_retained_replay_observer_v2.py',
 'minicpm5_retained_replay_finalizer_v3.py','prepare_minicpm5_retained_replay_binding_v2.py',
 'test_minicpm5_retained_replay_lifecycle_v1.py','test_prepare_minicpm5_retained_replay_binding_v2.py']
print(json.dumps(dict(schema='minicpm5_retained_lifecycle_independent_source_probe_v1',
 source_sha256={str(Q/p):hashlib.sha256((Q/p).read_bytes()).hexdigest()for p in paths},
 passed_cases=len(checks),checks=checks,audit_denials=blocked,
 actual_json_payload_history_lock_process_native_reads=False),indent=2))
