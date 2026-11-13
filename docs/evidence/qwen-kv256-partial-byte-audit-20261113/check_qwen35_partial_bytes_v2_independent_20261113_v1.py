"""Targeted source-only probes for v2; no actual audit/state imports or reads."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
import types

Q=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
SOURCE=Q/'audit_qwen35_kv256_partial_bytes_root_20261113_v2.py'
source=SOURCE.read_bytes()
tree=ast.parse(source)
main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
results={}
def require(ok,message):
    if not ok: raise ValueError(message)
def extract(name,scope):
    node=next(x for x in ast.walk(main) if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<extracted>','exec'),scope)
    return scope[name]
def rejected(name,fn,message=None):
    try:fn()
    except ValueError as error:
        if message is not None:assert str(error)==message,(name,str(error))
        results[name]=True
    else:raise AssertionError(name+' unexpectedly accepted')
def identity(path):
    s=Path(path).lstat()
    return dict(realpath=str(path),device=s.st_dev,inode=s.st_ino,size=s.st_size,
        mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def ref(path,cap=8*2**20):
    data=Path(path).read_bytes();assert len(data)<=cap
    return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest())

clock=[10.0];expected=dict(path='/synthetic/idle',identity=dict(inode=1));latest=[expected]
heavy=[False];checks=dict(latest=0,heavy=0)
def latest_handoff():checks['latest']+=1;return latest[0]
def active_heavy():
    checks['heavy']+=1
    require(not heavy[0],'synthetic heavy job')
scope=dict(require=require,lock=types.SimpleNamespace(check=lambda:None),
    time=types.SimpleNamespace(monotonic=lambda:clock[0]),started=0.0,last=[-1.0],
    identity=lambda p:expected['identity'],idle_ref=dict(path='/synthetic/idle'),expected=expected,
    latest_handoff=latest_handoff,process_api=lambda:types.SimpleNamespace(active_heavy=active_heavy))
guard=extract('guard',scope);guard()
clock[0]=10.1;latest[0]=dict(path='/synthetic/successor',identity=dict(inode=2))
rejected('forced_final_guard_rejects_new_successor_within_throttle',lambda:guard(True),'Unknown successor during raw analysis')
latest[0]=expected;heavy[0]=True
rejected('forced_final_guard_rejects_heavy_within_throttle',lambda:guard(True),'synthetic heavy job')
heavy[0]=False;guard(True);results['forced_final_guard_accepts_unchanged']=True
final_print=next(x for x in ast.walk(main) if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call)
    and isinstance(x.value.func,ast.Name) and x.value.func.id=='print')
try_node=next(x for x in main.body if isinstance(x,ast.Try))
idx=try_node.body.index(final_print)
last_node=try_node.body[idx-1]
assert isinstance(last_node,ast.Expr) and isinstance(last_node.value,ast.Call)
assert isinstance(last_node.value.func,ast.Name) and last_node.value.func.id=='guard'
assert len(last_node.value.args)==1 and last_node.value.args[0].value is True
results['forced_guard_is_immediately_before_publication']=True

with tempfile.TemporaryDirectory(prefix='qwen35-v2-source-review-') as tmp:
    root=Path(tmp);(root/'inputs').mkdir()
    path=root/'inputs'/'data.raw';data=b'\x12\x34'*8;path.write_bytes(data)
    reference=dict(ref(path),bytes=len(data));seen={}
    capture=extract('capture',dict(require=require,guard=lambda:None,STATE_BYTES=8*2**20,
        Path=Path,root=root,identity=identity,os=os,stat=stat,hashlib=hashlib,seen=seen))
    assert capture(reference,'inputs')==data
    first=copy.deepcopy(seen)
    assert capture(reference,'inputs')==data
    results['unchanged_repeated_capture_accepts']=seen==first
    replacement=root/'replacement.raw';replacement.write_bytes(data);os.replace(replacement,path)
    rejected('same_bytes_capture_inode_change_rejected',lambda:capture(reference,'inputs'),'Keep first raw reference and identity')
    assert seen==first;results['first_raw_identity_preserved_after_rejection']=True
    path.write_bytes(b'x'*len(data))
    rejected('changed_capture_bytes_rejected',lambda:capture(reference,'inputs'),'Full raw contents and identity unchanged')
    metadata=root/'analysis.py';metadata.write_text('original source\n');pinned={}
    pin=extract('pin',dict(require=require,identity=identity,ref=ref,pinned=pinned))
    metadata_ref=ref(metadata);pin(metadata_ref);pin(metadata_ref)
    results['unchanged_repeated_metadata_binding_accepts']=True
    final_loop=next(x for x in try_node.body if isinstance(x,ast.For) and isinstance(x.iter,ast.Call)
        and isinstance(x.iter.func,ast.Attribute) and isinstance(x.iter.func.value,ast.Name)
        and x.iter.func.value.id=='pinned')
    final_code=compile(ast.fix_missing_locations(ast.Module(body=[final_loop],type_ignores=[])),'<final-stability>','exec')
    final_scope=dict(require=require,identity=identity,ref=ref,pinned=pinned)
    exec(final_code,final_scope);results['final_metadata_unchanged_accepts']=True
    replacement=root/'source-replacement.py';replacement.write_text('original source\n');os.replace(replacement,metadata)
    rejected('same_bytes_metadata_inode_rebinding_rejected',lambda:pin(metadata_ref),'Keep first metadata/source identity')
    rejected('final_metadata_same_bytes_inode_change_rejected',lambda:exec(final_code,final_scope),'Every analysis source and metadata remains at its first binding')
    metadata.write_text('different source\n')
    rejected('metadata_content_change_rejected',lambda:pin(metadata_ref),'Metadata/source stable at initial binding')

host=next(x for x in ast.walk(main) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)
    and x.func.id=='require' and len(x.args)>1 and isinstance(x.args[1],ast.Constant)
    and x.args[1].value=='This prefix records unchanged host inputs')
host_code=compile(ast.Expression(host.args[0]),'<host-checks>','eval')
def host_ok(values,alias=1,returned=True,changed=None):
    row=dict(call_returned=returned)
    if changed is not None:row['changed_host_inputs']=changed
    return eval(host_code,dict(row=row,host_checks=values,graph=dict(alias=alias)))
assert host_ok([True,True]) and host_ok([True,True,True],alias=None)
results['exact_host_boolean_arities_accept']=True
for name,values,kwargs in [('empty',[],{}),('integer',[1,1],{}),('tuple',(True,True),{}),
    ('wrong_arity',[True,True,True],{}),('false',[True,False],{}),
    ('returned_integer',[True,True],dict(returned=1)),
    ('changed_capture',[True,True],dict(changed={'0':'synthetic'}))]:
    assert not host_ok(values,**kwargs),name
    results['host_'+name+'_rejected']=True

bytes_source=Q/'qwen35_kv256_alias_diagnostic_bytes_v1.py'
bytes_raw=bytes_source.read_bytes();pure={}
exec(compile(bytes_raw,str(bytes_source),'exec'),pure)
words=[0x0000,0x3f80,0x0001,0x8000,0x7f80,0x7fc1,0x7f81,0xbf80]
expected_bytes=struct.pack('<8H',*words);words[0]=0x8000;words[6]=0x7fc1
actual=struct.pack('<8H',*words)
observed,bits=pure['compare'](actual,expected_bytes,2,rows=1,tokens=4,width=2)
assert bits==bytes([2,16])
assert observed['differing_bytes']==observed['differing_words']==2
assert observed['word_class_transitions']=={'positive_zero->negative_zero':1,'signaling_nan->quiet_nan':1}
assert [x['differing_words'] for x in observed['partitions']]==[1,1]
results['pure_compare_prefix_tail_and_bf16_class_mock']=True
proof_check=next(x for x in ast.walk(main) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)
    and x.func.id=='require' and len(x.args)>1 and isinstance(x.args[1],ast.Constant)
    and x.args[1].value=='Every saved difference reproduced')
proof_code=compile(ast.Expression(proof_check.args[0]),'<full-proof-check>','eval')
proof=dict(copy.deepcopy(observed),actual='synthetic',expected='synthetic',difference_bitmap='synthetic')
def proof_ok(bitmap,proof):return eval(proof_code,dict(bitmap=bitmap,derived_bits=bits,observed=observed,proof=proof))
assert proof_ok(bits,proof);results['all_exact_comparison_fields_accept']=True
assert not proof_ok(bytes([3,16]),proof);results['modified_bitmap_rejected']=True
changed=copy.deepcopy(proof);changed['partitions'][0]['differing_words']=0
assert not proof_ok(bits,changed);results['modified_partition_field_rejected']=True
changed=copy.deepcopy(proof);changed['actual_sha256']='0'*64
assert not proof_ok(bits,changed);results['modified_raw_digest_proof_rejected']=True

assignment=next(x for x in ast.walk(main) if isinstance(x,ast.Assign)
    and any(isinstance(y,ast.Name) and y.id=='sources' for y in x.targets))
source_paths=eval(compile(ast.Expression(assignment.value),'<source-path-list>','eval'),dict(Q=Q))
assert all(x.is_file() for x in source_paths),source_paths
results['explicit_source_paths_exist']=True
assert (Q.parent/'minicpm5_layer0_execution_protocol_v4.py') in source_paths
assert (Q.parent/'qwen35_predicated_slot_protocol_v1.py') in source_paths
results['original_protocol_source_parent_paths_corrected']=True
assert SOURCE.read_bytes()==source,'source changed during review'
assert all(results.values())
print(json.dumps(dict(schema='qwen35_partial_bytes_v2_independent_source_probe_v1',
    source=dict(path=str(SOURCE),sha256=hashlib.sha256(source).hexdigest()),
    pure_compare_source=dict(path=str(bytes_source),sha256=hashlib.sha256(bytes_raw).hexdigest()),
    actual_audit_executed=False,real_metadata_raw_history_idle_lock_reads=False,
    passed_cases=len(results),checks=results),indent=2))
