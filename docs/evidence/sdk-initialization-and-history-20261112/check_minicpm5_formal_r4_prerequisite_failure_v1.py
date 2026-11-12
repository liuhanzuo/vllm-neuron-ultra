"""Read saved r4 failure; execute only selected frozen AST with mock OS ports.

No history/native imports, processes, services or locks are accessed. Restored
tuple fields are a source-derived fixture, not a recovered process heap.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / 'research'
sys.path.insert(0, str(ROOT))
NATIVE = {'ctypes','_ctypes','torch','numpy','neuronxcc','torch_neuronx',
          'neuronx_distributed','neuronx_distributed_inference'}
class BlockNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in NATIVE:
            raise AssertionError('Forbidden native import: '+fullname)
assert not NATIVE.intersection(sys.modules)
sys.meta_path.insert(0, BlockNative())
from research import minicpm5_formal_literal_codec_v1 as codec

PREFIX = 'minicpm5-formal-import-r4-20261112-'
sources = {}
def read(path):
    data = Path(path).read_bytes()
    return data, dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
def saved(name):
    data, reference = read(Q/(PREFIX+name))
    return json.loads(data),reference
review,review_ref = saved('root-concrete-review.json')
assert review_ref['sha256'] == 'c6bfb506aa16b6285cff3a98b024cbe6e30cbc302b32994ffbda3a3d26379882'
map_raw,map_ref=read(review['complete_source_map']['path'])
assert map_ref['sha256']==review['complete_source_map']['sha256']
pins=json.loads(map_raw)
def source(name):
    path=Q/name;data,reference=read(path)
    assert pins[str(path)]==reference['sha256']
    sources[name]=reference
    return ast.parse(data,filename=str(path))
def selected(name,kind,definition,namespace):
    tree=source(name)
    nodes=[node for node in ast.walk(tree) if isinstance(node,kind) and node.name==definition]
    assert len(nodes)==1
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(Q/name),'exec'),namespace)
    return namespace[definition]
def forbidden(*args,**kwargs):
    raise AssertionError('No process/native/wait/spawn or later conjunct may execute')

pv=json.loads(Path(review['pv_spec']['path']).read_text())
output=Path(pv['output_root'])
assert sorted(path.name for path in output.iterdir())==['admission.json']
wire,wire_ref=read(output/'admission.json')
literal,literal_ref=read(codec.LITERAL['path'])
assert literal_ref==codec.LITERAL
source('minicpm5_sdk_mapping_nocache_entry_v6.py')
source('minicpm5_formal_literal_codec_v1.py')
decoded=codec.decode('admission',wire,literal.decode(),wire_sha256=wire_ref['sha256'])
memory=copy.deepcopy(decoded)
observation=memory['source_entry_binding']['outer_types_source_observation']
assert type(observation['current_source_identity']) is list
assert type(observation['execution_receipt']['identity']) is list
observation['current_source_identity']=tuple(observation['current_source_identity'])
observation['execution_receipt']['identity']=tuple(observation['execution_receipt']['identity'])
assert codec.encode('admission',memory,literal.decode())==wire
assert codec.decode('admission',wire,literal.decode(),wire_sha256=wire_ref['sha256'])!=memory

class Guard:
    def __call__(self,**kwargs): return None
class Store: pass
guard=Guard();lease=NS(admission=copy.deepcopy(memory['lease']));guard.lease=lease
store=Store();store.guard=guard
original=NS(r=NS(digest=forbidden,p=NS(proc=forbidden)),serial=NS(),require=codec.require)
base_ns=dict(copy=copy,r=original.r,require=codec.require,os=NS(getpid=forbidden))
original.SingleSDKAdapter=selected('minicpm5_pv_sdk_import_diagnostic_v1.py',ast.ClassDef,'SingleSDKAdapter',base_ns)
observed_ns=dict(original=original,ObserverStore=Store,require=codec.require,copy=copy)
Observed=selected('minicpm5_sdk_mapping_observed_adapter_v4.py',ast.ClassDef,'ObservedSingleSDKAdapter',observed_ns)
source_ns=dict(base=NS(ObservedSingleSDKAdapter=Observed),copy=copy)
Adapter=selected('minicpm5_sdk_mapping_source_adapter_v2.py',ast.ClassDef,'SourceEntrySDKAdapter',source_ns)
admission_ref={key:wire_ref[key] for key in ('path','sha256')}
metadata=codec.AdmissionMetadata(NS(read=lambda ref:json.loads(wire)),admission_ref,literal.decode(),guard)
item=Adapter({'witness':memory['witness']},admission_ref,memory,lease,guard,metadata,store,
    outer_deadline=memory['source_entry_binding']['outer_deadline'],authority={'source_mock_only':True},
    authenticate=forbidden,verify_loaded_native=forbidden,sdk_entry_bytes=literal,sdk_entry_ref=codec.LITERAL)
assert item.child is None and item.child_started is False and item.state=='new'
try: item.authenticate_complete_prerequisites()
except ValueError as error: assert str(error)=='Actual original authenticated outer and exclusive admission'
else: raise AssertionError('Original first equality must reject tuple/list mismatch')
assert item.child is item.native_ports is item.observer is item.owner is item.argv is None
assert item.prerequisites_checked is False and item.child_started is False and item.state=='new'
proof=item.drain_owned()
assert proof==dict(physically_drained=True,no_services_started=True,exit_receipt_saved=False,child_started=False)

closed=[];alarms=[]
release=selected('minicpm5_layer1_operator_native_outer_support_v3.py',ast.FunctionDef,
    'release_after_physical_drain',dict(require=codec.require,os=NS(close=closed.append)))
outer_tree=source('minicpm5_sdk_mapping_outer_source_entry_v7.py')
outer=next(node for node in outer_tree.body if getattr(node,'name',None)=='source_entry')
tried=next(node for node in outer.body if isinstance(node,ast.Try))
namespace=dict(bootstrap=NS(fd=9,adopted=True),ports=item,require=codec.require,
    signal=NS(alarm=alarms.append),original=NS(prior=NS(audited=NS(support=NS(release_after_physical_drain=release)))),
    time=NS(sleep=forbidden),sys=sys,json=json)
exec(compile(ast.Module(body=tried.finalbody,type_ignores=[]),str(Q/'minicpm5_sdk_mapping_outer_source_entry_v7.py'),'exec'),namespace)
assert closed==[9] and alarms==[0] and item.child is None
waited,wait_ref=saved('launcher-wait.json');log,log_ref=read(Q/(PREFIX+'tool.log'))
assert waited['pid']==855446 and waited['actual_returncode']==1 and waited['popen_wait_observed'] is True
trace=log.decode().split('Traceback (most recent call last):\n')
assert len(trace)==2
assert 'line 163, in source_entry\n    ports.authenticate_complete_prerequisites()' in trace[1]
assert 'line 79, in authenticate_complete_prerequisites\n    super().authenticate_complete_prerequisites()' in trace[1]
assert 'line 275, in authenticate_complete_prerequisites' in trace[1]
assert trace[1].endswith('ValueError: Actual original authenticated outer and exclusive admission\n')
assert not NATIVE.intersection(sys.modules)
print(json.dumps(dict(schema='minicpm5_formal_r4_prerequisite_failure_source_check_v1',
    source_only=True,native_imports_blocked=sorted(NATIVE),root_review=review_ref,source_pins=sources,
    actual_saved_admission=wire_ref,actual_saved_log=log_ref,actual_saved_launcher_wait=wait_ref,
    output_names=['admission.json'],source_derived_tuple_fixture_reencodes_exact_saved_wire=True,
    original_first_conjunct_reproduced_false=True,later_conjuncts_not_evaluated=True,
    original_three_constructors_and_auth_with_mocks_start_no_child=True,
    original_observed_and_base_drain_return_no_child_proof=True,original_finally_mock_closes_once_after_proof=True,
    current_processes_services_locks_inspected=False,actual_physical_drain_asserted=False,
    actual_tool_terminal_independently_observed=False,native_ABI_qualified=False,
    full_model_qualified=False,new_speedup_claim=False),sort_keys=True,separators=(',',':')))
