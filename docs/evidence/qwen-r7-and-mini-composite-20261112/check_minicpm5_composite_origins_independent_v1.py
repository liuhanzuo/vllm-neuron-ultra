"""Independent review: saved JSON/source plus synthetic plaintext only."""
import ast
import copy
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
Q=Path(__file__).resolve().parent;ROOT=Q.parent
sys.path[:0]=[str(ROOT),str(Q)]
DENIED={'ctypes','_ctypes','torch','torch_neuronx','neuronxcc','nki','numpy','transformers','vllm','neuronx_distributed','neuronx_distributed_inference'}
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in DENIED or name.startswith('research.minicpm5_sdk_mapping_worker_native'):
            raise AssertionError('Native/runtime import forbidden: '+name)
sys.meta_path.insert(0,BlockNative())
OUTPUT=Q/'minicpm5-sdk310-composite-origins-independent-test-results-20261112-v1.json'
STDLIB=Path(os.__file__).parent
sys.excepthook=sys.__excepthook__
reads=set();writes=[]
def audit(event,args):
    if event=='open':
        path,mode,flags=args
        if isinstance(path,int):raise AssertionError('FD payload access forbidden')
        path=Path(path)
        writing=(mode is not None and any(ch in mode for ch in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        if writing:
            if path!=OUTPUT:raise AssertionError('Only independent test result write permitted: '+str(path))
            writes.append(str(path));return
        if path.suffix not in ('.py','.json'):
            if path.suffix=='.pyc':raise PermissionError('No bytecode read; source fallback only')
            raise AssertionError('Source/JSON only: '+str(path))
        if not (path.is_relative_to(ROOT) or path.is_relative_to(STDLIB) or (path.suffix=='.json' and path.is_relative_to('/mnt/nvme'))):
            raise AssertionError('Unapproved source/JSON namespace: '+str(path))
        reads.add(str(path))
    if event=='import' and args[0].split('.')[0] in DENIED:raise AssertionError('Blocked native import')
    if event.startswith(('ctypes.','subprocess.','os.exec','os.spawn')) or event in ('os.system','os.fork','os.forkpty'):
        raise AssertionError('No process/native operation: '+event)
sys.addaudithook(audit)
from research import check_minicpm5_sdk310_composite_origins_v1 as original_tests
from research import check_minicpm5_sdk310_composite_saved_metadata_v1 as saved_tests
from research import minicpm5_sdk310_composite_origins_v1 as m
DELIVERY=Q/'minicpm5-sdk310-composite-origins-source-delivery-20261112-v1.json'

def read_ref(ref):
    path=Path(ref['path']);before=path.stat();raw=path.read_bytes();after=path.stat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
    assert len(raw)<=8*2**20 and hashlib.sha256(raw).hexdigest()==ref['sha256']
    if 'bytes' in ref:assert len(raw)==ref['bytes']
    return json.loads(raw) if path.suffix=='.json' else raw

def ref(path):
    raw=Path(path).read_bytes();return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

class Independent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.delivery=json.loads(DELIVERY.read_text());cls.saved=read_ref(cls.delivery['actual_metadata_check'])

    def test_all_declared_frozen_sources_and_evidence_match(self):
        for r in self.delivery['runtime_sources']+self.delivery['unchanged_predecessors']+self.delivery['check_sources']:
            self.assertTrue(read_ref(r))
        self.assertFalse(self.delivery['runtime_admitted']);self.assertFalse(self.delivery['actual_native_authentication_executed'])

    def test_actual_saved_metadata_reproduces_exact_report_without_payload(self):
        self.assertEqual(saved_tests.check(),self.saved)
        self.assertFalse(self.saved['full_source_map_reconstructed_here'])
        self.assertFalse(self.saved['full_snapshot_plaintext_replayed_here'])
        self.assertEqual(self.saved['payload_reads'],0)

    def test_four_actual_archive_budgets_are_exact_original(self):
        budget=dict(address_space_bytes=2147483648,cpu_seconds=600,wall_seconds=900,single_file_bytes=67108864,
                    output_bytes=134217728,log_bytes=8388608,cpus=[5],nice=19,root_floor_bytes=30064771072,
                    nvme_floor_bytes=32212254720,memory_available_floor_bytes=1073741824)
        for branch in ('old_libc6','new_eight'):
            for role in ('producer','replay'):
                ended=read_ref(self.saved[branch][role+'_exit']);plan=read_ref(ended['plan'])
                self.assertEqual(plan['budget'],budget)
                self.assertFalse(plan['automatic_successor_launch'])
                if branch=='new_eight':self.assertEqual(plan['mini_scientific_constraints'],m.CONSTRAINTS)

    def test_106_stage_pins_are_only_provenance_subset(self):
        pins=self.saved['stage_source_union'];self.assertEqual(len(pins),106)
        total=0
        for path,sha in pins.items():
            self.assertEqual(Path(path).suffix,'.py');raw=read_ref(dict(path=path,sha256=sha));total+=len(raw)
            self.assertLessEqual(len(raw),2*2**20)
        self.assertLessEqual(total,8*2**20)
        for r in self.delivery['runtime_sources']:self.assertNotIn(r['path'],pins)
        for path in ('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch/__init__.py',
                     '/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/__init__.py'):
            self.assertNotIn(path,pins)

    def test_whole_module_ast_only_has_intended_dispatch_changes(self):
        def normalized(name,is_new):
            tree=ast.parse((Q/name).read_text())
            tree.body=[n for n in tree.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))
                       and not (isinstance(n,ast.FunctionDef) and n.name=='replay_package_origins')
                       and not (isinstance(n,ast.ImportFrom) and any(a.name=='minicpm5_sdk310_composite_origins_v1' for a in n.names))]
            class Normalize(ast.NodeTransformer):
                def visit_Constant(self,node):
                    if node.value=='minicpm5_sdk310_native_source_review_v2':node.value='minicpm5_sdk310_native_source_review_v1'
                    return node
                def visit_Call(self,node):
                    self.generic_visit(node)
                    if isinstance(node.func,ast.Name) and node.func.id=='replay_package_origins' and is_new:
                        self.assertion=len(node.args)==3;assert self.assertion;node.args=node.args[:2]
                    return node
                def visit_ImportFrom(self,node):
                    for a in node.names:
                        if a.name=='minicpm5_sdk_mapping_worker_native_v3':a.name='minicpm5_sdk_mapping_worker_native_v2'
                    return node
            return ast.dump(Normalize().visit(tree),include_attributes=False)
        self.assertEqual(normalized('minicpm5_sdk_mapping_worker_native_v2.py',False),normalized('minicpm5_sdk_mapping_worker_native_v3.py',True))
        self.assertEqual(normalized('minicpm5_sdk_mapping_worker_source_entry_v3.py',False),normalized('minicpm5_sdk_mapping_worker_source_entry_v4.py',True))

    def test_required_source_map_is_superset_and_conflicts_reject(self):
        f=original_tests.fixture();f[2]['/synthetic/additional-reviewed.py']='e'*64
        self.assertTrue(original_tests.Checks().run_fixture(f))
        f=original_tests.fixture();f[2]['/synthetic/accepted.py']='b'*64
        with self.assertRaises(ValueError):original_tests.Checks().run_fixture(f)

suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(original_tests.Checks),unittest.defaultTestLoader.loadTestsFromTestCase(Independent)])
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert not DENIED.intersection(sys.modules)
record=dict(schema='minicpm5_sdk310_composite_origins_independent_test_results_v1',agent='/root/mini_mapping_authority_source',
    passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
    delivery=ref(DELIVERY),review_test_source=ref(__file__),read_paths=sorted(reads),native_import_blocker_active=True,
    source_and_JSON_file_fence_active=True,process_native_operations_forbidden=True,loaded_forbidden_modules=[],
    actual_index_plaintext_read=False,current_binary_or_payload_read=False,ctypes_import=False,serial_or_admission=False,
    full_native_source_map_established=False,actual_mapping_authority_established=False,new_speedup_claim=False)
with OUTPUT.open('x') as stream:json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,read_files=len(reads),output=str(OUTPUT))))
raise SystemExit(0 if result.wasSuccessful() else 1)
