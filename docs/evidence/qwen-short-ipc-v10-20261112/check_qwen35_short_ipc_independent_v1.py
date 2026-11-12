"""Independent v10 source/mock review; no current process/socket/service work."""
import ast
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import stat
import sys
import unittest
from unittest.mock import patch
Q=Path(__file__).resolve().parent;ROOT=Q.parent;STDLIB=Path(os.__file__).parent
sys.path[:0]=[str(Q),str(ROOT)];sys.excepthook=sys.__excepthook__
DENIED={'torch','numpy','ctypes','_ctypes','transformers','vllm','torch_neuronx','neuronxcc','nki','neuronx_distributed','neuronx_distributed_inference'}
class Fence(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in DENIED:raise AssertionError('Forbidden native import: '+name)
sys.meta_path.insert(0,Fence())
OUTPUT=Q/'qwen35-short-ipc-v10-independent-test-results-20261112-v1.json'
read_paths=set()
def audit(event,args):
    if event=='open':
        path,mode,flags=args
        if isinstance(path,int):raise AssertionError('No current FD/payload read')
        path=Path(path)
        writing=(mode is not None and any(ch in mode for ch in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        if writing:
            if path!=OUTPUT:raise AssertionError('Only independent result JSON may be written')
            return
        if path.suffix=='.pyc':raise PermissionError('No bytecode read; source fallback')
        if path.suffix not in ('.py','.json','.patch') or not (path.is_relative_to(ROOT) or path.is_relative_to(STDLIB) or path.is_relative_to('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/vllm')):
            raise AssertionError('Source/JSON only: '+str(path))
        read_paths.add(str(path))
    if event=='import' and args[0].split('.')[0] in DENIED:raise AssertionError('Native import forbidden')
    if event.startswith(('ctypes.','subprocess.','os.exec','os.spawn','socket.','fcntl.')) or event in ('os.system','os.fork','os.forkpty'):
        raise AssertionError('No current physical operation: '+event)
sys.addaudithook(audit)
import check_qwen35_sampler_short_ipc_v10 as original
import qwen35_sampler_runtime_contract_v10 as c
import qwen35_sampler_runtime_io_v10 as io
MANIFEST=dict(path=str(Q/'qwen35-sampler-short-ipc-v10-sources-20261112-v1.json'),sha256='7bc928e5f6d4b0450a593b0e0aa3265512ff04a17bd6413e8b67dcd3143c986c')
DIFF=dict(path=str(Q/'qwen35-sampler-short-ipc-v10-source-diff-20261112-v1.patch'),sha256='38e4fe081cf6086e6b4c556a41d1761245700361db0f5731262c987fb7de6009')
SPEC=dict(path=str(Q/'qwen35-sampler-r7-20261112-spec-v9.json'),sha256='0572165e81926a8daf92675390537b7c837713f84f739ad4e202db3c13360f1b')

def read(ref):
    p=Path(ref['path']);before=p.stat();raw=p.read_bytes();after=p.stat()
    assert len(raw)<=64*2**20 and hashlib.sha256(raw).hexdigest()==ref['sha256']
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
    return json.loads(raw) if p.suffix=='.json' else raw

def reference(p):
    raw=Path(p).read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

class Independent(unittest.TestCase):
    def test_frozen_sources_rpc_selectors_and_manifest_limits_match(self):
        manifest=read(MANIFEST);read(DIFF);prior=read(manifest['previous_source_manifest'])
        for r in list(manifest['source_files'].values())+list(prior['source_files'].values()):read(r)
        for path,sha in manifest['rpc_source_sha256'].items():read(dict(path=path,sha256=sha))
        self.assertEqual(manifest['limits'],prior['limits']);self.assertEqual(c.LIMITS,prior['limits'])
        self.assertEqual(c.LIMITS['owned_cleanup_seconds'],45)

    def test_all_admitted_ascii_run_lengths_and_nonascii_reject(self):
        for size in range(1,41):
            value=original.spec('r'*size)
            if size<=23:
                rows=c.ipc_paths(value);self.assertEqual(len(rows),6)
                self.assertEqual(max(r['path_bytes'] for r in rows),84+size)
                self.assertTrue(all(r['path_bytes']==len(r['path'].encode('utf8')) for r in rows))
            else:
                with self.assertRaises(ValueError):c.ipc_paths(value)
        for name in ('aé','a中','a/escape','a..','/a'):
            with self.assertRaises(ValueError):c.ipc_paths(original.spec(name))

    def test_two_trees_share_original_budget_and_only_control_uuid_sockets(self):
        value=original.spec();out=Path(value['output_root']);control=c.control_root(value)
        uuid=c.IPC_UUID;target=c.server_temporary_roots(value)[0]/uuid
        files={out/'model.pt':original.fake_stat(stat.S_IFREG,size=4096,blocks=8),
               control/'phase.json':original.fake_stat(stat.S_IFREG,size=1024,blocks=2),
               target:original.fake_stat(stat.S_IFSOCK,blocks=3)}
        usage,inventory=original.scan_with(files,value)
        self.assertEqual(usage['allocated_bytes'],13*512)
        self.assertEqual(usage['classes']['model_payload_bytes'],8*512)
        self.assertEqual(usage['classes']['all_metadata_bytes'],2*512)
        self.assertEqual(usage['classes']['temporary_and_directory_bytes'],3*512)
        self.assertTrue({str(p) for p in files}<={r[0] for r in inventory})
        for wrong in (out/'baseline-capture'/'tmp'/uuid,control/'candidate-http'/'tmp'/'nested'/uuid,
                      target.with_name(uuid.replace('-4000-','-1000-')),target.with_name(uuid.replace('-8000-','-7000-')),
                      target.with_name('ffffffff-ffff-4fff-8fff-fffffffffffF')):
            with self.subTest(path=wrong),self.assertRaises(ValueError):original.scan_with({wrong:original.fake_stat(stat.S_IFSOCK)},value)
        for mode in (stat.S_IFCHR,stat.S_IFBLK,stat.S_IFIFO,stat.S_IFLNK):
            with self.subTest(mode=mode),self.assertRaises(ValueError):original.scan_with({target:original.fake_stat(mode)},value)

    def test_socket_temp_cap_is_exact_and_terminal_scan_stability_retained(self):
        value=original.spec();target=c.server_temporary_roots(value)[0]/c.IPC_UUID
        limit=c.LIMITS['temporary_and_directory_bytes'];files={target:original.fake_stat(stat.S_IFSOCK,blocks=limit//512)}
        usage,_=original.scan_with(files,value);self.assertEqual(usage['allocated_bytes'],limit)
        files[target]=original.fake_stat(stat.S_IFSOCK,blocks=limit//512+1)
        with self.assertRaises(ValueError):original.scan_with(files,value)
        with patch.object(io,'storage_scan',side_effect=[({'allocated_bytes':0},[]),({'allocated_bytes':512},[])]) as scans:
            with self.assertRaisesRegex(ValueError,'terminal output inventory'):io.storage(value,terminal=True)
        self.assertEqual(scans.call_count,2)

    def test_all48002_executed_pins_retained_in_additive_inmemory_union(self):
        spec=read(SPEC);manifest=read(MANIFEST);before=copy.deepcopy(spec['source_sha256'])
        self.assertEqual(len(before),48002)
        additions={r['path']:r['sha256'] for r in manifest['source_files'].values()}
        self.assertFalse(set(before)&set(additions))
        union=c.merge_sources(before,additions);self.assertEqual(len(union),48008)
        self.assertEqual({p:union[p] for p in before},before)
        self.assertEqual(spec['source_sha256'],before)
        self.assertTrue(all(before.get(p)==h for p,h in manifest['rpc_source_sha256'].items()))
        bad=copy.deepcopy(additions);bad[next(iter(before))]='f'*64
        with self.assertRaises(ValueError):c.merge_sources(before,bad)

    def test_whole_old_runtime_ast_preserved_except_exact_ipc_changes(self):
        for stem in ('contract','io','history','service','adapter'):
            old=(Q/('qwen35_sampler_runtime_'+stem+'_v9.py')).read_text().replace('_v9','_v10')
            new=(Q/('qwen35_sampler_runtime_'+stem+'_v10.py')).read_text()
            old_tree=ast.parse(old);new_tree=ast.parse(new)
            if stem=='contract':
                original_control=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name=='control_root')
                remove_functions={'ipc_paths','server_temporary_roots','phase_environment'}
                new_tree.body=[n for n in new_tree.body if not (isinstance(n,ast.FunctionDef) and n.name in remove_functions)
                    and not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'IPC_PATH_LIMIT','IPC_UUID','SERVER_PHASES'} for t in n.targets))]
                for i,n in enumerate(new_tree.body):
                    if isinstance(n,ast.FunctionDef) and n.name=='control_root':new_tree.body[i]=original_control
                    if isinstance(n,ast.FunctionDef) and n.name=='validate_spec':
                        n.body=[x for x in n.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id=='ipc_paths')]
            if stem=='io':
                new_tree.body=[n for n in new_tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='ipc_socket_path')
                    and not (isinstance(n,ast.Import) and any(a.name=='re' for a in n.names))]
                for n in new_tree.body:
                    if isinstance(n,ast.ImportFrom):n.names=[a for a in n.names if a.name!='server_temporary_roots']
                class RemoveSocket(ast.NodeTransformer):
                    def visit_If(self,node):
                        if isinstance(node.test,ast.Call) and isinstance(node.test.func,ast.Attribute) and node.test.func.attr=='S_ISSOCK':return None
                        return self.generic_visit(node)
                new_tree=RemoveSocket().visit(new_tree)
            if stem=='adapter':
                for n in new_tree.body:
                    if isinstance(n,ast.ImportFrom):n.names=[a for a in n.names if a.name!='phase_environment']
                class Environment(ast.NodeTransformer):
                    def visit_Assign(self,node):
                        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='phase_environment':
                            return ast.parse('env["TMPDIR"] = str(Path(phase["control"]) / "tmp")').body[0]
                        return self.generic_visit(node)
                new_tree=Environment().visit(new_tree)
            self.assertEqual(ast.dump(old_tree,include_attributes=False),ast.dump(new_tree,include_attributes=False),stem)
        old=(Q/'run_qwen35_sampler_runtime_v9.py').read_text().replace('_v9','_v10')
        new=(Q/'run_qwen35_sampler_runtime_v10.py').read_text()
        self.assertEqual(ast.dump(ast.parse(old)),ast.dump(ast.parse(new)))

suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(original.ShortIPC),unittest.defaultTestLoader.loadTestsFromTestCase(Independent)])
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert not DENIED.intersection(sys.modules)
record=dict(schema='qwen35_short_ipc_v10_independent_test_results_v1',reviewer='/root/mini_mapping_authority_source',
    passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
    test_source=reference(__file__),manifest=MANIFEST,diff=DIFF,saved_executed_r7_spec=SPEC,
    source_mock_only=True,read_paths=sorted(read_paths),native_imports=False,current_process_or_storage_or_socket_operations=False,
    actual_spec_or_admission_created=False,launch_authorized=False,new_speedup_claim=False)
with OUTPUT.open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,output=str(OUTPUT))))
raise SystemExit(0 if result.wasSuccessful() else 1)
