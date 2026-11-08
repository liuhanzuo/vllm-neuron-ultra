"""v11 storage integration: real sockets, full rescans and retained HTTP lifecycle."""
import ast,hashlib,json,os,socket,sys,tempfile,types,unittest,uuid
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v10 as prior
import nemotron9b_package_protocol_v11 as p
import nemotron9b_package_evidence_v11 as evidence
import nemotron9b_package_evidence_v10 as old_evidence

def replay(name,path,replacements):
 module=types.ModuleType(name);module.__file__=str(path)
 source=path.read_text()
 for before,after in replacements:source=source.replace(before,after)
 exec(compile(source,str(path),'exec'),module.__dict__)
 return module

class Closure(unittest.TestCase):
 def test_v10_parent_bytes_unchanged(self):
  spec=json.loads((R/'research/nemotron9b-package-runtime-v10-r4-20261108-spec-v2.json').read_text())
  for f in R.glob('nemotron9b_package_*v10.py'):self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),spec['source_sha256'][str(f)])
 def test_twelve_other_runtime_modules_only_version_rename(self):
  for f in R.glob('nemotron9b_package_*v10.py'):
   if f.name in ('nemotron9b_package_protocol_v10.py','nemotron9b_package_metadata_v10.py','nemotron9b_package_evidence_v10.py','nemotron9b_package_raw_verify_v10.py'):continue
   self.assertEqual(ast.dump(ast.parse(f.read_text().replace('_v10','_v11'))),ast.dump(ast.parse(f.with_name(f.name.replace('_v10','_v11')).read_text())))
 def test_protocol_changes_limited_to_inventory_and_source_closure(self):
  def funcs(version):return {n.name:ast.dump(n)for n in ast.parse((R/('nemotron9b_package_protocol_'+version+'.py')).read_text().replace('_v10','_v11')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  before,after=funcs('v10'),funcs('v11');self.assertEqual(set(before),set(after))
  for name in before:
   if name not in {'required_sources','_usage_once','usage','storage'}:self.assertEqual(before[name],after[name],name)
 def test_raw_verifier_changes_only_work_tree_context(self):
  before=(R/'nemotron9b_package_raw_verify_v10.py').read_text().replace('_v10','_v11').replace("work=tree(Path(spec['work_root']),guard)","work=tree(Path(spec['work_root']),guard,spec=spec)")
  self.assertEqual(ast.dump(ast.parse(before)),ast.dump(ast.parse((R/'nemotron9b_package_raw_verify_v11.py').read_text())))
  def funcs(version):return {n.name:ast.dump(n)for n in ast.parse((R/('nemotron9b_package_evidence_'+version+'.py')).read_text().replace('_v10','_v11')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  old,new=funcs('v10'),funcs('v11');self.assertEqual(set(new)-set(old),{'ipc_metadata'})
  for name in old:
   if name not in {'tree','verify_rows','verify_phase','final_audit'}:self.assertEqual(old[name],new[name],name)
 def test_historical_metadata_support_retained(self):
  for name,shape in prior.metadata.ENVELOPE_SCHEMAS.items():self.assertEqual(shape,p.metadata.ENVELOPE_SCHEMAS[name])
  self.assertEqual(p.metadata.ENVELOPE_SCHEMAS['nemotron9b_package_runtime_v11'],prior.metadata.ENVELOPE_SCHEMAS['nemotron9b_package_runtime_v10'])
  self.assertEqual(p.metadata.LIMITS,prior.metadata.LIMITS)
  def funcs(mod):return {n.name:ast.dump(n)for n in ast.parse(Path(mod.__file__).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  self.assertEqual(funcs(prior.metadata),funcs(p.metadata))
 def test_required_source_closure_is_strict_superset(self):
  required=set(p.required_sources());self.assertTrue(set(prior.required_sources())<=required)
  for name in ['nemotron9b_package_storage_inventory_candidate_v1.py','research/check_nemotron9b_storage_inventory_candidate_20261108_v1.py','research/check_nemotron9b_package_exec_v11.py']:self.assertIn(str(R/name),required)
  self.assertTrue(all(str(f)in required for f in R.glob('nemotron9b_package_*v11.py')))
  self.assertTrue(all(Path(f).is_file()for f in required))
 def test_science_and_budgets_unchanged(self):
  for name in ['STAGES','SCIENTIFIC_STAGES','ROOT_FLOOR','NVME_FLOOR','ROOT_CAP','NVME_CAP','CAPS','BUCKETS','SPLIT_FLAGS','ALIAS_FLAGS','PACKAGE','COMMIT']:self.assertEqual(getattr(prior,name),getattr(p,name),name)
  self.assertEqual(len(p.STAGES),52)
  self.assertEqual(prior.alias_jobs(),p.alias_jobs());self.assertEqual(prior.split_jobs(),p.split_jobs())
  for stage in (*p.STAGES,'controller'):self.assertEqual(prior.resources(stage),p.resources(stage))

class IntegratedStorage(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='nv11-');self.root=Path(self.tmp.name)
  self.spec={k:str(self.root/n)for k,n in [('control_root','control'),('evidence_root','evidence'),('work_root','work')]}
  for root in self.spec.values():Path(root).mkdir()
  self.work=Path(self.spec['work_root']);self.ipc=self.work/'temporary/tmp';self.ipc.mkdir(parents=True);self.sockets=[]
 def tearDown(self):
  for obj in self.sockets:obj.close()
  self.tmp.cleanup()
 def bind(self,path):
  obj=socket.socket(socket.AF_UNIX);obj.bind(str(path));self.sockets.append(obj)
 def test_storage_routes_exception_only_to_work_and_temporary(self):
  self.bind(self.ipc/str(uuid.uuid4()))
  with self.assertRaisesRegex(RuntimeError,'Ordinary new output file'):prior.storage(self.spec)
  with patch.object(p,'usage',wraps=p.usage)as calls,patch.object(p.os,'statvfs',return_value=types.SimpleNamespace(f_bavail=64*p.GIB,f_frsize=1)):
   result=p.storage(self.spec,reserve=True)
  self.assertEqual(result['nvme_used'],0)
  for call in calls.call_args_list:
   root=call.args[0];expected=self.ipc if root in [self.work,self.work/'temporary']else None
   self.assertEqual(call.kwargs.get('ipc_root'),expected)
 def test_storage_rejects_socket_in_evidence_or_wrong_category(self):
  for root in [Path(self.spec['evidence_root']),self.work/'artifact',self.ipc/'nested']:
   root.mkdir(exist_ok=True);path=root/str(uuid.uuid4());self.bind(path)
   with self.assertRaisesRegex(RuntimeError,'declared temporary'):p.storage(self.spec)
   path.unlink()
 def test_storage_floor_and_reservation_still_enforced(self):
  for reserve,free in [(False,27*p.GIB),(True,47*p.GIB)]:
   with patch.object(p.os,'statvfs',return_value=types.SimpleNamespace(f_bavail=free,f_frsize=1)):
    with self.assertRaisesRegex(RuntimeError,'floor/reservation'):p.storage(self.spec,reserve=reserve)
 def test_socket_count_is_bounded(self):
  for _ in range(64):self.bind(self.ipc/str(uuid.uuid4()))
  self.assertEqual(p.usage(self.work,ipc_root=self.ipc),0)
  self.bind(self.ipc/str(uuid.uuid4()))
  with self.assertRaisesRegex(RuntimeError,'socket count'):p.usage(self.work,ipc_root=self.ipc)
 def test_socket_disappearance_uses_complete_rescan(self):
  path=self.ipc/str(uuid.uuid4());self.bind(path);data=self.work/'data';data.write_bytes(b'retained');walk=os.walk;calls=[]
  def racing(*args,**kwargs):
   calls.append(1)
   for row in walk(*args,**kwargs):
    if len(calls)==1 and Path(row[0])==self.ipc:path.unlink()
    yield row
  with patch.object(p.os,'walk',side_effect=racing):actual=p.usage(self.work,ipc_root=self.ipc)
  info=data.stat();self.assertEqual(actual,max(info.st_size,info.st_blocks*512));self.assertEqual(len(calls),2)
 def test_temporary_category_cap_retained_with_socket(self):
  self.bind(self.ipc/str(uuid.uuid4()))
  with (self.work/'temporary/large').open('wb')as stream:stream.truncate(p.CAPS['temporary']+1)
  with self.assertRaisesRegex(RuntimeError,'category cap temporary'):p.storage(self.spec)
 def test_live_socket_hardlink_and_alias_paths_rejected(self):
  path=self.ipc/str(uuid.uuid4());self.bind(path);os.link(path,self.ipc/str(uuid.uuid4()))
  with self.assertRaisesRegex(RuntimeError,'Single-link'):p.usage(self.work,ipc_root=self.ipc)

class IntegratedRaw(unittest.TestCase):
 setUp=IntegratedStorage.setUp
 tearDown=IntegratedStorage.tearDown
 bind=IntegratedStorage.bind
 def closed_socket(self,root=None):
  path=(root or self.ipc)/str(uuid.uuid4());self.bind(path);self.sockets[-1].close();return path
 def test_original_final_manifest_rejects_detached_socket_new_manifest_captures_identity(self):
  path=self.closed_socket();data=self.work/'data.json';data.write_bytes(b'original evidence')
  with self.assertRaisesRegex(RuntimeError,'ordinary evidence'):old_evidence.tree(self.work,lambda:None)
  with patch.object(evidence,'digest',wraps=evidence.digest)as digest:
   rows=evidence.tree(self.work,lambda:None,spec=self.spec)
  self.assertEqual([c.args[0]for c in digest.call_args_list],[data])
  self.assertEqual(set(rows),{str(path),str(data)});self.assertTrue(rows[str(path)]['detached']);self.assertNotIn('sha256',rows[str(path)])
  self.assertEqual(rows[str(data)],old_evidence.digest(data,lambda:None));evidence.verify_rows(rows,lambda:None,self.spec)
 def test_live_socket_is_allowed_in_storage_but_rejected_by_final_manifest(self):
  self.bind(self.ipc/str(uuid.uuid4()));p.storage(self.spec)
  with self.assertRaisesRegex(RuntimeError,'Live Unix endpoint'):evidence.tree(self.work,lambda:None,spec=self.spec)
 def test_final_manifest_rejects_socket_without_exact_context(self):
  self.closed_socket()
  with self.assertRaisesRegex(RuntimeError,'exact work root'):evidence.tree(self.work,lambda:None)
  with self.assertRaisesRegex(RuntimeError,'exact work root'):evidence.tree(self.work,lambda:None,spec=dict(self.spec,work_root=str(self.root/'other')))
 def test_final_manifest_rejects_socket_outside_explicit_temporary_directory(self):
  self.closed_socket(self.work)
  with self.assertRaisesRegex(RuntimeError,'declared temporary'):evidence.tree(self.work,lambda:None,spec=self.spec)
 def test_final_manifest_socket_count_retained(self):
  for _ in range(65):self.closed_socket()
  with self.assertRaisesRegex(RuntimeError,'socket evidence count'):evidence.tree(self.work,lambda:None,spec=self.spec)
 def test_tampered_or_removed_socket_identity_is_rejected(self):
  path=self.closed_socket();rows=evidence.tree(self.work,lambda:None,spec=self.spec)
  rows[str(path)]['stat']['inode']+=1
  with self.assertRaisesRegex(RuntimeError,'identity changed'):evidence.verify_rows(rows,lambda:None,self.spec)
  path.unlink()
  with self.assertRaises(FileNotFoundError):evidence.verify_rows(rows,lambda:None,self.spec)
 def test_socket_changed_during_capture_is_rejected(self):
  path=self.closed_socket();count=[0]
  def guard():
   count[0]+=1
   if count[0]==2:path.unlink();path.write_bytes(b'changed')
  with self.assertRaisesRegex(RuntimeError,'socket evidence'):evidence.ipc_metadata(path,self.work,self.spec,guard)
 def test_manifest_cannot_reclassify_regular_data_as_socket(self):
  path=self.ipc/str(uuid.uuid4());path.write_bytes(b'data')
  with self.assertRaisesRegex(RuntimeError,'socket evidence'):evidence.ipc_metadata(path,self.work,self.spec,lambda:None)
 def test_fifo_and_socket_hardlinks_still_rejected_by_manifest(self):
  path=self.closed_socket();os.link(path,self.ipc/str(uuid.uuid4()))
  with self.assertRaisesRegex(RuntimeError,'single-link'):evidence.tree(self.work,lambda:None,spec=self.spec)

if __name__=='__main__':
 candidate=replay('candidate_v11',R/'research/check_nemotron9b_storage_inventory_candidate_20261108_v1.py',[
  ('import nemotron9b_package_storage_inventory_candidate_v1 as c','import nemotron9b_package_protocol_v11 as candidate_protocol\nc=types.SimpleNamespace(inventory=candidate_protocol.usage)'),
  ('import os,socket,sys,tempfile,unittest,uuid','import os,socket,sys,tempfile,types,unittest,uuid')])
 inventory=replay('inventory_v11',R/'research/check_nemotron9b_package_inventory_v7.py',[('import nemotron9b_package_protocol_v7 as p','import nemotron9b_package_protocol_v11 as p')])
 http=replay('http_v11',R/'check_nemotron9b_package_server_lifecycle_v4.py',[('_v4','_v11')])
 class TransientHttp(unittest.TestCase):
  def test_resolved_empty_argv_retains_original_http_success(self):
   with tempfile.TemporaryDirectory()as tmp:
    h=http.Harness(Path(tmp));original=h.proc;empty=[False]
    def proc(pid):
     if h.executed and not empty[0]:empty[0]=True;return dict(h.child_owner,argv=[])
     return original(pid)
    h.proc=proc;result=h.run();h.check(result)
    diagnostic=p.read(Path(tmp)/'quality-server-exec-diagnostic.json')
    self.assertTrue(empty[0]);self.assertEqual(diagnostic['status'],'transient_empty_argv_resolved');self.assertFalse(diagnostic['qualified']);self.assertTrue(result['passed'])
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)for cls in [Closure,IntegratedStorage,IntegratedRaw,candidate.Inventory,http.ActualHttpRegression,TransientHttp])
 for name in unittest.defaultTestLoader.getTestCaseNames(inventory.Inventory):
  if name not in {'test_other_fourteen_modules_only_import_rename','test_all_other_protocol_functions_unchanged'}:suite.addTest(inventory.Inventory(name))
 result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful()else 1)
