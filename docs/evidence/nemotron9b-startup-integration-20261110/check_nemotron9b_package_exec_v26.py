"""v26 source and full startup integration, with synthetic timing only."""
import ast,copy,hashlib,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];Q=R/'research'
sys.path[:0]=[str(R),str(Q)]
import nemotron9b_package_protocol_v24 as old
import nemotron9b_package_protocol_v26 as p
import nemotron9b_package_startup_history_v26 as h
import nemotron9b_package_startup_drain_v26 as drain

class Source(unittest.TestCase):
 def test_original52_stages_and_all_resource_budgets_unchanged(self):
  for name in ('STAGES','SCIENTIFIC_STAGES','ROOT_FLOOR','NVME_FLOOR','ROOT_CAP','NVME_CAP','CAPS','BUCKETS','SPLIT_FLAGS','ALIAS_FLAGS','PACKAGE','COMMIT'):
   self.assertEqual(getattr(old,name),getattr(p,name),name)
  self.assertEqual(len(p.STAGES),52)
  for phase in (*p.STAGES,'controller'):self.assertEqual(old.resources(phase),p.resources(phase))
 def test_twelve_scientific_execution_modules_ast_unchanged(self):
  excluded={'protocol','metadata','worker','server_lifecycle'}
  paths=[x for x in R.glob('nemotron9b_package_*v24.py')if x.name.removeprefix('nemotron9b_package_').removesuffix('_v24.py')not in excluded]
  self.assertEqual(len(paths),12)
  for path in paths:
   self.assertEqual(ast.dump(ast.parse(path.read_text().replace('_v24','_v26'))),ast.dump(ast.parse(path.with_name(path.name.replace('_v24','_v26')).read_text())),path.name)
 def test_original_worker_science_ast_unchanged(self):
  def funcs(path):
   return {n.name:ast.dump(n)for n in ast.parse(path.read_text().replace('_v24','_v26')).body if isinstance(n,ast.FunctionDef)and n.name not in ('http_stage','serve_child')}
  self.assertEqual(funcs(R/'nemotron9b_package_worker_v24.py'),funcs(R/'nemotron9b_package_worker_v26.py'))
 def test_bound_startup_helpers_match_reviewed_candidate(self):
  expected=(Q/'nemotron9b_startup_deadlines_v2.py').read_text().replace('nemotron9b_exec_transition_v2','nemotron9b_package_exec_transition_v26')
  self.assertEqual((R/'nemotron9b_package_startup_v26.py').read_text(),expected)
  self.assertEqual((R/'nemotron9b_package_exec_transition_v26.py').read_bytes(),(Q/'nemotron9b_exec_transition_v2.py').read_bytes())
 def test_all_original_sources_and_new_closure_present(self):
  required=set(p.required_sources())
  self.assertLess(set(old.required_sources()),required)
  self.assertTrue(all(Path(x).is_file()for x in required))
  self.assertTrue({str(x)for x in R.glob('nemotron9b_package_*v26.py')}<=required)
 def test_full_physical_drain_preserved_after_history_extension(self):
  a=ast.parse((R/'nemotron9b_completed_history_drain_v9.py').read_text())
  b=ast.parse((R/'nemotron9b_package_startup_drain_v26.py').read_text())
  af=next(n for n in a.body if isinstance(n,ast.FunctionDef)and n.name=='physical_drain')
  bf=next(n for n in b.body if isinstance(n,ast.FunctionDef)and n.name=='physical_drain')
  self.assertEqual(ast.dump(ast.Module(body=af.body[1:],type_ignores=[])),ast.dump(ast.Module(body=bf.body[2:],type_ignores=[])))
  self.assertIs(p.predecessor_drain,drain.physical_drain)
 def test_new_metadata_envelopes_bound_to_exact_mini_pins(self):
  self.assertEqual(p.metadata.LIMITS,old.metadata.LIMITS)
  for path,digest in h.LARGE.items():
   with self.subTest(path=path),patch.object(p.metadata,'stable_bytes')as read:
    with self.assertRaisesRegex(RuntimeError,'exact completed'):h.text(path,'0'*64,read)
    read.assert_not_called()
  self.assertFalse(h.accepts('/tmp/unreviewed.json'))
 def test_no_native_runtime_import(self):
  self.assertFalse({'torch','numpy','vllm','neuronxcc','torch_neuronx'}&set(sys.modules))
class Storage(unittest.TestCase):
 def setUp(self):
  self.s=dict(control_root='/mnt/nvme/nemotron9b-package-fixture-control',evidence_root='/mnt/nvme/nemotron9b-package-fixture-evidence',
   work_root='/mnt/nvme/nemotron9b-package-fixture',compiler_asset_policy={},output_layout=copy.deepcopy(h.LAYOUT))
  self.used={};self.free={'root':p.ROOT_FLOOR,'nvme':p.NVME_FLOOR+p.ROOT_CAP+p.NVME_CAP};self.same=False
 def check(self,reserve=True):
  def statfs(path):
   key='root'if Path(path)==p.ROOT else'nvme'
   return types.SimpleNamespace(f_bavail=self.free[key],f_frsize=1)
  def stat(path):return types.SimpleNamespace(st_dev=1 if self.same or Path(path)==p.ROOT else 2)
  with patch.object(p,'usage',side_effect=lambda path,*a,**kw:self.used.get(str(path),0)),patch.object(p.metadata,'load',return_value={}),patch.object(p.os,'statvfs',side_effect=statfs),patch.object(p.os,'stat',side_effect=stat):
   return p.storage(self.s,reserve=reserve)
 def test_full_19gib_reserved_on_nvme_at_exact_boundary(self):
  result=self.check();self.assertEqual(result['remaining_nvme_reservation'],19*2**30)
  self.assertTrue(result['metadata_on_nvme'])
 def test_one_byte_below_either_floor_or_total_reservation_rejected(self):
  for key in ('root','nvme'):
   with self.subTest(key=key):
    self.setUp();self.free[key]-=1
    with self.assertRaises(RuntimeError):self.check()
 def test_usage_reduces_only_its_remaining_reservation(self):
  self.used[self.s['control_root']]=100;self.used[self.s['evidence_root']]=200;self.used[self.s['work_root']]=300
  self.free['nvme']-=600
  result=self.check();self.assertEqual(result['remaining_nvme_reservation'],19*2**30-600)
 def test_each_original_category_cap_still_enforced(self):
  for path,cap in [(self.s['control_root'],p.ROOT_CAP),(self.s['work_root'],p.NVME_CAP),*[(self.s['work_root']+'/'+name,cap)for name,cap in p.CAPS.items()]]:
   with self.subTest(path=path):
    self.used={path:cap+1}
    with self.assertRaisesRegex(RuntimeError,'cap|reservation'):self.check()
 def test_wrong_filesystem_or_parent_rejected(self):
  self.same=True
  with self.assertRaisesRegex(RuntimeError,'Distinct'):self.check()
  self.same=False;self.s['control_root']=str(R/'wrong-control')
  with self.assertRaisesRegex(RuntimeError,'NVMe namespace'):self.check()
 def test_layout_cannot_reduce_any_floor_or_cap(self):
  for key in ('root_floor','nvme_floor','metadata_cap','work_cap'):
   self.setUp();self.s['output_layout'][key]-=1
   with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'quotas'):self.check()
 def test_timed_floor_check_cannot_spend_floor(self):
  self.free['nvme']=p.NVME_FLOOR;self.assertEqual(self.check(False)['remaining_nvme_reservation'],0)
  self.free['nvme']-=1
  with self.assertRaises(RuntimeError):self.check(False)
if __name__=='__main__':unittest.main(verbosity=2)
