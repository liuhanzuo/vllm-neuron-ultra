"""Actual metadata and tiny storage-interface fixtures; no torch or weight data."""
import copy,hashlib,sys,types,unittest
from unittest.mock import patch
import dory_reference_loader_bridge_v1 as b

class Bridge(unittest.TestCase):
 def fixture(self):
  rows=[dict(name='embedding',shape=[2,3],start=0,end=12),dict(name='query',shape=[2,2],start=12,end=20),dict(name='learned_scale',shape=[2],start=20,end=24)]
  contract=dict(rows=rows,tensor_count=3,full_sha256='a'*64,file_bytes=100,data_bytes=24)
  weights={};hashes=[]
  for i,row in enumerate(rows):
   size=row['end']-row['start'];shape=row['shape'];stride=[shape[1],1]if len(shape)==2 else[1]
   weights[row['name']]=dict(shape=shape,dtype='torch.bfloat16',device='cpu',stride=stride,offset=0,elements=size//2,element_bytes=2,contiguous=True,data_pointer=100+100*i,storage_pointer=100+100*i,storage_bytes=size,requires_grad=False)
   hashes.append(dict(name=row['name'],shape=shape,dtype='BF16',bytes=size,sha256='b'*64))
  proof=dict(full_file_hash_verified=True,full_sha256='a'*64,file_bytes_read=100,weight_bytes=24,tensors=hashes)
  return weights,copy.deepcopy(weights),contract,proof
 def run_fixture(self,w,a,c,p):return b.verify_bridge(w,a,c,p,lambda x:x,lambda:None)
 def test_complete_detached_aliases(self):
  v=self.run_fixture(*self.fixture());self.assertEqual(v['weight_bytes'],24);self.assertEqual(v['tensor_count'],3);self.assertTrue(v['all_detached_weights_share_original_storage']);self.assertFalse(v['model_forward_executed'])
 def test_missing_or_extra_physical_weight(self):
  for mode in ('missing','extra'):
   w,a,c,p=self.fixture()
   if mode=='missing':a.pop('learned_scale')
   else:a['other']=a['query']
   with self.assertRaises(ValueError):self.run_fixture(w,a,c,p)
 def test_clone_or_convert_rejected(self):
  for key,value in [('data_pointer',777),('storage_pointer',777),('dtype','torch.float32'),('stride',[1,2]),('offset',1)]:
   with self.subTest(key=key):
    w,a,c,p=self.fixture();a['query'][key]=value
    with self.assertRaises(ValueError):self.run_fixture(w,a,c,p)
 def test_shared_physical_buffers_rejected(self):
  w,a,c,p=self.fixture()
  for values in (w,a):values['query'].update(data_pointer=100,storage_pointer=100)
  with self.assertRaises(ValueError):self.run_fixture(w,a,c,p)
 def test_both_sides_wrong_storage_rejected(self):
  for key,value in [('device','cuda'),('dtype','torch.float32'),('storage_bytes',100),('requires_grad',True),('contiguous',False),('element_bytes',4),('offset',1)]:
   with self.subTest(key=key):
    w,a,c,p=self.fixture()
    for values in (w,a):values['query'][key]=value
    with self.assertRaises(ValueError):self.run_fixture(w,a,c,p)
 def test_unverified_or_partial_checkpoint_rejected(self):
  for key,value in [('full_file_hash_verified',False),('full_sha256','c'*64),('file_bytes_read',99),('weight_bytes',22)]:
   w,a,c,p=self.fixture();p[key]=value
   with self.assertRaises(ValueError):self.run_fixture(w,a,c,p)
 def test_missing_duplicate_or_corrupt_tensor_proof(self):
  for mode in ('missing','duplicate','hash','shape','bytes'):
   w,a,c,p=self.fixture()
   if mode=='missing':p['tensors'].pop()
   elif mode=='duplicate':p['tensors'].append(p['tensors'][0])
   else:p['tensors'][0][{'hash':'sha256','shape':'shape','bytes':'bytes'}[mode]]={'hash':'x'*64,'shape':[6],'bytes':10}[mode]
   with self.assertRaises(ValueError):self.run_fixture(w,a,c,p)
 def test_guard_failure_propagates_before_inspection(self):
  w,a,c,p=self.fixture();calls=[]
  def stop():raise InterruptedError('expired serial lease')
  with self.assertRaises(InterruptedError):b.verify_bridge(w,a,c,p,lambda x:calls.append(x),stop)
  self.assertEqual(calls,[])
 def test_guard_checked_during_each_tensor(self):
  w,a,c,p=self.fixture();calls=[]
  b.verify_bridge(w,a,c,p,lambda x:x,lambda:calls.append(True))
  self.assertGreaterEqual(len(calls),5)

class Sources(unittest.TestCase):
 def test_actual_original507metadata_and_candidate_pins(self):
  raw,contract_raw,c=b.source_bundle();self.assertEqual(hashlib.sha256(raw).hexdigest(),b.CANDIDATE_SHA);self.assertEqual(hashlib.sha256(contract_raw).hexdigest(),b.loader.CONTRACT_SHA)
  self.assertEqual(c['tensor_count'],507);self.assertEqual(c['parameters'],3975229440);self.assertEqual(c['data_bytes'],7950458880)
  self.assertEqual(c['config']['n_recurrent_loops'],2);self.assertEqual(c['config']['recurrent_kv_cache_mode'],'per_loop')
  names={r['name']for r in c['rows']};self.assertIn('lm_head.logit_scale',names);self.assertTrue(any(n.endswith('.sqk')for n in names));self.assertTrue(any(n.endswith('.suv_gate')for n in names))
 def test_missing_runtime_rejected_before_source_import(self):
  with patch.object(b,'source_bundle')as source:
   with self.assertRaises(ValueError):b.import_original(object(),lambda:None)
   source.assert_not_called()
 def test_existing_module_never_replaced(self):
  fake=types.ModuleType('torch');existing=types.ModuleType('dory_contract')
  with patch.dict(sys.modules,{'torch':fake,'dory_contract':existing}),patch.object(b,'source_bundle')as source:
   with self.assertRaises(ValueError):b.import_original(fake,lambda:None)
   self.assertIs(sys.modules['dory_contract'],existing);source.assert_not_called()
 def test_failed_import_restores_module_namespace(self):
  fake=types.ModuleType('torch');self.assertNotIn('dory_contract',sys.modules);self.assertNotIn(b.MODULE_NAME,sys.modules)
  with patch.dict(sys.modules,{'torch':fake}),patch.object(b,'source_bundle',return_value=(b"raise RuntimeError('fixture import failure')",b'pass',{})):
   with self.assertRaisesRegex(RuntimeError,'fixture import failure'):b.import_original(fake,lambda:None)
   self.assertNotIn('dory_contract',sys.modules);self.assertNotIn(b.MODULE_NAME,sys.modules)
 def test_expired_lease_prevents_loading(self):
  def stop():raise InterruptedError('expired')
  with patch.object(b.loader,'load_checkpoint')as load,patch.object(b,'import_original')as imported:
   with self.assertRaises(InterruptedError):b.build_reference('/unused',{},torch=object(),guard=stop)
   load.assert_not_called();imported.assert_not_called()
 def test_no_native_framework_imported(self):self.assertFalse({'torch','numpy','transformers','vllm'}&set(sys.modules))

if __name__=='__main__':unittest.main(verbosity=2)
