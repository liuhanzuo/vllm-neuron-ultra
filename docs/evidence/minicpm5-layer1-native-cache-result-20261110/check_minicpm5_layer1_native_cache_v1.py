"""Independent native-cache boundary checks using synthetic complete raw data."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_native_cache_results_v1 as verify
from research import minicpm5_layer1_after_kv_factorial_v1 as finished
import minicpm5_layer1_native_cache_runner_v1 as runner
import minicpm5_layer1_native_cache_worker_v1 as worker


def read(ref):
 p=Path(ref['path']);data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==ref['sha256'];return json.loads(data)

class NativeCacheChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory(prefix='minicpm-native-cache-check-');cls.root=Path(cls.tmp.name)
  hand=json.loads((Q/'continuation-handoff-20261110-v414-mini-kv-factorial-complete.json').read_text())
  spec=read(hand['factorial']['spec']);cls.plan=read(spec['plan']);cls.plan['output_root']=str(cls.root);cls.plan['layer1_native_cache']={'synthetic':True}
  cls.original={};records=[];cache=logits=taps=0
  mapping={(f['fixture'],f['step']):f for f in cls.plan['frames']}
  for fixture in cls.plan['fixtures']:
   for step in range(4):
    frame=mapping[fixture['id'],step];n=len(fixture['input_ids'])+step;ref=cls.plan['hf_frame_records'][frame['index']];old=read(ref)
    entries=[];pieces=[];offset=0
    for name,meta in old['metadata'].items():
     width=meta['source_shape'][-1];data=b'\x80\x3f'*width;sha=hashlib.sha256(data).hexdigest();old['tap_raw_sha256'][name]=sha
     entries.append(dict(name=name,offset=offset,bytes=len(data),shape=[1,1,width],sha256=sha));pieces.append(data);offset+=len(data)
    cls.original[ref['path']]=old
    folder=cls.root/'native'/('frame-%02d'%frame['index']);folder.mkdir(parents=True)
    output={};cachedata=b'\x80\x3f'*(n*128)+b'\xc0\x3f'*(n*128)
    rawlogits=b'\x00\x00\x80\x3f'*130560;rawtaps=b''.join(pieces)
    for name,filename,data in (('key','key.bf16',cachedata),('value','value.bf16',cachedata),('logits','logits.fp32',rawlogits),('taps','selected-taps.bf16',rawtaps)):
     p=folder/filename;p.write_bytes(data);output[name]=dict(path=str(p),raw_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),identity=verify.payload.file_identity(p,p.lstat()))
    row=dict(schema='minicpm5_layer1_native_cache_frame_v1',complete=True,frame=frame,native_fixture_order=fixture['id'],step=step,cache_length=n,
     original_frame=ref,original_reference=cls.plan['hf_reference'],cache_metadata={k:dict(shape=[1,2,n,128],stride=[2*n*128,n*128,128,1],offset=0,dtype='torch.bfloat16')for k in ('key','value')},
     tap_metadata=old['metadata'],tap_entries=entries,outputs=output,full_logits_raw_exact=True,all213_selected_taps_raw_exact=True,official_bf16=True,history_unmodified=True,
     full_model_qualified=False,new_speedup_claim=False)
    records.append(row);cache+=2*len(cachedata);logits+=len(rawlogits);taps+=len(rawtaps)
  cls.result=dict(schema='minicpm5_layer1_native_cache_result_v1',complete=True,rank=-1,records=records,frames=32,
   full_cache_bytes=cache,full_logits_bytes=logits,selected_tap_bytes=taps,native_provenance=cls.plan['native_provenance'],
   original_32_model_forwards=True,full_logits_raw_exact=True,all213_selected_taps_raw_exact=True,original_native_history_preserved=True,
   completed_factorial=cls.plan['layer1_native_cache'],full_model_qualified=False,new_speedup_claim=False,self_exit_claimed=False)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def validate(self,value=None):
  value=value or self.result
  for row in value['records']:
   folder=self.root/'native'/('frame-%02d'%row['frame']['index']);folder.mkdir(parents=True,exist_ok=True);(folder/'result.json').write_text(json.dumps(row))
  with patch.object(verify.p,'read_ref',side_effect=lambda ref:copy.deepcopy(self.original[ref['path']])):
   return verify.validate(value,self.plan,lambda:None)
 def reject(self,change):
  value=copy.deepcopy(self.result);change(value)
  with self.assertRaises((RuntimeError,ValueError,AssertionError)):self.validate(value)
 def test_all32_complete_caches_logits_and6816_taps(self):
  p=self.validate();self.assertEqual((p['frames'],p['cache_tensors'],p['full_cache_bytes'],p['full_logits_bytes'],p['selected_taps']),(32,64,5894144,16711680,6816))
 def test_missing_frames_or_reordered_fixture_history_rejected(self):
  self.reject(lambda r:r['records'].pop());self.reject(lambda r:r['records'].reverse())
 def test_wrong_head_count_prefix_or_precision_rejected(self):
  self.reject(lambda r:r['records'][0]['cache_metadata']['key'].update(shape=[1,1,24,128]))
  self.reject(lambda r:r['records'][0].update(cache_length=23));self.reject(lambda r:r['records'][0]['cache_metadata']['value'].update(dtype='torch.float32'))
 def test_wrong_frame_slot_or_original_metadata_rejected(self):
  self.reject(lambda r:r['records'][0]['frame'].update(slot=0));self.reject(lambda r:r['records'][0]['original_frame'].update(sha256='0'*64))
 def test_missing_or_short_selected_tap_rejected(self):
  self.reject(lambda r:r['records'][0]['tap_entries'].pop());self.reject(lambda r:r['records'][0]['tap_entries'][1].update(bytes=2))
 def test_wrong_source_layout_rejected(self):
  self.reject(lambda r:r['records'][0]['tap_metadata']['layers.1.self_attn'].update(source_stride=[1,1,1]))
 def mutate_bytes(self,index,name,modify,pattern):
  r=copy.deepcopy(self.result);row=r['records'][index];desc=row['outputs'][name];path=Path(desc['path']);old=path.read_bytes()
  try:
   new=modify(old);path.write_bytes(new);desc.update(raw_bytes=len(new),sha256=hashlib.sha256(new).hexdigest(),identity=verify.payload.file_identity(path,path.lstat()))
   with self.assertRaisesRegex((RuntimeError,ValueError),pattern):self.validate(r)
  finally:
   path.write_bytes(old);self.result['records'][index]['outputs'][name]['identity']=verify.payload.file_identity(path,path.lstat())
 def test_recomputed_hash_cannot_hide_changed_cache_prefix(self):
  self.mutate_bytes(1,'key',lambda x:b'\x00\x3f'+x[2:],'previous native cache bytes')
 def test_nonfinite_cache_rejected(self):
  self.mutate_bytes(0,'value',lambda x:b'\x80\x7f'+x[2:],'finite')
 def test_recomputed_hash_cannot_hide_wrong_logits(self):
  self.mutate_bytes(0,'logits',lambda x:b'\x01'+x[1:],'FP32 logits')
 def test_changed_selected_tap_rejected(self):
  self.mutate_bytes(0,'taps',lambda x:b'\x00\x3f'+x[2:],'selected tap independently')
 def test_guard_denial_precedes_native_import(self):
  class Denied:
   def guard(self,*args):raise RuntimeError('busy')
  with patch.object(worker.driver,'run')as native,self.assertRaisesRegex(RuntimeError,'busy'):worker.run(Denied())
  native.assert_not_called()
 def test_actual_completed_factorial_and_audit_required(self):
  h=json.loads((Q/'continuation-handoff-20261110-v414-mini-kv-factorial-complete.json').read_text())['factorial']
  b=dict(schema='minicpm5_layer1_completed_kv_factorial_v1',model_exit=h['actual_exit'],audit_exit=h['independent_audit_actual_exit'])
  got=finished.completed(b,read);self.assertEqual(len(got['owners']),5)
  for target in (b['model_exit'],b['audit_exit']):
   def wrong(ref):
    v=read(ref)
    if ref==target:v['actual_exit_code']=1
    return v
   with self.assertRaises(RuntimeError):finished.completed(b,wrong)
 def test_native_environment_has_two_threads_and_no_tp_or_neuron_route(self):
  env=runner.env('a'*32,29625)
  self.assertEqual(env['OMP_NUM_THREADS'],env['MKL_NUM_THREADS']);self.assertEqual(env['OMP_NUM_THREADS'],'2')
  self.assertFalse(set(env)&{'RANK','WORLD_SIZE','NXD_CPU_MODE','NEURON_RT_VISIBLE_CORES','PYTHONPATH'})
  with self.assertRaises(RuntimeError):runner.env('a'*32,29625,0)
 def test_drain_requires_native_child_gone(self):
  spec=dict(output_root=str(self.root),unit='synthetic')
  (self.root/'native-owner.json').write_text(json.dumps(dict(pid=999,start_ticks='1')))
  props=dict(LoadState='not-found',InvocationID='',MainPID='0',ControlPID='0',ExecMainPID='0')
  with patch.object(runner.p,'properties',return_value=props),patch.object(runner.p,'gone',side_effect=RuntimeError('native live')):
   with self.assertRaisesRegex(RuntimeError,'native live'):runner.verify_owned_drain(spec,None,None,None,False)
 def test_wrong_provenance_or_qualification_claim_rejected(self):
  self.reject(lambda r:r['native_provenance'].update(dtype='torch.float32'));self.reject(lambda r:r.update(full_model_qualified=True))

if __name__=='__main__':
 assert not {'torch','numpy','transformers'}&set(sys.modules)
 unittest.main(verbosity=2)
