"""Source/byte controls only; never executes a model or reads saved tensors."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
import zlib
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_cache_codec_v1 as c
from research import minicpm5_layer1_observation_v1 as o

class Codec(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.raw=bytes(range(256))*(c.CACHE_BYTES//256)
  cls.header,cls.packed=c.encode(cls.raw)
 def test_full_first_cache_roundtrip_includes_all4224_positions(self):
  self.assertEqual(c.decode(self.header,self.packed),self.raw)
  self.assertEqual(len(self.header['pages']),66)
 def test_nonzero_repeat_has_empty_delta_but_restores_complete_cache(self):
  h,p=c.encode(self.raw,self.raw);self.assertEqual((h['pages'],p),([],b''))
  self.assertEqual(c.decode(h,p,self.raw),self.raw)
 def test_changes_in_all_slots_and_physical_padding_roundtrip(self):
  b=bytearray(self.raw)
  for slot in range(4):
   for pos in (0,1231,4095,4096,4223):
    b[(slot*4224+pos)*256+127]^=255
  changed=bytes(b);h,p=c.encode(changed,self.raw)
  self.assertEqual(c.decode(h,p,self.raw),changed)
  self.assertLess(len(p),c.MAX_PACKED_BYTES)
 def test_wrong_history_is_rejected_even_when_size_matches(self):
  h,p=c.encode(self.raw,self.raw)
  with self.assertRaisesRegex(ValueError,'prior'):c.decode(h,p,bytes(c.CACHE_BYTES))
 def test_changed_packed_or_complete_raw_hash_is_rejected(self):
  with self.assertRaises(ValueError):c.decode(self.header,self.packed[:-1]+b'!')
  h=copy.deepcopy(self.header);h['sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'reconstructed'):c.decode(h,self.packed)
 def test_first_zero_snapshot_cannot_omit_zero_pages(self):
  h,p=c.encode(bytes(c.CACHE_BYTES));h.update(pages=[],packed_bytes=0,packed_sha256=c.sha(b''))
  with self.assertRaisesRegex(ValueError,'every page'):c.decode(h,b'')
 def test_duplicate_reordered_or_out_of_range_pages_rejected(self):
  for mutate in (lambda h:h['pages'].insert(1,copy.deepcopy(h['pages'][0])),
                 lambda h:h['pages'].reverse(),lambda h:h['pages'][0].update(index=66)):
   h=copy.deepcopy(self.header);mutate(h)
   with self.assertRaises(ValueError):c.decode(h,self.packed)
 def test_wrong_offsets_or_incomplete_page_length_rejected(self):
  for k,v in (('offset',1),('raw_bytes',65535),('packed_bytes',True)):
   h=copy.deepcopy(self.header);h['pages'][0][k]=v
   with self.assertRaises(ValueError):c.decode(h,self.packed)
 def test_zlib_truncation_or_trailing_member_rejected_after_outer_rehash(self):
  first=self.header['pages'][0]['packed_bytes']
  for altered in (self.packed[:first-1],self.packed[:first]+zlib.compress(b'extra')):
   p=altered+self.packed[first:];h=copy.deepcopy(self.header);change=len(altered)-first
   h['pages'][0]['packed_bytes']=len(altered)
   for item in h['pages'][1:]:item['offset']+=change
   h.update(packed_bytes=len(p),packed_sha256=c.sha(p))
   with self.assertRaises(ValueError):c.decode(h,p)
 def test_decompression_overflow_is_rejected(self):
  bomb=zlib.compress(bytes(c.PAGE_BYTES+1));first=self.header['pages'][0]['packed_bytes']
  h=copy.deepcopy(self.header);p=bomb+self.packed[first:];change=len(bomb)-first
  h['pages'][0]['packed_bytes']=len(bomb)
  for item in h['pages'][1:]:item['offset']+=change
  h.update(packed_bytes=len(p),packed_sha256=c.sha(p))
  with self.assertRaisesRegex(ValueError,'bounded compressed'):c.decode(h,p)
 def test_short_or_extended_cache_input_rejected(self):
  for raw in (self.raw[:-1],self.raw+b'\0'):
   with self.assertRaises(ValueError):c.encode(raw)

 def test_incompressible_full_cache_retains_original_single_file_bound(self):
  import random
  raw=random.Random(527).randbytes(c.CACHE_BYTES)
  h,p=c.encode(raw);self.assertLessEqual(len(p),c.MAX_PACKED_BYTES)
  self.assertEqual(c.decode(h,p),raw)

class Observation(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  root=Path('/mnt/nvme/minicpm5-original32-layer0-full-bf16-cpu-r1-20261108-v1/cpu-rank0/strict_anchor_s0')
  cls.templates=[json.loads((root/('call-%02d'%i)/'result.json').read_text())for i in range(14)]
  cls.cache={n:bytes([i+1])*c.CACHE_BYTES for i,n in enumerate(o.CACHES)}
 def fixture(self,i=0):
  row=copy.deepcopy(self.templates[i]);rows=len(row['physical_rows'])
  taps={n:bytes([k+3])*(rows*4096)for k,n in enumerate(o.TAPS)}
  for n,raw in taps.items():row['stage_sha256'][n]=c.sha(raw)
  for k,n in zip((2,3),o.CACHES):
   for field in ('before_metadata','input_after_metadata'):row[field][k]['sha256']=c.sha(self.cache[n])
  return row,taps
 def build(self,row,taps,previous=None,original=None):
  return o.build(row,copy.deepcopy(row)if original is None else original,
     dict(path='/synthetic-original-call.json',sha256='a'*64),taps,self.cache,previous)
 def test_complete14_call_chain_keeps_every_physical_frame(self):
  previous=decoded=None;frames=[];total=0
  for i in range(14):
   row,taps=self.fixture(i);d,b,previous=self.build(row,taps,previous)
   out,decoded=o.reconstruct(d,b,decoded)
   self.assertEqual(out,taps);self.assertEqual(decoded,previous)
   frames.extend(d['physical_frame_order']);total+=len(b)
  self.assertEqual(sorted(frames),list(range(32)));self.assertLess(total,128*2**20)
 def test_complete_payload_restores_all_taps_and_both_caches(self):
  row,taps=self.fixture();d,b,n=self.build(row,taps);out,state=o.reconstruct(d,b)
  self.assertEqual(out,taps);self.assertEqual(state,n);self.assertLess(len(b),16*2**20)
 def test_every_original_field_is_required_for_noninterference(self):
  row,taps=self.fixture()
  for field in o.REPEAT_FIELDS:
   original=copy.deepcopy(row);original[field]=None
   with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Observer changed'):
    self.build(row,taps,original=original)
 def test_layer1_input_vs_post_cache_confusion_is_rejected(self):
  row,taps=self.fixture();row['input_after_metadata'][2]['sha256']='b'*64
  with self.assertRaisesRegex(ValueError,'input caches'):self.build(row,taps)
 def test_wrong_tap_cache_bytes_and_partial_collection_rejected(self):
  row,taps=self.fixture()
  for bad in ({},dict(taps,**{o.TAPS[0]:b'bad'})):
   with self.assertRaises(ValueError):self.build(row,bad)
  cache=dict(self.cache);cache[o.CACHES[0]]=bytes(c.CACHE_BYTES)
  with self.assertRaisesRegex(ValueError,'input cache'):
   o.build(row,row,dict(path='/fixture.json',sha256='a'*64),taps,cache)
 def test_rank_arm_or_call_splice_rejected(self):
  row,taps=self.fixture();d,b,n=self.build(row,taps);row,taps=self.fixture(1)
  for key,value in (('rank',1),('arm',o.ARMS[1]),('call',8)):
   prior=copy.deepcopy(n);prior[key]=value
   with self.assertRaises(ValueError):self.build(row,taps,prior)
 def test_missing_first_snapshot_rejected(self):
  row,taps=self.fixture(1)
  with self.assertRaises(ValueError):self.build(row,taps)
 def test_payload_offset_and_physical_order_tampering_rejected(self):
  row,taps=self.fixture();d,b,n=self.build(row,taps)
  for edit in (lambda x:x['entries'][1].update(offset=0),lambda x:x.update(physical_frame_order=[31])):
   bad=copy.deepcopy(d);edit(bad)
   with self.assertRaises(ValueError):o.reconstruct(bad,b)
 def test_diagnostic_flags_cannot_be_promoted(self):
  row,taps=self.fixture();d,b,n=self.build(row,taps)
  for key,value in (('full_model_qualified',True),('all_original_fields_exact',False),('new_speedup_claim',True)):
   bad=copy.deepcopy(d);bad[key]=value
   with self.assertRaises(ValueError):o.reconstruct(bad,b)

class SourceIntegration(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from prepare_minicpm5_layer1_observation_source_v1 import assemble
  cls.plan,cls.checks=assemble()
 def test_actual84_call_metadata_and_original_plan_preserved(self):
  self.assertEqual(len(self.plan['layer1_observation']['original_calls']),84)
  self.assertEqual((len(self.plan['calls']),len(self.plan['frames'])),(14,32))
  self.assertFalse(self.checks['runtime_admitted'])
  self.assertEqual(self.checks['raw_cache_bytes_without_compression'],726663168)
 def test_original_output_caps_remain(self):
  self.assertEqual(self.plan['layer1_observation']['output_caps'],dict(per_rank=128*2**20,total=256*2**20,single_file=16*2**20))
 def test_projection_and_input_science_unchanged(self):
  old=ast.parse((R/'minicpm5_original32_layer0_full_bf16_cpu_worker_v1.py').read_text())
  new=ast.parse((R/'minicpm5_original32_layer1_observation_cpu_worker_v1.py').read_text())
  for name in ('OProjPatch','selected_targets','anchor_records','compare_upstream','save_selected_taps'):
   a=next(x for x in old.body if getattr(x,'name',None)==name)
   b=next(x for x in new.body if getattr(x,'name',None)==name)
   self.assertEqual(ast.dump(a),ast.dump(b),name)
 def test_observation_export_only_after_original_forward_and_checks(self):
  tree=ast.parse((R/'minicpm5_original32_layer1_observation_cpu_worker_v1.py').read_text())
  fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='run_arm')
  forwards=[x for x in ast.walk(fn)if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='model']
  exports=[x for x in ast.walk(fn)if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='save_layer1_observation']
  self.assertEqual(len(forwards),1);self.assertEqual(len(exports),1);self.assertGreater(exports[0].lineno,forwards[0].lineno)
 def test_no_native_runtime_imported(self):
  self.assertFalse({'torch','numpy','vllm','neuronxcc','torch_neuronx'}&set(sys.modules))
 def check_plan(self,plan):
  from prepare_minicpm5_layer1_observation_source_v1 import Reader
  import minicpm5_original32_layer1_observation_cpu_worker_v1 as worker
  reader=Reader()
  def read(path,digest):
   self.assertEqual(plan['inputs'][path]['sha256'],digest)
   return reader.read(path,digest)
  return worker.validate_plan(plan,read)
 def test_missing_permuted_or_wrong_rank_call_records_rejected(self):
  for action in ('missing','permuted','rank'):
   plan=copy.deepcopy(self.plan);rows=plan['layer1_observation']['original_calls']
   if action=='missing':rows.pop()
   elif action=='permuted':rows[0],rows[1]=rows[1],rows[0]
   else:rows[0]['rank']=1
   with self.subTest(action=action),self.assertRaises((ValueError,RuntimeError,AssertionError)):self.check_plan(plan)
 def test_original_quality_precision_or_source_cannot_change(self):
  for action in ('quality','dtype','source'):
   plan=copy.deepcopy(self.plan)
   if action=='quality':plan['remaining_gates']['original32']='weakened'
   elif action=='dtype':plan['graph_calls'][0]['input_metadata'][4]['dtype']='torch.float64'
   else:plan['source_sha256'][next(iter(plan['source_sha256']))]='0'*64
   with self.subTest(action=action),self.assertRaises((ValueError,RuntimeError,AssertionError)):self.check_plan(plan)
 def test_output_cap_increase_rejected(self):
  plan=copy.deepcopy(self.plan);plan['layer1_observation']['output_caps']['total']+=1
  with self.assertRaises((ValueError,RuntimeError,AssertionError)):self.check_plan(plan)
 def test_assembly_is_source_only_and_creates_no_runtime_output(self):
  self.assertFalse(Path(self.plan['output_root']).exists())
  self.assertFalse(self.checks['native_started'])
  self.assertFalse(self.checks['model_or_tensor_payloads_read'])

if __name__=='__main__':unittest.main(verbosity=2)
