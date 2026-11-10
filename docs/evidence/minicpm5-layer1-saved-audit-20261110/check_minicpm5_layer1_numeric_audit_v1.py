"""Synthetic FP32 archives/numerics and original JSON only; no native model."""
import array
import collections
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_numeric_audit_v1 as audit
from research import minicpm5_layer1_logits_archive_v1 as archive
from research import minicpm5_layer1_payload_io_v1 as owned

V=archive.V
def sha(data):return hashlib.sha256(data).hexdigest()
class FloatStorage:pass
class Storage:
 def __init__(self,key,count):self.key,self.count=key,count
class Tensor:
 def __init__(self,key,shape):
  self.key,self.shape=key,shape
 def __reduce__(self):
  product=1;stride=[]
  for size in reversed(self.shape):stride.insert(0,product);product*=size
  return archive.original.rebuild,(Storage(self.key,product),0,tuple(self.shape),tuple(stride),False,collections.OrderedDict())
class Encoder(pickle.Pickler):
 def persistent_id(self,value):
  if isinstance(value,Storage):return ('storage',FloatStorage,value.key,'cpu',value.count)
def archive_bytes(data,storages,change_pickle=None,change_entries=None):
 out=io.BytesIO();Encoder(out,protocol=2).dump(data);metadata=out.getvalue()
 old=('c'+archive.original.__name__+'\nrebuild\n').encode()
 assert old in metadata
 metadata=metadata.replace(old,b'ctorch._utils\n_rebuild_tensor_v2\n')
 old=('c'+FloatStorage.__module__+'\nFloatStorage\n').encode();assert old in metadata
 metadata=metadata.replace(old,b'ctorch\nFloatStorage\n')
 if change_pickle:metadata=change_pickle(metadata)
 entries={'synthetic/data.pkl':metadata,'synthetic/byteorder':b'little','synthetic/version':b'3\n'}
 entries.update({'synthetic/data/'+key:values.tobytes()for key,values in storages.items()})
 if change_entries:change_entries(entries)
 out=io.BytesIO()
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED)as z:
  for name,value in entries.items():z.writestr(name,value)
 return out.getvalue()
def fixture_values():
 reference=array.array('f',[-10.])*V;reference[7]=5.;reference[8]=2.
 bad=array.array('f',reference);bad[7]=1.;bad[8]=6.
 return reference,bad

class ArchiveTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.values,_=fixture_values()
  cls.data=archive_bytes({'output':Tensor('0',[1,1,V])},{'0':cls.values})
 def test_restricted_original_pytorch_style_pickle_and_complete_values(self):
  calls=[]
  with archive.Archive(self.data,3*2**20,lambda:calls.append(1))as source:
   values,digest=source.tensor(source.data['output'],[1,1,V])
  self.assertEqual(values,self.values);self.assertEqual(digest,sha(self.values.tobytes()))
  self.assertGreaterEqual(len(calls),5)
 def test_double_dtype_or_unapproved_callable_rejected(self):
  for edit in (lambda b:b.replace(b'ctorch\nFloatStorage\n',b'ctorch\nDoubleStorage\n'),
   lambda b:b.replace(b'ctorch._utils\n_rebuild_tensor_v2\n',b'cos\nsystem\n')):
   data=archive_bytes({'output':Tensor('0',[1,1,V])},{'0':self.values},change_pickle=edit)
   with self.assertRaises(AssertionError):archive.Archive(data,3*2**20,lambda:None)
 def test_trailing_pickle_or_wrong_byteorder_rejected(self):
  data=archive_bytes({'output':Tensor('0',[1,1,V])},{'0':self.values},change_pickle=lambda b:b+b'N.')
  with self.assertRaisesRegex(ValueError,'trailing'):archive.Archive(data,3*2**20,lambda:None)
  data=archive_bytes({'output':Tensor('0',[1,1,V])},{'0':self.values},
   change_entries=lambda e:e.update({'synthetic/byteorder':b'big'}))
  with self.assertRaisesRegex(ValueError,'endian'):archive.Archive(data,3*2**20,lambda:None)
 def test_nonfinite_truncated_or_changed_tensor_geometry_rejected(self):
  bad=array.array('f',self.values);bad[0]=float('nan')
  data=archive_bytes({'output':Tensor('0',[1,1,V])},{'0':bad})
  with archive.Archive(data,3*2**20,lambda:None)as source:
   with self.assertRaisesRegex(ValueError,'finite'):source.tensor(source.data['output'],[1,1,V])
  data=archive_bytes({'output':Tensor('0',[1,1,V])},{'0':self.values},
   change_entries=lambda e:e.update({'synthetic/data/0':e['synthetic/data/0'][:-4]}))
  with archive.Archive(data,3*2**20,lambda:None)as source:
   with self.assertRaisesRegex(ValueError,'Every FP32'):source.tensor(source.data['output'],[1,1,V])
  with archive.Archive(self.data,3*2**20,lambda:None)as source:
   desc=copy.deepcopy(source.data['output']);desc['offset']=1
   with self.assertRaisesRegex(ValueError,'Complete original'):source.tensor(desc,[1,1,V])
 def test_duplicate_or_compressed_zip_entries_rejected(self):
  for mode in ('duplicate','compressed'):
   out=io.BytesIO()
   with zipfile.ZipFile(io.BytesIO(self.data))as old,zipfile.ZipFile(out,'w')as new:
    for entry in old.infolist():
     new.writestr(entry.filename,old.read(entry.filename),compress_type=zipfile.ZIP_DEFLATED if mode=='compressed'else zipfile.ZIP_STORED)
    if mode=='duplicate':new.writestr('synthetic/version',b'3')
   with self.assertRaises(ValueError):archive.Archive(out.getvalue(),3*2**20,lambda:None)
 def test_guard_denial_precedes_archive_parse(self):
  def denied():raise RuntimeError('no serial admission')
  with patch.object(archive.zipfile,'ZipFile')as parse:
   with self.assertRaisesRegex(RuntimeError,'no serial admission'):archive.Archive(self.data,3*2**20,denied)
   parse.assert_not_called()
 def test_guarded_file_read_verifies_original_frozen_identity_and_sha(self):
  with tempfile.TemporaryDirectory(prefix='mini-synthetic-logits-')as d:
   path=Path(d)/'output.pt';path.write_bytes(self.data);identity=owned.file_identity(path,path.lstat())
   ref=dict(path=str(path),bytes=len(self.data),sha256=sha(self.data))
   self.assertEqual(archive.read_exact(ref,path,3*2**20,lambda:None,identity),self.data)
   with self.assertRaises(ValueError):archive.read_exact(dict(ref,sha256='0'*64),path,3*2**20,lambda:None,identity)
   with self.assertRaises(ValueError):archive.read_exact(ref,path,3*2**20,lambda:None,dict(identity,inode=identity['inode']+1))
 def test_missing_file_guard_midread_failure_symlink_and_bad_cap_rejected(self):
  with tempfile.TemporaryDirectory(prefix='mini-synthetic-logits-')as d:
   path=Path(d)/'output.pt';path.write_bytes(self.data)
   ref=dict(path=str(path),bytes=len(self.data),sha256=sha(self.data));calls=[0]
   def guard():
    calls[0]+=1
    if calls[0]==3:raise RuntimeError('lease lost')
   with patch.object(archive.os,'close',wraps=archive.os.close)as close:
    with self.assertRaisesRegex(RuntimeError,'lease lost'):archive.read_exact(ref,path,3*2**20,guard)
    self.assertEqual(close.call_count,1)
   with self.assertRaisesRegex(ValueError,'bounds'):archive.read_exact(ref,path,64*2**20,lambda:None)
   target=path.parent/'target.pt';path.rename(target);path.symlink_to(target)
   with self.assertRaisesRegex(ValueError,'Canonical'):archive.read_exact(ref,path,3*2**20,lambda:None)

class NumericTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.good,cls.bad=fixture_values()
  cls.plan=json.loads((Q/'minicpm5-layer1-runner-source-plan-20261110-v1.json').read_text())
 def test_float64_original_math_has_known_identical_and_shift_properties(self):
  same=archive.metrics(self.good,self.good)
  self.assertAlmostEqual(same['cosine'],1.,places=12);self.assertEqual(same['kl_reference_to_neuron'],0.)
  self.assertEqual(same['max_abs_error'],0.);self.assertTrue(same['argmax_match']);self.assertEqual(archive.failed(same),[])
  shifted=array.array('f',(x+1 for x in self.good))
  m=archive.metrics(shifted,self.good)
  self.assertAlmostEqual(m['kl_reference_to_neuron'],0.,places=12);self.assertEqual(m['max_abs_error'],1.)
  bad=archive.metrics(self.bad,self.good)
  self.assertIn('max_abs',archive.failed(bad));self.assertIn('high_margin_argmax',archive.failed(bad))
 def test_all_original_threshold_boundaries_and_tolerances_retained(self):
  self.assertIs(archive.metrics,archive.original.metrics);self.assertIs(archive.failed,archive.original.failed)
  self.assertEqual(audit.TOLERANCE,dict(cosine=3e-6,kl_reference_to_neuron=2e-5,max_abs_error=2e-6,
   mean_abs_error=2e-6,reference_margin=2e-6))
  m=dict(cosine=.995,kl_reference_to_neuron=.02,max_abs_error=2.,reference_margin=.5,argmax_match=False)
  self.assertEqual(archive.failed(m),[])
  for field,value,wanted in (('cosine',.99499,'cosine'),('kl_reference_to_neuron',.02001,'KL'),
   ('max_abs_error',2.0001,'max_abs'),('reference_margin',.50001,'high_margin_argmax')):
   self.assertIn(wanted,archive.failed(dict(m,**{field:value})))
 def test_both_rank_reference_witnesses_require_every32_frames_exactly(self):
  hashes={(f['fixture'],f['step']):'a'*64 for f in self.plan['frames']}
  witnesses=[dict(frame=f,original_logits_raw_exact=True,hf_reference_logits_sha256='a'*64)for f in self.plan['frames']]
  ranks=[dict(native32_evidence=copy.deepcopy(witnesses))for _ in range(2)]
  audit.check_reference_witnesses(ranks,self.plan,hashes)
  for action in ('missing','duplicate','permuted','hash','false'):
   bad=copy.deepcopy(ranks);rows=bad[1]['native32_evidence']
   if action=='missing':rows.pop()
   if action=='duplicate':rows[-1]=copy.deepcopy(rows[0])
   if action=='permuted':rows[0],rows[1]=rows[1],rows[0]
   if action=='hash':rows[0]['hf_reference_logits_sha256']='b'*64
   if action=='false':rows[0]['original_logits_raw_exact']=False
   with self.subTest(action=action),self.assertRaises(ValueError):audit.check_reference_witnesses(bad,self.plan,hashes)
 def pair(self,values):
  frame=self.plan['frames'][0];m=archive.metrics(values,self.good)
  mapping=dict(frame=frame,physical_row=0,current_position=23)
  call=dict(physical_rows=[mapping],HF_metrics=[dict(mapping,metrics=m,failed_criteria=archive.failed(m))])
  return [call,copy.deepcopy(call)],[values,array.array('f',values)],[sha(values.tobytes())]*2,{(frame['fixture'],frame['step']):self.good}
 def test_independent_frame_scores_keep_failure_instead_of_qualifying_model(self):
  calls,arrays,hashes,hf=self.pair(self.bad)
  out=audit.score_pair(calls,arrays,hashes,hf,{k:0. for k in audit.TOLERANCE},lambda:None)
  self.assertEqual(len(out),1);self.assertIn('high_margin_argmax',out[0]['failed_criteria'])
 def test_replica_current_position_missing_frame_and_metric_tampering_rejected(self):
  for action in ('replica','position','missing','metric','criterion','top1'):
   calls,arrays,hashes,hf=self.pair(self.bad)
   if action=='replica':arrays[1][0]+=1
   else:
    for call in calls:
     if action=='position':call['HF_metrics'][0]['current_position']+=1
     if action=='missing':call['HF_metrics']=[]
     if action=='metric':call['HF_metrics'][0]['metrics']['max_abs_error']=0.
     if action=='criterion':call['HF_metrics'][0]['failed_criteria']=[]
     if action=='top1':call['HF_metrics'][0]['metrics']['neuron_top1']=99
   with self.subTest(action=action),self.assertRaises(ValueError):
    audit.score_pair(calls,arrays,hashes,hf,{k:0. for k in audit.TOLERANCE},lambda:None)
 def test_actual84_metadata_mappings_are_compatible_without_tensor_reads(self):
  checked=0
  for ref in self.plan['layer1_observation']['original_calls']:
   data=Path(ref['path']).read_bytes();self.assertEqual(sha(data),ref['sha256']);call=json.loads(data)
   self.assertEqual([x['frame']for x in call['physical_rows']],[x['frame']for x in call['HF_metrics']])
   self.assertEqual(call['output_metadata']['shape'],[len(call['physical_rows']),1,V]);checked+=1
  self.assertEqual(checked,84)
 def test_numeric_failure_prevents_any_cache_read_and_cache_failure_prevents_completion(self):
  with patch.object(audit,'verify_numeric_trajectory',side_effect=ValueError('numeric failure')),\
    patch.object(audit.cache_io,'audit_trajectory')as cache:
   with self.assertRaisesRegex(ValueError,'numeric failure'):audit.audit_collection({},[],lambda *a:None,lambda:None)
   cache.assert_not_called()
  with patch.object(audit,'verify_numeric_trajectory',return_value=dict(complete=True,candidate_metric_frames=96)),\
    patch.object(audit.cache_io,'audit_trajectory',side_effect=ValueError('cache failure')):
   with self.assertRaisesRegex(ValueError,'cache failure'):audit.audit_collection({},[],lambda *a:None,lambda:None)
 def test_complete_combination_still_cannot_claim_runtime_model_or_causality(self):
  with patch.object(audit,'verify_numeric_trajectory',return_value=dict(complete=True,candidate_metric_frames=96)),\
    patch.object(audit.cache_io,'audit_trajectory',return_value=dict(complete=True,calls=84,full_cache_records=168)):
   proof=audit.audit_collection({},[],lambda *a:None,lambda:None)
  self.assertTrue(proof['complete']);self.assertTrue(proof['requires_separate_runtime_admission'])
  for key in ('model_reexecuted','attention_causality_proven','full_model_qualified','new_speedup_claim'):self.assertFalse(proof[key])
 def test_no_native_runtime_imported(self):
  self.assertFalse({'torch','numpy','torch_neuronx','neuronxcc','transformers','vllm'}&set(sys.modules))



class SyntheticTrajectory:
 def __init__(self,root):
  self.plan=json.loads((Q/'minicpm5-layer1-runner-source-plan-20261110-v1.json').read_text())
  self.plan['output_root']=str(root);self.ranks=[];self.documents={};self.arrays={};self.hashes={}
  hf_rows=[];storages={}
  for index,fixture in enumerate(self.plan['fixtures']):
   combined=array.array('f')
   for step in range(4):
    values,_=fixture_values()
    shift=(index*4+step)/64.
    values=array.array('f',(x+shift for x in values))
    self.arrays[fixture['id'],step]=values;self.hashes[fixture['id'],step]=sha(values.tobytes());combined.extend(values)
   storages[str(index)]=combined
   hf_rows.append(dict(id=fixture['id'],input_ids=fixture['input_ids'],
    continuation_ids=fixture['continuation_ids'],logits=Tensor(str(index),[4,V])))
  data=archive_bytes(hf_rows,storages);path=root/'synthetic-hf-reference.pt';path.write_bytes(data)
  identity=owned.file_identity(path,path.lstat());self.plan['hf_reference']=dict(path=str(path),sha256=sha(data),identity=identity)
  self.plan['inputs'][str(path)]=dict(sha256=sha(data),identity=identity)
  originals=[]
  for item in self.plan['layer1_observation']['original_calls'][:14]:
   raw=Path(item['path']).read_bytes();assert sha(raw)==item['sha256'];originals.append(json.loads(raw))
  base_calls={}
  for ai,arm in enumerate(audit.ARMS):
   for ci,original in enumerate(originals):
    row=copy.deepcopy(original);row['arm']=arm;row['rank']=0;arrays=[]
    physical=sorted(row['physical_rows'],key=lambda x:x['physical_row'])
    values=array.array('f');scores=[]
    for mapping in physical:
     frame=mapping['frame'];expected=self.arrays[frame['fixture'],frame['step']]
     got=array.array('f',expected)
     if frame['index']<ai+1:got[7]-=4.;got[8]+=4.
     values.extend(got)
    for mapping in row['physical_rows']:
     frame=mapping['frame'];index=mapping['physical_row']
     m=archive.metrics(values[index*V:(index+1)*V],self.arrays[frame['fixture'],frame['step']])
     scores.append(dict(mapping,metrics=m,failed_criteria=archive.failed(m)))
    row['HF_metrics']=scores;row['output_metadata'].update(shape=[len(physical),1,V],dtype='torch.float32',sha256=sha(values.tobytes()))
    payload=archive_bytes(dict(output=Tensor('0',[len(physical),1,V])),{'0':values})
    taps=[];entries=[];offset=0;size=len(physical)*4096
    for k,name in enumerate(audit.results.SELECTED_TAPS):
     raw=bytes([k+1])*size;row['stage_sha256'][name]=sha(raw)
     entries.append(dict(name=name,offset=offset,raw_bytes=size,dtype='torch.bfloat16',
      shape=[len(physical),1,2048],frame_indices=[x['frame']['index']for x in physical],
      sha256=sha(raw),source_metadata=row['stage_metadata'][name]))
     taps.append(raw);offset+=size
    base_calls[ai,ci]=(row,payload,b''.join(taps),entries)
  for rank in (0,1):
   rankrow=dict(rank=rank,arm_reports=[],summaries={},native32_evidence=[
    dict(frame=f,original_logits_raw_exact=True,hf_reference_logits_sha256=self.hashes[f['fixture'],f['step']])
    for f in self.plan['frames']])
   for ai,arm in enumerate(audit.ARMS):
    folder=root/('cpu-rank'+str(rank))/arm;folder.mkdir(parents=True)
    report=dict(reports=[])
    for ci in range(14):
     row,payload,taps,entries=base_calls[ai,ci];row=copy.deepcopy(row);row['rank']=rank
     path=folder/('call-%02d'%ci);path.mkdir()
     output=path/'output.pt';output.write_bytes(payload)
     row['payload']=dict(path=str(output),bytes=len(payload),sha256=sha(payload))
     selected=path/'selected-taps.bf16';selected.write_bytes(taps)
     row['selected_tap_bundle']=dict(path=str(selected),sha256=sha(taps),raw_bytes=len(taps),
      identity=owned.file_identity(selected,selected.lstat()),encoding='raw_bf16_le',entries=copy.deepcopy(entries))
     self.documents[str(path/'result.json')]=row
     report['reports'].append(dict(call=ci,report=str(path/'result.json'),sha256='a'*64))
    self.documents[str(folder/'result.json')]=report
    rankrow['arm_reports'].append(dict(arm=arm,report=str(folder/'result.json'),sha256='a'*64))
    rankrow['summaries'][arm]=dict(failed_frames=list(range(ai+1)),argmax_match_fraction=(31-ai)/32,passed=False)
   self.ranks.append(rankrow)
 def read_json(self,path,digest):
  assert digest=='a'*64
  return self.documents[path]

class CompleteSyntheticTrajectory(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory(prefix='mini-full-logits-synthetic-')
  cls.fixture=SyntheticTrajectory(Path(cls.tmp.name))
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def validate(self):
  f=self.fixture
  # Original metadata-validation implementation is tested separately with real
  # frozen metadata. Generated logits intentionally cannot match original model.
  with patch.object(audit.results,'validate_rank_pair',return_value=dict(synthetic_metadata_boundary=True)):
   return audit.verify_numeric_trajectory(f.plan,f.ranks,f.read_json,lambda:None)
 def test_all84_archives_both_ranks_full32_references_and96_metric_frames(self):
  proof=self.validate()
  self.assertTrue(proof['complete'])
  self.assertEqual((proof['logits_payload_count'],proof['candidate_metric_frames'],
   proof['official_reference_frames'],proof['original_selected_tap_bytes']),(84,96,32,3145728))
  self.assertTrue(proof['replicated_ranks_verified_and_counted_once'])
  for ai,arm in enumerate(audit.ARMS):
   summary=proof['summaries'][arm]
   self.assertEqual(summary,dict(passed_frames=31-ai,top1_matches=31-ai,failed_frames=list(range(ai+1))))
  self.assertFalse(proof['full_model_qualified']);self.assertFalse(proof['new_speedup_claim'])
 def test_missing_rank_reference_record_blocks_complete_trajectory(self):
  f=self.fixture;last=f.ranks[1]['native32_evidence'].pop()
  try:
   with self.assertRaisesRegex(ValueError,'missing, repeated'):self.validate()
  finally:f.ranks[1]['native32_evidence'].append(last)
 def test_recorded_summary_cannot_hide_independently_recomputed_failure(self):
  f=self.fixture;summary=f.ranks[1]['summaries'][audit.ARMS[0]];old=summary['failed_frames']
  summary['failed_frames']=[]
  try:
   with self.assertRaisesRegex(ValueError,'summaries'):self.validate()
  finally:summary['failed_frames']=old

if __name__=='__main__':unittest.main(verbosity=2)
