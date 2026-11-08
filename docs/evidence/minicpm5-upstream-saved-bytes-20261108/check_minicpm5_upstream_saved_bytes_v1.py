"""Stdlib miniature synthetic files only; no real model payloads/native math."""
import ast,copy,hashlib,json,math,random,struct,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(Q),str(R)]
import review_minicpm5_upstream_saved_bytes_v1 as a
import minicpm5_frame1_upstream_replay_v2 as new
import minicpm5_frame1_upstream_replay_v1 as old

def number(word):return struct.unpack('<f',struct.pack('<I',word<<16))[0]
def floating_add(x,y):
 value=number(x)+number(y)
 try:bits=struct.unpack('<I',struct.pack('<f',value))[0]
 except OverflowError:return 0xff80 if value<0 else 0x7f80
 return ((bits+0x7fff+((bits>>16)&1))>>16)&0xffff

def raw(words):return struct.pack('<'+str(len(words))+'H',*words)
def metric(left,right):
 av,bv=list(map(number,left)),list(map(number,right));delta=[abs(x-y)for x,y in zip(av,bv)];different=sum(x!=y for x,y in zip(left,right));den=math.sqrt(sum(x*x for x in av))*math.sqrt(sum(y*y for y in bv))
 return dict(raw_exact=different==0,different_elements=different,compared_elements=len(left),nonfinite_elements=0,max_abs=max(delta),mean_abs=sum(delta)/len(delta),cosine=sum(x*y for x,y in zip(av,bv))/den if den else None)
def scopes(left,right):return dict(full=metric(left,right),real_tokens=metric(left[:8],right[:8]),prediction=metric(left[4:8],right[4:8]))

def tensor(path,words,length,kind='observed',name='x'):
 path.parent.mkdir(parents=True,exist_ok=True);data=raw(words);path.write_bytes(data);digest=hashlib.sha256(data).hexdigest()
 value=dict(name=name,path=str(path),bytes=len(data),sha256=digest,identity=a.old.identity(path,path.stat()),metadata=dict(shape=[1,length,4],stride=[length*4,4,1],offset=0,dtype='torch.bfloat16',sha256=digest),encoding='packed_bf16_le')
 if kind=='observed':value['observed_not_recomputed']=True
 else:value.update(counterfactual_computed=True,observed_model_output=False)
 return value

class Addition(unittest.TestCase):
 def test_signed_zero(self):
  for x,y,wanted in [(0,0,0),(0x8000,0x8000,0x8000),(0,0x8000,0),(0x3f80,0xbf80,0)]:self.assertEqual(a.bf16_add(x,y),wanted)
 def test_ties_even(self):
  self.assertEqual(a.bf16_add(0x3f80,0x3b80),0x3f80);self.assertEqual(a.bf16_add(0x3f81,0x3b80),0x3f82)
 def test_subnormals_and_overflow(self):
  for x,y,wanted in [(1,1,2),(127,1,128),(0x8001,0x8001,0x8002),(0x7f7f,0x7f7f,0x7f80)]:self.assertEqual(a.bf16_add(x,y),wanted)
 def test_all_finite_words_plus_zero(self):
  for word in range(65536):
   if (word>>7)&255!=255:self.assertEqual(a.bf16_add(word,0),floating_add(word,0))
 def test_random_against_independent_float_rounding(self):
  rng=random.Random(7314);count=0
  while count<10000:
   x,y=rng.randrange(65536),rng.randrange(65536)
   if (x>>7)&255==255 or(y>>7)&255==255:continue
   self.assertEqual(a.bf16_add(x,y),floating_add(x,y));count+=1
 def test_nonfinite_rejected(self):
  for word in (0x7f80,0xff80,0x7fc1):
   with self.assertRaises(ValueError):a.bf16_add(word,0)

class TensorFiles(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory(prefix='mini-upstream-source-');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
 def fixture(self,kind='observed',words=None):
  path=self.root/'x.bf16';row=tensor(path,words or[0x3f80]*8,2,kind);return path,row
 def test_complete_file(self):
  path,row=self.fixture()
  with a.TensorFile(row,path,[1,2,4],'observed',lambda:None)as f:
   self.assertEqual(f.next_row(),(0x3f80,)*4);f.next_row();self.assertEqual(f.finish()['sha256'],row['sha256'])
 def test_incomplete_read_rejected(self):
  path,row=self.fixture()
  with a.TensorFile(row,path,[1,2,4],'observed',lambda:None)as f:
   f.next_row()
   with self.assertRaises(ValueError):f.finish()
 def test_truncated_file_rejected(self):
  path,row=self.fixture();path.write_bytes(b'xx')
  with self.assertRaises(ValueError):a.TensorFile(row,path,[1,2,4],'observed',lambda:None)
 def test_altered_same_size_bytes_rejected(self):
  path,row=self.fixture()
  with a.TensorFile(row,path,[1,2,4],'observed',lambda:None)as f:
   f.next_row();f.next_row();path.write_bytes(raw([0x4000]*8))
   with self.assertRaises(ValueError):f.finish()
 def test_nonfinite_padding_rejected(self):
  path,row=self.fixture(words=[0x3f80]*7+[0x7fc1])
  with a.TensorFile(row,path,[1,2,4],'observed',lambda:None)as f:
   f.next_row()
   with self.assertRaises(ValueError):f.next_row()
 def test_computed_cannot_claim_observed(self):
  path,row=self.fixture('computed');row['observed_model_output']=True
  with self.assertRaises(ValueError):a.TensorFile(row,path,[1,2,4],'computed',lambda:None)
 def test_computed_file_accepted(self):
  path,row=self.fixture('computed')
  with a.TensorFile(row,path,[1,2,4],'computed',lambda:None)as f:f.next_row();f.next_row();self.assertEqual(f.finish()['bytes'],16)
 def test_shape_shrink_rejected(self):
  path,row=self.fixture()
  with self.assertRaises(ValueError):a.TensorFile(row,path,[1,1,4],'observed',lambda:None)
 def test_symlink_rejected(self):
  path,row=self.fixture();link=self.root/'link.bf16';link.symlink_to(path);row['path']=str(link)
  with self.assertRaises(ValueError):a.TensorFile(row,link,[1,2,4],'observed',lambda:None)
 def test_overlapping_layout_rejected(self):
  path,row=self.fixture();row['metadata']['stride']=[1,1,1]
  with self.assertRaises(ValueError):a.TensorFile(row,path,[1,2,4],'observed',lambda:None)
 def test_false_exact_zero_distance_rejected(self):
  actual=metric([0],[0x8000]);claimed=dict(actual,raw_exact=True,different_elements=0)
  with self.assertRaises(ValueError):a.compare_metric(actual,claimed)

class SyntheticContrasts(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='mini-upstream-contrast-source-');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
  self.patches=[patch.object(a,'LENGTH',2),patch.object(a,'BUCKET',3),patch.object(a,'WIDTH',4)]
  for p in self.patches:p.start();self.addCleanup(p.stop)
  self.words={};self.reports={}
  for role in a.ROLES:
   native=role=='native';length=2 if native else 3;entry=[0x3f80]*(length*4);attention=[0x3f00 if native else 0x3e80]*(length*4);summed=[floating_add(x,y)for x,y in zip(entry,attention)]
   values=dict(layer_input=entry,inputnorm_output=entry,attention_input=entry,attention_output=attention,postnorm_input=summed,postnorm_output=summed,mlp_input=summed)
   self.words[role]=values;self.reports[role]=dict(captures=[tensor(self.root/role/(field+'.bf16'),values[field],length,name=field)for field in a.FIELDS])
 def claimed(self,label):
  cpu,native=self.words['cpu-rank0'],self.words['native'];entry=list(cpu['layer_input']);attention=list(cpu['attention_output'])
  if label in ('native_entry_only','native_entry_and_attention'):entry[:8]=native['layer_input']
  if label in ('native_attention_only','native_entry_and_attention'):attention[:8]=native['attention_output']
  summed=[floating_add(x,y)for x,y in zip(entry,attention)];post=list(summed)
  saved=tensor(self.root/'contrasts'/(label+'.postnorm.bf16'),post,3,'computed',label)
  em=dict(a.observed_row(self.reports['cpu-rank0'],'layer_input')['metadata'],sha256=hashlib.sha256(raw(entry)).hexdigest());am=dict(a.observed_row(self.reports['cpu-rank0'],'attention_output')['metadata'],sha256=hashlib.sha256(raw(attention)).hexdigest())
  return dict(saved_postnorm_output=saved,complete_shape=[1,3,4],only_real_prefix_substituted=2,cpu_padding_preserved=True,entry=em,attention=am,residual_vs_cpu=scopes(summed,cpu['postnorm_input']),postnorm_vs_cpu=scopes(post,cpu['postnorm_output']),residual_real_vs_native=metric(summed[:8],native['postnorm_input']),postnorm_real_vs_native=metric(post[:8],native['postnorm_output']))
 def test_all_three_saved_contrasts(self):
  for label in a.CONTRASTS:
   with self.subTest(label=label):
    result=a.audit_contrast(self.root,self.reports,label,self.claimed(label),lambda:None);self.assertTrue(result['whole_padding_exact']);self.assertFalse(result['norm_math_independently_reexecuted'])
 def test_independent_residuals(self):
  result=a.audit_residuals(self.root,self.reports,lambda:None);self.assertTrue(all(v['exact']for v in result.values()))
 def test_corrupt_original_residual_rejected(self):
  row=a.observed_row(self.reports['native'],'postnorm_input');path=Path(row['path']);replacement=tensor(path,[0x3f80]*8,2,name='postnorm_input');self.reports['native']['captures'][4]=replacement
  with self.assertRaises(ValueError):a.audit_residuals(self.root,self.reports,lambda:None)
 def test_false_counterfactual_metric_rejected(self):
  label=a.CONTRASTS[1];claimed=self.claimed(label);claimed['postnorm_vs_cpu']['full']['different_elements']=0
  with self.assertRaises(ValueError):a.audit_contrast(self.root,self.reports,label,claimed,lambda:None)
 def test_changed_padding_rejected_even_with_fresh_file_hash(self):
  label=a.CONTRASTS[1];claimed=self.claimed(label);row=claimed['saved_postnorm_output'];words=list(struct.unpack('<12H',Path(row['path']).read_bytes()));words[-1]=0x4000;claimed['saved_postnorm_output']=tensor(Path(row['path']),words,3,'computed',label)
  with self.assertRaises(ValueError):a.audit_contrast(self.root,self.reports,label,claimed,lambda:None)
 def test_false_complete_input_hash_rejected(self):
  label=a.CONTRASTS[1];claimed=self.claimed(label);claimed['entry']['sha256']='0'*64
  with self.assertRaises(ValueError):a.audit_contrast(self.root,self.reports,label,claimed,lambda:None)
 def test_short_prefix_rejected(self):
  label=a.CONTRASTS[1];claimed=self.claimed(label);claimed['only_real_prefix_substituted']=1
  with self.assertRaises(ValueError):a.audit_contrast(self.root,self.reports,label,claimed,lambda:None)

class MetadataAggregate(unittest.TestCase):
 def setUp(self):
  self.reports={};controls={}
  for role in a.ROLES:
   native=role=='native';length=1231 if native else 2048
   meta=dict(shape=[1,length,2048],stride=[length*2048,2048,1],offset=0,dtype='torch.bfloat16',sha256='1'*64)
   self.reports[role]=dict(schema='minicpm5_frame1_upstream_capture_v1',complete=True,native=native,rank=-1 if native else int(role[-1]),frame=dict(index=1,fixture=7,slot=1,step=0),captures=[dict(name=f,metadata=dict(meta))for f in a.FIELDS],precision_interventions=[],oproj_witnesses=[],actual_prefix_calls=1,anchor=dict(all_six_exact=True),cpu_full_anchor=dict(reference=dict(a.R3_REFS[role]),full_mlp_input_exact=True,adjacent_boundaries_exact=True,source_shape_preserved=True),history_consumed=False,downstream_layers_executed=0,full_model_qualified=False,new_speedup_claim=False)
   controls[role]={f:dict(expected=dict(meta),actual=dict(meta),raw_exact=True,layout_exact=True)for f in ('inputnorm_output','attention_input','postnorm_input','postnorm_output','mlp_input')}
  self.replay=dict(schema='minicpm5_frame1_upstream_replay_v2',frame=dict(index=1,fixture=7,slot=1,step=0),contrast_capture_plan=copy.deepcopy(a.CAPTURE_PLAN),complete=True,full_model_qualified=False,unique_final_KL_cause_proven=False,new_speedup_claim=False,controls=controls,all_own_input_controls_exact=True,passed=True,contrasts={label:{}for label in a.CONTRASTS},observed_boundaries={f:{}for f in a.FIELDS},classification='local_residual_and_postnorm_contrasts_only_not_attention_or_final_logit_cause')
 def test_success_metadata(self):self.assertTrue(a.verify_reports(self.reports,self.replay))
 def failed(self):
  control=self.replay['controls']['native']['postnorm_output'];control['actual']['sha256']='2'*64;control['raw_exact']=False
  self.replay.update(all_own_input_controls_exact=False,passed=False,contrasts={},classification='own_input_replay_failed_no_counterfactual_executed');self.replay.pop('observed_boundaries')
 def test_failed_control_preserved(self):self.failed();self.assertFalse(a.verify_reports(self.reports,self.replay))
 def test_failed_control_cannot_contrast(self):
  self.failed();self.replay['contrasts'][a.CONTRASTS[0]]={}
  with self.assertRaises(ValueError):a.verify_reports(self.reports,self.replay)
 def test_missing_observed_field(self):
  self.replay['observed_boundaries'].pop('attention_output')
  with self.assertRaises(ValueError):a.verify_reports(self.reports,self.replay)
 def test_wrong_r3_reference(self):
  self.reports['native']['cpu_full_anchor']['reference']['sha256']='0'*64
  with self.assertRaises(ValueError):a.verify_reports(self.reports,self.replay)
 def test_false_full_anchor(self):
  self.reports['cpu-rank1']['cpu_full_anchor']['full_mlp_input_exact']=False
  with self.assertRaises(ValueError):a.verify_reports(self.reports,self.replay)
 def test_wrong_frame(self):
  self.replay['frame']['step']=1
  with self.assertRaises(ValueError):a.verify_reports(self.reports,self.replay)
 def test_smaller_output_plan(self):
  self.replay['contrast_capture_plan']['shape'][1]=1231
  with self.assertRaises(ValueError):a.verify_reports(self.reports,self.replay)
 def test_false_control_classification(self):
  self.replay['controls']['native']['postnorm_output']['actual']['sha256']='2'*64
  with self.assertRaises(ValueError):a.verify_reports(self.reports,self.replay)
 def test_adjacent_boundary_mismatch(self):
  self.reports['native']['captures'][-1]['metadata']['sha256']='2'*64
  with self.assertRaises(ValueError):a.verify_reports(self.reports,self.replay)
 def test_old_metrics_only_replay_rejected(self):
  self.replay['schema']='minicpm5_frame1_upstream_replay_v1'
  with self.assertRaises(ValueError):a.verify_reports(self.reports,self.replay)
 def test_r3_saved_metadata_checked_independently(self):
  originals={ref['path']:dict(complete=True,native=role=='native',rank=-1 if role=='native'else int(role[-1]),frame=dict(index=1,fixture=7,slot=1,step=0),actual_prefix_calls=1,precision_interventions=[],anchor=dict(all_six_exact=True),captures=[dict(name='input',metadata=dict(a.observed_row(self.reports[role],'mlp_input')['metadata']))])for role,ref in a.R3_REFS.items()}
  with patch.object(a,'load_json',side_effect=lambda ref:originals[ref['path']]):
   self.assertEqual(len(a.verify_r3_anchors(self.reports,lambda:None)),3)
   originals[a.R3_REFS['native']['path']]['captures'][0]['metadata']['sha256']='3'*64
   with self.assertRaises(ValueError):a.verify_r3_anchors(self.reports,lambda:None)

class Version(unittest.TestCase):
 def test_arithmetic_helpers_unchanged(self):
  def funcs(path):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}
  previous,current=funcs(old.__file__),funcs(new.__file__)
  for name in ('restore','bind_norm','read_report','load_norm','compare','fixed_cpu_geometry','control_passed'):self.assertEqual(previous[name],current[name])
 def test_original_budget_preserved(self):
  plan=new.capture_plan();self.assertEqual(plan['contrast_bytes'],25165824);self.assertEqual(plan['all_raw_bytes'],177926144);self.assertLess(plan['all_raw_bytes'],192*2**20)
 def test_no_native_import(self):self.assertFalse(set(sys.modules)&{'torch','numpy','torch_neuronx','transformers'})
 def test_save_unknown_variant_rejected_before_io(self):
  with self.assertRaises(RuntimeError):new.save_contrast(None,None,Path('/unused'),'unreviewed')
 def test_saved_byte_metadata_schema_is_explicit(self):
  self.assertFalse(new.capture_plan()['independent_norm_arithmetic_reexecuted'])

if __name__=='__main__':unittest.main(verbosity=2)
