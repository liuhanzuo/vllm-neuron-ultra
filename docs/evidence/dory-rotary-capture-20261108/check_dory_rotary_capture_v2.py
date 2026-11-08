"""Source and synthetic raw-file counterexamples; no native execution."""
import ast,copy,hashlib,json,struct,sys,tempfile,unittest
from pathlib import Path
Q=Path(__file__).resolve().parent;sys.path.insert(0,str(Q))
import dory_rotary_arithmetic_v2 as original
import dory_rotary_arithmetic_capture_v2 as capture
import dory_rotary_capture_runtime_v1 as runtime
import review_dory_rotary_saved_boundaries_v1 as review

def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
def write(path,value):path.write_text(json.dumps(value,allow_nan=False));return ref(path)

class Source(unittest.TestCase):
 def test_original_prepared_sources_unchanged(self):
  prep=json.loads((Q/'dory-rotary-arithmetic-source-preparation-20261108-v1.json').read_text())
  for field in ('source','checker','plan'):
   row=prep[field];self.assertEqual(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest(),row['sha256'])
 def test_unexecuted_v1_duplicate_dtype_reproduced_and_fixed(self):
  # This is the original actual metadata merge, with a representative output
  # shape from compare_position; no Torch or arithmetic is invoked here.
  comparison=dict(shape=[16,256],dtype='torch.float32',passed=True)
  with self.assertRaises(TypeError):dict(theta=10000,dtype='float32',fixture='signed_ramp',tensor='query',position=0,**comparison)
  row=dict(theta=10000,fixture='signed_ramp',tensor='query',position=0,**comparison)
  self.assertEqual(row['dtype'],'torch.float32')
  old=(Q/'dory_rotary_arithmetic_v1.py').read_text()
  expected=old.replace("rows.append(dict(theta=theta, dtype=dtype_name, fixture=case, tensor=kind,","rows.append(dict(theta=theta, fixture=case, tensor=kind,").replace("schema='dory_rotary_arithmetic_result_v1'","schema='dory_rotary_arithmetic_result_v2'")
  self.assertEqual(expected,Path(original.__file__).read_text())
 def test_no_native_imports(self):self.assertFalse({'torch','numpy','vllm','transformers'}&set(sys.modules))
 def test_arithmetic_and_original_plan_unchanged(self):
  def funcs(path):return {n.name:ast.dump(n)for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
  before,after=funcs(Path(original.__file__)),funcs(Path(capture.__file__));self.assertEqual(set(before),set(after))
  for name in before:
   if name!='run_comparison':self.assertEqual(before[name],after[name],name)
  self.assertEqual(original.plan(),capture.plan())
 def test_capture_adds_only_explicit_writes(self):
  source=Path(capture.__file__).read_text().replace('def run_comparison(*, torch, guard, emit):','def run_comparison(*, torch, guard):')
  source=source.replace("    require(callable(emit), 'Explicit saved-boundary writer required')\n",'').replace("    emit('positions', positions)\n",'').replace("                emit(f'theta{theta}_{dtype_name}_cache_selected', selected)\n",'')
  start=source.index("                    prefix = f'theta{theta}_{dtype_name}_{case}'")
  end=source.index('                    del q, k, actual, expected, a, e',start)
  source=source[:start]+source[end:]
  def run(text):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='run_comparison'))
  self.assertEqual(run(source),run(Path(original.__file__).read_text()))
 def test_denied_guard_prevents_native_access_or_output(self):
  def denied():raise RuntimeError('active predecessor')
  with tempfile.TemporaryDirectory()as tmp:
   path=Path(tmp)/'new'
   with self.assertRaisesRegex(RuntimeError,'active predecessor'):runtime.run(torch=object(),guard=denied,output=path)
   self.assertFalse(path.exists())
 def test_expected_full_shapes_and53names(self):
  rows=review.expected_rows(Path('/tmp/raw'))
  self.assertEqual(len(rows),53);self.assertEqual(sum(row['bytes']for row in rows.values()),3*2*2*6*20*256*(4+2)+2*6*256*(4+2)+6*8)
  self.assertEqual(rows['theta10000_float32_signed_ramp_candidate_query']['shape'],[6,16,256])
 def test_known_bf16_ties_even_and_signed_zero(self):
  self.assertEqual(review.pack(1+1/256,'bfloat16'),bytes.fromhex('803f'))
  self.assertEqual(review.pack(1+3/256,'bfloat16'),bytes.fromhex('823f'))
  self.assertEqual(review.pack(-0.0,'bfloat16'),bytes.fromhex('0080'))
  self.assertEqual(review.pack(0.0,'bfloat16'),bytes.fromhex('0000'))
 def test_value_equal_signed_zero_is_not_bitwise_equal(self):
  v=review.comparison(bytes.fromhex('0000'),bytes.fromhex('0080'),'bfloat16')
  self.assertEqual(v['unequal_values'],0);self.assertFalse(v['passed']);self.assertFalse(v['bitwise_equal'])
 def test_nonfinite_rejected(self):
  with self.assertRaisesRegex(ValueError,'Finite'):review.comparison(bytes.fromhex('807f'),bytes.fromhex('0000'),'bfloat16')

class SavedFiles(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);raw=cls.root/'raw';raw.mkdir();rows=review.expected_rows(raw);values={}
  values['positions']=struct.pack('<6q',*review.POSITIONS)
  # Synthetic unit coefficients intentionally test byte/review plumbing only.
  # They are not claimed to be trigonometric coefficients for any model.
  for theta in review.THETAS:
   for dtype in review.DTYPES:
    prefix=f'theta{theta}_{dtype}';values[prefix+'_cache_selected']=review.pack(1.0,dtype)*(6*256)
    for case in review.FIXTURES:
     for kind,heads in [('query',16),('key',4)]:
      base=prefix+'_'+case;inp=review.fixture_bytes(case,heads,dtype);values[base+'_input_'+kind]=inp;xs=review.decode(inp,dtype);parts=[]
      for pos in range(6):
       for head in range(heads):
        offset=(pos*heads+head)*256;left=xs[offset:offset+128];right=xs[offset+128:offset+256]
        parts.extend(review.pack(a-b,dtype)for a,b in zip(left,right));parts.extend(review.pack(b+a,dtype)for a,b in zip(left,right))
      result=b''.join(parts)
      for arm in ['candidate','reference']:values[base+'_'+arm+'_'+kind]=result
  for name,row in rows.items():
   value=values[name];Path(row['path']).write_bytes(value);row['sha256']=hashlib.sha256(value).hexdigest();assert len(value)==row['bytes']
  cls.manifest=dict(schema='dory_rotary_saved_boundaries_v1',rows=rows,tensors=53,raw_bytes=sum(map(len,values.values())),all_files_rehashed=True)
  inputs=[];caches=[];comparisons=[]
  for theta in review.THETAS:
   for dtype in review.DTYPES:
    prefix=f'theta{theta}_{dtype}';caches.append(dict(theta=theta,dtype=dtype,shape=[262144,256],selected_positions_sha256=rows[prefix+'_cache_selected']['sha256']))
    for case in review.FIXTURES:
     base=prefix+'_'+case;inputs.append(dict(theta=theta,dtype=dtype,fixture=case,input_sha256={kind:rows[base+'_input_'+kind]['sha256']for kind in ['query','key']}))
     for kind,heads in [('query',16),('key',4)]:
      data=values[base+'_candidate_'+kind];width=heads*256*(4 if dtype=='float32'else 2)
      for i,pos in enumerate(review.POSITIONS):
       v=data[i*width:(i+1)*width];digest=hashlib.sha256(v).hexdigest()
       comparisons.append(dict(theta=theta,fixture=case,tensor=kind,position=pos,shape=[heads,256],dtype='torch.'+dtype,all_finite=True,actual_sha256=digest,expected_sha256=digest,bitwise_equal=True,unequal_values=0,max_abs_error=0.0,mean_abs_error=0.0,max_relative_nonzero=0.0,nonzero_against_zero=0,passed=True))
  cls.captured=dict(comparisons=comparisons,input_cases=inputs,cache_proofs=caches,passed=True,native_reference_qualified=False,new_speedup_claim=False)
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def setUp(self):self.manifest=copy.deepcopy(self.__class__.manifest);self.captured=copy.deepcopy(self.__class__.captured)
 def result(self,**changes):
  value=dict(schema='dory_rotary_control_capture_result_v1',complete=True,control=write(self.root/'control.json',self.captured),capture=write(self.root/'capture.json',self.captured),manifest=write(self.root/'raw-manifest.json',self.manifest),
   observer_invariant=True,saved_tensors=53,original_comparisons=96,captured_comparisons=96,unique_position_comparisons=96,executed_position_comparisons=192,original_arithmetic_passed=True,captured_arithmetic_passed=True,passed=True,native_reference_qualified=False,new_speedup_claim=False)
  value.update(changes);return write(self.root/'result.json',value)
 def test_complete_saved_matrix_and_independent_arithmetic(self):
  result=review.review(self.result(),lambda:None)
  self.assertEqual(len(result['comparisons']),96);self.assertEqual(len(result['independent_native_style_replay']),16)
  self.assertTrue(result['all_reference_outputs_reproduced']);self.assertTrue(result['all_candidate_outputs_reproduced']);self.assertFalse(result['native_reference_qualified'])
 def test_changed_raw_bytes_rejected(self):
  path=Path(self.manifest['rows']['positions']['path']);old=path.read_bytes();path.write_bytes(b'x'+old[1:])
  try:
   with self.assertRaisesRegex(ValueError,'bytes changed'):review.review(self.result(),lambda:None)
  finally:path.write_bytes(old)
 def test_omitted_boundary_rejected(self):
  self.manifest['rows'].pop('positions')
  with self.assertRaisesRegex(ValueError,'53exact'):review.review(self.result(),lambda:None)
 def test_shape_or_dtype_substitution_rejected(self):
  self.manifest['rows']['positions']['dtype']='float32'
  with self.assertRaisesRegex(ValueError,'shape/dtype/path'):review.review(self.result(),lambda:None)
 def test_false_pass_metric_rejected(self):
  self.captured['comparisons'][0]['unequal_values']=1
  with self.assertRaisesRegex(ValueError,'metrics'):review.review(self.result(),lambda:None)
 def test_false_observer_flag_rejected(self):
  with self.assertRaisesRegex(ValueError,'observer equality'):review.review(self.result(observer_invariant=False),lambda:None)
 def test_failed_arithmetic_cannot_be_promoted(self):
  self.captured['passed']=False
  with self.assertRaisesRegex(ValueError,'pass classification'):review.review(self.result(),lambda:None)
 def test_model_qualification_inferred_rejected(self):
  with self.assertRaisesRegex(ValueError,'full-model'):review.review(self.result(native_reference_qualified=True),lambda:None)
if __name__=='__main__':unittest.main(verbosity=2)
