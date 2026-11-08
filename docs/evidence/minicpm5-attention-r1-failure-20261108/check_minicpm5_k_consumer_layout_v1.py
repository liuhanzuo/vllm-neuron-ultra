"""Regression tests for cache-to-attention K metadata, not native math proof."""
import copy,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import minicpm5_attention_k_consumer_layout_v1 as k
import minicpm5_frame1_attention_observer_v2 as a

class Boundary(unittest.TestCase):
 def pair(self,native=True):
  n=1231 if native else 2048;heads=2 if native else 1
  p=dict(shape=[1,heads,n,128],stride=[heads*n*128,128,heads*128,1],offset=0,dtype='torch.bfloat16',sha256='a'*64)
  c=copy.deepcopy(p);c['stride']=[heads*n*128,n*128,128,1]
  return p,c
 def test_native_values_same_with_cache_layout(self):
  p,c=self.pair();row=k.compare(p,c,True);self.assertTrue(k.validate(row,True)['passed']);self.assertFalse(row['layout_exact'])
 def test_values_changed_always_fail(self):
  for native in [False,True]:
   p,c=self.pair(native);c['sha256']='b'*64;row=k.compare(p,c,native);self.assertFalse(row['passed'])
   with self.assertRaises(RuntimeError):k.validate(row,native)
 def test_cpu_layout_remains_exact(self):
  p,c=self.pair(False);c['offset']=1;row=k.compare(p,c,False);self.assertFalse(row['passed'])
  with self.assertRaises(RuntimeError):k.validate(row,False)
 def test_same_metadata_passes_both(self):
  for native in [False,True]:
   p,c=self.pair(native);self.assertTrue(k.validate(k.compare(p,p,native),native)['passed'])
 def test_changed_shape_dtype_missing_hash_rejected(self):
  for field,value in [('shape',[1,2,1230,128]),('dtype','torch.float32'),('sha256','a')]:
   p,c=self.pair();c[field]=value
   with self.assertRaises(RuntimeError):k.compare(p,c,True)
 def test_invalid_layout_rejected(self):
  for field,value in [('stride',[1,1,1,1]),('offset',-1),('stride',[1,1,1,-1])]:
   p,c=self.pair();c[field]=value
   with self.assertRaises(RuntimeError):k.compare(p,c,True)
 def test_forged_pass_or_consumer_hash_rejected(self):
  for change in [lambda x:x.update(layout_exact=True),lambda x:x.update(model_arithmetic_changed=True),lambda x:x['consumer'].update(sha256='b'*64)]:
   p,c=self.pair();row=k.compare(p,c,True);change(row)
   with self.assertRaises(RuntimeError):k.validate(row,True)
 def test_replay_uses_consumer_layout_and_original_bytes(self):
  p,c=self.pair();saved=dict(name='k_post_rope',metadata=p,sha256=p['sha256'],path='/original.raw',bytes=630272)
  out=k.replay_row(saved,k.compare(p,c,True),True);self.assertEqual(out['metadata'],c);self.assertEqual(out['path'],saved['path']);self.assertEqual(saved['metadata'],p);self.assertIsNot(out['metadata'],c)
 def test_foreign_saved_producer_rejected(self):
  p,c=self.pair();saved=dict(metadata=dict(p,sha256='b'*64),sha256='b'*64)
  with self.assertRaises(RuntimeError):k.replay_row(saved,k.compare(p,c,True),True)
 def test_boundary_written_before_failed_value_check(self):
  p,c=self.pair();c['sha256']='b'*64
  with tempfile.TemporaryDirectory()as tmp:
   obs=object.__new__(a.Observer);obs.k_consumer_boundary=None;obs.guard=lambda:True;obs.values={'k_post_rope':dict(metadata=p)};obs.native=True;obs.torch=None;obs.directory=Path(tmp)
   with patch.object(a.old,'metadata',return_value=c):
    with self.assertRaises(RuntimeError):obs.observe_k_consumer(object())
   saved=json.loads((Path(tmp)/'k-consumer-boundary.json').read_text());self.assertFalse(saved['passed']);self.assertEqual(saved['consumer'],c)
 def test_exact_boundary_recorded_and_duplicate_rejected(self):
  p,c=self.pair()
  with tempfile.TemporaryDirectory()as tmp:
   obs=object.__new__(a.Observer);obs.k_consumer_boundary=None;obs.guard=lambda:True;obs.values={'k_post_rope':dict(metadata=p)};obs.native=True;obs.torch=None;obs.directory=Path(tmp)
   with patch.object(a.old,'metadata',return_value=c):
    obs.observe_k_consumer(object());self.assertEqual(obs.k_consumer_boundary['record']['consumer'],c)
    with self.assertRaises(RuntimeError):obs.observe_k_consumer(object())
 def test_observer_does_not_change_model_implementation(self):
  import ast
  before=ast.parse((R/'minicpm5_frame1_attention_observer_v1.py').read_text());after=ast.parse(Path(a.__file__).read_text())
  funcs=lambda tree:{node.name:ast.dump(node)for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
  b,c=funcs(before),funcs(after)
  for name in set(b)-{'Observer','own_input_controls'}:self.assertEqual(b[name],c[name],name)
  calls=[x for x in ast.walk(after)if isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and x.func.attr in ('eager_attention_forward','perform_prefill')]
  self.assertEqual(len(calls),2)
  for call in calls:self.assertTrue(any(isinstance(x,ast.Name)and x.id=='consumer_k'for x in call.args))
 def test_no_framework_imported(self):self.assertFalse({'torch','numpy','transformers','torch_neuronx'}&set(sys.modules))

if __name__=='__main__':unittest.main()
