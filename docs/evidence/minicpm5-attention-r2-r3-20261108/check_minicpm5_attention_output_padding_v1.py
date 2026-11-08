"""Actual installed RowParallelLinear source contract and observer regressions."""
import ast,copy,sys,types,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'research')]
import check_minicpm5_attention_k_integration_v1 as prior
import minicpm5_frame1_attention_observer_v3 as a
import minicpm5_frame1_attention_worker_v3 as w
import minicpm5_attention_failed_r2_history_v1 as h
prior.a=prior.inherited.a=a;prior.w=prior.inherited.w=w
class RowParallelLinear:
 def __init__(self):
  self.input_is_parallel=True;self.reduce_output=True;self.reduce_dtype='bf16';self.tensor_parallel_group=types.SimpleNamespace(size=lambda:2);self.pad=False;self.sequence_parallel_enabled=False;self.input_size=self.output_size=2048;self.input_size_per_partition=1024;self.weight=types.SimpleNamespace(shape=[2048,1024],dtype='bf16');self.bias=None
class Padding(unittest.TestCase):
 def test_unpadded_missing_pad_size_is_supported(self):
  x=RowParallelLinear();self.assertFalse(hasattr(x,'pad_size'));self.assertTrue(a.verify_cpu_output_projection(types.SimpleNamespace(bfloat16='bf16'),x))
 def test_changed_padding_rejected(self):
  x=RowParallelLinear();x.pad=True;x.pad_size=0
  with self.assertRaises(RuntimeError):a.verify_cpu_output_projection(types.SimpleNamespace(bfloat16='bf16'),x)
 def test_changed_sizes_rejected(self):
  for field in ['input_size','output_size','input_size_per_partition']:
   x=RowParallelLinear();setattr(x,field,2049)
   with self.assertRaises(RuntimeError):a.verify_cpu_output_projection(types.SimpleNamespace(bfloat16='bf16'),x)
 def test_changed_shard_or_precision_rejected(self):
  for weight in [types.SimpleNamespace(shape=[2048,1025],dtype='bf16'),types.SimpleNamespace(shape=[2048,1024],dtype='fp32')]:
   x=RowParallelLinear();x.weight=weight
   with self.assertRaises(RuntimeError):a.verify_cpu_output_projection(types.SimpleNamespace(bfloat16='bf16'),x)
 def test_changed_reduction_parallel_or_bias_rejected(self):
  for field,value in [('reduce_output',False),('input_is_parallel',False),('sequence_parallel_enabled',True),('reduce_dtype','fp32'),('bias',object())]:
   x=RowParallelLinear();setattr(x,field,value)
   with self.assertRaises(RuntimeError):a.verify_cpu_output_projection(types.SimpleNamespace(bfloat16='bf16'),x)
 def test_installed_source_only_creates_pad_size_when_enabled(self):
  p=Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/parallel_layers/layers.py');tree=ast.parse(p.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='RowParallelLinear');init=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=='__init__');branches=[x for x in init.body if isinstance(x,ast.If)and ast.unparse(x.test)=='self.pad'];self.assertEqual(len(branches),1)
  assignments=[x for x in ast.walk(init)if isinstance(x,ast.Assign)and any(isinstance(t,ast.Attribute)and t.attr=='pad_size'for t in x.targets)];self.assertEqual(len(assignments),1);self.assertIn(assignments[0],branches[0].body)
 def test_r2_actual_failure_receipt_validates(self):self.assertTrue(h.validate(dict(prior_attention_failure=dict(terminal=h.TERMINAL,verifier_exit=h.EXIT)))['native_complete'])
 def test_no_framework_imported(self):self.assertFalse({'torch','numpy','transformers','torch_neuronx'}&set(sys.modules))
if __name__=='__main__':
 suite=unittest.TestSuite()
 for name in ['Profile','Source','Geometry','Upstream','Hooks','Controls']:suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(prior.inherited,name)))
 for cls in [prior.Wiring,Padding]:suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
 result=unittest.TextTestRunner().run(suite);raise SystemExit(not result.wasSuccessful())
