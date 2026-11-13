"""SOURCE ONLY: real observer bodies, fake tensors/SDK; no native imports or IO."""
import builtins
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import deepseek_weight_load_observer_v1 as m


class Tensor:
    layout = 'strided'
    device = 'cpu'
    dtype = 'torch.bfloat16'
    _version = 0

    def __init__(self, data=b'\x01\x80\x00\x00', shape=None):
        self.data = bytearray(data)
        self.shape = [len(data) // 2] if shape is None else shape

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda:len(self.data), data_ptr=lambda:id(self.data))

    def stride(self):return [1]
    def storage_offset(self):return 0
    def numel(self):return len(self.data) // 2
    def element_size(self):return 2
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):return self
    def reshape(self,n):
        if n != -1:raise AssertionError('only byte-view flatten permitted')
        return self
    def view(self, dtype):
        if dtype != 'uint8':raise AssertionError('numeric conversion forbidden')
        return SimpleNamespace(numpy=lambda:self.data)


TORCH = SimpleNamespace(Tensor=Tensor, strided='strided', uint8='uint8')


def ranks():
    return [{'weight':Tensor(), 'buffer':Tensor(b'\x00\x80')} for _ in range(2)]


class Tests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        calls=self.calls
        class Builder:
            tp_degree=2;start_rank_id=0;local_ranks_size=2
            def shard_checkpoint(self):
                calls.append('original_shard')
                self.returned=ranks()
                return self.returned
        class Base:
            def load_weights(self, path, *, skip=False):
                calls.append(('original_load',path,skip))
                values=self.builder.shard_checkpoint()
                self.values=values
                self.traced_model=SimpleNamespace(nxd_model=SimpleNamespace(weights=values,weight_loader=None))
                return 'unaltered_return'
        class Model(Base):
            def __init__(self):self.builder=Builder()
        self.Builder,self.Base,self.Model=Builder,Base,Model
        self.dep=patch.object(m,'_dependencies',return_value=(TORCH,Builder,Base));self.dep.start()
        self.addCleanup(self.dep.stop)
        self.original_load=Base.load_weights;self.original_shard=Builder.shard_checkpoint

    def observed(self,arm='off',callback=None):
        values=[];model=self.Model()
        with m.observe_load(self.Model,arm,{'checkpoint':'externally-pinned'},lambda:None,callback or values.append) as state:
            self.assertEqual(model.load_weights('/actual-artifact',skip=True),'unaltered_return')
            self.assertIs(model.values,model.builder.returned)
        self.assertIs(self.Base.load_weights,self.original_load)
        self.assertIs(self.Builder.shard_checkpoint,self.original_shard)
        return values[0] if values else state['report']

    def test_exact_original_call_and_return(self):
        report=self.observed()
        self.assertEqual(self.calls,[('original_load','/actual-artifact',True),'original_shard'])
        self.assertEqual(report['original_loader_calls'],1)
        self.assertEqual(report['original_shard_calls'],1)
        self.assertEqual(len(report['pre_layout_shards']),2)

    def test_all_values_hashed_without_numeric_cast(self):
        report=self.observed()
        for group in ('pre_layout_shards','loaded_cpu_views'):
            for rank in report[group]:
                row=next(x for x in rank['entries'] if x['key']=='weight')
                self.assertEqual(row['cpu_sha256'],hashlib.sha256(b'\x01\x80\x00\x00').hexdigest())
                self.assertEqual(row['cpu_bytes'],4)

    def test_callback_runs_before_original_load_returns(self):
        class Stop(Exception):pass
        def callback(report):
            self.assertIn('loaded_cpu_views',report)
            raise Stop('external layout gate remains unresolved')
        with self.assertRaises(Stop):self.observed(callback=callback)
        self.assertIs(self.Base.load_weights,self.original_load)
        self.assertIs(self.Builder.shard_checkpoint,self.original_shard)

    def test_driver_exception_preserved_and_hooks_restored(self):
        with self.assertRaisesRegex(RuntimeError,'driver failed'):
            with m.observe_load(self.Model,'off',{'x':'pinned'},lambda:None,lambda r:None):
                self.Model().load_weights('/artifact')
                raise RuntimeError('driver failed')
        self.assertIs(self.Base.load_weights,self.original_load)
        self.assertIs(self.Builder.shard_checkpoint,self.original_shard)

    def test_original_sharder_error_preserved(self):
        def fail(builder):raise RuntimeError('actual SDK sharder failure')
        self.Builder.shard_checkpoint=fail
        with self.assertRaisesRegex(RuntimeError,'actual SDK sharder failure'):
            with m.observe_load(self.Model,'off',{'x':'pinned'},lambda:None,lambda r:None):
                self.Model().load_weights('/artifact')
        self.assertIs(self.Builder.shard_checkpoint,fail)
        self.assertIs(self.Base.load_weights,self.original_load)

    def test_no_load_rejected(self):
        with self.assertRaisesRegex(ValueError,'without one full'):
            with m.observe_load(self.Model,'off',{'x':'pinned'},lambda:None,lambda r:None):pass

    def test_second_load_rejected(self):
        with self.assertRaisesRegex(ValueError,'Exactly one actual'):
            with m.observe_load(self.Model,'off',{'x':'pinned'},lambda:None,lambda r:None):
                self.Model().load_weights('/artifact');self.Model().load_weights('/artifact')

    def test_wrong_rank_geometry_rejected(self):
        self.Builder.local_ranks_size=1
        with self.assertRaisesRegex(ValueError,'TP2'):
            self.observed()
        self.assertIs(self.Base.load_weights,self.original_load)

    def test_tensor_storage_mutation_rejected(self):
        t=Tensor();count=[0]
        def guard():
            count[0]+=1
            if count[0]==3:t._version+=1
        with self.assertRaisesRegex(ValueError,'changed while observed'):m.tensor_record(t,TORCH,guard)

    def test_complete_large_tensor_stream(self):
        data=b'\xff\x7f'*(m.CHUNK_BYTES//2+3);guards=[]
        row=m.tensor_record(Tensor(data),TORCH,lambda:guards.append(1))
        self.assertEqual(row['cpu_bytes'],len(data))
        self.assertEqual(row['cpu_sha256'],hashlib.sha256(data).hexdigest())
        self.assertEqual(len(guards),4)

    def test_scalar_buffer_is_hashed_without_dtype_cast(self):
        row=m.tensor_record(Tensor(b'\x00\x80',shape=[]),TORCH,lambda:None)
        self.assertEqual(row['descriptor']['shape'],[])
        self.assertEqual(row['cpu_bytes'],2)
        self.assertEqual(row['cpu_sha256'],hashlib.sha256(b'\x00\x80').hexdigest())

    def test_inconsistent_tensor_geometry_rejected(self):
        a=self.observed('off');b=self.observed('on')
        b['pre_layout_shards'][0]['entries'][0]['descriptor']['shape']=[999]
        with self.assertRaises(ValueError):m.compare_observations(a,b)

    def test_incomplete_rank_map_rejected(self):
        with self.assertRaises(ValueError):m.tensor_map_record(ranks()[:1],TORCH,lambda:None)
        value=ranks();del value[1]['buffer']
        with self.assertRaises(ValueError):m.tensor_map_record(value,TORCH,lambda:None)

    def test_equal_cpu_views_still_do_not_prove_layout(self):
        report=m.compare_observations(self.observed('off'),self.observed('on'))
        self.assertTrue(report['pre_layout_full_rank_key_values_equal'])
        self.assertTrue(report['loaded_full_rank_key_cpu_views_equal'])
        for name in ('graph_weight_ABI_proven','layout_semantics_proven','inference_admitted','full_model_qualified','new_speedup_claim'):
            self.assertIs(report[name],False)

    def test_rank_value_mismatch_not_hidden_by_equal_shapes(self):
        a=self.observed('off');b=self.observed('on')
        b['pre_layout_shards'][1]['entries'][0]['cpu_sha256']='0'*64
        report=m.compare_observations(a,b)
        self.assertFalse(report['pre_layout_full_rank_key_values_equal'])
        self.assertTrue(report['loaded_full_rank_key_cpu_views_equal'])

    def test_loaded_view_mismatch_and_changed_provenance(self):
        a=self.observed('off');b=self.observed('on')
        b['loaded_cpu_views'][0]['entries'][0]['cpu_sha256']='0'*64
        self.assertFalse(m.compare_observations(a,b)['loaded_full_rank_key_cpu_views_equal'])
        b['checkpoint_provenance']={'checkpoint':'different'}
        with self.assertRaises(ValueError):m.compare_observations(a,b)

    def test_forged_qualification_rejected(self):
        a=self.observed('off');b=self.observed('on');b['layout_semantics_proven']=True
        with self.assertRaises(ValueError):m.compare_observations(a,b)

    def test_callback_mutation_does_not_rewrite_retained_report(self):
        report=self.observed(callback=lambda r:r.update(layout_semantics_proven=True))
        self.assertIs(report['layout_semantics_proven'],False)


if __name__=='__main__':
    original=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name.split('.')[0] in ('torch','torch_neuronx','torch_xla','numpy','neuronx_distributed','neuronx_distributed_inference'):
            raise AssertionError('No native imports in source-only tests: '+name)
        return original(name,*args,**kwargs)
    builtins.__import__=guarded_import
    def audit(event,args):
        if event in ('subprocess.Popen','os.system','socket.connect','socket.bind','os.listdir','os.scandir'):
            raise AssertionError('No runtime/process/filesystem traversal in source-only tests')
        if event=='open':
            name,mode,flags=args
            if not isinstance(name,str) or not name.endswith(('.py','.pyc')):
                raise AssertionError('No actual model/metadata/payload read in source-only tests')
            if flags & (1|2|64|512):raise AssertionError('No file mutation in source-only tests')
    sys.addaudithook(audit)
    unittest.main(verbosity=2)
