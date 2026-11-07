# SPDX-License-Identifier: Apache-2.0
import ast
import copy
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from vllm_neuron.models.nemotron9b import identity as i, state
P=Path(i.__file__).parent


class Contracts(unittest.TestCase):
    def test_original_124_mixed_dtype_buffers(self):
        rows=state.state_layout()
        self.assertEqual(len(rows),124);self.assertEqual(sum(r['dtype']=='torch.float32'for r in rows),54)
        self.assertEqual(sum(math.prod(r['shape'])*({'torch.bfloat16':2,'torch.float32':4}[r['dtype']])for r in rows),845283328)
        self.assertEqual([r['name']for r in rows[:62]],sorted(r['name']for r in rows[:62]))

    def test_no_serving_path_calls_diagnostic_clear(self):
        for name in ('modeling.py','registration.py','serve.py','artifacts.py','cache.py'):
            self.assertNotIn('clear_for_cold_validation',(P/name).read_text())
        source=ast.parse((P/'modeling.py').read_text())
        model=next(x for x in source.body if isinstance(x,ast.ClassDef)and x.name=='NeuronNemotron9BModel')
        apply=next(x for x in model.body if isinstance(x,ast.FunctionDef)and x.name=='_apply')
        self.assertIn('torch.float32',ast.unparse(apply));self.assertIn('current.data',ast.unparse(apply))

    def test_explicit_clear_preserves_bindings_and_rejects_changed_bindings(self):
        events=[]
        class Tensor:
            def __init__(self,row,index):
                self.shape=tuple(row['shape']);self.dtype=row['dtype'];self._cdata=index+1
                self.device=SimpleNamespace(type='privateuseone',index=row['rank'])
                self.nonzero=1
            def copy_(self,value):self.nonzero=value;events.append(self._cdata)
            def detach(self):return self
            def cpu(self):return self
            def contiguous(self):return self
            def view(self,dtype):return self
        model=SimpleNamespace(state=[{},{}])
        for index,row in enumerate(state.state_layout()):model.state[row['rank']][row['name']]=Tensor(row,index)
        expected=state.bindings(model)
        fake=ModuleType('torch');fake.uint8='uint8'
        fake.zeros=lambda shape,dtype,device:0
        fake.count_nonzero=lambda tensor:tensor.nonzero
        with patch.dict(sys.modules,{'torch':fake}):
            result=state.clear_for_cold_validation(model,expected)
            self.assertEqual(result['states_written'],124)
            self.assertEqual(len(set(events)),124)
            self.assertEqual(state.bindings(model),expected)
            events.clear()
            next(iter(model.state[0].values()))._cdata=1000
            with self.assertRaises(i.AdmissionError):state.clear_for_cold_validation(model,expected)
            self.assertEqual(events,[])

    def test_spawn_entry_registers_before_main_guard(self):
        source=(P/'serve.py').read_text()
        self.assertLess(source.index('register()'),source.index('if __name__'))
        self.assertIn('"spawn"',source)

    def test_historical_config_accepted(self):
        c=json.loads(Path(__file__).with_name('historical_config.json').read_text())
        i.validate_scope(c,c['neuron_config']);i.canonical_configuration(c)
        for key,bad in [('tp_degree',1),('max_length',8192),('batch_size',8),('ctx_batch_size',4),('context_encoding_buckets',[4096]),('torch_dtype','float32'),('quantized',True),('on_device_sampling_config',{})]:
            x=copy.deepcopy(c);x['neuron_config'][key]=bad
            with self.subTest(key=key),self.assertRaises(i.AdmissionError):i.validate_scope(x,x['neuron_config'])

    def test_positive_infinities_exact_and_no_nan_or_other_infinity(self):
        x={'time_step_limit':[0.,float('inf')],'neuron_config':{'quantize_clamp_bound':float('inf')}}
        self.assertEqual(i.canonical_configuration(x),i.canonical_configuration(json.loads(json.dumps(x))))
        for bad in ({'time_step_limit':[0.,float('-inf')]},{'other':float('inf')},{'time_step_limit':[0.,float('nan')]}):
            with self.assertRaises(i.AdmissionError):i.canonical_configuration(bad)
        self.assertNotEqual(i.canonical_configuration({'x':True}),i.canonical_configuration({'x':1}))

    def test_secondary_sdk_routes_and_unknown_fields_fail_closed(self):
        c=json.loads(Path(__file__).with_name('historical_config.json').read_text())
        for key,bad in [('kv_cache_quant',True),('qkv_kernel_enabled',True),
                        ('rpl_reduce_dtype','float32'),('allow_input_truncation',True),
                        ('future_unknown_option',True),('mlp_kernel_enabled',0)]:
            x=copy.deepcopy(c);x['neuron_config'][key]=bad
            with self.subTest(key=key),self.assertRaises(i.AdmissionError):i.validate_scope(x,x['neuron_config'])
        x=copy.deepcopy(c);del x['neuron_config']['qkv_kernel_enabled']
        with self.assertRaises(i.AdmissionError):i.validate_scope(x,x['neuron_config'])

    def test_standard_sampler_and_functionalization_required(self):
        env={'NEURON_NEMOTRON9B':'1','XLA_DISABLE_FUNCTIONALIZATION':'1'}
        with patch.dict(os.environ,env,clear=True):self.assertEqual(i.selected_variant(),'split_nki_v5')
        for extra in ({'NEURON_CPU_SAMPLER_FAST':'1'},{'XLA_DISABLE_FUNCTIONALIZATION':'0'},{'TRN1_NEMOTRON9B_TEXT_ADAPTER':'1'}):
            with patch.dict(os.environ,dict(env,**extra),clear=True),self.assertRaises(i.AdmissionError):i.selected_variant()

    def test_kernel_capacity_and_inactive_write_contract_remain_source_pinned(self):
        # Actual arithmetic equivalence is independently checked against frozen AST hashes.
        read=(P/'nki_read.py').read_text();write=(P/'nki_write.py').read_text()
        self.assertIn('capacity == 4',read);self.assertIn('nl.maximum',read)
        self.assertIn('width % 128 == 0',write);self.assertIn('nisa',write)


if __name__=='__main__':unittest.main()
