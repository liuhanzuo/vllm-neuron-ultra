"""Synthetic interfaces only: no Torch/NumPy/native imports or actual tensor reads.

The fake tensors below model bytes, strides, offsets and aliasing, not attention
arithmetic. These tests check orchestration and rejection, never model accuracy.
"""
import ast
import contextlib
import copy
import itertools
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
Q=Path(__file__).resolve().parent;sys.path.insert(0,str(Q.parent))
from research import minicpm5_layer1_sdk_batch_layout_callable_v1 as call
from research import check_minicpm5_layer1_current_history_factorial_v1 as fixture
core=call.core


def strides(shape):
    result=[];stride=1
    for size in reversed(shape):result.append(stride);stride*=size
    return tuple(reversed(result))


class Tensor:
    widths={'torch.bfloat16':2,'torch.bool':1,'torch.int64':8,'torch.int32':4}
    def __init__(self,raw,dtype,shape=None,stride=None,offset=0):
        self.backing=raw if isinstance(raw,bytearray) else bytearray(raw)
        self.dtype=dtype;self.width=self.widths[dtype];self.shape=tuple(shape or (len(raw)//self.width,))
        self.steps=tuple(stride or strides(self.shape));self.offset=offset;self.device=SimpleNamespace(type='cpu')
    def stride(self):return self.steps
    def storage_offset(self):return self.offset
    def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:id(self.backing),nbytes=lambda:len(self.backing))
    def as_strided(self,shape,stride,offset=0):return Tensor(self.backing,self.dtype,shape,stride,offset)
    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],(list,tuple)):shape=shape[0]
        assert math.prod(shape)==math.prod(self.shape)
        return Tensor(self.backing,self.dtype,shape,strides(shape),self.offset)
    def narrow(self,dim,start,length):
        shape=list(self.shape);shape[dim]=length
        return Tensor(self.backing,self.dtype,shape,self.steps,self.offset+start*self.steps[dim])
    def raw(self):
        # Copy contiguous suffix chunks, retaining actual offsets in source storage.
        split=len(self.shape);extent=1
        for index in range(len(self.shape)-1,-1,-1):
            if self.steps[index]!=extent:break
            split=index;extent*=self.shape[index]
        pieces=[]
        for index in itertools.product(*(range(s)for s in self.shape[:split])):
            offset=self.offset+sum(i*s for i,s in zip(index,self.steps))
            pieces.append(self.backing[offset*self.width:(offset+extent)*self.width])
        return b''.join(pieces)
    def copy_(self,other):
        assert len(self.backing)==len(other.raw())
        self.backing[:]=other.raw();return self


def meta(torch,value):
    return dict(shape=list(value.shape),stride=list(value.stride()),offset=value.offset,dtype=value.dtype,
        sha256=core.boundary.sha(value.raw()))


class FakeTorch:
    bfloat16='torch.bfloat16';bool='torch.bool';int64='torch.int64';int32='torch.int32'
    def __init__(self,rank):
        self.distributed=SimpleNamespace(is_initialized=lambda:True,get_backend=lambda:'gloo',get_world_size=lambda:2,get_rank=lambda:rank)
    def is_inference_mode_enabled(self):return True
    def get_num_threads(self):return 1
    def frombuffer(self,raw,dtype):return Tensor(raw,dtype)
    def tensor(self,values,dtype):
        def flat(v):return sum((flat(x)for x in v),[])if isinstance(v,(list,tuple))else [v]
        values=flat(values);fmt={self.bool:'?',self.int64:'q',self.int32:'i'}[dtype]
        return Tensor(struct.pack('<'+fmt*len(values),*values),dtype)
    def zeros(self,batch,dtype):return Tensor(bytes(batch*Tensor.widths[dtype]),dtype)
    def isfinite(self,tensor):
        return SimpleNamespace(all=lambda:core.factorial.history.finite_bf16(tensor.raw())if tensor.dtype==self.bfloat16 else True)


class RotaryEmbedding:
    __module__='neuronx_distributed_inference.modules.attention.utils'
    base=5000000;dim=128;factor=None
    def __init__(self):self.calls=[]
    def __call__(self,normalized,positions):
        self.calls.append((meta(None,normalized),meta(None,positions)))
        # Deliberately use nonzero offsets and nonstandard preserved source strides.
        outputs=[]
        for kind in (0,1):
            data=bytearray(2200)
            for slot in range(4):data[2*(7+slot*256):2*(7+slot*256+128)]=struct.pack('<H',0x3f80+kind*8+slot)*128
            outputs.append(Tensor(data,'torch.bfloat16',(4,1,128),(256,128,1),7))
        return outputs


class NeuronLlamaAttention:
    __module__='neuronx_distributed_inference.models.llama.modeling_llama'
    def __init__(self):
        self.calls=[];self._forward_hooks={};self._forward_pre_hooks={}
        self.torch_dtype='torch.bfloat16';self.tp_degree=2;self.cp_degree=self.dp_degree=1
        self.num_heads=8;self.num_key_value_heads=1;self.head_dim=128
        for k in ('sequence_parallel_enabled','k_cache_transposed','attn_block_tkg_nki_kernel_enabled',
            'attn_block_cte_nki_kernel_enabled','attn_tkg_builtin_kernel_enabled','attn_tkg_nki_kernel_enabled',
            'flash_decoding_enabled','use_qk_norm'):setattr(self,k,False)
        self.attention_chunk_size=self.sliding_window=None
        self.o_proj=SimpleNamespace(o_proj=SimpleNamespace(reduce_dtype='torch.bfloat16'))
        self.params={k:Tensor(b'\x80\x3f','torch.bfloat16')for k in call.WEIGHTS}
        self.mutation=None;self.bad_output=None;self.bad_cos=False
    def modules(self):return [self]
    def named_parameters(self):return self.params.items()
    def __call__(self,**kwargs):
        selected=kwargs['hidden_states'];batch=selected.shape[0]
        self.calls.append(kwargs)
        if self.mutation:self.mutation(kwargs)
        raw=selected.raw()
        output=Tensor(raw,'torch.bfloat16',(batch,1,2048))
        if self.bad_output:self.bad_output(output)
        return SimpleNamespace(hidden_states=output,residual=None,cos_cache=None if self.bad_cos else kwargs['cos_cache'],sin_cache=kwargs['sin_cache'])


class Manager:
    def __init__(self):
        self.calls=[];self.bad_getter=False
        self.past_key_values=[None,None,*[Tensor(bytes(4325376),'torch.bfloat16',(4,1,4224,128))for _ in range(2)]]
    def get_kv_by_layer_id(self,**kwargs):
        self.calls.append(kwargs)
        views=[v.as_strided((4,1,kwargs['seq_len'],128),(540672,540672,128,1))for v in self.past_key_values[2:4]]
        return [Tensor(v.raw(),v.dtype,v.shape)for v in views]if self.bad_getter else views


class CallableChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.CurrentHistoryChecks.setUpClass();f=fixture.CurrentHistoryChecks;cls.fixture=f
        cls.case=copy.deepcopy(f.case);cls.case['call']['frames']=sorted(cls.case['call']['frames'],key=lambda f:f['index'])
        cls.expected=bytes(16384)
        cls.case['expected_attention']=dict(logical_sha256=core.boundary.sha(cls.expected),metadata=core.boundary.OUTPUT_META)
        cls.case['attention_route']=dict(mask=core.boundary.tensor_meta(bytes(int(c<p)for p in cls.case['positions']for c in range(128)),
            [4,1,1,128],[128,128,128,1],'torch.bool'),positions=core.boundary.tensor_meta(struct.pack('<4q',*cls.case['positions']),[4,1],[1,1],'torch.int64'))
        cls.prepared=core.prepare(cls.case,f.normalized,f.own,f.prior,f.current)
        cls.previous={}
        for mode in ('own','official_both','restore'):
            raw=cls.prepared['normalized_raw']if mode=='official_both'else cls.expected;digest=core.boundary.sha(raw)
            result=dict(schema='minicpm5_layer1_current_history_factorial_callable_result_v1',complete=True,
                rank=cls.case['rank'],arm=cls.case['arm'],call=cls.case['call'],mode=mode,
                full_model_qualified=False,new_speedup_claim=False,attention_output_sha256=digest,
                intervention=cls.prepared['intervention']if mode=='official_both'else {})
            cls.previous[mode]=dict(result=result,output=dict(path='/synthetic/'+mode+'.bf16',raw_bytes=16384,sha256=digest,identity={'synthetic':True}),raw=raw)

    def setUp(self):
        self.torch=FakeTorch(self.case['rank']);self.attention=NeuronLlamaAttention();self.rope=RotaryEmbedding()
        config=SimpleNamespace(batch_size=4,max_batch_size=4,tkg_batch_size=4,kv_cache_batch_size=4,n_active_tokens=1,
            is_prefill_stage=False,token_generation_batches=None,rpl_reduce_dtype=None,sequence_parallel_enabled=False,
            is_prefix_caching=False,fused_qkv=False,qkv_kernel_enabled=False,qkv_nki_kernel_enabled=False,
            out_proj_kernel_enabled=False,torch_dtype='torch.bfloat16')
        self.model=SimpleNamespace(neuron_config=config,n_positions=128,kv_mgr=Manager(),
            layers=[SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=self.rope)),SimpleNamespace(self_attn=self.attention)]+[None]*40)
        self.hashes={k:core.boundary.sha(v.raw())for k,v in self.attention.params.items()};self.guards=0;self.controls=[]
    def guard(self):self.guards+=1
    def patches(self):
        stack=contextlib.ExitStack();stack.enter_context(patch.object(call,'tensor_bytes',side_effect=lambda torch,value:value.raw()))
        stack.enter_context(patch.object(call,'metadata',side_effect=meta));return stack
    def variants(self):
        with self.patches():return call.official_variants(self.torch,self.model,self.prepared,self.hashes,self.guard)
    def replay(self,previous=None,restore_bad=False):
        f=self.fixture
        def control(torch,model,case,mode,*args):
            self.controls.append((mode,len(self.attention.calls)))
            return dict(mode=mode,complete=True),b'\x00\x3f'*8192 if mode=='restore'and restore_bad else self.expected
        with self.patches(),patch.object(call.original,'replay',side_effect=control):
            return call.replay(self.torch,self.model,self.case,f.normalized,f.own,f.prior,f.current,self.expected,
                self.previous if previous is None else previous,self.hashes,self.guard)

    def test_ten_official_calls_one_b4_rope_four_complete_outputs(self):
        result,rows=self.variants()
        self.assertEqual(len(self.attention.calls),10);self.assertEqual(len(self.rope.calls),1)
        self.assertEqual([row['variant']for row in rows],list(core.VARIANTS));self.assertTrue(all(len(row['raw'])==16384 for row in rows))
        self.assertEqual([len(v['tasks'])for v in result['variants']],[1,1,4,4]);self.assertTrue(self.guards>20)
        self.assertTrue(all(row['raw']==self.prepared['normalized_raw']for row in rows))
        json.dumps(result)

    def test_six_row_api_with_full_own_and_restore_around_all_variants(self):
        result,rows=self.replay()
        self.assertEqual([r['variant']for r in rows],list(call.ORDER));self.assertEqual(self.controls,[('own',0),('restore',10)])
        self.assertEqual(result['physical_attention_invocations'],12);self.assertEqual(result['raw_output_bytes'],98304)
        self.assertTrue(all(set(row)=={'rank','call','variant','raw'}and row['call']==self.case['call']for row in rows))
        self.assertTrue(result['controls']['previous_both_official_full_bytes_exact']);json.dumps(result)

    def test_b1_uses_selected_physical_slots_with_full_cache_backings(self):
        result,_=self.variants()
        self.assertEqual(len(self.model.kv_mgr.calls),1)
        self.assertEqual(self.model.kv_mgr.calls[0]['seq_ids'].raw(),struct.pack('<4q',0,1,2,3))
        for inv in (self.attention.calls[2:6],self.attention.calls[6:10]):
            for slot,kwargs in enumerate(inv):
                self.assertEqual(kwargs['seq_ids'].raw(),struct.pack('<q',slot))
                self.assertEqual(kwargs['position_ids'].raw(),struct.pack('<q',self.case['positions'][slot]))
                self.assertEqual(kwargs['past_key_value'][0].offset,slot*540672)
                self.assertEqual(kwargs['past_key_value'][0].stride(),(540672,540672,128,1))
                self.assertEqual(kwargs['past_key_value'][0].untyped_storage().nbytes(),4325376)
                self.assertEqual(kwargs['seq_len'],128);self.assertFalse(kwargs['get_kv_per_layer']);self.assertFalse(kwargs['update_kv_per_layer'])

    def test_normalized_layout_offsets_and_holes_preserved(self):
        self.variants()
        self.assertEqual(self.attention.calls[0]['hidden_states'].stride(),(1,4,4))
        self.assertEqual(self.attention.calls[1]['hidden_states'].stride(),(2048,2048,1))
        for slot in range(4):
            a=self.attention.calls[2+slot]['hidden_states'];b=self.attention.calls[6+slot]['hidden_states']
            self.assertEqual(a.offset,slot);self.assertEqual(b.offset,slot*2048)
            self.assertEqual(a.raw(),b.raw());self.assertEqual(len(a.backing),16384);self.assertEqual(len(a.raw()),4096)

    def test_same_rope_storage_stride_nonzero_offset_and_bytes(self):
        result,_=self.variants();first=self.attention.calls[0]
        for slot,kwargs in enumerate(self.attention.calls[2:6]):
            for key in ('cos_cache','sin_cache'):
                v=kwargs[key];self.assertIs(v.backing,first[key].backing)
                self.assertEqual(v.offset,7+slot*256);self.assertEqual(v.stride(),(256,128,1))
                self.assertEqual(v.raw(),first[key].raw()[slot*256:(slot+1)*256])
        self.assertEqual(result['official_rope'][0]['offset'],7)

    def test_b4_getter_must_keep_original_storage_alias_and_stride(self):
        self.model.kv_mgr.bad_getter=True
        with self.assertRaises(ValueError):self.variants()
        self.assertFalse(self.attention.calls);self.assertFalse(self.rope.calls)

    def test_guard_rejects_before_any_model_or_source_access(self):
        def deny():raise RuntimeError('No lease')
        with self.assertRaises(RuntimeError):call.official_variants(None,None,None,None,deny)
        with self.assertRaises(RuntimeError):call.replay(*([None]*10),deny)

    def test_padding_mutation_is_rejected(self):
        def mutate(kwargs):kwargs['past_key_value'][0].backing[-2:]=b'\x80\x40'
        self.attention.mutation=mutate
        with self.assertRaises(ValueError):self.variants()
        self.assertEqual(len(self.attention.calls),1)

    def test_unselected_b1_normalized_hole_mutation_is_rejected(self):
        def mutate(kwargs):
            v=kwargs['hidden_states']
            if v.shape[0]==1:v.backing[2:4]=b'\x80\x40'
        self.attention.mutation=mutate
        with self.assertRaises(ValueError):self.variants()
        self.assertEqual(len(self.attention.calls),3)

    def test_bf16_collective_scope_cannot_be_promoted_to_fp32(self):
        self.attention.o_proj.o_proj.reduce_dtype='torch.float32'
        with self.assertRaises(ValueError):self.variants()
        self.assertFalse(self.attention.calls)

    def test_weight_or_config_mutation_after_attention_is_rejected(self):
        for mode in ('weight','config'):
            self.setUp()
            def mutate(kwargs):
                if mode=='weight':self.attention.params[call.WEIGHTS[0]].backing[:]=b'\x00\x3f'
                else:self.model.neuron_config.batch_size=1
            self.attention.mutation=mutate
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.variants()

    def test_corrupt_raw_predecessor_and_rehashed_wrong_both_fail(self):
        for rehash in (False,True):
            self.setUp();previous=copy.deepcopy(self.previous);row=previous['official_both'];row['raw']=b'\x00\x3f'*8192
            if rehash:row['result']['attention_output_sha256']=row['output']['sha256']=core.boundary.sha(row['raw'])
            with self.subTest(rehash=rehash),self.assertRaises(ValueError):self.replay(previous)

    def test_wrong_prior_physical_state_or_case_rejected_before_control(self):
        for change in [lambda p:p['official_both']['result']['intervention'].update(selected_storage_sha256='0'*64),
            lambda p:p['own']['result'].update(rank=0),lambda p:p['restore']['result'].update(call={})]:
            previous=copy.deepcopy(self.previous);change(previous)
            with self.assertRaises(ValueError):self.replay(previous)
            self.assertFalse(self.controls)

    def test_changed_restore_bytes_are_not_accepted(self):
        with self.assertRaises(ValueError):self.replay(restore_bad=True)

    def test_bad_shape_nonfinite_or_wrong_rope_return_rejected(self):
        for mode in ('shape','nonfinite','rope'):
            self.setUp()
            if mode=='shape':self.attention.bad_output=lambda o:setattr(o,'shape',(4,2,1024))
            elif mode=='nonfinite':self.attention.bad_output=lambda o:o.backing.__setitem__(slice(0,2),b'\x80\x7f')
            else:self.attention.bad_cos=True
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.variants()

    def test_real_attention_keywords_preserve_original_sdk_invocation(self):
        def invocation(path):
            tree=ast.parse(path.read_text())
            return next(n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='attention')
        old=invocation(Q/'minicpm5_layer1_current_history_factorial_callable_v1.py');new=invocation(Path(call.__file__))
        old={k.arg:k.value for k in old.keywords};new={k.arg:k.value for k in new.keywords}
        self.assertEqual(set(old),set(new))
        varying={'hidden_states','attention_mask','position_ids','past_key_value','adapter_ids','cos_cache','sin_cache','seq_ids'}
        for key in set(old)-varying:self.assertEqual(ast.dump(old[key]),ast.dump(new[key]))
        for entry in self.replay()[1]:self.assertFalse({'torch','numpy','neuronxcc','torch_neuronx'}&set(sys.modules))

    def test_forbidden_native_imports_and_no_io_or_launch_calls(self):
        self.assertFalse({'torch','numpy','torch_neuronx','neuronxcc'}&set(sys.modules))
        tree=ast.parse(Path(call.__file__).read_text())
        names={n.func.id for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)}
        attrs={n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)}
        self.assertFalse({'open','exec','eval','__import__'}&names)
        self.assertFalse({'read_bytes','read_text','write_bytes','write_text','Popen','acquire_lock'}&attrs)


if __name__=='__main__':unittest.main(verbosity=2)
