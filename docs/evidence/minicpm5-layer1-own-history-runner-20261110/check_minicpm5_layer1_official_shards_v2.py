"""Synthetic official-shard fixtures only; no native import or real weight read."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research import minicpm5_layer1_official_shards_v2 as m


def sha(value):return hashlib.sha256(value).hexdigest()


class Context:
    def __init__(self,rank,plan):
        self.plan=copy.deepcopy(plan);self.permit=dict(rank=rank)
        self._sources_checked=True;self.guards=0
    def guard(self,role):
        assert role=='cpu';self.guards+=1
    def verify_input(self,path):
        self.guard('cpu');row=self.plan['inputs'][path]
        assert m.identity(path)==row['identity'] and sha(Path(path).read_bytes())==row['sha256']
        return row
    def read_json(self,path,digest):
        self.guard('cpu');data=Path(path).read_bytes();assert sha(data)==digest
        return json.loads(data)


class TestShards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='mini-layer1-synthetic-')
        cls.root=Path(cls.tmp.name);cls.header={};offset=0;cls.parts=[];cls.expected={}
        for index,(name,rows,cols,axis) in enumerate((('q_proj',2048,2048,0),('k_proj',256,2048,0),
                ('v_proj',256,2048,0),('o_proj',2048,2048,1))):
            # Independent word formula includes row and column. Both TP shards
            # differ; O column halves cannot accidentally equal row halves.
            values=[[(r*31+c*7+index*113)%65536 for c in range(cols)]for r in range(rows)]
            data=b''.join(struct.pack('<'+'H'*cols,*row)for row in values)
            key='model.layers.1.self_attn.'+name+'.weight'
            cls.header[key]=dict(dtype='BF16',shape=[rows,cols],data_offsets=[offset,offset+len(data)])
            offset+=len(data);cls.parts.append(data)
            for rank in (0,1):
                selected=values[rank*(rows//2):(rank+1)*(rows//2)] if axis==0 else [row[rank*(cols//2):(rank+1)*(cols//2)]for row in values]
                shape=[len(selected),len(selected[0])]
                raw=b''.join(struct.pack('<'+'H'*len(row),*row)for row in selected)
                cls.expected[rank,m.local_key(name)]=(dict(shape=shape,dtype='torch.bfloat16',device='cpu'),raw)
            del values
        cls.rawheader=json.dumps(cls.header,sort_keys=True).encode()
        cls.path=cls.root/'synthetic.safetensors'
        cls.path.write_bytes(struct.pack('<Q',len(cls.rawheader))+cls.rawheader+b''.join(cls.parts))
        cls.headerpath=cls.root/'header.json';cls.headerpath.write_bytes(cls.rawheader)
        cls.plan=dict(official_layer0_weights=dict(path=str(cls.path)),safetensors_header=dict(path=str(cls.headerpath),sha256=sha(cls.rawheader)),
            inputs={str(cls.path):dict(identity=m.identity(cls.path),sha256=sha(cls.path.read_bytes()))})
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def verify(self,rank=0,context=None,describe=None,read=None):
        return m.verify(context or Context(rank,self.plan),rank,
            describe or (lambda k:copy.deepcopy(self.expected[rank,k][0])),
            read or (lambda k:self.expected[rank,k][1]))
    def test_literal_row_halves(self):
        raw=struct.pack('<8H',0,1,0x8000,0x7fc0,0xffff,0x7f80,6,7)
        self.assertEqual(m.shard_bytes(raw,[4,2],0,0),raw[:8])
        self.assertEqual(m.shard_bytes(raw,[4,2],0,1),raw[8:])
    def test_literal_column_halves_and_special_words(self):
        raw=struct.pack('<8H',0,1,0x8000,0x7fc0,0xffff,0x7f80,6,7)
        self.assertEqual(m.shard_bytes(raw,[2,4],1,0),struct.pack('<4H',0,1,0xffff,0x7f80))
        self.assertEqual(m.shard_bytes(raw,[2,4],1,1),struct.pack('<4H',0x8000,0x7fc0,6,7))
    def test_axis_and_extent_rejected(self):
        for raw,shape,axis,rank in ((b'00',[1,1],0,0),(b'0000',[2,2],0,0),(b'00000000',[2,2],2,0),(b'00000000',[2,2],1,True)):
            with self.subTest(args=(shape,axis,rank)),self.assertRaises(ValueError):m.shard_bytes(raw,shape,axis,rank)
    def test_all_four_rank0_match_independent_words(self):
        ctx=Context(0,self.plan);result=self.verify(context=ctx)
        self.assertTrue(result['all_four_projection_shards_raw_exact']);self.assertEqual(len(result['weights']),4)
        self.assertGreater(ctx.guards,100);self.assertFalse(result['full_model_qualified'])
        for key,digest in result['hashes'].items():self.assertEqual(digest,sha(self.expected[0,key][1]))
    def test_all_four_rank1_match_independent_words(self):
        result=self.verify(1);self.assertEqual(result['q_head_ids'],list(range(8,16)));self.assertEqual(result['kv_head_ids'],[1])
        for key,digest in result['hashes'].items():
            self.assertEqual(digest,sha(self.expected[1,key][1]));self.assertNotEqual(digest,sha(self.expected[0,key][1]))
    def test_changed_byte_in_every_projection_rejected(self):
        for badkey in (m.local_key(n)for n in m.PROJECTIONS):
            def read(key):
                raw=self.expected[0,key][1]
                return raw[:-1]+bytes([raw[-1]^1]) if key==badkey else raw
            with self.subTest(key=badkey),self.assertRaisesRegex(ValueError,'Every official'):self.verify(read=read)
    def test_wrong_rank_shards_rejected(self):
        with self.assertRaisesRegex(ValueError,'Every official'):self.verify(read=lambda k:self.expected[1,k][1])
    def test_wrong_shape_dtype_device_rejected(self):
        for field,value in (('shape',[1024,1024]),('dtype','torch.float32'),('device','cuda')):
            def describe(key):return dict(self.expected[0,key][0],**{field:value})
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'geometry'):self.verify(describe=describe)
    def test_rank_and_source_admission_rejected(self):
        for rank,sources,permit in ((True,True,True),(2,True,2),(0,False,0),(0,True,1)):
            ctx=Context(permit,self.plan);ctx._sources_checked=sources
            with self.subTest(rank=rank,sources=sources,permit=permit),self.assertRaises(ValueError):self.verify(rank,ctx)
    def test_guard_denial_precedes_file_and_callback_access(self):
        class Denied:
            def guard(self,role):raise RuntimeError('not admitted')
        with patch.object(m.os,'open',side_effect=AssertionError('unexpected file read')):
            with self.assertRaisesRegex(RuntimeError,'not admitted'):m.verify(Denied(),0,None,None)
    def test_native_wrapper_guard_precedes_native_object_access(self):
        class Denied:
            def guard(self,role):raise RuntimeError('not admitted')
        with self.assertRaisesRegex(RuntimeError,'not admitted'):m.verify_model(Denied(),None,None,0)
    def test_mismatched_header_rejected(self):
        ctx=Context(0,self.plan)
        def wrong(path,digest):
            header=copy.deepcopy(self.header);header['model.layers.1.self_attn.q_proj.weight']['data_offsets'][0]+=1
            return header
        ctx.read_json=wrong
        with self.assertRaises(ValueError):self.verify(context=ctx)
    def test_changed_official_identity_after_authentication_rejected(self):
        ctx=Context(0,self.plan);original=ctx.read_json
        def replace_identity(path,digest):
            ctx.plan['inputs'][str(self.path)]['identity']['mtime_ns']+=1
            return original(path,digest)
        ctx.read_json=replace_identity
        with self.assertRaisesRegex(ValueError,'identity'):self.verify(context=ctx)
    def test_mid_read_guard_revocation_propagates(self):
        ctx=Context(0,self.plan)
        def guard(role):
            ctx.guards+=1
            if ctx.guards==15:raise RuntimeError('lease revoked')
        ctx.guard=guard
        with self.assertRaisesRegex(RuntimeError,'lease revoked'):self.verify(context=ctx)
    def test_short_read_rejected(self):
        original=m.os.pread
        def short(fd,count,offset):return b'' if offset>=8+len(self.rawheader) else original(fd,count,offset)
        with patch.object(m.os,'pread',side_effect=short),self.assertRaisesRegex(ValueError,'short read'):self.verify()
    def test_column_half_cannot_be_substituted_with_row_half(self):
        key=m.local_key('o_proj');wrong=self.parts[3][:len(self.parts[3])//2]
        self.assertNotEqual(wrong,self.expected[0,key][1])
        with self.assertRaisesRegex(ValueError,'Every official'):
            self.verify(read=lambda k:wrong if k==key else self.expected[0,k][1])


class ParameterContract(unittest.TestCase):
    def metadata(self,rank):
        import minicpm5_layer0_cpu_constructor_v1 as original
        prefix='layers.1.self_attn.'
        result={key[len(prefix):]:dict(value,device='cpu')for key,value in original.expected_state_metadata('model').items()if key.startswith(prefix)}
        result['rank_util.rank'].update(values=[rank],requires_grad=False)
        return result
    def test_original_full_model_metadata_includes_int32_rank_parameter(self):
        for rank in (0,1):
            metadata=self.metadata(rank);self.assertEqual(len(metadata),5)
            self.assertTrue(m.check_parameter_contract(metadata,rank))
    def test_missing_wrong_or_trainable_rank_is_rejected(self):
        for action in ('missing','wrong','trainable','precision'):
            metadata=self.metadata(0)
            if action=='missing':metadata.pop('rank_util.rank')
            elif action=='wrong':metadata['rank_util.rank']['values']=[1]
            elif action=='trainable':metadata['rank_util.rank']['requires_grad']=True
            else:metadata['rank_util.rank']['dtype']='torch.bfloat16'
            with self.subTest(action=action),self.assertRaisesRegex(ValueError,'SPMDRank'):
                m.check_parameter_contract(metadata,0)
    def test_unknown_or_missing_projection_cannot_hide_behind_rank_exception(self):
        for action in ('extra','missing','shape'):
            metadata=self.metadata(1)
            if action=='extra':metadata['unknown']=metadata['rank_util.rank']
            elif action=='missing':metadata.pop('qkv_proj.k_proj.weight')
            else:metadata['qkv_proj.q_proj.weight']['shape']=[2048,2048]
            with self.subTest(action=action),self.assertRaises(ValueError):m.check_parameter_contract(metadata,1)


if __name__=='__main__':
    assert not {'torch','numpy','torch_neuronx','transformers'}&set(sys.modules)
    unittest.main(verbosity=2)
