"""SOURCE ONLY: synthetic archives and fake file ownership; no model payload IO."""
import builtins
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import stat
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import deepseek_weight_serialized_descriptor_v1 as m


class Obj:
    def __init__(self,origin,state):self.origin,self.state=origin,state


def pack(value):
    if isinstance(value,Obj):
        module,name=value.origin.rsplit('.',1)
        return b'c'+module.encode()+b'\n'+name.encode()+b'\n)\x81'+pack(value.state)+b'b'
    if value is None:return b'N'
    if value is True:return b'\x88'
    if value is False:return b'\x89'
    if type(value)is int:return b'J'+struct.pack('<i',value)
    if type(value)in (str,bytes):
        data=value.encode()if type(value)is str else value
        return b'X'+struct.pack('<I',len(data))+data
    if type(value)is dict:return b'}('+b''.join(pack(k)+pack(v)for k,v in value.items())+b'u'
    if type(value)is list:return b']('+b''.join(pack(v)for v in value)+b'e'
    if type(value)is tuple:return b'('+b''.join(pack(v)for v in value)+b't'
    raise AssertionError(type(value))


def varint(value):
    out=bytearray()
    while value>=128:out.append((value&127)|128);value>>=7
    out.append(value);return bytes(out)


def field(number,value):
    if type(value)is int:return varint(number<<3)+varint(value)
    return varint((number<<3)|2)+varint(len(value))+value


def tensor(name,key='',shape=(2,),kind=0,dtype=14):
    data=field(1,name.encode())+b''.join(field(2,n)for n in shape)+field(3,dtype)
    if key:data+=field(6,key.encode())
    if kind:data+=field(7,kind)
    return data


def metadata(*,wlt=False,large_graph=False):
    if wlt:
        inputs=[tensor('input'+str(i),key,kind=3)for i,key in enumerate(('a.weight','b.weight'))]
        outputs=[tensor('output'+str(i),key)for i,key in enumerate(('a.weight','b.weight'))]
        return b''.join(field(1,t)for t in inputs)+b''.join(field(2,t)for t in outputs)
    inputs=[tensor('input0',shape=(1,128),kind=1,dtype=2),tensor('input1',shape=(1,128),kind=1,dtype=2)]
    inputs += [tensor('input'+str(2+i),key,kind=3)for i,key in enumerate(('a.weight','b.weight'))]
    inputs += [tensor('input'+str(4+i),'kv_mgr.past_key_values.'+str(i),shape=(4,4,4096,128),kind=2)for i in range(72)]
    outputs=[tensor('output0',shape=(1,1,151936),dtype=1)]
    outputs += [tensor('output'+str(1+i),shape=(4,4,4096,128))for i in range(72)]
    data=b''.join(field(1,t)for t in inputs)+b''.join(field(2,t)for t in outputs)
    data+=field(4,b'G'*(1048600 if large_graph else 31))
    data+=b''.join(field(6,field(1,1+i)+field(2,4+i))for i in range(72))
    return data


def fixture(*,large_neff=True):
    neff=b'\x02'+b'\x00'*7+b'N'*(1048600 if large_neff else 64)
    small=b'\x02'+b'\x00'*7+b'GRAPH'
    loader=Obj('__torch__.torch.classes.neuron.LayoutTransformation',(neff,metadata(wlt=True),'v1',2))
    groups={};routes={}
    for tag in m.TAGS:
        groups[tag]=Obj('__torch__.synthetic.Bucket',{'models':[Obj('__torch__.torch.classes.neuron.SPMDModel',
            (small,metadata(),'v1',2,2))for _ in range(6)]})
        for index,bucket in enumerate(m.BUCKETS):
            batch=1 if tag==m.TAGS[0]else 4
            routes[json.dumps([[batch,bucket if batch==1 else 1],[batch,bucket]])]=(tag,index)
    state=dict(start_rank_id=0,weight_loader=loader,input_shape_map=routes,models=Obj('__torch__.synthetic.ModuleDict',groups))
    return Obj('__torch__.synthetic.Executor',{'nxd_model':Obj('__torch__.neuronx_distributed.trace.spmd.NxDModel',state)})


def archive_bytes(obj=None,*,trailing=b'',extra=None):
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w',compression=zipfile.ZIP_STORED)as archive:
        archive.writestr('model/data.pkl',b'\x80\x02'+pack(obj or fixture())+b'.'+trailing)
        archive.writestr('model/code/__torch__/spmd.py',b'def initialize(): pass\n')
        # This member must not be opened by descriptor parsing (whole archive hash still reads it).
        archive.writestr('model/data/0',b'SYNTHETIC TENSOR STORAGE')
        if extra:archive.writestr(*extra)
    return data.getvalue()


def read(data,*,binding_changes=None,limit_changes=None):
    st=SimpleNamespace(st_dev=1,st_ino=2,st_size=len(data),st_mtime_ns=3,st_ctime_ns=4,
                       st_mode=stat.S_IFREG|0o600,st_nlink=1)
    class P:
        parents=()
        def __init__(self,value):self.value=value
        def is_absolute(self):return True
        def resolve(self):return self
        def is_symlink(self):return False
        def stat(self):return st
    binding=dict(path='/synthetic/model.pt',sha256=hashlib.sha256(data).hexdigest(),identity=m.file_identity(st))
    binding.update(binding_changes or {})
    limits=dict(archive_bytes=32<<20,pickle_bytes=24<<20,members=16,code_bytes=1<<20)
    limits.update(limit_changes or {})
    with patch.object(m,'Path',P),patch.object(m.os,'open',return_value=321),patch.object(m.os,'close'),\
         patch.object(m.os,'fstat',return_value=st),patch.object(m.os,'fdopen',return_value=io.BytesIO(data)):
        return m.read_archive(binding,lambda:None,limits)


class Tests(unittest.TestCase):
    def test_real_inert_parser_full_WLT_stream_and12_graphs(self):
        value=read(archive_bytes())
        self.assertEqual(len(value['graphs']),12)
        self.assertEqual(value['named_weights_per_graph'],2)
        neff=fixture().state['nxd_model'].state['weight_loader'].state[0]
        self.assertEqual(value['wlt']['neff'],dict(bytes=len(neff),sha256=hashlib.sha256(neff).hexdigest()))
        self.assertTrue(value['whole_archive_sha256_verified'])

    def test_large_metaneff_graph_streamed_completely(self):
        raw=metadata(large_graph=True)
        value=m.BoundRuntimeReader(io.BytesIO(b'\x80\x02'+pack(raw)+b'.')).load()
        fp,parsed=m.interface(value)
        self.assertEqual(fp,dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        self.assertEqual(parsed['graph_definition']['bytes'],1048600)
        self.assertEqual(len(parsed['aliases']),72)

    def test_full_archive_digest_checked_before_parse(self):
        with self.assertRaisesRegex(ValueError,'archive SHA256'):read(archive_bytes(),binding_changes={'sha256':'0'*64})

    def test_archive_and_pickle_caps(self):
        data=archive_bytes()
        for cap in ('archive_bytes','pickle_bytes','members','code_bytes'):
            with self.subTest(cap=cap),self.assertRaises(ValueError):read(data,limit_changes={cap:1})

    def test_trailing_pickle_bytes_rejected(self):
        with self.assertRaisesRegex(ValueError,'trailing'):read(archive_bytes(trailing=b'other'))

    def test_unknown_pickle_global_not_executed(self):
        with self.assertRaisesRegex(ValueError,'Unsupported pickle global'):
            m.BoundRuntimeReader(io.BytesIO(b'\x80\x02cos\nsystem\n.')).load()

    def test_extension_registry_cannot_bypass_class_filter(self):
        with self.assertRaisesRegex(ValueError,'extension registry'):
            m.BoundRuntimeReader(io.BytesIO(b'\x80\x02\x82\x01.')).load()

    def test_wrong_WLT_rank_and_key_coverage_rejected(self):
        obj=fixture();loader=obj.state['nxd_model'].state['weight_loader']
        parts=list(loader.state);parts[-1]=1;loader.state=tuple(parts)
        with self.assertRaisesRegex(ValueError,'WLT TP2'):read(archive_bytes(obj))

    def test_missing_bucket_rejected(self):
        obj=fixture();obj.state['nxd_model'].state['models'].state[m.TAGS[0]].state['models'].pop()
        with self.assertRaisesRegex(ValueError,'six original buckets'):read(archive_bytes(obj))

    def test_reordered_or_missing_route_rejected(self):
        obj=fixture();obj.state['nxd_model'].state['input_shape_map'].popitem()
        with self.assertRaisesRegex(ValueError,'All12'):read(archive_bytes(obj))

    def test_exact_saved_equality_retains_no_semantic_qualification(self):
        value=read(archive_bytes());result=m.compare_descriptors(value,deepcopy(value))
        self.assertTrue(result['strict_saved_components_equal'])
        for key in ('loaded_values_observed','graph_weight_ABI_semantics_proven','layout_semantics_proven',
                    'inference_admitted','full_model_qualified','new_speedup_claim'):
            self.assertIs(result[key],False)

    def test_NEFF_byte_change_breaks_saved_identity(self):
        a=read(archive_bytes());b=deepcopy(a);b['wlt']['neff']['sha256']='0'*64
        self.assertFalse(m.compare_descriptors(a,b)['complete_saved_WLT_equal'])

    def test_same_key_shape_but_reordered_binding_is_not_exact(self):
        a=read(archive_bytes());b=deepcopy(a);b['graphs'][0]['weight_inputs'].reverse()
        self.assertFalse(m.compare_descriptors(a,b)['all12_ordered_weight_interfaces_equal'])

    def test_script_change_is_not_exact_path(self):
        a=read(archive_bytes());b=deepcopy(a)
        b['all_serialized_script_records']['model/code/__torch__/spmd.py']['sha256']='0'*64
        self.assertFalse(m.compare_descriptors(a,b)['strict_saved_components_equal'])

    def test_forged_qualification_rejected(self):
        a=read(archive_bytes());b=deepcopy(a);b['layout_semantics_proven']=True
        with self.assertRaises(ValueError):m.compare_descriptors(a,b)

    def test_guard_failure_propagates_before_IO(self):
        def stop():raise RuntimeError('owning guard revoked')
        with self.assertRaisesRegex(RuntimeError,'revoked'):
            m.describe(None,{},stop)


if __name__=='__main__':
    original=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name.split('.')[0]in ('torch','torch_neuronx','torch_xla','numpy','neuronx_distributed','neuronx_distributed_inference'):
            raise AssertionError('No native imports in source-only tests: '+name)
        return original(name,*args,**kwargs)
    builtins.__import__=guarded_import
    def audit(event,args):
        if event in ('subprocess.Popen','os.system','socket.connect','socket.bind','os.listdir','os.scandir'):
            raise AssertionError('No actual runtime/process/filesystem traversal')
        if event=='open':
            name,mode,flags=args
            if not isinstance(name,str)or not name.endswith(('.py','.pyc')):
                raise AssertionError('No actual model/metadata/payload file read')
            if flags&(1|2|64|512):raise AssertionError('No file mutation')
    sys.addaudithook(audit)
    unittest.main(verbosity=2)
