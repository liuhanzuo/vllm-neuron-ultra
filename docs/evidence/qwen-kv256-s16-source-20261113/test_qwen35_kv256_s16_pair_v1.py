"""Stdlib-only synthetic HLO, metadata, and source-contract checks."""
import ast
import base64
import copy
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parent
BLOCKED={'torch','torch_neuronx','torch_xla','neuronxcc','numpy','ctypes','nki','vllm','vllm_neuron'}
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in BLOCKED:raise AssertionError('Forbidden native import: '+fullname)
assert not BLOCKED.intersection(sys.modules)
sys.meta_path.insert(0,NoNative())
sys.path.insert(0,str(ROOT))
import qwen35_kv256_s16_pair_hlo_v1 as hlo
import qwen35_kv256_s16_pair_leaf_v1 as leaf
from qwen35_kv256_alias_diagnostic_bytes_v1 import compare


def vint(value):
    out=bytearray()
    while value>=128:out.append((value&127)|128);value>>=7
    out.append(value);return bytes(out)
def field(number,value):
    if type(value)is int:return vint(number<<3)+vint(value)
    if type(value)is str:value=value.encode()
    return vint((number<<3)|2)+vint(len(value))+value
def shp(kind,dims=(),children=()):
    types={'bf16':16,'s16':3,'s32':4,'s64':5,'tuple':13}
    return field(2,types[kind])+b''.join(field(3,x)for x in dims)+b''.join(field(4,x)for x in children)
def encode_shape(shape):return shp(shape[0],shape[1],tuple(encode_shape(c)for c in shape[2]))
def node(identifier,opcode,shape,operands=(),extra=()):
    return dict(id=identifier,op=opcode,shape=shape,operands=tuple(operands),extra=list(extra))
def enc_node(n):
    return (field(1,'node'+str(n['id']))+field(2,n['op'])+field(3,encode_shape(n['shape']))
            +field(35,n['id'])+b''.join(field(36,i)for i in n['operands'])
            +b''.join(field(k,v)for k,v in n['extra']))
def bounds(start,end):
    return [(17,b''.join(field(k,v)for k,v in ((1,a),(2,b),(3,c))))
            for a,b,c in ((0,4,1),(0,1,1),(start,end,1),(0,256,1))]


class Fixture:
    def __init__(self,arm='s16_bitcast_cat',called=False):
        self.arm=arm;self.called=called;self.alias_output=1;self.alias_parameter=0;self.alias_kind=2
        backend=base64.b64encode(json.dumps(dict(func_name='slot_cache_gather_predicated',platform_target='trn1',
                                               func_literal='synthetic source-only kernel')).encode())
        self.nodes=[node(1,'parameter',hlo.STATE,extra=[(9,0)]),node(2,'parameter',hlo.PREFIX,extra=[(9,1)]),
            node(3,'parameter',hlo.SLOTS,extra=[(9,2)]),node(4,'slice',hlo.PREFIX,[1],bounds(0,256)),
            node(5,'reshape',('bf16',(4,65536),()),[4]),node(6,'reshape',('bf16',(4,65536),()),[2]),
            node(7,'custom-call',('bf16',(4,65536),()),[5,6,3],[(28,b'AwsNeuronCustomNativeKernel'),(43,backend)]),
            node(8,'reshape',hlo.PREFIX,[7]),node(9,'slice',hlo.TAIL,[1],bounds(256,4096))]
        if arm=='s16_bitcast_cat':
            self.nodes += [node(10,'bitcast-convert',('s16',hlo.PREFIX[1],()),[8]),
                node(11,'bitcast-convert',('s16',hlo.TAIL[1],()),[9]),
                node(12,'concatenate',('s16',hlo.STATE[1],()),[10,11],[(14,2)]),
                node(13,'bitcast-convert',hlo.STATE,[12])]
        else:
            self.nodes += [node(13,'concatenate',hlo.STATE,[8,9],[(14,2)])]
        self.nodes += [node(14,'tuple',('tuple',(),(hlo.STATE,hlo.STATE)),[1,13])]

    def n(self,identifier):return next(x for x in self.nodes if x['id']==identifier)

    def encode(self):
        nodes=copy.deepcopy(self.nodes)
        comps=[]
        if self.called:
            # Move only the exact assembly to a real called computation. Call
            # operands remain original NKI prefix and original full-state tail.
            body=[n for n in nodes if n['id']in(10,11,12,13)]
            body=[node(8,'parameter',hlo.PREFIX,extra=[(9,0)]),node(9,'parameter',hlo.TAIL,extra=[(9,1)])]+body
            nodes=[n for n in nodes if n['id']not in(10,11,12,13)]
            nodes.append(node(13,'call',hlo.STATE,[8,9],[(38,2)]))
            comps.append(field(5,2)+field(6,13)+b''.join(field(2,enc_node(n))for n in body))
        comps.insert(0,field(5,1)+field(6,14)+b''.join(field(2,enc_node(n))for n in nodes))
        alias=field(1,field(1,self.alias_output)+field(2,self.alias_parameter)+field(4,self.alias_kind))
        return field(6,1)+b''.join(field(3,c)for c in comps)+field(8,alias)

    def proof(self):return hlo.assembly_hlo_proof(self.encode(),self.arm)


class HloTests(unittest.TestCase):
    def test_s16_full_entry_output_path(self):
        p=Fixture().proof()
        self.assertEqual((p['concat_dtype'],p['bitcast_converts'],p['numeric_converts_in_assembly']),('s16',3,0))
        self.assertEqual(p['state_alias_output'],1)
        self.assertTrue(p['exact_tail_slice']and p['original_writer_inputs'])

    def test_bf16_matched_control(self):
        p=Fixture('bf16_cat').proof()
        self.assertEqual((p['concat_dtype'],p['bitcast_converts']),('bf16',0))
        self.assertEqual(p['writer_func_literal_sha256'],Fixture().proof()['writer_func_literal_sha256'])

    def test_call_expansion_follows_actual_operands(self):
        p=Fixture(called=True).proof();self.assertEqual(p['computations'],2)
        self.assertEqual(p['bitcast_converts'],3)

    def test_numeric_convert_on_each_bitcast_edge_rejected(self):
        for identifier in (10,11,13):
            f=Fixture();f.n(identifier)['op']='convert'
            with self.subTest(identifier=identifier),self.assertRaises(ValueError):f.proof()

    def test_float_concat_with_unused_s16_markers_rejected(self):
        f=Fixture();f.n(12)['shape']=hlo.STATE
        with self.assertRaises(ValueError):f.proof()

    def test_wrong_concat_axis_rejected(self):
        f=Fixture();f.n(12)['extra']=[(14,3)]
        with self.assertRaises(ValueError):f.proof()

    def test_swapped_concat_operands_rejected(self):
        f=Fixture();f.n(12)['operands']=(11,10)
        with self.assertRaises(ValueError):f.proof()

    def test_wrong_tail_bound_rejected(self):
        f=Fixture();f.n(9)['extra']=bounds(255,4095)
        with self.assertRaises(ValueError):f.proof()

    def test_wrong_prefix_bound_rejected(self):
        f=Fixture();f.n(4)['extra']=bounds(1,257)
        with self.assertRaises(ValueError):f.proof()

    def test_wrong_tail_source_rejected(self):
        f=Fixture();f.n(9)['operands']=(2,)
        with self.assertRaises(ValueError):f.proof()

    def test_numeric_convert_before_prefix_writer_rejected(self):
        f=Fixture();f.n(5)['op']='convert'
        with self.assertRaises(ValueError):f.proof()

    def test_changed_writer_target_rejected(self):
        f=Fixture();f.n(7)['extra'][0]=(28,b'OtherKernel')
        with self.assertRaises(ValueError):f.proof()

    def test_wrong_writer_source_function_rejected(self):
        f=Fixture();f.n(7)['extra'][1]=(43,base64.b64encode(json.dumps(dict(func_name='other',platform_target='trn1',func_literal='x')).encode()))
        with self.assertRaises(ValueError):f.proof()

    def test_wrong_writer_updates_input_rejected(self):
        f=Fixture();f.n(7)['operands']=(5,5,3)
        with self.assertRaises(ValueError):f.proof()

    def test_alias0_and_wrong_state_and_may_alias_rejected(self):
        for key,value in (('alias_output',0),('alias_parameter',1),('alias_kind',1)):
            f=Fixture();setattr(f,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):f.proof()

    def test_widened_slot_dtype_rejected(self):
        f=Fixture();f.n(3)['shape']=('s64',(4,),())
        with self.assertRaises(ValueError):f.proof()

    def test_missing_tail_bitcast_rejected(self):
        f=Fixture();f.n(12)['operands']=(10,9)
        with self.assertRaises(ValueError):f.proof()

    def test_scalar_wire_overflow_or_truncation_rejected(self):
        for data in (b'\x80',b'\x80\x00',b'\xff'*10+b'\x00',b'\x1a\xff\xff\x7f'):
            with self.subTest(data=data),self.assertRaises(ValueError):hlo._protobuf_fields(data)

    def test_duplicate_instruction_rejected(self):
        f=Fixture();f.nodes.append(copy.deepcopy(f.n(10)))
        with self.assertRaises(ValueError):f.proof()

    def test_operand_cycle_rejected(self):
        f=Fixture();f.n(10)['operands']=(10,)
        with self.assertRaises(ValueError):f.proof()

    def test_no_optimized_lowering_or_old_gather_overclaim(self):
        p=Fixture().proof()
        self.assertFalse(p['optimized_backend_instructions_proven']or p['old_gather_semantics_proven'])


class SourceTests(unittest.TestCase):
    def test_fixed_sdk_descriptor_field_numbers_used_by_proof(self):
        path=Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/pyhlo/service/hlo_pb2.py')
        tree=ast.parse(path.read_text())
        literal=next(n.args[0]for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='AddSerializedFile')
        descriptor=hlo._protobuf_fields(ast.literal_eval(literal));messages={}
        def collect(raw,parent=''):
            fs=hlo._protobuf_fields(raw);name=hlo.blob(fs,1).decode();qualified=parent+name
            fields={hlo.blob(hlo._protobuf_fields(f),1).decode():hlo.scalar(hlo._protobuf_fields(f),3)
                    for f in hlo._field_values(fs,2,2)}
            messages[qualified]=fields
            for nested in hlo._field_values(fs,3,2):collect(nested,qualified+'.')
        for raw in hlo._field_values(descriptor,4,2):collect(raw)
        expected={'HloModuleProto':dict(computations=3,entry_computation_id=6,input_output_alias=8),
            'HloComputationProto':dict(instructions=2,id=5,root_id=6),
            'HloInstructionProto':dict(opcode=2,shape=3,parameter_number=9,tuple_index=13,dimensions=14,
                slice_dimensions=17,custom_call_target=28,id=35,operand_ids=36,control_predecessor_ids=37,
                called_computation_ids=38,backend_config=43),
            'HloInputOutputAliasProto':dict(entries=1),
            'HloInputOutputAliasProto.AliasEntryProto':dict(output_shape_index=1,parameter_number=2,parameter_shape_index=3,kind=4)}
        for name,fields in expected.items():
            self.assertEqual({k:messages[name][k]for k in fields},fields)

    def test_workload_is_extra_two_by_three_under_original_limits(self):
        w=leaf.workload()
        self.assertEqual([g['name']for g in w['graphs']],['bf16_cat','s16_bitcast_cat'])
        self.assertEqual((w['calls'],w['new_graphs'],w['saved_graphs']),(6,2,0))
        self.assertEqual(w['profiles'],['raw','finite','row_tagged_raw'])
        self.assertEqual((w['complete_owned_output_cap_bytes'],w['per_leaf_seconds']),(2**30,900))
        self.assertEqual(sum(w['output_class_caps'].values()),2**30)
        self.assertTrue(all(g['alias']==1 for g in w['graphs']))
        self.assertFalse(w['full_geometry_qualified']or w['full_model_qualified']or w['new_speedup_claim'])

    def test_same_fixed_flags_as_original_diagnostic(self):
        source=ast.parse((ROOT/'qwen35_kv256_alias_diagnostic_leaf_v1.py').read_text())
        flags=next(ast.literal_eval(n.value)for n in source.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='FLAGS'for t in n.targets))
        self.assertEqual(leaf.FLAGS,flags)

    def test_exact_original_writer_source_pins(self):
        source=ast.parse((ROOT/'qwen35_kv256_alias_diagnostic_leaf_v1.py').read_text())
        fixed=next(n.value for n in source.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='FIXED_SOURCES'for t in n.targets))
        self.assertEqual(set(leaf.FIXED_SOURCES.values()),{ast.literal_eval(v)for v in fixed.values})

    def test_future_leaf_imports_are_inert(self):
        self.assertFalse(BLOCKED.intersection(sys.modules))
        for filename in ('qwen35_kv256_s16_pair_leaf_v1.py','qwen35_kv256_s16_pair_hlo_v1.py'):
            tree=ast.parse((ROOT/filename).read_text())
            self.assertFalse(any(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)for n in tree.body))

    def test_preserved_complete_raw_difference_positions_on_tiny_fixture(self):
        expected=bytes.fromhex('0080817fc17f013f');actual=bytes.fromhex('0000c07fc07f013f')
        p,bits=compare(actual,expected,1,rows=1,tokens=2,width=2)
        self.assertEqual(p['differing_words'],3)
        self.assertEqual(p['word_class_transitions'],{'negative_zero->positive_zero':1,'signaling_nan->quiet_nan':1,'quiet_nan->quiet_nan':1})
        self.assertTrue(p['full_difference_positions_retained'])
        for i,(a,b)in enumerate(zip(actual,expected)):self.assertEqual(bool(bits[i//8]&(1<<(i%8))),a!=b)

    def test_compile_uses_same_alias_map_and_flags(self):
        tree=ast.parse((ROOT/'qwen35_kv256_s16_pair_leaf_v1.py').read_text())
        calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='trace']
        self.assertEqual(len(calls),1)
        kw={k.arg:ast.unparse(k.value)for k in calls[0].keywords}
        self.assertEqual(kw['input_output_aliases'],'assembly_control.alias_map(source)')
        self.assertEqual(kw['compiler_args'],'list(FLAGS)')

    def test_input_hlo_is_sdk_actual_fixed_written_path(self):
        sdk=Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/xla_impl/trace.py')
        tree=ast.parse(sdk.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='setup_compiler_dirs')
        literals={n.value for n in ast.walk(fn)if isinstance(n,ast.Constant)and isinstance(n.value,str)}
        self.assertIn('model',literals);self.assertIn('graph.hlo',literals)
        self.assertTrue(any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='SerializeToString'for n in ast.walk(fn)))


def metadata_fixture():
    """Synthetic metadata only; no actual full tensor/raw allocation or file IO."""
    root=Path('/synthetic-only-kv256-pair');sha='0'*64;bitmap='1'*64
    def ref(path,size,hash_=sha):return dict(path=str(path),bytes=size,sha256=hash_)
    inputs={p:dict(cache=ref(root/'inputs'/(p+'-cache.raw'),leaf.STATE_BYTES),
        old=ref(root/'inputs'/(p+'-expected-old.raw'),leaf.STATE_BYTES),
        slots=ref(root/'inputs'/'slots-int32.raw',16),
        updates={'256':ref(root/'inputs'/(p+'-updates256.raw'),524288)})for p in leaf.PROFILES}
    graphs=[];calls=[]
    for g in leaf.GRAPHS:
        proof=Fixture(g['name']).proof()
        artifact=dict(path=str(root/'graphs'/g['name']/'model.pt'),sha256=sha,size_bytes=123)
        href=dict(path=str(root/'graphs'/g['name']/'model'/'graph.hlo'),sha256=proof['hlo_sha256'],size_bytes=proof['hlo_bytes'])
        graphs.append(dict(g,calls=3,physical_output_order=['old','new'],artifact=artifact,origin='fresh_control',hlo=href,hlo_proof=proof))
        for profile in leaf.PROFILES:
            refs=inputs[profile];directory=root/'calls'/(g['name']+'-'+profile)
            partitions=[dict(row=r,partition=part,start_byte=(r*4096+lo)*256*2,end_byte_exclusive=(r*4096+hi)*256*2,
                             differing_bytes=0,differing_words=0)
                        for r in range(4)for part,lo,hi in (('prefix',0,256),('tail',256,4096))]
            proofs={name:dict(bytes_compared=leaf.STATE_BYTES,actual=ref(directory/(name+'.raw'),leaf.STATE_BYTES),
                expected=refs['old'if name=='old'else'cache'],difference_bitmap=ref(directory/(name+'.diffbits'),leaf.STATE_BYTES//8,bitmap),
                actual_sha256=sha,expected_sha256=sha,bitmap_sha256=bitmap,bitmap_bytes=leaf.STATE_BYTES//8,
                partitions=copy.deepcopy(partitions),differing_bytes=0,differing_words=0,identical=True)
                for name in ('old','new','state')}
            calls.append(dict(graph=g['name'],profile=profile,inputs=refs,slots=[-1]*4,bucket=256,alias=1,
                physical_output_order=['old','new'],call_returned=True,proofs=proofs,host_inputs_unchanged=[True,True],
                host_inputs_after=[ref(directory/'host-input-0.raw',524288),ref(directory/'host-input-1.raw',16)],
                reset_exact=True,reset_sha256=sha,all_expected_bytes_equal=True))
    return dict(schema='qwen35_kv256_s16_pair_result_v1',workload=leaf.workload(),output_root=str(root),
        inputs=inputs,graphs=graphs,calls=calls,started_calls=6,returned_calls=6,complete=True,diagnostic_completed=True,
        mismatch_calls=0,s16_all_profiles_byte_exact=True,both_finite_profiles_byte_exact=True,
        original_missing_controls_replaced=False,raw_bytes_written=1,original_failed_r3_result_preserved=True,
        independent_actual_exit_audit_required=True,full_geometry_qualified=False,full_model_qualified=False,new_speedup_claim=False)


class MetadataTests(unittest.TestCase):
    def test_complete_synthetic_metadata_passes_without_IO(self):
        p=leaf.validate_report(metadata_fixture());self.assertEqual(p['calls'],6)
        self.assertFalse(p['full_geometry_qualified'])

    def test_missing_profile_rejected(self):
        v=metadata_fixture();v['calls'].pop()
        with self.assertRaises(ValueError):leaf.validate_report(v)

    def test_wrong_alias_and_order_rejected(self):
        for key,value in (('alias',0),('physical_output_order',['new','old'])):
            v=metadata_fixture();v['calls'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):leaf.validate_report(v)

    def test_missing_device_state_rejected(self):
        v=metadata_fixture();del v['calls'][0]['proofs']['state']
        with self.assertRaises(ValueError):leaf.validate_report(v)

    def test_missing_tail_partition_rejected(self):
        v=metadata_fixture();v['calls'][0]['proofs']['new']['partitions'].pop()
        with self.assertRaises(ValueError):leaf.validate_report(v)

    def test_missing_or_truncated_host_input_rejected(self):
        for change in ('missing','short'):
            v=metadata_fixture()
            if change=='missing':v['calls'][0]['host_inputs_after'].pop()
            else:v['calls'][0]['host_inputs_after'][0]['bytes']-=1
            with self.subTest(change=change),self.assertRaises(ValueError):leaf.validate_report(v)

    def test_changed_writer_literal_between_arms_rejected(self):
        v=metadata_fixture();v['graphs'][1]['hlo_proof']['writer_func_literal_sha256']='a'*64
        with self.assertRaises(ValueError):leaf.validate_report(v)

    def test_failed_s16_call_cannot_be_promoted(self):
        v=metadata_fixture();row=v['calls'][3];proof=row['proofs']['new']
        proof.update(identical=False,differing_words=1,differing_bytes=1,actual_sha256='a'*64)
        proof['actual']['sha256']='a'*64;proof['partitions'][1].update(differing_words=1,differing_bytes=1)
        row['all_expected_bytes_equal']=False;v['mismatch_calls']=1
        with self.assertRaises(ValueError):leaf.validate_report(v)
        v['s16_all_profiles_byte_exact']=False
        self.assertEqual(leaf.validate_report(v)['mismatch_calls'],1)

    def test_partial_returned_counter_rejected(self):
        v=metadata_fixture();v['returned_calls']=5
        with self.assertRaises(ValueError):leaf.validate_report(v)

    def test_qualification_or_substitution_claim_rejected(self):
        for key in ('full_geometry_qualified','full_model_qualified','new_speedup_claim','original_missing_controls_replaced'):
            v=metadata_fixture();v[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):leaf.validate_report(v)


if __name__=='__main__':unittest.main(verbosity=2)
