"""Meaningful fixed diagnostic source/mock checks; no native or ownership IO."""
import ast
import base64
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
BLOCKED={'ctypes','_ctypes','torch','torch_xla','torch_neuronx','neuronxcc','numpy','ml_dtypes','libneuronxla','transformers','vllm'}
class BlockNative:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in BLOCKED:raise AssertionError('No native import: '+name)
assert not BLOCKED.intersection(sys.modules)
sys.meta_path.insert(0,BlockNative())
from research import qwen35_slot_special_value_protocol_v2 as d
from research import qwen35_slot_special_value_leaf_v2 as leaf
import qwen35_slot_special_value_serial_contract_v2 as c
import qwen35_slot_special_value_serial_v2 as serial
import qwen35_predicated_slot_serial_contract_v3 as original_contract
import qwen35_predicated_slot_serial_v3 as original_serial
from qwen35_sampler_runtime_io_v12 import RetainedLock
from qwen35_sampler_runtime_service_v12 import Service
from qwen35_package_preflight_history_v2 import History

def artifact_fixture():
    config=dict(func_name='qwen35_conv_identity_dma',platform_target='trn1',mac_count=0,
        has_collectives=False,grid=[],func_literal='explicit source/mock fixture only; no native compilation performed')
    return hlo_fixture(config)

def varint(value):
    out=bytearray()
    while value>127:out.append((value&127)|128);value>>=7
    out.append(value);return bytes(out)

def field(number,value):
    return varint(number*8+2)+varint(len(value))+value

def hlo_fixture(config,*,opcode=b'custom-call',target=b'AwsNeuronCustomNativeKernel',extra=b'',token=None):
    if token is None:token=base64.b64encode(json.dumps(config).encode())
    instruction=field(2,opcode)+field(28,target)+field(43,token)+extra
    return field(3,field(2,instruction))

REAL_HLO=Path('/mnt/nvme/neuron-work/qwen35-slot-special-value-r1-v1-20261112/conv-special-values/artifacts/identity/model/graph.hlo')
REAL_HLO_SHA='afaa40dae2d7555e2dfa8121a2404bd5e8816a833d88d462220b10dba5266144'
SDK_HLO=Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/pyhlo/service/hlo_pb2.py')
SDK_HLO_SHA='8d332b4180bc80e2b376a3018de0013920e6878b3eb260c52540ef07bc5b0922'

class HloParserTests(unittest.TestCase):
    def real_hlo(self):
        payload=REAL_HLO.read_bytes()
        self.assertEqual(len(payload),10244);self.assertEqual(hashlib.sha256(payload).hexdigest(),REAL_HLO_SHA)
        return payload
    def test_actual_saved_hlo_exposes_old_length_byte_bug_and_new_exact_fields_pass(self):
        from research import qwen35_slot_special_value_protocol_v1 as previous
        payload=self.real_hlo()
        with self.assertRaisesRegex(ValueError,'One complete NKI'):previous.identity_hlo_proof(payload)
        proof=d.identity_hlo_proof(payload)
        self.assertEqual(proof['backend_config_bytes'],9368)
        self.assertEqual(proof['backend_config_sha256'],'3cbe774e4b2ed1dd055e2ab2098a9b4d974fd45b1d406ad7a854ba61c8f0fbfb')
        self.assertEqual(proof['func_literal_sha256'],'6f8cfaf7f7946270c60495565a64237133f75fe222eb19aec84f147949cab8ed')
        self.assertEqual((proof['computations'],proof['instructions'],proof['nki_custom_calls']),(3,9,1))
        self.assertIn(b'\xda\x02\x98IeyJ',payload)
    def test_fixed_sdk_descriptors_define_exact_message_field_chain(self):
        payload=SDK_HLO.read_bytes();self.assertEqual(hashlib.sha256(payload).hexdigest(),SDK_HLO_SHA)
        blobs=[ast.literal_eval(node.args[0]) for node in ast.walk(ast.parse(payload))
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='AddSerializedFile']
        self.assertEqual(len(blobs),1);messages={}
        for blob in d._field_values(d._protobuf_fields(blobs[0]),4,2):
            fs=d._protobuf_fields(blob);name=d._field_values(fs,1,2)[0].decode();messages[name]={}
            for item in d._field_values(fs,2,2):
                row=d._protobuf_fields(item);key=d._field_values(row,1,2)[0].decode()
                messages[name][key]=(d._field_values(row,3,0)[0],d._field_values(row,4,0)[0],d._field_values(row,5,0)[0])
        self.assertEqual(messages['HloModuleProto']['computations'],(3,3,11))
        self.assertEqual(messages['HloComputationProto']['instructions'],(2,3,11))
        self.assertEqual({k:messages['HloInstructionProto'][k] for k in ('opcode','custom_call_target','backend_config')},
                         dict(opcode=(2,1,9),custom_call_target=(28,1,9),backend_config=(43,1,12)))
    def test_actual_hlo_rejects_missing_wrong_duplicate_or_detached_nki_identity(self):
        payload=self.real_hlo()
        mutations=[payload.replace(b'AwsNeuronCustomNativeKernel',b'AwsNeuronCustomWrong_Kernel'),
                   payload.replace(b'custom-call',b'custom-fall'),payload+payload,
                   field(1,b'custom-call AwsNeuronCustomNativeKernel '+base64.b64encode(b'{"func_name":"fake"}'))]
        for changed in mutations:
            with self.subTest(length=len(changed)),self.assertRaises(ValueError):d.identity_hlo_proof(changed)
    def test_duplicate_config_opcode_target_and_json_keys_rejected(self):
        config=dict(func_name='qwen35_conv_identity_dma',platform_target='trn1',mac_count=0,
            has_collectives=False,grid=[],func_literal='source fixture')
        mutations=[hlo_fixture(config,extra=field(43,base64.b64encode(json.dumps(config).encode()))),
                   hlo_fixture(config,extra=field(2,b'custom-call')),hlo_fixture(config,extra=field(28,b'AwsNeuronCustomNativeKernel')),
                   hlo_fixture(config,token=base64.b64encode(json.dumps(config).replace('"mac_count": 0','"mac_count": 9, "mac_count": 0').encode())),
                   hlo_fixture(dict(config,mac_count=False)),hlo_fixture(dict(config,mac_count=1)),
                   hlo_fixture(dict(config,func_name='other')),hlo_fixture(dict(config,has_collectives=True)),
                   hlo_fixture(dict(config,platform_target='trn2')),hlo_fixture(dict(config,grid=[1])),
                   hlo_fixture(dict(config,func_literal='')),hlo_fixture(config,opcode=b'parameter'),
                   hlo_fixture(config,token=b'!!!!'),hlo_fixture(config,token=base64.b64encode(json.dumps(config).encode())+b'='),
                   hlo_fixture(config,token=base64.b64encode(json.dumps(config).replace('"mac_count": 0','"mac_count": NaN').encode()))]
        for changed in mutations:
            with self.subTest(length=len(changed)),self.assertRaises(ValueError):d.identity_hlo_proof(changed)
    def test_complete_wire_boundaries_reject_truncation_and_unsupported_or_overlong_encoding(self):
        payload=self.real_hlo()
        for changed in (payload[:-1],payload[:202],payload+b'\x80',b'\x00',b'\x1b',b'\x1a\xff',
                        b'\x1a\xff\xff\xff\xff\xff\xff\xff\xff\xff\x02',b'\x9a\x00\x00'):
            with self.subTest(length=len(changed)),self.assertRaises(ValueError):d.identity_hlo_proof(changed)

class Tensor:
    def __init__(self,payload,dtype='torch.uint8',shape=None):
        self.payload=bytes(payload);self.dtype=dtype;self.shape=shape or [len(payload)]
    def view(self,dtype):return Tensor(self.payload,dtype,self.shape)
    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],(list,tuple)):shape=shape[0]
        return Tensor(self.payload,self.dtype,list(shape))
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):return self
    def clone(self):return Tensor(self.payload,self.dtype,self.shape[:])
    def numpy(self):return NS(tobytes=lambda:self.payload)

class Module:
    def eval(self):return self

class LeafTests(unittest.TestCase):
    def run_leaf(self,root,*,injected=None):
        calls=[];events=[];saved=[];artifacts=[];identity_trace=[]
        references={}
        for arm in ('reference','candidate'):
            path=root.parent/(arm+'-fixture.pt');path.write_bytes(('source-mock-'+arm).encode())
            references[arm]=d.file_ref(path)
        class Compiled:
            code='def forward(x):\n    return ops.neuron.forward_v2(x, model)\n'
            def __init__(self,arm):self.arm=arm
            def __call__(self,*args):
                calls.append(self.arm)
                if injected==self.arm:raise RuntimeError('injected runtime failure')
                payload=args[0].payload
                if self.arm=='reference':payload=d.original.normalized_bytes(payload)
                return Tensor(payload,'torch.bfloat16',d.SHAPE)
            def save(self,path):Path(path).write_bytes(b'source-mock-identity-model-not-native')
        def trace(module,example,**kwargs):
            actual=module.forward(*example);identity_trace.append(actual.payload)
            p=Path(kwargs['compiler_workdir'])/'model';p.mkdir();(p/'graph.hlo').write_bytes(artifact_fixture())
            return Compiled('identity')
        def load(path):
            return Compiled(next((arm for arm,row in references.items() if row['path']==path),'identity'))
        fake_torch=NS(Tensor=Tensor,uint8='torch.uint8',int64='torch.int64',bfloat16='torch.bfloat16',
            set_num_threads=lambda n:None,from_numpy=lambda value:Tensor(value.payload),
            nn=NS(Module=Module),inference_mode=contextlib.nullcontext,jit=NS(load=load))
        fake_np=NS(uint8='uint8',frombuffer=lambda payload,**kwargs:NS(copy=lambda:NS(payload=bytes(payload))))
        fake_neuron=NS(trace=trace,move_trace_to_device=lambda compiled,core:None)
        fake_kernel=NS(qwen35_conv_identity_dma=lambda value:value.clone())
        def save(report):saved.append(copy.deepcopy(report))
        def guard(event):events.append(event)
        mocked={'torch':fake_torch,'numpy':fake_np,'torch_neuronx':fake_neuron,
                'research.qwen35_slot_identity_dma_nki_v1':fake_kernel}
        with patch.dict(sys.modules,mocked),patch.object(d,'ARTIFACTS',references),patch.object(d,'verify_inputs'),patch.object(leaf.frozen,'verify_sources'):
            if injected:
                with self.assertRaisesRegex(RuntimeError,'injected runtime'):leaf.run_diagnostic(root,guard=guard,save_report=save,record_artifact=artifacts.append)
                self.assertFalse(saved[-1]['complete']);return calls,events,saved,artifacts
            report=leaf.run_diagnostic(root,guard=guard,save_report=save,record_artifact=artifacts.append)
            self.assertEqual(d.validate_report(report,root)['calls'],6)
            self.assertEqual(identity_trace,[d.inputs('original_raw')['cache']])
            self.assertEqual(calls,['reference']*2+['candidate']*2+['identity']*2)
            self.assertFalse(report['calls'][0]['bitwise_equal'])
            self.assertFalse(report['calls'][1]['bitwise_equal'])
            self.assertTrue(all(row['bitwise_equal'] for row in report['calls'][2:]))
            self.assertEqual(report['calls'][0]['different_bytes'],139)
            self.assertEqual(len(d.checked(report['calls'][0]['differences'])),139*4)
            for row in report['calls']:self.assertEqual(len(d.checked(row['output'])),d.BYTES)
            self.assertEqual(events[events.index('before_native_imports')+1],'before_compile:identity:diagnostic')
            self.assertEqual(len(artifacts),1)
            # Every byte/offset is recomputed, not merely trusted by count/hash.
            mutations=[lambda v:v['calls'].pop(),lambda v:v['calls'][0].__setitem__('different_bytes',0),
                lambda v:v['calls'][0].__setitem__('bitwise_equal',True),
                lambda v:v['calls'][2].__setitem__('arm','reference'),
                lambda v:v.__setitem__('qualification_matrix_completed',True),
                lambda v:v['graphs'][0].__setitem__('moved_to_device',False),
                lambda v:v['graphs'][0]['post_calls'].__setitem__('sha256','f'*64)]
            for mutation in mutations:
                value=copy.deepcopy(report);mutation(value)
                with self.assertRaises((ValueError,KeyError)):d.validate_report(value,root)
            # Corrupt a middle offset after its ref is updated: full recompute
            # still rejects, independently of the first-16 preview and count.
            value=copy.deepcopy(report);p=Path(value['calls'][0]['differences']['path']);payload=bytearray(p.read_bytes())
            payload[100:104]=struct.pack('<I',99999);p.write_bytes(payload);value['calls'][0]['differences']=d.file_ref(p)
            with self.assertRaises(ValueError):d.validate_report(value,root)
        return calls,events,saved,artifacts

    def test_real_leaf_source_six_mock_calls_keeps_candidate_after_reference_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'diag';root.mkdir();self.run_leaf(root)
    def test_runtime_exception_aborts_after_preserving_prior_complete_captures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'diag';root.mkdir();calls,events,saved,artifacts=self.run_leaf(root,injected='candidate')
            self.assertEqual(calls,['reference','reference','candidate'])
            self.assertEqual(len(saved[-1]['calls']),2)
            self.assertTrue((root/'original_raw-reference-output.raw').exists())
            self.assertFalse((root/'original_raw-candidate-output.raw').exists())

class PureTests(unittest.TestCase):
    def test_original_failure_and_whole_input_hashes(self):
        evidence=d.original.evidence();self.assertEqual(evidence['derived_different_bytes'],139)
        self.assertTrue(evidence['whole_sha256_match']);self.assertEqual(evidence['candidate_runtime_correctness_calls'],0)
        negative=struct.unpack('<49152H',d.inputs('negative_nan_raw')['cache'])
        self.assertEqual(sum(x&0xff80==0xff80 and bool(x&0x7f) for x in negative),8)
    def test_raw_outputs_are_exclusive_and_full_differences_not_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=d.inputs('original_raw')['cache'];changed=bytes(value^255 for value in original)
            row=d.record_call(root,'reference','original_raw',changed,d.SHAPE,'torch.bfloat16',d.inputs('original_raw'),lambda event:None)
            self.assertEqual(row['different_bytes'],98304);self.assertEqual(row['differences']['bytes'],393216)
            with self.assertRaises(ValueError):d.record_call(root,'reference','original_raw',changed,d.SHAPE,'torch.bfloat16',d.inputs('original_raw'),lambda event:None)
    def test_changed_artifact_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';path.write_bytes(b'original');reference=d.file_ref(path);path.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError,'SHA256'):d.checked(reference)
    def test_identity_source_is_exact_fresh_hbm_load_store_no_host_return(self):
        tree=ast.parse((Q/'qwen35_slot_identity_dma_nki_v1.py').read_text());function,=[x for x in tree.body if isinstance(x,ast.FunctionDef)]
        calls=[ast.unparse(n.func) for n in ast.walk(function) if isinstance(n,ast.Call)]
        self.assertEqual(sorted(calls),sorted(['nki.jit','nl.ndarray','nl.arange','nl.arange','nl.static_range','nl.load','nl.store']))
        self.assertIn('buffer=nl.shared_hbm',ast.unparse(function));self.assertEqual(ast.unparse(function.body[-1]),'return output')
        self.assertEqual(d.identity_hlo_proof(artifact_fixture())['func_name'],'qwen35_conv_identity_dma')
        with self.assertRaises(ValueError):d.identity_hlo_proof(b'host return x')

def definitions(path):
    return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(Path(path).read_text()).body
            if isinstance(node,(ast.FunctionDef,ast.ClassDef))}

class WiringTests(unittest.TestCase):
    def test_successor_changes_only_parser_version_wiring_and_additive_tests(self):
        import re
        for stem in ('leaf','serial_contract','serial'):
            previous=(Q/('qwen35_slot_special_value_'+stem+'_v1.py')).read_text()
            successor=(Q/('qwen35_slot_special_value_'+stem+'_v2.py')).read_text()
            normalized=re.sub(r'(qwen35_slot_special_value_[a-z_]+)_v2',r'\1_v1',successor)
            self.assertEqual(previous,normalized,stem)
        before=definitions(Q/'qwen35_slot_special_value_protocol_v1.py')
        successor=(Q/'qwen35_slot_special_value_protocol_v2.py').read_text()
        normalized=re.sub(r'(qwen35_slot_special_value_[a-z_]+)_v2',r'\1_v1',successor)
        after={node.name:ast.dump(node,include_attributes=False) for node in ast.parse(normalized).body
               if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        for name in before:
            if name!='identity_hlo_proof':self.assertEqual(before[name],after[name],name)
    def test_original_lifecycle_capture_resources_and_environment_reused(self):
        before=definitions(Q/'qwen35_predicated_slot_serial_v3.py');after=definitions(Q/'qwen35_slot_special_value_serial_v2.py')
        for name in ('capture_layout','only_supervisor','group_members','check_group','read_history_reference','loaded_sources','entry_argv','supervisor_argv'):
            self.assertEqual(before[name],after[name],name)
        before=definitions(Q/'qwen35_predicated_slot_serial_contract_v3.py');after=definitions(Q/'qwen35_slot_special_value_serial_contract_v2.py')
        for name in ('environment','check_environment','check_process_settings','bytecode_prefix','update_json','current_layout','scan_once','scan_output','resources'):
            self.assertEqual(before[name],after[name],name)
        self.assertIs(serial.History,History);self.assertIs(serial.Service,Service);self.assertIs(serial.RetainedLock,RetainedLock)
        self.assertEqual(c.LIMITS,original_contract.LIMITS);self.assertEqual(c.PROPERTIES,original_contract.PROPERTIES)
        self.assertEqual(len(original_contract.LAYOUTS),8);self.assertEqual(c.LAYOUTS,(d.NAME,))
        self.assertEqual(original_contract.protocol.counts()['total']['total_runtime_calls'],46368)
    def test_actual_leaf_guard_and_outer_cleanup_are_original_source(self):
        trees=[ast.parse((Q/name).read_text()) for name in ('qwen35_predicated_slot_serial_v3.py','qwen35_slot_special_value_serial_v2.py')]
        layouts=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='layout') for tree in trees]
        guards=[next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='guard') for node in layouts]
        self.assertEqual(ast.dump(guards[0],include_attributes=False),ast.dump(guards[1],include_attributes=False))
        handlers=[next(n for n in node.body if isinstance(n,ast.Try)).handlers for node in layouts]
        self.assertEqual([ast.dump(n,include_attributes=False) for n in handlers[0]],
                         [ast.dump(n,include_attributes=False) for n in handlers[1]])
        final=[]
        for tree in trees:
            outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='outer')
            final.append(next(n for n in outer.body if isinstance(n,ast.Try)).finalbody)
        class NormalizeDiagnosticReceipt(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value=='qwen35_slot_special_value_outer_terminal_v2':node.value='qwen35_predicated_slot_outer_terminal_v3'
                return node
            def visit_Call(self,node):
                self.generic_visit(node);node.keywords=[x for x in node.keywords if x.arg!='qualification_matrix_completed'];return node
        clean=NormalizeDiagnosticReceipt().visit(ast.Module(body=copy.deepcopy(final[1]),type_ignores=[]))
        self.assertEqual(ast.dump(ast.Module(body=final[0],type_ignores=[]),include_attributes=False),ast.dump(clean,include_attributes=False))
    def test_fixed_diagnostic_contract_rejects_qualification_or_resource_shrink(self):
        spec=dict(schema='qwen35_slot_special_value_serial_spec_v2',run_name='mock',
            output_root='/mnt/nvme/neuron-work/qwen35-slot-special-value-mock',history_overlay=dict(path='/overlay',sha256='1'*64),
            current_serial_catalog=dict(path='/catalog',sha256='2'*64),source_sha256={str(Q/'qwen35_slot_special_value_serial_v2.py'):'3'*64},
            input_extension={},running_handoff_path=str(Q/'continuation-handoff-mock.json'),lfm_r2_scope=dict(path='/lfm',sha256='4'*64),
            source_delivery=dict(path='/delivery',sha256='5'*64),primitive_manifest=c.MANIFEST,workload=d.workload(),
            limits=copy.deepcopy(c.LIMITS),lock_identity=c.LOCK_IDENTITY,native_python=c.PYTHON,automatic_successor=False)
        reference=dict(path='/spec',sha256='6'*64);review=dict(schema='qwen35_sampler_runtime_source_review_v1',reviewer='/root',passed=True,
            blockers=[],spec=reference,source_sha256=spec['source_sha256'],launch_authorized=True)
        c.validate(spec,review,reference)
        for change in (lambda s:s['workload'].__setitem__('calls',5),lambda s:s['workload'].__setitem__('full_model_qualified',True),
                       lambda s:s['limits'].__setitem__('memory_max_bytes',7*c.GIB),
                       lambda s:s.__setitem__('workload',original_contract.protocol.workload())):
            changed=copy.deepcopy(spec);change(changed)
            with self.assertRaises(ValueError):c.validate(changed,review,reference)
    def test_no_current_binding_in_source_and_separate_schema(self):
        source=(Q/'qwen35_slot_special_value_serial_v2.py').read_text()
        self.assertIn('run_diagnostic(work,',source);self.assertNotIn('run_case(',source)
        self.assertNotIn('v102',source);self.assertNotIn('v569',source)
        self.assertIn('qualification_matrix_completed=False',source)
        spec=dict(run_name='mock',output_root='/mnt/nvme/neuron-work/qwen35-slot-special-value-mock')
        self.assertTrue(c.names(spec)['coordinator'].endswith('-slot-special-values'))
    def test_exact_one_diagnostic_capture_and_no_second_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);owner=dict(process=dict(pid=1,cgroup='0::/mock'),service_owner={})
            def capture(*args):
                work=args[2];serial.once(work/'leaf-result.json',dict(layout=d.NAME));return {'source_mock':True}
            with patch.object(serial,'only_supervisor'),patch.object(serial,'capture_layout',side_effect=capture) as execute,\
                    patch.object(serial,'validate_layout',return_value={'layout':d.NAME}):
                self.assertEqual(serial.run_layouts({'output_root':str(root)},{'path':'/mock','sha256':'a'*64},owner,lambda:None),[{'layout':d.NAME}])
                self.assertEqual(execute.call_count,1)
    def test_real_source_inventory_contains_kernel_leaf_and_original_sdk_environment(self):
        from prepare_qwen35_slot_special_value_serial_v2 import static_local_sources
        pins=static_local_sources()
        for name in ('qwen35_slot_identity_dma_nki_v1.py','qwen35_slot_special_value_leaf_v2.py','qwen35_predicated_slot_sdk_environment_v1.py'):
            self.assertIn(str(Q/name),pins)
        self.assertIn(str(Q.parent/'qwen35_predicated_slot_device_leaf_v1.py'),pins)

if __name__=='__main__':
    result=unittest.main(exit=False)
    assert not BLOCKED.intersection(sys.modules)
    raise SystemExit(not result.result.wasSuccessful())
