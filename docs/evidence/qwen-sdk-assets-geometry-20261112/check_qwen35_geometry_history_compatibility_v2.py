"""Source/mock checks for exact v7 text dispatch; never run real History."""
import ast
import copy
import hashlib
import importlib.abc
import io
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch','numpy','torch_xla','torch_neuronx','neuronxcc',
                'nki','ml_dtypes','transformers','vllm','libneuronxla','ctypes','_ctypes'}:
            raise AssertionError('Native import forbidden: '+name)

sys.meta_path.insert(0,RejectNative())
Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
import research
import qwen35_package_preflight_history_v2 as old_history
import qwen35_package_preflight_history_v3 as new_history
from qwen35_package_serial_runtime_v7 import source_text,io as transport
from qwen35_writer_geometry_serial_v1 import run as old_run,audit as old_audit,contract,protocol
from qwen35_writer_geometry_serial_v2 import run,audit

def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}


class PreservationTests(unittest.TestCase):
    def test_all_runtime_function_bodies_identical(self):
        self.assertEqual(functions(Path(old_run.__file__)),functions(Path(run.__file__)))

    def test_all_audit_function_bodies_identical(self):
        self.assertEqual(functions(Path(old_audit.__file__)),functions(Path(audit.__file__)))

    def test_history_only_changes_final_text_dispatcher(self):
        before=Path(old_history.__file__).read_text()
        after=Path(new_history.__file__).read_text()
        normalized=after.replace('source_text.verify_text_sources(partition, self.guard)',
                                 'sources.verify_text_sources(partition, self.guard)')
        def nodes(text):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef)}
        self.assertEqual(nodes(before),nodes(normalized))
        self.assertEqual(after.count('source_text.verify_text_sources(partition, self.guard)'),1)

    def test_inherited_full_scope_and_authority_unchanged(self):
        for name in ('physical_pair','freshness','unchanged'):
            self.assertIs(getattr(new_history.History,name),getattr(old_history.History,name))
        self.assertIs(new_history.Authority,old_history.Authority)
        self.assertIs(run.History,new_history.History);self.assertIs(audit.History,new_history.History)
        self.assertIs(audit.stage,run)

    def test_original_science_resource_service_and_lock_objects_reused(self):
        self.assertIs(run.protocol,protocol);self.assertIs(audit.protocol,protocol)
        for name in ('LIMITS','PROPERTIES','resources','verify_delivery','validate_report','RetainedLock','Service'):
            self.assertIs(getattr(run,name),getattr(old_run,name))
        self.assertEqual(protocol.counts()['total']['correctness_calls'],40000)
        self.assertEqual(len(run.LAYOUTS),8)

    def test_no_global_source_or_cap_patch(self):
        self.assertIs(new_history.source_text,source_text)
        tree=ast.parse(Path(new_history.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Attribute):self.assertEqual(ast.unparse(target.value),'self')
        self.assertEqual(transport.LIMITS['metadata_bytes'],8*2**20)
        self.assertEqual(transport.HOST_SOURCE_ENVELOPE[2],8804235)


class ExactEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.path,self.sha,self.size=transport.HOST_SOURCE_ENVELOPE
        self.ident=dict(realpath=self.path,device=1,inode=2,size=self.size,mtime_ns=3,ctime_ns=4)

    def test_wrong_digest_rejected_before_any_read(self):
        with patch.object(transport,'raw') as reader,patch.object(transport,'identity') as identity:
            with self.assertRaises(ValueError):transport.host_source_bytes(self.path,'0'*64)
        reader.assert_not_called();identity.assert_not_called()

    def test_wrong_size_rejected_before_open(self):
        with patch.object(transport,'raw') as reader,patch.object(transport,'identity',return_value=dict(self.ident,size=self.size+1)):
            with self.assertRaises(ValueError):transport.host_source_bytes(self.path,self.sha)
        reader.assert_not_called()

    def test_exact_envelope_uses_unchanged_spec_cap_and_full_size(self):
        payload=b'x'*self.size
        with patch.object(transport,'raw',return_value=payload) as reader,\
                patch.object(transport,'identity',return_value=self.ident),\
                patch.object(transport.hashlib,'sha256',return_value=SimpleNamespace(hexdigest=lambda:self.sha)):
            self.assertIs(transport.host_source_bytes(self.path,self.sha),payload)
        reader.assert_called_once_with(self.path,64*2**20)

    def test_identity_change_and_truncation_rejected(self):
        for changed,payload in [(dict(self.ident,inode=3),b'x'*self.size),(self.ident,b'x')]:
            with self.subTest(size=len(payload)),patch.object(transport,'raw',return_value=payload),\
                    patch.object(transport,'identity',side_effect=[self.ident,changed]):
                with self.assertRaises(ValueError):transport.host_source_bytes(self.path,self.sha)

    def test_full_hash_still_required(self):
        with patch.object(transport,'raw',return_value=b'x'*self.size),patch.object(transport,'identity',return_value=self.ident):
            with self.assertRaisesRegex(ValueError,'digest'):transport.host_source_bytes(self.path,self.sha)

    def test_other_qwen_names_do_not_get_envelope_exception(self):
        fake=SimpleNamespace(text=Mock(return_value=b'ordinary'))
        with patch.object(source_text,'host_source_bytes',side_effect=AssertionError('unknown envelope')):
            for path in (self.path+'.other',self.path.replace('r10','r11'),'/tmp/'+Path(self.path).name):
                self.assertEqual(source_text.text_bytes(fake,path,'1'*64),b'ordinary')
        self.assertEqual(fake.text.call_count,3)


class CompletePartitionTests(unittest.TestCase):
    def fixture(self):
        ordinary='/mock/ordinary.py';alias='/mock/alias.json';special=transport.HOST_SOURCE_ENVELOPE[0]
        values={ordinary:b'ordinary',alias:b'alias',special:b'exact-envelope'}
        pins={p:hashlib.sha256(b).hexdigest() for p,b in values.items()}
        part=dict(categories=dict(text_sha256=pins,
            python_bytes_sha256={f'/mock/fixture{i}.py':'a'*64 for i in range(3)},
            historical_payload_sha256={'/mock/history.raw':'b'*64},
            query_native_binary_sha256={'/mock/native.so':'c'*64}),
            metadata_aliases={alias:dict(target='/mock/target.json')},source_map_sha256='s',partition_sha256='p')
        calls=[]
        def text(path,sha=None):
            self.assertNotIn(path,('/mock/history.raw','/mock/native.so'))
            calls.append(('text',path,sha));return values[path]
        def alias_sha(path,admitted,callback):
            self.assertIs(callback,text);self.assertEqual(admitted,part['metadata_aliases'])
            calls.append(('alias',path));return pins[path]
        def require(ok,message):
            if not ok:raise ValueError(message)
        def check(value):
            self.assertIs(value,part);self.assertEqual(set(value['categories']),{'text_sha256','python_bytes_sha256','historical_payload_sha256','query_native_binary_sha256'})
            calls.append(('partition',))
        fake=SimpleNamespace(require=require,check_partition=check,text=text,
            aliases=SimpleNamespace(source_sha=alias_sha),metadata=SimpleNamespace(python_source_bytes=lambda p,h:calls.append(('python',p,h))))
        return part,values,calls,fake

    def verify(self,part,values,fake,guard):
        with patch.dict(sys.modules,{'research.minicpm5_layer1_operator_sdk_sources_v1':fake}),\
                patch.object(research,'minicpm5_layer1_operator_sdk_sources_v1',fake,create=True),\
                patch.object(source_text,'host_source_bytes',side_effect=lambda p,h:values[p]):
            return source_text.verify_text_sources(part,guard)

    def test_all_categories_and_alias_and_three_byte_fixtures_retained(self):
        part,values,calls,fake=self.fixture();before=copy.deepcopy(part);guard=Mock()
        result=self.verify(part,values,fake,guard)
        self.assertEqual(part,before);self.assertEqual(result,dict(complete_text_verification=True,
            source_map_sha256='s',partition_sha256='p',historical_payload_bytes_read=0,native_bytes_read=0))
        self.assertEqual(sum(c[0]=='python' for c in calls),3)
        self.assertEqual(sum(c[0]=='alias' for c in calls),1)
        self.assertEqual(sum(c[0]=='text' for c in calls),1)
        self.assertEqual(guard.call_count,8)

    def test_partition_validation_failure_stops_before_any_text(self):
        part,values,calls,fake=self.fixture();fake.check_partition=Mock(side_effect=ValueError('complete partition'))
        with self.assertRaisesRegex(ValueError,'complete partition'):self.verify(part,values,fake,Mock())
        self.assertEqual(calls,[])

    def test_ordinary_hash_mismatch_cannot_be_hidden(self):
        part,values,calls,fake=self.fixture();part['categories']['text_sha256']['/mock/ordinary.py']='0'*64
        with self.assertRaisesRegex(ValueError,'Unchanged exact text source'):self.verify(part,values,fake,Mock())
        self.assertEqual(sum(c[0]=='python' for c in calls),0)

    def test_guard_exception_stops_full_iteration(self):
        part,values,calls,fake=self.fixture();guard=Mock(side_effect=[None,ValueError('guard')])
        with self.assertRaisesRegex(ValueError,'guard'):self.verify(part,values,fake,guard)
        self.assertEqual(calls,[('partition',)])


if __name__=='__main__':
    # Reuse the thirty unchanged accepted geometry controls with only their
    # module-under-test references switched to the source successor.
    from qwen35_writer_geometry_serial_v1 import test_source as inherited
    inherited.s=run;inherited.audit=audit;inherited.History=new_history.History
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]),
                              unittest.defaultTestLoader.loadTestsFromModule(inherited)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
