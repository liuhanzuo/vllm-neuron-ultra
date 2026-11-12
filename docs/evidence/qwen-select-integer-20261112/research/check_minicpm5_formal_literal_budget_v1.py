"""Stdlib/source mocks for one-call immutable budget reuse; no runtime replay."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
BLOCKED = {'ctypes','_ctypes','torch','torch_xla','torch_neuronx','numpy','neuronxcc',
           'neuronx_distributed','neuronx_distributed_inference','ml_dtypes','libneuronxla'}


class NoNative:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise AssertionError('Source/mock only: '+name)


assert not BLOCKED.intersection(sys.modules)
sys.meta_path.insert(0, NoNative())
from research import minicpm5_formal_literal_codec_v1 as codec
from research import minicpm5_formal_literal_result_v1 as result
from research import minicpm5_formal_literal_budget_v1 as prepared
from research import minicpm5_sdk_mapping_observer_backend_v4 as old_backend
from research import minicpm5_sdk_mapping_observer_backend_v5 as new_backend
from research import minicpm5_sdk_mapping_outer_source_entry_v10 as new_outer

EVIDENCE = Q/'minicpm5-fixed-input-pv-sdk-import-formal-r5-20261112'
FROZEN = {
    'minicpm5_sdk_mapping_outer_source_entry_v9.py':'c498fadfb42957b9e8ae66f461be4137c5546b27046de7b1ab7f4c9843624c57',
    'minicpm5_sdk_mapping_nocache_entry_v6.py':'bbf5affece28e3d6356ce27d48ccdb7f4d707fbf6717808a69ca1f61222763d0',
    'minicpm5_sdk_mapping_source_adapter_v3.py':'245f15f52cbfbd36a345d0ac16ed6000118026d055eff5fb9dc17f7ff792506f',
    'minicpm5_sdk_mapping_observed_adapter_v5.py':'561688b8012a97867f990ebd8f43ae29551fe0d37450bcbf75e7fdddb5ead378',
    'minicpm5_sdk_mapping_observer_backend_v4.py':'e944fe7cb35194fe052934d337c5d5d48e94aa40820742af68939010b7db26ef',
    'minicpm5_sdk_mapping_observer_logic_v2.py':'c7672fa51553edd3f46cc90049ee2c0f2c46c76e816fd9869a97d596f32fcbf8',
    'minicpm5_formal_literal_codec_v1.py':'b2909dec9b14aa55b0d543b597b04423fbb0d5bbc96c9ed3354d51cf5ea2eb20',
    'minicpm5_formal_literal_result_v1.py':'d96c90a5f1b36441574634fb5e6917c5bf9177753cbfe682d87bc3cbcf0027ea'}


class CountingDigest:
    """Exact SHA oracle with the original pure implementation's check schedule."""
    def __init__(self): self.sizes=[]
    def __call__(self, raw, check=lambda:None):
        self.sizes.append(len(raw))
        padded=((len(raw)+9+63)//64)*64
        for _ in range(0,padded,4096): check()
        check()
        return hashlib.sha256(raw).hexdigest()


def original_budget(wire, argv, literal, root):
    value=codec.future_runtime_budget(wire,argv,literal)
    value.update(result.future_stdout_budget(wire,argv,literal,root))
    return value


class MemoryStore:
    """Explicitly fake disk boundary; inherited writer/readback calls stay real."""
    def __init__(self,root,guard):
        self.root=Path(root);self.guard=guard;self.rows={};self.saved={};self.identities={}
        self.allocated=0;self.logical_metadata=0;self.events=[];self.before_read=None
    def inventory(self):
        self.guard();self.events.append(('inventory',tuple(sorted(self.saved))))
        return dict(names=sorted(self.saved))
    def write(self,name,raw):
        self.guard();assert name not in self.saved
        self.saved[name]=raw;self.identities[name]=1;self.rows[name]=(0,1,len(raw))
        self.logical_metadata+=len(raw);self.allocated+=len(raw)
        self.events.append(('write',name,len(raw)))
        return self.reference(name)
    def json(self,name,value):return self.write(name,codec.encoded(value))
    def reference(self,name):
        self.guard();self.events.append(('reference',name))
        if self.identities[name]!=1:raise ValueError('Mock fresh identity changed')
        raw=self.saved[name]
        return dict(path=str(self.root/name),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    def read(self,reference):
        self.guard();name=Path(reference['path']).name;self.events.append(('read',name))
        if self.before_read:self.before_read(self)
        raw=self.saved[name]
        if self.identities[name]!=1 or hashlib.sha256(raw).hexdigest()!=reference['sha256']:
            raise ValueError('Mock actual full read/identity mismatch')
        return raw


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.literal=(Q/codec.LITERAL['path'].split('/')[-1]).read_text()
        cls.wire=(EVIDENCE/'admission.json').read_bytes()
        owner=(EVIDENCE/'phase-1-owner.json').read_bytes()
        assert len(cls.wire)==44438 and hashlib.sha256(cls.wire).hexdigest()=='8e5b1b0959f3270e831963637c3bfd0e2e794593cdc1a3a53ef2dac0839677f0'
        assert len(owner)==990 and hashlib.sha256(owner).hexdigest()=='95ae403eaecef64ab7eb47c86c241805f99ae7db053ea9bb4bc37a246339d1cb'
        with patch.object(codec,'digest',CountingDigest()):
            cls.admission=codec.decode('admission',cls.wire,cls.literal,wire_sha256=hashlib.sha256(cls.wire).hexdigest())
            cls.owner=codec.decode('owner',owner,cls.literal,wire_sha256=hashlib.sha256(owner).hexdigest())
        cls.argv=cls.owner['argv'];cls.root=str(EVIDENCE)

    def test_counting_oracle_preserves_real_digest_values_and_every_guard_checkpoint(self):
        for raw in (b'',b'x',b'x'*55,b'x'*56,b'x'*4095,b'x'*4096,self.literal.encode()):
            real=[];mock=[];oracle=CountingDigest()
            expected=codec.digest(raw,lambda:real.append('guard'))
            self.assertEqual(oracle(raw,lambda:mock.append('guard')),expected)
            self.assertEqual(mock,real)

    def test_exact_saved_inputs_identical_budgets_six_fewer_hashes(self):
        outputs=[];counts=[]
        for function in (original_budget,prepared.combined_budget):
            counter=CountingDigest()
            with patch.object(codec,'digest',counter):
                before=copy.deepcopy((self.wire,self.argv,self.literal,self.root))
                outputs.append(function(*before))
                self.assertEqual(before,(self.wire,self.argv,self.literal,self.root))
            counts.append((len(counter.sizes),sum(counter.sizes)))
        self.assertEqual(outputs[0],outputs[1])
        self.assertEqual(codec.encoded(outputs[0]),codec.encoded(outputs[1]))
        self.assertEqual(counts,[(18,911953),(12,552023)])
        self.assertEqual(outputs[1]['runtime_cap'],79008)
        self.assertFalse(outputs[1]['actual_sdk_owner_observed'])

    def test_changed_admission_wire_literal_argv_and_caps_rejected_by_both(self):
        cases=[]
        wire=bytearray(self.wire);wire[100]^=1
        cases.append((bytes(wire),self.argv,self.literal,self.root))
        value=json.loads(self.wire);value['witness']['raw_capture_bytes_read']=1
        cases.append((codec.encoded(value),self.argv,self.literal,self.root))
        cases.append((self.wire,self.argv,self.literal[:-1]+'x',self.root))
        for index,replacement in ((7,self.literal[:-1]+'x'),(2,'-s'),(9,'outer312'),(13,'f'*64)):
            argv=self.argv[:];argv[index]=replacement
            cases.append((self.wire,argv,self.literal,self.root))
        with patch.object(codec,'digest',CountingDigest()):
            large=copy.deepcopy(self.admission)
            large['source_entry_binding']['source_mock_budget_padding']='x'*40000
            cases.append((codec.encode('admission',large,self.literal),self.argv,self.literal,self.root))
            for args in cases:
                for function in (original_budget,prepared.combined_budget):
                    with self.subTest(function=function.__name__),self.assertRaises((ValueError,UnicodeDecodeError)):
                        function(*args)
            for key,value in (('RUNTIME_CAP',1),('SUMMARY_CAP',1)):
                with patch.object(codec,key,value):
                    for function in (original_budget,prepared.combined_budget):
                        with self.assertRaises(ValueError):function(self.wire,self.argv,self.literal,self.root)

    def test_no_cross_call_cache_and_changed_valid_inputs_get_new_exact_result(self):
        with patch.object(codec,'digest',CountingDigest()) as digest:
            first=prepared.combined_budget(self.wire,self.argv,self.literal,self.root)
            calls=len(digest.sizes)
            second=prepared.combined_budget(self.wire,self.argv,self.literal,self.root)
            self.assertEqual(len(digest.sizes),calls*2)
            self.assertEqual(first,second);first['total_runtime_bound']=0
            self.assertNotEqual(first,second)
            admission=copy.deepcopy(self.admission)
            admission['source_entry_binding']['source_mock_budget_padding']='x'*37
            wire=codec.encode('admission',admission,self.literal)
            for root,argv in ((self.root,self.argv),(self.root+'-different',self.argv)):
                actual=prepared.combined_budget(wire,argv,self.literal,root)
                self.assertEqual(actual,original_budget(wire,argv,self.literal,root))
                self.assertNotEqual(actual,second)

    def store(self,backend):
        logic=old_backend.literal_backend.logic
        original=NS(Store=MemoryStore,RUNTIME_METADATA_CAP=logic.RUNTIME_METADATA_CAP,
            RUNTIME={'admission.json','phase-1-owner.json','phase-1-exit.json'},serial=NS(CAPS=logic.CAPS))
        events=[];cls=backend.make_store_class(original)
        store=cls(self.root,lambda:events.append('guard'),literal=self.literal)
        return store,events

    def reserve(self,store):
        _,reference=store.prospective_admission(self.admission)
        budget=store.reserve_runtime(self.admission,reference,self.argv)
        return reference,budget

    def test_actual_store_override_keeps_all_guard_and_fresh_readback_calls(self):
        values=[]
        for backend in (old_backend,new_backend):
            counter=CountingDigest()
            with patch.object(codec,'digest',counter):
                store,guards=self.store(backend)
                reference,budget=self.reserve(store)
                store.json('admission.json',self.admission)
                store.before_spawn(self.admission,reference,self.argv)
                # A second unchanged pre-spawn check must still freshly read.
                store.before_spawn(self.admission,reference,self.argv)
                self.assertEqual(store.saved['admission.json'],self.wire)
                self.assertEqual(store.events.count(('read','admission.json')),2)
                values.append((reference,budget,store.events,guards,len(counter.sizes),sum(counter.sizes)))
        self.assertEqual(values[0][:4],values[1][:4])
        self.assertEqual(values[0][4]-values[1][4],6)
        self.assertEqual(values[0][5]-values[1][5],359930)

    def test_store_rejects_changed_prepared_inputs_new_bytes_and_identity_at_spawn(self):
        with patch.object(codec,'digest',CountingDigest()):
            def ready():
                store,_=self.store(new_backend);reference,_=self.reserve(store)
                store.json('admission.json',self.admission)
                return store,reference
            store,reference=ready()
            with self.assertRaises(ValueError):store.reserve_runtime(self.admission,reference,self.argv)
            value=copy.deepcopy(self.admission);value['witness']['raw_capture_bytes_read']=1
            with self.assertRaises(ValueError):store.before_spawn(value,reference,self.argv)
            argv=self.argv[:];argv[10]+='-changed'
            with self.assertRaises(ValueError):store.before_spawn(self.admission,reference,argv)
            for mutate in (lambda s:s.saved.update({'admission.json':s.saved['admission.json']+b' '}),
                           lambda s:s.identities.update({'admission.json':2})):
                store,reference=ready();mutate(store)
                with self.assertRaises(ValueError):store.before_spawn(self.admission,reference,self.argv)
                store,reference=ready();store.before_read=mutate
                with self.assertRaises(ValueError):store.before_spawn(self.admission,reference,self.argv)
            store,_=self.store(new_backend)
            wrong=dict(path=str(store.root/'admission.json'),sha256='f'*64)
            with self.assertRaises(ValueError):store.reserve_runtime(self.admission,wrong,self.argv)

    def test_only_one_budget_expression_and_three_import_edges_change(self):
        pairs=(('minicpm5_sdk_mapping_outer_source_entry_v9.py','minicpm5_sdk_mapping_outer_source_entry_v10.py',
                'minicpm5_sdk_mapping_source_adapter_v3','minicpm5_sdk_mapping_source_adapter_v4'),
               ('minicpm5_sdk_mapping_source_adapter_v3.py','minicpm5_sdk_mapping_source_adapter_v4.py',
                'minicpm5_sdk_mapping_observed_adapter_v5','minicpm5_sdk_mapping_observed_adapter_v6'),
               ('minicpm5_sdk_mapping_observed_adapter_v5.py','minicpm5_sdk_mapping_observed_adapter_v6.py',
                'minicpm5_sdk_mapping_observer_backend_v4','minicpm5_sdk_mapping_observer_backend_v5'))
        for old,new,a,b in pairs:
            self.assertEqual((Q/old).read_text().replace(a,b),(Q/new).read_text())
        def method(name):
            tree=ast.parse((Q/name).read_text())
            return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='reserve_runtime')
        before=method('minicpm5_sdk_mapping_observer_backend_v3.py')
        after=method('minicpm5_sdk_mapping_observer_backend_v5.py')
        old_assign=next(i for i,n in enumerate(before.body) if isinstance(n,ast.Assign)
                        and any(isinstance(x,ast.Name) and x.id=='budget' for x in n.targets))
        new_assign=next(n for n in after.body if isinstance(n,ast.Assign)
                        and any(isinstance(x,ast.Name) and x.id=='budget' for x in n.targets))
        before.body[old_assign:old_assign+2]=[copy.deepcopy(new_assign)]
        self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))
        self.assertIs(new_backend.make_ports,old_backend.make_ports)
        self.assertIs(new_backend.NativePorts,old_backend.NativePorts)
        self.assertEqual(new_backend.ORIGINAL_NAMES,old_backend.ORIGINAL_NAMES)
        self.assertEqual(new_backend.OBSERVER_NAMES,old_backend.OBSERVER_NAMES)
        with patch.object(codec,'digest',CountingDigest()):
            old,_=self.store(old_backend);new,_=self.store(new_backend)
            for name in ('before_spawn','prospective_admission','write','load','inventory','validate_observer_budget'):
                self.assertIs(getattr(old,name).__func__.__code__,getattr(new,name).__func__.__code__)
        self.assertIs(new_outer.adapter.observed.backend,new_backend)

    def test_frozen_sources_native_science_limits_and_no_import_runtime(self):
        for name,sha in FROZEN.items():
            self.assertEqual(hashlib.sha256((Q/name).read_bytes()).hexdigest(),sha,name)
        limits=old_backend.literal_backend.logic.LIMITS
        self.assertEqual((limits['cpu_seconds'],limits['hard_cpu_seconds'],limits['wall_seconds']),(600,605,900))
        self.assertNotIn('research.minicpm5_pv_sdk_import_diagnostic_v1',sys.modules)
        self.assertNotIn('research.audit_minicpm5_layer1_saved_operators_v1',sys.modules)
        self.assertFalse(BLOCKED.intersection(sys.modules))


if __name__=='__main__':
    test=unittest.main(exit=False,verbosity=2)
    assert not BLOCKED.intersection(sys.modules)
    raise SystemExit(not test.result.wasSuccessful())
