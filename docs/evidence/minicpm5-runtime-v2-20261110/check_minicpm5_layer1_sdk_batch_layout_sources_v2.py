"""Synthetic partition/context tests; no real payload/native reads or launches."""
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock,patch
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_sdk_batch_layout_sources_v2 as core
import minicpm5_layer1_sdk_batch_layout_runtime_context_v2 as context_module

class Checks(unittest.TestCase):
    def setUp(self):
        self.guard=Mock()
        self.ref=dict(path=str(Q/'minicpm5-layer1-sdk-batch-layout-source-contract-synthetic.json'),sha256='1'*64)
        self.catalog_ref=dict(path='/synthetic/catalog.json',sha256='2'*64)
        self.contract=dict(schema=core.CONTRACT_SCHEMA,r12_spec=core.R12_SPEC,metadata_limits=core.metadata.LIMITS)
        self.categories=dict(text_sha256={'/synthetic/source.py':'3'*64,self.ref['path']:self.ref['sha256'],
            core.R12_SPEC['path']:core.R12_SPEC['sha256'],self.catalog_ref['path']:self.catalog_ref['sha256']},
            historical_payload_sha256={'/unopened/historical%d.%s'%(i,'raw'if i<14 else 'bin'):'4'*64 for i in range(20)},
            python_bytes_sha256={'/synthetic/nonutf8%d.py'%i:'5'*64 for i in range(3)},
            query_native_binary_sha256={'/unopened/lib%d.so.1.2'%i:'6'*64 for i in range(9)})
        files={};mappings={}
        for i,path in enumerate(self.categories['query_native_binary_sha256']):
            row=dict(sha256='6'*64,bytes=435600 if i<8 else 435608,device=1,inode=i+1)
            files[path]=row;mappings[path]={'device':1,'inode':i+1}
        for i in range(32):files['/synthetic/query%d.py'%i]=dict(sha256='7'*64,bytes=12700 if i<31 else 12708,device=1,inode=100+i)
        self.observation=dict(schema='dory_service_query_dependency_observation_v1',complete_observation=True,
            source_and_binary_pins=files,source_and_binary_bytes_hashed=4326816,new_native_file_mappings=mappings)
        self.pins={k:v for values in self.categories.values()for k,v in values.items()}
        self.anchor=dict(schema='nemotron9b_package_runtime_v27',run_id='r12-20261110',
            source_sha256=copy.deepcopy(self.pins),metadata_limits=core.metadata.LIMITS,metadata_aliases={},
            final_predecessor_catalog=self.catalog_ref,scientific={'must':'retain'})
        self.catalog={'synthetic':'catalog'}
        self.partition=dict(schema=core.PARTITION_SCHEMA,source_contract=self.ref,r12_spec=core.R12_SPEC,
            source_map_sha256=core.digest(self.pins),partition_sha256=core.digest(self.categories),
            categories=self.categories,metadata_aliases={},query_observation_ref=core.metadata.query_sources.OBSERVATION,
            query_observation=self.observation)
        self.identities={path:dict(realpath=path,device=row['device'],inode=row['inode'],size=row['bytes'],
            mtime_ns=1,ctime_ns=2)for path,row in files.items()if path in mappings}
        self.owner={'pid':os.getpid(),'start_ticks':'123','argv':['synthetic']}
        self.lock={'device':1,'inode':22};self.attempt='a'*32
        self.report=dict(complete=True,files=9,bytes=3920408,native_libraries_loaded_here=False,
            model_or_tensor_bytes_read=0,full_runtime_environment_verified=False)
        self.receipt=dict(schema=core.AUTH_SCHEMA,source_contract=self.ref,
            source_map_sha256=self.partition['source_map_sha256'],partition_sha256=self.partition['partition_sha256'],
            query_observation=self.partition['query_observation_ref'],native_sha256=self.categories['query_native_binary_sha256'],
            native_identities=self.identities,native_authentication=self.report,outer_owner=self.owner,
            lock_identity=self.lock,attempt_id=self.attempt,historical_payload_bytes_read=0,
            native_libraries_loaded_here=False,full_model_qualified=False,new_speedup_claim=False)

    def read(self,ref):
        for expected,value in ((self.ref,self.contract),(core.R12_SPEC,self.anchor),(self.catalog_ref,self.catalog),
            (core.metadata.query_sources.OBSERVATION,self.observation)):
            if ref==expected:return copy.deepcopy(value)
        raise AssertionError('Unexpected metadata read')

    def classify(self):
        with patch.object(core.metadata,'classify_sources',return_value=copy.deepcopy(self.categories)):
            return core.classify(self.pins,self.ref,self.read)

    def test_four_categories_lossless_no_mutation(self):
        before=copy.deepcopy((self.pins,self.anchor));result=self.classify()
        self.assertEqual(core.check_partition(result),self.pins)
        self.assertEqual(before,(self.pins,self.anchor))
        self.assertEqual([len(result['categories'][k])for k in core.CATEGORIES],[4,20,3,9])

    def test_full43912_map_preserved_without_projection(self):
        for i in range(43912-len(self.pins)):self.pins['/synthetic/additional%d.py'%i]='a'*64
        self.categories['text_sha256'].update({k:v for k,v in self.pins.items()if k not in {p for cat in core.CATEGORIES[1:]for p in self.categories[cat]}})
        with patch.object(core.metadata,'classify_sources',return_value=self.categories)as classify:
            result=core.classify(self.pins,self.ref,self.read)
            self.assertEqual(len(core.check_partition(result)),43912)
            view=classify.call_args.args[0]
            self.assertEqual(view['source_sha256'],self.pins)
            self.assertEqual(view['scientific'],self.anchor['scientific'])

    def test_changed_or_missing_inherited_pin_rejected_before_classifier(self):
        for mutation in ('changed','missing'):
            pins=copy.deepcopy(self.pins)
            if mutation=='changed':pins['/synthetic/source.py']='0'*64
            else:pins.pop('/synthetic/source.py')
            with patch.object(core.metadata,'classify_sources')as classifier:
                with self.assertRaises(RuntimeError):core.classify(pins,self.ref,self.read)
                classifier.assert_not_called()

    def test_loss_overlap_unknown_native_and_contract_rejected(self):
        for mutation in ('loss','overlap','native','contract'):
            with self.subTest(mutation=mutation):
                self.setUp()
                if mutation=='loss':self.categories['text_sha256'].pop('/synthetic/source.py')
                if mutation=='overlap':self.categories['text_sha256'].update(self.categories['historical_payload_sha256'])
                if mutation=='native':
                    self.pins['/unopened/new.so.9']='0'*64;self.categories['text_sha256']['/unopened/new.so.9']='0'*64
                if mutation=='contract':self.contract['metadata_limits']={'ordinary_file_bytes':999}
                with self.assertRaises(RuntimeError):self.classify()

    def test_source_only_auth_does_not_call_native_or_read_payload(self):
        def text(path,sha):return ('bytes:'+path).encode()
        for path in self.categories['text_sha256']:self.categories['text_sha256'][path]=hashlib.sha256(text(path,None)).hexdigest()
        self.partition['partition_sha256']=core.digest(self.categories)
        pins={k:v for category in self.categories.values()for k,v in category.items()}
        self.partition['source_map_sha256']=core.digest(pins)
        with patch.object(core,'text',side_effect=text)as opened,\
            patch.object(core.metadata,'python_source_bytes')as python,\
            patch.object(core.metadata.query_sources,'authenticate_native_sources',side_effect=AssertionError('native read'))as native:
            result=core.verify_text_sources(self.partition,self.guard)
            self.assertEqual(opened.call_count,4);self.assertEqual(python.call_count,3)
            native.assert_not_called();self.assertEqual(result['historical_payload_bytes_read'],0)

    def test_original_exact_alias_transport_is_used(self):
        name,row=next(iter(core.aliases.ALLOWED.items()))
        self.categories['text_sha256']={name:row['sha256']}
        self.partition['metadata_aliases']={name:{'synthetic':'captured'}}
        self.partition['source_map_sha256']=core.digest({k:v for cat in self.categories.values()for k,v in cat.items()})
        self.partition['partition_sha256']=core.digest(self.categories)
        with patch.object(core.aliases,'validate_spec'),patch.object(core.aliases,'source_sha',return_value=row['sha256'])as alias,\
            patch.object(core.metadata,'python_source_bytes'):
            core.verify_text_sources(self.partition,self.guard)
            alias.assert_called_once_with(name,self.partition['metadata_aliases'],core.text)

    def test_native_hash_only_explicit_outer_after_lock_and_once(self):
        events=[]
        def verify(pins,observation,guard):
            guard();events.append('hash_exact9');return self.report
        with patch.object(core,'native_identities',return_value=self.identities),\
            patch.object(core.p,'lock_proof',side_effect=lambda *a:events.append('lock')),\
            patch.object(core.metadata.query_sources,'authenticate_native_sources',side_effect=verify)as native:
            receipt=core.authenticate_native_sources(self.partition,self.owner,self.lock,self.attempt,self.guard)
            self.assertEqual(receipt,self.receipt);self.assertEqual(native.call_count,1)
            self.assertLess(events.index('lock'),events.index('hash_exact9'))
            self.assertEqual(events[-1],'lock')

    def test_nonouter_and_broken_lock_reject_before_native(self):
        with patch.object(core.metadata.query_sources,'authenticate_native_sources')as native:
            with self.assertRaises(RuntimeError):
                core.authenticate_native_sources(self.partition,dict(self.owner,pid=os.getpid()+1),self.lock,self.attempt,self.guard)
            native.assert_not_called()
        with patch.object(core.p,'lock_proof',side_effect=RuntimeError('lost lock')),\
            patch.object(core.metadata.query_sources,'authenticate_native_sources')as native:
            with self.assertRaisesRegex(RuntimeError,'lost lock'):
                core.authenticate_native_sources(self.partition,self.owner,self.lock,self.attempt,self.guard)
            native.assert_not_called()

    def test_native_identity_change_during_hash_rejects(self):
        with patch.object(core,'native_identities',side_effect=[self.identities,{}]),patch.object(core.p,'lock_proof'),\
            patch.object(core.metadata.query_sources,'authenticate_native_sources',return_value=self.report):
            with self.assertRaises(RuntimeError):
                core.authenticate_native_sources(self.partition,self.owner,self.lock,self.attempt,self.guard)

    def test_receipt_pure_validator_and_stat_only_checks(self):
        with patch.object(core.metadata.query_sources,'authenticate_native_sources',side_effect=AssertionError('no native read')), \
            patch.object(core,'identity',side_effect=lambda path:self.identities[path])as stat:
            proof=core.validate_native_receipt(self.receipt,self.partition,self.owner,self.lock,self.attempt)
            self.assertEqual(proof['native_bytes_read_here'],0);stat.assert_not_called()
            core.check_native_identities(self.receipt,self.partition,self.guard);self.assertEqual(stat.call_count,9)

    def test_receipt_binding_hash_identity_and_flags_tamper_rejected(self):
        for key,value in [('source_map_sha256','0'*64),('partition_sha256','0'*64),('outer_owner',{}),
            ('lock_identity',{}),('attempt_id','b'*32),('native_authentication',{}),('native_identities',{}),
            ('full_model_qualified',True),('native_sha256',{})]:
            with self.subTest(key=key):
                bad=copy.deepcopy(self.receipt);bad[key]=value
                with self.assertRaises(RuntimeError):core.validate_native_receipt(bad,self.partition,self.owner,self.lock,self.attempt)

    def test_ordinary_metadata_uses_unchanged_v27_transport(self):
        with patch.object(core.metadata,'text',return_value=b'old')as old,patch.object(core.metadata,'stable_bytes')as read:
            self.assertEqual(core.text('/untrusted/arbitrary.json','a'*64),b'old')
            old.assert_called_once();read.assert_not_called()
        self.assertEqual(core.metadata.LIMITS,{'ordinary_file_bytes':8388608,'named_json_envelope_bytes':33554432})

    def test_named_new_envelope_requires_exact_schema_and_digest(self):
        path=Q/'minicpm5-layer1-sdk-batch-layout-synthetic-spec.json'
        value=dict(schema=core.ENVELOPES['spec'],source_sha256={},source_contract=self.ref,
            latest_completion={'status':'pending'},output_root='/synthetic',plan=self.ref)
        raw=json.dumps(value).encode();sha=hashlib.sha256(raw).hexdigest()
        with patch.object(core.metadata,'stable_bytes',return_value=raw)as read:
            self.assertEqual(core.text(path,sha),raw)
            read.assert_called_with(path,33554432)
            with self.assertRaises(RuntimeError):core.text(path,None)
            with self.assertRaises(RuntimeError):core.text(path,'0'*64)
        value['schema']='untrusted';bad=json.dumps(value).encode()
        with patch.object(core.metadata,'stable_bytes',return_value=bad):
            with self.assertRaises(RuntimeError):core.text(path,hashlib.sha256(bad).hexdigest())

    def test_same_basename_elsewhere_not_new_envelope(self):
        self.assertIsNone(core.envelope_kind('/untrusted/minicpm5-layer1-sdk-batch-layout-synthetic-spec.json'))
        self.assertIsNone(core.envelope_kind(Q/'minicpm5-layer1-sdk-batch-layout-synthetic-payload.json'))

    def context_fixture(self):
        root='/synthetic/output';aref=dict(path=root+'/admission.json',sha256='a'*64)
        nref=dict(path=root+'/source-authentication.json',sha256='b'*64)
        planref=dict(path='/synthetic/plan.json',sha256='c'*64)
        permit=dict(runtime_admission=aref,runtime_plan=planref,outer_owner=self.owner,lock_identity=self.lock,
            attempt_id=self.attempt,source_authentication=nref)
        plan=dict(source_sha256=self.pins,source_contract=self.ref,output_root=root,latest_completion={'complete':'synthetic'},inputs={})
        a=dict(schema='minicpm5_layer1_sdk_batch_layout_cpu_admission_v2',source_contract=self.ref,plan=planref,
            outer_owner=self.owner,lock_identity=self.lock,attempt_id=self.attempt,latest_completion=plan['latest_completion'],
            source_authentication=nref,runtime_admitted=True,predecessors_checked=True)
        identities={r['path']:dict(realpath=r['path'],device=1,inode=i+200,size=100,mtime_ns=1,ctime_ns=1)for i,r in enumerate((aref,nref))}
        def load(ref):
            if ref==aref:return copy.deepcopy(a)
            if ref==nref:return copy.deepcopy(self.receipt)
            raise AssertionError('Unexpected context metadata read')
        context=object.__new__(context_module.RuntimeContext)
        context.plan=plan;context.permit=permit;context._sources_checked=False;context._verified={}
        return context,a,identities,load

    def test_context_uses_outer_native_receipt_without_native_byte_reads(self):
        context,a,identities,load=self.context_fixture()
        with patch.object(context_module.OriginalContext,'guard',return_value=context.permit),\
            patch.object(core,'classify',return_value=self.partition),patch.object(core,'verify_text_sources'),\
            patch.object(core,'load',side_effect=load),patch.object(context_module,'identity',side_effect=lambda p:identities[p]),\
            patch.object(core,'native_identities',return_value=self.identities),\
            patch.object(core.metadata.query_sources,'authenticate_native_sources',side_effect=AssertionError('worker native read'))as native:
            context.verify_sources();self.assertTrue(context._sources_checked);context.guard('cpu')
            native.assert_not_called()
            identities[a['source_authentication']['path']]=dict(identities[a['source_authentication']['path']],inode=999)
            with self.assertRaises(RuntimeError):context.guard('cpu')

    def test_context_missing_mismatched_receipt_or_live_identity_rejects(self):
        for mutation in ('missing','binding','native'):
            with self.subTest(mutation=mutation):
                context,a,identities,load=self.context_fixture()
                if mutation=='binding':a['outer_owner']={}
                def reader(ref):
                    if mutation=='missing'and ref['path'].endswith('source-authentication.json'):raise FileNotFoundError(ref['path'])
                    return load(ref)
                with patch.object(context_module.OriginalContext,'guard',return_value=context.permit),\
                    patch.object(core,'classify',return_value=self.partition),patch.object(core,'verify_text_sources'),\
                    patch.object(core,'load',side_effect=reader),patch.object(context_module,'identity',side_effect=lambda p:identities[p]),\
                    patch.object(core,'native_identities',return_value={}if mutation=='native'else self.identities):
                    with self.assertRaises((RuntimeError,FileNotFoundError)):context.verify_sources()
                    self.assertFalse(context._sources_checked)

    def test_context_large_json_uses_named_transport_before_any_payload_reader(self):
        context,a,identities,load=self.context_fixture();context._sources_checked=True
        path=str(Q/'minicpm5-layer1-sdk-batch-layout-synthetic-plan.json')
        ident=dict(realpath=path,device=1,inode=99,size=14*2**20,mtime_ns=1,ctime_ns=2)
        context.plan['inputs'][path]={'sha256':'d'*64,'identity':ident}
        with patch.object(context_module.OriginalContext,'guard',return_value=context.permit),\
            patch.object(context_module,'identity',return_value=ident),patch.object(core,'load',return_value={'metadata':'large'})as reader,\
            patch.object(context_module.OriginalContext,'verify_input',side_effect=AssertionError('generic payload read')):
            self.assertEqual(context.read_json(path,'d'*64),{'metadata':'large'})
            reader.assert_called_once_with(dict(path=path,sha256='d'*64,bytes=14*2**20))
            with self.assertRaises(RuntimeError):context.read_json('/synthetic/payload.bin','d'*64)

    def test_original_model_input_environment_and_activation_methods_inherited(self):
        for name in ('__init__','verify_input','record_cpu_import','activate'):
            self.assertIs(getattr(context_module.RuntimeContext,name),getattr(context_module.OriginalContext,name))
        self.assertFalse(any(n.split('.')[0]in {'torch','numpy','torch_neuronx','neuronxcc','vllm'}for n in sys.modules))

    def test_bad_partition_digest_rejects_before_text_read(self):
        self.partition['partition_sha256']='0'*64
        with patch.object(core,'text')as text,patch.object(core.metadata,'python_source_bytes')as python:
            with self.assertRaises(RuntimeError):core.verify_text_sources(self.partition,self.guard)
            text.assert_not_called();python.assert_not_called()

    def test_native_stat_wrong_inode_or_size_rejects_before_authentication(self):
        for key in ('inode','size'):
            identities=copy.deepcopy(self.identities);path=next(iter(identities));identities[path][key]+=1
            with patch.object(core,'identity',side_effect=lambda p:identities[p]),patch.object(core.p,'lock_proof'),\
                patch.object(core.metadata.query_sources,'authenticate_native_sources')as native:
                with self.assertRaises(RuntimeError):
                    core.authenticate_native_sources(self.partition,self.owner,self.lock,self.attempt,self.guard)
                native.assert_not_called()

    def test_incomplete_native_report_cannot_create_success_receipt(self):
        with patch.object(core,'native_identities',return_value=self.identities),patch.object(core.p,'lock_proof'),\
            patch.object(core.metadata.query_sources,'authenticate_native_sources',return_value=dict(self.report,files=8)):
            with self.assertRaises(RuntimeError):
                core.authenticate_native_sources(self.partition,self.owner,self.lock,self.attempt,self.guard)

    def test_json_ref_types_length_and_tensor_suffix_reject(self):
        for ref in ({'path':'relative.json','sha256':'a'*64},
            {'path':'/synthetic/a.json','sha256':'a'*64,'bytes':True},
            {'path':'/synthetic/a.bin','sha256':'a'*64}):
            with self.subTest(ref=ref),patch.object(core,'text')as read:
                with self.assertRaises(RuntimeError):core.load(ref)
                read.assert_not_called()
        with patch.object(core,'text',return_value=b'{}'):
            with self.assertRaises(RuntimeError):core.load({'path':'/synthetic/a.json','sha256':'a'*64,'bytes':3})

    def test_context_wrong_metadata_hash_or_identity_rejects_before_read(self):
        context,a,identities,load=self.context_fixture();context._sources_checked=True
        path='/synthetic/input.json';ident=dict(realpath=path,device=1,inode=9,size=100,mtime_ns=1,ctime_ns=1)
        context.plan['inputs'][path]={'sha256':'d'*64,'identity':ident}
        with patch.object(context_module.OriginalContext,'guard',return_value=context.permit),\
            patch.object(context_module,'identity',return_value=ident)as identity,patch.object(core,'load')as read:
            with self.assertRaises(RuntimeError):context.read_json(path,'e'*64)
            read.assert_not_called()
            identity.return_value=dict(ident,inode=999)
            with self.assertRaises(RuntimeError):context.read_json(path,'d'*64)
            read.assert_not_called()

    def test_ast_compile_and_no_bytecode_native_imports(self):
        self.assertTrue(sys.dont_write_bytecode)
        for module in (core,context_module):
            compile(Path(module.__file__).read_text(),module.__file__,'exec',dont_inherit=True)
        self.assertFalse(any(n.split('.')[0]in {'torch','numpy','torch_neuronx','neuronxcc','vllm'}for n in sys.modules))

if __name__=='__main__':unittest.main(verbosity=2)
