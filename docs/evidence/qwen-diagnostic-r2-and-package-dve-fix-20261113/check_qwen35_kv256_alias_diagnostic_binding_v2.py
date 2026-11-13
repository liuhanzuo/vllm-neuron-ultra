"""Source/synthetic tests only; real plans/current requests are never opened."""
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import patch

Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
import check_qwen35_kv256_alias_diagnostic_binding_v1 as old
import prepare_qwen35_kv256_alias_diagnostic_binding_v2 as v2
import derive_qwen35_kv256_alias_diagnostic_binding_v2 as derivation

READER_PINS={
    str(Q/'qwen35_package_preflight_history_v3.py'):'421b0488dae28a24b9b23eea0d437dfa0035c7ec92f8eb6a52b9e837c152433c',
    str(Q/'qwen35_sampler_runtime_io_v12.py'):'250e176e158c87728703593b6fb1f03d0da5e1e01b856277a7cdedc692bd533f',
    str(Q/'minicpm5_layer1_operator_sdk_sources_v1.py'):'cdc679752fdad493fd00e7f7339e700b91a65604760fcdbbc8f70f49d1f93c30',
    str(Q/'minicpm5_layer1_sdk_batch_layout_sources_v2.py'):'d1bd188a44f3d30cf52ccd171b1eed05e6fc7246d96d68dff39591449b6ced6b',
    str(Q.parent/'nemotron9b_package_metadata_v27.py'):'cc0cfb2beddf985673a8c286446ad7de460fd2823b79b5044b892b9c793ed4f8',
    str(Q.parent/'historical_b3_payload_partition_v2.py'):'3044095f99585575d088db631071d9c2712e94f18277270ad717abfa198932c0',
    str(Q.parent/'historical_b3_payload_partition_v1.py'):'8a3dab13e910f7657cda07e0ff9b1fe3c99d535c2a6a084c4f3e009857c5ab5c',
}


def v2_fixture(with_plan=False):
    with patch.object(old,'s',v2):future,values,identities=old.fixture()
    helper=values[future['preparer_delivery']['path']]
    helper.update(schema='qwen35_kv256_alias_diagnostic_binding_source_delivery_v2',
        predecessor_preparer=copy.deepcopy(v2.PREDECESSOR_PREPARER),retained_input_exception=copy.deepcopy(v2.INHERITED_PLAN))
    helper['source_sha256'][v2.PREDECESSOR_PREPARER['path']]=v2.PREDECESSOR_PREPARER['sha256']
    if with_plan:
        seed=values[future['seed_spec']['path']];overlay=values[seed['history_overlay']['path']]
        plan_ref={k:v2.INHERITED_PLAN[k]for k in('path','sha256')};path=plan_ref['path']
        plan=dict(schema='minicpm5_layer0_runtime_inputs_v1',source_sha256={'/synthetic/source.py':'1'*64},
            source_contract=dict(path='/synthetic/source-contract.json',sha256='2'*64),
            latest_completion={},output_root='/synthetic/historical-output',inputs={},synthetic=True)
        ident=dict(realpath=path,device=1,inode=98765,size=12803326,mtime_ns=1234,ctime_ns=5678)
        overlay['base_plan']=plan_ref;values[path]=plan;identities[path]=ident
        seed['input_extension'][path]=dict(sha256=plan_ref['sha256'],identity=copy.deepcopy(ident))
        seed['source_sha256'][path]=plan_ref['sha256']
        values[future['seed_review']['path']]['source_sha256']=copy.deepcopy(seed['source_sha256'])
    return future,values,identities


def build(future,values,identities):
    def read(ref):return dict(value=copy.deepcopy(values[ref['path']]),identity=copy.deepcopy(identities[ref['path']]))
    return v2.prepare(future,read,lambda _:True,lambda:None)


class Inherited27Tests(old.BindingTests):
    @classmethod
    def setUpClass(cls):cls.binding_patch=patch.object(old,'s',v2);cls.binding_patch.start()
    @classmethod
    def tearDownClass(cls):cls.binding_patch.stop()
    def setUp(self):self.f,self.v,self.i=v2_fixture()


def frozen_source(path):
    data=Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest()==READER_PINS[str(path)],path
    return data.decode()


def extracted_functions(source,names,namespace):
    # No module body/import is executed: only named pure reader functions with
    # synthetic callback globals. Their complete original text is authenticated.
    tree=ast.parse(source);selected=[node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name in names]
    assert {n.name for n in selected}==set(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<authenticated-source-synthetic-test>','exec'),namespace)
    return namespace


def reader_namespace(raw):
    require=v2.require;common=dict(Path=Path,Q=Q,re=re,require=require)
    inherited=extracted_functions(frozen_source(Q/'minicpm5_layer1_sdk_batch_layout_sources_v2.py'),
        {'envelope_kind'},dict(common))
    history=extracted_functions(frozen_source(Q.parent/'historical_b3_payload_partition_v1.py'),
        {'pin_shape'},dict(common))
    metadata=extracted_functions(frozen_source(Q.parent/'nemotron9b_package_metadata_v27.py'),
        {'pin_shape'},dict(common,history=types.SimpleNamespace(pin_shape=history['pin_shape'])))
    tree=ast.parse(frozen_source(Q.parent/'nemotron9b_package_metadata_v27.py'))
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in('LIMIT','ENVELOPE_LIMIT')for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<authenticated-limits>','exec'),constants)
    calls=[]
    def stable_bytes(path,cap):
        calls.append((str(path),cap));require(len(raw)<=cap,'synthetic exact original cap');return raw
    def ordinary(*args):raise ValueError('Synthetic ordinary reader rejects unclassified oversized payload')
    source=frozen_source(Q/'minicpm5_layer1_operator_sdk_sources_v1.py')
    values=dict(common,hashlib=hashlib,json=json,
        predecessor=types.SimpleNamespace(envelope_kind=inherited['envelope_kind']),
        metadata=types.SimpleNamespace(pin_shape=metadata['pin_shape'],ENVELOPE_LIMIT=constants['ENVELOPE_LIMIT'],
            stable_bytes=stable_bytes,text=ordinary),
        ENVELOPES={'spec':'minicpm5_layer1_operator_sdk_cpu_execution_spec_v1',
            'source-review':'minicpm5_layer1_operator_sdk_runtime_source_review_v1','plan':'minicpm5_layer0_runtime_inputs_v1'})
    return extracted_functions(source,{'reference','envelope_kind','envelope_shape','text','load'},values),calls,constants


class ExactInheritedPlanTests(unittest.TestCase):
    def setUp(self):self.f,self.v,self.i=v2_fixture(with_plan=True)
    def seed(self):return self.v[self.f['seed_spec']['path']]
    def plan(self):return self.v[v2.INHERITED_PLAN['path']]

    def test_exact_inherited_identity_sha_and_complete_seed_preserved(self):
        before=copy.deepcopy((self.f,self.v,self.i));result=build(self.f,self.v,self.i)['values'];path=v2.INHERITED_PLAN['path']
        self.assertEqual(before,(self.f,self.v,self.i))
        self.assertEqual(result['spec']['input_extension'][path],self.seed()['input_extension'][path])
        self.assertEqual(result['spec']['source_sha256'][path],v2.INHERITED_PLAN['sha256'])
        self.assertEqual(result['candidate']['preservation']['exact_retained_typed_inputs'],{path:self.seed()['input_extension'][path]})
        self.assertNotIn(path,result['candidate']['preparation_only_inputs'])
        self.assertEqual(result['spec']['limits'],old.c.LIMITS)
        for key in('full_geometry_qualified','full_model_qualified','new_speedup_claim','runtime_launched'):
            self.assertIs(result['candidate'][key],False)

    def test_v1_reproduces_only_size_rejection_on_same_synthetic_input(self):
        # Hand v1 its own valid source-delivery envelope; only the inherited
        # oversized input differs from its passing original fixture.
        future,values,identities=copy.deepcopy((self.f,self.v,self.i))
        helper=values[future['preparer_delivery']['path']]
        helper['schema']='qwen35_kv256_alias_diagnostic_binding_source_delivery_v1'
        helper['source_sha256'][str(old.s.SELF)]='9'*64
        def read(ref):return dict(value=copy.deepcopy(values[ref['path']]),identity=copy.deepcopy(identities[ref['path']]))
        with self.assertRaisesRegex(ValueError,'No large preparation envelope'):
            old.s.prepare(future,read,lambda _:True,lambda:None)

    def test_every_complete_plan_identity_field_must_match_current_observation(self):
        path=v2.INHERITED_PLAN['path']
        for key in('realpath','device','inode','size','mtime_ns','ctime_ns'):
            with self.subTest(key=key):
                self.setUp();ident=self.seed()['input_extension'][path]['identity']
                ident[key]=('/synthetic/wrong.json'if key=='realpath'else ident[key]+1)
                with self.assertRaises(ValueError):build(self.f,self.v,self.i)

    def test_wrong_plan_digest_or_base_reference_rejected(self):
        path=v2.INHERITED_PLAN['path']
        self.seed()['input_extension'][path]['sha256']='a'*64
        with self.assertRaises(ValueError):build(self.f,self.v,self.i)
        self.setUp();overlay=self.v[self.seed()['history_overlay']['path']];overlay['base_plan']['sha256']='b'*64
        with self.assertRaises(ValueError):build(self.f,self.v,self.i)

    def test_arbitrary_oversized_inherited_input_rejected(self):
        for path in('/synthetic/oversized.json',str(Q/'minicpm5-layer1-operator-sdk-other-plan.json')):
            row=copy.deepcopy(self.seed()['input_extension'][v2.INHERITED_PLAN['path']]);row['identity']['realpath']=path
            self.seed()['input_extension'][path]=row
            with self.assertRaisesRegex(ValueError,'original8MiB'):build(self.f,self.v,self.i)
            del self.seed()['input_extension'][path]

    def test_exact_plan_size_even_if_less_than_ordinary_cap(self):
        path=v2.INHERITED_PLAN['path']
        for size in(8*v2.MIB,12803325,12803327,32*v2.MIB+1,64*v2.MIB):
            with self.subTest(size=size):
                self.setUp();self.seed()['input_extension'][path]['identity']['size']=size;self.i[path]['size']=size
                with self.assertRaises(ValueError):build(self.f,self.v,self.i)

    def test_plan_cannot_be_newly_promoted_or_have_seed_identity_dropped(self):
        path=v2.INHERITED_PLAN['path'];row=self.seed()['input_extension'][path]
        with self.assertRaises(ValueError):v2.retained_input_caps({path:row},{},{path:row},
            {k:v2.INHERITED_PLAN[k]for k in('path','sha256')},self.plan())
        with self.assertRaises(ValueError):v2.retained_input_caps({}, {path:row},{path:row},
            {k:v2.INHERITED_PLAN[k]for k in('path','sha256')},self.plan())

    def test_ordinary_cap_boundaries_unchanged(self):
        path='/synthetic/ordinary.json';row=dict(sha256='a'*64,identity=dict(realpath=path,device=1,inode=2,size=8*v2.MIB,mtime_ns=3,ctime_ns=4))
        self.seed()['input_extension'][path]=row;build(self.f,self.v,self.i)
        row['identity']['size']+=1
        with self.assertRaisesRegex(ValueError,'original8MiB'):build(self.f,self.v,self.i)

    def test_invalid_original_typed_shape_fields_rejected(self):
        for key,value in(('schema','wrong'),('source_sha256',[]),('source_contract',{}),
            ('latest_completion',[]),('output_root',None),('inputs',[])):
            with self.subTest(key=key):
                self.setUp();self.plan()[key]=value
                with self.assertRaises((ValueError,KeyError)):build(self.f,self.v,self.i)

    def test_invalid_plan_pin_and_original_bound_rejected(self):
        self.plan()['source_sha256']={'relative.py':'1'*64}
        with self.assertRaises(ValueError):build(self.f,self.v,self.i)
        self.setUp();self.plan()['inputs']=dict.fromkeys(range(100001),None)
        with self.assertRaises(ValueError):build(self.f,self.v,self.i)

    def test_v2_and_frozen_v1_helper_sources_merged_before_preparation(self):
        result=build(self.f,self.v,self.i)['values']['spec'];source=result['source_sha256']
        self.assertIn(str(v2.SELF),source)
        self.assertEqual(source[v2.PREDECESSOR_PREPARER['path']],v2.PREDECESSOR_PREPARER['sha256'])
        self.v[self.f['preparer_delivery']['path']]['source_sha256'].pop(v2.PREDECESSOR_PREPARER['path'])
        with self.assertRaises(ValueError):build(self.f,self.v,self.i)

    def test_separate_helper_review_contract_required(self):
        for key,value in(('schema','qwen35_kv256_alias_diagnostic_binding_source_delivery_v1'),
            ('predecessor_preparer',{}),('retained_input_exception',{})):
            with self.subTest(key=key):
                self.setUp();self.v[self.f['preparer_delivery']['path']][key]=value
                with self.assertRaises(ValueError):build(self.f,self.v,self.i)

    def test_actual_executed_preparer_pin_checked_before_builder(self):
        helper=copy.deepcopy(self.v[self.f['preparer_delivery']['path']])
        helper['source_sha256'][str(v2.SELF)]=hashlib.sha256(v2.SELF.read_bytes()).hexdigest()
        v2.verify_executed_preparer(helper)
        helper['source_sha256'][str(v2.SELF)]='0'*64
        with self.assertRaises(ValueError):v2.verify_executed_preparer(helper)
        source=v2.SELF.read_text();main=next(n for n in ast.parse(source).body if getattr(n,'name',None)=='main')
        text=ast.get_source_segment(source,main)
        self.assertLess(text.index('verify_executed_preparer('),text.index('result = prepare('))

    def test_pure_builder_does_not_read_real_plan_or_source(self):
        with patch('builtins.open',side_effect=AssertionError('No real IO in builder')),\
             patch.object(Path,'read_bytes',side_effect=AssertionError('No real IO in builder')):
            build(self.f,self.v,self.i)


class ReaderSourceProofTests(unittest.TestCase):
    def test_original_typed_reader_32MiB_exception_with_synthetic_bytes(self):
        _,values,_=v2_fixture(True);plan=values[v2.INHERITED_PLAN['path']]
        plan['synthetic_padding']='x'*(8*v2.MIB+1)
        raw=json.dumps(plan).encode();namespace,calls,limits=reader_namespace(raw)
        self.assertGreater(len(raw),limits['LIMIT']);self.assertEqual(limits['LIMIT'],8*v2.MIB)
        self.assertEqual(limits['ENVELOPE_LIMIT'],32*v2.MIB)
        reference=dict(path=v2.INHERITED_PLAN['path'],sha256=hashlib.sha256(raw).hexdigest())
        result=namespace['load'](reference)
        self.assertEqual(result,plan);self.assertEqual(calls,[(reference['path'],32*v2.MIB)])
        with self.assertRaises(ValueError):namespace['load'](dict(reference,sha256='0'*64))
        with self.assertRaises(ValueError):namespace['load'](dict(reference,path='/synthetic/arbitrary.json'))

    def test_exact_plan_shape_matches_original_pure_branch(self):
        _,values,_=v2_fixture(True);plan=values[v2.INHERITED_PLAN['path']];namespace,_,_=reader_namespace(b'{}')
        self.assertEqual(namespace['envelope_kind'](v2.INHERITED_PLAN['path']),'plan')
        for key,value in((None,None),('schema','wrong'),('source_sha256',{}),('source_sha256',{'relative.py':'1'*64}),
            ('source_contract',dict(path='/synthetic/ref.json',sha256='1'*64,bytes=3)),
            ('source_contract',dict(path='/synthetic/ref.json',sha256='1'*64,bytes=True)),
            ('latest_completion',[]),('output_root',None),('inputs',[])):
            copy_plan=copy.deepcopy(plan)
            if key:copy_plan[key]=value
            outcomes=[]
            for check in(lambda:namespace['envelope_shape'](v2.INHERITED_PLAN['path'],copy_plan),
                          lambda:v2.inherited_plan_shape(copy_plan)):
                try:check();outcomes.append(True)
                except(ValueError,KeyError):outcomes.append(False)
            self.assertEqual(outcomes[0],outcomes[1],(key,value))

    def test_frozen_runtime_reader_chain_and_identity_checks_retained(self):
        history=frozen_source(Q/'qwen35_package_preflight_history_v3.py')
        io=frozen_source(Q/'qwen35_sampler_runtime_io_v12.py')
        transport=frozen_source(Q.parent/'nemotron9b_package_metadata_v27.py')
        self.assertIn("require(identity(path) == row['identity']",history)
        self.assertIn("self.meta.read({'path': path, 'sha256': row['sha256']})",history)
        self.assertIn('result = sources.load(value)',io)
        self.assertIn('historical_identity(value["path"]) == before',io)
        self.assertIn('before.st_size <= cap',transport)
        self.assertIn("('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')",transport)

    def test_exact_seven_edit_reversal_and_all_v1_runtime_sources_unchanged(self):
        source,edits=derivation.derive();self.assertEqual(len(edits),7);self.assertEqual(source,v2.SELF.read_text())
        delivery=json.loads((Q/'qwen35_kv256_alias_diagnostic_serial_v1/source_delivery.json').read_text())
        for path,sha in delivery['new_source_sha256'].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),sha,path)
        self.assertEqual(hashlib.sha256(Path(derivation.PREDECESSOR['path']).read_bytes()).hexdigest(),derivation.PREDECESSOR['sha256'])

    def test_source_imports_and_help_have_no_native_or_runtime_effect(self):
        self.assertFalse({'torch','torch_neuronx','neuronxcc','ctypes','numpy'}&set(sys.modules))
        with patch.object(v2,'api',side_effect=AssertionError('No runtime API for help')):
            with self.assertRaises(SystemExit)as cm:v2.main(['--help'])
        self.assertEqual(cm.exception.code,0)


if __name__=='__main__':unittest.main(verbosity=2)
