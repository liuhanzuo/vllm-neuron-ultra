"""Source definitions and small JSON only; no runtime imports, physical IO or writes.

Execute the actual successor catalog/prepare/overlay definitions with supplied
readers, retain prior negative controls, and prove the complete active import
chain uses them. Full execution/admission is deliberately outside these tests.
"""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research'
sys.path.insert(0,str(ROOT))
from research import check_prepare_minicpm5_layer1_operator_native_v2 as previous
from research import minicpm5_layer1_operator_native_manifest_v1 as manifest
from research import minicpm5_layer1_operator_native_output_catalog_v2 as output_catalog

require=previous.require;definitions=previous.definitions
PREP=Q/'prepare_minicpm5_layer1_operator_native_v3.py'
RUNTIME=Q/'minicpm5_layer1_operator_sdk_runtime_completion_v4.py'
CATALOG=dict(path=str(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261111-v44.json'),
    sha256='0fbc64e48c4652b73594127c4d969da27340a766df37f94ca25e8f424beff9a0')


def assignment(path,name,scope):
    nodes=[n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name)and t.id==name for t in n.targets)]
    assert len(nodes)==1,(path,name)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


def read_small(ref):
    require(Path(ref['path']).suffix=='.json','Only small explicit JSON metadata')
    with Path(ref['path']).open('rb')as source:raw=source.read(2*2**20+1)
    require(len(raw)<=2*2**20 and hashlib.sha256(raw).hexdigest()==ref['sha256'],
        'Original bounded authenticated JSON bytes')
    return json.loads(raw)


OriginalFixture=previous.Fixture
class Fixture(OriginalFixture):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        scope=self.r.completion.latest.serial_catalog.__globals__
        maximum=assignment(RUNTIME,'MAX_SERIAL_RUNS',scope)
        definitions(RUNTIME,{'serial_catalog'},scope)
        self.r.completion.latest=SimpleNamespace(serial_catalog=scope['serial_catalog'],MAX_SERIAL_RUNS=maximum,sdk=scope['sdk'])
        self.scope.update(SELF=str(PREP),RUNNER=ROOT/'minicpm5_layer1_operator_native_runner_v10.py')
        definitions(PREP,{'audit_policy','executed_extension','historical_auditors','merge','extension','source_additions',
            'loaded_sources','resolve_binding','object_ref','encoded','paths','preserved','overlay_value'},self.scope)

    def expand(self,count):
        self.serial['runs']=[]
        for index in range(count):
            row=copy.deepcopy(self.row);row['terminal']=self.ref('terminal-%d'%index)
            row['tool']['exit_code']=(0,1,137,255)[index%4]
            self.objects[row['terminal']['path']]=dict(copy.deepcopy(self.terminal),actual_tool=row['tool'])
            self.serial['runs'].append(row)
        return self

    def validate(self):
        pins={p:s for row in self.serial['runs']for p,s in row['source_sha256'].items()}
        return self.r.completion.latest.serial_catalog(dict(serial_predecessors=self.serial_ref,source_sha256=pins),self.read,lambda:None)

    def additions(self):
        old={str(ROOT/('closure-%d.py'%n)):'d'*64 for n in range(578)}
        closure=dict(source_only=True,original_complete_source_map_must_remain=True,
            dynamic_import_resolution_verified=False,runtime_import_identity_verified=False,
            runtime_graph=dict(source_sha256=old,files=578,native_executed=False))
        return self.scope['source_additions'](closure,self.serial,{},self.scope['ACCEPTED_CLOSURE'],self.serial_ref,self.history())


class PriorControls(previous.Tests):
    def setUp(self):
        changed=patch.object(previous,'Fixture',Fixture);changed.start();self.addCleanup(changed.stop)


class CompleteCatalog(unittest.TestCase):
    def test_64_65_84_256_rows_preserve_every_failure_owner_unit_and_source(self):
        for count in (64,65,84,256):
            with self.subTest(count=count):
                f=Fixture().expand(count);before=copy.deepcopy(f.serial);result=f.validate()
                self.assertEqual(result['runs'],before['runs']);self.assertEqual(f.serial,before)
                self.assertEqual(result['owners'],[o for row in before['runs']for o in row['owners']])
                self.assertEqual([r['tool']['exit_code']for r in result['runs']],[(0,1,137,255)[n%4]for n in range(count)])
                pins=f.additions()
                self.assertTrue(all(pins[p]==s for row in before['runs']for p,s in row['source_sha256'].items()))
                result['runs'][0]['tool']['exit_code']=99;self.assertEqual(f.serial,before)

    def test_257_rows_rejected_in_both_runtime_and_prepare(self):
        f=Fixture().expand(257)
        with self.assertRaisesRegex(RuntimeError,'complete current post-r2'):f.validate()
        # Exercise the actual prepare cap directly, rather than stop first at its historical validator.
        with self.assertRaisesRegex(RuntimeError,'Root complete current serial catalog'):
            f.scope['source_additions'](dict(source_only=True,original_complete_source_map_must_remain=True,
                dynamic_import_resolution_verified=False,runtime_import_identity_verified=False,
                runtime_graph=dict(source_sha256={str(ROOT/('c%d.py'%n)):'a'*64 for n in range(578)},files=578,native_executed=False)),
                f.serial,{},f.scope['ACCEPTED_CLOSURE'],f.serial_ref,{})

    def test_duplicate_conflict_malformed_owner_unknown_tool_and_live_handoff_rejected_above64(self):
        for mode in ('duplicate','conflict','owner','tool','bool_exit','negative_exit','live','incomplete'):
            f=Fixture().expand(84)
            if mode=='duplicate':f.serial['runs'][-1]=copy.deepcopy(f.serial['runs'][0])
            elif mode=='conflict':f.serial['runs'][-1]['source_sha256'][f.core]='f'*64
            elif mode=='owner':f.serial['runs'][-1]['owners'][0]['start_ticks']='bad'
            elif mode=='tool':f.serial['runs'][-1]['tool']['kind']='invented'
            elif mode=='bool_exit':f.serial['runs'][-1]['tool']['exit_code']=False
            elif mode=='negative_exit':f.serial['runs'][-1]['tool']['exit_code']=-1
            elif mode=='live':f.objects[f.serial['current_handoff']['path']]['live_tool_sessions']=[123]
            else:f.serial['all_later_serial_runs_included']=False
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):f.validate()

    def test_actual84_catalog_and_accepted578_source_closure_are_preserved(self):
        f=Fixture();f.scope['LIMITS']=assignment(Q/'audit_minicpm5_layer1_operator_sdk_result_v3.py','LIMITS',{})
        actual_binding=assignment(Q/'minicpm5_layer1_operator_sdk_completion_v2.py','BINDING',{'Q':Q})
        f.r.completion.latest.sdk.BINDING=actual_binding
        serial=read_small(CATALOG);before=copy.deepcopy(serial)
        self.assertEqual(len(serial['runs']),84)
        pins={p:s for row in serial['runs']for p,s in row['source_sha256'].items()}
        result=f.r.completion.latest.serial_catalog(dict(serial_predecessors=CATALOG,source_sha256=pins),read_small,lambda:None)
        self.assertEqual(result['runs'],before['runs']);self.assertEqual(serial,before)
        self.assertTrue(any(row['tool']['exit_code']!=0 for row in result['runs']))
        history=f.scope['historical_auditors'](CATALOG,serial,read_small,lambda:None)
        closure_ref=assignment(PREP,'ACCEPTED_CLOSURE',{'Q':Q});f.scope['ACCEPTED_CLOSURE']=closure_ref
        closure=read_small(closure_ref)
        added=f.scope['source_additions'](closure,serial,{},closure_ref,CATALOG,history)
        self.assertTrue(all(added[p]==s for p,s in pins.items()))
        self.assertTrue(all(added[p]==s for p,s in closure['runtime_graph']['source_sha256'].items()))
        self.assertEqual(result['owners'],[o for row in before['runs']for o in row['owners']])

    def test_actual_overlay_binding_keeps_44503_sources_8791_inputs_and_all84_rows(self):
        f=Fixture().expand(84);latest=f.r.completion.latest
        original_binding=assignment(Q/'minicpm5_layer1_operator_recovered_completion_v2.py','BINDING',{'Q':Q})
        completion=SimpleNamespace(require=require,sources=f.r.sources,BINDING=original_binding,latest=latest,
            physical=SimpleNamespace(GROUP_PREFIX='/synthetic/group/',group_unit=lambda s:s))
        fixture_scope={'hashlib':hashlib}
        definitions(Q/'check_minicpm5_layer1_operator_native_driver_v2.py',{'fixture'},fixture_scope)
        base,native=fixture_scope['fixture']()
        base.update(source_sha256={str(ROOT/'original-source'/(str(n)+'.py')):'a'*64 for n in range(44503)},
            inputs={'/synthetic/input/'+str(n):{'sha256':'b'*64,'identity':{'original':n}}for n in range(8791)},
            latest_completion={'original':True},layer1_operator_boundary=latest.sdk.BINDING,
            interpreters={'native':'/home/ubuntu/venvs/model_reference_2026/bin/python'},model={'original':'MiniCPM5'},dtype='torch.bfloat16')
        for index,key in enumerate(('actual_exit','recorder_terminal')):
            row=f.serial['runs'][index];row['terminal']=original_binding[key];row['tool']['exit_code']=0
            f.refs[row['terminal']['path']]=row['terminal'];f.objects[row['terminal']['path']]={'completed':'original recovery'}
        recovered=dict(metadata_complete=True,runtime_admitted=False,original_model_exit_code=1,binding=original_binding,
            base_plan_ref=f.ref('base-plan'),base_plan=base,source_sha256=dict(base['source_sha256']),native={'native':native},
            historical={'later_serial':{'runs':copy.deepcopy(f.serial['runs'][:65])}},done={'operator_sdk':{'owners':[]}},
            actual_exit={'owner':f.owner,'actual_tool':f.serial['runs'][0]['tool']},
            recorder_terminal={'owner':f.owner,'actual_tool':f.serial['runs'][1]['tool']},owners=[f.owner],complete_known_units=[])
        contract=manifest.build(base,native);manifest_ref=f.ref('native-manifest');f.objects[manifest_ref['path']]=contract
        overlay_scope=dict(copy=copy,Path=Path,re=re,completion=completion,manifest=manifest,output_contract=output_catalog,
            require=require,digest=manifest.digest,SCHEMA='minicpm5_layer1_operator_native_small_overlay_v2')
        definitions(Q/'minicpm5_layer1_operator_native_overlay_v3.py',{'resolve','output_catalog'},overlay_scope)
        f.r.overlays=SimpleNamespace(**overlay_scope);f.r.completion=completion;f.r.LOCK_POLICY={'unchanged':True}
        f.scope.update(BASE_SOURCE_COUNT=44503,BASE_INPUT_COUNT=8791)
        life_scope=dict(copy=copy,Path=Path,re=re,require=require,catalog=output_catalog,digest=manifest.digest,ROOT=str(ROOT))
        for name in ('PYTHON','PROPS'):assignment(Q/'minicpm5_layer1_operator_native_lifecycle_core_v2.py',name,life_scope)
        definitions(Q/'minicpm5_layer1_operator_native_lifecycle_core_v2.py',{'ref','build_contract'},life_scope)
        f.r.life.build_contract=life_scope['build_contract']
        sources={p:s for row in f.serial['runs']for p,s in row['source_sha256'].items()}
        sources[str(f.scope['RUNNER'])]='c'*64;before=copy.deepcopy(recovered)
        overlay=f.scope['overlay_value'](recovered,'catalog-memory',f.serial_ref,manifest_ref,sources,{})
        resolved,contract,spec=f.scope['resolve_binding'](recovered,overlay,f.read,lambda:None)
        self.assertEqual(recovered,before);self.assertEqual(resolved['current_scope']['later_serial']['runs'],f.serial['runs'])
        self.assertTrue(all(resolved['plan']['source_sha256'][p]==s for p,s in base['source_sha256'].items()))
        self.assertEqual(resolved['plan']['inputs'],base['inputs']);self.assertEqual(len(base['inputs']),8791)
        self.assertEqual(resolved['plan']['dtype'],'torch.bfloat16');self.assertEqual(resolved['plan']['model'],base['model'])
        self.assertEqual(contract['resources'],life_scope['PROPS']);self.assertEqual(spec['runner']['path'],str(f.scope['RUNNER']))
        self.assertEqual(len(resolved['native_runtime_output_catalog']['payload_files']),936)
        self.assertFalse(resolved['runtime_admitted']);self.assertEqual(recovered['original_model_exit_code'],1)
        f.serial['runs'].pop(64)
        with self.assertRaisesRegex(RuntimeError,'recovery-historical serial'):f.scope['resolve_binding'](recovered,overlay,f.read,lambda:None)


class SuccessorWiring(unittest.TestCase):
    def test_only_caps_and_explicit_import_entrypoint_wiring_change(self):
        # Normalize only the reviewed differences; equality covers every original
        # function, guard, resource/science constant, and prepare/bind/CLI branch.
        pairs=[
            (Q/'minicpm5_layer1_operator_sdk_runtime_completion_v3.py',RUNTIME),
            (Q/'minicpm5_layer1_operator_recovered_completion_v1.py',Q/'minicpm5_layer1_operator_recovered_completion_v2.py'),
            (Q/'minicpm5_layer1_operator_native_overlay_v2.py',Q/'minicpm5_layer1_operator_native_overlay_v3.py'),
            (Q/'minicpm5_layer1_operator_native_process_adapter_v6.py',Q/'minicpm5_layer1_operator_native_process_adapter_v7.py'),
            (ROOT/'minicpm5_layer1_operator_native_runner_v9.py',ROOT/'minicpm5_layer1_operator_native_runner_v10.py'),
            (ROOT/'minicpm5_layer1_operator_native_outer_runner_v3.py',ROOT/'minicpm5_layer1_operator_native_outer_runner_v4.py'),
            (Q/'prepare_minicpm5_layer1_operator_native_v2.py',PREP)]
        names={old.stem:new.stem for old,new in pairs}
        def normalized(path,successor):
            raw=path.read_text()
            if successor:
                raw=raw.replace('MAX_SERIAL_RUNS=256\n','').replace("len(value['runs'])<=MAX_SERIAL_RUNS","len(value['runs'])<=64")
                raw=raw.replace("len(serial['runs'])<=r.completion.latest.MAX_SERIAL_RUNS","len(serial['runs'])<=64")
                for old,new in names.items():raw=raw.replace(new,old)
            tree=ast.parse(raw)
            if isinstance(tree.body[0],ast.Expr)and isinstance(tree.body[0].value,ast.Constant):tree.body.pop(0)
            return ast.dump(tree)
        for old,new in pairs:
            with self.subTest(path=str(new)):self.assertEqual(normalized(old,False),normalized(new,True))
        self.assertEqual(assignment(RUNTIME,'MAX_SERIAL_RUNS',{}),256)
        # Prove these are the callable imports used by both actual entrypoints.
        expected={PREP:{'r':'minicpm5_layer1_operator_native_process_adapter_v7'},
            pairs[4][1]:{'runtime':'minicpm5_layer1_operator_native_process_adapter_v7'},
            pairs[5][1]:{'r':'minicpm5_layer1_operator_native_process_adapter_v7'},
            pairs[3][1]:{'completion':'minicpm5_layer1_operator_recovered_completion_v2','overlays':'minicpm5_layer1_operator_native_overlay_v3'},
            pairs[2][1]:{'completion':'minicpm5_layer1_operator_recovered_completion_v2'},
            pairs[1][1]:{'latest':'minicpm5_layer1_operator_sdk_runtime_completion_v4'}}
        for path,wanted in expected.items():
            imports={alias.asname:alias.name for node in ast.parse(path.read_bytes()).body
                if isinstance(node,ast.ImportFrom)for alias in node.names}
            self.assertTrue(all(imports.get(alias)==name for alias,name in wanted.items()),path)
        self.assertEqual(assignment(PREP,'RUNNER',{'ROOT':ROOT}),pairs[4][1])
        self.assertEqual(assignment(PREP,'OUTER',{'ROOT':ROOT}),pairs[5][1])

    def test_native_imports_never_loaded_by_source_metadata_checks(self):
        self.assertFalse(any(name=='torch'or name.startswith(('torch.','vllm','transformers','neuronxcc'))for name in sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
