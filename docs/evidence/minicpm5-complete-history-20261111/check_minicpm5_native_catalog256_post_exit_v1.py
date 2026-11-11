"""Source/definition and bounded JSON checks only; no runtime graph imports or IO.

The actual recorder/auditor/comparison functions stay byte-for-byte unchanged;
explicit successor imports fix their catalog path without global mutation.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research';sys.path.insert(0,str(ROOT))
from research import check_prepare_minicpm5_layer1_operator_native_v3 as prior
require=prior.require;definitions=prior.definitions;assignment=prior.assignment
AUDIT=Q/'audit_minicpm5_layer1_operator_native_result_v3.py'
RECORDER=Q/'record_minicpm5_layer1_operator_native_model_exit_v2.py'
COMPARISON=Q/'run_minicpm5_native_sdk_saved_comparison_v2.py'
PREP=Q/'prepare_minicpm5_layer1_operator_native_v4.py'
CORE4=Q/'minicpm5_layer1_operator_native_post_exit_core_v4.py'
CORE5=Q/'minicpm5_layer1_operator_native_post_exit_core_v5.py'
CATALOG=dict(path=str(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261111-v45.json'),
    sha256='bda6d214cf0a77e379f1aeba68c669d89f6977049bcea4e04bb79c0dfe91a426')
BINDING=dict(path=str(Q/'minicpm5-layer1-operator-native-r2-catalog256-20261111-binding-result.json'),
    sha256='f538a813abe84dd3c320c93ab5b4b97d51b3c7ca87d50a481635c20df4653438')
LIMITS=assignment(Q/'audit_minicpm5_layer1_operator_sdk_result_v3.py','LIMITS',{})


def load_helpers(path,names,**extra):
    scope=dict(copy=copy,hashlib=hashlib,json=json,Path=Path,re=re,require=require,ROOT=ROOT,Q=Q,
        LIMITS=LIMITS,SELF=str(path));scope.update(extra);definitions(path,names,scope)
    return SimpleNamespace(**scope)


class WitnessFixture(prior.Fixture):
    def __init__(self,auditor=False,current=True,code=0):
        super().__init__(auditor=auditor,code=code)
        definitions(PREP,{'executed_extension','audit_policy','historical_auditors'},self.scope)
        if current:
            script=str(AUDIT if auditor else RECORDER)
            pins={script:'a'*64,str(AUDIT):'a'*64,str(CORE4):'b'*64,str(CORE5):'c'*64}
            self.owner['argv'][2]=script;self.pins=pins;self.spec['source_extension']=pins
            self.review['source_extension']=pins;self.row['source_sha256']=dict(pins)
            if auditor:self.result['source']=dict(path=script,sha256=pins[script])
            else:
                self.result['recorder_source']=dict(path=script,sha256=pins[script])
                self.terminal['source_extension']=pins


class Wiring(unittest.TestCase):
    def test_only_imports_change_all_runtime_guards_and_schemas_remain_identical(self):
        changes=[
            (Q/'audit_minicpm5_layer1_operator_native_result_v2.py',AUDIT,[
                ('minicpm5_layer1_operator_native_process_adapter_v6','minicpm5_layer1_operator_native_process_adapter_v7'),
                ('minicpm5_layer1_operator_native_post_exit_core_v2','minicpm5_layer1_operator_native_post_exit_core_v5'),
                ('minicpm5_layer1_operator_native_outer_runner_v3','minicpm5_layer1_operator_native_outer_runner_v4')]),
            (Q/'record_minicpm5_layer1_operator_native_model_exit_v1.py',RECORDER,[
                ('audit_minicpm5_layer1_operator_native_result_v2','audit_minicpm5_layer1_operator_native_result_v3'),
                ('minicpm5_layer1_operator_native_post_exit_core_v3','minicpm5_layer1_operator_native_post_exit_core_v5')]),
            (Q/'run_minicpm5_native_sdk_saved_comparison_v1.py',COMPARISON,[
                ('audit_minicpm5_layer1_operator_native_result_v2','audit_minicpm5_layer1_operator_native_result_v3')]),
            (Q/'minicpm5_layer1_operator_native_post_exit_core_v2.py',CORE4,[]),
            (Q/'minicpm5_layer1_operator_native_post_exit_core_v3.py',CORE5,[
                ('minicpm5_layer1_operator_native_post_exit_core_v2','minicpm5_layer1_operator_native_post_exit_core_v4')])]
        for old,new,replacements in changes:
            raw=new.read_text()
            for before,after in replacements:
                self.assertIn(after,raw);raw=raw.replace(after,before)
            old_tree=ast.parse(old.read_bytes());new_tree=ast.parse(raw)
            old_tree.body.pop(0);new_tree.body.pop(0)
            with self.subTest(path=str(new)):self.assertEqual(ast.dump(old_tree),ast.dump(new_tree))
        self.assertEqual(CORE4.read_bytes(),(Q/'minicpm5_layer1_operator_native_post_exit_core_v2.py').read_bytes())
        def funcs(path):return {n.name:ast.dump(n)for n in ast.parse(path.read_bytes()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        before=funcs(Q/'prepare_minicpm5_layer1_operator_native_v3.py');after=funcs(PREP)
        self.assertEqual(set(before),set(after))
        for key in before:
            if key!='executed_extension':self.assertEqual(before[key],after[key],key)

    def test_actual_runtime_import_chain_reaches_shared256_validator(self):
        paths={RECORDER:('audit','audit_minicpm5_layer1_operator_native_result_v3'),
            COMPARISON:('a','audit_minicpm5_layer1_operator_native_result_v3'),
            AUDIT:('r','minicpm5_layer1_operator_native_process_adapter_v7'),
            Q/'minicpm5_layer1_operator_native_process_adapter_v7.py':('completion','minicpm5_layer1_operator_recovered_completion_v2'),
            Q/'minicpm5_layer1_operator_recovered_completion_v2.py':('latest','minicpm5_layer1_operator_sdk_runtime_completion_v4')}
        for path,(alias,wanted)in paths.items():
            imports={a.asname:a.name for node in ast.parse(path.read_bytes()).body if isinstance(node,ast.ImportFrom)for a in node.names}
            self.assertEqual(imports[alias],wanted)
        self.assertEqual(assignment(prior.RUNTIME,'MAX_SERIAL_RUNS',{}),256)
        core_import=next(n for n in ast.parse(CORE5.read_bytes()).body if isinstance(n,ast.ImportFrom))
        self.assertEqual(core_import.module,'research.minicpm5_layer1_operator_native_post_exit_core_v4')
        self.assertIn('validate',[n.name for n in core_import.names])

    def test_actual85_catalog_all_failed_rows_and_v10_contract_unchanged(self):
        f=WitnessFixture(current=False);f.scope['LIMITS']=LIMITS
        f.r.completion.latest.sdk.BINDING=assignment(Q/'minicpm5_layer1_operator_sdk_completion_v2.py','BINDING',{'Q':Q})
        catalog=prior.read_small(CATALOG);old=prior.read_small(prior.CATALOG)
        self.assertEqual(len(catalog['runs']),85);self.assertEqual(catalog['runs'][:-1],old['runs'])
        pins={p:s for row in catalog['runs']for p,s in row['source_sha256'].items()}
        got=f.r.completion.latest.serial_catalog(dict(serial_predecessors=CATALOG,source_sha256=pins),prior.read_small,lambda:None)
        self.assertEqual(got['runs'],catalog['runs']);self.assertEqual(sum(x['tool']['exit_code']!=0 for x in got['runs']),22)
        history=f.scope['historical_auditors'](CATALOG,catalog,prior.read_small,lambda:None)
        self.assertNotIn(str(AUDIT),history);self.assertNotIn(str(CORE4),history);self.assertNotIn(str(CORE5),history)
        binding=prior.read_small(BINDING);spec=prior.read_small(binding['spec']);overlay=prior.read_small(binding['overlay'])
        runner=spec['runner'];self.assertEqual(runner,binding['contract']['runner'])
        self.assertEqual(runner['path'],str(ROOT/'minicpm5_layer1_operator_native_runner_v10.py'))
        self.assertEqual(hashlib.sha256(Path(runner['path']).read_bytes()).hexdigest(),runner['sha256'])
        life_scope=dict(Path=Path,re=re,require=require)
        definitions(Q/'minicpm5_layer1_operator_native_lifecycle_core_v2.py',{'ref','command'},life_scope)
        admission=dict(path=binding['contract']['output_root']+'/admission.json',sha256='a'*64)
        for mode in ('coordinator','native'):
            self.assertEqual(life_scope['command'](binding['contract'],mode,admission)[2],runner['path'])
        self.assertEqual(binding['base_source_files'],44503);self.assertEqual(binding['base_inputs'],8791)
        for path in (Q/'audit_minicpm5_layer1_operator_native_result_v2.py',Q/'minicpm5_layer1_operator_native_post_exit_core_v2.py',Q/'minicpm5_layer1_operator_native_post_exit_core_v3.py'):
            self.assertIn(str(path),overlay['source_extension'])

    def test_no_native_imports(self):
        self.assertFalse(any(x=='torch'or x.startswith(('torch.','vllm','transformers','neuronxcc'))for x in sys.modules))


class Witnesses(unittest.TestCase):
    def test_exact_old_and_new_producers_preserve_actual_failed_recorder_results(self):
        for current in (False,True):
            for auditor in (False,True):
                for code in ((0,)if auditor else(0,1,137,255)):
                    f=WitnessFixture(auditor,current,code);before=copy.deepcopy(f.serial);history=f.history()
                    self.assertEqual(f.serial,before)
                    self.assertIn(str(AUDIT)if current else f.auditor,history)
                    if current:self.assertEqual({str(CORE4),str(CORE5),str(AUDIT)},set(history))

    def test_new_complete_pair_required_no_old_or_future_source_preenrollment(self):
        for mode in ('missing_core4','missing_core5','future','old_audit','wrong_argv','wrong_source','changed_hash','unwitnessed','failed_audit'):
            f=WitnessFixture(auditor=mode=='failed_audit')
            if mode.startswith('missing_'):
                path=str(CORE4 if mode=='missing_core4'else CORE5);f.pins.pop(path);f.row['source_sha256'].pop(path)
            elif mode=='future':f.pins[str(Q/'audit_minicpm5_layer1_operator_native_result_v99.py')]='a'*64;f.row['source_sha256']=dict(f.pins)
            elif mode=='old_audit':f.pins[str(Q/'audit_minicpm5_layer1_operator_native_result_v2.py')]='a'*64;f.row['source_sha256']=dict(f.pins)
            elif mode=='wrong_argv':f.owner['argv'][2]=str(Q/'record_minicpm5_layer1_operator_native_model_exit_v1.py')
            elif mode=='wrong_source':f.result['recorder_source']['path']=str(Q/'record_minicpm5_layer1_operator_native_model_exit_v99.py')
            elif mode=='changed_hash':f.row['source_sha256'][str(CORE4)]='f'*64
            elif mode=='unwitnessed':f.terminal['schema']='source_assertion_only'
            else:f.row['tool']['exit_code']=1
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):f.history()

    def test_old_producer_cannot_enroll_new_auditor_core_pair(self):
        for auditor in (False,True):
            f=WitnessFixture(auditor,current=False)
            f.pins[str(AUDIT)]='a'*64;f.pins[str(CORE4)]='b'*64;f.pins[str(CORE5)]='c'*64
            f.row['source_sha256']=dict(f.pins)
            with self.assertRaises(RuntimeError):f.history()

    def test_missing_model_receipt_review_or_owner_still_denies(self):
        for mode in ('receipt','review','owner','result_owner','result_spec','flags'):
            f=WitnessFixture()
            if mode=='receipt':del f.objects[f.ref('producer-result')['path']]
            elif mode=='review':f.review['passed']=False
            elif mode=='owner':f.row['owners']=[]
            elif mode=='result_owner':f.result['recorder']=dict(f.owner,pid=999)
            elif mode=='result_spec':f.result['recorder_spec']=f.ref('wrong')
            else:f.result['self_exit_claimed']=True
            with self.subTest(mode=mode),self.assertRaises((RuntimeError,KeyError)):f.history()


class SourceExtensions(unittest.TestCase):
    def setUp(self):
        f=WitnessFixture();self.r=f.r
        self.audit=load_helpers(AUDIT,{'validate_spec','merged_plan'},r=self.r,core=SimpleNamespace(__file__=str(CORE5)))
        self.recorder=load_helpers(RECORDER,{'validate_spec','actual_tool'},r=self.r,core=SimpleNamespace(__file__=str(CORE5)))
        self.comparison=load_helpers(COMPARISON,{'audit_terminal','catalog_binding'},r=self.r,a=self.audit)
        core=load_helpers(CORE4,{'actual_tool'},life=self.r.life,__file__=str(CORE5))
        self.audit.core=core;self.audit.validate_spec.__globals__['core']=SimpleNamespace(__file__=str(CORE5))
        self.comparison.audit_terminal.__globals__['a']=self.audit

    def test_old_map_preserved_disjoint_extensions_only_no_equal_or_changed_overlaps(self):
        binding=prior.read_small(BINDING);overlay=prior.read_small(binding['overlay'])
        old=dict(overlay['source_extension']);before=copy.deepcopy(old)
        new={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in (RECORDER,AUDIT,CORE4,CORE5)}
        result=self.audit.merged_plan({'source_sha256':old,'inputs':{'all':'original8791'}},new)
        self.assertEqual(old,before);self.assertEqual(result['inputs'],{'all':'original8791'})
        self.assertTrue(all(result['source_sha256'][p]==s for p,s in before.items()))
        for old_path in (Q/'audit_minicpm5_layer1_operator_native_result_v2.py',Q/'minicpm5_layer1_operator_native_post_exit_core_v2.py',Q/'minicpm5_layer1_operator_native_post_exit_core_v3.py'):
            for sha in (old[str(old_path)],'f'*64):
                with self.assertRaisesRegex(RuntimeError,'cannot replace'):self.audit.merged_plan({'source_sha256':old},{str(old_path):sha})

    def test_new_recorder_auditor_specs_keep_old_schema_limits_and_required_new_pins(self):
        f=WitnessFixture();sr=f.ref('spec');rr=f.ref('review');tool=dict(kind='tool_session',session_id=55,terminal_chunk='abcdef',exit_code=1)
        pins={str(p):'a'*64 for p in (RECORDER,AUDIT,CORE4,CORE5)}
        spec=dict(schema='minicpm5_native_model_exit_recorder_spec_v1',live=f.ref('live'),model_spec=f.ref('model-spec'),
            model_review=f.ref('model-review'),actual_tool=tool,source_extension=pins,output=str(Q/'minicpm5-layer1-operator-native-synthetic-model-actual-exit-v1.json'),limits=LIMITS)
        review=dict(schema='minicpm5_native_model_exit_recorder_source_review_v1',reviewer='/root',passed=True,blockers=[],spec=sr,source_extension=pins)
        self.recorder.validate_spec(sr,spec,review)
        audit=dict(schema='minicpm5_native_post_exit_audit_spec_v1',model_exit=f.ref('exit'),recorder_exit=f.ref('recorder'),current_serial_catalog=CATALOG,
            source_extension=pins,output=str(Q/'minicpm5-layer1-operator-native-synthetic-post-exit-audit-v1.json'),limits=LIMITS)
        ar=dict(review,schema='minicpm5_native_post_exit_audit_source_review_v1')
        self.audit.validate_spec(sr,rr,audit,ar)
        for obj,fn,args in ((spec,self.recorder.validate_spec,(sr,spec,review)),(audit,self.audit.validate_spec,(sr,rr,audit,ar))):
            obj['limits']=dict(LIMITS,cpu=0)
            with self.assertRaises(RuntimeError):fn(*args)

    def test_comparison_accepts_only_exact_completed_new_auditor(self):
        scope={'e':SimpleNamespace(a=self.audit,r=self.r,Q=Q,LIMITS=LIMITS),'Path':Path,'copy':copy}
        definitions(Q/'check_minicpm5_native_sdk_saved_comparison_entry_v1.py',{'ref','terminal_fixture'},scope)
        terminal,result,spec,review=scope['terminal_fixture']();tool=self.comparison.audit_terminal(terminal,result,spec,review)
        self.assertEqual(tool,terminal['actual_tool']);self.assertEqual(result['source']['path'],str(AUDIT))
        for mode in ('old_source','old_argv','false_exit','failed_exit','self_exit','raw_missing','sdk_promoted'):
            t,v,s,w=scope['terminal_fixture']()
            if mode=='old_source':v['source']['path']=str(Q/'audit_minicpm5_layer1_operator_native_result_v2.py')
            elif mode=='old_argv':t['owner']['argv'][2]=str(Q/'audit_minicpm5_layer1_operator_native_result_v2.py')
            elif mode=='false_exit':t['actual_tool']['exit_code']=False
            elif mode=='failed_exit':t['actual_tool']['exit_code']=1
            elif mode=='self_exit':v['self_exit_claimed']=True
            elif mode=='raw_missing':v['every_raw_byte_reread_after_all_model_processes_exited']=False
            else:v['original_sdk_model_exit_code']=0
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):self.comparison.audit_terminal(t,v,s,w)

    def test_comparison_preserves_full_current_catalog_and_rejects_missing_predecessor(self):
        f=WitnessFixture().expand(85);serial=f.validate();before=copy.deepcopy(serial)
        prior_scope=dict(serial,runs=serial['runs'][:84]);last=serial['runs'][-1]
        self.comparison.catalog_binding(serial,prior_scope,[(last['terminal'],last['owners'][0],last['tool'])])
        self.assertEqual(serial,before);serial['runs'].pop(17)
        with self.assertRaises(RuntimeError):self.comparison.catalog_binding(serial,prior_scope,[])


if __name__=='__main__':unittest.main(verbosity=2)
