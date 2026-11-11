"""Pure successor-consumer tests; no runtime imports, process or payload IO.

Run the extracted actual preparation functions with authenticated in-memory
metadata callbacks. The original completed recovery and overlay callbacks are
observed mocks: this is not a reconstruction of the real 47,823-source plan.
The real producer's pure source_binding function supplies independent fixtures.
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

Q=Path(__file__).resolve().parent;ROOT=Q.parent
PREP=Q/'prepare_minicpm5_layer1_operator_native_v7.py'
OLD=Q/'prepare_minicpm5_layer1_operator_native_v6.py'
spec=importlib.util.spec_from_file_location('prior_preparation_checks',Q/'check_prepare_minicpm5_layer1_operator_native_v2.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
require=previous.require;definitions=previous.definitions
NAMES={'audit_policy','legacy_executed_extension','inherited_source_plan','inherited_source_binding',
       'completed_inherited_extension','executed_extension','historical_auditors','merge','extension',
       'source_additions','loaded_sources','resolve_binding','object_ref','encoded','paths','preserved',
       'current_history_freshness'}
REC=str(Q/'record_minicpm5_layer1_operator_native_model_exit_v3.py')
AUD=str(Q/'audit_minicpm5_layer1_operator_native_result_v4.py')
CORE4=str(Q/'minicpm5_layer1_operator_native_post_exit_core_v4.py')
CORE5=str(Q/'minicpm5_layer1_operator_native_post_exit_core_v5.py')
RUNNER=str(ROOT/'minicpm5_layer1_operator_native_runner_v12.py')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class LegacyFixture(previous.Fixture):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.scope['digest']=digest
        self.r.completion.latest.MAX_SERIAL_RUNS=256
        definitions(PREP,NAMES,self.scope)


class PriorControls(previous.Tests):
    def setUp(self):
        substitution=patch.object(previous,'Fixture',LegacyFixture)
        substitution.start();self.addCleanup(substitution.stop)


class Fixture(LegacyFixture):
    def __init__(self,model_exit=0,inherited_entries=False):
        super().__init__()
        self.calls=[];self.guard_calls=0
        self.r.LOCK_POLICY={'original':'retained-descriptor'}
        self.r.completion.BINDING={'original':'completed-recovery'}
        self.r.completion.latest.MAX_SERIAL_RUNS=256
        self.scope.update(BASE_SOURCE_COUNT=5,BASE_INPUT_COUNT=2)
        self.base={'source_sha256':{RUNNER:'a'*64,CORE4:'b'*64,CORE5:'c'*64,
                   str(ROOT/'package-a.py'):'d'*64,str(ROOT/'package-b.py'):'e'*64},
                   'inputs':{'first':{'identity':1},'second':{'identity':2}},'model':'original-bf16'}
        self.plan=copy.deepcopy(self.base)
        if inherited_entries:self.plan['source_sha256'].update({REC:'f'*64,AUD:'9'*64})
        self.recovered={'binding':self.r.completion.BINDING,'metadata_complete':True,'runtime_admitted':False,
                        'original_model_exit_code':1,'base_plan':self.base,'source_sha256':self.base['source_sha256']}
        self.r.completion.completed=self.recovery
        self.r.overlays=SimpleNamespace(resolve=self.resolve)
        self.r.sources.text=lambda *args:(_ for _ in ()).throw(AssertionError('No actual source IO'))
        self.r.digest=digest
        self.model_spec_ref=self.ref('model-spec');self.model_review_ref=self.ref('model-review')
        self.overlay_ref=self.ref('model-overlay')
        self.overlay={'synthetic':'original-overlay'}
        self.model_spec={'schema':'minicpm5_native_process_spec_v1','overlay':self.overlay_ref,
                         'runner':{'path':RUNNER,'sha256':'a'*64},'unit':'trn1-mini-synthetic',
                         'lock_policy':self.r.LOCK_POLICY}
        self.model_review={'schema':'minicpm5_native_process_source_review_v1','reviewer':'/root','passed':True,
                           'blockers':[],'spec':self.model_spec_ref,'overlay':self.overlay_ref,
                           'lock_policy':self.r.LOCK_POLICY,'resolved_plan_sha256':digest(self.plan)}
        self.objects.update({self.model_spec_ref['path']:self.model_spec,self.model_review_ref['path']:self.model_review,
                             self.overlay_ref['path']:self.overlay})
        self.pins={}if inherited_entries else{REC:'f'*64,AUD:'9'*64}
        self.spec['source_extension']=self.pins;self.review['source_extension']=self.pins
        self.owner['argv'][2]=REC
        self.result.update(recorder_source={'path':REC,'sha256':'f'*64},schema='minicpm5_native_model_actual_exit_v1'if model_exit==0
                           else'minicpm5_native_model_failed_actual_exit_v1',resolved_plan_sha256=digest(self.plan),
                           actual_exit_observed=True,exact_owner_gone=True,actual_exit_code=model_exit,
                           lock_identity=self.r.ORIGINAL_LOCK,full_model_qualified=False,new_speedup_claim=False,
                           owner={'pid':10,'start_ticks':'100','ppid':1,'argv':['synthetic-model'],'cgroup':'0::/synthetic'})
        self.spec['actual_tool']['exit_code']=model_exit
        self.result['actual_tool']=self.spec['actual_tool']
        self.terminal['source_extension']=self.pins
        self.row['source_sha256']={REC:'f'*64,AUD:'9'*64,CORE4:'b'*64,CORE5:'c'*64}
        # Use the real producer's pure function as an independent expected receipt.
        a_scope=dict(copy=copy,Path=Path,re=re,ROOT=ROOT,Q=Q,SELF=AUD,r=self.r,require=require,
                     core=SimpleNamespace(__file__=CORE5))
        definitions(Q/'audit_minicpm5_layer1_operator_native_result_v4.py',{'merged_plan','source_binding'},a_scope)
        self.produce_binding=a_scope['source_binding']
        _,proof=self.produce_binding(self.plan,self.pins,self.model_spec_ref,self.model_review_ref,self.overlay_ref,REC)
        self.result['source_binding']=proof
        self.prior_rows=[]

    def guard(self):self.guard_calls+=1

    def recovery(self,binding,read,guard,source_read):
        self.calls.append('complete');require(binding==self.r.completion.BINDING,'exact recovery input');guard()
        return copy.deepcopy(self.recovered)

    def resolve(self,overlay,recovered,read,guard):
        self.calls.append('resolve');require(overlay==self.overlay and recovered==self.recovered,'actual full recovery before overlay')
        guard()
        return {'metadata_complete':True,'runtime_admitted':False,'plan':copy.deepcopy(self.plan),
                'plan_sha256':digest(self.plan),'current_scope':{'later_serial':{'runs':copy.deepcopy(self.prior_rows)}}}

    def consume(self,cache=None):
        return self.scope['executed_extension'](self.terminal,self.row,self.read,self.guard,{}if cache is None else cache)

    def as_auditor(self):
        self.recorder_terminal=self.terminal;self.recorder_row=self.row;self.recorder_spec=self.spec
        self.recorder_result=self.result
        self.recorder_ref=self.row['terminal'];self.model_exit_ref=self.recorder_terminal['model_exit']
        model_row={'terminal':self.model_exit_ref,'tool':self.result['actual_tool'],'owners':[self.result['owner']],
                   'units':[],'source_sha256':{}}
        catalog={'schema':'minicpm5_operator_later_serial_metadata_catalog_v1','reviewer':'/root','passed':True,
                 'blockers':[],'completed_sdk':{'synthetic':'original-sdk'},'current_handoff':self.ref('handoff'),
                 'all_later_serial_runs_included':True,'runs':[model_row,self.recorder_row]}
        cat_ref=self.ref('audit-catalog');self.objects[cat_ref['path']]=catalog
        self.audit_catalog=catalog
        sr=self.ref('audit-spec');rr=self.ref('audit-review');out=self.ref('audit-result')
        owner=copy.deepcopy(self.owner);owner['pid']=31;owner['start_ticks']='311'
        owner['argv']=['python3','-B',AUD,'--spec',sr['path'],'--spec-sha256',sr['sha256'],
                       '--review',rr['path'],'--review-sha256',rr['sha256']]
        self.spec={'schema':'minicpm5_native_post_exit_audit_spec_v1','model_exit':self.model_exit_ref,
                   'recorder_exit':self.recorder_ref,'current_serial_catalog':cat_ref,
                   'source_extension':copy.deepcopy(self.pins),'output':out['path'],'limits':self.scope['LIMITS']}
        self.review={'schema':'minicpm5_native_post_exit_audit_source_review_v1','reviewer':'/root','passed':True,
                     'blockers':[],'spec':sr,'source_extension':self.spec['source_extension']}
        _,proof=self.produce_binding(self.plan,self.spec['source_extension'],self.model_spec_ref,
                                     self.model_review_ref,self.overlay_ref,AUD)
        verified={'verified':True,'recorder_spec':self.recorder_result['recorder_spec'],
                  'recorder_review':self.recorder_result['recorder_review'],'producer':REC,
                  'source_binding_sha256':digest(self.recorder_result['source_binding']),
                  'complete_recorder_sources':len(self.plan['source_sha256'])+len(self.pins),
                  'complete_inherited_plan_reconstructed':True,'all_recorded_sources_covered_by_auditor':True}
        self.result={'schema':'minicpm5_native_post_exit_independent_audit_v1','complete':True,'reviewer':'/root',
                     'auditor_owner':owner,'audit_spec':sr,'audit_review':rr,'lock_identity':self.r.ORIGINAL_LOCK,
                     'actual_auditor_tool_exit_verified':False,'self_exit_claimed':False,'full_model_qualified':False,
                     'new_speedup_claim':False,'source':{'path':AUD,'sha256':'9'*64},'source_binding':proof,
                     'recorder_source_binding_verified':verified,'model_exit':self.model_exit_ref,'recorder_exit':self.recorder_ref}
        tool={'kind':'tool_session','session_id':27,'terminal_chunk':'abc123','exit_code':0}
        self.terminal={'schema':'minicpm5_native_independent_audit_actual_exit_v1','reviewer':'/root','audit_spec':sr,
                       'audit_review':rr,'result':out,'owner':owner,'actual_tool':tool,'actual_tool_exit_observed':True,
                       'exact_owner_gone':True,'lock_identity':self.r.ORIGINAL_LOCK,'self_exit_claimed':False,
                       'full_model_qualified':False,'new_speedup_claim':False}
        tr=self.ref('audit-terminal')
        self.row={'terminal':tr,'tool':tool,'owners':[owner],'units':[],
                  'source_sha256':{REC:'f'*64,AUD:'9'*64,CORE4:'b'*64,CORE5:'c'*64}}
        self.objects.update({sr['path']:self.spec,rr['path']:self.review,out['path']:self.result,tr['path']:self.terminal})
        return self


class InheritedControls(unittest.TestCase):
    def test_recorder_full_reconstruction_and_producer_receipt_agree(self):
        for code in (0,1,137,-9):
            f=Fixture(code);before=copy.deepcopy(f.result);result=f.consume()
            self.assertEqual(f.calls,['complete','resolve']);self.assertGreater(f.guard_calls,0)
            self.assertEqual(result,{REC:'f'*64,AUD:'9'*64,CORE4:'b'*64,CORE5:'c'*64})
            self.assertEqual(f.result,before);self.assertEqual(f.result['actual_exit_code'],code)

    def test_inherited_entries_allow_empty_extension_without_duplicate_pins(self):
        f=Fixture(inherited_entries=True);result=f.consume()
        self.assertEqual(f.pins,{});self.assertEqual(len(result),4)
        self.assertEqual(f.result['source_binding']['complete_source_files'],7)

    def test_audit_reconstructs_recorder_catalog_join_and_complete_coverage(self):
        f=Fixture().as_auditor();before=copy.deepcopy(f.audit_catalog);result=f.consume()
        self.assertEqual(f.calls,['complete','resolve'])
        self.assertEqual(f.audit_catalog,before);self.assertEqual(result[AUD],'9'*64)
        for ref in (f.recorder_ref,f.model_exit_ref,f.recorder_result['recorder_spec'],f.recorder_result['recorder_review']):
            self.assertIn(ref['path'],f.reads)

    def test_no_guard_or_unknown_producer_cannot_route_to_new_branch(self):
        f=Fixture()
        with self.assertRaisesRegex(RuntimeError,'actual preparation guard'):
            f.scope['executed_extension'](f.terminal,f.row,f.read)
        f.result['recorder_source']['path']=str(Q/'record_minicpm5_layer1_operator_native_model_exit_v99.py')
        with self.assertRaisesRegex(RuntimeError,'Only exact original'):f.consume()

    def test_tampered_binding_fields_are_rejected_even_with_recomputed_outer_hash(self):
        for field in ('inherited_plan_sha256','inherited_source_map_sha256','complete_source_map_sha256',
                      'extension_source_map_sha256','inherited_source_files','complete_source_files',
                      'extension_source_files','producer','complete_source_reconstruction_required'):
            f=Fixture();proof=f.result['source_binding']
            proof[field]=not proof[field]if type(proof[field])is bool else 999 if type(proof[field])is int else'f'*64
            with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,'independently recomputed'):f.consume()

    def test_plan_review_overlay_or_inherited_data_substitution_rejected(self):
        for mode in ('review_digest','result_digest','review_pass','review_overlay','runner','base_removed','base_input_removed','overlay','proof_ref'):
            f=Fixture()
            if mode=='review_digest':f.model_review['resolved_plan_sha256']='0'*64
            elif mode=='result_digest':f.result['resolved_plan_sha256']='0'*64
            elif mode=='review_pass':f.model_review['passed']=False
            elif mode=='review_overlay':f.model_review['overlay']=f.ref('other-overlay')
            elif mode=='runner':f.model_spec['runner']['sha256']='0'*64
            elif mode=='base_removed':f.base['source_sha256'].pop(str(ROOT/'package-a.py'))
            elif mode=='base_input_removed':f.base['inputs'].pop('first')
            elif mode=='overlay':f.objects[f.overlay_ref['path']]={'replaced':True}
            else:f.result['source_binding']['model_review']=f.ref('other-review')
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):f.consume()

    def test_only_new_local_extension_and_exact_catalog_map(self):
        for mode in ('overlap','external','future_auditor','unbound_row','missing_row_pin','entry_hash','review_map','cap'):
            f=Fixture()
            if mode=='overlap':f.pins[CORE4]='b'*64
            elif mode=='external':f.pins['/tmp/not-local.py']='b'*64
            elif mode=='future_auditor':f.pins[str(Q/'audit_minicpm5_layer1_operator_native_result_v99.py')]='b'*64
            elif mode=='unbound_row':f.row['source_sha256'][str(ROOT/'unbound.py')]='a'*64
            elif mode=='missing_row_pin':del f.row['source_sha256'][REC]
            elif mode=='entry_hash':f.result['recorder_source']['sha256']='e'*64
            elif mode=='review_map':f.review['source_extension']={}
            else:f.pins.update({str(ROOT/('extra-'+str(n)+'.py')):'a'*64 for n in range(2048)})
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):f.consume()

    def test_true_exit_identity_and_command_are_required(self):
        for mode in ('bool_tool','tool_mismatch','owner_missing','owner_live','self_exit','model_bool','model_code','model_schema','model_lock','argv','review_sha'):
            f=Fixture()
            if mode=='bool_tool':f.row['tool']['exit_code']=False
            elif mode=='tool_mismatch':f.terminal['actual_tool']=dict(f.row['tool'],session_id=99)
            elif mode=='owner_missing':f.row['owners']=[]
            elif mode=='owner_live':f.terminal['owner_gone']=False
            elif mode=='self_exit':f.terminal['self_exit_claimed']=True
            elif mode=='model_bool':f.result['actual_exit_code']=False
            elif mode=='model_code':f.result['actual_exit_code']=1
            elif mode=='model_schema':f.result['schema']='minicpm5_native_model_failed_actual_exit_v1'
            elif mode=='model_lock':f.result['lock_identity']={}
            elif mode=='argv':f.owner['argv'][2]=AUD
            else:f.owner['argv'][-1]='f'*64
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):f.consume()

    def test_audit_requires_actual_separate_recorder_and_successful_model(self):
        for mode in ('failed_model','failed_recorder','same_session','recorder_dropped','model_dropped','recorder_owner','recorder_binding','recorder_review','verified_false','rebound_terminal'):
            f=Fixture(1 if mode=='failed_model'else 0).as_auditor()
            if mode=='failed_recorder':f.recorder_terminal['actual_tool']['exit_code']=1
            elif mode=='same_session':f.recorder_terminal['actual_tool']['session_id']=f.recorder_result['actual_tool']['session_id']
            elif mode=='recorder_dropped':f.audit_catalog['runs'].pop()
            elif mode=='model_dropped':f.audit_catalog['runs'].pop(0)
            elif mode=='recorder_owner':f.recorder_terminal['owner']=dict(f.recorder_terminal['owner'],pid=999)
            elif mode=='recorder_binding':f.recorder_result['source_binding']['complete_source_files']=99
            elif mode=='recorder_review':f.objects[f.recorder_result['recorder_review']['path']]['passed']=False
            elif mode=='verified_false':f.result['recorder_source_binding_verified']['verified']=False
            elif mode=='rebound_terminal':f.recorder_terminal['model_exit']=f.ref('another-model-exit')
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):f.consume()

    def test_original_model_predecessors_cannot_be_dropped_from_auditor_catalog(self):
        f=Fixture().as_auditor();f.prior_rows=[dict(f.recorder_row,terminal=f.ref('older-terminal'))]
        with self.assertRaisesRegex(RuntimeError,'full original model predecessor'):f.consume()

    def test_complete_recorder_source_coverage_required_even_outside_entry_chain(self):
        f=Fixture().as_auditor();extra=str(ROOT/'actual-recorder-extra.py')
        f.recorder_spec['source_extension'][extra]='a'*64
        f.recorder_row['source_sha256'][extra]='a'*64
        _,f.recorder_result['source_binding']=f.produce_binding(f.plan,f.recorder_spec['source_extension'],
            f.model_spec_ref,f.model_review_ref,f.overlay_ref,REC)
        with self.assertRaises(RuntimeError):f.consume()

    def test_per_traversal_cache_cannot_mask_changed_inputs(self):
        f=Fixture();cache={};f.consume(cache);f.consume(cache)
        self.assertEqual(f.calls,['complete','resolve'])
        f.model_review['resolved_plan_sha256']='0'*64
        with self.assertRaisesRegex(RuntimeError,'reconstructed original model'):f.consume(cache)

    def test_boolean_zero_inherited_binding_counter_is_not_integer_zero(self):
        f=Fixture(inherited_entries=True);f.result['source_binding']['extension_source_files']=False
        with self.assertRaisesRegex(RuntimeError,'independently recomputed'):f.consume()

    def test_catalog_requires_inherited_entry_and_core_witnesses(self):
        for key in (REC,AUD,CORE4,CORE5):
            f=Fixture(inherited_entries=True);del f.row['source_sha256'][key]
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'entry/core witnesses'):f.consume()

    def test_old_catalog256_paths_and_required_core_chain_still_validate(self):
        for is_auditor in (False,True):
            f=LegacyFixture(auditor=is_auditor)
            replacements={str(Q/'record_minicpm5_layer1_operator_native_model_exit_v1.py'):str(Q/'record_minicpm5_layer1_operator_native_model_exit_v2.py'),
                          str(Q/'audit_minicpm5_layer1_operator_native_result_v2.py'):str(Q/'audit_minicpm5_layer1_operator_native_result_v3.py'),
                          str(Q/'minicpm5_layer1_operator_native_post_exit_core_v2.py'):CORE4,
                          str(Q/'minicpm5_layer1_operator_native_post_exit_core_v3.py'):CORE4}
            f.pins.clear();f.pins.update({replacements.get(f.script,f.script):'a'*64,
                str(Q/'audit_minicpm5_layer1_operator_native_result_v3.py'):'a'*64,CORE4:'b'*64,CORE5:'c'*64})
            script=replacements.get(f.script,f.script);f.owner['argv'][2]=script
            key='source'if is_auditor else'recorder_source';f.result[key]={'path':script,'sha256':'a'*64}
            f.row['source_sha256']=dict(f.pins)
            self.assertEqual(f.scope['executed_extension'](f.terminal,f.row,f.read),f.pins)

    def test_new_history_authenticates_idle_tip_before_complete_recovery_and_again_after(self):
        f=Fixture();events=[]
        recovery_terminal=f.ref('fixed-recovery-terminal');recovery_spec=f.ref('fixed-recovery-spec');base_ref=f.ref('fixed-original-plan')
        f.r.completion.BINDING={'actual_exit':recovery_terminal}
        f.recovered['binding']=f.r.completion.BINDING
        f.objects[recovery_terminal['path']]={'spec':recovery_spec}
        f.objects[recovery_spec['path']]={'original_plan':base_ref,'source_extension':{}}
        f.objects[base_ref['path']]={**f.base,'latest_completion':{'original':'latest'}}
        original_completed=f.r.completion.completed
        def complete(*args):events.append('complete');return original_completed(*args)
        def fresh(view,read,guard):
            events.append('fresh');self.assertEqual(view['serial_predecessors'],f.serial_ref)
            self.assertTrue(all(view['source_sha256'][p]==s for p,s in f.row['source_sha256'].items()))
            return {'same_idle_tip':True}
        f.r.completion.completed=complete;f.r.completion.latest.require_latest_terminal=fresh
        result=f.scope['historical_auditors'](f.serial_ref,f.serial,f.read,f.guard)
        self.assertEqual(events,['fresh','complete','fresh']);self.assertEqual(result[AUD],'9'*64)
        events.clear()
        def reject_fresh(*args):events.append('fresh');raise RuntimeError('not idle')
        f.r.completion.latest.require_latest_terminal=reject_fresh
        with self.assertRaisesRegex(RuntimeError,'not idle'):f.scope['historical_auditors'](f.serial_ref,f.serial,f.read,f.guard)
        self.assertEqual(events,['fresh'])


class SourcePreservation(unittest.TestCase):
    def test_actual_complete_92_row_catalog_unchanged_through_legacy_consumers(self):
        def assignment(path,name,namespace):
            nodes=[node for node in ast.parse(path.read_bytes()).body if isinstance(node,ast.Assign)
                   and any(isinstance(target,ast.Name)and target.id==name for target in node.targets)]
            self.assertEqual(len(nodes),1)
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
            return namespace[name]
        read_paths=set()
        def read(ref):
            path=Path(ref['path']);self.assertEqual(path.suffix,'.json')
            with path.open('rb')as stream:raw=stream.read(8*2**20+1)
            self.assertLessEqual(len(raw),8*2**20)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),ref['sha256'])
            read_paths.add(str(path));return json.loads(raw)
        ref={'path':str(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261111-v50.json'),
             'sha256':'55db2de4ecd4232c3d7e232d5d34ef148a6df785b6a5a9bbf5fe3f4b3de5bc7a'}
        serial=read(ref);before=copy.deepcopy(serial);self.assertEqual(len(serial['runs']),92)
        f=LegacyFixture();scope=f.r.completion.latest.serial_catalog.__globals__
        scope['MAX_SERIAL_RUNS']=256
        scope['sdk'].BINDING=assignment(Q/'minicpm5_layer1_operator_sdk_completion_v2.py','BINDING',{'Q':Q})
        definitions(Q/'minicpm5_layer1_operator_sdk_runtime_completion_v4.py',{'serial_catalog'},scope)
        f.r.completion.latest.serial_catalog=scope['serial_catalog']
        f.scope['LIMITS']=assignment(Q/'audit_minicpm5_layer1_operator_sdk_result_v3.py','LIMITS',{})
        result=f.scope['historical_auditors'](ref,serial,read,lambda:None)
        self.assertEqual(serial,before)
        self.assertTrue({str(Q/'audit_minicpm5_layer1_operator_native_result_v2.py'),
                         str(Q/'audit_minicpm5_layer1_operator_native_result_v3.py'),CORE4,CORE5}<=set(result))
        self.assertTrue({row['terminal']['path']for row in serial['runs']}<=read_paths)

    def test_original_legacy_validator_exactly_preserved_and_all_other_bodies_unchanged(self):
        old=ast.parse(OLD.read_bytes());new=ast.parse(PREP.read_bytes())
        old_nodes={n.name:n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        new_nodes={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        legacy=copy.deepcopy(new_nodes['legacy_executed_extension']);legacy.name='executed_extension'
        self.assertEqual(ast.dump(legacy,include_attributes=False),ast.dump(old_nodes['executed_extension'],include_attributes=False))
        for name,node in old_nodes.items():
            if name in ('executed_extension','historical_auditors'):continue
            self.assertEqual(ast.dump(node,include_attributes=False),ast.dump(new_nodes[name],include_attributes=False),name)
        old_assign=[ast.dump(n,include_attributes=False)for n in old.body if isinstance(n,ast.Assign)]
        new_assign=[ast.dump(n,include_attributes=False)for n in new.body if isinstance(n,ast.Assign)]
        self.assertEqual(old_assign,new_assign)

    def test_actual_historical_consumer_passes_guard_and_local_cache(self):
        tree=ast.parse(PREP.read_bytes())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='historical_auditors')
        calls=[n for n in ast.walk(fn)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='executed_extension']
        self.assertEqual(len(calls),1);self.assertEqual([ast.unparse(x)for x in calls[0].args],['terminal','row','read','guard','reconstructed'])
        new_names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertTrue({'inherited_source_plan','inherited_source_binding','completed_inherited_extension'}<=new_names)
        self.assertFalse(any(isinstance(n,ast.ImportFrom)and any(a.name=='audit_minicpm5_layer1_operator_native_result_v4'for a in n.names)for n in ast.walk(tree)))


if __name__=='__main__':unittest.main(verbosity=2)
