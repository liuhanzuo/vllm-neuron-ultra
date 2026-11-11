"""Source definitions plus small authenticated-reader mocks; no original data IO."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research'


def definitions(path,names,scope):
    tree=ast.parse(path.read_bytes())
    nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))and node.name in names]
    assert {node.name for node in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)


def require(ok,message):
    if not ok:raise RuntimeError(message)


def pin_shape(pins):
    require(type(pins)is dict and len(pins)<=100000,'Bounded synthetic source map')
    for path,sha in pins.items():
        require(type(path)is str and Path(path).is_absolute()and '..'not in Path(path).parts
            and type(sha)is str and re.fullmatch('[a-f0-9]{64}',sha),'Exact synthetic pin')


class Fixture:
    def __init__(self,auditor=False,code=0):
        self.objects={};self.refs={};self.reads=[]
        life=dict(Path=Path,re=re,copy=copy,require=require,PROCESS_KEYS=('pid','start_ticks','ppid','argv','cgroup'))
        definitions(Q/'minicpm5_layer1_operator_native_lifecycle_core_v2.py',{'ref','process'},life)
        sources=SimpleNamespace(reference=life['ref'],metadata=SimpleNamespace(pin_shape=pin_shape))
        physical=dict(re=re,require=require)
        definitions(Q/'minicpm5_sdk_physical_evidence_v3.py',{'unit_name'},physical)
        sdk=SimpleNamespace(BINDING={'synthetic':'original-sdk'})
        serial_scope=dict(copy=copy,Path=Path,re=re,require=require,sources=sources,sdk=sdk,
            physical_evidence=SimpleNamespace(unit_name=physical['unit_name']))
        definitions(Q/'minicpm5_layer1_operator_sdk_runtime_completion_v3.py',{'serial_catalog'},serial_scope)
        self.r=SimpleNamespace(life=SimpleNamespace(ref=life['ref'],process=life['process']),sources=sources,
            completion=SimpleNamespace(latest=SimpleNamespace(serial_catalog=serial_scope['serial_catalog'])),
            ORIGINAL_LOCK={'device':66306,'inode':8294421})
        self.scope=dict(copy=copy,hashlib=hashlib,json=json,Path=Path,re=re,sys=SimpleNamespace(modules={}),
            ROOT=ROOT,Q=Q,r=self.r,require=require,LIMITS={'synthetic':'original-unchanged'},
            SELF=str(Q/'prepare_minicpm5_layer1_operator_native_v2.py'),
            RUNNER=ROOT/'minicpm5_layer1_operator_native_runner_v9.py',
            AUDITOR_NAMES=re.compile(r'(?:audit_minicpm5_layer1_operator_native_result_v[0-9]+|minicpm5_layer1_operator_native_post_exit_core_v[0-9]+)\.py'),
            ACCEPTED_CLOSURE={'path':str(Q/'accepted-synthetic-closure.json'),'sha256':'a'*64})
        definitions(Q/'prepare_minicpm5_layer1_operator_native_v2.py',
            {'audit_policy','executed_extension','historical_auditors','merge','extension','source_additions',
             'loaded_sources','resolve_binding','object_ref','encoded','paths'},self.scope)
        self.script=str(Q/('audit_minicpm5_layer1_operator_native_result_v2.py'if auditor else'record_minicpm5_layer1_operator_native_model_exit_v1.py'))
        self.auditor=str(Q/'audit_minicpm5_layer1_operator_native_result_v2.py')
        self.core=str(Q/('minicpm5_layer1_operator_native_post_exit_core_v2.py'if auditor else'minicpm5_layer1_operator_native_post_exit_core_v3.py'))
        self.pins={self.script:'a'*64,self.auditor:'a'*64,self.core:'b'*64}
        sr=self.ref('executed-spec');rr=self.ref('executed-review');out=self.ref('producer-result')
        self.owner=dict(pid=23,start_ticks='123',ppid=1,cgroup='0::/synthetic',argv=[
            'python3','-B',self.script,'--spec',sr['path'],'--spec-sha256',sr['sha256'],
            '--review',rr['path'],'--review-sha256',rr['sha256']])
        tool=dict(kind='tool_session',session_id=17,terminal_chunk='abcdef',exit_code=code)
        common=dict(reviewer='/root',owner=self.owner,actual_tool=tool,actual_tool_exit_observed=True,
            self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
        self.spec=dict(source_extension=self.pins,output=out['path'],limits=self.scope['LIMITS'])
        if auditor:
            self.spec.update(schema='minicpm5_native_post_exit_audit_spec_v1',model_exit=self.ref('model-exit'),
                recorder_exit=self.ref('recorder-exit'),current_serial_catalog=self.ref('prior-catalog'))
            review_schema='minicpm5_native_post_exit_audit_source_review_v1'
            result=dict(schema='minicpm5_native_post_exit_independent_audit_v1',complete=True,reviewer='/root',
                auditor_owner=self.owner,audit_spec=sr,audit_review=rr,lock_identity=self.r.ORIGINAL_LOCK,
                source=dict(path=self.script,sha256=self.pins[self.script]),model_exit=self.spec['model_exit'],
                recorder_exit=self.spec['recorder_exit'],actual_auditor_tool_exit_verified=False,
                self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
            self.terminal=dict(common,schema='minicpm5_native_independent_audit_actual_exit_v1',
                audit_spec=sr,audit_review=rr,result=out,exact_owner_gone=True,lock_identity=self.r.ORIGINAL_LOCK)
        else:
            self.spec.update(schema='minicpm5_native_model_exit_recorder_spec_v1',live=self.ref('live'),
                model_spec=self.ref('model-spec'),model_review=self.ref('model-review'),actual_tool=dict(tool,session_id=16,exit_code=1))
            review_schema='minicpm5_native_model_exit_recorder_source_review_v1'
            result=dict(schema='minicpm5_native_model_failed_actual_exit_v1',reviewer='/root',recorder=self.owner,
                recorder_spec=sr,recorder_review=rr,recorder_source=dict(path=self.script,sha256=self.pins[self.script]),
                live=self.spec['live'],spec=self.spec['model_spec'],review=self.spec['model_review'],
                actual_tool=self.spec['actual_tool'],self_exit_claimed=False,actual_recorder_tool_exit_verified=False)
            self.terminal=dict(common,schema='minicpm5_native_model_exit_recorder_actual_exit_v1',model_exit=out,
                owner_gone=True,recorder_spec=sr,recorder_review=rr,source_extension=self.pins)
        self.review=dict(schema=review_schema,reviewer='/root',passed=True,blockers=[],spec=sr,source_extension=self.pins)
        self.objects.update({sr['path']:self.spec,rr['path']:self.review,out['path']:result})
        tr=self.ref('terminal');self.objects[tr['path']]=self.terminal
        handoff=self.ref('handoff');self.objects[handoff['path']]=dict(live_tool_sessions=[],active_heavy_units=[])
        self.row=dict(terminal=tr,tool=tool,owners=[self.owner],units=[],source_sha256=dict(self.pins))
        self.serial=dict(schema='minicpm5_operator_later_serial_metadata_catalog_v1',reviewer='/root',passed=True,
            blockers=[],completed_sdk=sdk.BINDING,current_handoff=handoff,all_later_serial_runs_included=True,runs=[self.row])
        self.serial_ref=self.ref('catalog');self.objects[self.serial_ref['path']]=self.serial
        self.result=result

    def ref(self,name):
        path='/synthetic/'+name+'.json'
        self.refs.setdefault(path,dict(path=path,sha256=hashlib.sha256(name.encode()).hexdigest()))
        return self.refs[path]

    def read(self,ref):
        require(self.refs.get(ref['path'])==ref and ref['path']in self.objects,'Injected exact authenticated metadata ref')
        self.reads.append(ref['path']);return copy.deepcopy(self.objects[ref['path']])

    def history(self):return self.scope['historical_auditors'](self.serial_ref,self.serial,self.read,lambda:None)


class Tests(unittest.TestCase):
    def test_completed_recorder_and_true_nonzero_exit_keep_exact_sources(self):
        for code in (0,1,137,255):
            f=Fixture(code=code);before=copy.deepcopy(f.serial)
            self.assertEqual(f.history(),{f.auditor:'a'*64,f.core:'b'*64})
            self.assertEqual(f.serial,before);self.assertIn(f.ref('executed-review')['path'],f.reads)
        f=Fixture(code=1)
        tool=dict(kind='tool_call',terminal_chunk='fedcba',exit_code=1)
        f.row['tool']=tool;f.terminal['actual_tool']=tool
        self.assertEqual(f.history(),{f.auditor:'a'*64,f.core:'b'*64})
        tool['terminal_chunk']='abc'
        with self.assertRaises(RuntimeError):f.history()

    def test_completed_auditor_exact_contract_and_unchanged_failed_sdk_outcome(self):
        f=Fixture(auditor=True);self.assertEqual(f.history(),{f.auditor:'a'*64,f.core:'b'*64})
        for bad in (dict(f.terminal,extra=True),dict(f.terminal,exact_owner_gone=False)):
            with self.assertRaises(RuntimeError):f.scope['executed_extension'](bad,f.row,f.read)

    def test_future_unwitnessed_and_changed_auditor_pins_are_rejected(self):
        for mode in ('future','future_in_spec','changed','unwitnessed'):
            f=Fixture()
            if mode=='future':f.row['source_sha256'][str(Q/'audit_minicpm5_layer1_operator_native_result_v99.py')]='c'*64
            elif mode=='future_in_spec':
                path=str(Q/'audit_minicpm5_layer1_operator_native_result_v99.py')
                f.pins[path]='c'*64;f.row['source_sha256'][path]='c'*64
            elif mode=='changed':f.row['source_sha256'][f.core]='c'*64
            else:f.terminal['schema']='unexecuted-source-review'
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):f.history()

    def test_incomplete_duplicate_bad_tool_owner_and_negative_exit_rejected(self):
        for mode in ('incomplete','duplicate','tool','owner','negative','bool','live'):
            f=Fixture()
            if mode=='incomplete':f.serial['all_later_serial_runs_included']=False
            elif mode=='duplicate':f.serial['runs'].append(copy.deepcopy(f.row))
            elif mode=='tool':f.terminal['actual_tool']=dict(f.row['tool'],session_id=99)
            elif mode=='owner':f.row['owners']=[]
            elif mode=='negative':f.row['tool']['exit_code']=-9
            elif mode=='bool':f.row['tool']['exit_code']=False
            else:f.objects[f.serial['current_handoff']['path']]['live_tool_sessions']=[17]
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):f.history()

    def test_result_spec_review_and_claim_only_sources_cannot_substitute(self):
        for mode in ('missing_result','result_owner','result_spec','review','spec_output','claimed','argv','audit_failure'):
            f=Fixture(auditor=mode=='audit_failure')
            if mode=='missing_result':del f.objects[f.ref('producer-result')['path']]
            elif mode=='result_owner':f.result['recorder']=dict(f.owner,pid=999)
            elif mode=='result_spec':f.result['recorder_spec']=f.ref('other-spec')
            elif mode=='review':f.review['source_extension']={}
            elif mode=='spec_output':f.spec['output']='/wrong.json'
            elif mode=='claimed':f.terminal['source_extension']={}
            elif mode=='argv':f.owner['argv'][4]='/wrong-spec.json'
            else:f.row['tool']['exit_code']=1
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):f.history()

    def test_full_catalog_rows_preserved_with_inherited_history_pins(self):
        f=Fixture();other=f.ref('unrelated-completed-terminal');f.objects[other['path']]={'schema':'another-real-terminal'}
        f.serial['runs'].insert(0,dict(terminal=other,tool=dict(kind='tool_call',terminal_chunk='fedcba',exit_code=1),
            owners=[],units=[],source_sha256=dict(f.pins)))
        before=copy.deepcopy(f.serial);self.assertEqual(f.history(),{f.auditor:'a'*64,f.core:'b'*64})
        self.assertEqual(f.serial,before)

    def test_original_closure_and_all_old_pins_survive_additive_merge(self):
        f=Fixture();history=f.history();old={str(ROOT/('old%d.py'%n)):'d'*64 for n in range(578)}
        closure=dict(source_only=True,original_complete_source_map_must_remain=True,
            dynamic_import_resolution_verified=False,runtime_import_identity_verified=False,
            runtime_graph=dict(source_sha256=old,files=578,native_executed=False))
        additions={str(f.scope['RUNNER']):'e'*64}
        pins=f.scope['source_additions'](closure,f.serial,additions,f.scope['ACCEPTED_CLOSURE'],f.serial_ref,history)
        self.assertTrue(all(pins[p]==s for p,s in old.items()));self.assertEqual(pins[f.core],'b'*64)
        self.assertEqual(f.scope['extension'](old,pins,'source'),{p:s for p,s in pins.items()if p not in old})
        for additions in ({next(iter(old)):'c'*64},{str(Q/'minicpm5_layer1_operator_native_post_exit_core_v99.py'):'a'*64}):
            with self.assertRaises(RuntimeError):f.scope['source_additions'](closure,f.serial,additions,f.scope['ACCEPTED_CLOSURE'],f.serial_ref,history)

    def test_loaded_sources_obey_same_exact_completed_hash_allowance(self):
        f=Fixture();raw=b'synthetic source';sha=hashlib.sha256(raw).hexdigest()
        f.scope['sys'].modules={'x':SimpleNamespace(__file__=f.core)};f.r.sources.text=lambda path:raw
        self.assertEqual(f.scope['loaded_sources'](lambda:None,{f.core:sha}),{f.core:sha})
        for history in ({},{f.core:'a'*64}):
            with self.assertRaises(RuntimeError):f.scope['loaded_sources'](lambda:None,history)

    def test_runner_v9_binding_keeps_original_base_maps(self):
        f=Fixture();base={'source_sha256':{'/synthetic/old.py':'b'*64},'inputs':{'/synthetic/input':{'sha256':'c'*64}}}
        plan=copy.deepcopy(base);plan['source_sha256'][str(f.scope['RUNNER'])]='a'*64
        resolved={'plan':plan};f.r.overlays=SimpleNamespace(resolve=lambda *args:resolved)
        seen=[];f.r.life.build_contract=lambda res,ref,runner,unit:seen.append(runner)or{'unit':unit}
        f.r.LOCK_POLICY={'synthetic':'original'}
        got,contract,spec=f.scope['resolve_binding']({'base_plan':base},{'output_root':'/mnt/nvme/minicpm5-layer1-operator-native-test'},lambda:None,lambda:None)
        self.assertIs(got,resolved);self.assertEqual(spec['runner']['path'],str(f.scope['RUNNER']))
        self.assertEqual(seen,[spec['runner']]);self.assertEqual(plan['inputs'],base['inputs'])

    def test_frozen_preparation_functions_and_guarded_work_are_retained(self):
        def funcs(path):return {n.name:n for n in ast.parse(path.read_bytes()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        old=funcs(Q/'prepare_minicpm5_layer1_operator_native_v1.py');new=funcs(Q/'prepare_minicpm5_layer1_operator_native_v2.py')
        changed={'loaded_sources','source_additions','prepare','bind'}
        for name in set(old)-changed:self.assertEqual(ast.dump(old[name]),ast.dump(new[name]),name)
        # Changed orchestration keeps all previous calls except blanket name rejection,
        # and adds only actual-history authentication / policy arguments.
        class Normalize(ast.NodeTransformer):
            def visit_Assign(self,node):
                if any(isinstance(t,ast.Name)and t.id=='history'for t in node.targets):return None
                return self.generic_visit(node)
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name):
                    call=node.value
                    if call.func.id=='audit_policy':return None
                    if call.func.id=='require'and 'AUDITOR_NAMES'in ast.unparse(call):return None
                return self.generic_visit(node)
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name)and node.func.id in ('loaded_sources','source_additions'):
                    node.args=[a for a in node.args if not(isinstance(a,ast.Name)and a.id=='history')]
                return node
        for name in ('prepare','bind'):
            self.assertEqual(ast.dump(Normalize().visit(old[name])),ast.dump(Normalize().visit(new[name])),name)


if __name__=='__main__':unittest.main(verbosity=2)
