"""Native context_v3/runner_v9 integration: source definitions and small mocks."""
import ast
from contextlib import ExitStack, redirect_stdout
import copy
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'research')]
from research import check_minicpm5_native_source_phase_seal_v1 as memory
from research import minicpm5_native_source_phase_seal_v3 as phase_v2

sources=memory.sources
sources.metadata.LIMIT=8*2**20
sources.metadata.query_sources.NATIVE_BYTES=3920408
memory.scope.update(AUTH_SCHEMA='minicpm5_layer1_sdk_batch_layout_source_authentication_v2',
    re=re,Path=Path,identity=lambda path:None)
memory.definitions(ROOT/'research/minicpm5_layer1_operator_sdk_sources_v1.py',
    {'native_identities','validate_native_receipt','check_native_identities','reference'},memory.scope)
for name in ('native_identities','validate_native_receipt','check_native_identities','reference','AUTH_SCHEMA'):
    setattr(sources,name,memory.scope[name])


def isolated_contexts():
    """Run unchanged class definitions without historical import-time JSON IO."""
    first=ROOT/'minicpm5_layer1_operator_native_runtime_context_v1.py'
    scope=dict(copy=copy,Path=Path,OriginalContext=memory.original.RuntimeContext,
        identity=lambda path:None,digest=memory.original.digest,sources=sources,require=memory.seal.require)
    memory.definitions(first,set(),scope)
    tree=ast.parse(first.read_bytes())
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(first),'exec'),scope)
    second=ROOT/'minicpm5_layer1_operator_native_runtime_context_v3.py'
    other=dict(copy=copy,Path=Path,PreviousContext=scope['RuntimeContext'],sources=sources,
        identity=lambda path:None,SourcePhase=phase_v2.SourcePhase,require=memory.seal.require,_COMPLETE=object())
    tree=ast.parse(second.read_bytes());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(second),'exec'),other)
    return scope,other


class Harness(memory.Harness):
    def __init__(self):
        super().__init__(4)
        self.previous,self.current=isolated_contexts()
        self.plan.update(sdk_operator_recovery={'schema':'synthetic-recovered','failed_exit':1},
            latest_completion={'schema':'synthetic-complete','rows':[{'pid':19,'start_ticks':'20'}]},
            layer1_operator_boundary={'schema':'synthetic-sdk-boundary'},
            serial_predecessors={'path':'/synthetic/catalog.json','sha256':'a'*64})
        self.refs=dict(admission={'path':self.plan['output_root']+'/admission.json','sha256':'b'*64},
            authenticated={'path':self.plan['output_root']+'/source-authentication.json','sha256':'c'*64},
            overlay={'path':'/synthetic/overlay.json','sha256':'d'*64})
        self.permit.update(runtime_admission=self.refs['admission'],source_authentication=self.refs['authenticated'],
            runtime_plan=self.refs['overlay'],sdk_operator_recovery=self.plan['sdk_operator_recovery'],
            outer_owner={'pid':10,'start_ticks':'11'},lock_identity={'device':1,'inode':2})
        # The actual new runner and helpers must be additive original source pins.
        for relative in ('minicpm5_layer1_operator_native_runner_v9.py',
                         'minicpm5_layer1_operator_native_runtime_context_v3.py',
                         'research/minicpm5_native_source_phase_seal_v3.py',
                         'research/minicpm5_native_source_phase_seal_v1.py'):
            path=str(ROOT/relative);self.raw[path]=(ROOT/relative).read_bytes()
        self.raw[self.plan['source_contract']['path']]=b'{}'
        self.plan['source_contract']['sha256']=hashlib.sha256(b'{}').hexdigest()
        self.partition['source_contract']=copy.deepcopy(self.plan['source_contract'])
        self.partition['categories']['text_sha256']={path:hashlib.sha256(raw).hexdigest() for path,raw in self.raw.items()}
        self.plan['source_sha256']={p:s for part in self.partition['categories'].values()for p,s in part.items()}
        self.partition['source_map_sha256']=memory.original.digest(self.plan['source_sha256'])
        self.partition['partition_sha256']=memory.original.digest(self.partition['categories'])
        self.resolved['plan_sha256']=memory.original.digest(self.plan)
        self.permit['plan_sha256']=self.resolved['plan_sha256']
        runner=str(ROOT/'minicpm5_layer1_operator_native_runner_v9.py')
        self.contract=memory.seal.life.build_contract(self.resolved,self.refs['overlay'],
            dict(path=runner,sha256=self.plan['source_sha256'][runner]),'trn1-mini-layer1-operator-native-source-seal')
        self.ports.contract=self.contract
        owners=[dict(pid=21,start_ticks='200',argv=['synthetic'],cgroup='0::/old')]
        leaf=dict(owners=owners,history={'rows':[{'unit':'old','samples':[{'properties':{'MainPID':'0'},'group':{'absent':True}}]}]},
            source_sha256=copy.deepcopy(self.plan['source_sha256']))
        # Include the actual overlay return classes omitted by the old tiny
        # fixture: full historical branch, current_scope, native_bundle and view.
        self.resolved.update(current_scope=dict(leaf,_original_done=copy.deepcopy(leaf),
            native={'plan':copy.deepcopy(self.plan),'native':{'records':[{'metadata':[1,2,None]}]}},
            later_serial={'runs':[{'owners':copy.deepcopy(owners),'source_sha256':copy.deepcopy(self.plan['source_sha256'])}]}),
            native_bundle={'plan':copy.deepcopy(self.plan),'records':{'native':{'rows':[{'frame':[1,2,3]}]}},'owners':owners},
            current_serial_view={'latest_completion':self.plan['latest_completion'],'serial_predecessors':self.plan['serial_predecessors'],
                'source_sha256':copy.deepcopy(self.plan['source_sha256'])})
        self.identities={}
        for path,pin in self.partition['query_observation']['source_and_binary_pins'].items():
            if path in self.partition['categories']['query_native_binary_sha256']:
                self.identities[path]=dict(realpath=path,device=pin['device'],inode=pin['inode'],size=pin['bytes'],mtime_ns=1,ctime_ns=1)
        self.auth=dict(schema=sources.AUTH_SCHEMA,source_contract=copy.deepcopy(self.partition['source_contract']),
            source_map_sha256=self.partition['source_map_sha256'],partition_sha256=self.partition['partition_sha256'],
            query_observation=self.partition['query_observation_ref'],native_sha256=self.partition['categories']['query_native_binary_sha256'],
            native_identities=copy.deepcopy(self.identities),native_authentication=dict(complete=True,files=9,bytes=3920408,
                native_libraries_loaded_here=False,model_or_tensor_bytes_read=0,full_runtime_environment_verified=False),
            outer_owner=self.permit['outer_owner'],lock_identity=self.permit['lock_identity'],attempt_id=self.permit['attempt_id'],
            historical_payload_bytes_read=0,native_libraries_loaded_here=False,full_model_qualified=False,new_speedup_claim=False)
        self.admitted=dict(schema='minicpm5_layer1_operator_native_admission_v1',source_contract=self.plan['source_contract'],
            overlay=self.refs['overlay'],resolved_plan_sha256=self.resolved['plan_sha256'],
            sdk_operator_recovery=self.plan['sdk_operator_recovery'],outer_owner=self.permit['outer_owner'],
            lock_identity=self.permit['lock_identity'],attempt_id=self.permit['attempt_id'],
            latest_completion=self.plan['latest_completion'],completed_sdk_layout=self.plan['layer1_operator_boundary'],
            serial_predecessors=self.plan['serial_predecessors'],source_authentication=self.refs['authenticated'],
            runtime_admitted=True,predecessors_checked=True)
        self.values={self.refs['admission']['path']:self.admitted,self.refs['authenticated']['path']:self.auth,
            self.plan['source_contract']['path']:{'synthetic':True}}
        for path,value in self.values.items():
            self.identities[path]=dict(realpath=path,device=1,inode=50+len(self.identities),
                size=len(json.dumps(value)),mtime_ns=1,ctime_ns=1)
        self.loaded=[];self.context_value=None;self.phase_refs=[]

    def patches(self):
        stack=super().patches()
        def load(ref):
            self.loaded.append(ref['path'])
            if ref['path'] in (self.refs['admission']['path'],self.refs['authenticated']['path']):
                assert type(self.context_value.plan)is dict and not self.context_value._sources_checked
            return copy.deepcopy(self.values[ref['path']])
        def classify(pins,ref,read):
            assert type(pins)is dict and pins==self.plan['source_sha256']
            self.events.append(('classify',))
            read(ref)
            return copy.deepcopy(self.partition)
        stack.enter_context(patch.object(sources,'load',side_effect=load,create=True))
        stack.enter_context(patch.object(sources,'classify',side_effect=classify))
        stack.enter_context(patch.dict(memory.scope,identity=lambda path:copy.deepcopy(self.identities[path])))
        stack.enter_context(patch.dict(self.previous,identity=lambda path:copy.deepcopy(self.identities[path])))
        stack.enter_context(patch.dict(self.current,identity=lambda path:copy.deepcopy(self.identities[path])))
        def construct(*args):
            value=phase_v2.SourcePhase(*args);self.phase_refs.append(weakref.ref(value));return value
        stack.enter_context(patch.dict(self.current,SourcePhase=construct))
        return stack

    def context(self):
        self.context_value=self.current['RuntimeContext'](self.plan,self.resolved['plan_sha256'],self.permit,self.old_guard,
            contract=self.contract,resolved=self.resolved,lease=self.lease,ports=self.ports)
        return self.context_value


class Tests(unittest.TestCase):
    def test_complete_integrated_verify_and_plain_receipts_then_permission(self):
        h=Harness()
        with h.patches():
            context=h.context();self.assertIsNot(context.plan,h.plan)
            before=copy.deepcopy(context.plan)
            context.verify_sources()
            self.assertTrue(context._sources_checked)
            self.assertIs(type(context.plan),dict);self.assertEqual(context.plan,before)
            self.assertEqual(context._inherited_native_receipt,h.auth)
            self.assertEqual(set(context._source_runtime_identities),{h.refs['admission']['path'],h.refs['authenticated']['path']})
            context.guard('native')
        self.assertEqual(sum(e[0]=='classify'for e in h.events),1)
        self.assertEqual([e[1]for e in h.events if e[0]=='source'],list(h.raw))
        self.assertEqual(sum(e[0]=='python-bytes'for e in h.events),3)
        self.assertEqual(h.contract['runner']['path'],str(ROOT/'minicpm5_layer1_operator_native_runner_v9.py'))

    def test_normal_exit_releases_phase_and_full_snapshots_without_gc(self):
        h=Harness();enabled=gc.isenabled();gc.disable()
        try:
            with h.patches():
                context=h.context();context.verify_sources()
                self.assertTrue(context._sources_checked)
                self.assertEqual(len(h.phase_refs),1)
                self.assertIsNone(h.phase_refs[0]())
        finally:
            if enabled:gc.enable()

    def test_classifier_source_receipt_identity_errors_latch_every_entry(self):
        for fault in ('classify','source','admission','receipt','native_identity'):
            h=Harness()
            with h.patches(),self.subTest(fault=fault):
                context=h.context()
                if fault=='source':h.raw[next(iter(h.raw))]=b'bad'
                elif fault=='admission':h.admitted['resolved_plan_sha256']='0'*64
                elif fault=='receipt':h.auth['native_authentication']['complete']=False
                elif fault=='native_identity':h.identities[next(iter(h.auth['native_identities']))]['mtime_ns']=3
                with ExitStack()as stack:
                    if fault=='classify':stack.enter_context(patch.object(sources,'classify',side_effect=ValueError('classifier rejected')))
                    with self.assertRaises((RuntimeError,ValueError))as first:context.verify_sources()
                self.assertFalse(context._sources_checked)
                for operation in (context.verify_sources,context.guard,context.external_guard,
                                  lambda:context.verify_input('/synthetic/model')):
                    with self.assertRaises(type(first.exception))as again:operation()
                    self.assertIs(again.exception,first.exception)

    def test_early_return_repeated_verification_and_partial_phase_cannot_enable_model(self):
        for fault in ('early_return','partial_phase','repeated'):
            h=Harness()
            with h.patches(),self.subTest(fault=fault):
                context=h.context()
                if fault=='repeated':context.verify_sources();self.assertTrue(context._sources_checked)
                with ExitStack()as stack:
                    if fault=='early_return':stack.enter_context(patch.object(context,'_verify_sources_once',return_value=None))
                    if fault=='partial_phase':stack.enter_context(patch.object(phase_v2.SourcePhase,'verify_text_sources',return_value=None))
                    with self.assertRaises(RuntimeError):context.verify_sources()
                self.assertFalse(context._sources_checked)
                with self.assertRaises(RuntimeError):context.verify_input('/synthetic/model')

    def test_rich_history_unknown_types_and_old_alias_changes_rejected(self):
        for fault in ('set','tuple','path','mutation'):
            h=Harness()
            with h.patches(),self.subTest(fault=fault):
                context=h.context()
                if fault!='mutation':
                    h.resolved['current_scope']['history']['unsupported']={'set':{1},'tuple':(1,),'path':Path('/synthetic')}[fault]
                else:
                    text=sources.text
                    def changed(path,wanted=None):
                        result=text(path,wanted)
                        h.resolved['current_scope']['history']['rows'][0]['unit']='changed'
                        return result
                with ExitStack()as stack:
                    if fault=='mutation':stack.enter_context(patch.object(sources,'text',side_effect=changed))
                    with self.assertRaises(RuntimeError):context.verify_sources()
                self.assertFalse(context._sources_checked)
                with self.assertRaises(RuntimeError):context.guard()

    def test_original_classifier_and_full_receipt_tail_ast_unchanged(self):
        def method(path,name):
            tree=ast.parse(path.read_bytes());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef))
            return next(n for n in klass.body if isinstance(n,ast.FunctionDef)and n.name==name)
        old=method(ROOT/'minicpm5_layer1_operator_native_runtime_context_v1.py','verify_sources')
        new=method(ROOT/'minicpm5_layer1_operator_native_runtime_context_v3.py','_verify_sources_once')
        old_head=next(i for i,n in enumerate(old.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
                      and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=='verify_text_sources')
        new_head=next(i for i,n in enumerate(new.body)if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Tuple))
        self.assertEqual([ast.dump(n)for n in old.body[:old_head]],[ast.dump(n)for n in new.body[:new_head]])
        # Original final _sources_checked assignment is intentionally moved to
        # the outer completion-token gate. All receipt/identity nodes remain.
        self.assertEqual([ast.dump(n)for n in old.body[old_head+1:-1]],
                         [ast.dump(n)for n in new.body[new_head+2:-1]])

    def test_runner_only_changes_context_wiring_and_six_bounded_logs(self):
        old=ast.parse((ROOT/'minicpm5_layer1_operator_native_runner_v7.py').read_bytes())
        new=ast.parse((ROOT/'minicpm5_layer1_operator_native_runner_v9.py').read_bytes())
        class Normalize(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name)and node.value.func.id=='phase_log':return None
                return self.generic_visit(node)
            def visit_If(self,node):
                self.generic_visit(node)
                return node if node.body else None
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name)and node.func.id=='RuntimeContext':node.keywords=[]
                return node
        a={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
        b={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
        for name in a:self.assertEqual(ast.dump(a[name]),ast.dump(Normalize().visit(b[name])),name)
        scope=dict(json=json,os=os,time=SimpleNamespace(monotonic=lambda:5.0),_PHASE_START=1.0,require=memory.seal.require)
        exec(compile(ast.Module(body=[b['phase_log']],type_ignores=[]),'phase_log','exec'),scope)
        output=io.StringIO()
        with redirect_stdout(output):
            scope['phase_log']('bootstrap.begin',dict(pid=os.getpid(),start_ticks='123'))
        raw=output.getvalue();value=json.loads(raw)
        self.assertLessEqual(len(raw.encode()),513)
        self.assertEqual(set(value),{'phase','elapsed_seconds','pid','start_ticks'})
        self.assertEqual(value['elapsed_seconds'],4.0)
        with self.assertRaises(RuntimeError):scope['phase_log']('made_up',dict(pid=os.getpid(),start_ticks='123'))



class BoundedSnapshotTests(unittest.TestCase):
    def test_stream_digest_exact_original_bytes(self):
        import random
        rng=random.Random(912)
        values=[None,True,False,0,1,-1,-0.0,1e-300,1e300,{'é雪\ud800':['\n\t\"',1.25,None]}, {}, []]
        for _ in range(100):
            values.append({'z':[rng.random(), rng.randrange(-100000,100000), str(rng.random())],
                           'a':{'雪':['x',False,None]}, 'shared':[1,2,3]})
        for value in values:
            self.assertEqual(phase_v2.stream_digest(value),memory.original.digest(value))

    def test_shared_containers_sealed_once_and_cycles_rejected(self):
        leaf={'rows':[1,2,3]}; value={'a':leaf,'b':[leaf,leaf]}
        frozen=phase_v2.seal_json(value)
        self.assertIs(frozen['a'],frozen['b'][0]);self.assertIs(frozen['b'][0],frozen['b'][1])
        leaf['rows'].append(4);self.assertEqual(frozen['a']['rows'],(1,2,3))
        cyclic=[];cyclic.append(cyclic)
        class Custom(dict):pass
        for bad in (cyclic,Custom(),(1,),{1:'x'},float('nan'),float('inf'),set(),object()):
            for func in (phase_v2.seal_json,phase_v2.stream_digest):
                with self.subTest(kind=type(bad).__name__,function=func.__name__),self.assertRaises(RuntimeError):func(bad)

    def test_history_not_snapshotted_and_all_aliases_still_checked(self):
        for alias in ('resolved','context_plan','contract','partition','permit'):
            h=memory.Harness()
            h.resolved['current_scope']={'history':[{'payload':[1,2,3]}]}
            with h.patches(),self.subTest(alias=alias):
                context=h.context()
                with self.assertRaisesRegex(RuntimeError,'Mutable alias changed'):
                    with phase_v2.SourcePhase(h.contract,h.resolved,context,h.partition,h.lease,h.ports,sources)as phase:
                        self.assertEqual(set(phase.snapshots),{'contract','context_plan','partition','catalog'})
                        phase.verify_text_sources()
                        target={'resolved':h.resolved,'context_plan':phase.original_plan,'contract':h.contract,
                                'partition':h.partition,'permit':context.permit}[alias]
                        target['changed']=[1]
                self.assertFalse(context._sources_checked)
                with self.assertRaises(RuntimeError):context.guard()

    def test_stream_peak_does_not_materialize_full_serialized_history(self):
        import tracemalloc
        # A shared history amplifies serialized size without a large source fixture.
        leaf={'row':[{'name':'abcd'*30,'number':n}for n in range(300)]}
        value={'history':[leaf]*12}
        def measured(function):
            tracemalloc.start()
            try:
                result=function(value);peak=tracemalloc.get_traced_memory()[1]
            finally:tracemalloc.stop()
            return result,peak
        old,old_peak=measured(memory.original.digest)
        new,new_peak=measured(phase_v2.stream_digest)
        self.assertEqual(old,new);self.assertLess(new_peak,old_peak/3)
        print(json.dumps({'synthetic_only':True,'original_peak_bytes':old_peak,'stream_peak_bytes':new_peak}),flush=True)

    def test_only_import_wiring_changes_in_context_and_runner(self):
        for old,new,before,after in (
            ('minicpm5_layer1_operator_native_runtime_context_v2.py','minicpm5_layer1_operator_native_runtime_context_v3.py',
             'minicpm5_native_source_phase_seal_v2','minicpm5_native_source_phase_seal_v3'),
            ('minicpm5_layer1_operator_native_runner_v8.py','minicpm5_layer1_operator_native_runner_v9.py',
             'minicpm5_layer1_operator_native_runtime_context_v2','minicpm5_layer1_operator_native_runtime_context_v3')):
            self.assertEqual((ROOT/old).read_text().replace(before,after),(ROOT/new).read_text())

if __name__=='__main__':
    unittest.main(verbosity=2)
