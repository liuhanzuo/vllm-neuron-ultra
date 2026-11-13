"""Independent synthetic reproduction. No real History, lock, proc or payload IO."""
import copy,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
Q=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research');sys.path[:0]=[str(Q),str(Q.parent)]
import minicpm5_retained_replay_finalizer_v2 as o
import minicpm5_retained_replay_caller_v1 as caller
import minicpm5_retained_replay_history_v1 as history
import qwen35_package_preflight_serial_v2 as serial
import test_minicpm5_retained_replay_v1 as fixtures

def probe(mode):
    case=fixtures.ObserverTests();case.setUp();args=case.final_inputs()
    previous,_,observation,live_ref,live,actual,updated_at,_=args
    with tempfile.TemporaryDirectory(prefix='mini-final-guard-source-',dir='/tmp') as temp:
        root=Path(temp);control=root/'control';control.mkdir()
        spec=dict(output_root=str(control),observer_root=str(control/'independent-exit'),units={'pair':'fake-pair','replay':'fake-replay'},terminal_handoff_path=str(root/'complete.json'))
        observation_ref=dict(path=spec['observer_root']+'/observation.json',sha256='b'*64)
        observation.update(spec=dict(path=str(root/'spec.json'),sha256='b'*64),review=dict(path=str(root/'review.json'),sha256='b'*64),
            outer_terminal=dict(path=str(root/'terminal.json'),sha256='b'*64),independent_live=live_ref)
        request_ref=dict(path=str(root/'request.json'),sha256='b'*64)
        request=dict(observation=observation_ref,observer_live=live_ref,actual_observer_tool=actual,updated_at=updated_at)
        values={request_ref['path']:request,observation_ref['path']:observation,live_ref['path']:live,
            observation['admission']['path']:case.admission,observation['outer_terminal']['path']:case.terminal,observation['running']['path']:previous}
        expected=dict(path=observation['running']['path'],identity=observation['running_identity'])
        latest=[copy.deepcopy(expected)];drains=[];writes=[];events=[];metadata_changed=[False];running_changed=[False]
        if mode=='changed_before':latest[0]=dict(path=str(root/'unknown-newer.json'),identity=dict(inode=333))
        class Lock:
            def acquire(self):events.append('lock-acquire')
            def check(self):events.append('lock-check')
            def release(self,proof):assert proof is True;events.append('lock-release')
        class History:
            def __init__(self,*args):pass
            def resolve(self):events.append('full-history-resolve');return self
            def physical_pair(self,*args):events.append('full-history-physical');return {'complete':True}
            def unchanged(self):
                events.append('history-unchanged')
                if metadata_changed[0]:raise ValueError('Inherited metadata identity changed')
        def current():events.append('latest-read');return copy.deepcopy(latest[0])
        def drain(*args):
            drains.append(len(drains)+1);events.append('drain-'+str(len(drains)))
            if mode=='changed_during_last_drain' and len(drains)==3:
                latest[0]=dict(path=str(root/'unknown-newer.json'),identity=dict(inode=333));events.append('newer-handoff-published')
            if mode=='metadata_during_last_drain' and len(drains)==3:
                metadata_changed[0]=True;events.append('inherited-metadata-changed')
            if mode=='running_identity_during_last_drain' and len(drains)==3:
                running_changed[0]=True;events.append('running-identity-changed')
            return dict(physically_drained=True,all_actual_phase_states=[])
        def once(path,value):
            writes.append(dict(path=str(path),value=value,latest=copy.deepcopy(latest[0])));events.append('publish-complete')
            return dict(path=str(path),sha256='c'*64)
        fakeio=SimpleNamespace(read=lambda r,*a:copy.deepcopy(values[r['path']]),identity=lambda p:dict(inode=444)if running_changed[0]else copy.deepcopy(expected['identity']),
            process_api=lambda:SimpleNamespace(gone=lambda owner:None),RetainedLock=Lock,once=once)
        with patch.object(caller,'api',return_value=(fakeio,None)),patch.object(caller,'refs',return_value=spec),\
             patch.object(caller,'latest_observation',side_effect=current),patch.object(caller,'source_unchanged'),\
             patch.object(caller,'storage'),patch.object(o,'verify_self'),patch.object(o,'drain_phases',side_effect=drain),\
             patch.object(history,'History',History),patch.object(history,'Authority',return_value=None),\
             patch.object(serial,'save_history',return_value=[dict(path=str(root/'history.json'),sha256='d'*64)]),\
             patch('fcntl.flock',side_effect=AssertionError('real lock forbidden')),patch('subprocess.Popen',side_effect=AssertionError('process forbidden')):
            error=None
            try:o.publish_completed(request_ref)
            except ValueError as exc:error=str(exc)
        if mode=='unchanged':assert error is None and len(writes)==1
        else:assert error and not writes
        return dict(mode=mode,error=error,published=bool(writes),drains=len(drains),events=events,
            published_despite_newer_handoff=bool(writes and writes[0]['latest']!=expected))


import ast,hashlib,unittest

class LastDrainRegression(unittest.TestCase):
    def test_unchanged_scope_publishes_after_postdrain_metadata_and_latest_checks(self):
        result=probe('unchanged');events=result['events'];pivot=events.index('drain-3');published=events.index('publish-complete')
        self.assertIn('history-unchanged',events[pivot+1:published]);self.assertIn('latest-read',events[pivot+1:published])
        self.assertTrue(result['published']);self.assertFalse(result['published_despite_newer_handoff'])
    def test_newer_handoff_before_finalizer_rejected(self):
        self.assertFalse(probe('changed_before')['published'])
    def test_newer_handoff_during_last_drain_rejected(self):
        result=probe('changed_during_last_drain');self.assertFalse(result['published'])
        self.assertIn('Unknown newer handoff',result['error'])
    def test_inherited_metadata_during_last_drain_rejected(self):
        result=probe('metadata_during_last_drain');self.assertFalse(result['published'])
        self.assertIn('Inherited metadata identity changed',result['error'])
    def test_running_identity_during_last_drain_rejected(self):
        result=probe('running_identity_during_last_drain');self.assertFalse(result['published'])
        self.assertIn('Same final running metadata identity',result['error'])


class SourceEnrollment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=fixtures.binding();cls.source=Path(o.SELF).read_bytes()
    def test_pure_binding_retains_all_v1_maps_and_adds_exact_finalizer(self):
        context,source_bytes,plan=copy.deepcopy(self.fixture)
        original=candidate=o.c.render_binding(context,source_bytes,plan)
        with patch('builtins.open',side_effect=AssertionError('No IO')),patch.object(Path,'open',side_effect=AssertionError('No IO')):
            result=o.render_binding(context,source_bytes,plan,self.source)
        old=original['spec'];new=result['spec']
        for key in old:
            if key!='source_sha256':self.assertEqual(old[key],new[key],key)
        self.assertEqual(new['source_sha256'],dict(old['source_sha256'],**{o.SELF:hashlib.sha256(self.source).hexdigest()}))
        self.assertEqual(result['mandatory_source_sha256'][o.SELF],hashlib.sha256(self.source).hexdigest())
        self.assertNotIn(o.SELF,context['inherited_source_sha256'])
        self.assertFalse(result['runtime_enrolled']);self.assertFalse(result['spec']['deletion_authorized'])
    def test_cannot_replace_prior_finalizer_pin(self):
        context,sources,plan=copy.deepcopy(self.fixture);context['inherited_source_sha256'][o.SELF]='0'*64
        with self.assertRaisesRegex(ValueError,'replacement'):o.render_binding(context,sources,plan,self.source)
    def test_actual_finalizer_requires_exact_source_enrollment(self):
        io=SimpleNamespace(raw=Mock(return_value=self.source))
        with patch.object(caller,'api',return_value=(io,None)):
            with self.assertRaisesRegex(ValueError,'reviewed complete source closure'):o.verify_self({'source_sha256':{}})
            io.raw.assert_not_called()
            spec={'source_sha256':{o.SELF:hashlib.sha256(self.source).hexdigest()}}
            o.verify_self(spec);io.raw.assert_called_once_with(o.SELF,1<<20)
            spec['source_sha256'][o.SELF]='0'*64
            with self.assertRaisesRegex(ValueError,'source changed'):o.verify_self(spec)
    def test_frozen_v1_finalizer_body_retained_except_three_explicit_additions(self):
        def fn(path):return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='publish_completed')
        old=fn(Q/'minicpm5_retained_replay_observer_v1.py');new=fn(o.SELF)
        text=ast.unparse(new)
        self.assertEqual(text.count('verify_self(spec)'),2)
        class RemoveSelf(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name)and node.value.func.id=='verify_self':return None
                return self.generic_visit(node)
        new=RemoveSelf().visit(new)
        main_try=next(n for n in new.body if isinstance(n,ast.Try))
        self.assertEqual(ast.unparse(main_try.body[-3]),'history.unchanged()')
        self.assertEqual(ast.unparse(main_try.body[-2]),'guard(True)')
        del main_try.body[-3:-1]
        self.assertEqual(ast.dump(new,include_attributes=False),ast.dump(old,include_attributes=False))
    def test_real_observer_external_exit_and_full_history_constraints_are_reused(self):
        import minicpm5_retained_replay_observer_v1 as old
        self.assertIs(o.render_completed,old.render_completed);self.assertIs(o.reconcile,old.reconcile);self.assertIs(o.drain_phases,old.drain_phases)
        tree=ast.parse(Path(o.SELF).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='publish_completed')
        body=ast.unparse(fn)
        for name in ('history.physical_pair','actual_observer_tool','observer_owner','retain_until_drained','io.once','reconcile'):
            self.assertIn(name,body)


if __name__=='__main__':unittest.main(verbosity=2)
