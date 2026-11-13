"""Synthetic writer binding/observer tests; no live History or native work."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import test_prepare_qwen35_package_serial_candidate_v7 as old
import prepare_qwen35_writer_pair_candidate_v1 as p
from qwen35_writer_pair_validation_v1 import geometry, runtime
from qwen35_package_serial_runtime_v8.wiring import STAGES
from qwen35_writer_pair_serial_runtime_v2.test_source import GeometryMetadata


def load_observer():
    path = Path('/home/ubuntu/qwen35_writer_pair_post_exit_v1.py')
    spec = importlib.util.spec_from_file_location('writer_observer_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


observer = load_observer()


class Fixture(old.Fixture):
    def __init__(self):
        super().__init__(version=8)
        self.directory = p.RUNTIME_DIRECTORY
        previous_runtime = self.value(self.request['runtime_delivery'])
        runtime_files = {}
        for name in p.c.RUNTIME_FILES:
            target = str(self.directory/name)
            source = str(old.p.c.Q/'qwen35_package_serial_runtime_v8'/name)
            if source in self.metadata:
                ref = self.meta(target, self.value(self.metadata[source]['ref']))
            else:
                ref = self.source(target)
            runtime_files[target] = dict(sha256=ref['sha256'], bytes=self.identities[target]['size'])
        self.request['runtime_directory'] = str(self.directory)
        self.request['runtime_delivery'] = self.meta(str(self.directory/'source_delivery.json'), dict(
            schema='qwen35_writer_pair_serial_runtime_source_delivery_v2', source_only=True,
            runtime_admitted=False, files=runtime_files))
        self.request['runtime_review'] = self.meta('mock-writer-runtime-review.json', dict(
            schema='qwen35_writer_pair_runtime_v2_root_source_review_v1', reviewer='/root', source_accepted=True,
            blockers=[], runtime_admitted=False, source_delivery=self.request['runtime_delivery']))
        science_pins = {}
        for name in ('input_sources.json','capture_sources.json','geometry_sources.json'):
            sub = self.source(str(p.c.Q/('mock-writer-'+name+'.py')))
            ref = self.meta(str(p.c.Q/'qwen35_writer_pair_validation_v1'/name), {sub['path']:sub['sha256']})
            science_pins[ref['path']] = dict(sha256=ref['sha256'])
        for name in ('__init__.py','runtime.py'):
            ref = self.source(str(p.c.Q/'qwen35_writer_pair_validation_v1'/name))
            science_pins[ref['path']] = dict(sha256=ref['sha256'])
        self.meta(p.c.SCIENCE_DELIVERY['path'], dict(files=science_pins), fixed=p.c.SCIENCE_DELIVERY)
        # Exact runtime startup sources and imported predecessor utilities.
        for path in (*p.c.STARTUP_FILES, p.SELF, str(p.c.Q/'prepare_qwen35_package_serial_candidate_v7.py'),
                     str(p.c.Q/'qwen35_sampler_pair_protocol_v1.py'),
                     str(p.c.Q/'qwen35_sampler_history_supplement_v1.py'),
                     str(p.c.Q/'qwen35_sampler_torch_namespace_sources_v1.py')):
            ref = self.source(path); self.request['preparation_sources'][path] = ref['sha256']
        oldcase = self.request.pop('case')
        cases = {}
        for arm in ('baseline','candidate'):
            case = dict(oldcase, name='qwen35-writer-pair-v1-20261112', writer_arm=arm,
                writer_mode=runtime.WRITERS[arm], sampler_fast='1')
            for key in ('model_path','artifact','compiler_cache','compiler_workdir'):
                case[key] = str(Path(case[key]).parent/('qwen35-writer-pair-v1-mock-'+arm+'-'+key))
            cases[arm] = case
        self.request['writer_cases'] = cases
        self.request['future']['output_root'] = '/mnt/nvme/neuron-work/qww-mock-new'
        budget = copy.deepcopy(self.value(self.request['resource_budget']))
        budget.pop('case'); budget.update(schema='qwen35_writer_pair_reviewed_budget_v1', writer_cases=cases,
            output_root=self.request['future']['output_root'])
        prior_classes=budget['classes']
        budget['classes']={arm+'_'+kind:copy.deepcopy(prior_classes[kind]) for arm in ('baseline','candidate')
                          for kind in ('view','artifact','compiler_cache','compiler_workdir')}
        budget['classes']['build_auxiliary']=copy.deepcopy(prior_classes['build_auxiliary'])
        self.request['resource_budget']=self.meta('mock-writer-budget.json',budget)
        self.report_refs={name:self.meta('mock-writer-geometry-'+name+'.json',
            dict(source_sha256={str(p.c.Q/'mock-geometry-source.py'):'b'*64})) for name in geometry.LAYOUTS}
        gtool=dict(kind='tool_session',session_id=21,terminal_chunk='mock_geometry',exit_code=0)
        gowners=[dict(pid=21,start_ticks=21)]; gunits=['geometry']
        self.geometry_exit=self.meta('mock-writer-geometry-exit.json',dict(
            schema='qwen35_writer_geometry_root_actual_exit_v1',reviewer='/root',actual_tool_exit_observed=True,
            actual_tool=gtool,geometry_qualified=True,owners_gone=True,physically_drained=True,owners=gowners,units=gunits,
            reports=self.report_refs,source_sha256={str(p.c.Q/'mock-geometry-source.py'):'b'*64},full_model_qualified=False))
        self.request['geometry_qualification']=self.meta('mock-geometry-qualification.json',dict(
            schema='qwen35_writer_geometry_independent_qualification_v1',reviewer='/root',passed=True,blockers=[],
            reports=self.report_refs,actual_exit=self.geometry_exit,
            source_sha256={str(p.c.Q/'mock-geometry-source.py'):'b'*64},full_model_qualified=False))
        package_sources={str(p.c.Q/'mock-package-source.py'):'c'*64}
        self.package_spec=self.meta('mock-package-spec.json',dict(case=oldcase,source_sha256=package_sources))
        self.package_prefix=[]
        for index,(arm,stage) in enumerate(STAGES[:-1],1):
            result=self.meta(f'mock-package-result-{index}.json',dict(mock=True))
            self.package_prefix.append(self.meta(f'mock-package-stage-{index}.json',dict(index=index,arm=arm,
                stage=stage,case=oldcase,completed=True,result=result)))
        self.package_admission=self.meta('mock-package-admission.json',dict(mock=True))
        self.package_external=self.meta('mock-package-external.json',dict(mock=True))
        self.package_terminal=self.meta('mock-package-terminal.json',dict(mock=True))
        self.package_audit=self.meta('mock-package-audit.json',dict(schema='qwen35_package_independent_final_audit_v1',
            reviewer='/root',all25_stages_audited=True,spec=self.package_spec,admission=self.package_admission,
            external_tool_exit=self.package_external,outer_terminal=self.package_terminal,
            stage_receipts=self.package_prefix,storage=dict(mock_actual_inventory=True)))
        self.package_stage25=self.meta('mock-package-stage25.json',dict(index=25,arm=STAGES[-1][0],
            stage=STAGES[-1][1],case=oldcase,completed=True,result=self.package_audit))
        ptool=dict(kind='tool_session',session_id=22,terminal_chunk='mock_package',exit_code=0)
        powners=[dict(pid=22,start_ticks=22)]; punits=['package']
        self.request['package_completion']=self.meta('mock-package-completion.json',dict(
            schema='qwen35_package_root_actual_exit_v1',reviewer='/root',actual_tool_exit_observed=True,
            actual_tool=ptool,owners_gone=True,physically_drained=True,completed_stages=24,
            completed_stage_prefix=self.package_prefix,outcome=dict(all25_stages_audited=True,
                stage25=self.package_stage25,stage25_error=None),stage25=self.package_stage25,
            owners=powners,all_declared_units=punits,spec=self.package_spec,admission=self.package_admission,
            external_tool_exit=self.package_external,outer_terminal=self.package_terminal))
        self.value(self.catalog_ref)['runs'].extend([
            dict(terminal=self.geometry_exit,tool=gtool,owners=gowners,units=gunits,
                 source_sha256={str(p.c.Q/'mock-geometry-source.py'):'b'*64}),
            dict(terminal=self.request['package_completion'],tool=ptool,owners=powners,units=punits,
                 source_sha256=package_sources)])

    def run(self, *, real_geometry=False, **kwargs):
        with patch('builtins.open',side_effect=AssertionError('unexpected file IO')), \
             patch('os.open',side_effect=AssertionError('unexpected fd IO')), \
             patch('subprocess.Popen',side_effect=AssertionError('unexpected process')), \
             patch('fcntl.flock',side_effect=AssertionError('unexpected lock')), \
             patch.object(Path,'is_symlink',return_value=False):
            if real_geometry:
                return p.prepare(self.request,self.read_metadata,self.read_source,self.identity,self.guard)
            with patch.object(geometry,'check_complete_geometry',return_value={'synthetic_binding_only':True}):
                return p.prepare(self.request,kwargs.get('read_metadata',self.read_metadata),
                    kwargs.get('read_source',self.read_source),kwargs.get('identity_only',self.identity),self.guard)


class Preparation(unittest.TestCase):
    def test_complete_candidate_preserves_history_and_never_authorizes(self):
        f=Fixture(); out=f.run()
        self.assertFalse(out['passed']);self.assertFalse(out['launch_authorized']);self.assertFalse(out['history_resolved'])
        self.assertEqual(out['spec']['writer_cases'],f.request['writer_cases'])
        self.assertEqual(out['spec']['case'],f.request['writer_cases']['baseline'])
        self.assertEqual(out['preservation']['current_catalog_rows'],7)
        self.assertEqual(out['spec']['geometry_qualification'],f.request['geometry_qualification'])
        for path,sha in f.value(f.seed_ref)['source_sha256'].items():self.assertEqual(out['spec']['source_sha256'][path],sha)
        for key in ('geometry_qualification','package_completion'):
            ref=f.request[key];self.assertEqual(out['spec']['input_extension'][ref['path']]['sha256'],ref['sha256'])
        self.assertNotIn(f.binary_base_input['identity']['realpath'],out['spec']['input_extension'])
        self.assertFalse(any(path in f.request['binary_inputs'] for path,_,kind in f.reads))
        self.assertEqual(len([path for path in f.stats if path.endswith(('_bkt.bin','_ctrl.bin'))]),20)

    def test_all40000_synthetic_geometry_calls_checked_without_payload_reads(self):
        f=Fixture();GeometryMetadata.setUpClass()
        for name,report in GeometryMetadata.reports.items():
            value=copy.deepcopy(report);value['source_sha256']=f.value(f.geometry_exit)['source_sha256']
            f.metadata[f.report_refs[name]['path']]['value']=value
            f.identities[f.report_refs[name]['path']]['size']=len(p.encode(value))
        out=f.run(real_geometry=True);self.assertFalse(out['launch_authorized'])
        f.value(f.report_refs['conv'])['calls'][0]['proofs']['full_new']['identical']=False
        with self.assertRaises((ValueError, KeyError)):f.run(real_geometry=True)

    def test_incomplete_geometry_or_wrong_dtype_does_not_reach_budget(self):
        f=Fixture()
        with self.assertRaises((ValueError, KeyError)):f.run(real_geometry=True)
        for field,value in (('geometry_qualified',False),('physically_drained',False),('actual_tool_exit_observed',False)):
            with self.subTest(field=field):
                f=Fixture();f.value(f.geometry_exit)[field]=value
                with self.assertRaises(ValueError):f.run()

    def test_failed_or_unrecorded_geometry_rejected(self):
        f=Fixture();f.value(f.geometry_exit)['actual_tool']['exit_code']=1
        with self.assertRaises(ValueError):f.run()
        f=Fixture();f.value(f.catalog_ref)['runs'].pop(-2)
        with self.assertRaisesRegex(ValueError,'geometry row'):f.run()

    def test_package_exit0_alone_is_insufficient(self):
        for change in ('all25','stage25','prefix','storage','case','outer','catalog'):
            with self.subTest(change=change):
                f=Fixture();end=f.value(f.request['package_completion']);audit=f.value(f.package_audit)
                if change=='all25':end['outcome']['all25_stages_audited']=False
                if change=='stage25':f.value(f.package_stage25)['completed']=False
                if change=='prefix':f.value(f.package_prefix[8])['index']=3
                if change=='storage':audit['storage']={}
                if change=='case':f.value(f.package_prefix[0])['case']={}
                if change=='outer':audit['outer_terminal']=f.geometry_exit
                if change=='catalog':f.value(f.catalog_ref)['runs'].pop()
                with self.assertRaises(ValueError):f.run()

    def test_both_writers_same_sampler_and_disjoint_paths_required(self):
        for change in ('sampler','writer','paths'):
            f=Fixture();case=f.request['writer_cases']['candidate']
            if change=='sampler':case['sampler_fast']='0'
            if change=='writer':case['writer_mode']='torch_where'
            if change=='paths':case['artifact']=f.request['writer_cases']['baseline']['artifact']
            with self.subTest(change=change),self.assertRaises(ValueError):f.run()

    def test_dual_budget_cannot_reuse_single_arm_or_missing_cap(self):
        for change in ('class','aux','root','headroom','review'):
            f=Fixture();budget=f.value(f.request['resource_budget'])
            if change=='class':budget['classes'].pop('candidate_artifact')
            if change=='aux':budget['classes']['build_auxiliary']['allocated_bytes']=2*p.c.LIMITS['leaf_log_bytes']
            if change=='root':budget['output_root']='/mnt/nvme/neuron-work/qwp-mock-new'
            if change=='headroom':budget['headroom_bytes']['nvme']=0
            if change=='review':budget['passed']=False
            with self.subTest(change=change),self.assertRaises(ValueError):f.run()

    def test_old_catalog_prefix_and_sources_cannot_be_replaced(self):
        f=Fixture();f.value(f.catalog_ref)['runs'][0]['owners']=[]
        with self.assertRaises(ValueError):f.run()
        f=Fixture();f.value(f.package_spec)['source_sha256'][str(p.c.Q/'mock-geometry-source.py')]='c'*64
        with self.assertRaisesRegex(ValueError,'conflict'):f.run()

    def test_geometry_source_crossing_and_missing_helper_rejected(self):
        f=Fixture();f.value(f.report_refs['conv'])['source_sha256']={}
        with self.assertRaises(ValueError):f.run()
        f=Fixture();del f.request['preparation_sources'][p.SELF]
        with self.assertRaises(ValueError):f.run()

    def test_identity_changed_at_final_recheck_rejected(self):
        f=Fixture();calls={}
        def identity(path):
            value=f.identity(path);calls[path]=calls.get(path,0)+1
            if path==next(iter(f.request['binary_inputs'])) and calls[path]>1:value['inode']+=1
            return value
        with self.assertRaisesRegex(ValueError,'stat-only payload'):f.run(identity_only=identity)

    def test_retained_metadata_cannot_be_overwritten(self):
        f=Fixture();f.request['future']['spec_path']=f.package_audit['path']
        with self.assertRaisesRegex(ValueError,'cannot overwrite'):f.run()


class Observer(unittest.TestCase):
    def test_failed_outer_never_invokes_stage28(self):
        for terminal in (None,dict(failed=True,result_present=False)):
            result=observer.audit_outcome({'exit_code':1},terminal,None,None,None,None,
                lambda *_:self.fail('stage28 must not run'))
            self.assertFalse(result['all28_stages_audited']);self.assertFalse(result['full_model_qualified'])

    def test_exit0_missing_terminal_rejected(self):
        with self.assertRaises(ValueError):observer.audit_outcome({'exit_code':0},None,None,None,None,None,lambda *_:None)

    def test_stage28_failure_preserved_independent_of_exit0(self):
        def fail(*_):raise ValueError('numeric mismatch')
        result=observer.audit_outcome({'exit_code':0},dict(failed=False,result_present=True),None,None,None,None,fail)
        self.assertEqual(result['actual_outer_exit_code'],0);self.assertFalse(result['all28_stages_audited'])
        self.assertEqual(result['stage28_error']['message'],'numeric mismatch')

    def test_successful_stage28_still_requires_root_qualification(self):
        result=observer.audit_outcome({'exit_code':0},dict(failed=False,result_present=True),None,None,None,None,
                                     lambda *_:{'path':'/mock/stage28.json','sha256':'a'*64})
        self.assertTrue(result['all28_stages_audited']);self.assertFalse(result['full_model_qualified'])

    def test_exact_predecessor_rows_required_and_retained(self):
        f=Fixture();oldcat=f.value(f.catalog_ref)
        ref=dict(path='/mock/next.json',sha256='a'*64)
        out=observer.append_catalog(oldcat,ref,ref,{'kind':'tool_session'},[],[],{},p.catalog_shape,7)
        self.assertEqual(out['runs'][:-1],oldcat['runs'])
        with self.assertRaises(ValueError):observer.append_catalog(oldcat,ref,ref,{},[],[],{},p.catalog_shape,6)

    def test_actual_tool_session_identity_and_terminal_required(self):
        value=dict(schema='qwen35_package_external_tool_result_v1',reviewer='/root',completed=True,
            observation_source='actual_external_tool_result',observed_unix_ns=1,
            actual_tool=dict(kind='tool_session',session_id=33,terminal_chunk='mock_terminal',exit_code=0))
        self.assertEqual(observer.actual_tool(value,dict(kind='tool_session',session_id=33))['exit_code'],0)
        with self.assertRaises(ValueError):observer.actual_tool(value,dict(kind='tool_session',session_id=34))
        value['actual_tool'].pop('terminal_chunk')
        with self.assertRaises(ValueError):observer.actual_tool(value,dict(kind='tool_session',session_id=33))

    def test_source_contains_complete_scope_and_all28_not_package25(self):
        source=Path(observer.__file__).read_text()
        self.assertIn("value['writer_cases'] == spec['writer_cases']",source)
        self.assertIn('qwen35_writer_pair_serial_runtime_v2',source)
        self.assertIn("spec['source_sha256'].get(r['path']) == r['sha256']",source)
        self.assertNotIn('stage25',source)
        self.assertNotIn('shared-build',source)
        for path in (Path(p.__file__),Path(observer.__file__),Path('/home/ubuntu/ack_qwen35_writer_pair_root_v1.py')):
            ast.parse(path.read_text())
        self.assertFalse(any(n.split('.')[0] in ('torch','numpy','vllm','neuronxcc') for n in sys.modules))



# Reuse complete-scope failure mocks against this exact writer observer.
import test_qwen35_package_r5_post_exit_source_v1 as old_scope

class ScopeChecks(old_scope.ScopeChecks):
    def setUp(self):
        selected=patch.object(old_scope,'p',observer);selected.start();self.addCleanup(selected.stop)

class Envelope(unittest.TestCase):
    def test_envelope_is_small_exact_and_never_an_admission(self):
        f=Fixture();out=f.run();value,payload=p.candidate_envelope(out,f.request)
        self.assertEqual(json.loads(payload),value);self.assertFalse(value['launch_authorized'])
        self.assertEqual(value['completed_prerequisites'],out['completed_prerequisites'])
        self.assertIn(f.package_spec['path'],value['preparation_only_references'])
        self.assertNotIn(f.package_spec['path'],out['spec']['input_extension'])
        self.assertLess(len(payload),8*2**20)

    def test_changed_bytes_or_different_catalog_cannot_be_enveloped(self):
        f=Fixture();out=f.run();out['spec_bytes']+=b' '
        with self.assertRaises(ValueError):p.candidate_envelope(out,f.request)
        out=f.run();f.request['current_catalog']=dict(path='/mock/foreign.json',sha256='a'*64)
        with self.assertRaises(ValueError):p.candidate_envelope(out,f.request)

class RootHelpers(unittest.TestCase):
    def load(self,name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

    def setup_acceptance(self):
        import types
        from unittest.mock import Mock
        from qwen35_writer_pair_serial_runtime_v2 import io,budget,history
        # Replace the historical dependency before import; do not load its
        # ctypes lifecycle helper while testing caller wiring.
        latest=types.ModuleType('research.minicpm5_layer1_operator_sdk_runtime_completion_v4')
        latest.latest_handoff=Mock()
        module=self.load('writer_accept_test','/home/ubuntu/accept_qwen35_writer_pair_root_v1.py')
        f=Fixture()
        for path in ('/home/ubuntu/accept_qwen35_writer_pair_root_v1.py',
                     '/home/ubuntu/ack_qwen35_writer_pair_root_v1.py',
                     '/home/ubuntu/qwen35_writer_pair_post_exit_v1.py'):
            ref=f.source(path);f.request['preparation_sources'][path]=ref['sha256']
        out=f.run();candidate,_=p.candidate_envelope(out,f.request)
        f.value(f.idle_ref)['active_serial']=None
        candidate_ref=f.meta('mock-accepted-candidate.json',candidate)
        caller_delivery=f.meta('mock-caller-delivery.json',dict(files={path:dict(sha256=out['spec']['source_sha256'][path])
            for path in ('/home/ubuntu/accept_qwen35_writer_pair_root_v1.py',)}))
        review=f.meta('mock-caller-review.json',dict(reviewer='/root',source_accepted=True,blockers=[],runtime_admitted=False,
            source_delivery=caller_delivery))
        request=dict(schema='qwen35_writer_pair_root_accept_request_v1',candidate=candidate_ref,caller_source_review=review,
            review_path=str(p.c.Q/'mock-root-final-review.json'),binding_path=str(p.c.Q/'mock-root-final-binding.json'))
        request_ref=f.meta('mock-root-accept-request.json',request)
        values={path:row['value'] for path,row in f.metadata.items()}
        values[out['spec_ref']['path']]=out['spec'];values[out['review_ref']['path']]=out['review']
        def reference(path,*_):
            path=str(path)
            if path in out['spec']['source_sha256']:return dict(path=path,sha256=out['spec']['source_sha256'][path])
            return dict(path=path,sha256='a'*64)
        tip=dict(path=f.idle_ref['path'],identity=f.identities[f.idle_ref['path']])
        lease=types.SimpleNamespace(acquire=Mock(),check=Mock(),release=Mock())
        bound=types.SimpleNamespace(resolve=Mock(),unchanged=Mock(),freshness=Mock(return_value={'latest_handoff':f.idle_ref}),
            supplement={'scope':{'children':[None]*874}});bound.resolve.return_value=bound
        publications=[]
        def once(path,value):publications.append((str(path),copy.deepcopy(value)));return reference(path)
        return locals()

    def run_acceptance(self,ctx,*,history_error=None,tip_changed=False):
        import contextlib
        from unittest.mock import Mock, mock_open
        import resource,signal
        f=ctx['f'];io=ctx['io'];budget=ctx['budget'];history=ctx['history'];latest=ctx['latest']
        if history_error:ctx['bound'].resolve.side_effect=ValueError(history_error)
        with contextlib.ExitStack() as stack:
            import research
            stack.enter_context(patch.object(research,'minicpm5_layer1_operator_sdk_runtime_completion_v4',latest,create=True))
            stack.enter_context(patch.dict(sys.modules,{'research.minicpm5_layer1_operator_sdk_runtime_completion_v4':latest}))
            for target,value in ((resource,'setrlimit'),(signal,'alarm')):stack.enter_context(patch.object(target,value))
            stack.enter_context(patch.object(io,'read',side_effect=lambda ref,*_:copy.deepcopy(ctx['values'][ref['path']])) )
            stack.enter_context(patch.object(io,'ref',side_effect=ctx['reference']))
            stack.enter_context(patch.object(io,'identity',side_effect=lambda path:copy.deepcopy(f.identities[str(path)])))
            stack.enter_context(patch.object(io,'once',side_effect=ctx['once']))
            stack.enter_context(patch.object(io,'RetainedLock',return_value=ctx['lease']))
            stack.enter_context(patch.object(io,'process_api',return_value=Mock()))
            stack.enter_context(patch.object(io,'resources',return_value={'mock_reserved':True}))
            stack.enter_context(patch.object(io,'loaded_local_sources'))
            stack.enter_context(patch.object(budget,'fresh_paths'))
            stack.enter_context(patch.object(budget,'unchanged_inputs'))
            stack.enter_context(patch.object(history,'Bundle',return_value=ctx['bound']))
            stack.enter_context(patch.object(latest,'latest_handoff',return_value=ctx['tip'] if not tip_changed else {}))
            prereq=stack.enter_context(patch.object(p,'completed_prerequisites'))
            stack.enter_context(patch.object(Path,'exists',return_value=False))
            stack.enter_context(patch.object(Path,'is_symlink',return_value=False))
            stack.enter_context(patch.object(Path,'resolve',lambda x:x))
            stack.enter_context(patch.object(Path,'open',mock_open()))
            stack.enter_context(patch('os.fsync'))
            stack.enter_context(patch('builtins.print'))
            ctx['module'].main(['--request',ctx['request_ref']['path'],'--request-sha256',ctx['request_ref']['sha256']])
            prereq.assert_called_once()

    def test_root_acceptance_requires_complete_history_before_binding(self):
        ctx=self.setup_acceptance();self.run_acceptance(ctx)
        ctx['bound'].resolve.assert_called_once();ctx['bound'].unchanged.assert_called_once()
        ctx['lease'].acquire.assert_called_once();ctx['lease'].release.assert_called_once_with(True)
        self.assertEqual(len(ctx['publications']),1)
        result=ctx['publications'][0][1]
        self.assertEqual(result['lfm_children'],874);self.assertTrue(result['full_source_history_resolution_passed'])
        self.assertFalse(result['runtime_launched']);self.assertFalse(result['full_model_qualified'])

    def test_history_failure_never_publishes_binding(self):
        ctx=self.setup_acceptance()
        with self.assertRaisesRegex(ValueError,'history incomplete'):self.run_acceptance(ctx,history_error='history incomplete')
        self.assertFalse(ctx['publications']);ctx['lease'].release.assert_called_once_with(True)

    def test_stale_idle_never_acquires_lock_or_publishes(self):
        ctx=self.setup_acceptance()
        with self.assertRaisesRegex(ValueError,'current idle'):self.run_acceptance(ctx,tip_changed=True)
        ctx['lease'].acquire.assert_not_called();self.assertFalse(ctx['publications'])

    def test_missing_inherited_child_refuses_binding(self):
        ctx=self.setup_acceptance();ctx['bound'].supplement['scope']['children'].pop()
        with self.assertRaisesRegex(ValueError,'874'):self.run_acceptance(ctx)
        self.assertFalse(ctx['publications'])


import contextlib
from types import SimpleNamespace
from unittest.mock import Mock
import qwen35_writer_pair_audit_physical_v1 as physical_adapter

def observation():return dict(path='/mock/running.json',identity=dict(inode=7,device=1,mtime_ns=2))

class FreshnessTests(unittest.TestCase):
    def test_first_at_zero_every_lock_and_exact_half_second_boundary(self):
        lock=SimpleNamespace(check=Mock());guard=observer.FreshnessGuard(lock,observation())
        with patch.object(observer.time,'monotonic',side_effect=[0.,.1,.499999,.5,.75,1.]),patch.object(observer,'latest_handoff',return_value=observation()) as latest:
            for _ in range(6):guard()
        self.assertEqual(lock.check.call_count,6);self.assertEqual(latest.call_count,3)

    def test_force_inside_interval_does_full_scan(self):
        lock=SimpleNamespace(check=Mock());guard=observer.FreshnessGuard(lock,observation())
        with patch.object(observer.time,'monotonic',return_value=0.),patch.object(observer,'latest_handoff',return_value=observation()) as latest:
            guard();guard();guard(force=True);guard()
        self.assertEqual(lock.check.call_count,4);self.assertEqual(latest.call_count,2)

    def test_changed_path_and_same_path_identity_rejected(self):
        for changed in [dict(observation(),path='/mock/new.json'),dict(observation(),identity=dict(inode=8))]:
            lock=SimpleNamespace(check=Mock());guard=observer.FreshnessGuard(lock,observation())
            with self.subTest(changed=changed),patch.object(observer.time,'monotonic',side_effect=[0.,.1,.5]),                    patch.object(observer,'latest_handoff',side_effect=[observation(),changed]) as latest:
                guard();guard()
                with self.assertRaisesRegex(ValueError,'Running binding changed'):guard()
            self.assertEqual(latest.call_count,2);self.assertEqual(lock.check.call_count,3)
            self.assertEqual(guard.last_full_started,0.)

    def test_force_rejects_change_before_half_second(self):
        guard=observer.FreshnessGuard(SimpleNamespace(check=Mock()),observation())
        with patch.object(observer.time,'monotonic',return_value=0.),patch.object(observer,'latest_handoff',side_effect=[observation(),dict(observation(),path='/new')]):
            guard()
            with self.assertRaises(ValueError):guard(force=True)

    def test_lock_failure_cannot_hide_in_interval(self):
        lock=SimpleNamespace(check=Mock(side_effect=[None,ValueError('lost lock')]))
        guard=observer.FreshnessGuard(lock,observation())
        with patch.object(observer.time,'monotonic',return_value=0.),patch.object(observer,'latest_handoff',return_value=observation()) as latest:
            guard()
            with self.assertRaisesRegex(ValueError,'lost lock'):guard()
        self.assertEqual(latest.call_count,1)

    def test_slow_scan_uses_start_time_and_forces_next_due_call(self):
        now=[0.];guard=observer.FreshnessGuard(SimpleNamespace(check=Mock()),observation())
        def latest():now[0]+=1.;return observation()
        with patch.object(observer.time,'monotonic',side_effect=lambda:now[0]),patch.object(observer,'latest_handoff',side_effect=latest) as scan:
            guard();guard()
        self.assertEqual(scan.call_count,2);self.assertEqual(guard.last_full_started,1.)

    def test_backward_or_nonfinite_clock_fails_closed(self):
        for bad in [-1.,float('nan'),float('inf')]:
            guard=observer.FreshnessGuard(SimpleNamespace(check=Mock()),observation())
            with self.subTest(bad=bad),patch.object(observer.time,'monotonic',side_effect=[0.,bad]),                    patch.object(observer,'latest_handoff',return_value=observation()):
                guard()
                with self.assertRaisesRegex(ValueError,'Monotonic'):guard()

    def test_caller_cannot_rebind_saved_expected_identity(self):
        expected=observation();guard=observer.FreshnessGuard(SimpleNamespace(check=Mock()),expected)
        expected['identity']['inode']=999
        with patch.object(observer.time,'monotonic',return_value=0.),patch.object(observer,'latest_handoff',return_value=expected):
            with self.assertRaises(ValueError):guard()

    def test_missing_original_ack_observation_fails_closed(self):
        for bad in [{},dict(path='/mock'),dict(path='/mock',identity=None),dict(observation(),extra=True)]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):observer.FreshnessGuard(SimpleNamespace(check=Mock()),bad)

    def test_48627_guard_calls_preserve_all_locks_with_bounded_scans(self):
        lock=SimpleNamespace(check=Mock());guard=observer.FreshnessGuard(lock,observation());now=[0.]
        with patch.object(observer.time,'monotonic',side_effect=lambda:now[0]),patch.object(observer,'latest_handoff',return_value=observation()) as latest:
            for i in range(48627):
                now[0]=i/10000;guard()
        self.assertEqual(lock.check.call_count,48627);self.assertEqual(latest.call_count,10)

class PhysicalPassTests(unittest.TestCase):

    def exercise(self, change_at_full_scan=None):
        from research import minicpm5_sdk_physical_evidence_v3 as original
        units = {'unit-%04d' % i for i in range(1308)}
        known = units | {'extra'}
        done = dict(nemotron={'history': {'rows': [{'unit': u} for u in sorted(units)]}}, owners=[{'pid': 1}], lock_identity={'inode': 2})
        expected = observation()
        events = []
        count = [0]
        lock = SimpleNamespace(check=lambda : events.append('lock'))
        guard = observer.FreshnessGuard(lock, expected)

        def latest():
            count[0] += 1
            events.append('latest:' + str(count[0]))
            return dict(expected, path='/changed') if count[0] == change_at_full_scan else copy.deepcopy(expected)

        def physical(value, check):
            self.assertIs(value, done)
            self.assertIs(check, guard)
            events.append('physical')
            check()
            return dict(all_additional_owners_checked=done['owners'], lock_identity=done['lock_identity'])

        def props(name):
            self.assertEqual(name, 'extra')
            events.append('properties')
            return {'mock': True}

        def group(path):
            events.append('group')
            return {'empty': True}

        def validate_observation(value, known_units):
            self.assertEqual(known_units, known)
            self.assertEqual(len(value['supplemental_observations']), 1)
            events.append('pass_validated')
            return known

        def validate_pair(pair, required, owners, lock_identity, *, known_units):
            self.assertEqual(required, units)
            self.assertEqual(owners, done['owners'])
            self.assertEqual(lock_identity, done['lock_identity'])
            self.assertEqual(known_units, known)
            self.assertEqual(pair['first'], pair['second'])
            events.append('pair_validated')
        with contextlib.ExitStack() as stack:
            patches = [patch.object(observer.time, 'monotonic', return_value=0.0), patch.object(observer, 'latest_handoff', side_effect=latest), patch.object(original, 'unit_name', side_effect=lambda x: x), patch.object(original, '_tree_header'), patch.object(original, 'inspect_tree', return_value={'units': known, 'needs_supplement': {'extra'}}), patch.object(original, 'observation'), patch.object(original, 'validate_observation', side_effect=validate_observation), patch.object(original, 'validate_pair', side_effect=validate_pair)]
            for p in patches:
                stack.enter_context(p)
            if change_at_full_scan:
                with self.assertRaisesRegex(ValueError, 'Running binding changed'):
                    physical_adapter._collect_pair(done, guard, physical, props, group, known_units=known)
                pair = None
            else:
                pair = physical_adapter._collect_pair(done, guard, physical, props, group, known_units=known)
        return (pair, events)

    def test_each_complete_physical_pass_forces_first_and_last_at_same_clock(self):
        (pair, events) = self.exercise()
        self.assertIsNotNone(pair)
        self.assertEqual(events.count('physical'), 2)
        self.assertEqual(events.count('properties'), 2)
        self.assertEqual(events.count('group'), 2)
        self.assertEqual(events.count('pass_validated'), 2)
        self.assertEqual(events.count('pair_validated'), 1)
        self.assertEqual([x for x in events if x.startswith('latest:')], ['latest:1', 'latest:2', 'latest:3', 'latest:4'])
        for (i, event) in enumerate(events):
            if event == 'physical':
                self.assertTrue(events[i - 1].startswith('latest:'))
            if event == 'pass_validated':
                self.assertEqual(events[i + 1], 'lock')
                self.assertTrue(events[i + 2].startswith('latest:'))

    def test_change_at_each_pass_boundary_fails_before_next_work(self):
        for (at, physical_count) in [(1, 0), (2, 1), (3, 1), (4, 2)]:
            with self.subTest(at=at):
                (pair, events) = self.exercise(change_at_full_scan=at)
                self.assertIsNone(pair)
                self.assertEqual(events.count('physical'), physical_count)
                self.assertNotIn('pair_validated', events)

    def test_current_physical_wiring_keeps_original_callbacks_and_disjoint_scope(self):
        import research
        original = SimpleNamespace(p=SimpleNamespace(properties=Mock(), empty_group=Mock()), physical=Mock(return_value={'all_scope': True}))
        done = {'complete_known_units': ['one', 'two']}
        guard = Mock()
        authority = SimpleNamespace(disjoint=Mock())

        def collector(value, check, physical, props, groups, *, known_units):
            self.assertIs(value, done)
            self.assertIs(check, guard)
            self.assertEqual(known_units, done['complete_known_units'])
            self.assertIs(props, original.p.properties)
            self.assertIs(groups, original.p.empty_group)
            return physical(value, check)
        with patch.object(physical_adapter, '_collect_pair', side_effect=collector), patch.object(research, 'minicpm5_native_current_physical_v1', original, create=True), patch.dict(sys.modules, {'research.minicpm5_native_current_physical_v1': original}):
            self.assertEqual(physical_adapter.collect_pair(done, guard, authority), {'all_scope': True})
        original.physical.assert_called_once_with(done, guard, authority)
        authority.disjoint.assert_called_once_with(done['complete_known_units'])


class AuditPreservation(unittest.TestCase):
    def test_each_final_audit_publication_has_its_own_forced_discovery(self):
        import qwen35_writer_pair_post_audit_v1 as successor
        tree=ast.parse(Path(successor.__file__).read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
        publications=[]
        for index,node in enumerate(function.body):
            if (isinstance(node,ast.Assign) and isinstance(node.value,ast.Call)
                    and isinstance(node.value.func,ast.Name) and node.value.func.id=='once'):
                prior=function.body[index-1]
                self.assertIsInstance(prior,ast.Expr);self.assertIsInstance(prior.value,ast.Call)
                self.assertEqual(prior.value.func.id,'guard')
                self.assertEqual([(k.arg,getattr(k.value,'value',None)) for k in prior.value.keywords],[('force',True)])
                publications.append(node.targets[0].id)
        self.assertEqual(publications,['result_ref','receipt'])

    def test_complete_pair_collector_is_exact_accepted_v5_body(self):
        def fn(path, name):
            return next(n for n in ast.parse(Path(path).read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
        self.assertEqual(ast.dump(fn(physical_adapter.__file__, '_collect_pair'), include_attributes=False),
                         ast.dump(fn(p.c.Q/'qwen35_geometry_audit_physical_v5.py', '_collect_pair'), include_attributes=False))

    def test_full28_audit_science_and_exit_statements_preserved(self):
        from qwen35_writer_pair_serial_runtime_v2 import final_audit
        import qwen35_writer_pair_post_audit_v1 as successor
        source=Path(successor.__file__).read_text().replace('qwen35_writer_pair_serial_runtime_v2.', '.')
        source=source.replace('from .history import Authority\nfrom qwen35_writer_pair_audit_physical_v1 import Bundle',
                              'from .history import Bundle, Authority')
        source=source.replace('import importlib\n','').replace('import sys\n','')
        source=source.replace('PHASES, LIMITS, PACKAGE','PHASES, LIMITS')
        source=source.replace('def guard(force=False):', 'def guard():').replace('execution_guard(force=force)', 'execution_guard()')
        class RemoveGuardExpr(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id in ('guard','bind_pure_package_sources'):
                    return None
                return self.generic_visit(node)
            def visit_FunctionDef(self,node):
                if node.name=='bind_pure_package_sources':return None
                return self.generic_visit(node)
        new=RemoveGuardExpr().visit(ast.parse(source))
        oldtree=RemoveGuardExpr().visit(ast.parse(Path(final_audit.__file__).read_text()))
        self.assertEqual(ast.dump(new,include_attributes=False),ast.dump(oldtree,include_attributes=False))
        self.assertIs(successor.Bundle,physical_adapter.Bundle)

    def test_history_resolve_inputs_and_sources_inherited_unchanged(self):
        from qwen35_writer_pair_serial_runtime_v2.history import Bundle
        for name in ('__init__','resolve','unchanged','freshness','scope'):
            self.assertIs(getattr(physical_adapter.Bundle,name),getattr(Bundle,name))

class AckChecks(unittest.TestCase):
    def exercise(self,*,missing_owner=False,stale_tip=False):
        import types
        from qwen35_writer_pair_serial_runtime_v2 import io,adapter,history
        import research
        helper=RootHelpers();module=helper.load('writer_ack_test','/home/ubuntu/ack_qwen35_writer_pair_root_v1.py')
        f=Fixture();self_ref=f.source('/home/ubuntu/ack_qwen35_writer_pair_root_v1.py')
        f.request['preparation_sources'][self_ref['path']]=self_ref['sha256'];out=f.run();spec=out['spec']
        root=p.c.control_root(spec);review_ref=dict(path=str(p.c.Q/'mock-final-review.json'),sha256='a'*64)
        admission_ref=dict(path=str(root/'admission.json'),sha256='a'*64)
        owner_ref=dict(path=str(root/'coordinator-owner.json'),sha256='a'*64)
        outer={'pid':11,'start_ticks':'11'};co={'pid':22,'start_ticks':'22'}
        owner={'process':co,'service_owner':{'unit':'mock-writer-coordinator'}}
        admission=dict(outer_owner=outer,retained_fd=9,spec=out['spec_ref'],review=review_ref,
            freshness={'latest_handoff':f.idle_ref},current_serial_catalog=f.catalog_ref,units=p.c.units(spec['run_name']))
        binding_ref=dict(path=str(p.c.Q/'mock-binding.json'),sha256='a'*64)
        binding=dict(schema='qwen35_writer_pair_root_concrete_binding_v1',reviewer='/root',passed=True,
            full_source_history_resolution_passed=True,spec=out['spec_ref'],review=review_ref)
        values={binding_ref['path']:binding,out['spec_ref']['path']:spec,
            review_ref['path']:dict(out['review'],passed=True,blockers=[],launch_authorized=True),
            admission_ref['path']:admission,owner_ref['path']:owner,f.idle_ref['path']:f.value(f.idle_ref)}
        refs={r['path']:r for r in (self_ref,admission_ref,owner_ref)}
        process=SimpleNamespace(proc=Mock(side_effect=lambda pid:None if missing_owner else outer if pid==11 else co),active_heavy=Mock())
        service=SimpleNamespace(unit='mock-writer-coordinator',live=Mock())
        latest=types.ModuleType('research.minicpm5_layer1_operator_sdk_runtime_completion_v4')
        latest.latest_handoff=Mock(return_value={} if stale_tip else dict(path=f.idle_ref['path'],identity=f.identities[f.idle_ref['path']]))
        publications=[]
        def once(path,value):publications.append((str(path),copy.deepcopy(value)));return dict(path=str(path),sha256='a'*64)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(research,'minicpm5_layer1_operator_sdk_runtime_completion_v4',latest,create=True))
            stack.enter_context(patch.dict(sys.modules,{'research.minicpm5_layer1_operator_sdk_runtime_completion_v4':latest}))
            stack.enter_context(patch.object(io,'read',side_effect=lambda r,*_:copy.deepcopy(values[r['path']])))
            stack.enter_context(patch.object(io,'ref',side_effect=lambda path,*_:refs[str(path)]))
            stack.enter_context(patch.object(io,'identity',side_effect=lambda path:f.identities[str(path)]))
            stack.enter_context(patch.object(io,'descriptor',return_value={'retained':True}))
            stack.enter_context(patch.object(io,'process_api',return_value=process))
            stack.enter_context(patch.object(io,'once',side_effect=once))
            stack.enter_context(patch.object(adapter,'admit_shape'))
            stack.enter_context(patch.object(adapter,'coordinator_service',return_value=service))
            stack.enter_context(patch.object(Path,'exists',return_value=False))
            stack.enter_context(patch.object(Path,'is_symlink',return_value=False))
            stack.enter_context(patch('builtins.print'))
            args=['--binding',binding_ref['path'],'--binding-sha256',binding_ref['sha256'],
                  '--root-live',str(p.c.Q/'mock-new-root-live.json'),'--session-id','53']
            if missing_owner or stale_tip:
                with self.assertRaises(ValueError):module.main(args)
            else:module.main(args)
        return publications,process,service

    def test_exact_live_owners_and_actual_session_bind_running(self):
        publications,process,service=self.exercise();self.assertEqual(len(publications),2)
        self.assertEqual(publications[0][1]['live_tool_sessions'],[53])
        self.assertEqual(publications[1][1]['actual_tool'],dict(kind='tool_session',session_id=53))
        process.active_heavy.assert_called_once_with(('mock-writer-coordinator',));service.live.assert_called_once()

    def test_absent_owner_never_publishes_running(self):
        self.assertFalse(self.exercise(missing_owner=True)[0])

    def test_unknown_successor_never_publishes_running(self):
        self.assertFalse(self.exercise(stale_tip=True)[0])


if __name__=='__main__':unittest.main(verbosity=2)
