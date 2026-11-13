"""Pure/mock helper coverage; no real runtime metadata, History or native IO."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent),'/home/ubuntu']
import test_qwen35_kv256_s16_pair_serial_v1 as serial
import check_qwen35_kv256_alias_diagnostic_binding_v1 as original
import check_qwen35_kv256_alias_diagnostic_binding_v2 as predecessor
import prepare_qwen35_kv256_s16_pair_binding_v1 as prepare
import qwen35_kv256_s16_pair_root_helpers_v1 as helpers
import qwen35_kv256_s16_pair_audit_v1 as auditor
import render_qwen35_kv256_s16_pair_helpers_v1 as renderer


def fixture(with_plan=False):
    with patch.object(original,'s',prepare),patch.object(original,'c',serial.contract),patch.object(predecessor,'v2',prepare):
        future,values,identities=predecessor.v2_fixture(with_plan)
    future['schema']='qwen35_kv256_s16_pair_future_binding_v1'
    future['paths']['output']='/mnt/nvme/neuron-work/qwen35-kv256-s16-pair-'+future['run_name']
    review=values[future['diagnostic_source_review']['path']]
    values[review['source_delivery']['path']]['schema']='qwen35_kv256_s16_pair_serial_source_delivery_v1'
    values[future['preparer_delivery']['path']]['schema']='qwen35_kv256_s16_pair_binding_source_delivery_v1'
    return future,values,identities


def build(future,values,identities):
    def read(ref):return dict(value=copy.deepcopy(values[ref['path']]),identity=copy.deepcopy(identities[ref['path']]))
    return prepare.prepare(future,read,lambda _:True,lambda:None)


class InheritedBindingTests(original.BindingTests):
    """All original binding controls, with the exact pair workload substituted."""
    def setUp(self):
        for name,value in [('s',prepare),('c',serial.contract)]:
            p=patch.object(original,name,value);p.start();self.addCleanup(p.stop)
        self.f,self.v,self.i=fixture()

    def test_complete_synthetic_candidate_preserves_science_and_no_authority(self):
        before=copy.deepcopy((self.f,self.v,self.i));result=self.run_builder()['values']
        self.assertEqual(before,(self.f,self.v,self.i))
        self.assertEqual(result['spec']['workload'],serial.protocol.workload())
        self.assertEqual(result['spec']['limits'],serial.contract.LIMITS)
        self.assertEqual(result['spec']['workload']['calls'],6)
        self.assertEqual(len(result['spec']['workload']['graphs']),2)
        self.assertFalse(result['review_draft']['passed']);self.assertFalse(result['review_draft']['launch_authorized'])
        for key in ('passed','launch_authorized','physical_pair_performed','lock_acquired','native_executed',
                    'compiler_executed','runtime_launched','full_geometry_qualified','full_model_qualified','new_speedup_claim'):
            self.assertIs(result['candidate'][key],False)
        self.assertEqual(result['candidate']['catalog_rows'],2)
        self.assertEqual(result['candidate']['preservation']['seed_lfm_children'],874)
        self.assertEqual(result['spec']['lfm_r2_scope'],self.value('seed_spec')['lfm_r2_scope'])


class RetainedPlanTests(unittest.TestCase):
    def test_exact_plan_identity_and_predecessor_pin_retained(self):
        f,v,i=fixture(True);before=copy.deepcopy((f,v,i));result=build(f,v,i)['values'];path=prepare.INHERITED_PLAN['path']
        row=v[f['seed_spec']['path']]['input_extension'][path]
        self.assertEqual(result['spec']['input_extension'][path],row)
        self.assertEqual(result['candidate']['preservation']['exact_retained_typed_inputs'],{path:row})
        self.assertNotIn(path,result['candidate']['preparation_only_inputs'])
        self.assertEqual(result['spec']['source_sha256'][prepare.PREDECESSOR_PREPARER['path']],prepare.PREDECESSOR_PREPARER['sha256'])
        self.assertEqual(before,(f,v,i))

    def test_changed_inherited_identity_or_size_rejected(self):
        for key in ('device','inode','size','mtime_ns','ctime_ns'):
            with self.subTest(key=key):
                f,v,i=fixture(True);path=prepare.INHERITED_PLAN['path']
                i[path][key]+=1
                with self.assertRaises(ValueError):build(f,v,i)

    def test_no_new_oversize_input_or_dropped_seed_input(self):
        f,v,i=fixture(True);seed=v[f['seed_spec']['path']];path=prepare.INHERITED_PLAN['path']
        row=seed['input_extension'][path];ref={k:prepare.INHERITED_PLAN[k]for k in ('path','sha256')}
        with self.assertRaises(ValueError):prepare.retained_input_caps({path:row},{},{path:row},ref,v[path])
        with self.assertRaises(ValueError):prepare.retained_input_caps({}, {path:row},{path:row},ref,v[path])
        seed['input_extension']['/synthetic/other.json']=dict(sha256='a'*64,identity=dict(i[path],realpath='/synthetic/other.json'))
        with self.assertRaisesRegex(ValueError,'original8MiB'):build(f,v,i)

    def test_executed_preparer_requires_both_actual_and_frozen_pin(self):
        f,v,_=fixture();helper=v[f['preparer_delivery']['path']]
        helper['source_sha256'][str(prepare.SELF)]=hashlib.sha256(prepare.SELF.read_bytes()).hexdigest()
        prepare.verify_executed_preparer(helper)
        for name in (str(prepare.SELF),prepare.PREDECESSOR_PREPARER['path']):
            wrong=copy.deepcopy(helper);wrong['source_sha256'][name]='0'*64
            with self.assertRaises(ValueError):prepare.verify_executed_preparer(wrong)


class RendererTests(unittest.TestCase):
    def test_exact_frozen_predecessors_reproduce_all_three_helpers(self):
        sources={name:(Q/name).read_text()for name in renderer.BASE_SHA256}
        for name,text in renderer.render(sources).items():self.assertEqual(text,(Q/name).read_text())

    def test_changed_missing_or_extra_predecessor_rejected(self):
        for mode in ('changed','missing','extra'):
            sources={name:(Q/name).read_text()for name in renderer.BASE_SHA256}
            if mode=='changed':sources[next(iter(sources))]+='\n'
            elif mode=='missing':sources.pop(next(iter(sources)))
            else:sources['unexpected.py']=''
            with self.subTest(mode=mode),self.assertRaises(ValueError):renderer.render(sources)


class AuditTests(unittest.TestCase):
    def observe(self,report,artifacts):
        root=serial.WORK.parent;values={str(serial.WORK/'report.json'):report}
        for info in artifacts:values[str(serial.WORK/('artifact-'+info['name']+'.json'))]=info
        existing={str(serial.WORK),*values}
        def ref(path,*args):return dict(path=str(path),sha256='1'*64)
        with patch.object(Path,'exists',lambda p:str(p)in existing),\
             patch.object(auditor,'ref',side_effect=ref),\
             patch.object(auditor,'read',side_effect=lambda r,*a:copy.deepcopy(values[r['path']])),\
             patch.object(auditor.stage,'artifact_record',side_effect=lambda info,*a:info),\
             patch.object(serial.protocol,'verify_sources',return_value=serial.PINS):
            return auditor.observe_layouts(root,{'process':{}},{'exit_code':1},{'failed':True},lambda:None)

    def test_actual_hlo_failure_boundary_keeps_graph_without_receipt(self):
        report=serial.prefix(proof=False);report['error']='synthetic HLO rejection'
        result=self.observe(report,[]);row=result['layouts'][0]
        self.assertEqual((row['compiled_graphs'],row['artifact_receipts'],row['completed_calls']),(1,0,0))
        self.assertEqual(row['graphs_without_receipt'],1);self.assertTrue(row['pending_hlo_proof'])
        self.assertEqual(row['failure'],report['error']);self.assertIsNone(row['summary'])
        self.assertEqual(result['owners'],[])

    def test_second_graph_failure_keeps_three_completed_calls(self):
        report=serial.prefix(calls=3,returned=3,started=3,graphs=2,proof=False)
        row=self.observe(report,[serial.protocol.receipt(report['graphs'][0])])['layouts'][0]
        self.assertEqual((row['compiled_graphs'],row['artifact_receipts'],row['completed_calls']),(2,1,3))
        self.assertIsNone(row['summary'])


class PublicationTests(unittest.TestCase):
    def publication(self,**changes):
        f,v,_=fixture();old=v[f['catalog']['path']]
        ended=dict(diagnostic_completed=False,geometry_qualified=False,full_model_qualified=False,new_speedup_claim=False,
            running_handoff=dict(path='/synthetic/running.json',sha256='1'*64),completed_calls=0,retained_mismatch_calls=0,
            owners=[],units=[],lock_identity=serial.contract.LOCK_IDENTITY)
        ended.update(changes)
        actual=dict(kind='tool_session',session_id=100,terminal_chunk='abc123',exit_code=1)
        audit_tool=dict(kind='tool_session',session_id=101,terminal_chunk='def456',exit_code=0)
        result=helpers.publication_values(old,{},dict(current_serial_catalog=f['catalog']),ended,
            dict(path='/synthetic/actual-exit.json',sha256='2'*64),None,actual,audit_tool,
            dict(idle='/synthetic/idle.json',catalog='/synthetic/catalog.json',receipt='/synthetic/receipt.json'),
            {'/synthetic/new-source.py':'3'*64},{'synthetic':True},[{},{}],'synthetic-time')
        return old,result,actual,audit_tool

    def test_failed_hlo_diagnostic_appends_one_true_row_and_retains_auditor(self):
        old,(values,refs),actual,audit_tool=self.publication()
        self.assertEqual(values['catalog']['runs'][:-1],old['runs'])
        self.assertEqual(values['catalog']['runs'][-1]['tool'],actual)
        self.assertEqual(values['receipt']['actual_independent_auditor_tool'],audit_tool)
        current=values['idle']['qwen35_kv256_s16_pair_current']
        self.assertEqual(current['status'],'failed');self.assertIsNone(current['completion'])
        self.assertIs(values['receipt']['no_post_run_source_retrofit'],True)
        self.assertIs(values['receipt']['new_speedup_claim'],False)
        self.assertEqual(values['catalog']['current_handoff'],refs['idle'])

    def test_diagnostic_cannot_promote_geometry_model_or_speedup(self):
        for key in ('geometry_qualified','full_model_qualified','new_speedup_claim'):
            with self.subTest(key=key),self.assertRaises(ValueError):self.publication(**{key:True})

    def test_all_five_helpers_must_be_preenrolled(self):
        refs={str(p):dict(path=str(p),sha256='a'*64)for p in helpers.HELPERS}
        spec=dict(source_sha256={p:r['sha256']for p,r in refs.items()})
        lookup=lambda p:refs[str(p)]
        self.assertEqual(len(helpers.pinned_helpers(spec,lookup)),5)
        for path in refs:
            wrong=copy.deepcopy(spec);wrong['source_sha256'].pop(path)
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'no source retrofit'):
                helpers.pinned_helpers(wrong,lookup)

    def test_actual_tool_requires_complete_observed_identity(self):
        valid=dict(kind='tool_session',session_id=1,terminal_chunk='abc123',exit_code=1)
        self.assertEqual(helpers.actual_tool(valid),valid)
        for key in valid:
            wrong=copy.deepcopy(valid);wrong.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):helpers.actual_tool(wrong)

    def test_no_native_module_imported(self):self.assertFalse(serial.BLOCKED.intersection(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
