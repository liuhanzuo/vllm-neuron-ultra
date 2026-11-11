"""Real authenticated original88 metadata regression plus narrow source closure.

No builder/completed/physical/bundle/native/payload operations. The original
historical JSON is read with full file SHA and a32MiB cap; negative fixtures are
copies in memory, never replacement production records.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research';sys.path[:0]=[str(ROOT),str(Q)]
from research import minicpm5_layer1_sdk_batch_layout_completion_v3 as c
from research import minicpm5_layer1_sdk_batch_layout_post_exit_core_v3 as core
from research import check_minicpm5_layer1_sdk_batch_layout_recovered_closure_v1 as previous_checks
from research import check_minicpm5_layer1_sdk_batch_layout_post_exit_core_v1 as original_checks
from research import prepare_minicpm5_original88_python_trace_20261107_v3 as science_source
COLLECTOR=dict(path=str(Q/'minicpm5-layer1-collector-r2-20261110-spec.json'),
    sha256='906caf171b18b17a9ca02da1365fe4205ae7f63fc6963cec22b7d752e018ef46')
MAPPING={
 'minicpm5_layer1_sdk_batch_layout_cpu_runner_v3':'minicpm5_layer1_sdk_batch_layout_cpu_runner_v4',
 'minicpm5_layer1_sdk_batch_layout_cpu_worker_v3':'minicpm5_layer1_sdk_batch_layout_cpu_worker_v4',
 'research/minicpm5_layer1_sdk_batch_layout_post_exit_core_v2':'research/minicpm5_layer1_sdk_batch_layout_post_exit_core_v3',
 'research/prepare_minicpm5_layer1_sdk_batch_layout_v6':'research/prepare_minicpm5_layer1_sdk_batch_layout_v7',
 'research/record_minicpm5_layer1_sdk_batch_layout_exit_v2':'research/record_minicpm5_layer1_sdk_batch_layout_exit_v3',
 'research/audit_minicpm5_layer1_sdk_batch_layout_result_v3':'research/audit_minicpm5_layer1_sdk_batch_layout_result_v4',
 'research/record_minicpm5_layer1_sdk_batch_layout_audit_exit_v3':'research/record_minicpm5_layer1_sdk_batch_layout_audit_exit_v4',
}
REPLACEMENTS={a.split('/')[-1]:b.split('/')[-1]for a,b in MAPPING.items()}
REPLACEMENTS['minicpm5_layer1_sdk_batch_layout_completion_v2']='minicpm5_layer1_sdk_batch_layout_completion_v3'


class ActualOriginal88(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values={};cls.references={};cls.read_bytes=0
        def read(ref):
            if set(ref)!={'path','sha256'}:raise AssertionError('Only normalized path/file-hash refs may reach transport')
            path=Path(ref['path'])
            if path.suffix!='.json' or not path.is_absolute():raise AssertionError('Metadata JSON only')
            before=path.stat();assert before.st_size<=32*2**20
            raw=path.read_bytes();after=path.stat()
            assert (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
            assert hashlib.sha256(raw).hexdigest()==ref['sha256']
            value=json.loads(raw);cls.values[ref['path']]=value;cls.references[ref['path']]=dict(ref);cls.read_bytes+=len(raw)
            return value
        cls.collector=read(COLLECTOR)
        cls.actual=c.original88_bound(cls.collector,read)
        cls.spec_path=cls.values[cls.collector['finalization']['path']]['references']['execution_spec']['path']
        cls.plan_path=cls.actual['spec']['input_plan']['path']

    def setUp(self):
        self.patches=[patch.object(c,'completed',side_effect=AssertionError('full completion forbidden')),
            patch.object(c,'physical',side_effect=AssertionError('physical forbidden'))]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop()for p in reversed(self.patches)])

    def check(self,overrides=None):
        overrides=overrides or {};seen=[]
        def read(ref):
            self.assertEqual(set(ref),{'path','sha256'});self.assertEqual(ref,self.references[ref['path']]);seen.append(ref)
            return overrides.get(ref['path'],self.values[ref['path']])
        result=c.original88_bound(self.collector,read)
        return result,seen

    def test_real_no_plan_field_and_three_field_input_plan(self):
        bound,seen=self.check();self.assertNotIn('plan',bound['spec'])
        self.assertEqual(bound['spec']['schema'],'minicpm5_original88_python_trace_execution_spec_v1')
        self.assertEqual(set(bound['spec']['input_plan']),{'path','sha256','canonical_sha256'})
        self.assertEqual(bound['plan'],self.actual['plan']);self.assertEqual(bound['catalog'],self.actual['catalog'])
        self.assertEqual(len(bound['plan']['inputs']),7357);self.assertEqual(len(bound['plan']['writer_contexts']),88)
        self.assertEqual(bound['plan']['remaining_gates'],self.values[science_source.ORIGINAL['path']]['remaining_gates'])
        self.assertEqual(bound['plan']['output_root'],bound['spec']['output_root'])
        self.assertTrue(all(set(ref)=={'path','sha256'}for ref in seen))

    def test_no_fake_fallback_to_old_plan_field(self):
        spec=copy.deepcopy(self.actual['spec']);spec['plan']=dict(path='/synthetic/fake.json',sha256='a'*64)
        self.assertEqual(self.check({self.spec_path:spec})[0]['plan'],self.actual['plan'])

    def test_input_ref_shape_and_canonical_hash(self):
        for kind in ('missing','hash','extra'):
            spec=copy.deepcopy(self.actual['spec'])
            if kind=='missing':spec['input_plan'].pop('canonical_sha256')
            if kind=='hash':spec['input_plan']['canonical_sha256']='0'*64
            if kind=='extra':spec['input_plan']['bytes']=1
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):self.check({self.spec_path:spec})

    def test_actual_execution_namespace_must_match(self):
        spec=copy.deepcopy(self.actual['spec']);spec['output_root']+='/replacement'
        with self.assertRaises(RuntimeError):self.check({self.spec_path:spec})

    def test_scientific_transform_rejects_changed_gates_even_with_new_canonical_hash(self):
        plan=copy.deepcopy(self.actual['plan']);plan['remaining_gates']['quality']='relaxed'
        spec=copy.deepcopy(self.actual['spec']);spec['input_plan']['canonical_sha256']=science_source.digest(plan)
        with self.assertRaises(RuntimeError):self.check({self.spec_path:spec,self.plan_path:plan})

    def test_all_original_input_identities_preserved(self):
        plan=copy.deepcopy(self.actual['plan']);plan['inputs'].pop(next(iter(plan['inputs'])))
        spec=copy.deepcopy(self.actual['spec']);spec['input_plan']['canonical_sha256']=science_source.digest(plan)
        with self.assertRaises(RuntimeError):self.check({self.spec_path:spec,self.plan_path:plan})

    def test_exact_science_and_root_acceptance(self):
        for key in ('science_handoff','scientific_review'):
            spec=copy.deepcopy(self.actual['spec']);spec[key]=dict(path='/synthetic/wrong.json',sha256='a'*64)
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.check({self.spec_path:spec})
        path=science_source.ACCEPT['path'];accepted=copy.deepcopy(self.values[path]);accepted['passed']=False
        with self.assertRaises(RuntimeError):self.check({path:accepted})

    def test_original_finalizer_guards_retained(self):
        path=self.collector['finalization']['path']
        for key in ('passed','all_planned_launches_accounted_for','all_groups_drained'):
            value=copy.deepcopy(self.values[path]);value[key]=False
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.check({path:value})
        path=self.collector['finalizer_exit']['path'];value=copy.deepcopy(self.values[path]);value['actual_exit_observed']=False
        with self.assertRaises(RuntimeError):self.check({path:value})

    def test_only_historical_metadata_and_no_native_imports(self):
        self.assertLess(self.read_bytes,32*2**20)
        self.assertFalse(set(sys.modules)&{'torch','numpy','torch_neuronx','neuronxcc'})
        self.assertEqual(len(self.values),9)


class SourceClosure(unittest.TestCase):
    def test_frozen_completion_other_functions_and_all1308_path_unchanged(self):
        before=ast.parse((Q/'minicpm5_layer1_sdk_batch_layout_completion_v2.py').read_text())
        after=ast.parse((Q/'minicpm5_layer1_sdk_batch_layout_completion_v3.py').read_text())
        a={x.name:x for x in before.body if isinstance(x,ast.FunctionDef)}
        b={x.name:x for x in after.body if isinstance(x,ast.FunctionDef)}
        for name in a:
            if name!='completed':self.assertEqual(ast.dump(a[name]),ast.dump(b[name]),name)
        expected=(Q/'minicpm5_layer1_sdk_batch_layout_completion_v2.py').read_text()
        start=expected.index("    cs=collector['collector_spec'];final=load(cs['finalization']);final_exit=load(cs['finalizer_exit'])")
        end=expected.index('    co=[',start)
        expected=expected[:start]+"    bound=original88_bound(collector['collector_spec'],load)\n"+expected[end:]
        expected=expected.replace("co,[final_exit['owner']])","co,[bound['finalizer_owner']])")
        fn=next(x for x in ast.parse(expected).body if isinstance(x,ast.FunctionDef)and x.name=='completed')
        self.assertEqual(ast.dump(fn),ast.dump(b['completed']))
        self.assertIn('minicpm5_after_nemotron_r12_recovered_failure_v1',ast.unparse(b['completed']))

    def test_runtime_sources_only_import_and_exact_path_successors(self):
        for old,new in MAPPING.items():
            expected=(ROOT/(old+'.py')).read_text()
            for a,b in REPLACEMENTS.items():expected=expected.replace(a,b)
            expected=ast.parse(expected);actual=ast.parse((ROOT/(new+'.py')).read_text())
            if 'prepare_'in old:
                for tree in (expected,actual):tree.body=[n for n in tree.body if not(isinstance(n,ast.FunctionDef)and n.name=='entrypoints')]
            self.assertEqual(ast.dump(expected),ast.dump(actual),new)

    def test_builder_preserves_all_old_entries_and_adds_new(self):
        def entries(name):
            path=Q/name;t=ast.parse(path.read_text());fn=next(x for x in t.body if isinstance(x,ast.FunctionDef)and x.name=='entrypoints')
            env=dict(ROOT=ROOT,Q=Q,Path=Path,__file__=str(path));exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
            return env['entrypoints']()
        old=entries('prepare_minicpm5_layer1_sdk_batch_layout_v6.py');new=entries('prepare_minicpm5_layer1_sdk_batch_layout_v7.py')
        self.assertTrue(set(old)<=set(new))
        for p in new:self.assertTrue(p.is_file(),str(p))
        self.assertIn(Q/Path(__file__).name,new)
        self.assertIn(ROOT/'minicpm5_layer1_sdk_batch_layout_cpu_runner_v4.py',new)


class OriginalPostExitChecks(original_checks.PostExit):
    def setUp(self):
        p=patch.object(original_checks,'c',core);p.start();self.addCleanup(p.stop)
        # Fixture is synthetic and contains no real completion/runtime operation.
        def convert(v):
            if isinstance(v,dict):return {k:convert(x)for k,x in v.items()}
            if isinstance(v,list):return [convert(x)for x in v]
            if isinstance(v,tuple):return tuple(convert(x)for x in v)
            if isinstance(v,str):
                for a,b in REPLACEMENTS.items():v=v.replace(a,b)
            return v
        self.f=convert(previous_checks.fixture())


if __name__=='__main__':unittest.main(verbosity=2)
