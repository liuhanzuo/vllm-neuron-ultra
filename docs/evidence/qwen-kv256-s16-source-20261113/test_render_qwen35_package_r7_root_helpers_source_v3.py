"""Source/synthetic tests; no actual current state, native imports or execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock,patch

import render_qwen35_package_r7_root_helpers_source_v3 as p
import render_qwen35_package_r7_root_helpers_source_v2 as prior_v2
import render_qwen35_package_r7_root_helpers_source_v1 as prior
Q=Path(str(p.Q))
PAYLOADS={key:(Path(name) if Path(name).is_absolute() else Q/name).read_bytes() for key,(name,_) in p.PINS.items()}
AUDIT_SOURCES={key:Path(name).read_bytes() for key,(name,_) in p.AUDIT_PINS.items()}

def fixture():
    def ref(name):return dict(path=str(Q/name),sha256='a'*64)
    return dict(catalog=ref('minicpm5-operator-post-r2-serial-workloads-root-catalog-20261120-v900.json'),
        idle=ref('continuation-handoff-20261120-v900-synthetic-only-idle.json'),
        running_handoff=str(Q/'continuation-handoff-20261120-v901-qwen35-package-full-r7-running.json'),catalog_rows=200,
        runtime_review=ref('qwen35-package-serial-runtime-v9-root-source-review-20261120-v1.json'),
        preparer_review=ref('qwen35-package-serial-preparer-v8-root-source-review-20261120-v1.json'))

def render(bindings=None,payloads=None,audit_sources=None):
    with patch('builtins.open',side_effect=AssertionError('no file IO')),patch('os.open',side_effect=AssertionError('no fd IO')),\
         patch('os.system',side_effect=AssertionError('no shell')),patch('subprocess.Popen',side_effect=AssertionError('no process')),\
         patch('fcntl.flock',side_effect=AssertionError('no lock')):
        return p.render(PAYLOADS if payloads is None else payloads,fixture() if bindings is None else bindings,AUDIT_SOURCES if audit_sources is None else audit_sources)

def path(key):
    name=Path(p.PINS[key][0]);return p.successor_name(str(name if name.is_absolute() else Q/name))

def source(result,key):return result['files'][path(key)].decode()

def assignment(text,name):
    nodes=[n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    assert len(nodes)==1
    return ast.literal_eval(nodes[0].value)

def definition(text,name):
    node=next(n for n in ast.parse(text).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
    return ast.dump(node,include_attributes=False)

class Renderer(unittest.TestCase):
    def test_six_source_only_outputs_without_io(self):
        out=render();self.assertEqual(len(out['files']),6)
        for key in ('passed','launch_authorized','actual_bindings_checked','actual_prepare_executed','history_resolved','lock_acquired','native_executed'):self.assertFalse(out[key])
        self.assertTrue(out['source_only'])
        self.assertEqual(set(out['files']),{path(key) for key in p.PINS})

    def test_all_frozen_predecessors_required_and_authenticated(self):
        for key in p.PINS:
            altered=copy.deepcopy(PAYLOADS);altered[key]+=b'\n'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'frozen predecessor'):render(payloads=altered)
            altered.pop(key)
            with self.assertRaises(ValueError):render(payloads=altered)

    def test_all_static_audit_sources_required_and_authenticated(self):
        for key in p.AUDIT_PINS:
            altered=copy.deepcopy(AUDIT_SOURCES);altered[key]+=b'\n'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'frozen audit'):render(audit_sources=altered)
            altered.pop(key)
            with self.assertRaises(ValueError):render(audit_sources=altered)

    def test_complete_predecessor_source_byte_reversal(self):
        out=render()
        for key in ('prepare','accept','ack','observer'):
            restored=source(out,key)
            for row in reversed(out['exact_predecessor_reversal'][path(key)]):
                self.assertEqual(restored.count(row['after']),row['count']);restored=restored.replace(row['after'],row['before'])
            self.assertEqual(restored.encode(),PAYLOADS[key])

    def test_budget_original_limits_headroom_and_source_identities_preserved(self):
        out=render();old=json.loads(PAYLOADS['budget']);new=json.loads(out['files'][path('budget')])
        for key in set(old)-{'case','output_root','control_root'}:self.assertEqual(old[key],new[key],key)
        for key in set(old['case'])-{'model_path','artifact','compiler_workdir','compiler_cache'}:self.assertEqual(old['case'][key],new['case'][key],key)
        for key in ('model_path','artifact','compiler_workdir','compiler_cache'):self.assertTrue(new['case'][key].endswith('-20261113-r7'))

    def test_stat_input_original_binary_identities_unchanged(self):
        out=render();old=json.loads(PAYLOADS['stat-inputs']);new=json.loads(out['files'][path('stat-inputs')])
        for key in set(old)-{'case','output_root','run_name','resource_budget'}:self.assertEqual(old[key],new[key],key)
        self.assertEqual(new['resource_budget']['sha256'],hashlib.sha256(out['files'][path('budget')]).hexdigest())

    def test_explicit_binding_and_versioned_review_refs_required(self):
        for key in fixture():
            bad=fixture();del bad[key]
            with self.subTest(key=key),self.assertRaises(ValueError):render(bad)
        for key in ('catalog','idle','runtime_review','preparer_review'):
            for kind in ('sha','path','alias'):
                bad=fixture()
                if kind=='sha':bad[key]['sha256']='guess'
                elif kind=='path':bad[key]['path']='/tmp/unbound.json'
                else:bad[key]['path']=str(Q)+'/../research/'+Path(bad[key]['path']).name
                with self.subTest(key=key,kind=kind),self.assertRaises(ValueError):render(bad)
        for key in ('runtime_review','preparer_review'):
            bad=fixture();bad[key]['path']=bad[key]['path'].replace('-v9-', '-v8-').replace('preparer-v8-', 'preparer-v7-')
            with self.subTest(key=key),self.assertRaises(ValueError):render(bad)

    def test_old_catalog_idle_row_count_and_running_epoch_rejected(self):
        for mode in ('catalog','idle','rows','same','earlier','invalid_date','bool_rows'):
            bad=fixture()
            if mode=='catalog':bad['catalog']['path']=bad['catalog']['path'].replace('-v900.', '-v118.')
            if mode=='idle':bad['idle']['path']=bad['idle']['path'].replace('-v900-', '-v597-')
            if mode=='rows':bad['catalog_rows']=165
            if mode=='bool_rows':bad['catalog_rows']=True
            if mode=='same':bad['running_handoff']=bad['running_handoff'].replace('-v901-', '-v900-')
            if mode=='earlier':bad['running_handoff']=bad['running_handoff'].replace('20261120','20261119')
            if mode=='invalid_date':bad['running_handoff']=bad['running_handoff'].replace('20261120','20261320')
            with self.subTest(mode=mode),self.assertRaises(ValueError):render(bad)

    def test_root_assignments_bind_exact_selected_refs(self):
        out=render()
        for key in ('prepare','accept'):
            refs=assignment(source(out,key),'EXACT_REFS')
            for name,(target,sha) in refs.items():self.assertEqual(dict(path=target,sha256=sha),out['required_actual_refs'][name])
        self.assertIn(Path(fixture()['running_handoff']).name,source(out,'prepare'))
        self.assertIn(Path(fixture()['running_handoff']).name,source(out,'accept'))
        self.assertIn("len(old['runs']) == 200",source(out,'observer'))

    def test_preparer_and_runtime_deliveries_are_fixed(self):
        text=source(render(),'prepare')
        for ref in (p.RUNTIME_DELIVERY,p.PREPARER_DELIVERY):
            self.assertIn(ref['sha256'],text);self.assertIn(Path(ref['path']).name,text)
            actual=Path(ref['path']).read_bytes();self.assertEqual(hashlib.sha256(actual).hexdigest(),ref['sha256'])
        self.assertIn('import prepare_qwen35_package_serial_candidate_v8 as prep',text)

    def test_pre_enrolled_generated_and_static_audit_helpers(self):
        out=render();text=source(out,'prepare');pins=assignment(text,'expected_audit_helper_pins')
        self.assertEqual(len(pins),4);self.assertEqual(pins,out['audit_source_prebinding']['generated_and_static_helpers'])
        for target,sha in pins.items():
            raw=out['files'].get(target)
            if raw is None:raw=AUDIT_SOURCES[next(key for key,(name,_) in p.AUDIT_PINS.items() if name==target)]
            self.assertEqual(hashlib.sha256(raw).hexdigest(),sha);self.assertIn(Path(target).name,text)
        self.assertLess(text.index('expected_audit_helper_pins'),text.index('candidate = prep.prepare'))
        self.assertIn('preparation_sources=sources, required_helpers=[str(p) for p in helpers]',text)
        observer_pins=assignment(source(out,'observer'),'audit_support_pins')
        self.assertEqual(observer_pins,out['audit_source_prebinding']['static_sources'])

    def test_accept_original_bounded_history_and_874_children(self):
        old=PAYLOADS['accept'].decode();new=source(render(),'accept')
        for literal in ('resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))',
            'resource.setrlimit(resource.RLIMIT_CPU, (600, 605)); signal.alarm(600)',
            'lock = io.RetainedLock(); lock.acquire()',"assert len(bound.supplement['scope']['children']) == 874",
            'bound = Bundle(spec_ref, review_ref, guard).resolve()',"assert time.monotonic() - started < 600"):
            self.assertIn(literal,old);self.assertIn(literal,new)
        for key in ('prepare','accept','ack','observer'):self.assertNotIn('qwen35_package_serial_runtime_v8',source(render(),key))

    def test_observer_freshness_definition_and_forced_boundaries_preserved(self):
        old=PAYLOADS['observer'].decode();new=source(render(),'observer')
        for name in ('FreshnessGuard','latest_handoff','collect_scope','evaluate'):
            # Only evaluate can be absent in a predecessor; actual defined names
            # below determine the immutable function equality check.
            if any(isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name for n in ast.parse(old).body):
                self.assertEqual(definition(old,name),definition(new,name))
        self.assertEqual(old.count('guard(force=True)'),new.count('guard(force=True)'))

    def test_adapter_sources_are_only_namespace_successors(self):
        for key,stem in (('post_audit','qwen35_package_post_audit'),('physical','qwen35_package_audit_physical')):
            before=(Q/(stem+'_v1.py')).read_text().replace('qwen35_package_serial_runtime_v8','qwen35_package_serial_runtime_v9').replace('qwen35_package_audit_physical_v1','qwen35_package_audit_physical_v2')
            self.assertEqual(AUDIT_SOURCES[key].decode(),before)

    def test_seed_reference_and_science_remain_original(self):
        text=source(render(),'prepare')
        self.assertIn("at('qwen35-sampler-r10-root-actual-exit-20261112-v1.json')",text)
        self.assertIn("at('qwen35-package-build-storage-source-review-20261112-v1.json')",text)
        self.assertIn('qwen35_package_full_validation_v2/runtime.py',text)



def retained_fixture():
    value=fixture()
    value['catalog']['path']=str(Q/'minicpm5-retained-synthetic-r2-20261120-serial-catalog.json')
    value['idle']['path']=str(Q/'continuation-handoff-mini-retained-synthetic-r2-20261120-complete.json')
    value['running_handoff']=str(Q/'continuation-handoff-qwen35-package-full-r7-20261120-running.json')
    value['catalog_rows']=169
    return value


class RetainedLifecycleRenderer(unittest.TestCase):
    def test_legacy_outputs_identical_to_frozen_renderer(self):
        old=prior.render(PAYLOADS,fixture(),AUDIT_SOURCES);new=render()
        self.assertEqual(new.pop('schema'),'qwen35_package_r7_rendered_source_candidate_v3')
        self.assertEqual(old.pop('schema'),'qwen35_package_r7_rendered_source_candidate_v1')
        self.assertEqual(old,new)

    def test_retained_reproduces_original_rejection_and_new_pure_success(self):
        with self.assertRaisesRegex(ValueError,'complete post-r6 catalog'):
            prior.render(PAYLOADS,retained_fixture(),AUDIT_SOURCES)
        new=render(retained_fixture())
        self.assertEqual(len(new['files']),6)
        for key in ('passed','launch_authorized','actual_bindings_checked','actual_prepare_executed',
                    'history_resolved','lock_acquired','native_executed'):
            self.assertIs(new[key],False)

    def test_retained_injected_refs_and_nonnumeric_running_name_are_exact(self):
        bindings=retained_fixture();new=render(bindings)
        for key in ('prepare','accept'):
            refs=assignment(source(new,key),'EXACT_REFS')
            for name in ('catalog','idle'):
                self.assertEqual(refs[name],(bindings[name]['path'],bindings[name]['sha256']))
            self.assertIn(Path(bindings['running_handoff']).name,source(new,key))
            self.assertNotIn('v597-qwen35-package-full-r7',source(new,key))
        self.assertIn("len(old['runs']) == 169",source(new,'observer'))

    def test_retained_requires_same_completed_lifecycle_and_valid_dated_run(self):
        variations=(
            ('idle','path','continuation-handoff-mini-retained-synthetic-r3-20261120-complete.json'),
            ('idle','path','continuation-handoff-mini-retained-synthetic-r2-20261121-complete.json'),
            ('idle','path','continuation-handoff-mini-retained-synthetic-r2-20261120-replay.json'),
            ('idle','path','continuation-handoff-mini-retained-synthetic-r2-20261120-pair.json'),
            ('idle','path','continuation-handoff-20261120-v999-synthetic-idle.json'),
            ('catalog','path','minicpm5-retained-synthetic-r2-serial-catalog.json'),
            ('catalog','path','minicpm5-retained-synthetic-r2-20261320-serial-catalog.json'),
            ('catalog','path','minicpm5-retained-'+('a'*48)+'-20261120-serial-catalog.json'),
            ('catalog','path','minicpm5-unrecognized-synthetic-r2-20261120-serial-catalog.json'))
        for key,field,name in variations:
            bad=retained_fixture();bad[key][field]=str(Q/name)
            with self.subTest(name=name),self.assertRaises(ValueError):render(bad)

    def test_retained_cannot_mint_numeric_successor_or_wrong_family(self):
        for name in (
            'continuation-handoff-20261120-v999-qwen35-package-full-r7-running.json',
            'continuation-handoff-qwen35-package-full-r6-20261120-running.json',
            'continuation-handoff-qwen35-package-full-r7-20261120-complete.json',
            'continuation-handoff-qwen35-package-full-r7-20261119-running.json',
            'continuation-handoff-qwen35-package-full-r7-20261320-running.json'):
            bad=retained_fixture();bad['running_handoff']=str(Q/name)
            with self.subTest(name=name),self.assertRaises(ValueError):render(bad)

    def test_retained_row_lower_bound_types_dates_and_canonical_paths(self):
        for count in (0,166,167,True,168.0):
            bad=retained_fixture();bad['catalog_rows']=count
            with self.subTest(count=count),self.assertRaises(ValueError):render(bad)
        good=retained_fixture();good['catalog_rows']=168;render(good)
        later=retained_fixture();later['running_handoff']=later['running_handoff'].replace('20261120','20261121');render(later)
        for field in ('catalog','idle','running_handoff'):
            bad=retained_fixture()
            if field=='running_handoff':bad[field]=str(Q)+'/../research/'+Path(bad[field]).name
            else:bad[field]['path']=str(Q)+'/../research/'+Path(bad[field]['path']).name
            with self.subTest(field=field),self.assertRaises(ValueError):render(bad)

    def test_retained_complete_original_helper_reversal_and_budgets(self):
        out=render(retained_fixture());legacy=render()
        for key in ('prepare','accept','ack','observer'):
            restored=source(out,key)
            for row in reversed(out['exact_predecessor_reversal'][path(key)]):
                self.assertEqual(restored.count(row['after']),row['count'])
                restored=restored.replace(row['after'],row['before'])
            self.assertEqual(restored.encode(),PAYLOADS[key])
        for key in ('budget','stat-inputs'):
            self.assertEqual(out['files'][path(key)],legacy['files'][path(key)])
        for literal in ('tip = latest.latest_handoff()',"assert tip['path'] == idle_ref['path']"):
            self.assertIn(literal,source(out,'prepare'))
        for literal in ('tip = latest.latest_handoff()',"assert tip['path'] == catalog['current_handoff']['path']"):
            self.assertIn(literal,source(out,'accept'))
        self.assertIn("catalog['current_handoff'] == candidate['idle_handoff']",source(out,'accept'))
        for name in ('FreshnessGuard','latest_handoff','collect_scope'):
            self.assertEqual(definition(PAYLOADS['observer'].decode(),name),definition(source(out,'observer'),name))

    @staticmethod
    def frozen_latest_function():
        # Compile only the original definition: no historical module imports,
        # current metadata scans, native work or runtime History are executed.
        source_text=(Q/'minicpm5_layer1_operator_sdk_runtime_completion_v4.py').read_text()
        parsed=ast.parse(source_text)
        function=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='latest_handoff')
        namespace=dict(Path=Path,re=__import__('re'),require=p.require,
                       handoff_candidates=lambda:(),identity=lambda path:None,prior=types.SimpleNamespace(Q=Q))
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<frozen-latest-definition>','exec'),namespace)
        return namespace['latest_handoff']

    def test_original_latest_scan_recognizes_retained_using_mtime_not_numeric_suffix(self):
        latest=self.frozen_latest_function()
        numeric=Path(fixture()['idle']['path']);retained=Path(retained_fixture()['idle']['path'])
        identities={str(numeric):dict(mtime_ns=1),str(retained):dict(mtime_ns=2)}
        self.assertEqual(latest(lambda:[numeric,retained],identities.__getitem__),dict(path=str(retained),identity=dict(mtime_ns=2)))
        identities[str(numeric)]['mtime_ns']=3
        self.assertEqual(latest(lambda:[numeric,retained],identities.__getitem__)['path'],str(numeric))

    def test_original_latest_scan_rejects_ambiguous_timestamp_with_retained_name(self):
        latest=self.frozen_latest_function()
        paths=[Path(fixture()['idle']['path']),Path(retained_fixture()['idle']['path'])]
        with self.assertRaisesRegex(ValueError,'Ambiguous latest'):
            latest(lambda:paths,lambda _:dict(mtime_ns=1))



def retirement_fixture():
    value=retained_fixture()
    value['catalog']['path']=str(Q/'minicpm5-retirement-synthetic-r3-20261120-serial-catalog.json')
    value['idle']['path']=str(Q/'continuation-handoff-mini-retirement-synthetic-r3-20261120-complete.json')
    value['catalog_rows']=170
    return value


class RetirementLifecycleRenderer(unittest.TestCase):
    def test_all_numeric_and_retained_outputs_preserve_v2_exactly(self):
        for bindings in (fixture(),retained_fixture()):
            old=prior_v2.render(PAYLOADS,bindings,AUDIT_SOURCES);new=render(bindings)
            self.assertEqual(old.pop('schema'),'qwen35_package_r7_rendered_source_candidate_v2')
            self.assertEqual(new.pop('schema'),'qwen35_package_r7_rendered_source_candidate_v3')
            self.assertEqual(old,new)

    def test_retirement_v2_rejection_and_v3_exact_source_bindings(self):
        bindings=retirement_fixture()
        with self.assertRaises(ValueError):prior_v2.render(PAYLOADS,bindings,AUDIT_SOURCES)
        out=render(bindings)
        self.assertEqual(len(out['files']),6)
        for key in ('prepare','accept'):
            refs=assignment(source(out,key),'EXACT_REFS')
            for name in ('catalog','idle'):
                self.assertEqual(refs[name],(bindings[name]['path'],bindings[name]['sha256']))
            self.assertIn(Path(bindings['running_handoff']).name,source(out,key))
        self.assertIn("len(old['runs']) == 170",source(out,'observer'))
        for key in ('passed','launch_authorized','actual_bindings_checked','actual_prepare_executed','history_resolved','lock_acquired','native_executed'):
            self.assertIs(out[key],False)

    def test_retirement_rejects_mismatched_kind_run_incomplete_phase_and_fake_epoch(self):
        for name in (
            'continuation-handoff-mini-retained-synthetic-r3-20261120-complete.json',
            'continuation-handoff-mini-retirement-synthetic-r2-20261120-complete.json',
            'continuation-handoff-mini-retirement-synthetic-r3-20261121-complete.json',
            'continuation-handoff-mini-retirement-synthetic-r3-20261120-running.json',
            'continuation-handoff-mini-retirement-synthetic-r3-20261120-replay.json'):
            bad=retirement_fixture();bad['idle']['path']=str(Q/name)
            with self.subTest(name=name),self.assertRaises(ValueError):render(bad)
        bad=retained_fixture();bad['idle']['path']=bad['idle']['path'].replace('mini-retained-','mini-retirement-')
        with self.assertRaises(ValueError):render(bad)
        for name in ('continuation-handoff-20261120-v999-qwen35-package-full-r7-running.json',
                     'continuation-handoff-qwen35-package-full-r7-20261119-running.json'):
            bad=retirement_fixture();bad['running_handoff']=str(Q/name)
            with self.subTest(name=name),self.assertRaises(ValueError):render(bad)

    def test_retirement_bounds_and_original_helper_reversal(self):
        for count in (166,167,168,True,169.0):
            bad=retirement_fixture();bad['catalog_rows']=count
            with self.subTest(count=count),self.assertRaises(ValueError):render(bad)
        bindings=retirement_fixture();bindings['catalog_rows']=169;out=render(bindings)
        for key in ('prepare','accept','ack','observer'):
            restored=source(out,key)
            for row in reversed(out['exact_predecessor_reversal'][path(key)]):
                self.assertEqual(restored.count(row['after']),row['count']);restored=restored.replace(row['after'],row['before'])
            self.assertEqual(restored.encode(),PAYLOADS[key])
        old=render(retained_fixture())
        for key in ('budget','stat-inputs'):self.assertEqual(out['files'][path(key)],old['files'][path(key)])
        for kind in ('20261320', '20261112'):
            bad=retirement_fixture()
            for name in ('catalog','idle'):bad[name]['path']=bad[name]['path'].replace('20261120',kind)
            with self.subTest(kind=kind),self.assertRaises(ValueError):render(bad)

    def test_original_latest_scan_accepts_retirement_without_numeric_epoch(self):
        latest=RetainedLifecycleRenderer.frozen_latest_function()
        numeric=Path(fixture()['idle']['path']);retained=Path(retained_fixture()['idle']['path']);retired=Path(retirement_fixture()['idle']['path'])
        identities={str(numeric):dict(mtime_ns=1),str(retained):dict(mtime_ns=2),str(retired):dict(mtime_ns=3)}
        self.assertEqual(latest(lambda:[numeric,retained,retired],identities.__getitem__)['path'],str(retired))
        identities[str(numeric)]['mtime_ns']=4
        self.assertEqual(latest(lambda:[numeric,retained,retired],identities.__getitem__)['path'],str(numeric))

if __name__=='__main__':unittest.main(verbosity=2)
