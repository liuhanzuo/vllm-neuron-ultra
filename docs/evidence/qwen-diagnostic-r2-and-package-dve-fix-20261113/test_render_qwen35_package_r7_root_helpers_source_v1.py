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

import render_qwen35_package_r7_root_helpers_source_v1 as p
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

if __name__=='__main__':unittest.main(verbosity=2)
