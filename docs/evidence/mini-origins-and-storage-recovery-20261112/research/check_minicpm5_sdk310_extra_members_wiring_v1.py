"""Source, fixed saved JSON and synthetic ports only; no native or archive work."""
import ast
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

DENY={'ctypes','_ctypes','torch','numpy','nki','neuronxcc','torch_neuronx','torch_xla'}
class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in DENY:raise AssertionError('Source-only check forbids '+fullname)
sys.meta_path.insert(0,DenyNative())
import minicpm5_sdk310_extra_members_binding_v1 as b
import prepare_minicpm5_sdk310_extra_members_plan_v1 as builder
from check_minicpm5_sdk310_stdlib_extra_members_v1 import Tests as HelperTests, synthetic

Q=b.c.Q


def manifest():
    old=b.c.load(b.BASE);pins=old['source_sha256'].copy()
    pins.update({str(Q/name):hashlib.sha256((Q/name).read_bytes()).hexdigest() for name in b.NEW_NAMES})
    return dict(schema='minicpm5_sdk310_extra_members_source_manifest_v1',base_plan=b.BASE,
                helper_review=b.HELPER_REVIEW,source_sha256=pins,budget=b.c.INVENTORY,
                mini_scientific_constraints=b.helper.CONSTRAINTS)


def reference(name):
    ref,_=b.c.reference(Q/name);return {k:ref[k] for k in ('path','sha256')}


def functions(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}


class WiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request=b.helper.prepare()
        cls.synthetic_request,inspection,_,_=synthetic(cls.request)
        test=HelperTests('test_09_collect_is_one_inspection_two_reads_and_preserves_nonpromotion')
        transport,_=test.fake_transport(cls.synthetic_request,inspection)
        with patch.object(b.helper,'prepare',return_value=cls.synthetic_request),patch.dict(sys.modules,{'dory_deb_transport_v3':transport}):
            cls.result=b.helper.collect(lambda:None,lambda event:None)
        cls.manifest=manifest()

    def test_01_complete_original_sources_and_resource_limits(self):
        self.assertIs(b.validate_sources_manifest(self.manifest),self.manifest)
        self.assertEqual(len(self.manifest['source_sha256']),112)
        self.assertEqual(b.c.verify_sources(self.manifest['source_sha256'])>0,True)
        self.assertEqual(b.c.INVENTORY['output_bytes'],128*2**20)
        self.assertEqual(b.c.INVENTORY['root_floor_bytes'],28*2**30)
        self.assertEqual(b.c.INVENTORY['nvme_floor_bytes'],30*2**30)

    def test_02_changed_or_omitted_retained_source_and_budget_are_rejected(self):
        for mutate in (lambda m:m['source_sha256'].pop(next(iter(m['source_sha256']))),
                       lambda m:m['source_sha256'].update({next(iter(m['source_sha256'])):'0'*64}),
                       lambda m:m['budget'].update(wall_seconds=901),
                       lambda m:m['mini_scientific_constraints'].update(native_checks=8)):
            value=copy.deepcopy(self.manifest);mutate(value)
            with self.assertRaises(RuntimeError):b.validate_sources_manifest(value)

    def test_03_supplemental_full_result_validates_without_eight_package_schema(self):
        self.assertIs(b.validate_result(self.result,self.synthetic_request),self.result)
        self.assertNotIn('packages',self.result)
        self.assertEqual(self.result['prior_target_count'],12)
        self.assertEqual(len(self.result['comparisons']),2)

    def test_04_identity_and_association_tampering_rejected(self):
        mutations=[lambda x:x['comparisons'][0]['current_identity'].update(ino=1),
                   lambda x:x['comparisons'][0]['current_identity'].update(dev=os.makedev(1,1)),
                   lambda x:x['comparisons'][0]['current'].update(sha256='0'*64),
                   lambda x:x['comparisons'][0]['member'].update(sha256='0'*64),
                   lambda x:x['comparisons'].reverse(),
                   lambda x:x['comparisons'][0].update(observation_source={}),
                   lambda x:x['comparisons'][0]['current_identity'].update(nlink=2)]
        for mutate in mutations:
            value=copy.deepcopy(self.result);mutate(value)
            with self.assertRaises(RuntimeError):b.validate_result(value,self.synthetic_request)

    def test_05_model_or_historical_promotion_and_count_relabel_are_rejected(self):
        mutations=[lambda x:x.update(full_model_qualified=True),lambda x:x.update(prior_target_count=14),
                   lambda x:x.update(download_performed=True),lambda x:x.update(actual_process_exit_claimed=True),
                   lambda x:x['comparisons'][0].update(historical_full_file_content_proven=True),
                   lambda x:x['comparisons'][0].update(original_matches=True)]
        for mutate in mutations:
            value=copy.deepcopy(self.result);mutate(value)
            with self.assertRaises(RuntimeError):b.validate_result(value,self.synthetic_request)

    def events(self):
        r=self.result;request=self.synthetic_request
        ref=dict(path='/synthetic/extra-members-result.json',bytes=123,sha256='1'*64)
        decoder=r['inspection']['decompressor']
        events=[dict(event='supplemental_archive_replay_start',package=b.helper.PACKAGE,archive=request['archive']),
                dict(event='dpkg_deb_start',argv=decoder['argv']),
                dict(event='dpkg_deb_complete',package=b.helper.PACKAGE,receipt=decoder),
                dict(event='minicpm5_extra_members_complete',result=ref,prior_packages=8,prior_targets=12,
                     supplemental_packages=1,supplemental_targets=2,current_matches=2)]
        return events,ref

    def test_06_full_four_event_success_and_mutations(self):
        events,ref=self.events()
        self.assertEqual(b.validate_success(events,self.synthetic_request,self.result,ref)['events'],4)
        for mutate in (lambda x:x.pop(),lambda x:x.append(x[0]),lambda x:x.reverse(),
                       lambda x:x[2].update(package='wrong'),lambda x:x[3].update(prior_targets=14),
                       lambda x:x[0]['archive'].update(sha256='0'*64)):
            value=copy.deepcopy(events);mutate(value)
            with self.assertRaises(RuntimeError):b.validate_success(value,self.synthetic_request,self.result,ref)

    def test_07_run_passes_exact_guard_and_emitter_to_helper(self):
        calls=[];events=[]
        def guard():calls.append('guard')
        # Bound methods are new objects on each attribute access; retain one.
        emit=events.append
        def collect(actual_guard,actual_emit):
            self.assertIs(actual_guard,guard);self.assertIs(actual_emit,emit);guard()
            return copy.deepcopy(self.result)
        ref=dict(path='/synthetic/extra-members-result.json',bytes=123,sha256='1'*64)
        with patch.object(b,'authenticated',return_value=self.synthetic_request),patch.object(b.helper,'collect',side_effect=collect),patch.object(b.c,'save',return_value=ref) as save:
            value=b.run({},guard,emit,Path('/synthetic'))
        self.assertEqual(value,dict(result=ref));self.assertGreaterEqual(len(calls),3)
        self.assertEqual(save.call_args.args[0],Path(ref['path']))
        self.assertEqual(save.call_args.args[2],b.helper.METADATA_CAP)

    def test_08_failed_admission_guard_prevents_leaf(self):
        def denied():raise RuntimeError('synthetic missing retained descriptor')
        with patch.object(b.helper,'collect') as leaf:
            with self.assertRaisesRegex(RuntimeError,'retained descriptor'):
                b.run({'extra_member_metadata':b.helper.REFS},denied,lambda row:None,Path('/synthetic'))
            leaf.assert_not_called()

    def test_09_parent_and_worker_guards_drain_ack_limits_preserved_in_AST(self):
        old=functions(Q/'run_minicpm5_sdk310_snapshot_archive_v2.py')
        new=functions(Q/'run_minicpm5_sdk310_extra_members_v1.py')
        for name in ('parent_main','worker_main','recorder_main'):
            def named(tree,n):return next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==n)
            self.assertEqual(ast.dump(named(old[name],'guard')),ast.dump(named(new[name],'guard')))
        # Parent differs only in the literal worker file; every other AST node matches.
        class Normalize(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value=='run_minicpm5_sdk310_extra_members_v1.py':node.value='run_minicpm5_sdk310_snapshot_archive_v2.py'
                return node
        self.assertEqual(ast.dump(old['parent_main']),ast.dump(Normalize().visit(copy.deepcopy(new['parent_main']))))
        self.assertEqual(ast.dump(old['main']),ast.dump(new['main']))
        self.assertEqual(ast.dump(ast.Module(body=old['parent_main'].body[-1].finalbody,type_ignores=[])),
                         ast.dump(ast.Module(body=new['parent_main'].body[-1].finalbody,type_ignores=[])))

    def build_fixture(self,mutate=lambda x:None):
        catalog_ref=reference('minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v87.json')
        catalog=b.c.load(catalog_ref);predecessor=catalog['runs'][-1]['terminal']
        manifest_ref=dict(path=str(Q/'synthetic-extra-members-manifest.json'),sha256='1'*64)
        acceptance_ref=dict(path=str(Q/'synthetic-extra-members-root-acceptance.json'),sha256='2'*64)
        accepted=dict(reviewer='/root',passed=True,blockers=[],manifest=manifest_ref,source_sha256=self.manifest['source_sha256'])
        docs={manifest_ref['path']:self.manifest,acceptance_ref['path']:accepted,
              catalog_ref['path']:catalog,catalog['current_handoff']['path']:b.c.load(catalog['current_handoff'])}
        docs=copy.deepcopy(docs);mutate(docs)
        def read(ref):return copy.deepcopy(docs[ref['path']]) if ref['path'] in docs else b.c.load(ref)
        return builder.build(read,catalog_ref,predecessor,manifest_ref,acceptance_ref,
                             '/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-extra-members-synthetic')

    def test_10_saved_JSON_history_composition_preserves_all_actual_rows_and_children(self):
        plan=self.build_fixture()
        self.assertEqual(len(plan['source_sha256']),112)
        self.assertFalse(plan['runtime_admitted']);self.assertFalse(plan['automatic_successor_launch'])
        self.assertEqual(plan['predecessor_expected_exit_code'],1)
        old=b.c.load(b.BASE);scope=b.c.load(old['r2_predecessor_scope'])
        self.assertTrue(all(child in plan['predecessor_owners'] for child in scope['children']))
        self.assertTrue(all(k not in plan for k in ('archive_mode','archive_inputs','replay_from')))
        self.assertEqual(plan['extra_member_metadata'],b.helper.REFS)

    def test_11_running_handoff_or_changed_catalog_prefix_rejected(self):
        for what in ('running','history'):
            def mutate(docs):
                if what=='running':
                    key=next(k for k in docs if 'continuation-handoff-' in k);docs[key]['running']=True
                else:
                    key=next(k for k in docs if 'root-catalog-' in k);docs[key]['runs'][0]['tool']['exit_code']=99
            with self.assertRaises(RuntimeError):self.build_fixture(mutate)

    def test_12_existing_helper_and_supplemental_leaf_have_no_download_or_install(self):
        tree=ast.parse(Path(b.__file__).read_text())
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertEqual(calls.count('helper.collect'),1)
        self.assertFalse(any(x in calls for x in ('subprocess.Popen','io.download','os.system','runtime.acquire')))
        self.assertFalse(any(name.split('.')[0] in DENY for name in sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
