"""Saved metadata and source/mock wiring only; no current binding or native IO."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
from research import check_minicpm5_completed_qwen_catalog_v1 as old_checks
from research import minicpm5_completed_qwen_package_text_v2 as completed
from research import minicpm5_fixed_input_pv_catalog_auth_v2 as runtime
from research import prepare_minicpm5_pv_sdk_import_catalog_v3 as preparation
from research import prepare_minicpm5_catalog_formal_binding_v3 as wiring
from research import minicpm5_sdk_mapping_outer_source_entry_v9 as outer
from research import prepare_minicpm5_formal_r5_metadata_root_v1 as root_prepare

MEASUREMENTS={}
read,text,ref=old_checks.read,old_checks.text,old_checks.ref
CURRENT=dict(path=str(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v105.json'),
    sha256='d24e0eab8002ee92bd6824d60db5fae0f571ebef488261e3ead15e3691cfae11')


def extension():
    old=read(ref(Q/'minicpm5-formal-import-r4-20261112-original-preparation.json'))
    return dict(old['pv_spec_candidate']['source_extension'],**completed.SLOT_TEXT)


class Checks(unittest.TestCase):
    def test_actual_saved_v104_and_append_only_v105_support_exact_two_new_paths(self):
        for catalog in (completed.SLOT_CATALOG,CURRENT):
            pins=extension()
            with self.assertRaisesRegex(ValueError,'Only exact ended Qwen'):
                completed.previous.extension_shape(pins,completed_lfm_text=old_checks.original_lfm(),
                    catalog_ref=catalog,read=read)
            self.assertIs(completed.extension_shape(pins,completed_lfm_text=old_checks.original_lfm(),
                catalog_ref=catalog,read=read),pins)
        self.assertEqual(len(read(completed.SLOT_CATALOG)['runs']),153)
        self.assertEqual(len(read(CURRENT)['runs']),154)
        for path,sha in completed.SLOT_TEXT.items():self.assertEqual(hashlib.sha256(text(path,sha)).hexdigest(),sha)
    def test_reject_partial_changed_or_arbitrary_external_text(self):
        for change in ('partial','sha','unknown_tmp','unknown_repository','ceiling'):
            pins=extension()
            if change=='partial':pins.pop(next(iter(completed.SLOT_TEXT)))
            elif change=='sha':pins[next(iter(completed.SLOT_TEXT))]='f'*64
            elif change=='unknown_tmp':pins['/tmp/unreviewed.py']='f'*64
            elif change=='unknown_repository':pins['/home/ubuntu/src/unreviewed.py']='f'*64
            else:pins.update({str(Q/('extra-%d.py'%i)):'a'*64 for i in range(2049)})
            with self.subTest(change=change),self.assertRaises(ValueError):
                completed.extension_shape(pins,completed_lfm_text=old_checks.original_lfm(),catalog_ref=CURRENT,read=read)
    def test_actual_terminal_and_complete_catalog_mutations_rejected(self):
        for change in ('prefix','missing_row','missing_terminal','owner','tool','active','pin','promoted'):
            def changed_read(reference):
                if change=='missing_terminal' and reference==completed.SLOT_ENDED[0]['terminal']:raise FileNotFoundError()
                value=read(reference)
                if reference==CURRENT:
                    if change=='prefix':value['runs'][0]['units'].append('unknown.service')
                    if change=='missing_row':value['runs'].pop(151)
                if reference==completed.SLOT_CATALOG and change=='pin':
                    value['runs'][151]['source_sha256'][completed.SLOT_ENDED[0]['path']]='f'*64
                if reference==completed.SLOT_ENDED[1]['terminal']:
                    if change=='owner':value['owners']=[]
                    if change=='tool':value['actual_tool']['exit_code']=1
                    if change=='active':value['physically_drained']=False
                    if change=='promoted':value['full_model_qualified']=True
                return value
            with self.subTest(change=change),self.assertRaises((ValueError,FileNotFoundError)):
                completed.extension_shape(extension(),completed_lfm_text=old_checks.original_lfm(),
                    catalog_ref=CURRENT,read=changed_read)
    def test_shared_auth_and_preparation_keep_original_bodies_and_outer_all_gates(self):
        replacements=[('minicpm5_fixed_input_pv_catalog_auth_v1.py','minicpm5_fixed_input_pv_catalog_auth_v2.py',
                       'minicpm5_completed_qwen_package_text_v1','minicpm5_completed_qwen_package_text_v2'),
                      ('prepare_minicpm5_pv_sdk_import_catalog_v2.py','prepare_minicpm5_pv_sdk_import_catalog_v3.py',
                       'minicpm5_completed_qwen_package_text_v1','minicpm5_completed_qwen_package_text_v2'),
                      ('minicpm5_sdk_mapping_outer_source_entry_v8.py','minicpm5_sdk_mapping_outer_source_entry_v9.py',
                       'minicpm5_fixed_input_pv_catalog_auth_v1','minicpm5_fixed_input_pv_catalog_auth_v2')]
        for old,new,a,b in replacements:
            self.assertEqual((Q/old).read_text(),(Q/new).read_text().replace(b,a))
        self.assertIs(preparation.completed,completed);self.assertIs(runtime.completed,completed)
        self.assertIs(outer.catalog_auth,runtime)
    def test_root_prep_interface_uses_explicit_append_only_inputs_and_only_drafts(self):
        original,_,_=old_checks.fixture_original()
        delivery=dict(schema='minicpm5_completed_slot_catalog_source_delivery_v1',source_only=True,
            runtime_binding_created=False,implementation_sources=[])
        for name in root_prepare.NEW_SOURCES:
            row=ref(Q/name);row['bytes']=len(text(row['path'],row['sha256']));delivery['implementation_sources'].append(row)
        delivery_ref=dict(path=str(Q/'source-test-slot-delivery.json'),sha256=old_checks.digest(delivery))
        handoff=dict(complete_current_serial_catalog_next=CURRENT['path'])
        handoff_ref=dict(path=str(Q/'source-test-root-handoff.json'),sha256=old_checks.digest(handoff))
        inputs=dict(catalog=CURRENT,handoff=handoff_ref,predecessor=read(CURRENT)['runs'][-1]['terminal'],
            handoff_catalog_field='complete_current_serial_catalog_next',
            artifact_prefix=str(Q/'minicpm5-formal-import-root-source-test'),
            output_root=str(Q/'minicpm5-fixed-input-pv-sdk-import-root-source-test'),
            attempt_id='b'*32,unit='trn1-mini-root-source-test',source_delivery=delivery_ref)
        def current_read(reference):
            if reference==delivery_ref:return copy.deepcopy(delivery)
            if reference==handoff_ref:return copy.deepcopy(handoff)
            return read(reference)
        artifacts=root_prepare.assemble_candidates(inputs,read=current_read,text=text,guard=Mock(),original=original)
        receipt=artifacts['preparation-receipt']['value']
        self.assertEqual(receipt['catalog_rows'],154);self.assertTrue(receipt['reviews_still_pending'])
        self.assertFalse(receipt['runtime_admitted']);self.assertFalse(receipt['launch_authorized'])
        self.assertEqual(artifacts['outer-entry-plan']['value']['entry_module'],'research.minicpm5_sdk_mapping_outer_source_entry_v9')
        for name,row in artifacts.items():
            payload=wiring.launch.encoded(row['value'])
            self.assertEqual(len(payload),row['bytes']);self.assertEqual(hashlib.sha256(payload).hexdigest(),row['reference']['sha256'])
            if name.endswith('review-draft'):
                self.assertFalse(row['value']['passed']);self.assertIsNone(row['value']['reviewer'])
        self.assertEqual(receipt['original_nine_native_checks_pending'],9)
        self.assertEqual(receipt['original_scientific_rows_still_pending'],1284)
        self.assertIn(str(Q/'prepare_minicpm5_formal_r5_metadata_root_v1.py'),artifacts['complete-source-map']['value'])
    def test_full_saved154_wiring_preserves_all_sources_original_literal_and_science(self):
        original,old_request,base=old_checks.fixture_original()
        old_request.update(current_serial_catalog=CURRENT,
            output_root=str(Q/'minicpm5-fixed-input-pv-sdk-import-source-test-v105'))
        request=read(ref(Q/'minicpm5-formal-import-r4-20261112-request.json'))
        catalog=read(CURRENT)
        original_ref=dict(path=str(Q/'source-test-original-request.json'),sha256=old_checks.digest(old_request))
        handoff=dict(complete_current_serial_catalog_next=CURRENT['path'])
        handoff_ref=dict(path=str(Q/'source-test-handoff.json'),sha256=old_checks.digest(handoff))
        request.update(original_request=original_ref,current_handoff=handoff_ref,
            expected_predecessor_exit=catalog['runs'][-1]['terminal'],
            artifact_prefix=str(Q/'minicpm5-formal-import-source-test-v105'),attempt_id='a'*32,
            unit='trn1-mini-formal-source-test-v105')
        def current_read(reference):
            if reference==original_ref:return copy.deepcopy(old_request)
            if reference==handoff_ref:return copy.deepcopy(handoff)
            return read(reference)
        additions=[ref(Q/name) for name in wiring.SOURCES]
        result=wiring.prepare(request,literal_source_manifest=ref(Q/'minicpm5-formal-literal-source-manifest-20261112-v1.json'),
            preparation_source=ref(Q/'prepare_minicpm5_literal_formal_binding_v1.py'),catalog_source_refs=additions,
            read=current_read,text=text,guard=Mock(),original=original)
        artifacts=result['artifacts'];pins=artifacts['complete-source-map']['value']
        sdk=artifacts['sdk-entry-template']['value'];plan=artifacts['outer-entry-plan']['value'];bootstrap=plan['bootstrap_sources']
        self.assertEqual(result['catalog_rows'],len(catalog['runs']))
        self.assertTrue(all(pins.get(p)==s for row in catalog['runs'] for p,s in row['source_sha256'].items()))
        self.assertTrue(all(pins.get(p)==s for p,s in base['plan']['source_sha256'].items()))
        self.assertEqual(sdk['bootstrap_sources'],bootstrap);self.assertEqual(artifacts['outer-spec']['value']['source_extension'],bootstrap)
        self.assertEqual(plan['entry_module'],'research.minicpm5_sdk_mapping_outer_source_entry_v9')
        self.assertEqual(sdk['entry_module'],'research.minicpm5_sdk_mapping_worker_source_entry_v6')
        pending={str(Q/name) for name in wiring.RUNTIME};visited=set()
        while pending:
            path=pending.pop()
            if path in visited:continue
            visited.add(path);self.assertIn(path,bootstrap)
            self.assertEqual(hashlib.sha256(text(path,bootstrap[path])).hexdigest(),bootstrap[path])
            for node in ast.walk(ast.parse(text(path,bootstrap[path]))):
                if isinstance(node,ast.ImportFrom) and node.module=='research':
                    pending.update(str(Q/(item.name+'.py')) for item in node.names)
                elif isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('research.'):
                    pending.add(str(Q.parent/(node.module.replace('.','/')+'.py')))
        for row in artifacts.values():
            payload=wiring.launch.encoded(row['value'])
            self.assertEqual(len(payload),row['bytes']);self.assertEqual(hashlib.sha256(payload).hexdigest(),row['reference']['sha256'])
        self.assertEqual(result['original_nine_native_checks_pending'],9)
        self.assertEqual(result['original_scientific_rows_still_pending'],1284)
        self.assertFalse(result['runtime_admitted']);self.assertFalse(result['launch_authorized'])
        literal=text(result['entry_source']['path'],result['entry_source']['sha256'])
        self.assertEqual(wiring.launch.launch(literal,result['entry_source'],plan,result['outer_entry_plan'])[7],literal.decode())
        self.assertEqual(artifacts['pv-spec']['value']['limits'],old_checks.logic.LIMITS)
        self.assertEqual(artifacts['pv-spec']['value']['artifact_limits'],old_checks.logic.CAPS)
        MEASUREMENTS.update(catalog_rows=result['catalog_rows'],complete_source_files=len(pins),
            bootstrap_source_files=len(bootstrap),bootstrap_import_closure=len(visited),
            source_release_bytes=result['original_preparation']['complete_source_release_bytes'],
            original_reconstruct_mocked=True,current_binding_written=False)


if __name__=='__main__':
    outcome=unittest.main(exit=False)
    print(json.dumps(MEASUREMENTS,sort_keys=True))
    assert not old_checks.BLOCKED.intersection(sys.modules)
    raise SystemExit(not outcome.result.wasSuccessful())
