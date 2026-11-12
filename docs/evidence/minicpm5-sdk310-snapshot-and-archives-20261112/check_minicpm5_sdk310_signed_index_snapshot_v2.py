"""Metadata/pure decoder tests only; no XZ reads, native decode, output or process IO."""
import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import Mock
import prepare_minicpm5_sdk310_signed_index_snapshot_v1 as b
m=b.snapshot
REVIEW=dict(path=str(b.Q/'minicpm5-sdk310-signed-index-snapshot-root-source-review-20261112-v1.json'),sha256='d7e63d20738c3d8d73ac3484c86467ec94e0e9ddb46776b99a79334350062de2')
OUTPUT='/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-index-snapshot-synthetic-test'


def read_with(overrides):
    def read(ref):return copy.deepcopy(overrides[ref['path']]) if ref['path'] in overrides else b.c.load(ref)
    return read


def fixture_result():
    a=m.metadata();rows=[]
    for row in a['rows']:
        rows.append(dict(suite=row['suite'],logical_ref=row['original_logical_index_ref'],
            snapshot_ref=dict(path=str(Path(OUTPUT)/'indexes'/row['proposed_snapshot_basename']),**row['expected_uncompressed']),
            compressed_ref=row['compressed'],release=row['release'],status_text=row['status_text'],saved_signature_proof=row['saved_signature_proof']))
    value=dict(schema=m.SCHEMA,assessment=m.ASSESSMENT,original_inputs=a['frozen_mini_inputs'],signed_metadata=a['signed_metadata'],
        producer_actual_exit=a['producer_actual_exit'],recorder_actual_exit=a['recorder_actual_exit'],rows=rows,raw_total_bytes=m.RAW_TOTAL,
        complete=True,actual_process_exit_claimed=False,system_APT_files_changed=False,native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False)
    return value,a


class Tests(unittest.TestCase):
    def test_exact124_history_and100_sources_no_runtime_plan_write(self):
        plan=b.build(b.c.load,REVIEW,OUTPUT);old=b.c.load(m.OLD_MANIFEST)
        self.assertEqual(len(plan['source_sha256']),100);self.assertTrue(all(plan['source_sha256'][p]==h for p,h in old['source_sha256'].items()))
        self.assertEqual(plan['complete_current_serial_catalog'],m.CATALOG)
        self.assertEqual(len(plan['predecessor_owners']),2811);self.assertEqual(len(plan['predecessor_services']),1455)
        self.assertEqual(plan['predecessor_expected_exit_code'],1);self.assertEqual(plan['archive_mode'],'snapshot')
        self.assertEqual(plan['budget'],b.c.INVENTORY);self.assertFalse(plan['runtime_admitted'])
        self.assertNotIn('archive_inputs',plan)
    def test_original876_scope_and_three_retirements_retained(self):
        plan=b.build(b.c.load,REVIEW,OUTPUT);scope=b.c.load(plan['r2_predecessor_scope'])
        for owner in scope['predecessor_owners']:self.assertIn(owner,plan['predecessor_owners'])
        self.assertEqual(plan['retired_lfm_r2']['invocation_id'],'ea961c63d568400c803def561cf7b6cd')
        self.assertEqual([x['actual_tool']['exit_code']for x in plan['retired_mini_history']],[1,0])
    def test_removed_old_source_cannot_pass_even_matching_review(self):
        manifest=b.c.load(b.MANIFEST);review=b.c.load(REVIEW)
        manifest['source_sha256'].pop(next(iter(b.c.load(m.OLD_MANIFEST)['source_sha256'])))
        review['source_sha256']=manifest['source_sha256']
        with self.assertRaises(RuntimeError):b.build(read_with({b.MANIFEST['path']:manifest,REVIEW['path']:review}),REVIEW,OUTPUT)
    def test_source_review_and_original_budget_required(self):
        review=b.c.load(REVIEW);review['passed']=False
        with self.assertRaises(RuntimeError):b.build(read_with({REVIEW['path']:review}),REVIEW,OUTPUT)
        manifest=b.c.load(b.MANIFEST);manifest['budget']['wall_seconds']+=1
        with self.assertRaises(RuntimeError):b.build(read_with({b.MANIFEST['path']:manifest}),REVIEW,OUTPUT)
    def test_latest_failed_download_and_full124_rows_cannot_be_erased(self):
        catalog=b.c.load(m.CATALOG);last=catalog['runs'][-1]['terminal'];receipt=b.c.load(last);receipt['actual_tool']['exit_code']=0
        with self.assertRaises(RuntimeError):b.build(read_with({last['path']:receipt}),REVIEW,OUTPUT)
        catalog['runs'].pop()
        with self.assertRaises(RuntimeError):b.build(read_with({m.CATALOG['path']:catalog}),REVIEW,OUTPUT)
    def test_wrong_namespace_rejected(self):
        with self.assertRaises(RuntimeError):b.build(b.c.load,REVIEW,'/var/lib/apt/lists')
    def test_original_producer_must_bind_exact_signed_result(self):
        assessment=b.c.load(m.ASSESSMENT);producer=b.c.load(assessment['producer_actual_exit']);producer['evidence']['signed-metadata/sources-result.json']['sha256']='f'*64
        with self.assertRaises(RuntimeError):m.metadata(read_with({assessment['producer_actual_exit']['path']:producer}))
    def test_complete_original_trust_anchor_required(self):
        assessment=b.c.load(m.ASSESSMENT);signed=b.c.load(assessment['signed_metadata']);signed['trust_anchor']['keyring_sha256']='f'*64
        with self.assertRaises(RuntimeError):m.metadata(read_with({assessment['signed_metadata']['path']:signed}))
    def test_snapshot_manifest_keeps_old_logical_refs_and_exact_owned_paths(self):
        value,a=fixture_result();plan=dict(output_root=OUTPUT)
        self.assertIs(m.validate_snapshot_result(value,plan,a),value)
        for kind in ('logical','physical','size','ordered','system'):
            changed=copy.deepcopy(value)
            if kind=='logical':changed['rows'][0]['logical_ref']['path']='/var/lib/apt/lists/other'
            elif kind=='physical':changed['rows'][0]['snapshot_ref']['path']='/var/lib/apt/lists/current'
            elif kind=='size':changed['rows'][1]['snapshot_ref']['bytes']+=1
            elif kind=='ordered':changed['rows'].reverse()
            else:changed['system_APT_files_changed']=True
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):m.validate_snapshot_result(changed,plan,a)
    def test_seven_events_bind_every_verified_input_and_output(self):
        value,a=fixture_result();ref=dict(path=OUTPUT+'/snapshot-manifest.json',bytes=1,sha256='a'*64);events=[]
        for row in value['rows']:events.extend([dict(event='snapshot_xz_verified',suite=row['suite'],compressed_ref=row['compressed_ref']),dict(event='snapshot_plaintext_written',suite=row['suite'],snapshot_ref=row['snapshot_ref'])])
        events.append(dict(event='snapshot_complete',snapshot_manifest=ref,raw_total_bytes=m.RAW_TOTAL,indexes=3));m.validate_events(events,value,ref)
        with self.assertRaises(RuntimeError):m.validate_events(events[:-1],value,ref)
        changed=copy.deepcopy(events);changed[0],changed[1]=changed[1],changed[0]
        with self.assertRaises(RuntimeError):m.validate_events(changed,value,ref)
    def test_corrupt_full_xz_rejected_before_factory_is_called(self):
        row=dict(compressed=dict(bytes=3,sha256=hashlib.sha256(b'abc').hexdigest()),expected_uncompressed=dict(bytes=1,sha256=hashlib.sha256(b'x').hexdigest()))
        factory=Mock()
        with self.assertRaises(RuntimeError):m.decode_exact(b'abd',row,factory)
        factory.assert_not_called()
    def test_pure_decoder_factory_has_exact_limits_and_no_third_stream(self):
        compressed=b'abc';plain=b'old';row=dict(compressed=dict(bytes=3,sha256=hashlib.sha256(compressed).hexdigest()),expected_uncompressed=dict(bytes=3,sha256=hashlib.sha256(plain).hexdigest()))
        first=Mock(eof=True,unused_data=m.EMPTY_TAIL);first.decompress.return_value=plain
        second=Mock(eof=True,unused_data=b'');second.decompress.return_value=b'';factory=Mock(side_effect=[first,second])
        self.assertEqual(m.decode_exact(compressed,row,factory),plain)
        first.decompress.assert_called_once_with(compressed,max_length=4);second.decompress.assert_called_once_with(m.EMPTY_TAIL,max_length=1)
        self.assertEqual([call.args for call in factory.call_args_list],[(64*1024*1024,),(64*1024*1024,)])
        first=Mock(eof=True,unused_data=m.EMPTY_TAIL);first.decompress.return_value=plain
        second=Mock(eof=True,unused_data=b'third');second.decompress.return_value=b''
        with self.assertRaises(RuntimeError):m.decode_exact(compressed,row,Mock(side_effect=[first,second]))
    def test_combined_runner_worker_command_keeps_role_and_no_automatic_next_stage(self):
        source=(b.Q/'run_minicpm5_sdk310_signed_index_snapshot_v1.py').read_text()
        self.assertIn("argv=[python,'-B','-s',worker,'--mode','worker','--plan'",source)
        self.assertIn('inputs.run_snapshot(plan,guard,journal.emit)',source)
        self.assertIn("stat.S_IMODE(path.lstat().st_mode)==0o400",source)
        self.assertNotIn('deb_members_adapter',source)


if __name__=='__main__':unittest.main(verbosity=2)
