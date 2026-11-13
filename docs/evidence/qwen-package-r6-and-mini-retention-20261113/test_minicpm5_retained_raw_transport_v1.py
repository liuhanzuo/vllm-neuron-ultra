"""Only /tmp synthetic files. No production metadata, raw, model or History IO."""
import copy
import json
import os
import stat
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import minicpm5_retained_raw_transport_v1 as t


def fixture(root, count=2):
    rows=[]
    for i in range(count):
        data=bytes(range(64))*(i+1)
        a,b=root/('original-%d.raw'%i),root/('keeper-%d.raw'%i)
        a.write_bytes(data);b.write_bytes(data)
        original,keeper=t.identity(a),t.identity(b)
        def producer(ident):
            return dict(file=ident['realpath'],identity=ident,encoding='raw_bf16_le',
                raw_bytes=len(data),payload_sha256=t.digest(data),raw_sha256=t.digest(data),
                shape=[len(data)//2],dtype='torch.bfloat16',rank=0,name='synthetic',nonfinite_elements=0)
        rows.append(dict(index=i,original=original,keeper=keeper,sha256=t.digest(data),bytes=len(data),
            original_row=producer(original),keeper_row=producer(keeper)))
    return dict(schema=t.PLAN_SCHEMA,scope='synthetic-tests-only',provenance={},entries=rows)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mini-raw-test-',dir='/tmp')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.plan=fixture(self.root);self.guard=lambda:None

    def verified(self):
        return t._verify_pairs(self.plan,self.root/'verified.json',self.guard)

    def replay(self, ref=None):
        return t.Replay(self.plan,ref or self.verified(),self.guard,_synthetic=True)

    def admission(self):
        refs={k:dict(path=str(self.root/(k+'.json')),sha256='a'*64) for k in
            ('independent_verification','remover_source','replay_source','serial_lease','full_history_compatibility')}
        return dict(plan_sha256=t.digest(t.canonical(self.plan)),**refs)

    def test_full_pair_receipt_and_replay_preserve_bytes_and_two_identities(self):
        ref=self.verified();receipt=t.read_receipt(ref)
        self.assertEqual(receipt['original_and_keeper_bytes_read'],384)
        replay=self.replay(ref)
        for row in self.plan['entries']:
            data,proof=replay.read_state(row['original_row'])
            self.assertEqual(data,Path(row['original']['realpath']).read_bytes())
            self.assertEqual(proof['capture_identity'],row['original'])
            self.assertEqual(proof['storage_identity'],row['keeper'])
            self.assertFalse(proof['original_identity_is_current_stat'])

    def test_public_production_api_rejects_synthetic_or_modified_plan(self):
        with self.assertRaises(ValueError):t.verify_pairs(self.plan,self.root/'no.json',self.guard)
        with self.assertRaises(ValueError):t.Replay(self.plan,self.verified(),self.guard)
        plan=copy.deepcopy(self.plan);plan['scope']=t.PRODUCTION_SCOPE
        with self.assertRaises(ValueError):t.validate_plan(plan)

    def test_synthetic_bypass_cannot_point_at_production(self):
        plan=copy.deepcopy(self.plan)
        for row in plan['entries']:
            for field,source in [('original','original_row'),('keeper','keeper_row')]:
                name='/mnt/nvme/'+Path(row[field]['realpath']).name
                row[field]['realpath']=name;row[source]['file']=name
        with self.assertRaisesRegex(ValueError,'production'):t.synthetic_plan(plan)

    def test_missing_original_and_missing_keeper_fail_closed(self):
        Path(self.plan['entries'][0]['original']['realpath']).unlink()
        with self.assertRaises(FileNotFoundError):self.verified()
        self.assertFalse((self.root/'verified.json').exists())

    def test_equal_length_wrong_bytes_rejected_before_receipt(self):
        row=self.plan['entries'][0];path=Path(row['keeper']['realpath']);path.write_bytes(b'x'*row['bytes'])
        ident=t.identity(path);row['keeper']=ident;row['keeper_row']['identity']=ident
        with self.assertRaisesRegex(ValueError,'bytes differ'):self.verified()
        self.assertFalse((self.root/'verified.json').exists())

    def test_both_equal_but_wrong_digest_rejected(self):
        row=self.plan['entries'][0]
        row['sha256']='0'*64
        for key in ('original_row','keeper_row'):row[key]['payload_sha256']=row[key]['raw_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA'):self.verified()

    def test_interruption_never_publishes_complete_verification(self):
        count=0
        def guard():
            nonlocal count
            count+=1
            if count==4:raise InterruptedError('synthetic interruption')
        with self.assertRaises(InterruptedError):t._verify_pairs(self.plan,self.root/'verified.json',guard)
        self.assertFalse((self.root/'verified.json').exists())
        self.verified()  # Entire pass is safely repeated; no half-pass promotion.

    def test_replacement_during_read_and_earlier_pair_replacement_are_detected(self):
        calls=0
        def guard():
            nonlocal calls
            calls+=1
            if calls==3:
                p=Path(self.plan['entries'][0]['keeper']['realpath']);replacement=self.root/'replacement.raw'
                replacement.write_bytes(p.read_bytes());replacement.replace(p)
        with self.assertRaisesRegex(ValueError,'changed'):t._verify_pairs(self.plan,self.root/'v.json',guard)
        self.assertFalse((self.root/'v.json').exists())

    def test_replay_rejects_missing_replaced_and_modified_keeper(self):
        replay=self.replay();row=self.plan['entries'][0];p=Path(row['keeper']['realpath'])
        p.unlink()
        with self.assertRaises(FileNotFoundError):replay.read_state(row['original_row'])
        p.write_bytes(bytes(range(64)))
        with self.assertRaisesRegex(ValueError,'identity changed'):replay.read_state(row['original_row'])

    def test_manifest_shape_or_unknown_logical_path_cannot_be_changed(self):
        replay=self.replay();row=copy.deepcopy(self.plan['entries'][0]['original_row']);row['shape']=[999]
        with self.assertRaisesRegex(ValueError,'entire original'):replay.read_state(row)
        row['file']=str(self.root/'other.raw')
        with self.assertRaisesRegex(ValueError,'Unknown'):replay.read_state(row)

    def test_hardlink_and_symlink_rejected(self):
        row=self.plan['entries'][0];p=Path(row['keeper']['realpath'])
        other=self.root/'hardlink.raw';os.link(p,other)
        with self.assertRaisesRegex(ValueError,'Singleton'):t.identity(p)
        other.unlink();other.symlink_to(p)
        with self.assertRaisesRegex(ValueError,'Canonical|Symlink'):t.identity(other)

    def test_export_recreates_bytes_with_honest_new_identity(self):
        replay=self.replay();row=self.plan['entries'][0]
        ref=replay.export_state(row['original_row'],self.root/'export.raw',self.root/'export.json')
        value=t.read_receipt(ref)
        self.assertEqual((self.root/'export.raw').read_bytes(),bytes(range(64)))
        self.assertNotEqual(value['export_identity']['inode'],row['original']['inode'])
        self.assertFalse(value['capture_inode_recreated'])
        with self.assertRaises(FileExistsError):replay.export_state(row['original_row'],self.root/'export.raw',self.root/'again.json')

    def test_export_does_not_touch_original_or_keeper(self):
        replay=self.replay();row=self.plan['entries'][0]
        for key in ('original','keeper'):
            with self.assertRaisesRegex(ValueError,'overwrite'):replay.export_state(row['original_row'],row[key]['realpath'],self.root/'x.json')

    def test_export_parent_is_fsynced_before_receipt_in_another_directory(self):
        replay=self.replay();directory=self.root/'export-directory';directory.mkdir();seen=[]
        original_fsync=os.fsync;original_write=t._write_once
        def fsync(fd):
            value=os.fstat(fd)
            if stat.S_ISDIR(value.st_mode):seen.append(value.st_ino)
            return original_fsync(fd)
        def receipt(path,value):
            self.assertIn(directory.stat().st_ino,seen)
            self.assertEqual(t.digest((directory/'restored.raw').read_bytes()),value['sha256'])
            return original_write(path,value)
        with patch.object(t.os,'fsync',side_effect=fsync),patch.object(t,'_write_once',side_effect=receipt):
            replay.export_state(self.plan['entries'][0]['original_row'],directory/'restored.raw',self.root/'receipt.json')

    def test_export_change_during_reread_never_publishes_completion(self):
        replay=self.replay();destination=self.root/'export.raw';changed=False
        def guard():
            nonlocal changed
            if destination.exists()and not changed:
                changed=True;destination.write_bytes(b'z'*64)
        replay.guard=guard
        with self.assertRaises(ValueError):replay.export_state(self.plan['entries'][0]['original_row'],destination,self.root/'receipt.json')
        self.assertFalse((self.root/'receipt.json').exists())

    def test_durable_intent_before_partial_removal_and_resumed_postcondition(self):
        ref=self.verified();intent=t._prepare_intent(self.plan,ref,self.root/'intent.json',self.admission(),self.guard)
        Path(self.plan['entries'][0]['original']['realpath']).unlink()  # Synthetic external remover only.
        with self.assertRaisesRegex(ValueError,'still present'):t._audit_retired(self.plan,ref,intent,self.root/'done.json',self.guard)
        self.assertFalse((self.root/'done.json').exists())
        # Exact original capture identity is retained even after logical retirement.
        self.assertEqual(self.replay(ref).read_state(self.plan['entries'][0]['original_row'])[0],bytes(range(64)))
        Path(self.plan['entries'][1]['original']['realpath']).unlink()
        done=t._audit_retired(self.plan,ref,intent,self.root/'done.json',self.guard)
        value=t.read_receipt(done);self.assertEqual(value['original_paths_absent'],2)
        self.assertFalse(value['unlink_execution_observed_here'])

    def test_missing_or_partial_verification_cannot_admit_intent(self):
        ref=self.verified();value=t.read_receipt(ref);value['records'].pop()
        bad=t._write_once(self.root/'partial.json',value)
        with self.assertRaisesRegex(ValueError,'partial'):t._prepare_intent(self.plan,bad,self.root/'intent.json',self.admission(),self.guard)

    def test_receipt_cannot_be_replaced_or_overwritten(self):
        ref=self.verified()
        with self.assertRaises(FileExistsError):t._verify_pairs(self.plan,self.root/'verified.json',self.guard)
        Path(ref['path']).write_text('{}')
        with self.assertRaises(ValueError):t.read_receipt(ref)

    def test_truncated_or_interrupted_json_is_not_a_receipt(self):
        p=self.root/'partial.json';p.write_bytes(b'{')
        ref=dict(path=str(p),bytes=1,sha256=t.digest(b'{'))
        with self.assertRaises(json.JSONDecodeError):t.read_receipt(ref)

    def test_full_shape_rejects_chains_duplicate_indices_and_broken_producer(self):
        for variant in ('chain','index','producer'):
            with self.subTest(variant=variant):
                plan=copy.deepcopy(self.plan)
                if variant=='index':plan['entries'][1]['index']=0
                elif variant=='producer':plan['entries'][0]['original_row']['raw_bytes']+=1
                else:
                    plan['entries'][1]['keeper']=copy.deepcopy(plan['entries'][0]['original'])
                    plan['entries'][1]['keeper_row']=copy.deepcopy(plan['entries'][0]['original_row'])
                with self.assertRaises(ValueError):t.validate_plan(plan,production=False)


if __name__=='__main__':unittest.main(verbosity=2)
