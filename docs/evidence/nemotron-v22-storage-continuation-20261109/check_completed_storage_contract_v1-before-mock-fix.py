"""No real maintenance: fixed metadata checks and disposable two-phase fixtures."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import completed_storage_contract_v1 as c
import completed_storage_files_v1 as f
import run_completed_storage_serial_v1 as worker
import completed_storage_privileged_v1 as privileged


class Contract(unittest.TestCase):
    def setUp(self):
        self.candidate,self.weights,self.journals = c.fixed_targets()
        self.plan = dict(schema=c.SCHEMA,output=str(c.OUT),executor=str(c.EXECUTOR),
            privileged_helper=str(c.PRIVILEGED),candidate=c.CANDIDATE,audit=c.AUDIT,
            protected_model_spec=c.PARENT,budget=copy.deepcopy(c.BUDGET),
            weights=copy.deepcopy(self.weights),journals=copy.deepcopy(self.journals),
            scientific_constraints_changed=False,runtime_admitted=False,automatic_retry=False,
            original_qwen_bf16_restore_required=True,
            source_sha256={name:'0'*64 for name in [*c.p.required_sources(),*[str(c.Q/n) for n in c.NAMES]]})

    def test_actual_fixed_target_metadata(self):
        c.validation(self.plan)
        self.assertEqual(sum(r['stat']['st_size'] for r in self.weights),16381516776)
        self.assertEqual(sum(r['stat']['st_size'] for r in self.journals),1073741824)

    def test_actual_restore_and_preserved_dependency_metadata(self):
        c.fixed_evidence(self.candidate)

    def test_retirement_cannot_expand_to_int8(self):
        self.plan['weights'][0]['path'] = '/mnt/nvme/models/Qwen3-8B-int8/model.safetensors'
        with self.assertRaises(RuntimeError): c.validation(self.plan)

    def test_active_journal_cannot_be_substituted(self):
        self.plan['journals'][0]['path'] = str(c.JOURNAL_ROOT/'user-1000.journal')
        with self.assertRaises(RuntimeError): c.validation(self.plan)

    def test_no_weakened_budget(self):
        self.plan['budget']['root_floor'] = 1
        with self.assertRaises(RuntimeError): c.validation(self.plan)

    def test_no_missing_source(self):
        self.plan['source_sha256'].pop(str(c.PRIVILEGED))
        with self.assertRaises(RuntimeError): c.validation(self.plan)

    def test_completion_gate_precedes_output_or_target_reads(self):
        with patch.object(c.residual,'completed',side_effect=RuntimeError('still live')) as done, \
             patch.object(c,'fixed_targets') as targets, patch.object(c,'once') as save:
            with self.assertRaisesRegex(RuntimeError,'still live'): c.build({'actual':'missing'})
        done.assert_called_once();targets.assert_not_called();save.assert_not_called()

    def test_metadata_limit_and_no_invented_plan_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'plan.json'
            raw = json.dumps({'schema':'unrecognized'}).encode()
            path.write_bytes(raw)
            with self.assertRaises(RuntimeError):
                c.load(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()),plan=True)

    def test_root_helper_refuses_unadmitted_receipt_before_unlink(self):
        with patch.object(c,'load',side_effect=[self.plan,{'plan':{},'fully_drained':False}]), \
             patch.object(f,'unlink_exact') as unlink:
            with self.assertRaises(RuntimeError):
                privileged.retire(dict(plan={'path':'/invalid','sha256':'0'*64},
                    admission={'path':str(c.OUT/'admission.json'),'sha256':'0'*64}))
        unlink.assert_not_called()

    def test_privileged_inspection_fails_closed_on_unreadable_process(self):
        with patch.object(privileged,'scan',return_value=dict(processes=1,matches=[],errors=[{}],complete=False)):
            with self.assertRaises(RuntimeError): privileged.unused(self.weights)

    def test_privileged_inspection_rejects_live_use(self):
        with patch.object(privileged,'scan',return_value=dict(processes=1,matches=[{'kind':'fd'}],errors=[],complete=True)):
            with self.assertRaises(RuntimeError): privileged.unused(self.weights)


class TwoPhase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='completed-storage-two-phase-')
        self.root = Path(self.tmp.name).resolve()
        out = self.root/'output';out.mkdir()
        self.plan = dict(output=str(out),budget=dict(gzip_cap=2**20),weights=[],journals=[])
        for kind,n in [('weights',5),('journals',8)]:
            for i in range(n):
                path = self.root/f'{kind}-{i}'
                content = (kind+str(i)).encode()*100
                path.write_bytes(content)
                row = dict(path=str(path),stat=f.ordinary(path))
                if kind == 'weights': row['sha256_from_official_metadata'] = hashlib.sha256(content).hexdigest()
                self.plan[kind].append(row)

    def tearDown(self): self.tmp.cleanup()

    def prepare(self):
        with redirect_stdout(io.StringIO()):
            return worker.prepare_all(self.plan,lambda:None,availability=lambda rows,guard:[])

    def intact(self):
        self.assertTrue(all(Path(r['path']).exists() for r in self.plan['weights']+self.plan['journals']))

    def test_success_keeps_all_originals_until_separate_retirement(self):
        result = self.prepare()
        self.assertEqual((len(result['weights']),len(result['journals'])),(5,8))
        self.assertTrue(result['all_eight_journals_decompressed'])
        self.intact()

    def test_bad_last_weight_leaves_every_original_and_no_archive(self):
        self.plan['weights'][-1]['sha256_from_official_metadata'] = '0'*64
        with self.assertRaises(RuntimeError): self.prepare()
        self.intact()
        self.assertFalse(list(Path(self.plan['output']).iterdir()))

    def test_last_journal_failure_never_removes_verified_earlier_sources(self):
        last = Path(self.plan['journals'][-1]['path'])
        last.write_bytes(b'changed after admission')
        with self.assertRaises(RuntimeError): self.prepare()
        self.intact()
        self.assertEqual(len(list(Path(self.plan['output']).glob('*.gz'))),7)

    def test_archive_budget_failure_keeps_every_source(self):
        self.plan['budget']['gzip_cap'] = 1
        with self.assertRaises(RuntimeError): self.prepare()
        self.intact()

    def test_official_availability_failure_precedes_hash_or_archive(self):
        def failed(rows,guard): raise RuntimeError('not downloadable')
        with patch.object(f,'file_hash') as hashed, patch.object(f,'archive_journal') as archived:
            with self.assertRaises(RuntimeError): worker.prepare_all(self.plan,lambda:None,availability=failed)
        hashed.assert_not_called();archived.assert_not_called();self.intact()


if __name__ == '__main__': unittest.main(verbosity=2)
