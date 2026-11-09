"""Metadata transport checks, including real exact small preserved text files."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import completed_storage_contract_v1 as storage
import nemotron9b_completed_storage_metadata_after_retry_v1 as c


def plan_value():
    return dict(schema='completed_qwen_storage_after_retry_plan_v1',source_sha256=dict(c.EXACT_TEXT),
        budget={},completed_residual_review={},weights=[{}]*5,journals=[{}]*8,
        runtime_admitted=False,scientific_constraints_changed=False)


class Metadata(unittest.TestCase):
    def test_all_six_actual_text_pins_match_original_candidate(self):
        candidate,_,_=storage.fixed_targets()
        expected={r['path']:r['sha256'] for r in storage.evidence(candidate)
                  if Path(r['path']).suffix in ('.metadata','.sh')}
        self.assertEqual(c.EXACT_TEXT,expected)
        for path,digest in c.EXACT_TEXT.items():
            raw=c.text(path,digest,storage.p.metadata.stable_bytes)
            self.assertLess(len(raw),1024)

    def test_other_sh_or_metadata_path_cannot_be_opened(self):
        reader=Mock()
        for path in ('/tmp/other.sh','/tmp/other.metadata','/tmp/model.safetensors',str(c.PLAN)+'.other'):
            with self.assertRaises(RuntimeError):c.text(path,'0'*64,reader)
        reader.assert_not_called()

    def test_rebound_original_text_hash_rejected_before_open(self):
        reader=Mock()
        with self.assertRaises(RuntimeError):c.text(next(iter(c.EXACT_TEXT)),'0'*64,reader)
        reader.assert_not_called()

    def test_named_plan_envelope_can_exceed_ordinary_limit(self):
        value=plan_value();value['padding']='x'*c.LIMIT
        raw=json.dumps(value).encode()
        reader=Mock(return_value=raw)
        self.assertEqual(c.text(c.PLAN,hashlib.sha256(raw).hexdigest(),reader),raw)
        reader.assert_called_once_with(c.PLAN,c.ENVELOPE)

    def test_named_review_envelope_and_plan_binding(self):
        value=dict(schema='completed_qwen_storage_after_retry_source_review_v1',source_sha256=dict(c.EXACT_TEXT),
            reviewer='/root',passed=True,blockers=[],plan=dict(path=str(c.PLAN),sha256='0'*64))
        self.assertTrue(c.shape(c.REVIEW,value))
        raw=json.dumps(value).encode();reader=Mock(return_value=raw)
        c.text(c.REVIEW,hashlib.sha256(raw).hexdigest(),reader)
        reader.assert_called_once_with(c.REVIEW,c.LIMIT)
        value['plan']['path']='/other/plan.json'
        with self.assertRaises(RuntimeError):c.shape(c.REVIEW,value)

    def test_unknown_envelope_schema_rejected(self):
        value=plan_value();value['schema']='unrecognized'
        raw=json.dumps(value).encode()
        with self.assertRaises(RuntimeError):c.text(c.PLAN,hashlib.sha256(raw).hexdigest(),lambda path,cap:raw)

    def test_shape_cannot_omit_preserved_source_pins(self):
        value=plan_value();value['source_sha256'].popitem()
        with self.assertRaises(RuntimeError):c.shape(c.PLAN,value)

    def test_plan_cannot_change_scientific_scope(self):
        for key in ('runtime_admitted','scientific_constraints_changed'):
            value=plan_value();value[key]=True
            with self.assertRaises(RuntimeError):c.shape(c.PLAN,value)

    def test_payload_expansion_not_accepted_as_plan(self):
        value=plan_value();value['weights'].append({})
        with self.assertRaises(RuntimeError):c.shape(c.PLAN,value)

    def test_previous_named_envelopes_remain_accepted(self):
        from unittest.mock import patch
        for path in (c.previous.PLAN,c.previous.REVIEW):
            self.assertTrue(c.accepts(path))
            with patch.object(c.previous,'text',return_value=b'original') as reader:
                self.assertEqual(c.text(path,'0'*64,None),b'original')
            reader.assert_called_once_with(path,'0'*64,None)

    def test_previous_schema_cannot_impersonate_retry_plan(self):
        value=plan_value();value['schema']='completed_qwen_storage_maintenance_plan_v1'
        with self.assertRaises(RuntimeError):c.shape(c.PLAN,value)


if __name__=='__main__':unittest.main(verbosity=2)
