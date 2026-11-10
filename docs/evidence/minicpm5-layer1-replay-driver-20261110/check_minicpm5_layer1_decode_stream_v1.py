"""Complete synthetic cache streams; no real tensor, SDK or model execution."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research import minicpm5_layer1_decode_stream_v1 as stream
from research import check_minicpm5_layer1_results_v1 as fixtures
raw=stream.raw


def write_json(path,value):
    data=(json.dumps(value,sort_keys=True)+'\n').encode();path.write_bytes(data)
    return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest())


def read_json(path,digest):
    path=Path(path);assert path.suffix=='.json'and path.stat().st_size<8*2**20
    data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==digest
    return json.loads(data)


class Fixture:
    def __init__(self,root):
        self.root=root;self.plan=json.loads((ROOT/'research/minicpm5-layer1-runner-source-plan-20261110-v1.json').read_text())
        self.plan['output_root']=str(root);self.ranks=[];self.call_paths={};self.originals={};rows=[]
        fixtures.IndependentRaw.setUpClass();maker=fixtures.IndependentRaw()
        for rank in (0,1):
            rankrow=dict(schema='minicpm5_layer1_observation_cpu_rank_v1',complete=True,rank=rank,arms=list(raw.ARMS),arm_reports=[])
            for ai,arm in enumerate(raw.ARMS):
                folder=root/('cpu-rank'+str(rank))/arm;folder.mkdir(parents=True)
                armrow=dict(schema='minicpm5_layer1_observation_cpu_arm_v1',complete=True,rank=rank,arm=arm,reports=[])
                encoded=None;previous=None
                for ci in range(14):
                    call,original,payload,encoded=maker.fixture(ci,rank,ai,encoded)
                    callfolder=folder/('call-%02d'%ci);callfolder.mkdir()
                    original_ref=write_json(callfolder/'synthetic-original.json',original)
                    self.originals[rank,ai,ci]=original_ref
                    self.plan['layer1_observation']['original_calls'][rank*42+ai*14+ci]=dict(rank=rank,arm=arm,call=ci,**original_ref)
                    call['schema']='minicpm5_layer1_observation_cpu_call_v1'
                    payload_path=callfolder/'layer1-observation.bin';payload_path.write_bytes(payload)
                    call['layer1_observation'].update(original_call=original_ref,path=str(payload_path),
                        identity=stream.payload_io.file_identity(payload_path,payload_path.lstat()))
                    proof,previous=raw.verify_call(call,original,original_ref,payload,previous);rows.append(proof)
                    call_ref=write_json(callfolder/'result.json',call);self.call_paths[rank,ai,ci]=call_ref
                    armrow['reports'].append(dict(call=ci,report=call_ref['path'],sha256=call_ref['sha256']))
                arm_ref=write_json(folder/'result.json',armrow)
                rankrow['arm_reports'].append(dict(arm=arm,report=arm_ref['path'],sha256=arm_ref['sha256']))
            self.ranks.append(rankrow)
        # This fixture isolates stream loading. It does not run or certify the
        # separately tested full-logits/HF numeric audit or external exit chain.
        numeric=dict(schema='minicpm5_layer1_full_logits_numeric_audit_v1',complete=True,
            logits_payload_count=84,official_reference_frames=32,candidate_metric_frames=96,
            original_selected_tap_bytes=3145728,replicated_ranks_verified_and_counted_once=True,
            metrics_independently_recomputed=True,exact_classification_and_argmax=True)
        cache=dict(schema='minicpm5_layer1_independent_byte_verification_v1',complete=True,ranks=2,arms=3,
            calls=84,frames_per_arm=32,full_cache_records=168,complete_cache_bytes=168*raw.CACHE_BYTES,
            every_tap_and_full_cache_hash_exact=True,all_own_history_links_exact=True,raw_records=rows)
        self.audit=dict(schema='minicpm5_layer1_complete_saved_data_audit_v1',complete=True,
            numeric=numeric,layer1_cache=cache,attention_causality_proven=False,full_model_qualified=False,new_speedup_claim=False)


class StreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='mini-decode-stream-synthetic-');cls.fixture=Fixture(Path(cls.tmp.name))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def make(self,rank=0,audit=None,result=None,reader=read_json,guard=lambda:None):
        f=self.fixture
        return stream.RankStream(f.plan,result or f.ranks[rank],audit or f.audit,reader,guard)
    def test_both_ranks_all_histories_and36_complete_decode_calls(self):
        total=0
        for rank in (0,1):
            s=self.make(rank);seen=[]
            for value in s:
                case=value['case'];self.assertEqual(case['rank'],rank)
                self.assertEqual(len(value['normalized_raw']),16384);self.assertEqual(len(value['expected_attention_raw']),16384)
                self.assertEqual(case['normalized_input']['metadata']['stride'],[1,4,4])
                self.assertEqual(set(value['cache_raw']),set(raw.CACHES))
                ci=case['call']['index'];self.assertIn(ci,stream.boundary.DECODE_CALLS)
                for name,data in value['cache_raw'].items():
                    self.assertEqual(len(data),raw.CACHE_BYTES)
                    self.assertEqual(data,fixtures.IndependentRaw.snapshots[ci][name])
                seen.append((case['arm'],ci));total+=1
            self.assertEqual(seen,[(arm,ci)for arm in raw.ARMS for ci in stream.boundary.DECODE_CALLS])
            proof=s.finish();self.assertEqual(proof['all_history_calls'],42);self.assertEqual(proof['decode_calls'],18)
            self.assertFalse(proof['attention_reexecuted']);self.assertFalse(proof['new_speedup_claim'])
        self.assertEqual(total,36)
    def test_no_early_completion_or_second_consumption(self):
        s=self.make()
        with self.assertRaisesRegex(ValueError,'exhaust'):s.finish()
        gen=iter(s);next(gen)
        with self.assertRaisesRegex(ValueError,'exhaust'):s.finish()
        with self.assertRaisesRegex(ValueError,'only once'):next(iter(s))
        gen.close()
        with self.assertRaisesRegex(ValueError,'exhaust'):s.finish()
    def test_missing_or_changed_audit_scope_rejected_before_payload(self):
        for section,field,value in ((None,'complete',False),('numeric','official_reference_frames',31),
            ('numeric','candidate_metric_frames',95),('numeric','logits_payload_count',83),
            ('numeric','exact_classification_and_argmax',False),('layer1_cache','full_cache_records',167),
            ('layer1_cache','complete_cache_bytes',4096),('layer1_cache','all_own_history_links_exact',False)):
            audit=copy.deepcopy(self.fixture.audit)
            target=audit if section is None else audit[section];target[field]=value
            with self.subTest(section=section,field=field),patch.object(stream.payload_io,'read_owned_payload',side_effect=AssertionError('must not read')):
                with self.assertRaises(ValueError):self.make(audit=audit)
    def test_reordered_or_omitted_audited_calls_rejected(self):
        for action in ('omit','swap'):
            audit=copy.deepcopy(self.fixture.audit);rows=audit['layer1_cache']['raw_records']
            if action=='omit':rows.pop()
            else:rows[0],rows[42]=rows[42],rows[0]
            with self.subTest(action=action),self.assertRaises(ValueError):self.make(audit=audit)
    def test_audited_payload_hash_mismatch_rejected(self):
        audit=copy.deepcopy(self.fixture.audit);audit['layer1_cache']['raw_records'][0]['payload_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'independently audited'):next(iter(self.make(audit=audit)))
    def test_rank_result_wrong_order_rejected(self):
        result=copy.deepcopy(self.fixture.ranks[0]);result['arm_reports'].reverse()
        with self.assertRaisesRegex(ValueError,'arms in order'):next(iter(self.make(result=result)))
    def test_missing_prefill_history_call_rejected(self):
        target=self.fixture.ranks[0]['arm_reports'][0]['report']
        def reader(path,digest):
            value=read_json(path,digest)
            if path==target:value['reports'].pop(0)
            return value
        with self.assertRaisesRegex(ValueError,'All14'):next(iter(self.make(reader=reader)))
    def test_call_cannot_point_to_other_rank_payload(self):
        target=self.fixture.call_paths[0,0,0]['path']
        def reader(path,digest):
            value=read_json(path,digest)
            if path==target:value['layer1_observation']['path']=value['layer1_observation']['path'].replace('cpu-rank0','cpu-rank1')
            return value
        with self.assertRaisesRegex(ValueError,'admitted call observation'):next(iter(self.make(reader=reader)))
    def test_original_call_json_authentication_required(self):
        original=self.fixture.originals[0,0,0]['path']
        def reader(path,digest):
            if path==original:raise ValueError('original JSON hash mismatch')
            return read_json(path,digest)
        with self.assertRaisesRegex(ValueError,'JSON hash'):next(iter(self.make(reader=reader)))
    def test_decode_layout_cannot_silently_change(self):
        target=self.fixture.originals[0,0,4]['path']
        def reader(path,digest):
            value=read_json(path,digest)
            if path==target:value['stage_metadata'][stream.boundary.NORMALIZED]['stride']=[2048,2048,1]
            return value
        with self.assertRaisesRegex(ValueError,'stage_metadata'):next(iter(self.make(reader=reader)))
    def test_guard_denial_before_any_metadata_read(self):
        def denied():raise RuntimeError('not admitted')
        with self.assertRaisesRegex(RuntimeError,'not admitted'):
            stream.RankStream(None,None,None,lambda *a:(_ for _ in ()).throw(AssertionError('no read')),denied)
    def test_revocation_after_yield_prevents_further_reads(self):
        allowed=[True]
        def guard():
            if not allowed[0]:raise RuntimeError('lease revoked')
        gen=iter(self.make(guard=guard));next(gen);allowed[0]=False
        with self.assertRaisesRegex(RuntimeError,'lease revoked'):next(gen)
    def test_whole_payload_corruption_rejected(self):
        ref=self.fixture.call_paths[0,0,0];call=read_json(ref['path'],ref['sha256']);path=Path(call['layer1_observation']['path'])
        original=path.read_bytes()
        try:
            path.write_bytes(b'!'+original[1:])
            with self.assertRaisesRegex(ValueError,'identity'):next(iter(self.make()))
        finally:
            path.write_bytes(original)
            # Restoring bytes changes metadata; update only this synthetic fixture
            # record and its exact authenticated arm reference for later tests.
            call['layer1_observation']['identity']=stream.payload_io.file_identity(path,path.lstat())
            changed=write_json(Path(ref['path']),call);self.fixture.call_paths[0,0,0]=changed
            entry=self.fixture.ranks[0]['arm_reports'][0];arm=read_json(entry['report'],entry['sha256'])
            arm['reports'][0]['sha256']=changed['sha256'];armref=write_json(Path(entry['report']),arm);entry['sha256']=armref['sha256']


if __name__=='__main__':
    assert not {'torch','numpy','torch_neuronx','transformers'}&set(sys.modules)
    unittest.main(verbosity=2)
