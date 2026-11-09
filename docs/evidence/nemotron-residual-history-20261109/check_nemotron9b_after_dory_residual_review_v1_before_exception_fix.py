"""Real admitted snapshot plus synthetic terminal metadata; no actual completion."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
import nemotron9b_after_dory_residual_review_v1 as c


def fixture(start):
    start=copy.deepcopy(start);store={}
    def put(path,value):
        ref=dict(path=str(path),sha256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest())
        store[ref['path']]=copy.deepcopy(value)
        return ref
    producer=start['spec'];out=Path(producer['output']);reviewout=Path(producer['review_output'])
    owner,child=start['admission']['owner'],start['child']
    native=dict(reviewer='/root',actual_external_exit_observed=True,session_id=29930,exit_code=0,
                tool_chunk='abcdef',spec=c.PRODUCER,owner=owner,child_owner=child)
    native_ref=put(Q/'synthetic-native-exit-not-written.json',native)
    review=copy.deepcopy(producer)
    review.update(execution_mode='review',review_only_spec=True,
                  budget=c.load(c.PROPOSAL)['proposed_saved_review_budget'],
                  native_exit=native_ref,producer_spec=c.PRODUCER)
    records={
        'admission.json':start['admission'],'child-owner.json':child,'predecessor-history.json':start['history'],
        'child.json':dict(command=child['argv'],pid=child['pid']),
        'child-exit.json':dict(returncode=0,child_pid=child['pid'],child_owner=child),
        'result.json':dict(complete=True,mode='native',review=False,owner=owner,child_exit=0,
                           actual_self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)}
    for name,value in records.items():
        ref=put(out/name,value);review['source_sha256'][ref['path']]=ref['sha256']
    review['source_sha256'][native_ref['path']]=native_ref['sha256']
    review['essential_sources']=sorted(review['source_sha256'])
    review_ref=put(Q/'synthetic-review-spec-not-written.json',review)
    source_review=dict(reviewer='/root',passed=True,blockers=[],spec_sha256=review_ref['sha256'],
                       source_sha256=review['source_sha256'])
    sr_ref=put(Q/'synthetic-review-source-approval-not-written.json',source_review)
    reviewer=dict(pid=990001,start_ticks='999999999',ppid=1,cgroup='synthetic-only',argv=[
        '/usr/bin/python3','-B',review['executor'],'--mode','review','--spec',review_ref['path'],
        '--spec-sha256',review_ref['sha256'],'--review',sr_ref['path'],'--review-sha256',sr_ref['sha256']])
    prior=copy.deepcopy(start['history']);prior['processes'].extend([owner,child])
    prior['known_admitted_directories'].append(out.name)
    history_ref=put(reviewout/'predecessor-history.json',prior)
    admission=dict(spec_sha256=review_ref['sha256'],mode='review',review=True,owner=reviewer,
                   performance_measurement=False,budget=review['budget'],
                   canonical_sha256=prior['canonical_sha256'],predecessor_history=history_ref,
                   predecessor_drain=dict(complete=True,services=1149,processes=347))
    ar_ref=put(reviewout/'admission.json',admission)
    result=dict(complete=True,mode='review',review=True,owner=reviewer,actual_self_exit_claimed=False,
                full_model_qualified=False,new_speedup_claim=False)
    rr_ref=put(reviewout/'result.json',result)
    archives=dict(complete=True,original_events=6200,original_calls=6185,raw_bytes_verified=c.RAW,
                  tensor_labels=114,cache_histories=56,full_cache_hashes_reconstructed=True,
                  hidden_and_logits_bytes_verified=True)
    saved=dict(schema='dory_residual_combined_saved_review_v1',complete=True,diagnostic_only=True,
               original_events=6200,original_successful_calls=6185,original_pointwise_failures=7,
               both_original_and_new_successful_payloads_verified=True,all_producer_output_names_accounted=True,
               recorded_source_checkpoint_exit_bound=True,original_archives=archives,replay_archives=copy.deepcopy(archives),
               observations=dict(complete=True,selected_calls=89,selected_rows=153,residual_bytes_verified=175472640,
                                 injected_bytes_verified=8345600,original_pointwise_failures=7),
               checkpoint_review=dict(checkpoint_bytes_independently_authenticated=True,checkpoint_bytes_read=7950515472,
                                      original_tensors=507,mapped_physical_parameters=399,complete_packed_mapping_verified=True),
               source_file_review=start['sources'])
    for key in ('original_top1_recomputed','original_FP32_metrics_recomputed','production_lifecycle_qualified',
                'root_cause_established','full_model_qualified','new_speedup_claim','actual_loaded_libraries_JIT_verified',
                'complete_history_drain_verified_here'):saved[key]=False
    saved_ref=put(reviewout/'saved-residual-review.json',saved)
    receipt=dict(reviewer='/root',actual_external_exit_observed=True,session_id=1234,exit_code=0,
                 tool_chunk='fedcba',owner=reviewer,spec=review_ref,source_review=sr_ref,admission=ar_ref,
                 result=rr_ref,saved_review=saved_ref,predecessor_history=history_ref)
    ref=put(Q/'synthetic-review-exit-not-written.json',receipt)
    dirs=set(prior['known_admitted_directories'])|{reviewout.name}
    return dict(start=start,store=store,ref=ref,dirs=dirs,native=native_ref,review=review_ref,
                admission=ar_ref,result=rr_ref,saved=saved_ref,source_review=sr_ref,history=history_ref,
                reviewer=reviewer,out=out)


class Completion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader=c.Reader();cls.actual=c.anchor(cls.reader)

    def setUp(self):self.f=fixture(self.actual)

    def value(self,key):return self.f['store'][self.f[key]['path']]

    def run_chain(self,lookup=lambda pid:None,gone=lambda owner:None):
        # In-memory synthetic terminal metadata is not a real authenticated
        # receipt; production Reader authenticates exact bytes separately.
        return c.extend(self.f['start'],self.f['ref'],lambda ref:copy.deepcopy(self.f['store'][ref['path']]),
                        lookup=lookup,gone=gone,directories=self.f['dirs'])

    def test_real_anchor_is_admission_only_and_keeps_r9_failed(self):
        self.assertEqual((len(self.actual['history']['rows']),len(self.actual['history']['processes'])),(1149,345))
        self.assertTrue(self.actual['history']['r9_outcome']['failed_release'])
        self.assertFalse(self.actual['history']['r9_outcome']['producer_qualified'])
        self.assertEqual(len([p for p in self.actual['spec']['source_sha256']if Path(p).suffix=='.bin']),6)
        self.assertTrue(all(Path(p).suffix=='.json'for p in self.reader.pins))

    def test_synthetic_complete_chain_preserves_every_prior_row_and_owner(self):
        h,pins=self.run_chain()
        self.assertEqual(h['rows'],self.actual['history']['rows'])
        self.assertEqual(h['processes'][:345],self.actual['history']['processes'])
        self.assertEqual((len(h['processes']),len(h['known_admitted_directories'])),(348,37))
        self.assertFalse(h['current_physical_release_verified'])
        self.assertTrue(all(pins[k]==v for k,v in self.actual['spec']['source_sha256'].items()))

    def test_actual_live_native_owner_cannot_be_success(self):
        with self.assertRaisesRegex(RuntimeError,'still live'):
            self.run_chain(lookup=c.current_process)

    def test_native_live_child_cannot_be_success(self):
        child=self.actual['child']
        with self.assertRaisesRegex(RuntimeError,'still live'):
            self.run_chain(lookup=lambda pid:dict(pid=pid,start_ticks=child['start_ticks'],state='R')
                           if pid==child['pid']else None)

    def test_live_reviewer_cannot_be_success(self):
        def gone(owner):
            if owner==self.f['reviewer']:raise RuntimeError('reviewer still live')
        with self.assertRaisesRegex(RuntimeError,'reviewer still live'):self.run_chain(gone=gone)

    def test_failed_or_unobserved_review_exit_rejected(self):
        for key,value in (('exit_code',1),('actual_external_exit_observed',False),('session_id',0),('tool_chunk','')):
            with self.subTest(key=key):
                self.setUp();self.value('ref')[key]=value
                with self.assertRaises(RuntimeError):self.run_chain()

    def test_failed_or_unobserved_native_exit_rejected(self):
        for key,value in (('exit_code',1),('actual_external_exit_observed',False)):
            with self.subTest(key=key):
                self.setUp();self.value('native')[key]=value
                with self.assertRaises(RuntimeError):self.run_chain()

    def test_native_owner_replacement_rejected(self):
        self.value('native')['owner']['start_ticks']='999'
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_child_exit_owner_replacement_rejected(self):
        self.f['store'][str(self.f['out']/'child-exit.json')]['child_owner']['start_ticks']='999'
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_original_failed_history_cannot_be_omitted(self):
        self.value('history')['processes'].pop(0)
        with self.assertRaisesRegex(RuntimeError,'post-native history'):self.run_chain()

    def test_prior_service_cannot_be_omitted(self):
        self.value('history')['rows'].pop()
        with self.assertRaisesRegex(RuntimeError,'post-native history'):self.run_chain()

    def test_unaccounted_directory_rejected(self):
        self.f['dirs'].add('unknown-admission')
        with self.assertRaisesRegex(RuntimeError,'intervening'):self.run_chain()

    def test_missing_failed_native_directory_rejected(self):
        self.f['dirs'].remove('residual-replay-r1')
        with self.assertRaisesRegex(RuntimeError,'intervening'):self.run_chain()

    def test_saved_scope_cannot_be_partial(self):
        for key,value in (('original_events',6199),('original_successful_calls',6184),('original_pointwise_failures',0)):
            with self.subTest(key=key):
                self.setUp();self.value('saved')[key]=value
                with self.assertRaises(RuntimeError):self.run_chain()

    def test_both_full_archives_mandatory(self):
        for side in ('original_archives','replay_archives'):
            with self.subTest(side=side):
                self.setUp();self.value('saved')[side]['raw_bytes_verified']-=1
                with self.assertRaises(RuntimeError):self.run_chain()

    def test_residual_or_injected_scope_cannot_be_reduced(self):
        for key in ('selected_calls','selected_rows','residual_bytes_verified','injected_bytes_verified'):
            with self.subTest(key=key):
                self.setUp();self.value('saved')['observations'][key]-=1
                with self.assertRaises(RuntimeError):self.run_chain()

    def test_no_model_or_speedup_or_root_cause_promotion(self):
        for key in ('full_model_qualified','new_speedup_claim','root_cause_established','original_FP32_metrics_recomputed'):
            with self.subTest(key=key):
                self.setUp();self.value('saved')[key]=True
                with self.assertRaises(RuntimeError):self.run_chain()

    def test_original_checkpoint_mapping_required(self):
        self.value('saved')['checkpoint_review']['original_tensors']=506
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_review_source_approval_must_match_actual_command(self):
        self.value('source_review')['spec_sha256']='0'*64
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_reviewer_command_cannot_name_other_approval(self):
        # mutate all equal owner observations together so command check is exercised
        for value in (self.value('ref')['owner'],self.value('admission')['owner'],self.value('result')['owner']):
            value['argv'][-1]='0'*64
        with self.assertRaisesRegex(RuntimeError,'invocation'):self.run_chain()

    def test_source_pin_rebinding_rejected(self):
        key=next(iter(self.actual['spec']['source_sha256']))
        self.value('review')['source_sha256'][key]='0'*64
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_saved_review_must_authenticate_all_original_sources(self):
        self.value('saved')['source_file_review']['files'].popitem()
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_wrong_saved_result_namespace_rejected(self):
        self.value('ref')['saved_review']['path']='/wrong/saved-residual-review.json'
        with self.assertRaisesRegex(RuntimeError,'namespace'):self.run_chain()

    def test_duplicate_owner_cannot_be_appended(self):
        h=copy.deepcopy(self.actual['history'])
        with self.assertRaisesRegex(RuntimeError,'appended once'):c.append_owner(h,h['processes'][0])


class Reader(unittest.TestCase):
    def test_real_anchor_digest_corruption_rejected(self):
        reader=c.Reader()
        with self.assertRaisesRegex(RuntimeError,'Exact completion metadata'):
            reader(dict(c.HISTORY,sha256='0'*64))

    def test_binary_input_refused_before_open(self):
        with patch.object(c,'stable_bytes')as read:
            with self.assertRaisesRegex(RuntimeError,'JSON-only'):
                c.Reader()(dict(path='/unopened/payload.bin',sha256='0'*64))
            read.assert_not_called()

    def test_rebinding_refused_before_second_read(self):
        reader=c.Reader();reader(c.CHILD)
        with patch.object(c,'stable_bytes')as read:
            with self.assertRaisesRegex(RuntimeError,'rebind'):
                reader(dict(c.CHILD,sha256='0'*64))
            read.assert_not_called()

    def test_no_future_receipts_or_model_execution_created(self):
        self.assertFalse(list(Q.glob('synthetic-*-not-written.json')))
        self.assertFalse(any(x.split('.')[0]in{'torch','numpy','vllm','neuronxcc'}for x in sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
