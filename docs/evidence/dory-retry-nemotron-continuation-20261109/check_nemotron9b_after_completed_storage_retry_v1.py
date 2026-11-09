"""In-memory completion-chain tests; no synthetic runtime receipts are written."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nemotron9b_after_completed_storage_retry_v1 as c


def fixture(anchor):
    data = {}
    def put(path,value):
        ref = dict(path=str(path),sha256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest())
        data[ref['path']] = copy.deepcopy(value)
        return ref
    prior = copy.deepcopy(anchor['history'])
    for i in range(4):
        prior['processes'].append(dict(pid=990000+i,start_ticks=str(90000000+i),ppid=1,
                                     argv=['synthetic-test-only'],cgroup='synthetic-test-only'))
    prior['known_admitted_directories'].extend(['synthetic-native-not-written','synthetic-failed-review-not-written','synthetic-review-not-written'])
    residual_exit = dict(path=str(c.Q/'synthetic-residual-exit-not-written.json'),sha256='1'*64)
    prior['completed_residual_review'] = residual_exit
    prior['failed_residual_saved_review'] = dict(path='/synthetic/failed-review.json',sha256='9'*64)
    done = dict(history=prior,source_sha256={'/synthetic/original-source.py':'2'*64})
    _,weights,journals = c.storage.fixed_targets()
    plan = dict(completed_residual_review=residual_exit,weights=weights,journals=journals,
                source_sha256=copy.deepcopy(done['source_sha256']),protected_trees={'synthetic':{'entries':1,'sha256':'3'*64}})
    pref = put(c.PLAN,plan)
    sr = dict(schema='completed_qwen_storage_after_retry_source_review_v1',reviewer='/root',passed=True,
              blockers=[],plan=pref,source_sha256=plan['source_sha256'])
    srref = put(c.SOURCE_REVIEW,sr)
    owner = dict(pid=990100,start_ticks='90100000',ppid=1,cgroup='synthetic-test-only',argv=[
        '/usr/bin/python3','-B',str(c.storage.EXECUTOR),'--plan',pref['path'],
        '--plan-sha256',pref['sha256'],'--review',srref['path'],'--review-sha256',srref['sha256']])
    refs = {}
    refs['predecessor_history'] = put(c.OUT/c.OUTPUTS['predecessor_history'],prior)
    refs['predecessor_drain'] = put(c.OUT/c.OUTPUTS['predecessor_drain'],dict(
        complete=True,services=1149,processes=349,observations=[dict(unit=r['unit'],samples=[{},{}])for r in prior['rows']]))
    before = dict(root=32*2**30,nvme=40*2**30)
    lock = dict(device=1,inode=2)
    intent = dict(owner=owner,lock=lock,plan=pref,source_review=srref,actual_self_exit_claimed=False)
    ad = dict(owner=owner,lock=lock,plan=pref,source_review=srref,
        predecessor_history=refs['predecessor_history'],predecessor_drain=refs['predecessor_drain'],
        fully_drained=True,canonical_sha256=prior['canonical_sha256'],before_space=before,
        maintenance_only=True,performance_measurement=False)
    refs['intent'] = put(c.OUT/c.OUTPUTS['intent'],intent)
    refs['admission'] = put(c.OUT/c.OUTPUTS['admission'],ad)
    unused = dict(complete=True,matches=[],errors=[],processes=10)
    refs['protected_before'] = put(c.OUT/c.OUTPUTS['protected_before'],dict(trees=plan['protected_trees'],unused=unused))
    verified = dict(plan=pref,weights=[],journals=[],availability=[],all_five_weights_authenticated=True,
                    all_eight_journals_decompressed=True)
    for row in weights:
        verified['weights'].append(dict(path=row['path'],stat=row['stat'],bytes=row['stat']['st_size'],
                                         sha256=row['sha256_from_official_metadata']))
        verified['availability'].append(dict(path=row['path'],status=200,bytes=row['stat']['st_size'],body_bytes_read=0))
    for i,row in enumerate(journals):
        archive_stat = dict(row['stat'],st_ino=800+i,st_size=100,st_blocks=8)
        verified['journals'].append(dict(source=row['path'],source_stat=row['stat'],
            original_bytes=row['stat']['st_size'],original_sha256='4'*64,
            archive=str(c.OUT/(Path(row['path']).name+'.gz')),archive_stat=archive_stat,
            gzip_bytes=100,gzip_sha256='5'*64,independently_decompressed=True))
    refs['verification'] = put(c.OUT/c.OUTPUTS['verification'],verified)
    actions = []
    for i,row in enumerate(weights+journals):
        actions.append(dict(phase='before',path=row['path'],verification=refs['verification']))
        if i < 5:
            actions.append(dict(phase='after',retired=row['path'],restore_url=row['restore_url'],
                                sha256=row['sha256_from_official_metadata']))
        else:
            actions.append(dict(phase='after',retired=row['path'],archive=str(c.OUT/(Path(row['path']).name+'.gz')),unused=unused))
    refs['actions'] = put(c.OUT/c.OUTPUTS['actions'],actions)
    result = dict(complete=True,owner=owner,plan=pref,verification=refs['verification'],
        before_space=before,after_space=dict(root=33*2**30,nvme=55*2**30),
        retired_weight_files=5,archived_journal_files=8,
        weight_bytes_verified=sum(r['stat']['st_size'] for r in weights),journal_bytes_retained=1073741824,
        gzip_bytes=800,protected_trees_unchanged=True,original_payloads_and_runtime_unchanged=True,
        canonical_unchanged=True,original_qwen_bf16_restore_required=True,actual_self_exit_claimed=False,
        runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)
    refs['result'] = put(c.OUT/c.OUTPUTS['result'],result)
    receipt = dict(reviewer='/root',actual_external_exit_observed=True,session_id=12345,exit_code=0,
                   tool_chunk='abcdef',owner=owner,plan=pref,source_review=srref,**refs)
    exit_ref = put(c.Q/'synthetic-storage-exit-not-written.json',receipt)
    refs.update(plan=pref,source_review=srref,exit=exit_ref)
    return dict(data=data,done=done,plan=plan,receipt=receipt,exit_ref=exit_ref,refs=refs)


class Chain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reader = c.residual.Reader()
        cls.anchor = c.residual.old.anchor(reader)
        cls.template = fixture(cls.anchor)

    def setUp(self): self.f = copy.deepcopy(self.template)

    def value(self,key): return self.f['data'][self.f['refs'][key]['path']]

    def run_chain(self,*,gone=lambda owner:None,layout=lambda plan,verified:None):
        return c.extend(self.f['done'],self.f['exit_ref'],self.f['receipt'],self.f['plan'],
            lambda ref:copy.deepcopy(self.f['data'][ref['path']]),gone=gone,layout=layout)

    def test_synthetic_complete_chain_retains_all_original_history(self):
        h,pins = self.run_chain()
        self.assertEqual(h['rows'],self.anchor['history']['rows'])
        self.assertEqual(h['processes'][:-1],self.f['done']['history']['processes'])
        self.assertEqual((len(h['processes']),len(h['known_admitted_directories'])),(350,38))
        self.assertFalse(h['current_physical_release_verified'])
        self.assertEqual(pins,self.f['done']['source_sha256'])

    def test_failed_or_unobserved_exit_rejected(self):
        for key,value in [('exit_code',1),('actual_external_exit_observed',False),('session_id',0)]:
            with self.subTest(key=key):
                self.setUp();self.f['receipt'][key]=value
                with self.assertRaises(RuntimeError): self.run_chain()

    def test_wrong_plan_namespace_rejected(self):
        self.f['receipt']['plan']['path']='/wrong/plan.json'
        with self.assertRaises(RuntimeError): self.run_chain()

    def test_different_residual_completion_rejected(self):
        self.f['plan']['completed_residual_review']={'path':'/wrong','sha256':'0'*64}
        with self.assertRaises(RuntimeError): self.run_chain()

    def test_missing_predecessor_owner_rejected(self):
        self.value('predecessor_history')['processes'].pop()
        with self.assertRaises(RuntimeError): self.run_chain()

    def test_missing_predecessor_service_rejected(self):
        self.value('predecessor_history')['rows'].pop()
        with self.assertRaises(RuntimeError): self.run_chain()

    def test_incomplete_double_drain_rejected(self):
        self.value('predecessor_drain')['observations'][-1]['samples'].pop()
        with self.assertRaises(RuntimeError): self.run_chain()

    def test_source_review_must_bind_exact_plan(self):
        self.value('source_review')['plan']['sha256']='0'*64
        with self.assertRaises(RuntimeError): self.run_chain()

    def test_source_closure_cannot_be_omitted(self):
        self.value('source_review')['source_sha256'].clear()
        with self.assertRaises(RuntimeError): self.run_chain()

    def test_original_pin_cannot_be_rebound(self):
        self.f['plan']['source_sha256']['/synthetic/original-source.py']='6'*64
        self.value('source_review')['source_sha256']['/synthetic/original-source.py']='6'*64
        with self.assertRaisesRegex(RuntimeError,'rebind'): self.run_chain()

    def test_actual_invocation_cannot_point_to_other_review(self):
        bad=copy.deepcopy(self.f['receipt']['owner']);bad['argv'][-1]='0'*64
        self.f['receipt']['owner']=bad
        for key in ('intent','admission','result'):self.value(key)['owner']=copy.deepcopy(bad)
        with self.assertRaises(RuntimeError): self.run_chain()

    def test_live_maintenance_owner_rejected(self):
        def live(owner):raise RuntimeError('owner still live')
        with self.assertRaisesRegex(RuntimeError,'still live'):self.run_chain(gone=live)

    def test_wrong_original_weight_hash_rejected(self):
        self.value('verification')['weights'][4]['sha256']='0'*64
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_partial_journal_archive_rejected(self):
        self.value('verification')['journals'][-1]['independently_decompressed']=False
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_missing_weight_or_archive_rejected(self):
        for key in ('weights','journals'):
            with self.subTest(key=key):
                self.setUp();self.value('verification')[key].pop()
                with self.assertRaises(RuntimeError):self.run_chain()

    def test_restoration_availability_cannot_be_omitted(self):
        self.value('verification')['availability'].pop()
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_all_durable_actions_required(self):
        self.value('actions').pop()
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_retirement_cannot_precede_verification(self):
        actions=self.value('actions');actions[0],actions[1]=actions[1],actions[0]
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_restore_url_cannot_change_model_revision(self):
        self.value('actions')[1]['restore_url']='https://example.invalid/other-model'
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_open_or_mapped_journal_cannot_be_accepted(self):
        self.value('actions')[-1]['unused']['matches'].append({'kind':'fd'})
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_maintenance_cannot_promote_model_success(self):
        for key in ('full_model_qualified','new_speedup_claim','runtime_admitted','actual_self_exit_claimed'):
            with self.subTest(key=key):
                self.setUp();self.value('result')[key]=True
                with self.assertRaises(RuntimeError):self.run_chain()

    def test_original_weight_bytes_cannot_be_reduced(self):
        self.value('result')['weight_bytes_verified']-=1
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_original_checkpoint_restore_requirement_retained(self):
        self.value('result')['original_qwen_bf16_restore_required']=False
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_physical_output_validation_is_mandatory(self):
        def failed(plan,verified):raise RuntimeError('archive changed')
        with self.assertRaisesRegex(RuntimeError,'archive changed'):self.run_chain(layout=failed)

    def test_fresh_capacity_not_inferred_from_maintenance_success(self):
        self.value('result')['after_space']=dict(root=c.storage.BUDGET['root_floor'],nvme=c.storage.BUDGET['nvme_floor'])
        h,_=self.run_chain()
        self.assertFalse(h['current_physical_release_verified'])

    def test_failed_review_record_cannot_disappear_even_with_matching_counts(self):
        self.f['done']['history'].pop('failed_residual_saved_review')
        self.value('predecessor_history').pop('failed_residual_saved_review')
        with self.assertRaisesRegex(RuntimeError,'truncated history'):
            self.run_chain()

    def test_old_pre_retry_source_review_schema_is_not_accepted(self):
        self.value('source_review')['schema']='completed_qwen_storage_source_review_v1'
        with self.assertRaisesRegex(RuntimeError,'Final exact source review'):
            self.run_chain()


class ReaderTests(unittest.TestCase):
    def test_no_payload_path_can_be_read(self):
        with self.assertRaises(RuntimeError):c.Reader()({'path':'/tmp/model.safetensors','sha256':'0'*64})

    def test_corrupt_digest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'receipt.json';path.write_text('{}')
            with self.assertRaises(RuntimeError):c.Reader()({'path':str(path),'sha256':'0'*64})

    def test_missing_real_exit_stops_before_residual_completion(self):
        with patch.object(c.residual,'completed') as residual_done:
            with self.assertRaises(FileNotFoundError):
                c.completed({'path':'/tmp/no-such-real-maintenance-exit.json','sha256':'0'*64})
        residual_done.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
