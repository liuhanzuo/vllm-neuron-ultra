"""Synthetic retry-completion mutations; no real failed/successful exit written."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import nemotron9b_after_dory_saved_retry_v2 as c
import check_dory_saved_review_retry_contract_v1 as prior_tests
from check_nemotron9b_after_dory_residual_review_v1 import fixture as original_fixture


def fixture(previous, metadata):
    previous = copy.deepcopy(previous)
    store = copy.deepcopy(metadata)
    def put(path, value):
        ref = dict(path=str(path),sha256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest())
        store[str(path)] = copy.deepcopy(value)
        return ref
    parent = previous['parent']
    prep = c.old.load(c.PREPARATION)
    query_prep = c.old.load(c.QUERY_PREPARATION)
    observation = c.old.load(c.retry.QUERY_OBSERVATION)
    for ref,value in [(c.PREPARATION,prep),(c.QUERY_PREPARATION,query_prep),(c.retry.QUERY_OBSERVATION,observation)]:
        store[ref['path']] = value
    pins = dict(parent['source_sha256'])
    pins.update(prep['source_sha256'])
    for name in ('dory_service_query_auth_v1.py','dory_saved_review_service_query_v1.py','dory_saved_review_guard_v1.py'):
        path=str(c.Q/name)
        pins[path]=query_prep['source_sha256'][path]
    pins.update({p:r['sha256'] for p,r in observation['source_and_binary_pins'].items()})
    for ref in (c.retry.PARENT_SPEC,c.retry.PARENT_REVIEW,c.retry.QUERY_OBSERVATION,
                previous['history']['failed_residual_saved_review']):
        pins[ref['path']]=ref['sha256']
    spec=dict(schema=c.retry.SCHEMA,parent_review_spec=c.retry.PARENT_SPEC,output=str(c.OUT),
        executor=str(c.retry.EXECUTOR),reviewer=parent['reviewer'],producer_spec=parent['producer_spec'],
        native_exit=parent['native_exit'],failed_review_exit=previous['history']['failed_residual_saved_review'],
        budget=parent['budget'],query_dependency_observation=c.retry.QUERY_OBSERVATION,
        source_sha256=pins,essential_sources=sorted(pins),source_read_cap_bytes=parent['source_read_cap_bytes']+4*2**20,
        diagnostic_only=True,runtime_admitted=True,automatic_launch=False,full_model_qualified=False,new_speedup_claim=False)
    spec_ref=put(c.Q/'synthetic-saved-retry-spec-not-written.json',spec)
    review=dict(reviewer='/root',passed=True,blockers=[],spec_sha256=spec_ref['sha256'],source_sha256=pins)
    source_ref=put(c.Q/'synthetic-saved-retry-source-review-not-written.json',review)
    owner=dict(pid=990003,start_ticks='999999999',ppid=1,cgroup='synthetic-only',argv=[
        '/usr/bin/python3','-B',str(c.retry.EXECUTOR),'--spec',spec_ref['path'],'--spec-sha256',spec_ref['sha256'],
        '--review',source_ref['path'],'--review-sha256',source_ref['sha256']])
    h=previous['history']
    h_ref=put(c.OUT/'predecessor-history.json',h)
    drain=dict(complete=True,services=1149,processes=348,
        observations=[dict(unit=row['unit'],samples=[{},{}]) for row in h['rows']])
    d_ref=put(c.OUT/'predecessor-drain.json',drain)
    admission=dict(owner=owner,spec_sha256=spec_ref['sha256'],source_review=source_ref,
        parent_review_spec=c.retry.PARENT_SPEC,failed_review_exit=spec['failed_review_exit'],mode='review',review=True,
        performance_measurement=False,budget=spec['budget'],predecessor_history=h_ref,
        predecessor_drain_record=d_ref,predecessor_drain=drain,canonical_sha256=h['canonical_sha256'])
    result=dict(complete=True,owner=owner,failed_review_exit=spec['failed_review_exit'],mode='review',review=True,
        actual_self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
    sources=dict(source_and_binary_files_authenticated=True,files={p:dict(sha256=s,bytes=1)for p,s in pins.items()})
    frozen=copy.deepcopy(store[str(c.retry.OUT/'post-exit-contract.json')])
    anchor=c.old.anchor(c.old.Reader())
    original=original_fixture(anchor)
    saved=original['store'][original['saved']['path']]
    auth=dict(complete=True,dependency_observation=c.retry.QUERY_OBSERVATION,
        authenticated_files_before_import=41,authenticated_bytes_before_import=4326816,
        original_import_policy_restored=True,model_speedup_claim=False,
        guard_accounting=dict(closed=True,poisoned=False,poll_results_cached=False,guard_frequency_reduced=False,
            original_service_checks=45,persistent_service_checks=100,successful_guards=145),
        final_dependency_audit=dict(new_native_files=9,authenticated_imports={
            'dory_saved_review_service_query_v1':str(c.Q/'dory_saved_review_service_query_v1.py')}))
    values=dict(admission=admission,predecessor_history=h,predecessor_drain=drain,source_files=sources,
        frozen_contract=frozen,saved_review=saved,query_authentication=auth,result=result)
    refs={key:put(c.OUT/name,values[key])for key,name in c.OUTPUTS.items()}
    receipt=dict(reviewer='/root',actual_external_exit_observed=True,session_id=12345,exit_code=0,
        tool_chunk='abcdef',owner=owner,spec=spec_ref,source_review=source_ref,**refs)
    exit_ref=put(c.Q/'synthetic-saved-retry-exit-not-written.json',receipt)
    return dict(previous=previous,store=store,receipt=exit_ref,spec=spec_ref,source_review=source_ref,
        directories=set(h['known_admitted_directories'])|{c.OUT.name},names=set(c.OUTPUTS.values()),**refs)


class Completion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This synthetic r2 fixture uses its immutable pre-r3 history. The
        # production completed() still checks all current admitted directories.
        old=prior_tests.c;reader=c.Reader();data={}
        admission=reader(dict(path=str(old.OUT/'admission.json'),sha256=old.ANCHORS['admission.json']))
        reference=dict(path='/synthetic-only/dory-failed-tool-exit.json',sha256='1'*64)
        receipt=dict(schema='dory_residual_saved_review_r2_failed_actual_exit_v1',reviewer='/root',
            actual_external_exit_observed=True,session_id=86437,exit_code=1,tool_chunk='abcd01',
            owner=admission['owner'],spec=old.PARENT_SPEC,source_review=old.PARENT_REVIEW,
            output_files={name:dict(path=str(old.OUT/name),sha256=digest)for name,digest in old.ANCHORS.items()},
            full_model_qualified=False,new_speedup_claim=False,saved_review_completed=False)
        def read(ref):
            if ref==reference:return copy.deepcopy(receipt)
            value=reader(ref);data[ref['path']]=value;return value
        historical=read(dict(path=str(old.OUT/'predecessor-history.json'),sha256=old.ANCHORS['predecessor-history.json']))
        directories=set(historical['known_admitted_directories'])|{old.OUT.name}
        previous=old.failed_history(reference,read,gone=lambda owner:None,lookup=lambda pid:None,
                                    output_names=old.ANCHORS,directories=directories)
        cls.template=fixture(previous,data)

    def setUp(self):
        self.f=copy.deepcopy(self.template)

    def value(self,key):
        return self.f['store'][self.f[key]['path']]

    def run_chain(self,gone=lambda owner:None):
        return c.extend(self.f['previous'],self.f['receipt'],
            lambda ref:copy.deepcopy(self.f['store'][ref['path']]),gone=gone,
            directories=self.f['directories'],names=self.f['names'])

    def test_full_synthetic_chain_keeps_failed_review_and_every_prior_owner(self):
        h,pins=self.run_chain()
        prior=self.f['previous']['history']
        self.assertEqual(h['processes'][:348],prior['processes'])
        self.assertEqual(h['failed_residual_saved_review'],prior['failed_residual_saved_review'])
        self.assertEqual((len(h['rows']),len(h['processes']),len(h['known_admitted_directories'])),(1149,349,38))
        self.assertFalse(h['current_physical_release_verified'])
        self.assertEqual(h['completed_residual_review'],self.f['receipt'])

    def test_failed_retry_external_exit_rejected(self):
        self.value('receipt')['exit_code']=1
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_poll_without_exit_rejected(self):
        self.value('receipt')['actual_external_exit_observed']=False
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_live_retry_owner_rejected(self):
        def gone(owner):raise RuntimeError('retry still live')
        with self.assertRaisesRegex(RuntimeError,'still live'):self.run_chain(gone=gone)

    def test_failure_file_prevents_completion_even_with_success_result(self):
        self.f['names'].add('failure.json')
        with self.assertRaisesRegex(RuntimeError,'cleanup-failed'):self.run_chain()

    def test_query_cleanup_failure_prevents_completion(self):
        self.f['names'].add('query-cleanup-failure.json')
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_missing_query_final_record_rejected(self):
        self.f['names'].remove('service-query-authentication.json')
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_failed_prior_review_owner_cannot_be_omitted(self):
        self.value('predecessor_history')['processes'].pop()
        with self.assertRaisesRegex(RuntimeError,'truncated'):self.run_chain()

    def test_unknown_admission_cannot_be_hidden(self):
        self.f['directories'].add('unknown-review')
        with self.assertRaisesRegex(RuntimeError,'unknown admission'):self.run_chain()

    def test_wrong_invocation_rejected(self):
        for k in ('receipt','admission','result'):
            self.value(k)['owner']['argv'][-1]='0'*64
        with self.assertRaisesRegex(RuntimeError,'invocation'):self.run_chain()

    def test_single_service_observation_cannot_replace_double_drain(self):
        for key in ('admission','predecessor_drain'):
            drain=self.value(key)['predecessor_drain'] if key=='admission' else self.value(key)
            drain['observations'][0]['samples'].pop()
        with self.assertRaisesRegex(RuntimeError,'physically drained'):self.run_chain()

    def test_original_source_file_cannot_be_omitted(self):
        self.value('source_files')['files'].popitem()
        with self.assertRaisesRegex(RuntimeError,'source and dependency'):self.run_chain()

    def test_source_preparation_does_not_replace_final_source_review(self):
        self.value('source_review')['passed']=False
        with self.assertRaisesRegex(RuntimeError,'Final exact runtime'):self.run_chain()

    def test_full_raw_archive_scope_is_still_required(self):
        self.value('saved_review')['replay_archives']['raw_bytes_verified']-=1
        with self.assertRaisesRegex(RuntimeError,'Both complete'):self.run_chain()

    def test_all_residual_and_fault_bytes_still_required(self):
        self.value('saved_review')['observations']['injected_bytes_verified']-=1
        with self.assertRaisesRegex(RuntimeError,'selected residual'):self.run_chain()

    def test_original_seven_failures_cannot_be_erased(self):
        self.value('saved_review')['original_pointwise_failures']=0
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_model_or_speedup_promotion_rejected(self):
        for key in ('full_model_qualified','new_speedup_claim'):
            with self.subTest(key=key):
                self.setUp();self.value('saved_review')[key]=True
                with self.assertRaises(RuntimeError):self.run_chain()

    def test_incomplete_frozen_output_count_rejected(self):
        self.value('frozen_contract')['raw_file_count']-=1
        with self.assertRaisesRegex(RuntimeError,'full post-exit'):self.run_chain()

    def test_cached_query_results_are_rejected(self):
        self.value('query_authentication')['guard_accounting']['poll_results_cached']=True
        with self.assertRaisesRegex(RuntimeError,'reduced cadence'):self.run_chain()

    def test_missing_or_failed_guard_is_rejected(self):
        self.value('query_authentication')['guard_accounting']['successful_guards']-=1
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_final_query_poison_is_rejected(self):
        self.value('query_authentication')['guard_accounting']['poisoned']=True
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_unrestored_import_policy_rejected(self):
        self.value('query_authentication')['original_import_policy_restored']=False
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_unreviewed_query_import_rejected(self):
        self.value('query_authentication')['final_dependency_audit']['authenticated_imports']['unknown']='/unknown.py'
        with self.assertRaisesRegex(RuntimeError,'pinned dependency'):self.run_chain()

    def test_new_native_count_cannot_exceed_complete_expected_mapping_set(self):
        self.value('query_authentication')['final_dependency_audit']['new_native_files']=10
        with self.assertRaises(RuntimeError):self.run_chain()

    def test_preexisting_expected_mapping_is_not_counted_as_new(self):
        for count in(0,8,9):
            with self.subTest(delta=count):
                self.value('query_authentication')['final_dependency_audit']['new_native_files']=count
                history,_=self.run_chain()
                self.assertEqual(len(history['processes']),349)

    def test_invalid_new_mapping_counts_still_fail(self):
        for count in(-1,True,8.0,None):
            with self.subTest(delta=count):
                self.value('query_authentication')['final_dependency_audit']['new_native_files']=count
                with self.assertRaises(RuntimeError):self.run_chain()

    def test_all_nine_mapping_pins_required_even_when_delta_is_eight(self):
        self.value('query_authentication')['final_dependency_audit']['new_native_files']=8
        observation=self.f['store'][c.retry.QUERY_OBSERVATION['path']]
        observation['new_native_file_mappings'].popitem()
        with self.assertRaisesRegex(RuntimeError,'nine exact pinned'):self.run_chain()

    def test_actual_runtime_mapping_algorithm_handles_preloaded_and_missing_paths(self):
        from dory_service_query_auth_v1 import check_mapping_delta
        required={'/fixture/query-'+str(i)+'.so':dict(device=1,inode=i+1)for i in range(9)}
        first=next(iter(required));before={first:required[first]}
        self.assertEqual(len(check_mapping_delta(before,required,required)),8)
        missing=dict(required);missing.pop(first)
        with self.assertRaises(RuntimeError):check_mapping_delta({},missing,required)
        unknown=dict(required);unknown['/fixture/unapproved.so']=dict(device=1,inode=99)
        with self.assertRaises(RuntimeError):check_mapping_delta(before,unknown,required)


if __name__=='__main__':
    unittest.main(verbosity=2)
