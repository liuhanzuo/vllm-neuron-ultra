"""Actual old metadata with future completions mocked solely in test memory."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'research')]
import prepare_nemotron9b_package_runtime_v24 as b
from research import nemotron9b_completed_storage_metadata_after_retry_v2 as extras


class Builder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=b.p.metadata.load(b.PARENT);cls.catalog=b.p.metadata.load(b.CATALOG)
        reader=b.completion.residual.Reader();cls.anchor=b.completion.residual.old.anchor(reader)
        cls.sources=b.merge(cls.anchor['spec']['source_sha256'],reader.pins)
        cls.pins={name:b.p.metadata.sha(name)for name in b.p.required_sources()}
        cls.pins.update(b.downstream_pins())
        cls.pins.update({ref['path']:ref['sha256']for ref in b.TEST_REFS})
        cls.observation=b.p.metadata.load(b.QUERY_OBSERVATION)
        cls.query_pins={p:row['sha256']for p,row in cls.observation['source_and_binary_pins'].items()}
        cls.query_pins[b.QUERY_OBSERVATION['path']]=b.QUERY_OBSERVATION['sha256']
        cls.sources=b.merge(cls.sources,cls.query_pins)
        cls.residual_exit=dict(path=str(b.Q/'synthetic-v24-residual-exit-not-written.json'),sha256='a'*64)
        cls.storage_exit=dict(path=str(b.Q/'synthetic-v24-storage-exit-not-written.json'),sha256='b'*64)
        cls.observer=dict(path=str(b.Q/'nemotron9b-package-runtime-v24-fixture-observer.json'),sha256='c'*64)

    def setUp(self):
        h=copy.deepcopy(self.anchor['history'])
        h['processes'].extend([dict(pid=990001+i,start_ticks=str(99999999+i))for i in range(4)])
        h['known_admitted_directories']+=['residual-replay-r2','residual-replay-review-r2','residual-replay-review-r3']
        h['completed_residual_review']=self.residual_exit
        h['failed_residual_saved_review']=dict(path='/synthetic/failed-review-r2.json',sha256='f'*64)
        self.residual_proof=dict(history=h,
            source_sha256=b.merge(self.sources,{self.residual_exit['path']:self.residual_exit['sha256']}),
            historical_binary_sha256=b.previous.partition.exact_dory_payloads(self.anchor),
            runtime_admitted=False,fresh_physical_drain_verified=False,full_model_qualified=False,new_speedup_claim=False)
        self.storage_proof=copy.deepcopy(self.residual_proof)
        self.storage_proof['history']['processes'].append(dict(pid=990100,start_ticks='99999900'))
        self.storage_proof['history']['completed_storage_maintenance']=self.storage_exit
        self.storage_proof['source_sha256']=b.merge(self.storage_proof['source_sha256'],extras.EXACT_TEXT,
            {self.storage_exit['path']:self.storage_exit['sha256']})

    def compose(self,**changes):
        args=dict(parent=self.parent,catalog=self.catalog,run_id='r10-v24-builder-fixture',
            observer_ref=self.observer,pins=self.pins,storage_exit_ref=self.storage_exit)
        args.update(changes)
        with patch.object(b.completion.residual,'completed',return_value=self.residual_proof), \
             patch.object(b.completion.maintenance,'completed',return_value=self.storage_proof):
            return b.compose(**args)

    def test_all_original_science_and_twenty_payload_pins_retained(self):
        before=copy.deepcopy(self.parent);spec=self.compose()
        self.assertEqual(before,self.parent)
        self.assertEqual(set(spec)-set(self.parent),b.ADDED)
        for key,value in self.parent.items():
            if key not in b.CHANGED:self.assertEqual(spec[key],value,key)
        parts=b.p.metadata.classify_sources(spec,self.catalog)
        self.assertEqual(len(parts['historical_payload_sha256']),20)
        self.assertEqual(len(parts['query_native_binary_sha256']),9)
        self.assertEqual(len(parts['python_bytes_sha256']),3)
        self.assertEqual(spec['required_saved_retry_query_observation'],b.QUERY_OBSERVATION)
        self.assertEqual(len(spec['historical_payload_sha256']),14)
        self.assertEqual(len(spec['stages']),52)
        self.assertIsNone(spec['speedup_denominator'])
        self.assertFalse(spec['runtime_readiness']['ready'])
        self.assertEqual(spec['required_completed_storage'],self.storage_exit)
        self.assertLessEqual(len(b.encoded(b.Q/'nemotron9b-package-runtime-v24-fixture-spec.json',spec)),32*2**20)

    def test_changed_model_input_science_or_budget_rejected(self):
        for key in ('package_commit','stages','resources','disk_budget','payloads','references'):
            changed=copy.deepcopy(self.parent);changed[key]=None
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'Exact actual admitted r9'):
                self.compose(parent=changed)

    def test_rebound_source_or_missing_new_worker_rejected(self):
        pins=dict(self.pins);pins[next(iter(self.parent['source_sha256']))]='0'*64
        with self.assertRaisesRegex(RuntimeError,'Never rebind'):self.compose(pins=pins)
        pins=dict(self.pins);pins.pop(str(R/'nemotron9b_package_worker_v24.py'))
        with self.assertRaisesRegex(RuntimeError,'Complete new source'):self.compose(pins=pins)

    def test_missing_original_dory_payload_pin_rejected(self):
        key=next(iter(self.storage_proof['historical_binary_sha256']))
        self.storage_proof['source_sha256'].pop(key);self.storage_proof['historical_binary_sha256'].pop(key)
        with self.assertRaises(RuntimeError):self.compose()

    def test_actual_failed_predecessor_namespace_cannot_be_reused(self):
        with self.assertRaisesRegex(RuntimeError,'Fresh explicit'):self.compose(run_id=self.parent['run_id'])

    def test_incomplete_maintenance_history_rejected(self):
        for key in ('rows','processes','known_admitted_directories'):
            self.setUp();self.storage_proof['history'][key].pop()
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.compose()

    def test_original_r9_failure_cannot_be_promoted(self):
        self.residual_proof['history']['r9_outcome']['producer_qualified']=True
        with self.assertRaises(RuntimeError):self.compose()

    def test_missing_completion_ref_rejected_before_future_read_or_write(self):
        with patch.object(b.completion.maintenance,'completed') as done,patch.object(b.p,'once') as write:
            with self.assertRaisesRegex(RuntimeError,'Actual completed storage'):
                b.compose(self.parent,self.catalog,'r10-fixture',self.observer,self.pins,None)
        done.assert_not_called();write.assert_not_called()

    def test_missing_actual_completion_creates_no_spec(self):
        self.assertFalse(Path(self.storage_exit['path']).exists())
        before=set(b.Q.glob('nemotron9b-package-runtime-v24-*'))
        with patch.object(b.p,'once') as write:
            with self.assertRaises(FileNotFoundError):b.build('r10-fixture',self.storage_exit)
        write.assert_not_called()
        self.assertEqual(set(b.Q.glob('nemotron9b-package-runtime-v24-*')),before)

    def test_failed_or_live_maintenance_blocks_all_parent_authentication_and_output(self):
        with patch.object(b.completion.maintenance,'completed',side_effect=RuntimeError('actual success missing')), \
             patch.object(b.p.metadata,'load') as load,patch.object(b.p,'once') as write:
            with self.assertRaisesRegex(RuntimeError,'actual success missing'):b.build('r10-fixture',self.storage_exit)
        load.assert_not_called();write.assert_not_called()

    def test_observer_contract_keeps_root_only_participation(self):
        old=b.p.metadata.load(self.parent['external_observer_contract'])
        new=b.observer_policy(self.parent,old,'f'*64)
        self.assertEqual(new['participants'],['/root'])
        self.assertEqual({k:v for k,v in new.items()if k!='source_sha256'},
                         {k:v for k,v in old.items()if k!='source_sha256'})

    def test_no_runtime_spec_framework_import_or_automatic_launch(self):
        self.assertFalse(list(b.Q.glob('nemotron9b-package-runtime-v24-*-spec.json')))
        self.assertFalse(any(name.split('.')[0]in{'torch','numpy','vllm','neuronxcc'}for name in sys.modules))
        spec=self.compose()
        self.assertFalse(spec['automatic_successor_launch']);self.assertEqual(spec['retirement_policy'],'none')

    def test_query_pin_categories_required_without_unknown_native_expansion(self):
        native=next(iter(self.observation['new_native_file_mappings']))
        python=next(p for p in self.query_pins if p.endswith('.py') and p not in self.pins)
        required=str(b.Q/'dory_saved_review_service_query_v1.py')
        for key in (native,python,required,b.QUERY_OBSERVATION['path']):
            self.setUp();self.storage_proof['source_sha256'].pop(key)
            pins=dict(self.pins);pins.pop(key,None)
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.compose(pins=pins)
        self.setUp()
        self.storage_proof['source_sha256']['/synthetic/unreviewed.so']='0'*64
        with self.assertRaises(RuntimeError):self.compose()

    def test_failed_original_review_retained_and_distinct_from_success(self):
        self.residual_proof['history']['failed_residual_saved_review']=self.residual_exit
        with self.assertRaises(RuntimeError):self.compose()

    def test_native_bytes_accounted_separately_from_model_payloads(self):
        parts=dict(text_sha256={},python_bytes_sha256={},historical_payload_sha256={},
                   query_native_binary_sha256={'/synthetic/query.so':'a'*64})
        with patch.object(b.p.metadata,'classify_sources',return_value=parts), \
             patch.object(b.p.metadata,'load',return_value=self.observation), \
             patch.object(b.p.metadata.query_sources,'authenticate_native_sources',return_value=dict(
                 files=9,bytes=3920408,native_libraries_loaded_here=False)) as authenticate:
            stats=b.current_source_checks(dict(payloads={},runtime_binary_stat={}),{}, {})
        authenticate.assert_called_once_with(parts['query_native_binary_sha256'],self.observation,b.p.quiet)
        self.assertEqual(stats['query_native_files_rehashed'],9)
        self.assertEqual(stats['query_native_bytes_rehashed'],3920408)
        self.assertEqual(stats['payload_bytes_read'],0)
        self.assertEqual(stats['historical_raw_reads'],0)
        self.assertFalse(stats['query_native_libraries_loaded'])

    def test_query_dependency_failure_stops_preparation(self):
        with patch.object(b.p.metadata,'classify_sources',return_value=dict(text_sha256={},
                 python_bytes_sha256={},query_native_binary_sha256={})), \
             patch.object(b.p.metadata,'load',return_value={}), \
             patch.object(b.p.metadata.query_sources,'authenticate_native_sources',side_effect=RuntimeError('query changed')), \
             patch.object(b.p,'file_stat') as payload_stat:
            with self.assertRaisesRegex(RuntimeError,'query changed'):b.current_source_checks({}, {}, {})
        payload_stat.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
