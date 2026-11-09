"""Concrete builder checks against admitted r8 and actual latest completion."""
import copy
from pathlib import Path
import sys
import unittest

R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R),str(R/'research')]
import prepare_nemotron9b_package_runtime_v18 as b


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=b.p.metadata.load(b.PARENT)
        cls.catalog=b.p.metadata.load(b.CATALOG)
        cls.pins={path:b.p.metadata.sha(path) for path in b.p.required_sources()}
        cls.pins[str(b.TESTS_PATH)]='a'*64
        cls.observer=dict(path=str(b.Q/'nemotron9b-package-runtime-v18-fixture-observer.json'),sha256='a'*64)

    def compose(self,**changes):
        args=dict(parent=self.parent,catalog=self.catalog,run_id='r9-builder-fixture',observer_ref=self.observer,pins=self.pins)
        args.update(changes)
        return b.compose(**args)

    def test_all_original_model_scientific_and_provenance_fields_preserved(self):
        value=self.compose()
        self.assertEqual(set(value)-set(self.parent),b.ADDED)
        for key,expected in self.parent.items():
            if key not in b.CHANGED:self.assertEqual(value[key],expected,key)
        self.assertEqual(len(value['stages']),52)
        self.assertFalse(value['runtime_readiness']['ready'])
        self.assertIsNone(value['speedup_denominator'])

    def test_actual_full_history_and_completed_maintenance(self):
        proof=b.validate_history(self.compose(),self.catalog)
        self.assertEqual((proof['complete_services'],proof['complete_processes']),(1096,291))
        self.assertFalse(proof['fresh_physical_drain_proven'])

    def test_model_and_budget_and_original_stages_cannot_be_relaxed(self):
        for field in ['package_commit','stages','resources','disk_budget','automatic_successor_launch','speedup_denominator']:
            value=copy.deepcopy(self.parent)
            if field=='package_commit':value[field]='0'*40
            elif field=='stages':value[field].remove('long')
            elif field=='resources':value[field]['compile']['MemoryMax']='1'
            elif field=='disk_budget':value[field]['root_floor']-=1
            elif field=='automatic_successor_launch':value[field]=True
            else:value[field]='failed-r8'
            with self.subTest(field=field),self.assertRaises(RuntimeError):self.compose(parent=value)

    def test_old_source_pin_cannot_be_rebound(self):
        pins=dict(self.pins)
        pins[next(iter(self.parent['source_sha256']))]='0'*64
        with self.assertRaisesRegex(RuntimeError,'Never rebind'):self.compose(pins=pins)

    def test_new_worker_cannot_be_omitted(self):
        pins=dict(self.pins);del pins[str(R/'nemotron9b_package_worker_v18.py')]
        with self.assertRaisesRegex(RuntimeError,'Complete new source'):self.compose(pins=pins)

    def test_admitted_r8_service_name_cannot_be_reused(self):
        with self.assertRaises(RuntimeError):self.compose(run_id=self.parent['run_id'])

    def test_unsafe_new_run_names_rejected(self):
        for name in ['../escape','bad_name','x'*41]:
            with self.assertRaises(RuntimeError):self.compose(run_id=name)

    def test_only_root_observer_is_retained(self):
        old=b.p.metadata.load(self.parent['external_observer_contract'])
        new=b.observer_policy(self.parent,old,'f'*64)
        self.assertEqual(new['participants'],['/root'])
        self.assertEqual({k:v for k,v in old.items() if k!='source_sha256'},
                         {k:v for k,v in new.items() if k!='source_sha256'})


if __name__=='__main__':unittest.main(verbosity=2)
