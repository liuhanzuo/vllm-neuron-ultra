"""Actual scientific metadata plus in-memory future fixtures; no runtime launch."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import prepare_minicpm5_frame1_runtime_v1 as b
r=b.r


class BuilderControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=r.p.read_ref(r.BASE_PLAN)
        cls.prep=r.p.read_ref(b.PREPARATION)
        cls.anchors=b.actual_anchors()
        cls.pins=b.actual_sources(cls.base,cls.prep)
        cls.smol_ref=dict(path=str(b.SMOL),sha256='1'*64)
        cls.catalog_ref=dict(path='/test/current-catalog.json',sha256='2'*64)
        cls.chain=[dict(catalog=cls.smol_ref)]
        cls.catalog=dict(reviewer='/root',root_finalized=True,all_planned_launches_accounted_for=True,
                         rows=[dict(unit='trn1-memory-only-previous')],source_sha256={})

    def make(self,**updates):
        values=dict(base=self.base,catalog_ref=self.catalog_ref,catalog=self.catalog,smol_ref=self.smol_ref,chain=self.chain,
                    aliases={},new_sources=self.pins,added_json=self.anchors,run_id='r1-20261108-v1',port=29571)
        values.update(updates);return b.compose(**values)

    def test_import_has_no_frameworks(self):
        self.assertFalse({'torch','torch_neuronx','transformers','neuronx_distributed'}&set(sys.modules))

    def test_original_science_and_gates_unchanged(self):
        spec=self.make();plan=b.expanded_plan(spec,self.base)
        for key,value in self.base.items():
            if key not in ('output_root','source_sha256','inputs'):self.assertEqual(plan[key],value,key)
        self.assertEqual(plan['frame1_mlp'],r.worker.science())
        self.assertEqual(len(plan['inputs']),len(self.base['inputs'])+4)
        self.assertTrue(all(plan['source_sha256'][k]==v for k,v in self.base['source_sha256'].items()))
        self.assertTrue(all(plan['inputs'][k]==v for k,v in self.base['inputs'].items()))

    def test_original_bounds_and_no_launch_claim(self):
        spec=self.make();self.assertEqual(spec['resources'],{p:r.props(p)for p in r.PHASES})
        self.assertEqual(spec['output_cap_bytes'],192*2**20);self.assertEqual(spec['root_floor_bytes'],28*2**30)
        self.assertEqual(spec['canonical_sha256'],r.CANONICAL)
        self.assertFalse(spec['runtime_admitted']);self.assertFalse(spec['model_qualification_claim']);self.assertFalse(spec['automatic_full_gates_launch'])
        self.assertNotIn('reviewer',spec);self.assertNotIn('passed',spec)

    def test_names_distinct_and_fresh(self):
        spec=self.make();self.assertEqual(len(set(spec['units'].values())),3)
        self.assertTrue(all('r1-20261108-v1'in name for name in spec['units'].values()))
        c=copy.deepcopy(self.catalog);c['rows'].append(dict(unit=spec['units']['cpu']))
        with self.assertRaises(RuntimeError):self.make(catalog=c)

    def test_invalid_run_id(self):
        for run in ('../r1','r1-20261107-v1','r1-20261108-v1/other','', 'r1-20261108-vx'):
            with self.subTest(run=run),self.assertRaises(RuntimeError):self.make(run_id=run)

    def test_invalid_port(self):
        for port in (True,29000,30000,'29571'):
            with self.subTest(port=port),self.assertRaises(RuntimeError):self.make(port=port)

    def test_incomplete_catalog_rejected(self):
        for key,value in (('reviewer','other'),('root_finalized',False),('all_planned_launches_accounted_for',False)):
            c=copy.deepcopy(self.catalog);c[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.make(catalog=c)

    def test_exact_completed_smol_anchor_required(self):
        with self.assertRaises(RuntimeError):self.make(smol_ref=dict(path='/test/wrong.json',sha256='1'*64))
        with self.assertRaises(RuntimeError):self.make(chain=[])
        with self.assertRaises(RuntimeError):self.make(chain=[dict(catalog=self.catalog_ref)])

    def test_missing_source_and_rebound_source_rejected(self):
        for missing in (r.SELF,str(b.ROOT/'minicpm5_frame1_mlp_worker_v2.py'),str(b.ROOT/'minicpm5_frame1_mlp_replay_v2.py')):
            pins=dict(self.pins);pins.pop(missing)
            with self.subTest(missing=missing),self.assertRaises(RuntimeError):self.make(new_sources=pins)
        pins=dict(self.pins);key=next(iter(self.base['source_sha256']));pins[key]='f'*64
        with self.assertRaises(RuntimeError):self.make(new_sources=pins)

    def test_anchors_must_be_exact_original_four(self):
        anchors=copy.deepcopy(self.anchors);anchors.pop(next(iter(anchors)))
        with self.assertRaises(RuntimeError):self.make(added_json=anchors)
        anchors=copy.deepcopy(self.anchors);anchors[next(iter(anchors))]['sha256']='f'*64
        with self.assertRaises(RuntimeError):self.make(added_json=anchors)

    def test_copies_cannot_mutate_input_science(self):
        spec=self.make();spec['science']['cpu_anchors'][0]['sha256']='f'*64
        self.assertNotEqual(r.worker.science()['cpu_anchors'][0]['sha256'],'f'*64)
        spec['new_sources'].clear();self.assertTrue(self.pins)
        spec['canonical_sha256'].clear();self.assertTrue(r.CANONICAL)

    def test_lossless_pin_merge(self):
        self.assertEqual(b.merge({'a':'x'},{'a':'x','b':'y'}),{'a':'x','b':'y'})
        with self.assertRaises(RuntimeError):b.merge({'a':'x'},{'a':'z'})

    def test_missing_actual_completion_prevents_any_write(self):
        with patch.object(r.history_metadata,'load',side_effect=FileNotFoundError('No actual completed catalog')),patch.object(r.worker,'once')as write:
            with self.assertRaises(FileNotFoundError):b.build(self.catalog_ref,self.smol_ref,'r1-20261108-v1',29571)
            write.assert_not_called()

    def test_chain_selection_preserves_exact_audit_references(self):
        audit_ref=dict(path='/test/audit.json',sha256='3'*64)
        refs={name:dict(path='/test/'+name+'.json',sha256='4'*64)for name in r.smol_chain.FIELDS}
        current=dict(appended_terminal_audit=audit_ref,appended_from=r.HISTORY_ANCHOR)
        docs={audit_ref['path']:dict(references=refs),r.HISTORY_ANCHOR['path']:{}}
        selected=b.select_chain(self.smol_ref,current,lambda ref:docs[ref['path']])
        self.assertEqual(selected,[dict(catalog=self.smol_ref,terminal_audit=audit_ref,**refs)])

    def test_chain_cycle_and_wrong_floor_rejected(self):
        audit_ref=dict(path='/test/audit.json',sha256='3'*64)
        refs={name:dict(path='/test/'+name+'.json',sha256='4'*64)for name in r.smol_chain.FIELDS}
        for previous in (self.smol_ref,dict(path='/test/other-completed.json',sha256='5'*64)):
            current=dict(appended_terminal_audit=audit_ref,appended_from=previous)
            def load(ref):return dict(references=refs)if ref==audit_ref else current
            with self.assertRaises(RuntimeError):b.select_chain(self.smol_ref,current,load)

    def test_composed_spec_loads_actual_science_sources(self):
        spec=self.make();sref=dict(path='/test/in-memory-spec.json',sha256='6'*64);review_ref=dict(path='/test/in-memory-review.json',sha256='7'*64)
        review=dict(reviewer='/root',passed=True,source_only=True,independent_review=True,blockers=[],spec=sref)
        original=r.p.read_ref
        def read(ref):return spec if ref==sref else review if ref==review_ref else original(ref)
        with patch.object(r.p,'read_ref',side_effect=read),patch.object(r,'current_history',return_value=self.catalog)as completed:
            actual,plan=r.load_spec(sref,review_ref);self.assertEqual(actual,spec);self.assertEqual(plan,b.expanded_plan(spec,self.base));completed.assert_called_once_with(spec)
        self.assertFalse(Path(spec['output_root']).exists())


if __name__=='__main__':unittest.main()
