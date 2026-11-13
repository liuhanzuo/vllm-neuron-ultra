"""Complete-sized synthetic JSON comparisons; never reads real payloads."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import minicpm5_retained_raw_transport_v1 as t
import minicpm5_retained_raw_compare_v1 as c
from test_minicpm5_retained_raw_transport_v1 import fixture


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mini-compare-test-',dir='/tmp');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.storage=fixture(self.root)
        self.old=self.root/'old';self.new=self.root/'new'
        self.old.mkdir();self.new.mkdir();(self.old/'analysis').mkdir();(self.new/'analysis').mkdir()
        self.plan={'synthetic_original_plan':True};self.plan_hash=t.digest(t.canonical(self.plan)+b'\n')
        self.refs=copy.deepcopy(t.REFERENCES);self.refs['recovery_plan']['sha256']=self.plan_hash
        self.reports={};self.results={}
        for root in (self.old,self.new):
            self.write(root/'plan.json',self.plan)
            admit=dict(passed=True,payloads=11859,sha256={str(i):'a'*64 for i in range(11859)})
            if root==self.new:
                admit.update(original_capture_identities_preserved=True,physical_resolutions={r['original']['realpath']:
                    dict(logical_path=r['original']['realpath'],capture_identity=r['original'],storage_identity=r['keeper'],
                        sha256=r['sha256'],relocated=True,original_identity_is_current_stat=False)for r in self.storage['entries']})
            layout=dict(outputs=98,records=[dict(index=i,value='layout')for i in range(98)])
            report=dict(reports=[],hf_summaries=[dict(arm=i,rows=[dict(index=j,value=j/31)for j in range(32)],failed_frames=[4,31])for i in range(7)],
                full_model_qualified=False,new_speedup_claim=False,original_failure_preserved=True)
            sizes=[1,1,1,1,4,4,4,1,1,1,1,4,4,4]
            for i,size in enumerate(sizes):
                call=dict(complete=True,frames=[dict(effects=[dict(metric=1.25)for _ in range(8)])for _ in range(size)],
                    caches=[dict(value=k)for k in range(672)],cpu_rank_hooks=[dict(prediction_vectors=size)for _ in range(426)])
                ref=self.write(root/'analysis'/('call-%02d.json'%i),call);report['reports'].append({k:ref[k]for k in ('path','sha256')})
            self.reports[root]=report
            outputs={str(root/name):self.write(root/name,value)['sha256']for name,value in
                [('payload-admission.json',admit),('saved-layout-admission.json',layout),('comparison.json',report)]}
            result=dict(passed=True,phase='compare',plan_sha256=self.plan_hash,outputs=outputs,
                full_model_qualified=False,new_speedup_claim=False)
            self.results[root]=result
        self.oldref=self.write(self.old/'compare-result.json',self.results[self.old])
        self.newref=self.write(self.new/'compare-result.json',self.results[self.new])

    def write(self,path,value):
        raw=t.canonical(value)+b'\n';path.write_bytes(raw)
        return dict(path=str(path),sha256=t.digest(raw),bytes=len(raw))

    def refresh(self):
        self.results[self.new]['outputs'][str(self.new/'comparison.json')]=self.write(self.new/'comparison.json',self.reports[self.new])['sha256']
        self.newref=self.write(self.new/'compare-result.json',self.results[self.new])

    def compare(self):
        with patch.object(t,'REFERENCES',self.refs):return c._compare(self.storage,{},self.oldref,self.newref,lambda:None)

    def test_every_original_scope_matches_under_exact_namespace_translation(self):
        value=self.compare();self.assertTrue(value['full32_replay_and_all_scientific_outputs_equal'])
        self.assertEqual(value['stage_vectors13632'],13632);self.assertEqual(len(value['exact_namespace_translations']),14)

    def test_late_call_cache_difference_is_rejected_after_authenticating_changed_new_refs(self):
        path=self.new/'analysis/call-13.json';call=t.json.loads(path.read_bytes());call['caches'][-1]['value']+=1
        ref=self.write(path,call);self.reports[self.new]['reports'][-1]['sha256']=ref['sha256'];self.refresh()
        with self.assertRaisesRegex(ValueError,'byte-identical'):self.compare()

    def test_hf_last_frame_numeric_difference_and_original_failure_cannot_disappear(self):
        self.reports[self.new]['hf_summaries'][-1]['rows'][-1]['value']=0.99
        self.reports[self.new]['hf_summaries'][-1]['failed_frames']=[];self.refresh()
        with self.assertRaisesRegex(ValueError,'full32'):self.compare()

    def test_last_of98_layout_records_must_match(self):
        p=self.new/'saved-layout-admission.json';value=t.json.loads(p.read_bytes());value['records'][-1]['value']='changed'
        ref=self.write(p,value);self.results[self.new]['outputs'][str(p)]=ref['sha256'];self.refresh()
        with self.assertRaisesRegex(ValueError,'All98'):self.compare()

    def test_only_explicit_two_admission_fields_can_change(self):
        p=self.new/'payload-admission.json';value=t.json.loads(p.read_bytes());value['unexpected']='extra'
        ref=self.write(p,value);self.results[self.new]['outputs'][str(p)]=ref['sha256'];self.refresh()
        with self.assertRaisesRegex(ValueError,'admission'):self.compare()


if __name__=='__main__':unittest.main(verbosity=2)
