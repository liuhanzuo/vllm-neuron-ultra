"""Only in-memory synthetic future receipts; never writes a plan or exit record."""
import ast
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import prepare_minicpm5_sdk310_deb_plan_v4 as p
from check_minicpm5_sdk310_deb_sources_v1 import AdapterHarness
Q=Path(__file__).resolve().parent


def r(name):return dict(path='/synthetic/'+name+'.json',sha256='a'*64)


CATALOG = dict(path=str(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v78.json'),sha256='16083aac6255c989edda5b1e4f5a0e8cb48155a51724e386f2af2ef572421c60')


def fixture():
    successor=p.c.load(p.RETIREMENT_MANIFEST);retired=successor['actual_retirement_binding']
    cat=p.c.load(CATALOG)
    refs=dict(catalog=CATALOG,scope=retired['scope'],exit=retired['actual_exit'],audit=retired['full_audit'],handoff=cat['current_handoff'],acceptance=r('retirement-source-acceptance'),latest=cat['runs'][-1]['terminal'])
    manifest=p.c.load(p.MANIFEST);accepted=p.c.load(p.ACCEPTANCE);base=p.c.load(p.BASE)
    scope=p.c.load(refs['scope']);actual=p.c.load(refs['exit'])
    review=dict(reviewer='/root',passed=True,blockers=[],manifest=p.RETIREMENT_MANIFEST,retired_lfm_r2=copy.deepcopy(retired),retired_mini_history=copy.deepcopy(successor['retired_mini_history']))
    values={p.MANIFEST['path']:manifest,p.ACCEPTANCE['path']:accepted,p.BASE['path']:base,
        p.RETIREMENT_MANIFEST['path']:successor,refs['acceptance']['path']:review,
        refs['catalog']['path']:cat,refs['scope']['path']:scope,refs['exit']['path']:actual,
        refs['latest']['path']:p.c.load(refs['latest']),
        p.PREVIOUS_RETIREMENT_MANIFEST['path']:p.c.load(p.PREVIOUS_RETIREMENT_MANIFEST),
        p.PREVIOUS_RETIREMENT_ACCEPTANCE['path']:p.c.load(p.PREVIOUS_RETIREMENT_ACCEPTANCE)}
    for name in ('full_audit','service_exit','cleanup'):values[retired[name]['path']]=p.c.load(retired[name])
    for item in p.retirement.EXTRA:
        ended=p.c.load(item['actual_exit']);values[item['actual_exit']['path']]=ended
        values[item['recorder_exit']['path']]=p.c.load(item['recorder_exit'])
        for name in ('coordinator_owner','native_owner','cleanup_drained','outer_failure','service_exit','children_exits'):
            if name in ended['outputs']:
                ref=ended['outputs'][name];values[ref['path']]=p.c.load(ref)
    original122=copy.deepcopy(cat);original122['runs']=original122['runs'][:122]
    rebuilt=dict(catalog=original122,r2_scope=copy.deepcopy(scope))
    return refs,values,rebuilt


def build(refs,values,rebuilt,mode='download',**kw):
    with patch.object(p.history,'build',return_value=rebuilt):
        return p.build(lambda ref:copy.deepcopy(values[ref['path']]),refs['catalog'],refs['scope'],refs.get('latest',refs['exit']),mode,'/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-synthetic',retirement_acceptance_ref=refs['acceptance'],**kw)


class Tests(unittest.TestCase):
    def test_final_history_helper_and_auditor_pins(self):
        self.assertEqual(p.HISTORY_SOURCE['sha256'],'b5d957a6aaae30efefaf1a266897824b34986c61ff65579ef4ef6c7375fb821c')
        self.assertEqual(p.AUDITOR_SOURCE['sha256'],'fc4749e5a8508b2e8b59ab6703f0c595c60c24d2613032782c1aaea4cf730903')
        self.assertEqual(p.history.VERIFIERS[p.AUDITOR_SOURCE['path']],p.AUDITOR_SOURCE['sha256'])
        refs,values,rebuilt=fixture();plan=build(refs,values,rebuilt)
        self.assertEqual(plan['preparation_history_source'],p.HISTORY_SOURCE)
        self.assertEqual(plan['required_lfm_full_auditor_source'],p.AUDITOR_SOURCE)
    def test_complete_old_history_and_all874_children_retained(self):
        refs,values,rebuilt=fixture();plan=build(refs,values,rebuilt);base=values[p.BASE['path']]
        self.assertEqual(plan['predecessor_owners'][:1738],base['predecessor_owners'])
        self.assertEqual(len(plan['predecessor_owners']),2810)
        self.assertEqual(plan['predecessor_services'][:1319],base['predecessor_services'])
        self.assertEqual(len({r['unit'] for r in plan['predecessor_services']}),1454)
        self.assertEqual(len(plan['predecessor_services']),1454+1)
        self.assertEqual(plan['predecessor_owners'][-874:],values[refs['scope']['path']]['children'])
        self.assertEqual(len(plan['source_sha256']),98)
        self.assertEqual(plan['mini_scientific_constraints'],p.binding.CONSTRAINTS)
        self.assertFalse(plan['runtime_admitted']);self.assertTrue(plan['root_plan_review_required'])
    def test_real_current121_catalog_cannot_be_future122(self):
        refs,values,rebuilt=fixture();values[refs['catalog']['path']]['runs']=values[refs['catalog']['path']]['runs'][:121]
        with self.assertRaises(RuntimeError):build(refs,values,rebuilt)
    def test_missing_actual_terminal_never_builds_plan(self):
        refs,values,rebuilt=fixture();del values[refs['exit']['path']]
        with self.assertRaises(KeyError):build(refs,values,rebuilt)
    def test_history_helper_rejection_propagates(self):
        refs,values,_=fixture()
        with patch.object(p.history,'build',side_effect=RuntimeError('Not actually ended/audited')):
            with self.assertRaises(RuntimeError):p.build(lambda ref:values[ref['path']],refs['catalog'],refs['scope'],refs['exit'],'download','/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-synthetic',retirement_acceptance_ref=refs['acceptance'])
    def test_missing_child_and_modified_old_catalog_rejected(self):
        for kind in ('child','old'):
            refs,values,rebuilt=fixture()
            if kind=='child':values[refs['scope']['path']]['children'].pop()
            else:values[refs['catalog']['path']]['runs'][0]['owners']=[]
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):build(refs,values,rebuilt)
    def test_catalog_tool_and_actual_tool_must_match(self):
        refs,values,rebuilt=fixture();values[refs['exit']['path']]['actual_tool']['exit_code']=1
        with self.assertRaises(RuntimeError):build(refs,values,rebuilt)
    def test_frozen_source_acceptance_and_namespace_required(self):
        refs,values,rebuilt=fixture();values[p.ACCEPTANCE['path']]['passed']=False
        with self.assertRaises(RuntimeError):build(refs,values,rebuilt)
        refs,values,rebuilt=fixture()
        with self.assertRaises(RuntimeError):p.build(lambda ref:values[ref['path']],refs['catalog'],refs['scope'],refs['exit'],'download','/mnt/nvme/other',retirement_acceptance_ref=refs['acceptance'])
    def test_conflicting_service_invocation_is_not_overwritten(self):
        refs,values,rebuilt=fixture();base=values[p.BASE['path']];base['predecessor_services'].append(dict(unit=p.history.UNIT,invocation_id='wrong'))
        with self.assertRaises(RuntimeError):build(refs,values,rebuilt)
    def test_download_rejects_replay_references(self):
        refs,values,rebuilt=fixture()
        with self.assertRaises(RuntimeError):build(refs,values,rebuilt,replay_from=r('wrong'))
    def test_replay_requires_actual_download_and_actual_recorder(self):
        refs,values,rebuilt=fixture();h=AdapterHarness();h.run();saved=h.saved[-1]
        result=r('download-result');producer=r('producer-exit');recorder=r('recorder-exit')
        producer_tool=dict(kind='tool_session',session_id=9001,exit_code=0,terminal_chunk='abcdef')
        recorder_tool=dict(kind='tool_session',session_id=9002,exit_code=0,terminal_chunk='abcdef')
        values[result['path']]=saved;values[p.binding.NEXT['path']]=h.n
        values[producer['path']]=dict(schema='minicpm5_sdk310_deb_stage_actual_exit_v1',stage_success_observed=True,mode='download',actual_tool=producer_tool,owners_gone=True,child_group_empty=True,result=result)
        values[recorder['path']]=dict(actual_tool=recorder_tool,owner_gone=True,result=producer)
        values[refs['catalog']['path']]['runs'].extend([dict(terminal=producer,tool=producer_tool,owners=[dict(pid=1002,start_ticks='22')],units=[],source_sha256={}),dict(terminal=recorder,tool=recorder_tool,owners=[dict(pid=1003,start_ticks='23')],units=[],source_sha256={})])
        refs['latest']=recorder
        kw=dict(replay_from=result,producer_exit_ref=producer,producer_recorder_exit_ref=recorder)
        plan=build(refs,values,rebuilt,'replay',**kw);self.assertEqual(plan['replay_from'],result)
        self.assertEqual(plan['predecessor_exit'],recorder)
        for kind in ('failed_download','recorder_live','wrong_result'):
            changed=copy.deepcopy(values)
            if kind=='failed_download':changed[producer['path']]['stage_success_observed']=False
            elif kind=='recorder_live':changed[recorder['path']]['owner_gone']=False
            else:changed[producer['path']]['result']=r('other')
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):build(refs,changed,rebuilt,'replay',**kw)
    def test_new_manifest_all88_old_pins_and_exact98_required(self):
        for kind in ('remove','change','extra'):
            refs,values,rebuilt=fixture();pins=values[p.RETIREMENT_MANIFEST['path']]['source_sha256'];old=next(iter(values[p.MANIFEST['path']]['source_sha256']))
            if kind=='remove':pins.pop(old)
            elif kind=='change':pins[old]='f'*64
            else:pins['/synthetic/extra.py']='f'*64
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):build(refs,values,rebuilt)
    def test_separate_root_retirement_source_acceptance_required(self):
        for kind in ('missing','failed','wrong_manifest','wrong_binding'):
            refs,values,rebuilt=fixture();review=values[refs['acceptance']['path']]
            if kind=='missing':values.pop(refs['acceptance']['path'])
            elif kind=='failed':review['passed']=False
            elif kind=='wrong_manifest':review['manifest']=r('wrong')
            else:review['retired_lfm_r2']['invocation_id']=''
            with self.subTest(kind=kind),self.assertRaises((RuntimeError,KeyError)):build(refs,values,rebuilt)
    def test_actual_retirement_binding_preserves_known_invocation(self):
        refs,values,rebuilt=fixture();plan=build(refs,values,rebuilt)
        self.assertEqual(plan['retired_lfm_r2'],values[p.RETIREMENT_MANIFEST['path']]['actual_retirement_binding'])
        self.assertEqual([s for s in plan['predecessor_services'] if s['unit']==p.retirement.base.UNIT],[dict(unit=p.retirement.base.UNIT,invocation_id=p.retirement.base.INVOCATION)])
        self.assertEqual(plan['source_manifest'],p.RETIREMENT_MANIFEST)
        self.assertEqual(plan['source_only_acceptance'],refs['acceptance'])
        self.assertEqual(plan['prior_source_manifest'],p.MANIFEST)
    def test_real_retirement_failure_propagates_without_mock(self):
        refs,values,rebuilt=fixture();values[refs['audit']['path']]['physical_drain']['service_retirement_verified']=False
        with self.assertRaises(RuntimeError):build(refs,values,rebuilt)
    def test_all93_previously_tested_pins_must_remain(self):
        refs,values,rebuilt=fixture();pins=values[p.RETIREMENT_MANIFEST['path']]['source_sha256']
        pins[str(Q/'run_minicpm5_sdk310_deb_members_root_v2.py')]='f'*64
        with self.assertRaises(RuntimeError):build(refs,values,rebuilt)
    def test_real_failed_download_stays_latest_exit1_and_full_owner_retained(self):
        refs,values,rebuilt=fixture();plan=build(refs,values,rebuilt)
        self.assertEqual(plan['predecessor_exit'],refs['latest']);self.assertEqual(plan['predecessor_expected_exit_code'],1)
        self.assertEqual(values[refs['latest']['path']]['actual_tool'],dict(kind='tool_session',session_id=74561,terminal_chunk='6b73fb',exit_code=1))
        self.assertIn(values[refs['latest']['path']]['owner'],plan['predecessor_owners'])
    def test_both_extra_retirements_require_exact_root_acceptance(self):
        for kind in ('omitted','changed'):
            refs,values,rebuilt=fixture();review=values[refs['acceptance']['path']]
            if kind=='omitted':review.pop('retired_mini_history')
            else:review['retired_mini_history'][0]['actual_tool']['exit_code']=0
            with self.subTest(kind=kind),self.assertRaises((KeyError,RuntimeError)):build(refs,values,rebuilt)
        refs,values,rebuilt=fixture();plan=build(refs,values,rebuilt)
        self.assertEqual(plan['retired_mini_history'],values[p.RETIREMENT_MANIFEST['path']]['retired_mini_history'])
    def test_no_runtime_calls_or_review_handoff_creation_in_source(self):
        source=Path(p.__file__).read_text();tree=ast.parse(source)
        calls={ast.unparse(x.func)for x in ast.walk(tree)if isinstance(x,ast.Call)}
        for forbidden in ('subprocess.run','subprocess.Popen','os.system','runtime.acquire','binding.authenticated','io.download','io.inspect'):
            self.assertNotIn(forbidden,calls)
        self.assertIn("not destination.name.startswith('continuation-handoff-')",source)
        self.assertNotIn("reviewer='/root'",source)


if __name__=='__main__':unittest.main(verbosity=2)
