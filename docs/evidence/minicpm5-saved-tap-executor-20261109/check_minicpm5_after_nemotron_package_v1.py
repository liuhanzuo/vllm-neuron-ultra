"""Synthetic terminal records; no real completion or payloads are asserted."""
import copy
from pathlib import Path
import unittest

import minicpm5_after_nemotron_package_v1 as c
import nemotron9b_package_protocol_v22 as p22
import nemotron9b_package_protocol_v23 as p23


def fixture(profile='retry-v23'):
    v, _, owners, directories = c.PROFILES[profile]
    p = p23 if v == 23 else p22
    ref = lambda name: dict(path=str(c.Q/(name+'.json')), sha256='a'*64)
    spec_ref, review_ref = ref('synthetic-spec'), ref('synthetic-review')
    run = 'synthetic-mini-predecessor'
    spec = dict(schema='nemotron9b_package_runtime_v'+str(v), root_finalized=True,
        stages=list(p.STAGES), resources={s:p.resources(s) for s in (*p.STAGES,'controller')},
        source_sha256={str(c.Q/'fixture.py'):'b'*64}, controller=str(c.ROOT/f'nemotron9b_package_controller_v{v}.py'),
        worker=str(c.ROOT/f'nemotron9b_package_worker_v{v}.py'), automatic_successor_launch=False,
        speedup_denominator=None, run_id=run, control_root=str(c.ROOT/('nemotron9b-package-'+run+'-control')),
        units={s:'trn1-latest-verify-nemotron9b-package-'+run+'-'+s.replace('_','-') for s in (*p.STAGES,'controller')},
        required_completed_storage=ref('synthetic-storage'), required_residual_completion=ref('synthetic-dory'))
    review = dict(schema='nemotron9b_package_runtime_source_review_v'+str(v),passed=True,source_only=True,
        independent_review=True,reviewer='/root',blockers=[],execution_spec_sha256=spec_ref['sha256'],
        source_sha256=copy.deepcopy(spec['source_sha256']))
    prior = dict(rows=[dict(unit='synthetic-old-'+str(i)) for i in range(1149)],
        processes=[dict(pid=1000+i,start_ticks=str(100+i),ppid=1,argv=['/synthetic/old'],cgroup='0::/fixture') for i in range(owners)],
        known_admitted_directories=['/synthetic/old/'+str(i) for i in range(directories)],
        completed_storage_maintenance=spec['required_completed_storage'],
        completed_residual_review=spec['required_residual_completion'],canonical_sha256={'fixture':'c'*64})
    if profile == 'retry-v23': prior['failed_residual_saved_review']=ref('synthetic-failed-review')
    argv=['/usr/bin/python3','-B',spec['controller'],'--launch','--spec',spec_ref['path'],
          '--spec-sha256',spec_ref['sha256'],'--review',review_ref['path'],'--review-sha256',review_ref['sha256']]
    external=dict(pid=9000,start_ticks='9000',ppid=1,argv=argv,cgroup='0::/fixture')
    stage_owners={s:dict(pid=10000+i,start_ticks=str(10000+i),ppid=1,argv=['/synthetic/stage',s],cgroup='0::/fixture')
                  for i,s in enumerate(('controller',*p.STAGES))}
    refs={k:ref('synthetic-'+k) for k in ('intent','pre','admission','release','audit','terminal')}
    records=dict(intent=dict(execution_spec=spec_ref,source_review=review_ref,canonical_sha256=prior['canonical_sha256'],external_owner=external),
        admission=dict(execution_spec=spec_ref,source_review=review_ref,canonical_sha256=prior['canonical_sha256'],controller=stage_owners['controller']),
        pre=dict(launch_intent=refs['intent'],auditor=external,rows={s:dict(owner=o) for s,o in stage_owners.items()}),
        release=dict(released=True,all_current_groups_recursively_drained=True,canonical_unchanged=True,
            successful=True,failed=False,model_qualified=True,explicit_failed_release=False,audit=refs['audit'],
            automatic_successor_launch=False,new_speedup_claim=False),
        audit=dict(pre_stop=refs['pre'],canonical_unchanged=True,passed=True,model_qualified=True,
            automatic_successor_launch=False,new_speedup_claim=False,independent_retained_exits=True,failures=[],
            completion=dict(full_model_qualified=True,awaiting_external_raw_byte_audit=False,external_raw_byte_audit={'fixture':True})),
        terminal=dict(passed=True,diagnostic_complete=True,all_groups_physically_drained=True,
            successful_units_retained_for_external=True,admission=refs['admission']))
    receipt=dict(reviewer='/root',actual_tool_exit_observed=True,execution_spec=spec_ref,source_review=review_ref,
        owner=external,exact_owner_gone=True,actual_exit=dict(session_id=123,exit_code=0,tool_chunk='abc123'))
    rows=[dict(unit=spec['units'][s]) for s in ('controller',*p.STAGES)]
    summaries=[dict(stage=s,owner=o,never_launched=False,original_exit_success=True,worker_reported_complete=True,
                    worker_reported_passed=True,normal_retained_exit_success=True) for s,o in stage_owners.items()]
    captured=[(refs['pre'],list(stage_owners.values())+[external])]
    return dict(profile=profile,p=p,spec=spec,review=review,spec_ref=spec_ref,review_ref=review_ref,
        prior=prior,refs=refs,records=records,receipt=receipt,rows=rows,summaries=summaries,captured=captured)


def failed_fixture():
    f=fixture();f['receipt']['actual_exit']['exit_code']=1
    f['records']['release'].update(successful=False,failed=True,model_qualified=False,explicit_failed_release=True)
    f['records']['audit'].update(passed=False,model_qualified=False,completion=None,failures=['synthetic failure'])
    f['records']['failed']=dict(passed=False,all_started_groups_physically_drained=True)
    f['records']['terminal']=None
    extra=copy.deepcopy(f['receipt']);extra['owner'].update(pid=9001,start_ticks='9001')
    extra['owner']['argv'][3]='--audit';extra['owner']['argv'].insert(4,'--release-failed')
    extra['actual_exit']=dict(session_id=124,exit_code=0,tool_chunk='def456')
    f['failed_release_receipt']=extra
    f['captured'][0][1].append(extra['owner'])
    return f


class Completion(unittest.TestCase):
    def scope(self,f):return c.validate_scope(f['profile'],f['spec'],f['review'],f['spec_ref'],f['prior'],f['p'])
    def terminal(self,f):return c.validate_terminal(f['spec'],f['spec_ref'],f['review_ref'],f['receipt'],f['records'],f['refs'],f.get('failed_release_receipt'))

    def test_both_exact_profiles_and_terminal_success(self):
        self.assertEqual(p22.STAGES,p23.STAGES)
        self.assertEqual(len(p23.STAGES),52)
        for profile in c.PROFILES:
            f=fixture(profile);self.scope(f);out=self.terminal(f)
            self.assertTrue(out['producer_qualified']);self.assertFalse(out['new_speedup_claim'])

    def test_history_cannot_omit_service_owner_or_dory_directory(self):
        for key in ('rows','processes','known_admitted_directories'):
            f=fixture();f['prior'][key].pop()
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.scope(f)

    def test_retry_history_cannot_disappear_or_be_invented(self):
        for profile in c.PROFILES:
            f=fixture(profile)
            if profile=='retry-v23':del f['prior']['failed_residual_saved_review']
            else:f['prior']['failed_residual_saved_review']={}
            with self.subTest(profile=profile),self.assertRaises(RuntimeError):self.scope(f)

    def test_scientific_stages_resources_and_raw_audit_not_reduced(self):
        for field in ('stages','resources','units'):
            f=fixture();f['spec'][field].pop() if field=='stages' else f['spec'][field].pop('verify-raw')
            with self.subTest(field=field),self.assertRaises(RuntimeError):self.scope(f)

    def test_review_source_or_spec_binding_cannot_change(self):
        for field in ('execution_spec_sha256','source_sha256','blockers','independent_review'):
            f=fixture();f['review'][field]={'source_sha256':{},'blockers':['pending'],'independent_review':False}.get(field,'0'*64)
            with self.subTest(field=field),self.assertRaises(RuntimeError):self.scope(f)

    def test_storage_and_residual_order_is_bound(self):
        for key in ('completed_storage_maintenance','completed_residual_review'):
            f=fixture();f['prior'][key]={}
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.scope(f)

    def test_no_new_speedup_or_failed_denominator_from_completion(self):
        f=fixture();f['spec']['speedup_denominator']='failed-control'
        with self.assertRaises(RuntimeError):self.scope(f)

    def test_actual_tool_exit_and_exact_owner_required(self):
        for key in ('actual_tool_exit_observed','exact_owner_gone'):
            f=fixture();f['receipt'][key]=False
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.terminal(f)
        f=fixture();f['receipt']['owner']=dict(f['receipt']['owner'],start_ticks='1')
        with self.assertRaises(RuntimeError):self.terminal(f)

    def test_nonzero_external_exit_not_promoted(self):
        f=fixture();f['receipt']['actual_exit']['exit_code']=1
        with self.assertRaises(RuntimeError):self.terminal(f)

    def test_missing_raw_audit_not_promoted(self):
        f=fixture();f['records']['audit']['completion']['external_raw_byte_audit']={}
        with self.assertRaises(RuntimeError):self.terminal(f)

    def test_wrong_launch_args_rejected(self):
        for change in ('mode','digest','extra'):
            f=fixture();args=f['receipt']['owner']['argv']
            if change=='mode':args[3]='--check'
            elif change=='digest':args[7]='f'*64
            else:args.append('--release-failed')
            with self.subTest(change=change),self.assertRaises(RuntimeError):self.terminal(f)

    def test_every_stage_snapshot_required(self):
        f=fixture();f['records']['pre']['rows'].pop('verify-raw')
        with self.assertRaises(RuntimeError):self.terminal(f)

    def test_full_history_extended_without_modifying_old_history(self):
        f=fixture();before=copy.deepcopy(f['prior']);outcome=self.terminal(f)
        result=c.extend(f['prior'],f['spec'],f['rows'],f['summaries'],f['captured'],f['refs'],outcome,{'/synthetic/source':'a'*64})
        self.assertEqual(f['prior'],before);self.assertEqual(result['rows'][:1149],before['rows'])
        self.assertEqual(len(result['rows']),1202)
        self.assertEqual(result['processes'][:len(before['processes'])],before['processes'])
        self.assertEqual(len(result['processes']),len(before['processes'])+54)
        self.assertTrue(all(p['role']=='nemotron_package_recorded_owner' for p in result['processes'][len(before['processes']):]))
        self.assertFalse(result['full_model_qualified']);self.assertFalse(result['new_speedup_claim'])

    def test_partial_stage_or_missing_owner_cannot_extend_history(self):
        for mode in ('failed','missing','unlaunched'):
            f=fixture();outcome=self.terminal(f)
            if mode=='failed':f['summaries'][-1]['worker_reported_passed']=False
            elif mode=='missing':f['captured'][0][1].pop(0)
            else:f['summaries'][-1]['never_launched']=True
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):
                c.extend(f['prior'],f['spec'],f['rows'],f['summaries'],f['captured'],f['refs'],outcome,{})

    def test_historical_pin_cannot_be_rebound(self):
        with self.assertRaises(RuntimeError):c.merge({'/synthetic/source':'a'*64},{'/synthetic/source':'b'*64})

    def test_extension_cannot_promote_failed_outcome(self):
        f=fixture();outcome=self.terminal(f);outcome['producer_qualified']=False
        with self.assertRaises(RuntimeError):
            c.extend(f['prior'],f['spec'],f['rows'],f['summaries'],f['captured'],f['refs'],outcome,{})

    def test_actual_explicit_failed_release_can_advance_without_qualification(self):
        f=failed_fixture();outcome=self.terminal(f)
        self.assertFalse(outcome['producer_qualified']);self.assertTrue(outcome['failed_release'])
        # Unlaunched stages remain in the catalog, with their failed status.
        last=f['summaries'][-1];last.update(never_launched=True,original_exit_success=False,
            worker_reported_complete=False,worker_reported_passed=False,normal_retained_exit_success=False)
        last.pop('owner');f['captured'][0][1].pop(-3)
        result=c.extend(f['prior'],f['spec'],f['rows'],f['summaries'],f['captured'],f['refs'],outcome,{})
        self.assertEqual(len(result['rows']),1202);self.assertFalse(result['nemotron_outcome']['producer_qualified'])
        self.assertTrue(result['nemotron_stages'][-1]['never_launched'])

    def test_failed_release_requires_its_actual_successful_exit(self):
        for mode in ('missing','nonzero','not-observed','same-session','same-owner','wrong-wrapper'):
            f=failed_fixture();extra=f['failed_release_receipt']
            if mode=='missing':del f['failed_release_receipt']
            elif mode=='nonzero':extra['actual_exit']['exit_code']=1
            elif mode=='not-observed':extra['actual_tool_exit_observed']=False
            elif mode=='same-session':extra['actual_exit']['session_id']=123
            elif mode=='same-owner':extra['owner'].update(pid=9000,start_ticks='9000')
            else:extra['owner']['argv'][2]='/synthetic/unreviewed-wrapper.py'
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):self.terminal(f)

    def test_recovery_retains_original_audit_and_cannot_promote_it(self):
        f=failed_fixture();f['refs']['original_audit']=dict(path='/synthetic/original-audit.json',sha256='f'*64)
        f['records']['original_audit']=copy.deepcopy(f['records']['audit'])
        f['records']['audit'].update(recovery_only=True,success_not_promoted=True,original_audit=f['refs']['original_audit'])
        self.assertFalse(self.terminal(f)['producer_qualified'])
        f['records']['audit']['success_not_promoted']=False
        with self.assertRaises(RuntimeError):self.terminal(f)

    def test_success_does_not_accept_invented_failed_release(self):
        f=fixture();f['failed_release_receipt']=failed_fixture()['failed_release_receipt']
        with self.assertRaises(RuntimeError):self.terminal(f)


if __name__=='__main__':unittest.main(verbosity=2)
