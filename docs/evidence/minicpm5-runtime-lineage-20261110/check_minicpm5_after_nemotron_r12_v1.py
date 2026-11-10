"""Synthetic metadata/service doubles only. Never enroll or drain the live r12."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import minicpm5_after_nemotron_r12_v1 as h


def owner(pid, argv=None):
    return dict(pid=pid, start_ticks=str(pid+100), ppid=1,
        argv=argv or ['python3', '-B', '/synthetic/worker.py'], cgroup='0::/synthetic')


def ref(name): return dict(path='/synthetic/'+name+'.json', sha256='a'*64)


def fixture(success=True):
    spec = dict(schema='nemotron9b_package_runtime_v27', run_id='r12-20261110', root_finalized=True,
        stages=list(h.p.STAGES), controller=str(h.ROOT/'nemotron9b_package_controller_v27.py'),
        worker=str(h.ROOT/'nemotron9b_package_worker_v27.py'),
        control_root='/mnt/nvme/nemotron9b-package-r12-20261110-control',
        evidence_root='/mnt/nvme/nemotron9b-package-r12-20261110-evidence',
        work_root='/mnt/nvme/nemotron9b-package-r12-20261110',
        units={s:'trn1-latest-verify-nemotron9b-package-r12-20261110-'+s.replace('_','-') for s in (*h.p.STAGES,'controller')},
        resources={s:h.p.resources(s) for s in (*h.p.STAGES,'controller')}, source_sha256={},
        final_predecessor_catalog=ref('catalog'), automatic_successor_launch=False,
        speedup_denominator=None, post_r11_completion=copy.deepcopy(h.p.successor.BINDING))
    review=dict(schema='nemotron9b_package_runtime_source_review_v27', execution_spec_sha256=h.SPEC['sha256'],
        source_sha256={}, reviewer='/root', passed=True, source_only=True, independent_review=True, blockers=[])
    prior=dict(rows=[dict(unit='old-unit-%d'%i) for i in range(1255)], processes=[owner(10000+i) for i in range(806)],
        canonical_sha256='c'*64, nemotron_r11_outcome=dict(failed_release=True, producer_qualified=False))
    current=[owner(20000+i) for i in range(5)]
    previous=dict(previous=dict(history=prior, source_sha256={}), source_sha256={}, handoff=dict(canonical_sha256='c'*64),
        lock_identity={'device':1,'inode':2}, current_history=dict(decode_calls=180,previous_control_calls=108,
        raw_bytes=2949120,owners=current,complete_model_and_independent_auditor_authenticated=True),
        official_admission=dict(later_owners=[owner(21000)]),
        exec_variants=[owner(22000,['python3','/synthetic/before.py']),owner(22000,['python3','/synthetic/after.py'])])
    binding=dict(schema=h.SCHEMA,actual_exit=ref('actual-exit'),failed_release_exit=None if success else ref('failed-release-exit'))
    receipt=dict(reviewer='/root',actual_tool_exit_observed=True,execution_spec=h.SPEC,source_review=h.REVIEW,
        owner=h.LIVE_OWNER,exact_owner_gone=True,actual_exit=dict(session_id=h.LIVE_SESSION,exit_code=0 if success else 1,tool_chunk='synthetic-exit'))
    live=dict(schema='nemotron9b_r12_actual_external_live_v1',actual_live_identity_verified=True,external_owner=h.LIVE_OWNER,
        actual_execution_handle=dict(session_id=h.LIVE_SESSION),execution_spec=h.SPEC,source_review=h.REVIEW)
    refs={k:ref(k) for k in ('intent','pre','admission','release','audit','terminal','failed','original_audit')}
    if success:refs.pop('failed')
    else:refs.pop('terminal')
    stages=['controller',*h.p.STAGES];stage_owners={s:owner(30000+i) for i,s in enumerate(stages)}
    pre=dict(launch_intent=refs['intent'],rows={s:dict(stage=s,owner=stage_owners[s],properties={}) for s in stages},auditor=h.LIVE_OWNER)
    admission=dict(execution_spec=h.SPEC,source_review=h.REVIEW,canonical_sha256='c'*64,
        lock_identity=previous['lock_identity'],controller=stage_owners['controller'])
    intent=dict(admission,external_owner=h.LIVE_OWNER)
    release=dict(released=True,all_current_groups_recursively_drained=True,canonical_unchanged=True,successful=success,
        failed=not success,model_qualified=success,explicit_failed_release=not success,audit=refs['audit'],
        automatic_successor_launch=False,new_speedup_claim=False)
    audit=dict(pre_stop=refs['pre'],canonical_unchanged=True,passed=success,model_qualified=success,
        automatic_successor_launch=False,new_speedup_claim=False,independent_retained_exits=True,failures=[] if success else ['synthetic failure'],
        completion=dict(full_model_qualified=success,awaiting_external_raw_byte_audit=False,external_raw_byte_audit={'synthetic':True}))
    records=dict(intent=intent,pre=pre,admission=admission,release=release,audit=audit)
    if success:records['terminal']=dict(passed=True,diagnostic_complete=True,all_groups_physically_drained=True,
        successful_units_retained_for_external=True,admission=refs['admission'])
    else:records['failed']=dict(passed=False,all_started_groups_physically_drained=True)
    data={binding['actual_exit']['path']:receipt,h.LIVE['path']:live,h.SPEC['path']:spec,h.REVIEW['path']:review,
        spec['final_predecessor_catalog']['path']:{'synthetic_catalog':True}}
    if not success:
        release_owner=owner(40000,[h.p.PYTHON,'-B',spec['controller'],'--audit','--release-failed',
            '--spec',h.SPEC['path'],'--spec-sha256',h.SPEC['sha256'],'--review',h.REVIEW['path'],'--review-sha256',h.REVIEW['sha256']])
        data[binding['failed_release_exit']['path']]=dict(receipt,owner=release_owner,
            actual_exit=dict(session_id=99999,exit_code=0,tool_chunk='synthetic-release'))
    rows=[dict(unit=spec['units'][s],owner=stage_owners[s],never_launched=False) for s in stages]
    summaries=[dict(stage=s,owner=stage_owners[s],never_launched=False,original_exit_success=success,
        worker_reported_complete=success,worker_reported_passed=success,normal_retained_exit_success=success) for s in stages]
    return dict(spec=spec,review=review,previous=previous,binding=binding,data=data,refs=refs,records=records,rows=rows,summaries=summaries)


class CompletionChecks(unittest.TestCase):
    def enroll(self,f):
        stages=['controller',*h.p.STAGES];seen=[]
        def read(ref):return copy.deepcopy(f['data'][ref['path']])
        def retained(spec,stage,spec_ref,review_ref,protocol,control,saved,read):
            self.assertIs(protocol,h.p);self.assertEqual(spec_ref,h.SPEC);self.assertEqual(review_ref,h.REVIEW)
            self.assertEqual(saved,f['records']['pre']['rows'][stage]);seen.append(stage);i=stages.index(stage)
            return copy.deepcopy(f['rows'][i]),copy.deepcopy(f['summaries'][i])
        with patch.object(h.p.successor,'metadata',return_value=copy.deepcopy(f['previous'])), \
             patch.object(h,'terminal_records',return_value=copy.deepcopy((f['refs'],f['records']))), \
             patch.object(h,'retained_stage',side_effect=retained),patch.object(Path,'glob',return_value=[]), \
             patch.object(h.p,'bundle',side_effect=AssertionError('native/source bundle forbidden')), \
             patch.object(h.p,'predecessor_drain',side_effect=AssertionError('physical metadata access forbidden')), \
             patch.object(h.p.low,'gone',side_effect=AssertionError('live process access forbidden')), \
             patch.object(h.p.low,'properties',side_effect=AssertionError('service access forbidden')):
            result=h.completed(f['binding'],read,lambda:None)
        self.assertEqual(seen,stages)
        return result

    def test_success_metadata_full1308_and_no_physical_calls(self):
        f=fixture();d=self.enroll(f)
        self.assertEqual(len(d['history']['rows']),1308);self.assertEqual(d['history']['rows'][:1255],f['previous']['previous']['history']['rows'])
        self.assertEqual(d['history']['processes'][:806],f['previous']['previous']['history']['processes'])
        self.assertEqual(d['previous'],f['previous']);self.assertEqual(d['lock_identity'],f['previous']['lock_identity'])
        self.assertTrue(d['outcome']['producer_qualified']);self.assertFalse(d['fresh_physical_drain_verified']);self.assertFalse(d['runtime_admitted'])

    def test_preserves_same_pid_start_exec_variants_and_current_five(self):
        f=fixture();d=self.enroll(f)
        variants=[o for o in d['owners'] if o['pid']==22000]
        self.assertEqual({tuple(o['argv']) for o in variants},{('python3','/synthetic/before.py'),('python3','/synthetic/after.py')})
        for o in f['previous']['current_history']['owners']+[f['previous']['official_admission']['later_owners'][0]]:self.assertIn(o,d['owners'])

    def test_failed_requires_real_distinct_release_and_stays_failed(self):
        d=self.enroll(fixture(False));self.assertTrue(d['outcome']['failed_release']);self.assertFalse(d['outcome']['producer_qualified'])

    def test_failed_unlaunched_stages_preserved_not_promoted(self):
        f=fixture(False);f['rows'][-1]['owner']=None;f['rows'][-1]['never_launched']=True
        f['summaries'][-1]=dict(stage=h.p.STAGES[-1],never_launched=True)
        d=self.enroll(f);self.assertTrue(d['history']['nemotron_r12_stages'][-1]['never_launched'])

    def test_success_missing_stage_cannot_promote(self):
        f=fixture();f['summaries'][-1]['never_launched']=True
        with self.assertRaises(RuntimeError):self.enroll(f)

    def test_binding_schema_extra_missing_or_live_refs_rejected_early(self):
        for change in [lambda b:b.update(extra=True),lambda b:b.update(schema='old'),lambda b:b.update(actual_exit=None),lambda b:b.update(actual_exit=dict(path='/fake.bin',sha256='a'*64))]:
            b=fixture()['binding'];change(b)
            with self.assertRaises(RuntimeError),patch.object(h.p.successor,'metadata',side_effect=AssertionError('must not load old history')):
                h.completed(b,lambda r:(_ for _ in ()).throw(AssertionError('must reject before read')),lambda:None)

    def test_pending_receipt_rejected_before_history_or_spec(self):
        f=fixture();seen=[];f['data'][f['binding']['actual_exit']['path']]['actual_tool_exit_observed']=False
        def read(ref):seen.append(ref);return f['data'][ref['path']]
        with self.assertRaises(RuntimeError),patch.object(h.p.successor,'metadata',side_effect=AssertionError('no history')):
            h.completed(f['binding'],read,lambda:None)
        self.assertEqual(seen,[f['binding']['actual_exit']])

    def test_missing_receipt_propagates_before_old_history(self):
        with self.assertRaises(FileNotFoundError),patch.object(h.p.successor,'metadata',side_effect=AssertionError('no history')):
            h.completed(fixture()['binding'],lambda r:(_ for _ in ()).throw(FileNotFoundError('synthetic missing exit')),lambda:None)

    def test_actual_session_owner_and_exit_fields_not_substitutable(self):
        for kind in ('session','owner','gone','review','boolean_code'):
            f=fixture();r=f['data'][f['binding']['actual_exit']['path']]
            if kind=='session':r['actual_exit']['session_id']=28399
            if kind=='owner':r['owner']=owner(99)
            if kind=='gone':r['exact_owner_gone']=False
            if kind=='review':r['source_review']=ref('other')
            if kind=='boolean_code':r['actual_exit']['exit_code']=False
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):self.enroll(f)

    def test_failed_release_argv_session_owner_code(self):
        for kind in ('missing','argv','session','owner','exit'):
            f=fixture(False);r=f['data'][f['binding']['failed_release_exit']['path']]
            if kind=='missing':f['binding']['failed_release_exit']=None
            if kind=='argv':r['owner']['argv'][3]='--custom-release'
            if kind=='session':r['actual_exit']['session_id']=h.LIVE_SESSION
            if kind=='owner':r['owner']=h.LIVE_OWNER
            if kind=='exit':r['actual_exit']['exit_code']=1
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):self.enroll(f)

    def test_unqualified_or_incomplete_old_current_history_rejected(self):
        for kind in ('old_rows','old_owners','current180','current_owner','promote_r11'):
            f=fixture();p=f['previous']
            if kind=='old_rows':p['previous']['history']['rows'].pop()
            if kind=='old_owners':p['previous']['history']['processes'].pop()
            if kind=='current180':p['current_history']['decode_calls']=179
            if kind=='current_owner':p['current_history']['owners'].pop()
            if kind=='promote_r11':p['previous']['history']['nemotron_r11_outcome']['producer_qualified']=True
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):self.enroll(f)

    def test_lock_canonical_unit_stage_mismatch_rejected(self):
        for kind in ('lock','canonical','unit','stage'):
            f=fixture()
            if kind=='lock':f['records']['admission']['lock_identity']={'different':True}
            if kind=='canonical':f['records']['admission']['canonical_sha256']='f'*64
            if kind=='unit':f['spec']['units'][h.p.STAGES[0]]='replaced-unit'
            if kind=='stage':f['summaries'][-1]['stage']='missing-stage'
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):self.enroll(f)

    def test_recovery_only_requires_original_failed_audit_chain(self):
        f=fixture(False);f['records']['audit'].update(recovery_only=True,original_audit=f['refs']['original_audit'],success_not_promoted=True)
        f['records']['original_audit']=dict(pre_stop=f['refs']['pre'])
        self.assertTrue(self.enroll(f)['outcome']['failed_release'])
        f['records']['audit']['success_not_promoted']=False
        with self.assertRaises(RuntimeError):self.enroll(f)

    def test_plain_callback_adapter_path_and_conflict(self):
        seen=[];r=h.TrackedRead(lambda ref:seen.append(ref)or {'owner':owner(7)},lambda:None)
        with patch.object(h.terminal,'ref',return_value=ref('path')):r.path(Path('/synthetic/path.json'))
        self.assertEqual(seen,[ref('path')]);self.assertEqual(len(r.observations),1)
        with self.assertRaises(RuntimeError):r(dict(path=ref('path')['path'],sha256='b'*64))

    def physical(self,done,variant=None):
        count={};gone=[]
        def props(unit):
            count[unit]=count.get(unit,0)+1
            return dict(ActiveState='active' if variant=='active' else 'inactive',
                sample=count[unit] if variant=='changing' else 0)
        def group(name):
            if variant=='group':raise RuntimeError('occupied group')
            return dict(group=name,absent=True)
        def gone_check(o):
            gone.append(o)
            if variant=='live':raise RuntimeError('still live')
        with patch.object(h.p,'predecessor_drain',return_value={'synthetic_old_chain':True}) as old, \
             patch.object(h.p.low,'gone',side_effect=gone_check),patch.object(h.p.low,'properties',side_effect=props), \
             patch.object(h.p,'validate_after',side_effect=lambda spec,stage,sr,rr,saved,value:value), \
             patch.object(h.p.low,'terminal',return_value=True),patch.object(h.p.low,'empty_group',side_effect=group), \
             patch.object(h.p.low,'active_heavy') as heavy,patch.object(h.p,'canonical',return_value='x'*64 if variant=='canonical' else 'c'*64):
            result=h.physical(done,lambda:None)
        self.assertEqual(old.call_args.kwargs,{'spec':done['spec']});self.assertEqual(sum(count.values()),106)
        self.assertEqual(len(count),53);self.assertTrue(all(v==2 for v in count.values()));heavy.assert_called_once_with()
        for o in done['owners']:self.assertIn(o,gone)
        return result

    def test_physical_all_old_chain_and53_double_samples_no_exemptions(self):
        d=self.enroll(fixture());r=self.physical(d);self.assertEqual(r['services'],1308);self.assertEqual(len(r['current']),53)
        self.assertTrue(r['fresh_physical_drain_verified']);self.assertFalse(d['fresh_physical_drain_verified'])

    def test_physical_live_unreleased_unstable_group_canonical_fail_closed(self):
        d=self.enroll(fixture())
        for variant in ('live','active','changing','group','canonical'):
            with self.subTest(variant=variant),self.assertRaises(RuntimeError):self.physical(d,variant)

    def test_original_retained_stage_kept_and_unlaunched_absence_checked(self):
        from dory_nemotron_r8_retained_failure_v1 import retained_stage
        self.assertIs(h.retained_stage,retained_stage)
        f=fixture();stage=h.p.STAGES[0];saved=dict(stage=stage,owner=None,properties=dict(LoadState='not-found',MainPID='0',ControlPID='0',InvocationID=''))
        with tempfile.TemporaryDirectory(prefix='mini-r12-helper-synthetic-') as td:
            row,summary=h.retained_stage(f['spec'],stage,h.SPEC,h.REVIEW,h.p,Path(td),saved,read=lambda p:(_ for _ in ()).throw(AssertionError('no metadata for unlaunched')))
            self.assertTrue(row['never_launched']);self.assertTrue(summary['never_launched'])
            (Path(td)/(stage+'-owner.json')).write_text('{}')
            with self.assertRaises(RuntimeError):h.retained_stage(f['spec'],stage,h.SPEC,h.REVIEW,h.p,Path(td),saved,read=lambda p:{})

    def test_guard_required_and_propagates_before_read(self):
        with self.assertRaises(RuntimeError):h.completed(fixture()['binding'],lambda r:{},None)
        def stop():raise ValueError('revoked')
        with self.assertRaisesRegex(ValueError,'revoked'):h.completed(fixture()['binding'],lambda r:(_ for _ in ()).throw(AssertionError('not read')),stop)

    def test_native_not_imported(self):
        self.assertFalse(set(sys.modules)&{'torch','numpy','torch_neuronx','neuronxcc'})


if __name__=='__main__':unittest.main(verbosity=2)
