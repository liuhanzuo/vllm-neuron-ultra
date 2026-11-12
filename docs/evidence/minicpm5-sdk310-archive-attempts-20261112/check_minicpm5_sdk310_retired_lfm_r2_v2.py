"""Saved JSON and injected fake protocol only; no live checks or runtime writes."""
import copy
import unittest
from unittest.mock import patch
import minicpm5_sdk310_retired_lfm_r2_v2 as r
import minicpm5_sdk310_deb_input_binding_v3 as binding
import check_minicpm5_sdk310_retired_lfm_r2_v1 as previous


def fixture():
    plan,values,read=previous.fixture();scope=read(plan['r2_predecessor_scope'])
    base=r.c.load(previous.old_builder.BASE);catalog=r.c.load(previous.old_builder.history.PRIOR_CATALOG)
    catalog['runs'].append(dict(terminal=scope['actual_exit'],tool=scope['actual_tool'],owners=scope['owners'],units=[r.base.UNIT]))
    owners,services=previous.old_builder.complete_history(base,catalog,scope)
    catalog_ref=dict(path='/synthetic/complete-serial-catalog.json',sha256='b'*64)
    values[catalog_ref['path']]=catalog
    plan.update(predecessor_owners=owners,predecessor_services=services,complete_current_serial_catalog=catalog_ref)
    for item in r.EXTRA:
        terminal=r.c.load(item['actual_exit']);values[item['actual_exit']['path']]=terminal
        values[item['recorder_exit']['path']]=r.c.load(item['recorder_exit'])
        for name in ('coordinator_owner','native_owner','cleanup_drained','outer_failure','service_exit','children_exits'):
            if name in terminal['outputs']:
                ref=terminal['outputs'][name];values[ref['path']]=r.c.load(ref)
    plan['retired_mini_history']=r.bind_extra(read)
    return plan,values,read


class FakeProtocol(previous.FakeProtocol):
    def __init__(self,plan,values):
        super().__init__(plan,values)
        for item in r.EXTRA:self.overrides[item['unit']]=dict(LoadState='not-found',InvocationID='')


class Tests(unittest.TestCase):
    def reject(self,mutate):
        plan,values,read=fixture();mutate(plan,values)
        with self.assertRaises((RuntimeError,KeyError)):r.validate(plan,read=read)
    def test_both_genuine_histories_and_r2_bind_without_current_operations(self):
        plan,values,read=fixture();result=r.validate(plan,read=read)
        self.assertEqual(result,plan['retired_mini_history'])
        self.assertEqual([x['actual_tool']['exit_code'] for x in result],[1,0])
        self.assertEqual([x['original_sdk_model_exit_code'] for x in result],[1,1])
        self.assertNotIn('service_exit',result[0]['outputs']);self.assertIn('children_exits',result[1]['outputs'])
    def test_complete1455_rows_and2809_owners_three_only_exceptions(self):
        plan,values,read=fixture();p=FakeProtocol(plan,values);rows=r.preflight_history(plan,p,read)
        self.assertEqual(len(rows),1455);self.assertEqual(len(p.gone_owners),2809)
        self.assertEqual(p.calls,[s['unit']for s in plan['predecessor_services'] for _ in range(2)])
        self.assertEqual(sum('retirement_binding'in row for row in rows),3)
        old_unknown=[row for row,want in zip(rows,plan['predecessor_services']) if want['unit']==r.EXTRA[0]['unit'] and not want.get('invocation_id')]
        self.assertEqual(len(old_unknown),1);self.assertNotIn('retirement_binding',old_unknown[0])
    def test_no_broader_missing_invocation_exception(self):
        plan,values,read=fixture();p=FakeProtocol(plan,values)
        other=next(s for s in plan['predecessor_services'] if s.get('invocation_id') and s['unit'] not in [r.base.UNIT]+[i['unit']for i in r.EXTRA])
        p.overrides[other['unit']]={'LoadState':'not-found','InvocationID':''}
        with self.assertRaisesRegex(RuntimeError,'Known original invocation'):r.preflight_history(plan,p,read)
    def test_wrong_or_removed_known_invocations_rejected(self):
        for item in r.EXTRA:
            for changed in (None,'wrong'):
                def mutate(plan,values):
                    next(s for s in plan['predecessor_services'] if s['unit']==item['unit'] and s.get('invocation_id'))['invocation_id']=changed
                with self.subTest(unit=item['unit'],changed=changed):self.reject(mutate)
    def test_all_eight_historical_owners_required_in_plan(self):
        for item_index in (0,1):
            for owner_index in range(4):
                def mutate(plan,values):
                    owner=plan['retired_mini_history'][item_index]['owners'][owner_index]
                    plan['predecessor_owners']=[o for o in plan['predecessor_owners'] if (o['pid'],o['start_ticks'])!=(owner['pid'],owner['start_ticks'])]
                with self.subTest(item=item_index,owner=owner_index):self.reject(mutate)
    def test_each_model_and_recorder_catalog_row_required(self):
        for i in range(2):
            for key in ('actual_exit','recorder_exit'):
                def mutate(plan,values):
                    target=plan['retired_mini_history'][i][key]
                    catalog=values[plan['complete_current_serial_catalog']['path']]
                    catalog['runs']=[row for row in catalog['runs'] if row['terminal']!=target]
                with self.subTest(item=i,key=key):self.reject(mutate)
    def test_original_r1_failure_and_missing_receipts_preserved(self):
        for key,value in [('actual_exit_code',0),('successful_model_terminal',True),('original_nonzero_exit_preserved',False),('missing_runtime_receipts',[])]:
            with self.subTest(key=key):self.reject(lambda p,v:v[r.EXTRA[0]['actual_exit']['path']].__setitem__(key,value))
        def mutate(plan,values):
            ref=values[r.EXTRA[0]['actual_exit']['path']]['outputs']['outer_failure'];values[ref['path']]['error']='success'
        self.reject(mutate)
    def test_r3_loaded_success_and_actual_native_exit_required(self):
        for key,value in [('InvocationID',''),('ExecMainCode','0'),('ExecMainStatus','1')]:
            def mutate(plan,values):
                ref=values[r.EXTRA[1]['actual_exit']['path']]['outputs']['service_exit'];values[ref['path']]['properties'][key]=value
            with self.subTest(key=key):self.reject(mutate)
        def mutate(plan,values):
            ref=values[r.EXTRA[1]['actual_exit']['path']]['outputs']['children_exits'];values[ref['path']]['exit_codes']=[1]
        self.reject(mutate)
    def test_original_sdk_failure_and_old_recorder_flag_remain(self):
        for i in range(2):
            for key,value in [('original_sdk_model_exit_code',0),('actual_recorder_tool_exit_verified',True),('independent_raw_audit_pending',False)]:
                with self.subTest(item=i,key=key):self.reject(lambda p,v:v[r.EXTRA[i]['actual_exit']['path']].__setitem__(key,value))
    def test_actual_recorder_owner_success_and_report_binding_required(self):
        for i in range(2):
            for key,value in [('owner_gone',False),('actual_tool_exit_observed',False),('model_exit',{})]:
                with self.subTest(item=i,key=key):self.reject(lambda p,v:v[r.EXTRA[i]['recorder_exit']['path']].__setitem__(key,value))
    def test_current_two_units_reloaded_job_live_pid_or_group_rejected(self):
        for item in r.EXTRA:
            for changes in ({'LoadState':'loaded'},{'InvocationID':item['invocation_id']},{'Job':'x'},{'MainPID':'5'},{'ExecMainPID':'5'}):
                plan,values,read=fixture();p=FakeProtocol(plan,values);p.overrides[item['unit']].update(changes)
                with self.subTest(unit=item['unit'],changes=changes),self.assertRaises(RuntimeError):r.preflight_history(plan,p,read)
            plan,values,read=fixture();p=FakeProtocol(plan,values);group=r.exact(item);p.groups[group]=dict(group=group,absent=False,cgroups=1)
            with self.assertRaises(RuntimeError):r.preflight_history(plan,p,read)
    def test_any_historical_recorder_still_live_rejected(self):
        plan,values,read=fixture();p=FakeProtocol(plan,values);p.live=plan['retired_mini_history'][0]['owners'][3]
        with self.assertRaises(RuntimeError):r.preflight_history(plan,p,read)
    def test_recorder_rechecks_all876_r2_and8_mini_owners(self):
        plan,values,read=fixture();p=FakeProtocol(plan,values);rows=r.preflight_history(plan,p,read)
        p.gone_owners=[];proof=r.verify_recorded_history(plan,rows,p,read)
        self.assertEqual(len(p.gone_owners),884);self.assertEqual(len(proof['mini_history']),2)
        p.overrides[r.EXTRA[1]['unit']]['InvocationID']='new'
        with self.assertRaises(RuntimeError):r.verify_recorded_history(plan,rows,p,read)
    def test_missing_saved_invocation_or_changed_binding_rejected_by_recorder(self):
        plan,values,read=fixture();p=FakeProtocol(plan,values);rows=r.preflight_history(plan,p,read)
        i=next(i for i,row in enumerate(rows) if row['unit']==r.EXTRA[0]['unit'] and 'retirement_binding'in row)
        for key in ('original_invocation_id','retirement_binding'):
            changed=copy.deepcopy(rows);changed[i].pop(key)
            with self.subTest(key=key),self.assertRaises(KeyError):r.verify_recorded_history(plan,changed,p,read)
    def test_review_must_accept_both_complete_bindings(self):
        plan,values,read=fixture();review=dict(retired_lfm_r2=plan['retired_lfm_r2'])
        with self.assertRaises(KeyError):r.validate(plan,review,read)
    def test_five_runtime_successors_are_only_narrow_wiring(self):
        for name in ('run_minicpm5_sdk310_deb_members_root','minicpm5_sdk310_deb_members_worker','record_minicpm5_sdk310_deb_stage_exit'):
            old=(r.c.Q/(name+'_v2.py')).read_text();new=(r.c.Q/(name+'_v3.py')).read_text()
            expected=old.replace('import minicpm5_sdk310_deb_input_binding_v2 as','import minicpm5_sdk310_deb_input_binding_v3 as').replace('minicpm5_sdk310_retired_lfm_r2_v1','minicpm5_sdk310_retired_lfm_r2_v2')
            if name=='run_minicpm5_sdk310_deb_members_root':expected=expected.replace('minicpm5_sdk310_deb_members_worker_v2.py','minicpm5_sdk310_deb_members_worker_v3.py')
            self.assertEqual(new,expected)
        plan,values,read=fixture();plan['source_sha256']={str(r.c.Q/n):'f'*64 for n in binding.SUCCESSOR_NAMES};review={}
        with patch.object(binding.previous,'validate',return_value=plan)as first,patch.object(binding.retirement,'validate')as second:
            self.assertIs(binding.validate(plan,review,{}),plan);first.assert_called_once();second.assert_called_once()


if __name__=='__main__':unittest.main(verbosity=2)
