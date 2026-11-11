"""Small fully injected saved-entry flow; never real plan/raw/process/lock IO."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'research'),str(ROOT)]
from research import run_minicpm5_native_sdk_saved_comparison_v1 as e
from research.check_minicpm5_native_sdk_saved_comparison_entry_v1 import ref,terminal_fixture


class Flow(unittest.TestCase):
    def execute(self,*,mutate_inventory=False,packed_size=0,mutate_physical=False):
        external,audit_result,audit_spec,audit_review=terminal_fixture();spec_ref=ref('compare');review_ref=ref('compare-review')
        spec=dict(native_audit_exit=ref('audit-external'),current_serial_catalog=ref('current-catalog'),source_extension={e.SELF:'c'*64})
        owner=dict(pid=20,start_ticks='2',argv=['original'],cgroup='0::/synthetic');rec_owner=dict(owner,pid=21)
        outputs={key:ref(key)for key in ('admission','intent','source_authentication','coordinator_owner')}
        outputs['native_result']=ref('native-result-wrapper')
        ended=dict(actual_tool=dict(kind='tool_session',session_id=11,terminal_chunk='abcdef',exit_code=0),
            actual_exit_observed=True,actual_exit_code=0,owner=owner,coordinator=dict(owner,pid=22),native=dict(owner,pid=23),
            live=ref('live'),spec=ref('model-spec'),review=ref('model-review'),outputs=outputs)
        recorder=dict(owner=rec_owner,actual_tool=dict(kind='tool_call',terminal_chunk='bcdefa',exit_code=0))
        model_spec=dict(schema='minicpm5_native_process_spec_v1',overlay=ref('overlay'),runner=ref('runner'),unit='original-unit',lock_policy=e.r.LOCK_POLICY)
        model_review=dict(schema='minicpm5_native_process_source_review_v1',reviewer='/root',passed=True,blockers=[],
            spec=ended['spec'],overlay=model_spec['overlay'],lock_policy=e.r.LOCK_POLICY,resolved_plan_sha256='d'*64)
        source_contract=ref('source-contract');latest=ref('latest');boundary=ref('sdk-boundary');serial_ref=ref('model-catalog')
        plan=dict(source_sha256={e.a.SELF:'a'*64,str(Path(e.a.core.__file__).resolve()):'b'*64},source_contract=source_contract,
            latest_completion=latest,layer1_operator_boundary=boundary,serial_predecessors=serial_ref,output_root='/synthetic/native')
        contract=dict(unit='original-unit',output_root=plan['output_root'])
        admission=dict(overlay=model_spec['overlay'],source_contract=source_contract,sdk_operator_recovery=e.r.completion.BINDING,
            latest_completion=latest,completed_sdk_layout=boundary,serial_predecessors=serial_ref,attempt_id='a'*32)
        original_scope=dict(owners=[],complete_known_units=[],nemotron=dict(history=dict(rows=[])),later_serial=dict(runs=[]))
        resolved=dict(plan=plan,plan_sha256='d'*64,current_scope=original_scope,native_bundle={'native':'synthetic'})
        driver={'schema':'synthetic-unwrapped-driver'};checked=dict(driver=driver,contract=contract,model_owners=[owner])
        sdk_refs=[ref('sdk-rank%d'%rank)for rank in (0,1)];sdk=[{'rank':0},{'rank':1}]
        recovered=dict(base_plan={'output_root':'/synthetic/sdk'},base_plan_ref=ref('sdk-plan'),
            recovery=dict(original_model_outputs={'rank%d_result'%rank:sdk_refs[rank]for rank in (0,1)}))
        audit_result.update(inherited_native_source_authentication={'inherited':True},sdk_rank_references=sdk_refs,
            model_drain={'physically_drained':True},inventory={'complete':True})
        mapping={spec_ref['path']:spec,review_ref['path']:{},spec['native_audit_exit']['path']:external,
            external['result']['path']:audit_result,external['audit_spec']['path']:audit_spec,external['audit_review']['path']:audit_review,
            audit_spec['model_exit']['path']:ended,audit_spec['recorder_exit']['path']:recorder,ended['live']['path']:{},
            ended['spec']['path']:model_spec,ended['review']['path']:model_review,model_spec['overlay']['path']:{}}
        mapping.update({value['path']:{}for value in outputs.values()});mapping[outputs['admission']['path']]=admission
        mapping.update({reference['path']:value for reference,value in zip(sdk_refs,sdk)})
        events=[];metadata=Mock();metadata.read.side_effect=lambda reference:mapping[reference['path']]
        guard=Mock();guard.storage={};lease=Mock();lease.admission={'outer_owner':dict(owner,pid=90)}
        io=Mock();io.native.final_inventory.side_effect=lambda:events.append('inventory')
        def finish():events.append('unchanged');return {'complete':not mutate_inventory}
        io.finish.side_effect=finish;cases=[[{'rank':0}],[{'rank':1}]];case_refs=[ref('extra%d'%i)for i in range(12)]
        report={'synthetic_report':True}
        def compare(*args):
            events.append('compare');self.assertIs(args[0],driver);self.assertIs(args[1],plan);self.assertIs(args[2],audit_result)
            self.assertEqual(args[3],sdk);self.assertIs(args[4],cases);self.assertIs(args[5],recovered['base_plan'])
            self.assertIs(args[6],recovered['recovery']);self.assertIs(args[7],io.read_bytes);self.assertIs(args[8],guard)
            return report
        serial={'runs':[],'owners':[],'units':[]};freshness={'latest_handoff':ref('latest-idle')}
        pairs=[{'synthetic_pair':1},{'synthetic_pair':2 if mutate_physical else 1}]
        owned=Mock();owned.verify_drained.return_value=audit_result['model_drain']
        with ExitStack()as stack:
            def p(obj,name,**kwargs):return stack.enter_context(patch.object(obj,name,**kwargs))
            p(e,'validate_spec');p(e,'audit_terminal',return_value=external['actual_tool']);p(e,'catalog_binding');p(e,'historical_pairs')
            p(e.a,'AuditGuard',return_value=guard);p(e.r,'Metadata',return_value=metadata);p(e.r.sources,'text',return_value=b'synthetic')
            p(e.r.p,'gone');p(e.r.completion,'completed',return_value=recovered);p(e.r.overlays,'resolve',return_value=resolved)
            p(e.a.core,'validate',return_value=checked);p(e.r,'validate_initial_descriptor');p(e.r.execution,'validate_intent')
            p(e.r.sources,'classify',return_value={});p(e.r.sources,'validate_native_receipt',return_value={'inherited':True})
            p(e.a,'merged_plan',side_effect=lambda old,pins:dict(old,source_sha256=dict(old['source_sha256'],**pins)))
            p(e.r.sources,'verify_text_sources');p(e.r.sources,'check_native_identities');p(e.r,'verify_loaded')
            p(e.r.completion.latest,'serial_catalog',return_value=serial);p(e.r.completion.latest,'require_latest_terminal',return_value=freshness)
            p(e.r.completion.latest,'latest_handoff',return_value={'path':'/synthetic/latest-idle.json','identity':{}})
            p(e.r.current_authority,'Authority');p(e.a,'pair',side_effect=pairs);p(e.a.outer,'OwnedService',return_value=owned)
            p(e,'anchor_cases',return_value=(cases,case_refs));p(e,'ReadIO',return_value=io);p(e.comparison,'compare',side_effect=compare)
            p(e.transport,'encode_report',return_value=json.dumps({'encoded':'x'*packed_size}).encode())
            p(e.a.prior_audit,'packed_physical',return_value={'packed_physical':'synthetic'})
            publish=p(e.r,'atomic_once')
            result=e.compare_saved(spec_ref,review_ref,lease);publish.assert_not_called()
        self.assertEqual(events,['inventory','compare','unchanged']);return result
    def test_success_flow_preserves_unwrapped_arguments_prior_refs_and_pending_own_exit(self):
        result=self.execute();self.assertTrue(result['complete']);self.assertFalse(result['actual_comparator_tool_exit_verified'])
        self.assertEqual(result['original_sdk_model_exit_code'],1);self.assertEqual(result['comparator_reads'],1632)
        self.assertEqual(result['extra_anchor_metadata_reads'],12);self.assertEqual(len(result['extra_anchor_call_metadata']),12)
        self.assertFalse(result['self_exit_claimed']);self.assertFalse(result['full_model_qualified']);self.assertFalse(result['new_speedup_claim'])
    def test_changed_final_inventory_rejects_no_publish(self):
        with self.assertRaisesRegex(RuntimeError,'inventory'):self.execute(mutate_inventory=True)
    def test_changed_complete_physical_pair_rejects_no_publish(self):
        with self.assertRaisesRegex(RuntimeError,'physical scope'):self.execute(mutate_physical=True)
    def test_entire_wrapper_cap_not_only_report_cap(self):
        with self.assertRaisesRegex(RuntimeError,'wrapper'):self.execute(packed_size=e.LIMITS['output_file_bytes']-100)


if __name__=='__main__':unittest.main(verbosity=2)
