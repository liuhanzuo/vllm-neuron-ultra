"""Transaction mutations occur only in synthetic /tmp fixtures."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import minicpm5_retained_raw_transport_v1 as t
import minicpm5_retained_raw_transaction_v1 as x
from test_minicpm5_retained_raw_transport_v1 import fixture


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mini-retirement-test-',dir='/tmp')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.plan=fixture(self.root)
        self.guard=lambda:None
        self.verified=t._verify_pairs(self.plan,self.root/'verified.json',self.guard)
        admission=dict(plan_sha256=t.digest(t.canonical(self.plan)),**{k:dict(path=str(self.root/(k+'.json')),sha256='a'*64)
            for k in ('independent_verification','remover_source','replay_source','serial_lease','full_history_compatibility')})
        self.intent=t._prepare_intent(self.plan,self.verified,self.root/'intent.json',admission,self.guard)
        self.journal=self.root/'journal'

    def run_transaction(self):return x._retire(self.plan,self.verified,self.intent,self.journal,self.guard)

    def test_complete_and_repeated_audit_preserve_all_bytes(self):
        first=t.read_receipt(self.run_transaction());self.assertTrue(first['complete'])
        self.assertEqual(len(first['pairs']),2)
        for row in self.plan['entries']:
            self.assertFalse(Path(row['original']['realpath']).exists())
            self.assertEqual(t.identity(row['keeper']['realpath']),row['keeper'])
        second=t.read_receipt(self.run_transaction());self.assertEqual(first['pairs'],second['pairs'])
        self.assertNotEqual(first['postcondition'],second['postcondition'])

    def test_crash_after_unlink_resumes_from_exact_durable_intent(self):
        original=x._unlink_exact;count=0
        def crash(row,guard):
            nonlocal count
            original(row,guard);count+=1
            if count==1:raise InterruptedError('after unlink before done')
        with patch.object(x,'_unlink_exact',side_effect=crash):
            with self.assertRaises(InterruptedError):self.run_transaction()
        self.assertTrue((self.journal/'0000-intent.json').exists())
        self.assertFalse((self.journal/'0000-done.json').exists())
        result=t.read_receipt(self.run_transaction());done=t.read_receipt(result['pairs'][0])
        self.assertTrue(done['recovered_after_unrecorded_unlink'])
        self.assertFalse(done['unlink_observed_in_this_attempt'])

    def test_crash_before_unlink_preserves_original_and_retries_verified_pair(self):
        with patch.object(x,'_unlink_exact',side_effect=InterruptedError('before unlink')):
            with self.assertRaises(InterruptedError):self.run_transaction()
        self.assertTrue(Path(self.plan['entries'][0]['original']['realpath']).exists())
        self.assertTrue(t.read_receipt(self.run_transaction())['complete'])

    def test_missing_without_pair_intent_never_promoted_to_success(self):
        Path(self.plan['entries'][0]['original']['realpath']).unlink()
        with self.assertRaises(FileNotFoundError):self.run_transaction()
        self.assertTrue(Path(self.plan['entries'][1]['original']['realpath']).exists())

    def test_replacement_after_intent_is_not_unlinked(self):
        real=x._unlink_exact
        def replace_then_check(row,guard):
            original=Path(row['original']['realpath']);replacement=self.root/'replaced.raw'
            replacement.write_bytes(b'z'*row['bytes']);replacement.replace(original)
            real(row,guard)
        with patch.object(x,'_unlink_exact',side_effect=replace_then_check):
            with self.assertRaisesRegex(ValueError,'identity changed'):self.run_transaction()
        self.assertEqual(Path(self.plan['entries'][0]['original']['realpath']).read_bytes(),b'z'*64)
        self.assertFalse((self.journal/'0000-done.json').exists())

    def test_partial_pair_journal_is_retained_and_requires_explicit_review(self):
        self.journal.mkdir();p=self.journal/'0000-intent.json';p.write_bytes(b'{')
        with self.assertRaises(ValueError):self.run_transaction()
        self.assertEqual(p.read_bytes(),b'{')
        self.assertTrue(Path(self.plan['entries'][0]['original']['realpath']).exists())

    def test_interrupted_mid_done_write_retains_partial_and_blocks_automatic_resume(self):
        original=t._write_once
        def interrupted(path,value):
            if Path(path).name=='0000-done.json':
                Path(path).write_bytes(b'{"schema":')
                raise OSError('synthetic middle of receipt write')
            return original(path,value)
        with patch.object(t,'_write_once',side_effect=interrupted):
            with self.assertRaises(OSError):self.run_transaction()
        self.assertFalse(Path(self.plan['entries'][0]['original']['realpath']).exists())
        partial=self.journal/'0000-done.json';self.assertEqual(partial.read_bytes(),b'{"schema":')
        with self.assertRaisesRegex(ValueError,'partial completion'):self.run_transaction()
        self.assertEqual(partial.read_bytes(),b'{"schema":')
        self.assertTrue(Path(self.plan['entries'][1]['original']['realpath']).exists())
        self.assertEqual(t.identity(self.plan['entries'][0]['keeper']['realpath']),self.plan['entries'][0]['keeper'])

    def test_production_entry_rejects_synthetic_even_with_guard(self):
        with self.assertRaises(ValueError):x.retire(self.plan,self.verified,self.intent,self.journal,self.guard)


class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mini-source-binding-',dir='/tmp');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.plan=fixture(self.root);self.pin=t.digest(t.canonical(self.plan))
        self.sources={}
        for stem in ('transport','transaction','legacy','compare'):
            p=self.root/('minicpm5_retained_raw_'+stem+'_v1.py');p.write_text('synthetic_source = '+repr(stem)+'\n')
            self.sources[stem]=dict(path=str(p),sha256=t.digest(p.read_bytes()))
        p=self.root/'original.py';p.write_text('original = True\n');self.original_source={str(p):t.digest(p.read_bytes())}
        self.oldplan=self.write(self.root/'old-plan.json',dict(source_sha256=self.original_source))
        self.map={**self.original_source,**{r['path']:r['sha256']for r in self.sources.values()}}
        self.verified={'path':str(self.root/'verified.json'),'sha256':'c'*64,'bytes':1}
        self.equality=dict(complete=True,replay_result=dict(path=str(self.root/'replay.json'),sha256='a'*64,bytes=1))
        equality_ref=t._write_once(self.root/'equality.json',self.equality)
        self.review=dict(schema='minicpm5_retained_raw_replay_and_removal_review_v1',plan_sha256=self.pin,verification=self.verified,
            remover_source=self.sources['transaction'],replay_source=self.sources['legacy'],passed=True,blockers=[],
            full11859_logical_inputs_preserved=True,full32_replay_and_all_scientific_outputs_equal=True,
            all_original_failures_preserved=True,original_weights_precision_and_remaining_gates_unchanged=True,
            source_sha256=self.map,replay_equality=equality_ref)
        self.history=self.write(self.root/'history.json',dict(schema='minicpm5_retained_raw_full_history_compatibility_v1',
            plan_sha256=self.pin,passed=True,complete_current_input_and_source_maps_checked=True,
            all_predecessor_owner_service_cgroup_scope_preserved=True,candidate_current_input_intersection=0))
        self.lease=self.write(self.root/'lease.json',dict(schema='minicpm5_retained_raw_exclusive_retirement_lease_v1',
            plan_sha256=self.pin,all_writers_and_holders_drained=True,retirement_source=self.sources['transaction']))

    def write(self,p,value):
        p.write_bytes(t.canonical(value)+b'\n');return dict(path=str(p),sha256=t.digest(p.read_bytes()))

    def admit(self,wrong_remover=False):
        ref=self.write(self.root/'review.json',self.review)
        admission=dict(plan_sha256=self.pin,remover_source=self.sources['legacy']if wrong_remover else self.sources['transaction'],
            replay_source=self.sources['legacy'],independent_verification=ref,full_history_compatibility=self.history,serial_lease=self.lease)
        intent=t._write_once(self.root/'intent.json',dict(plan_sha256=self.pin,verification=self.verified,admission=admission))
        expected=dict(plan_sha256=self.pin,intent=intent,serial_lease=self.lease,
            all_writers_and_holders_drained=True,exclusive_namespace_write_lease=True)
        stack=ExitStack();self.addCleanup(stack.close)
        for obj,name,value in [(t,'Q',self.root),(t,'ROOT',self.root),(t,'__file__',self.sources['transport']['path']),
            (x,'__file__',self.sources['transaction']['path']),(x.comparison,'__file__',self.sources['compare']['path']),
            (t,'REFERENCES',{'recovery_plan':self.oldplan})]:stack.enter_context(patch.object(obj,name,value))
        stack.enter_context(patch.object(x.comparison,'_compare',return_value=self.equality))
        return x._admission(self.plan,self.verified,intent,lambda:expected)

    def test_exact_complete_executing_source_map_is_accepted(self):self.admit()()

    def test_missing_transport_source_rejected(self):
        del self.map[self.sources['transport']['path']]
        with self.assertRaisesRegex(ValueError,'source closure'):self.admit()

    def test_wrong_existing_python_file_cannot_replace_current_remover(self):
        with self.assertRaisesRegex(ValueError,'actual remover'):self.admit(wrong_remover=True)

    def test_executing_source_change_after_admission_fails_guard(self):
        guard=self.admit();Path(self.sources['transport']['path']).write_text('changed_after_admission = True\n')
        with self.assertRaisesRegex(ValueError,'changed after admission'):guard()

    def test_original_source_map_cannot_be_shortened(self):
        del self.map[next(iter(self.original_source))]
        with self.assertRaisesRegex(ValueError,'Entire unchanged original'):self.admit()


if __name__=='__main__':unittest.main(verbosity=2)
