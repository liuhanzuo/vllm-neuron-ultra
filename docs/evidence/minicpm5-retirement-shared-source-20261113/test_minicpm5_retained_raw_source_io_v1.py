"""Only source reads and injected in-memory stat/read fixtures; no actual .pth IO."""
import ast
import copy
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import minicpm5_retained_raw_source_io_v1 as source
import minicpm5_retained_raw_transaction_v1 as frozen
import minicpm5_retained_raw_transaction_v2 as transaction
import minicpm5_retained_retirement_contract_v3 as c


def stat_value(**changes):
    ident=source.SHARED_IDENTITY
    value=dict(st_dev=ident['device'],st_ino=ident['inode'],st_size=ident['size'],
        st_mtime_ns=ident['mtime_ns'],st_ctime_ns=ident['ctime_ns'],st_mode=ident['mode'],st_nlink=ident['nlink'])
    return SimpleNamespace(**dict(value,**changes))


class ReaderFixture:
    def __init__(self):
        self.value=stat_value();self.parts=[b'x'*40,b'x'*49,b''];self.events=[]
        self.hash=source.SHARED_SHA256;self.after_read=None
    def __enter__(self):
        self.stack=ExitStack()
        def read(fd,cap):
            self.events.append(('read',fd));out=self.parts.pop(0)
            if not out and self.after_read:self.after_read()
            return out
        def open_(path,flags):
            self.events.append(('open',str(path),flags));return 42
        for obj,name,value in ((Path,'resolve',lambda p:p),(Path,'is_symlink',lambda p:False),
            (Path,'lstat',lambda p:self.value),(source.os,'open',open_),
            (source.os,'fstat',lambda fd:self.value),(source.os,'read',read),
            (source.os,'close',lambda fd:self.events.append(('close',fd))),
            (source.t,'digest',lambda data:self.hash)):
            self.stack.enter_context(patch.object(obj,name,value))
        return self
    def __exit__(self,*args):return self.stack.__exit__(*args)
    def read(self):return source.read(dict(path=source.SHARED_PATH,sha256=source.SHARED_SHA256))


class ExactSharedReader(unittest.TestCase):
    def test_exact_historical_two_link_pin_is_read_without_mutation(self):
        with ReaderFixture()as f:
            self.assertEqual(source.identity(source.SHARED_PATH),source.SHARED_IDENTITY)
            self.assertEqual(f.read(),b'x'*89)
        self.assertEqual(f.events[0],('open',source.SHARED_PATH,source.os.O_RDONLY|source.os.O_NOFOLLOW|source.os.O_NONBLOCK))
        self.assertEqual(f.events[-1],('close',42));self.assertEqual(len([e for e in f.events if e[0]=='open']),1)

    def test_nlink_one_or_three_never_rebinds_the_known_shared_source(self):
        for nlink in (0,1,3):
            with self.subTest(nlink=nlink),ReaderFixture()as f:
                f.value=stat_value(st_nlink=nlink)
                with self.assertRaisesRegex(ValueError,'nlink2'):f.read()
                self.assertFalse(f.events)

    def test_every_pinned_stat_field_is_required(self):
        for name in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_mode'):
            with self.subTest(field=name),ReaderFixture()as f:
                f.value=stat_value(**{name:getattr(f.value,name)+1})
                with self.assertRaises(ValueError):f.read()
                self.assertFalse(f.events)

    def test_another_alias_of_same_inode_does_not_inherit_the_exception(self):
        with ReaderFixture()as f:
            alias=source.SHARED_PATH+'.other-alias'
            with self.assertRaisesRegex(ValueError,'Singleton'):source.identity(alias)
            with self.assertRaisesRegex(ValueError,'Singleton'):source.read(dict(path=alias,sha256=source.SHARED_SHA256))
            self.assertFalse(f.events)

    def test_raw_transport_still_rejects_even_the_exact_shared_pin(self):
        with ReaderFixture()as f:
            with self.assertRaisesRegex(ValueError,'Singleton'):source.t.identity(source.SHARED_PATH)
            six={k:v for k,v in source.SHARED_IDENTITY.items()if k not in ('mode','nlink')}
            with self.assertRaisesRegex(ValueError,'Singleton'):source.t._open(six)
            with self.assertRaisesRegex(ValueError,'changed across read'):source.t._unchanged(42,six)
            self.assertFalse(f.events)

    def test_ordinary_singleton_source_keeps_frozen_identity_open_and_unchanged(self):
        with ReaderFixture()as f:
            f.value=stat_value(st_nlink=1);path='/synthetic/ordinary.py'
            self.assertNotIn('nlink',source.identity(path))
            self.assertEqual(source.read(dict(path=path,sha256=source.SHARED_SHA256)),b'x'*89)
        self.assertEqual(f.events[-1],('close',42))

    def test_wrong_reviewed_hash_rejected_before_open_and_content_drift_after_read(self):
        with ReaderFixture()as f:
            with self.assertRaisesRegex(ValueError,'SHA pin'):source.read(dict(path=source.SHARED_PATH,sha256='0'*64))
            self.assertFalse(f.events)
        with ReaderFixture()as f:
            f.hash='0'*64
            with self.assertRaisesRegex(ValueError,'digest changed'):f.read()
            self.assertEqual(f.events[-1],('close',42))

    def test_alias_write_or_relink_during_read_fails_and_closes_descriptor(self):
        for changed in (dict(st_ctime_ns=1),dict(st_nlink=3),dict(st_ino=1)):
            with self.subTest(change=changed),ReaderFixture()as f:
                f.after_read=lambda:f.__setattr__('value',stat_value(**changed))
                with self.assertRaises(ValueError):f.read()
                self.assertEqual(f.events[-1],('close',42))

    def test_opened_descriptor_replacement_is_rejected(self):
        with ReaderFixture()as f,patch.object(source.os,'fstat',return_value=stat_value(st_ino=1)):
            with self.assertRaises(ValueError):f.read()
            self.assertEqual(f.events[-1],('close',42));self.assertFalse(any(e[0]=='read'for e in f.events))

    def test_symlink_or_noncanonical_path_is_not_admitted(self):
        with ReaderFixture()as f,patch.object(Path,'is_symlink',return_value=True):
            with self.assertRaisesRegex(ValueError,'nonsymlink'):f.read()
        with ReaderFixture()as f,patch.object(Path,'resolve',return_value=Path('/synthetic/rebound')):
            with self.assertRaisesRegex(ValueError,'nonsymlink'):f.read()


def admission_fixture():
    old={str(c.Q/('synthetic-original-%04d.py'%i)):'a'*64 for i in range(1868)}
    old[source.SHARED_PATH]=source.SHARED_SHA256
    execution={str(c.Q/('synthetic-added-%04d.py'%i)):'b'*64 for i in range(573)}
    own={str(c.Q/name):'d'*64 for name in ('minicpm5_retained_raw_transaction_v2.py',
        'minicpm5_retained_raw_transport_v1.py','minicpm5_retained_raw_compare_v1.py',
        'minicpm5_retained_raw_source_io_v1.py','minicpm5_retained_raw_legacy_v1.py')}
    sources=dict(old,**execution,**own);assert len(old)==1869 and len(sources)==2447
    ref=lambda name:dict(path='/synthetic/'+name+'.json',sha256='e'*64,bytes=100)
    plan=dict(synthetic='no actual plan');verification=ref('verification');intent_ref=ref('intent')
    pin=transaction.t.digest(transaction.t.canonical(plan))
    a=dict(plan_sha256=pin,remover_source=dict(path=str(c.Q/'minicpm5_retained_raw_transaction_v2.py'),sha256='d'*64),
        replay_source=dict(path=str(c.Q/'minicpm5_retained_raw_legacy_v1.py'),sha256='d'*64),
        independent_verification=ref('review'),full_history_compatibility=ref('history'),serial_lease=ref('lease'))
    review=dict(schema='minicpm5_retained_raw_replay_and_removal_review_v1',plan_sha256=pin,verification=verification,
        remover_source=a['remover_source'],replay_source=a['replay_source'],passed=True,blockers=[],
        full11859_logical_inputs_preserved=True,full32_replay_and_all_scientific_outputs_equal=True,
        all_original_failures_preserved=True,original_weights_precision_and_remaining_gates_unchanged=True,
        source_sha256=sources,replay_equality=ref('equality'))
    history=dict(schema='minicpm5_retained_raw_full_history_compatibility_v1',plan_sha256=pin,passed=True,
        complete_current_input_and_source_maps_checked=True,all_predecessor_owner_service_cgroup_scope_preserved=True,
        candidate_current_input_intersection=0)
    lease=dict(schema='minicpm5_retained_raw_exclusive_retirement_lease_v1',plan_sha256=pin,
        all_writers_and_holders_drained=True,retirement_source=a['remover_source'])
    intent=dict(admission=a,plan_sha256=pin,verification=verification)
    equality=dict(complete=True,replay_result=ref('replayed-science'))
    strict={a['remover_source']['path']:b'source',a['replay_source']['path']:b'source',
        a['independent_verification']['path']:review,transaction.t.REFERENCES['recovery_plan']['path']:dict(source_sha256=old),
        a['full_history_compatibility']['path']:history,a['serial_lease']['path']:lease}
    receipts={intent_ref['path']:intent,review['replay_equality']['path']:equality}
    expected=dict(plan_sha256=pin,intent=intent_ref,serial_lease=a['serial_lease'],
        all_writers_and_holders_drained=True,exclusive_namespace_write_lease=True)
    return SimpleNamespace(old=old,sources=sources,plan=plan,verification=verification,intent_ref=intent_ref,
        strict=strict,receipts=receipts,expected=expected,equality=equality,review=review)


class AdmissionIntegration(unittest.TestCase):
    def test_all_2447_hashes_original_1869_and_repeated_shared_checks_are_retained(self):
        f=admission_fixture();reads=[];identities=[];strict_reads=[];comparator=[]
        def identity(path):identities.append(str(path));return dict(path=str(path),nlink=2 if str(path)==source.SHARED_PATH else 1)
        def read(ref):self.assertEqual(ref['sha256'],f.sources[ref['path']]);reads.append(ref['path']);return b'synthetic'
        def evidence(ref,suffix):strict_reads.append((ref['path'],suffix));return copy.deepcopy(f.strict[ref['path']])
        def compare(*args):comparator.append(args);args[-1]();return f.equality
        with patch.object(transaction,'_load_evidence',side_effect=evidence),patch.object(transaction.t,'read_receipt',side_effect=lambda ref:copy.deepcopy(f.receipts[ref['path']])),\
                patch.object(source,'read',side_effect=read),patch.object(source,'identity',side_effect=identity),\
                patch.object(transaction.comparison,'_compare',side_effect=compare),patch.dict(sys.modules,clear=True):
            admitted=transaction._admission(f.plan,f.verification,f.intent_ref,lambda:f.expected)
            self.assertEqual(set(reads),set(f.sources));self.assertEqual(len(set(reads)),2447)
            before=reads.count(source.SHARED_PATH);admitted();admitted.check_complete_sources()
            self.assertEqual(reads.count(source.SHARED_PATH),before+2)
        self.assertEqual(len(comparator),1);self.assertEqual(len(f.old),1869)
        self.assertIn((transaction.t.REFERENCES['recovery_plan']['path'],'.json'),strict_reads)
        self.assertTrue(all(suffix=='.json'for _,suffix in strict_reads[2:]))
        self.assertGreaterEqual(identities.count(source.SHARED_PATH),4)

    def test_missing_original_pin_and_repeated_shared_identity_change_are_rejected(self):
        for change in ('missing_original','shared_identity_drift'):
            f=admission_fixture();count=[0]
            if change=='missing_original':f.sources.pop(next(k for k in f.old if k!=source.SHARED_PATH))
            def identity(path):
                if str(path)==source.SHARED_PATH:count[0]+=1
                return dict(path=str(path),revision=int(str(path)==source.SHARED_PATH and count[0]>1))
            with patch.object(transaction,'_load_evidence',side_effect=lambda ref,suffix:copy.deepcopy(f.strict[ref['path']])),\
                    patch.object(transaction.t,'read_receipt',side_effect=lambda ref:copy.deepcopy(f.receipts[ref['path']])),\
                    patch.object(source,'read',return_value=b'synthetic'),patch.object(source,'identity',side_effect=identity),\
                    patch.object(transaction.comparison,'_compare',return_value=f.equality),patch.dict(sys.modules,clear=True):
                with self.subTest(change=change),self.assertRaises(ValueError):
                    transaction._admission(f.plan,f.verification,f.intent_ref,lambda:f.expected)

    def test_frozen_deletion_bodies_metadata_reader_and_scope_behavior_are_byte_identical(self):
        def functions(path):
            text=Path(path).read_text();tree=ast.parse(text)
            return {node.name:ast.get_source_segment(text,node)for node in tree.body if isinstance(node,ast.FunctionDef)}
        old=functions(frozen.__file__);new=functions(transaction.__file__)
        for name in ('_load_evidence','_unlink_exact','_retire','retire','durable','expected_ref'):
            self.assertEqual(old[name],new[name],name)
        q=Path(c.Q)
        old_scope=(q/'minicpm5_retained_retirement_scope_v1.py').read_text()
        new_scope=(q/'minicpm5_retained_retirement_scope_v2.py').read_text()
        self.assertEqual(old_scope.replace('contract_v2 as c','contract_v3 as c'),new_scope)
        calls=[ast.unparse(n.func)for n in ast.walk(ast.parse(Path(source.__file__).read_text()))if isinstance(n,ast.Call)]
        self.assertFalse(any(x.split('.')[-1]in ('unlink','link','replace','rename','chmod','chown','kill','flock')for x in calls))


if __name__=='__main__':unittest.main(verbosity=2)
