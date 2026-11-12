"""Bounded source/metadata tests; no archive, native, serial or index payload IO."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
import check_minicpm5_sdk310_snapshot_reader_v1 as fixture_v1
import minicpm5_sdk310_snapshot_read_binding_v2 as b
import minicpm5_sdk310_snapshot_contract_v2 as c
import minicpm5_sdk310_snapshot_contract_v1 as c1
import prepare_minicpm5_sdk310_snapshot_archive_v2 as builder
EVIDENCE=dict(path=str(Q/'minicpm5-sdk310-dpkg-status-delta-evidence-20261112-v1.json'),sha256='f19d6c702312bb3c68e77822715e6666681fd1d7f87ec50e60bcc52655ee6fab')


def fixture():
    plan,n,values,_=fixture_v1.fixture()
    evidence=c.load(EVIDENCE);gap=c.load(n['source_gap_report'])
    observation={k:copy.deepcopy(evidence[k]) for k in ('old_snapshot','current_snapshot','current_ref')}
    observation['delta_evidence']=copy.deepcopy(EVIDENCE)
    plan['dpkg_status_observation']=observation;plan['metadata_source_count_limit']=106
    values[b.READER_V1_MANIFEST['path']]=c.load(b.READER_V1_MANIFEST)
    for name in b.STATUS_NAMES:plan['source_sha256'][str(Q/name)]=hashlib.sha256((Q/name).read_bytes()).hexdigest()
    values[EVIDENCE['path']]=copy.deepcopy(evidence);values[n['source_gap_report']['path']]=copy.deepcopy(gap)
    def read(ref):return copy.deepcopy(values[ref['path']])
    return plan,n,gap,evidence,values,read


class StatusReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence=c.load(EVIDENCE)
        cls.old=b.immutable_status(cls.evidence['old_snapshot'],lambda:None)
        cls.current=b.immutable_status(cls.evidence['current_snapshot'],lambda:None)

    def test_real_full_delta_matches_root_saved_evidence(self):
        self.assertEqual(b.status_delta(self.old,self.current),self.evidence['changed_packages'])

    def test_real_current_status_is_explicitly_reauthenticated(self):
        plan,n,gap,evidence,_,_=fixture();before=copy.deepcopy(n)
        self.assertEqual(b.authenticate_status(plan,lambda:None,n,gap,b.OLD_STATUS),self.current)
        self.assertEqual(n,before)
        for row in n['packages']:
            ref=row['dpkg_list'];b.original.read_text(ref,lambda:None,{ref['path']:ref})

    def test_delta_rejects_any_other_bytes_or_package_change(self):
        pieces=self.current.split(b'\n\n')
        candidates=[]
        candidates.append(self.current.replace(b'Package: libffi8\n',b'Package: libffi9\n',1))
        target=next(i for i,p in enumerate(pieces) if p.startswith(b'Package: libffi8\n'))
        changed=next(i for i,p in enumerate(pieces) if p.startswith(b'Package: gstreamer1.0-plugins-base\n'))
        for index,old,new in ((target,b'Status: install ok installed',b'Status: hold ok installed'),
                              (target,b'Architecture: amd64',b'Architecture: arm64'),
                              (changed,b'Installed-Size: 2401',b'Installed-Size: 2402'),
                              (changed,b'Version: 1.20.1-1ubuntu0.8',b'Version: 1.20.1-1ubuntu0.9')):
            tmp=list(pieces);self.assertIn(old,tmp[index]);tmp[index]=tmp[index].replace(old,new,1);candidates.append(b'\n\n'.join(tmp))
        tmp=list(pieces);tmp[0],tmp[1]=tmp[1],tmp[0];candidates.append(b'\n\n'.join(tmp))
        candidates += [self.current+b'\n',b'\n\n'.join(pieces[1:]),self.old,self.current.replace(b'Package: ',b'Package:  ',1)]
        for raw in candidates:
            with self.subTest(sha=hashlib.sha256(raw).hexdigest()),self.assertRaises((RuntimeError,ValueError,KeyError)):
                b.status_delta(self.old,raw)

    def test_status_metadata_rejects_swapped_or_inexact_observation(self):
        for mode in ('old','current','snapshot','missing','evidence_failed','evidence_gap','evidence_fields','evidence_lists','old_gap'):
            plan,n,gap,evidence,values,read=fixture()
            bound=plan['dpkg_status_observation'];saved=values[EVIDENCE['path']]
            if mode=='old':bound['old_snapshot']['sha256']=b.CURRENT_STATUS['sha256']
            if mode=='current':bound['current_ref']=copy.deepcopy(b.OLD_STATUS)
            if mode=='snapshot':bound['current_snapshot']['path']='/tmp/status'
            if mode=='missing':del bound['old_snapshot']
            if mode=='evidence_failed':saved['passed']=False
            if mode=='evidence_gap':saved['source_gap_report']['sha256']='f'*64
            if mode=='evidence_fields':saved['changed_packages'][0]['fields']['Version']['new']='other'
            if mode=='evidence_lists':saved['unchanged_package_lists'].pop()
            if mode=='old_gap':gap['metadata_inputs'].append(copy.deepcopy(b.OLD_STATUS))
            with self.subTest(mode=mode),self.assertRaises((RuntimeError,KeyError)):
                b.status_metadata(plan,read,n,gap,b.OLD_STATUS)

    def test_live_read_uses_current_ref_and_rejects_later_drift(self):
        plan,n,gap,evidence,_,_=fixture();seen=[]
        def read(ref,guard,allowed,cap):
            seen.append(copy.deepcopy(ref));self.assertEqual(ref,b.CURRENT_STATUS);return b'drift'
        with patch.object(b,'immutable_status',side_effect=[self.old,self.current]),patch.object(b.original,'read_text',read):
            with self.assertRaises(RuntimeError):b.authenticate_status(plan,lambda:None,n,gap,b.OLD_STATUS)
        self.assertEqual(seen,[b.CURRENT_STATUS])

    def test_snapshot_rejects_writable_and_changed_full_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'status';p.write_bytes(b'tiny plaintext');raw=p.read_bytes()
            ref=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            with self.assertRaises(RuntimeError):b.immutable_status(ref,lambda:None)
            p.chmod(0o400);self.assertEqual(b.immutable_status(ref,lambda:None),raw)
            with self.assertRaises(RuntimeError):b.immutable_status(dict(ref,sha256='f'*64),lambda:None)

    def test_snapshot_chain_keeps_all103_frozen_pins(self):
        plan,n,gap,evidence,values,read=fixture();before=copy.deepcopy(n)
        self.assertEqual(len(b.snapshot_chain(plan,read,n)['rows']),3);self.assertEqual(n,before)
        key=str(Q/'minicpm5_sdk310_snapshot_read_binding_v1.py');plan['source_sha256'][key]='f'*64
        with self.assertRaises(RuntimeError):b.snapshot_chain(plan,read,n)

    def test_new_count_only_contract_retains_all_byte_budgets(self):
        plan,*_=fixture();self.assertEqual(len(plan['source_sha256']),106)
        self.assertLess(c.verify_sources(plan['source_sha256']),8*c.MIB)
        with self.assertRaises(RuntimeError):c1.verify_sources(plan['source_sha256'])
        with self.assertRaises(RuntimeError):c.verify_sources(dict(plan['source_sha256'],**{str(Q/'extra.py'):'f'*64}))
        with patch.object(c,'reference',return_value=({'bytes':8*c.MIB,'sha256':'a'*64},b'')),self.assertRaises(RuntimeError):
            c.verify_sources({str(Q/'fake.py'):'a'*64,str(Q/'fake2.py'):'a'*64})
        self.assertEqual(c.INVENTORY,c1.INVENTORY)

    def test_five_entry_paths_keep_explicit_plan_and_status_receipt(self):
        old=(Q/'run_minicpm5_sdk310_snapshot_archive_v1.py').read_text()
        new=(Q/'run_minicpm5_sdk310_snapshot_archive_v2.py').read_text()
        for stem in ('minicpm5_sdk310_snapshot_contract','minicpm5_sdk310_snapshot_read_binding','run_minicpm5_sdk310_snapshot_archive'):
            old=old.replace(stem+'_v1',stem+'_v2')
        self.assertEqual(ast.dump(ast.parse(old)),ast.dump(ast.parse(new)))
        plan,*_=fixture()
        with patch.object(b,'authenticated') as authenticate:
            receipt=b.authenticate(plan,lambda:None);self.assertIs(authenticate.call_args.args[0],plan)
        self.assertEqual(receipt['dpkg_status_observation'],plan['dpkg_status_observation'])
        source=(Q/'minicpm5_sdk310_snapshot_read_binding_v2.py').read_text();tree=ast.parse(source)
        functions={f.name:ast.get_source_segment(source,f) for f in tree.body if isinstance(f,ast.FunctionDef)}
        self.assertIn('current_status = authenticate_status(plan,guard,n,gap,status_refs[0])',functions['authenticated'])
        self.assertIn('metadata.stanzas(current_status, 8*c.MIB)',functions['authenticated'])
        self.assertNotIn('read(status_refs[0])',functions['authenticated'])
        self.assertIn('n, _ = authenticated(plan,guard)',functions['save_collected'])
        self.assertIn('n, authority = authenticated(plan,guard)',functions['run'])

    def test_builder_preserves_history_and_requires_status_evidence(self):
        plan,n,gap,evidence,values,read=fixture()
        manifest=dict(source_sha256=plan['source_sha256'],complete_retained_source_count=106,
                      archive_stage_budget=c.INVENTORY,scientific_constraints=b.CONSTRAINTS)
        mr=dict(path=str(Q/'mock-status-reader-manifest.json'),sha256='b'*64)
        rr=dict(path=str(Q/'mock-status-reader-review.json'),sha256='c'*64)
        values[mr['path']]=manifest;values[rr['path']]=dict(reviewer='/root',passed=True,blockers=[],manifest=mr,metadata_source_count_limit=106)
        base=copy.deepcopy(plan);base['source_sha256']=c.load(b.OLD_MANIFEST)['source_sha256'];base['history_marker']=['same']
        kwargs=dict(snapshot_binding=plan['signed_index_snapshot'],status_observation=plan['dpkg_status_observation'],reader_source_manifest=mr,reader_source_acceptance=rr)
        with patch.object(builder.old,'build',return_value=base):result=builder.build(read,**kwargs)
        self.assertEqual(result['history_marker'],['same']);self.assertEqual(len(result['source_sha256']),106)
        values[EVIDENCE['path']]['passed']=False
        with patch.object(builder.old,'build',return_value=copy.deepcopy(base)),self.assertRaises(RuntimeError):builder.build(read,**kwargs)

    def test_no_native_modules_loaded(self):
        self.assertFalse(fixture_v1.DENIED.intersection(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
