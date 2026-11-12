"""Bounded stdlib/mock checks; no archive/native/serial operation or real snapshot."""
import ast
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch,Mock
Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
DENIED={'torch','numpy','ctypes','_ctypes','transformers','vllm','torch_neuronx','nki','neuronxcc','neuronx_distributed','neuronx_distributed_inference'}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in DENIED:raise RuntimeError('native forbidden: '+fullname)
sys.meta_path.insert(0,Blocker())
import minicpm5_sdk310_snapshot_read_binding_v1 as b
import minicpm5_sdk310_snapshot_contract_v1 as c
import dory_serial_stage_contract_v2 as old_c
import prepare_minicpm5_sdk310_snapshot_archive_v1 as builder


def local_json(ref):
    p=Path(ref['path']);raw=p.read_bytes()
    assert p.suffix=='.json' and len(raw)<4*2**20 and hashlib.sha256(raw).hexdigest()==ref['sha256']
    return json.loads(raw)


def fixture():
    n=local_json(b.NEXT);assessment=local_json(b.ASSESSMENT)
    old=local_json(b.OLD_MANIFEST);stage_sources=local_json(b.STAGE_MANIFEST)['source_sha256']
    pins=dict(stage_sources)
    for name in b.READER_NAMES:pins[str(Q/name)]=hashlib.sha256((Q/name).read_bytes()).hexdigest()
    values={b.NEXT['path']:n,b.ASSESSMENT['path']:assessment,b.OLD_MANIFEST['path']:old,
            b.STAGE_MANIFEST['path']:{'source_sha256':stage_sources}}
    def ref(name,value):
        path=str(Q/('mock-only-'+name+'.json'))
        values[path]=value
        return dict(path=path,sha256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest())
    root=Path('/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-snapshot-mock-only')
    rows=[dict(suite=r['suite'],logical_ref=r['original_logical_index_ref'],
               snapshot_ref=dict(path=str(root/'indexes'/r['proposed_snapshot_basename']),**r['expected_uncompressed']),
               compressed_ref=r['compressed'],release=r['release'],status_text=r['status_text'],
               saved_signature_proof=r['saved_signature_proof']) for r in assessment['rows']]
    manifest=dict(schema=b.SNAPSHOT_SCHEMA,complete=True,actual_process_exit_claimed=False,
                  system_APT_files_changed=False,native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False,
                  assessment=copy.deepcopy(b.ASSESSMENT),original_inputs=copy.deepcopy(b.NEXT),rows=rows,raw_total_bytes=50319218,
                  **{k:n[k] for k in ('signed_metadata','producer_actual_exit','recorder_actual_exit')})
    mr=dict(path=str(root/'snapshot-manifest.json'),sha256='a'*64,bytes=1000);values[mr['path']]=manifest
    stage_plan=dict(stage='inventory',budget=copy.deepcopy(c.INVENTORY),automatic_successor_launch=False,
                    source_sha256=stage_sources,output_root=str(root))
    sp=ref('stage-plan',stage_plan)
    sr=ref('stage-review',dict(reviewer='/root',passed=True,blockers=[],plan=sp,source_sha256=stage_sources))
    owners=[dict(pid=i,start_ticks=str(i*10)) for i in (101,102,103)]
    ended=dict(schema='minicpm5_sdk310_snapshot_stage_actual_exit_v1',stage_success_observed=True,
               owners_gone=True,child_group_empty=True,recorder_self_exit_claimed=False,result=mr,
               actual_tool=dict(kind='tool_session',session_id=90001,exit_code=0,terminal_chunk='abcdef'),
               owner=owners[0],child_owner=owners[1],recorder=owners[2],plan=sp,review=sr,output_root=str(root),
               evidence={'snapshot-manifest.json':mr,**{'indexes/'+r['suite']+'-main-Packages':r['snapshot_ref'] for r in rows}})
    er=ref('stage-exit',ended)
    recorded=dict(schema='minicpm5_sdk310_snapshot_recorder_actual_exit_v1',actual_tool_exit_observed=True,
                  owner_gone=True,owner=owners[2],result=er,
                  actual_tool=dict(kind='tool_call',exit_code=0,terminal_chunk='fedcba'))
    rr=ref('recorder-exit',recorded)
    catalog=ref('catalog',dict(runs=[dict(terminal=r,tool=v['actual_tool'],source_sha256=stage_sources)
                                   for r,v in ((er,ended),(rr,recorded))]))
    plan=dict(signed_index_snapshot=dict(manifest=mr,stage_exit=er,recorder_exit=rr),source_sha256=pins,
              complete_current_serial_catalog=catalog,predecessor_owners=owners,metadata_source_count_limit=103)
    # Each saved JSON is independent; mutation tests must not share references
    # with global fixed pins, other records or the supplied plan.
    values={key:copy.deepcopy(value) for key,value in values.items()}
    plan=copy.deepcopy(plan);n=copy.deepcopy(n)
    def read(r):return copy.deepcopy(values[r['path']])
    return plan,n,values,read


class ReaderTests(unittest.TestCase):
    def test_actual_zero_accepts_real_tool_calls_and_sessions_only(self):
        call=dict(kind='tool_call',terminal_chunk='c08c7b',exit_code=0)
        session=dict(kind='tool_session',session_id=35357,terminal_chunk='b7f150',exit_code=0)
        b.actual_zero(call);b.actual_zero(session)
        for invalid in (None,{},dict(call,kind='manifest'),dict(call,exit_code=1),
                        dict(call,exit_code=False),dict(call,terminal_chunk=''),
                        dict(call,terminal_chunk='not-a-tool-chunk'),dict(call,session_id=90002),
                        dict(call,session_id=None),dict(call,extra='unobserved'),
                        {k:v for k,v in session.items() if k!='session_id'},
                        dict(session,session_id=0),dict(session,session_id=True),
                        dict(session,session_id='35357')):
            with self.subTest(value=invalid),self.assertRaises(RuntimeError):b.actual_zero(invalid)

    def test_complete_chain_keeps_original_inputs(self):
        plan,n,values,read=fixture();before=copy.deepcopy(n)
        result=b.snapshot_chain(plan,read,n)
        self.assertEqual(n,before);self.assertEqual(len(result['rows']),3)
        resolver=b.SnapshotReadResolver(plan,lambda:None,n,read_metadata=read)
        self.assertEqual(set(resolver.rows),{r['current_uncompressed_index']['path'] for r in n['signed_index_reuse']})

    def test_completion_chain_rejections(self):
        cases=[('manifest',lambda v:v.update(complete=False)),
               ('manifest',lambda v:v.update(actual_process_exit_claimed=True)),
               ('manifest',lambda v:v.update(system_APT_files_changed=True)),
               ('manifest',lambda v:v.update(raw_total_bytes=0)),
               ('manifest',lambda v:v['original_inputs'].update(sha256='f'*64)),
               ('stage_exit',lambda v:v.update(stage_success_observed=False)),
               ('stage_exit',lambda v:v['actual_tool'].update(exit_code=1)),
               ('stage_exit',lambda v:v.update(owners_gone=False)),
               ('stage_exit',lambda v:v.update(child_group_empty=False)),
               ('stage_exit',lambda v:v.update(recorder_self_exit_claimed=True)),
               ('stage_exit',lambda v:v['result'].update(sha256='f'*64)),
               ('recorder_exit',lambda v:v.update(actual_tool_exit_observed=False)),
               ('recorder_exit',lambda v:v.update(owner_gone=False)),
               ('recorder_exit',lambda v:v['actual_tool'].update(exit_code=1)),
               ('recorder_exit',lambda v:v['owner'].update(pid=90000)),
               ('recorder_exit',lambda v:v['result'].update(sha256='f'*64))]
        for key,mutate in cases:
            with self.subTest(key=key,mutate=mutate):
                plan,n,values,read=fixture();mutate(values[plan['signed_index_snapshot'][key]['path']])
                with self.assertRaises((RuntimeError,KeyError)):b.snapshot_chain(plan,read,n)
        for key in b.BINDING_KEYS:
            plan,n,values,read=fixture();del plan['signed_index_snapshot'][key]
            with self.assertRaises(RuntimeError):b.snapshot_chain(plan,read,n)

    def test_mapping_rejects_alias_hash_size_duplicate_order_and_extra(self):
        changes=[lambda r:r['logical_ref'].update(path='/tmp/other'),
                 lambda r:r['snapshot_ref'].update(path='/tmp/other'),
                 lambda r:r['snapshot_ref'].update(sha256='f'*64),
                 lambda r:r['snapshot_ref'].update(bytes=1),
                 lambda r:r['compressed_ref'].update(sha256='f'*64),
                 lambda r:r['release'].update(sha256='f'*64),
                 lambda r:r['saved_signature_proof'].update(verified=False),
                 lambda r:r.update(extra='undeclared')]
        for mutate in changes:
            plan,n,values,read=fixture();m=values[plan['signed_index_snapshot']['manifest']['path']]
            mutate(m['rows'][0])
            with self.assertRaises(RuntimeError):b.snapshot_chain(plan,read,n)
        for mode in ('missing','duplicate','reordered'):
            plan,n,values,read=fixture();rows=values[plan['signed_index_snapshot']['manifest']['path']]['rows']
            if mode=='missing':rows.pop()
            if mode=='duplicate':rows[1]=copy.deepcopy(rows[0])
            if mode=='reordered':rows.reverse()
            with self.assertRaises(RuntimeError):b.snapshot_chain(plan,read,n)

    def test_source_and_history_rejections(self):
        for mode in ('old_pin','new_source_removed','owner_removed','catalog_missing','catalog_tool','catalog_source'):
            plan,n,values,read=fixture()
            if mode=='old_pin':plan['source_sha256'][next(iter(local_json(b.OLD_MANIFEST)['source_sha256']))]='f'*64
            if mode=='new_source_removed':del plan['source_sha256'][str(Q/b.READER_NAMES[0])]
            if mode=='owner_removed':plan['predecessor_owners'].pop()
            catalog=values[plan['complete_current_serial_catalog']['path']]
            if mode=='catalog_missing':catalog['runs'].pop()
            if mode=='catalog_tool':catalog['runs'][0]['tool']=dict(kind='tool_session',session_id=1,exit_code=0,terminal_chunk='aab')
            if mode=='catalog_source':catalog['runs'][0]['source_sha256']={}
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):b.snapshot_chain(plan,read,n)

    def test_reader_full_hash_stability_and_plaintext_passthrough(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);physical=root/'Packages';physical.write_bytes(b'Package: fixture\n\n');physical.chmod(0o400)
            logical=dict(path='/var/lib/apt/lists/mock-only-Packages',bytes=physical.stat().st_size,
                         sha256=hashlib.sha256(physical.read_bytes()).hexdigest())
            target=dict(logical,path=str(physical))
            resolver=b.SnapshotReadResolver.__new__(b.SnapshotReadResolver)
            resolver.guard=lambda:None;resolver.rows={logical['path']:dict(logical_ref=logical,snapshot_ref=target)}
            self.assertEqual(resolver.read(logical,{logical['path']:logical}),b'Package: fixture\n\n')
            other=root/'status';other.write_bytes(b'fixed metadata')
            ref=dict(path=str(other),bytes=other.stat().st_size,sha256=hashlib.sha256(other.read_bytes()).hexdigest())
            self.assertEqual(resolver.read(ref,{ref['path']:ref}),b'fixed metadata')
            with self.assertRaises(RuntimeError):resolver.read(dict(logical,sha256='f'*64),{logical['path']:logical})
            with self.assertRaises(RuntimeError):resolver.read(logical,{})
            physical.chmod(0o600)
            with self.assertRaises(RuntimeError):resolver.read(logical,{logical['path']:logical})
            physical.chmod(0o400);target['sha256']='f'*64
            with self.assertRaises(RuntimeError):resolver.read(logical,{logical['path']:logical})
            physical.unlink()
            with self.assertRaises(FileNotFoundError):resolver.read(logical,{logical['path']:logical})

    def test_reader_rejects_hardlink_symlink_and_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'Packages';p.write_bytes(b'bytes');p.chmod(0o400)
            logical=dict(path='/var/lib/apt/lists/mock-only',bytes=5,sha256=hashlib.sha256(b'bytes').hexdigest())
            target=dict(logical,path=str(p));r=b.SnapshotReadResolver.__new__(b.SnapshotReadResolver)
            r.guard=lambda:None;r.rows={logical['path']:dict(logical_ref=logical,snapshot_ref=target)}
            link=p.with_name('link');os.link(p,link)
            with self.assertRaises(RuntimeError):r.read(logical,{logical['path']:logical})
            link.unlink();p.rename(link);p.symlink_to(link)
            with self.assertRaises(RuntimeError):r.read(logical,{logical['path']:logical})
            p.unlink();link.rename(p)
            def mutate(*args):p.chmod(0o600);return b'bytes'
            with patch.object(b.original,'read_text',mutate),self.assertRaises(RuntimeError):r.read(logical,{logical['path']:logical})

    def test_all_original_ancestry_reads_use_explicit_resolver(self):
        plan,n,values,_=fixture();seen=[];before=copy.deepcopy(n)
        status=dict(path='/var/lib/dpkg/status',bytes=10,sha256='a'*64)
        plain={status['path']:('\n\n'.join('Package: '+name+'\nStatus: install ok installed\nVersion: '+version+'\nArchitecture: amd64' for name,version in b.PACKAGE_VERSIONS.items())+'\n\n').encode()}
        for row in n['packages']:plain[row['dpkg_list']['path']]=('\n'.join(row['saved_package_names'])+'\n').encode()
        def read(ref,*args):
            path=ref['path']
            if path==b.NEXT['path']:return copy.deepcopy(n)
            if path==n['signed_metadata']['path']:return dict(status='exact_addresses_ready',missing=[])
            if path==n['producer_actual_exit']['path']:return dict(stage_success_observed=True,actual_tool={'exit_code':0},complete_events_audit={'recomputed':{'entire_discovery_reproduced':True}},evidence={'signed-metadata/sources-result.json':n['signed_metadata']})
            if path==n['recorder_actual_exit']['path']:return dict(actual_tool={'exit_code':0},owner_gone=True,result=n['producer_actual_exit'])
            if path==n['trust_authority']['path']:return {}
            if path==n['source_gap_report']['path']:return {'metadata_inputs':[status]}
            if path==b.TOOL['path']:return dict(schema='dory_dpkg_deb_tool_authority_v1',reviewer='/root',tool={'path':'/usr/bin/dpkg-deb'})
            if path==n['dpkg_deb_tool_actual_exit']['path']:return dict(actual_tool={'exit_code':0},owner_gone=True,result=b.TOOL)
            raise AssertionError(path)
        self_test=self
        class FakeResolver:
            def __init__(self,actual,guard,manifest):self_test.assertIs(actual,plan);self_test.assertEqual(manifest,n)
            def read(self,ref,allowed):seen.append(ref);self_test.assertEqual(allowed[ref['path']],ref);return plain.get(ref['path'],b'fake plaintext')
        def ancestry(manifest,signed,authority,read,guard,now):
            for chain in manifest['signed_index_reuse']:
                for key in ('status_text','release','current_uncompressed_index'):read(chain[key])
            return 14
        with patch.object(c,'load',read),patch.object(b,'SnapshotReadResolver',FakeResolver),patch.object(b,'verify_ancestry',ancestry):
            result=b.authenticate(plan,lambda:None)
        self.assertEqual(result['signed_index_snapshot'],plan['signed_index_snapshot'])
        self.assertEqual(n,before)
        for chain in n['signed_index_reuse']:self.assertIn(chain['current_uncompressed_index'],seen)
        self.assertEqual(len(seen),18)

    def test_save_collected_reauthenticates_same_plan(self):
        plan,n,_,_=fixture();out=Path('/tmp/mock-collected-only');ref=dict(path=str(out/'packages/sources-result.json'),sha256='a'*64)
        auth={'signed_index_snapshot':plan['signed_index_snapshot']};saved=[]
        with patch.object(b,'authenticated',return_value=(n,{})) as authenticate,patch.object(c,'load',return_value={}),patch.object(b,'validate_result'),patch.object(c,'save',side_effect=lambda p,v:saved.append(v) or {'path':str(p)}),patch.object(b,'storage_usage',return_value=0):
            b.save_collected(out,dict(result=ref,mode='download',replay_from=None),auth,{}, {},lambda:None,plan=plan)
            self.assertIs(authenticate.call_args.args[0],plan)
        self.assertEqual(saved[0]['metadata_input_binding'],auth)
        with self.assertRaises(RuntimeError):b.save_collected(out,dict(result=ref),{}, {},{},lambda:None,plan=plan)

    def test_adapter_full_body_keeps_eight_packages_twelve_targets(self):
        plan,n,_,_=fixture();saved=[];events=[]
        targets={t['observed_path']:t for row in n['packages'] for t in row['targets']}
        def inspect(row,archive,tool,guard,emit):
            return dict(archive=dict(path=str(archive),bytes=row['bytes'],sha256=row['sha256']),
                        tar=dict(stream_bytes=1,stream_sha256='a'*64,targets={t['observed_path']:dict(size=1,sha256='b'*64,terminal=t['archive_member']) for t in row['targets']}))
        def file_ref(path,guard,expected,**kwargs):
            t=targets[path];return dict(path=path,**expected),{k:t['current_lstat']['st_'+k] for k in ('dev','ino','mode','nlink','size','mtime_ns','ctime_ns')}
        with tempfile.TemporaryDirectory() as td,patch.object(b,'authenticated',return_value=(n,{'tool':{}})) as authenticate,patch.object(b.io,'download'),patch.object(b.io,'inspect',side_effect=inspect),patch.object(b.io,'file_ref',side_effect=file_ref),patch.object(c,'save',side_effect=lambda p,v,*a:saved.append(v) or {'path':str(p),'sha256':'a'*64}):
            result=b.run(plan,lambda:None,events.append,b.TRUST,Path(td)/'packages','download')
            self.assertIs(authenticate.call_args.args[0],plan)
        self.assertEqual((result['packages'],result['targets']),(8,12))
        self.assertEqual(saved[0]['inputs'],b.TRUST)
        self.assertEqual([p['package'] for p in saved[0]['packages']],list(b.PACKAGE_VERSIONS))
        self.assertFalse(saved[0]['complete_native_dependency_closure_established'])

    def test_new_contract_only_source_count_changes(self):
        pins={str(Q/f'mock-{i}.py'):'a'*64 for i in range(103)}
        with patch.object(c,'reference',side_effect=lambda p,cap:({'bytes':10,'sha256':pins[p]},b'')):
            self.assertEqual(c.verify_sources(pins),1030)
        with self.assertRaises(RuntimeError):old_c.verify_sources(pins)
        with self.assertRaises(RuntimeError):c.verify_sources(dict(pins,**{str(Q/'extra.py'):'a'*64}))
        with patch.object(c,'reference',return_value=({'bytes':8*c.MIB,'sha256':'a'*64},b'')),self.assertRaises(RuntimeError):c.verify_sources(pins)
        self.assertEqual(c.INVENTORY,old_c.INVENTORY)

    def test_runtime_bodies_preserve_original_control_flow(self):
        def fun(file,name):
            source=(Q/file).read_text();return ast.get_source_segment(source,next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name))
        parent=fun('run_minicpm5_sdk310_deb_members_root_v3.py','main').replace('def main():','def parent_main(argv=None):').replace('a=parser.parse_args()','a=parser.parse_args(argv)').replace('inputs.authenticate(guard)','inputs.authenticate(plan,guard)').replace("worker=str(Q/'minicpm5_sdk310_deb_members_worker_v3.py')","worker=str(Q/'run_minicpm5_sdk310_snapshot_archive_v1.py')").replace("argv=[python,'-B','-s',worker,'--plan'","argv=[python,'-B','-s',worker,'--mode','worker','--plan'")
        worker=fun('minicpm5_sdk310_deb_members_worker_v3.py','main').replace('def main():','def worker_main(argv=None):').replace('a=parser.parse_args()','a=parser.parse_args(argv)').replace('from minicpm5_sdk310_deb_members_adapter_v1 import run','from minicpm5_sdk310_snapshot_read_binding_v1 import run').replace('inputs.authenticate(guard)','inputs.authenticate(plan,guard)').replace('value=run(guard,','value=run(plan,guard,').replace("sha256=a.admission_sha256),guard)","sha256=a.admission_sha256),guard,plan=plan)")
        recorder=fun('record_minicpm5_sdk310_deb_stage_exit_v3.py','main').replace('def main():','def recorder_main(argv=None):').replace('a=cli.parse_args()','a=cli.parse_args(argv)').replace('binding.','inputs.').replace('inputs.authenticate(guard)','inputs.authenticate(plan,guard)').replace('inputs.authenticated(guard)','inputs.authenticated(plan,guard)')
        for name,want in (('parent_main',parent),('worker_main',worker),('recorder_main',recorder)):
            self.assertEqual(ast.dump(ast.parse(fun('run_minicpm5_sdk310_snapshot_archive_v1.py',name))),ast.dump(ast.parse(want)))
        tree=ast.parse((Q/'run_minicpm5_sdk310_snapshot_archive_v1.py').read_text())
        imports=[n for n in tree.body if isinstance(n,ast.Import)]
        self.assertTrue(any(a.name=='minicpm5_sdk310_snapshot_contract_v1' and a.asname=='c' for n in imports for a in n.names))
        self.assertFalse(any(a.name=='dory_serial_stage_contract_v2' and a.asname=='c' for n in imports for a in n.names))

    def test_builder_retains_old_plan_and_requires_completed_chain(self):
        plan,n,values,read=fixture();manifest=dict(source_sha256=plan['source_sha256'],complete_retained_source_count=103,
            archive_stage_budget=c.INVENTORY,scientific_constraints=b.CONSTRAINTS)
        mr={'path':str(Q/'mock-reader-manifest.json'),'sha256':'b'*64};rr={'path':str(Q/'mock-reader-review.json'),'sha256':'c'*64}
        values[mr['path']]=manifest;values[rr['path']]=dict(reviewer='/root',passed=True,blockers=[],manifest=mr,metadata_source_count_limit=103)
        baseline=copy.deepcopy(plan);baseline['source_sha256']=local_json(b.OLD_MANIFEST)['source_sha256']
        baseline['original_gate_marker']={'unchanged':True}
        with patch.object(builder.old,'build',return_value=baseline) as old:
            result=builder.build(read,snapshot_binding=plan['signed_index_snapshot'],reader_source_manifest=mr,reader_source_acceptance=rr,mode='download',output_root='/unused')
            old.assert_called_once_with(read,mode='download',output_root='/unused')
        self.assertEqual(result['original_gate_marker'],{'unchanged':True})
        self.assertEqual(len(result['source_sha256']),103)
        values[rr['path']]['passed']=False
        with patch.object(builder.old,'build',return_value=copy.deepcopy(baseline)),self.assertRaises(RuntimeError):
            builder.build(read,snapshot_binding=plan['signed_index_snapshot'],reader_source_manifest=mr,reader_source_acceptance=rr)

    def test_imports_are_native_free(self):
        self.assertFalse(DENIED.intersection(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
