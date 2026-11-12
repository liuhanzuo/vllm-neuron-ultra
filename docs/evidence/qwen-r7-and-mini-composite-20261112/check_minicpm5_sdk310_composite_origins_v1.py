"""Bounded source/mock tests: only existing JSON/source reads, no native IO."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research import minicpm5_sdk310_composite_origins_v1 as m


def digest(value):return hashlib.sha256(value).hexdigest()


class Reader:
    def __init__(self):self.data={};self.raw={};self.reads=[];self.guards=0
    def guard(self):self.guards+=1
    def add(self,name,value):
        path='/synthetic/'+name+'.json';raw=json.dumps(value,sort_keys=True).encode()
        self.data[path]=copy.deepcopy(value)
        return dict(path=path,sha256=digest(raw),bytes=len(raw))
    def text(self,path,raw):
        ref=dict(path=path,bytes=len(raw),sha256=digest(raw));self.raw[path]=raw;return ref
    def json(self,ref):self.reads.append(ref['path']);return copy.deepcopy(self.data[ref['path']])
    def read(self,ref,cap):
        self.reads.append(ref['path']);raw=self.raw[ref['path']]
        if len(raw)>cap or len(raw)!=ref['bytes'] or digest(raw)!=ref['sha256']:raise ValueError('mock full text hash/size')
        return raw,dict(identity=dict(st_mode=0o100400,st_nlink=1))


def fixture():
    r=Reader();pins={'/synthetic/accepted.py':'a'*64}
    n=json.loads((ROOT/'research/minicpm5-sdk310-archive-member-next-inputs-20261112-v1.json').read_text())
    original_signed=json.loads(Path(n['signed_metadata']['path']).read_text())
    old=json.loads(Path(m.OLD_REPLAY['path']).read_text())
    tool=r.add('tool',{'synthetic':True});n['dpkg_deb_tool_authority']=tool
    trust=r.add('trust',dict(schema='dory_ubuntu_archive_key_authority_v1',independent_authority_verified=True,
        keyring=dict(sha256='a'*64),primary_fingerprints=['trusted']))
    n['trust_authority']=trust
    signed=dict(status='exact_addresses_ready',missing=[],packages=copy.deepcopy(original_signed['packages']),
        trust_anchor=dict(keyring_sha256='a'*64,primary_fingerprints=['trusted'],authority_ref=trust),releases=[])
    rows=[];assessment_rows=[];chains=[];raw_total=0
    for p in n['packages']:p['sources']=[]
    snapshot_root='/mnt/nvme/synthetic-snapshot'
    for index,suite in enumerate(('jammy','jammy-updates','jammy-security')):
        package_rows=n['packages'] if index==0 else n['packages'][:6] if index==1 else []
        raw=b'';stanzas={}
        for p in package_rows:
            fs=dict(package=p['package'],version=p['version'],architecture=p['architecture'],filename=p['filename'],
                    size=str(p['bytes']),sha256=p['sha256'])
            block='\n'.join(k+': '+v for k,v in fs.items()).encode()
            stanzas[p['package']]=(len(raw),block,fs);raw+=block+b'\n\n'
        if not raw:raw=b'Package: irrelevant\nVersion: 0\nArchitecture: amd64\n\n'
        logical=dict(path='/synthetic/logical/'+suite,bytes=len(raw),sha256=digest(raw))
        physical=r.text(snapshot_root+'/indexes/'+suite+'-main-Packages',raw);raw_total+=len(raw)
        release_raw=('Suite: '+suite+'\nSHA256:\n '+logical['sha256']+' '+str(len(raw))+' main/binary-amd64/Packages\n').encode()
        release_ref=r.text('/synthetic/'+suite+'-Release',release_raw)
        status=r.text('/synthetic/'+suite+'-status.txt',b'[GNUPG:] synthetic trusted proof\n')
        compressed=dict(path='/synthetic/'+suite+'.xz',bytes=1,sha256='c'*64)
        release_url='https://archive.ubuntu.com/ubuntu/dists/'+suite+'/Release'
        index_url=release_url.rsplit('/',1)[0]+'/main/binary-amd64/Packages.xz'
        proof=dict(verified=True,process_returncode=0,primary_fingerprint='trusted',keyring_sha256='a'*64,status_reference=status)
        release=dict(release_url=release_url,release_sha256=release_ref['sha256'],signature_proof=proof,
            indexes=[dict(url=index_url,compressed={k:compressed[k] for k in ('bytes','sha256')},
                          uncompressed={k:logical[k] for k in ('bytes','sha256')})])
        signed['releases'].append(release)
        chain=dict(suite=suite,current_uncompressed_index=logical,release=release_ref,status_text=status,
                   saved_signature_proof=proof,index_url=index_url)
        chains.append(chain)
        rows.append(dict(suite=suite,logical_ref=logical,snapshot_ref=physical,compressed_ref=compressed,
                         release=release_ref,status_text=status,saved_signature_proof=proof))
        assessment_rows.append(dict(suite=suite,original_logical_index_ref=logical,
            proposed_snapshot_basename=suite+'-main-Packages',expected_uncompressed={k:logical[k] for k in ('bytes','sha256')},
            compressed=compressed,release=release_ref,status_text=status,saved_signature_proof=proof))
        for p in package_rows:
            start,block,fs=stanzas[p['package']]
            p['sources'].append(dict(index=logical,index_url=index_url,release_url=release_url,
                release_sha256=release_ref['sha256'],url='https://archive.ubuntu.com/ubuntu/'+p['filename'],
                matched_stanza=dict(index=logical,fields=fs,byte_offset=start,bytes=len(block),sha256=digest(block))))
    for p in signed['packages']:
        p['sources']=[dict(release_url=signed['releases'][0]['release_url'],release_sha256=signed['releases'][0]['release_sha256'])]
    n['signed_index_reuse']=chains
    n['signed_metadata']=r.add('signed',signed)
    n['producer_actual_exit']=r.add('signed-stage',dict(stage_success_observed=True,
        schema='dory_ubuntu_metadata_actual_exit_v1',parent_owner_gone=True,child_owner_gone=True,
        child_group_empty=True,recorder=dict(pid=9,start_ticks='100'),
        complete_events_audit=dict(recomputed=dict(entire_discovery_reproduced=True)),
        evidence={'signed-metadata/sources-result.json':n['signed_metadata']},
        actual_tool=dict(kind='tool_call',terminal_chunk='abc123',exit_code=0)))
    n['recorder_actual_exit']=r.add('signed-recorder',dict(owner_gone=True,result=n['producer_actual_exit'],
        schema='dory_ubuntu_metadata_recorder_actual_exit_v1',reviewer='/root',actual_tool_exit_observed=True,
        owner=dict(pid=9,start_ticks='100'),self_exit_claimed=False,
        actual_tool=dict(kind='tool_call',terminal_chunk='abc124',exit_code=0)))
    next_ref=r.add('next',n)

    def stage(name,prefix,result_ref,mode=None,inputs=None,prior=None,extra=None):
        root=str(Path(result_ref['path']).parent if mode is None else Path(result_ref['path']).parents[1])
        plan=dict(source_sha256=pins,stage='inventory',automatic_successor_launch=False,output_root=root)
        if mode is not None:plan.update(archive_mode=mode,archive_inputs=inputs,replay_from=prior)
        if extra:plan.update(extra)
        plan_ref=r.add(name+'-plan',plan)
        review=dict(reviewer='/root',passed=True,blockers=[],plan=plan_ref,source_sha256=pins)
        if mode is not None:review.update(archive_mode=mode,archive_inputs=inputs,replay_from=prior)
        review_ref=r.add(name+'-review',review)
        owner=dict(pid=100+len(r.data),start_ticks='1234')
        ended=dict(schema=prefix+'_stage_actual_exit_v1',stage_success_observed=True,owners_gone=True,
            child_group_empty=True,recorder_self_exit_claimed=False,result=result_ref,mode=mode,
            actual_tool=dict(kind='tool_session',terminal_chunk='def123',exit_code=0,session_id=71),
            recorder=owner,plan=plan_ref,review=review_ref,output_root=root,lock_identity=dict(device=1,inode=2),
            evidence={str(Path(result_ref['path']).relative_to(root)):result_ref},
            full_model_qualified=False,new_speedup_claim=False)
        if mode is None:
            ended['evidence'].update({'indexes/'+x['suite']+'-main-Packages':x['snapshot_ref'] for x in rows})
        ended_ref=r.add(name+'-exit',ended)
        recorded=dict(schema=prefix+'_recorder_actual_exit_v1',reviewer='/root',actual_tool_exit_observed=True,
            owner_gone=True,owner=owner,result=ended_ref,full_model_qualified=False,new_speedup_claim=False,
            actual_tool=dict(kind='tool_call',terminal_chunk='def124',exit_code=0),actual_self_exit_claimed=False,
            original_lock_nonblocking_acquired_and_released=True,plan=plan_ref,review=review_ref,
            lock_identity=ended['lock_identity'])
        return ended_ref,r.add(name+'-recorder',recorded),plan_ref

    manifest=dict(schema='minicpm5_sdk310_signed_index_snapshot_v1',complete=True,original_inputs=next_ref,
        signed_metadata=n['signed_metadata'],producer_actual_exit=n['producer_actual_exit'],recorder_actual_exit=n['recorder_actual_exit'],
        assessment=r.add('assessment',dict(rows=assessment_rows)),rows=rows,raw_total_bytes=raw_total,
        actual_process_exit_claimed=False,system_APT_files_changed=False,native_ABI_qualified=False,
        full_model_qualified=False,new_speedup_claim=False)
    manifest_ref=r.add('snapshot-manifest',manifest)
    new_path=snapshot_root+'/snapshot-manifest.json';r.data[new_path]=r.data.pop(manifest_ref['path']);manifest_ref['path']=new_path
    se,sr,_=stage('snapshot','minicpm5_sdk310_snapshot',manifest_ref)
    snapshot=dict(manifest=manifest_ref,stage_exit=se,recorder_exit=sr)

    packages=[]
    for p in n['packages']:
        targets={};comparisons=[]
        for target in p['targets']:
            path=target['observed_path'];member=dict(size=target['current_lstat']['st_size'],sha256=digest(path.encode()),terminal=target['archive_member'])
            targets[path]=member
            comparisons.append(dict(target=target,member=dict(bytes=member['size'],sha256=member['sha256'],terminal=member['terminal']),
                current=dict(path=path,bytes=member['size'],sha256=member['sha256']),
                current_identity={k:target['current_lstat']['st_'+k] for k in ('dev','ino','mode','nlink','size','mtime_ns','ctime_ns')},
                original_matches=None,current_matches=True,new_observation_only=True))
        packages.append(dict(package=p['package'],version=p['version'],necessity=p['necessity'],actual_loaded=None,
            inspection=dict(archive={k:p[k] for k in ('bytes','sha256')},decompressor=dict(returncode=0),
                tar=dict(complete=True,stream_bytes=10240,targets=targets)),comparisons=comparisons))
    new=dict(schema='minicpm5_sdk310_deb_members_result_v1',complete=True,package_content_authenticated=True,
        packages=packages,archive_total_bytes=7110864,decompressed_total_bytes=81920,target_count=12,current_match_count=12,
        original_match_count=0,required_package_count=3,candidate_package_count=5,complete_native_dependency_closure_established=False,
        **{k:False for k in m.FALSE_CLAIMS})
    new['inputs']=dict(next_inputs=next_ref,tool_authority=tool)
    old['inputs']=dict(next_inputs=r.add('old-next',dict(signed_metadata=n['signed_metadata'])),tool_authority=tool)

    def make_bundle(name,result,prefix,extra=None):
        bound={};producer_ref=None
        for role,mode in [('producer','download'),('replay','replay')]:
            result=copy.deepcopy(result);result.update(mode=mode,replay_from=producer_ref)
            ref=r.add(name+'-'+role,result)
            path='/mnt/nvme/synthetic-'+name+'-'+role+'/packages/sources-result.json'
            r.data[path]=r.data.pop(ref['path']);ref['path']=path
            ended,recorded,_=stage(name+'-'+role,prefix,ref,mode,result['inputs'],producer_ref,extra)
            bound.update({role+'_result':ref,role+'_exit':ended,role+'_recorder_exit':recorded})
            producer_ref=ref
        return bound
    old_bound=make_bundle('old',old,'dory_deb')
    new_bound=make_bundle('new',new,'minicpm5_sdk310_deb',dict(signed_index_snapshot=snapshot,mini_scientific_constraints=m.CONSTRAINTS))
    native={};pure={}
    for p in [old['packages'][0],*packages]:
        for path,member in p['inspection']['tar']['targets'].items():
            row=dict(path=path,sha256=member['sha256'],bytes=member['size'],package=p['package'],version=p['version'])
            (pure if path.endswith('.py') else native)[path]=row
    decision=dict(schema='minicpm5_sdk310_native_source_review_v2',native_files=native,pure_sources=pure,
        package_origins=dict(schema='minicpm5_sdk310_composite_package_origins_v1',old_libc6=old_bound,new_eight=new_bound))
    return r,decision,pins,next_ref,old_bound['replay_result'],raw_total


class Checks(unittest.TestCase):
    def run_fixture(self,f):
        r,d,pins,n,old,total=f
        with patch.object(m,'NEXT',n),patch.object(m,'OLD_REPLAY',old),patch.object(m,'RAW_TOTAL',total):
            return m.replay_package_origins(r,d,pins)

    def test_complete_two_origins_and_full_three_plaintexts(self):
        f=fixture();r,d,*_=f
        out=self.run_fixture(f)
        self.assertEqual(out,{**d['native_files'],**d['pure_sources']})
        self.assertEqual(len([p for p in r.reads if p.endswith('-main-Packages')]),3)
        self.assertFalse(any(p.endswith(('.deb','.so','.xz')) for p in r.reads))
        self.assertGreater(r.guards,0)
        self.assertEqual(m.RAW_TOTAL,50319218)

    def test_failed_or_missing_external_completion_rejected(self):
        for branch in ('old_libc6','new_eight'):
            for role in ('producer','replay'):
                f=fixture();r,d,*_=f
                ref=d['package_origins'][branch][role+'_recorder_exit']
                r.data[ref['path']]['actual_tool']['exit_code']=1
                with self.assertRaises(ValueError):self.run_fixture(f)

    def test_immediate_completion_and_real_session_shapes(self):
        m.actual_zero(dict(kind='tool_call',terminal_chunk='abc123',exit_code=0))
        m.actual_zero(dict(kind='tool_session',terminal_chunk='abc123',session_id=1,exit_code=0))
        for value in [dict(kind='tool_call',terminal_chunk='abc123',session_id=1,exit_code=0),
                      dict(kind='tool_session',terminal_chunk='abc123',exit_code=0),
                      dict(kind='tool_call',terminal_chunk='abc123',exit_code=False)]:
            with self.assertRaises(ValueError):m.actual_zero(value)

    def test_independent_replay_cannot_reuse_producer_or_change_member(self):
        f=fixture();r,d,*_=f;b=d['package_origins']['new_eight'];b['replay_result']=b['producer_result']
        with self.assertRaises(ValueError):self.run_fixture(f)

    def test_actual_two_field_prior_ref_binds_three_field_result(self):
        f=fixture();r,d,*_=f;b=d['package_origins']['new_eight']
        r.data[b['replay_result']['path']]['replay_from'].pop('bytes')
        ended=r.data[b['replay_exit']['path']]
        r.data[ended['plan']['path']]['replay_from'].pop('bytes')
        r.data[ended['review']['path']]['replay_from'].pop('bytes')
        self.assertTrue(self.run_fixture(f))
        b['producer_result']['bytes']+=1
        with self.assertRaises(ValueError):self.run_fixture(f)
        f=fixture();r,d,*_=f;b=d['package_origins']['new_eight']
        r.data[b['replay_result']['path']]['packages'][0]['inspection']['tar']['complete']=False
        with self.assertRaises(ValueError):self.run_fixture(f)

    def test_absent_stage_source_and_changed_stage_review_rejected(self):
        f=fixture();f[2].clear()
        with self.assertRaises(ValueError):self.run_fixture(f)
        f=fixture();r,d,*_=f;e=r.data[d['package_origins']['new_eight']['producer_exit']['path']]
        r.data[e['review']['path']]['passed']=False
        with self.assertRaises(ValueError):self.run_fixture(f)

    def test_stanza_offset_hash_or_archive_tuple_rejected(self):
        for key,value in [('byte_offset',1),('sha256','0'*64)]:
            f=fixture();r,_,_,n,*_=f;r.data[n['path']]['packages'][0]['sources'][0]['matched_stanza'][key]=value
            with self.assertRaises(ValueError):self.run_fixture(f)

    def test_untrusted_signature_and_changed_snapshot_binding_rejected(self):
        f=fixture();r,_,_,n,*_=f;signed_ref=r.data[n['path']]['signed_metadata']
        r.data[signed_ref['path']]['releases'][0]['signature_proof']['primary_fingerprint']='untrusted'
        with self.assertRaises(ValueError):self.run_fixture(f)
        f=fixture();r,d,*_=f;ended=r.data[d['package_origins']['new_eight']['replay_exit']['path']]
        r.data[ended['plan']['path']]['signed_index_snapshot']['manifest']['sha256']='0'*64
        with self.assertRaises(ValueError):self.run_fixture(f)

    def test_complete_index_bytes_cannot_be_omitted_or_corrupted(self):
        f=fixture();r,*_=f;path=next(p for p in r.raw if p.endswith('-main-Packages'))
        r.raw[path]+=b'Package: injected\n\n'
        with self.assertRaises(ValueError):self.run_fixture(f)

    def test_no_fabricated_old_hash_or_current_loader_claim(self):
        for key,value in [('original_match_count',12),('loader_selection_established',True)]:
            f=fixture();r,d,*_=f;b=d['package_origins']['new_eight']
            for role in ('producer','replay'):r.data[b[role+'_result']['path']][key]=value
            with self.assertRaises(ValueError):self.run_fixture(f)

    def test_native_source_mismatch_and_unverified_provider_rejected(self):
        f=fixture();d=f[1];next(iter(d['native_files'].values()))['sha256']='0'*64
        with self.assertRaises(ValueError):self.run_fixture(f)
        f=fixture();d=f[1];next(iter(d['native_files'].values()))['package']='unknown'
        with self.assertRaises(ValueError):self.run_fixture(f)

    def test_worker_successor_only_changes_origin_dispatch_and_schema(self):
        q=ROOT/'research';old=ast.parse((q/'minicpm5_sdk_mapping_worker_native_v2.py').read_text())
        new=ast.parse((q/'minicpm5_sdk_mapping_worker_native_v3.py').read_text())
        class Normalize(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value=='minicpm5_sdk310_native_source_review_v2':node.value='minicpm5_sdk310_native_source_review_v1'
                return node
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name) and node.func.id=='replay_package_origins':node.args=node.args[:2]
                return node
        def selected(tree):
            return {n.name:ast.dump(Normalize().visit(n),include_attributes=False) for n in tree.body
                    if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name!='replay_package_origins'}
        self.assertEqual(selected(old),selected(new))
        old_entry=ast.parse((q/'minicpm5_sdk_mapping_worker_source_entry_v3.py').read_text())
        new_entry=ast.parse((q/'minicpm5_sdk_mapping_worker_source_entry_v4.py').read_text())
        self.assertEqual(selected(old_entry),selected(new_entry))
        self.assertNotIn('ctypes',sys.modules)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
