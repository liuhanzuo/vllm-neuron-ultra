"""Stdlib and synthetic/mocked checks only; no native/archive/network/lifecycle IO."""
import ast
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import minicpm5_sdk310_deb_input_binding_v1 as b
import minicpm5_sdk310_deb_members_adapter_v1 as a
import minicpm5_sdk310_deb_events_v1 as e
import dory_deb_input_binding_v3 as old
Q = Path(__file__).resolve().parent


def inputs():
    return json.loads((Q/'minicpm5-sdk310-archive-member-next-inputs-20261112-v1.json').read_text())


def text_ref(path, raw):
    return dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def ancestry_fixture():
    n = inputs(); blobs = {}; releases = []
    trust = dict(keyring_sha256='a'*64, primary_fingerprints=['B'*40], authority_ref={k:n['trust_authority'][k] for k in ('path','sha256')})
    authority = dict(schema='dory_ubuntu_archive_key_authority_v1',independent_authority_verified=True,keyring={'sha256':'a'*64},primary_fingerprints=['B'*40])
    for chain in n['signed_index_reuse']:
        path = chain['current_uncompressed_index']['path']; raw = b''; slices = {}
        for row in n['packages']:
            selected = [s for s in row['sources'] if s['index']['path'] == path]
            if not selected: continue
            fields = dict(package=row['package'],version=row['version'],architecture=row['architecture'],filename=row['filename'],size=str(row['bytes']),sha256=row['sha256'])
            block = '\n'.join(k+': '+v for k,v in fields.items()).encode()
            slices[row['package']] = dict(byte_offset=len(raw),bytes=len(block),sha256=hashlib.sha256(block).hexdigest(),fields=fields)
            raw += block+b'\n\n'
        idx = text_ref(path,raw); chain['current_uncompressed_index'] = idx; blobs[path] = raw
        rr = chain['release']; release_raw = chain['suite'].encode(); chain['release'] = text_ref(rr['path'],release_raw); blobs[rr['path']] = release_raw
        status = text_ref(chain['status_text']['path'],b'synthetic status only'); chain['status_text'] = status; blobs[status['path']] = b'synthetic status only'
        proof = dict(verified=True,process_returncode=0,primary_fingerprint='B'*40,keyring_sha256='a'*64,status_reference=status,release_sha256=chain['release']['sha256'])
        chain['saved_signature_proof'] = proof
        release_url = chain['index_url'].rsplit('/main/',1)[0]+'/Release'
        releases.append(dict(release_url=release_url,release_sha256=chain['release']['sha256'],signature_proof=copy.deepcopy(proof),indexes=[dict(url=chain['index_url'],uncompressed={k:idx[k] for k in ('bytes','sha256')})]))
        for row in n['packages']:
            for source in row['sources']:
                if source['index']['path'] != path: continue
                source['index'] = idx; source['release_sha256'] = chain['release']['sha256']; source['matched_stanza'] = dict(index=idx,**slices[row['package']])
    signed = dict(trust_anchor=trust,releases=releases)
    def read(ref):
        raw = blobs[ref['path']]
        if text_ref(ref['path'],raw) != ref: raise RuntimeError('Synthetic text hash mismatch')
        return raw
    def release_indexes(raw,suite,now):
        chain = next(x for x in n['signed_index_reuse'] if x['suite'] == suite)
        return {'main/binary-amd64/Packages':{k:chain['current_uncompressed_index'][k] for k in ('bytes','sha256')}},{}
    return n,signed,authority,read,release_indexes


def run_ancestry(fixture):
    n,signed,authority,read,parser = fixture
    with patch.object(b.metadata,'release_indexes',side_effect=parser):
        return b.verify_ancestry(n,signed,authority,read,Mock(),dt.datetime(2026,11,12,tzinfo=dt.timezone.utc))


class AdapterHarness:
    def __init__(self):
        self.n = inputs(); self.saved = []; self.events = []; self.inspections = {}; self.targets = {}
        for row in self.n['packages']:
            archive = dict(path='/mock/packages/'+row['package']+'.deb',bytes=row['bytes'],sha256=row['sha256'])
            members = {}
            for target in row['targets']:
                host = target['observed_path']; self.targets[host] = target
                members[host] = dict(size=target['current_lstat']['st_size'],sha256=hashlib.sha256(host.encode()).hexdigest(),terminal=target['archive_member'])
            self.inspections[row['package']] = dict(archive=archive,archive_identity={'stable':'archive-'+row['package']},tar=dict(stream_bytes=1024,stream_sha256=hashlib.sha256(row['package'].encode()).hexdigest(),targets=members),decompressor=dict(argv=['/usr/bin/dpkg-deb','--fsys-tarfile',archive['path']],returncode=0,stdout_bytes=1024,stderr='',stderr_sha256=hashlib.sha256(b'').hexdigest()))
    def download(self,row,path,guard,emit):
        receipt=self.inspections[row['package']]
        emit(dict(event='archive_download_start',package=row['package'],url=row['url'],expected=dict(bytes=row['bytes'],sha256=row['sha256'])))
        emit(dict(event='archive_download_complete',package=row['package'],saved=receipt['archive'],identity=receipt['archive_identity']))
    def inspect(self,row,path,tool,guard,emit):
        receipt=copy.deepcopy(self.inspections[row['package']]);emit(dict(event='dpkg_deb_start',argv=receipt['decompressor']['argv']));emit(dict(event='dpkg_deb_complete',package=row['package'],receipt=receipt['decompressor']));return receipt
    def file_ref(self,path,guard,expected,**kwargs):
        host=str(path)
        if host.endswith('.deb'):return dict(path=host,bytes=expected['bytes'],sha256=expected['sha256']),{}
        target=self.targets[host]
        return dict(path=host,**expected),{k:target['current_lstat']['st_'+k] for k in ('dev','ino','mode','nlink','size','mtime_ns','ctime_ns')}
    def save(self,path,value,cap):
        self.saved.append(copy.deepcopy(value));return dict(path=str(path),bytes=1,sha256='f'*64)
    def run(self,mode='download',prior=None):
        self.events=[]
        with patch.object(a.binding,'authenticated',return_value=(self.n,{'tool':{}})),patch.object(Path,'mkdir'),patch.object(a.io,'download',side_effect=self.download),patch.object(a.io,'inspect',side_effect=self.inspect),patch.object(a.io,'file_ref',side_effect=self.file_ref),patch.object(a.c,'save',side_effect=self.save),patch.object(a.c,'load',return_value=prior):
            return a.run(Mock(),self.events.append,b.TRUST,Path('/mock/packages'),mode,{'path':'prior'} if mode=='replay' else None)


class ManifestTests(unittest.TestCase):
    def test_exact_input_and_scope(self):
        self.assertEqual(b.validate_manifest(inputs())['counts']['required_packages'],3)
    def test_role_and_actual_mapping_are_not_inferred(self):
        for field,value in [('necessity','required'),('actual_loaded',True),('version','3.12')]:
            n=inputs();n['packages'][3][field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):b.validate_manifest(n)
    def test_duplicate_and_reordered_packages_rejected(self):
        for change in ('duplicate','order'):
            n=inputs()
            if change=='duplicate':n['packages'][1]=copy.deepcopy(n['packages'][0])
            else:n['packages'][0],n['packages'][1]=n['packages'][1],n['packages'][0]
            with self.subTest(change=change),self.assertRaises(RuntimeError):b.validate_manifest(n)
    def test_target_old_hash_and_alias_promotions_rejected(self):
        for change in ('path','old','alias'):
            n=inputs();row=n['packages'][0]
            if change=='path':row['targets'][0]['observed_path']='/usr/bin/python3.12'
            elif change=='old':row['targets'][0]['original_observed_reference']={'sha256':'a'*64}
            else:row['saved_package_names'].append('/unverified-alias')
            with self.subTest(change=change),self.assertRaises(RuntimeError):b.validate_manifest(n)
    def test_old_runner_binding_rejects_new_inputs(self):
        n=inputs();plan=dict(work_kind=b.WORK_KIND,archive_inputs=b.TRUST,archive_mode='download',replay_from=None,external_authorities=None,source_sha256=dict(b.SOURCE_PINS))
        with patch.object(old.runtime,'validate'):
            with self.assertRaises(RuntimeError):old.validate(plan,copy.deepcopy(plan),{})
        with patch.object(old.c,'load',return_value=n):
            with self.assertRaises(RuntimeError):old.authenticated(Mock())
    def test_new_binding_requires_new_source_closure_and_scientific_envelope(self):
        plan=dict(work_kind=b.WORK_KIND,archive_inputs=b.TRUST,archive_mode='download',replay_from=None,external_authorities=None,source_sha256=dict(b.SOURCE_PINS),mini_scientific_constraints=b.CONSTRAINTS,output_root='/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-test')
        plan['source_sha256'].update({str(Q/name):'a'*64 for name in b.NEW_NAMES})
        with patch.object(b.runtime,'validate'):
            b.validate(plan,copy.deepcopy(plan),{})
            for key,value in [('archive_mode','replay'),('mini_scientific_constraints',{}),('source_sha256',{}),('output_root','/mnt/nvme/other')]:
                bad=copy.deepcopy(plan);bad[key]=value
                with self.subTest(key=key),self.assertRaises(RuntimeError):b.validate(bad,copy.deepcopy(plan),{})


class AncestryTests(unittest.TestCase):
    def test_fourteen_complete_source_associations(self):self.assertEqual(run_ancestry(ancestry_fixture()),14)
    def test_cross_package_stanza_rejected(self):
        fixture=ancestry_fixture();n=fixture[0];n['packages'][0]['sources'][0]['matched_stanza']=copy.deepcopy(n['packages'][1]['sources'][0]['matched_stanza'])
        with self.assertRaises(RuntimeError):run_ancestry(fixture)
    def test_missing_and_duplicate_source_rejected(self):
        for duplicate in (False,True):
            fixture=ancestry_fixture();sources=fixture[0]['packages'][0]['sources']
            if duplicate:sources.append(copy.deepcopy(sources[0]))
            else:sources.pop()
            with self.subTest(duplicate=duplicate),self.assertRaises(RuntimeError):run_ancestry(fixture)
    def test_wrong_release_proof_and_url_rejected(self):
        for kind in ('proof','url','index'):
            fixture=ancestry_fixture();n,signed=fixture[:2]
            if kind=='proof':signed['releases'][0]['signature_proof']['process_returncode']=1
            elif kind=='url':n['packages'][0]['sources'][0]['url']='https://archive.ubuntu.com/ubuntu/pool/wrong.deb'
            else:signed['releases'][0]['indexes'][0]['uncompressed']['sha256']='0'*64
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):run_ancestry(fixture)
    def test_full_stanza_slice_cannot_be_replaced_by_subset(self):
        fixture=ancestry_fixture();fixture[0]['packages'][0]['sources'][0]['matched_stanza']['byte_offset']+=1
        with self.assertRaises(RuntimeError):run_ancestry(fixture)
    def test_fixed_text_reader_rejects_wrong_hash_and_unlisted_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metadata.txt';raw=b'plain text metadata\n';path.write_bytes(raw);ref=text_ref(str(path),raw)
            self.assertEqual(b.read_text(ref,Mock(),{str(path):ref}),raw)
            bad=dict(ref,sha256='0'*64)
            with self.assertRaises(RuntimeError):b.read_text(bad,Mock(),{str(path):bad})
            with self.assertRaises(RuntimeError):b.read_text(ref,Mock(),{})


class RuntimeMockTests(unittest.TestCase):
    def test_eight_packages_and_independent_replay(self):
        h=AdapterHarness();out=h.run();producer=copy.deepcopy(h.saved[-1]);self.assertEqual(len(h.events),41)
        e.validate_success(h.events,h.n['packages'],'download',producer,out['result'])
        out=h.run('replay',producer);self.assertEqual(len(h.events),33)
        e.validate_success(h.events,h.n['packages'],'replay',h.saved[-1],out['result'])
        self.assertEqual(h.saved[-1]['original_match_count'],0)
        self.assertTrue(all(p['actual_loaded'] is None for p in h.saved[-1]['packages']))
    def test_download_failure_cannot_save_success(self):
        h=AdapterHarness();h.download=Mock(side_effect=OSError('synthetic download failure'))
        with self.assertRaises(OSError):h.run()
        self.assertEqual(h.saved,[]);self.assertFalse(any(x['event']=='minicpm5_eight_packages_complete' for x in h.events))
    def test_inspection_failure_cannot_save_success(self):
        h=AdapterHarness();h.inspect=Mock(side_effect=RuntimeError('synthetic decoder failure'))
        with self.assertRaises(RuntimeError):h.run()
        self.assertEqual(h.saved,[]);self.assertEqual(len(h.events),2)
    def test_host_identity_change_rejected(self):
        h=AdapterHarness();original=h.file_ref
        def replaced(*args,**kwargs):
            ref,identity=original(*args,**kwargs);identity=dict(identity,ino=-1);return ref,identity
        h.file_ref=replaced
        with self.assertRaises(RuntimeError):h.run()
        self.assertEqual(h.saved,[])
    def test_replay_modified_member_and_mapping_claim_rejected(self):
        h=AdapterHarness();h.run();prior=copy.deepcopy(h.saved[-1])
        for kind in ('member','mapped','old'):
            bad=copy.deepcopy(prior)
            if kind=='member':bad['packages'][0]['comparisons'][0]['member']['sha256']='0'*64
            elif kind=='mapped':bad['packages'][0]['actual_loaded']=True
            else:bad['original_match_count']=11
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):h.run('replay',bad)
    def test_every_missing_or_duplicate_success_event_rejected(self):
        h=AdapterHarness();out=h.run();result=h.saved[-1]
        for i in range(41):
            for duplicate in (False,True):
                events=copy.deepcopy(h.events)
                if duplicate:events.insert(i,copy.deepcopy(events[i]))
                else:events.pop(i)
                with self.subTest(i=i,duplicate=duplicate),self.assertRaises(RuntimeError):e.validate_success(events,h.n['packages'],'download',result,out['result'])
    def test_adjacent_swaps_and_cross_package_event_rejected(self):
        h=AdapterHarness();out=h.run();result=h.saved[-1]
        for i in range(40):
            events=copy.deepcopy(h.events);events[i],events[i+1]=events[i+1],events[i]
            with self.subTest(i=i),self.assertRaises(RuntimeError):e.validate_success(events,h.n['packages'],'download',result,out['result'])
        events=copy.deepcopy(h.events);events[0]['package']=h.n['packages'][1]['package']
        with self.assertRaises(RuntimeError):e.validate_success(events,h.n['packages'],'download',result,out['result'])
    def test_decoder_failure_and_changed_final_result_not_success(self):
        h=AdapterHarness();out=h.run()
        for kind in ('failed','result'):
            events=copy.deepcopy(h.events);result=copy.deepcopy(h.saved[-1])
            if kind=='failed':events[3]['receipt']['returncode']=1;result['packages'][0]['inspection']['decompressor']['returncode']=1
            else:events[-1]['result']['sha256']='0'*64
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):e.validate_success(events,h.n['packages'],'download',result,out['result'])
    def test_same_count_event_duplication_is_rejected(self):
        h=AdapterHarness();out=h.run();events=copy.deepcopy(h.events);events[6]=copy.deepcopy(events[1])
        with self.assertRaises(RuntimeError):e.validate_success(events,h.n['packages'],'download',h.saved[-1],out['result'])


class SourcePreservationTests(unittest.TestCase):
    def test_parent_worker_ast_only_dispatch_changes(self):
        for oldname,newname,mapping in [('run_dory_deb_members_root_v3.py','run_minicpm5_sdk310_deb_members_root_v1.py',{'dory_deb_input_binding_v3':'minicpm5_sdk310_deb_input_binding_v1','dory_deb_members_worker_v3.py':'minicpm5_sdk310_deb_members_worker_v1.py'}),('dory_deb_members_worker_v3.py','minicpm5_sdk310_deb_members_worker_v1.py',{'dory_deb_input_binding_v3':'minicpm5_sdk310_deb_input_binding_v1','dory_deb_members_adapter_v3':'minicpm5_sdk310_deb_members_adapter_v1'})]:
            text=(Q/oldname).read_text()
            for left,right in mapping.items():text=text.replace(left,right)
            self.assertEqual(ast.dump(ast.parse(text)),ast.dump(ast.parse((Q/newname).read_text())))
    def test_reused_implementation_source_hashes_unchanged(self):
        for path,digest in b.SOURCE_PINS.items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),digest,path)
    def test_exact_finite_recorder_protocol_present(self):
        text=(Q/'record_minicpm5_sdk310_deb_stage_exit_v1.py').read_text();tree=ast.parse(text)
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertIn('event_protocol.validate_success',calls);self.assertIn('binding.validate_result',calls)
        self.assertIn('len(events)<=event_protocol.MAX_EVENTS',text);self.assertIn("else:c.require('failure.json'in values",text)
        self.assertEqual(e.MAX_EVENTS,41)


if __name__=='__main__':unittest.main(verbosity=2)
