"""Tiny synthetic checks only. No real SITE/OFFICIAL reads or subprocesses.

Fixtures live in a disposable temporary directory. The executable check adds an
audit hook rejecting real payload reads, native ML imports and child processes.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import dory_stage_a_native_metadata_census_v2 as census


class SyntheticTree(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='dory-census-synthetic-v1-')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.base=self.root/'base';self.env=self.root/'env'
        self.site=self.env/'lib/python3.12/site-packages';self.official=self.base/'vllm'
        self.site.mkdir(parents=True);self.official.mkdir(parents=True)
        self.anchor=self.base/'installation.json'
        self.cfg=self.env/'pyvenv.cfg'
        self.guard_calls=0
        changes=dict(BASE=self.base,ENV=self.env,SITE=self.site,OFFICIAL=self.official,
                     ROOT=self.root,ANCHORS=(self.anchor,self.cfg),LIMITS=dict(census.LIMITS))
        for key,value in changes.items():
            p=patch.object(census,key,value);p.start();self.addCleanup(p.stop)
        self.dist=self.site/'torch-2.14.0+cpu.dist-info';self.dist.mkdir()
        self.write(self.dist/'METADATA',b'Name: torch\nVersion: 2.14.0+cpu\n\n')
        self.write(self.dist/'INSTALLER',b'pip\n')
        self.write(self.dist/'WHEEL',b'Wheel-Version: 1.0\n')
        self.write(self.dist/'RECORD',b'torch/__init__.py,,\n../../../unobserved.so,sha256=claim,17\n')
        self.write(self.site/'torch/__init__.py',b'')
        self.write(self.site/'torch/native.so',b'synthetic unread bytes')
        self.write(self.site/'torch/model.safetensors',b'synthetic unread weights')
        self.write(self.site/'torch/__pycache__/module.pyc',b'ignored cache')
        self.write(self.site/'ordinary.pyc/child.py',b'ordinary source name')
        self.write(self.official/'tracked.py',b'tracked synthetic source')
        self.write(self.official/'untracked.py',b'untracked synthetic source')
        self.write(self.official/'.git/objects/opaque',b'never enumerated')
        self.write(self.anchor,b'{"install": []}')
        self.write(self.cfg,b'version = 3.12.16\n')

    def write(self,path,raw):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)

    def guard(self):self.guard_calls+=1
    def budget(self):return census.Budget(self.guard)

    def command_results(self):
        values=['75fc05a53a366068f534a61ab3ee6b50485419f2\n',
                '100644 '+('a'*40)+' 0\ttracked.py\0','python\n','libiomp.so\n','tcmalloc\n','libc6\tversion\tamd64\n']
        return [dict(argv=argv,returncode=0,stdout_stderr=value,bytes=len(value))
                for argv,value in zip(census.commands(),values)]

    def collect(self):
        results=iter(self.command_results())
        with patch.object(census,'run_command',side_effect=lambda argv,budget:next(results)):
            return census.collect(guard=self.guard)


class CensusTests(SyntheticTree):
    def test_complete_names_and_opaque_git_boundary(self):
        result=self.collect();rows={x['path']:x for x in result['entries']}
        self.assertIn(str(self.site/'torch/__init__.py'),rows)
        self.assertIn(str(self.site/'torch/native.so'),rows)
        self.assertIn(str(self.site/'torch/model.safetensors'),rows)
        self.assertIn(str(self.site/'ordinary.pyc/child.py'),rows)
        self.assertIn(str(self.official/'.git'),rows)
        self.assertNotIn(str(self.official/'.git/objects'),rows)
        self.assertEqual(rows[str(self.official/'.git')]['excluded'],'git_administrative_tree')
        self.assertEqual(rows[str(self.site/'torch/__pycache__/module.pyc')]['excluded'],'cached_bytecode')
        self.assertEqual(result['summary']['empty_files'],1)
        self.assertGreater(self.guard_calls,10)
        self.assertFalse(result['globally_atomic_snapshot'])

    def test_record_is_claim_only_and_external_claim_not_followed(self):
        result=self.collect();rows={x['path']:x for x in result['entries']}
        self.assertEqual(len(rows[str(self.site/'torch/__init__.py')]['record_claims']),1)
        self.assertIn(str(self.site/'torch/native.so'),result['summary']['unclaimed_names'])
        self.assertEqual(result['summary']['record_claims_outside_site'],1)
        self.assertFalse(result['summary']['record_paths_followed'])
        self.assertFalse(result['summary']['record_is_independent_trust'])
        self.assertFalse(result['complete_authenticated_inventory'])
        self.assertIn('unknown',result['summary']['all_payload_provenance'])

    def test_torch_missing_wheel_origin_and_cpython_unknown(self):
        result=self.collect();torch=result['summary']['torch']
        self.assertEqual(torch['packages'][0]['version'],'2.14.0+cpu')
        self.assertEqual(torch['direct_url_metadata'],[dict(path=str(self.dist/'direct_url.json'),status='missing')])
        self.assertEqual(torch['installation_report_candidates'],[])
        self.assertFalse(torch['independently_authenticated_wheel'])
        self.assertFalse(result['summary']['cpython']['trusted_distribution_source_verified'])

    def test_torch_install_report_is_only_candidate(self):
        self.write(self.anchor,json.dumps({'install':[{'metadata':{'name':'torch'},'download_info':{'url':'https://example.invalid/torch.whl'}}]}).encode())
        result=self.collect()['summary']['torch']
        self.assertEqual(len(result['installation_report_candidates']),1)
        self.assertFalse(result['independently_authenticated_wheel'])

    def test_symlink_hardlink_fifo_listed_without_open(self):
        (self.site/'link.so').symlink_to(self.site/'torch/native.so')
        os.link(self.site/'torch/__init__.py',self.site/'hardlinked.py')
        os.mkfifo(self.site/'pipe')
        result=self.collect();summary=result['summary']
        self.assertEqual(summary['symlinks'],1);self.assertEqual(summary['hardlinked_regular_files'],2)
        for name in ('link.so','hardlinked.py','pipe'):
            self.assertIn(str(self.site/name),summary['abnormal_file_names'])
        row=next(x for x in result['entries'] if x['path']==str(self.site/'link.so'))
        self.assertEqual(row['link_target'],str(self.site/'torch/native.so'))

    def test_symlinked_metadata_is_explicit_unsafe_row(self):
        (self.dist/'direct_url.json').symlink_to(self.site/'torch/native.so')
        result=self.collect()
        row=next(x for x in result['metadata'] if x['path']==str(self.dist/'direct_url.json'))
        self.assertEqual(row['status'],'not_canonical_single_link_regular')

    def test_inventory_bounds_and_counts_preserved(self):
        result=self.collect()
        self.assertEqual(result['limits']['files'],100000)
        self.assertEqual(result['limits']['inventory_bytes'],8*2**30)
        self.assertEqual(result['usage']['files'],result['summary']['inventory_files'])
        self.assertEqual(result['usage']['inventory_bytes'],result['summary']['inventory_bytes'])
        self.assertEqual(result['limits']['metadata_total_bytes'],32*2**20)

    def test_untracked_large_build_name_and_raw_bytes_retained_without_inventory_charge(self):
        target=self.official/'build/duplicate.so'
        self.write(target,b'synthetic placeholder')
        original=Path.lstat
        def large_stat(path,*args,**kwargs):
            info=original(path,*args,**kwargs)
            if path==target:
                fields=census.identity(info);fields['st_size']=9*2**30
                return SimpleNamespace(**fields)
            return info
        with patch.object(Path,'lstat',large_stat):result=self.collect()
        row=next(x for x in result['entries'] if x['path']==str(target))
        self.assertFalse(row['inventory_eligible'])
        self.assertEqual(row['identity']['st_size'],9*2**30)
        self.assertIn(str(target),result['summary']['unclaimed_names'])
        self.assertGreater(result['summary']['by_scope']['official']['observed_regular_bytes'],8*2**30)
        self.assertEqual(result['summary']['by_scope']['official']['eligible_inventory_files'],1)
        self.assertEqual(result['summary']['by_scope']['official']['eligible_inventory_bytes'],len(b'tracked synthetic source'))
        self.assertLess(result['summary']['inventory_bytes'],2**20)

    def test_untracked_file_count_does_not_consume_eligible_file_budget(self):
        before=self.collect();census.LIMITS['files']=before['usage']['files']
        self.write(self.official/'build/one.py',b'1')
        self.write(self.official/'build/two.so',b'2')
        result=self.collect()
        self.assertEqual(result['usage']['files'],before['usage']['files'])
        self.assertEqual(result['summary']['observed_regular_files'],before['summary']['observed_regular_files']+2)

    def test_tracked_file_over_limit_still_rejected(self):
        target=self.official/'tracked.py';original=Path.lstat
        def large_stat(path,*args,**kwargs):
            info=original(path,*args,**kwargs)
            if path==target:
                fields=census.identity(info);fields['st_size']=3*2**30
                return SimpleNamespace(**fields)
            return info
        with patch.object(Path,'lstat',large_stat):
            with self.assertRaisesRegex(RuntimeError,'single inventory'):self.collect()

    def test_unavailable_index_keeps_all_official_names_without_claiming_eligibility(self):
        values=self.command_results();values[1].update(returncode=1,stdout_stderr='unavailable')
        results=iter(values)
        with patch.object(census,'run_command',side_effect=lambda argv,budget:next(results)):
            result=census.collect(guard=self.guard)
        self.assertFalse(result['summary']['official_index_available'])
        self.assertEqual(result['summary']['by_scope']['official']['eligible_inventory_files'],0)
        self.assertIn(str(self.official/'tracked.py'),result['summary']['unclaimed_names'])
        self.assertFalse(result['inventory_eligibility_is_authentication'])

    def test_file_bound_rejects(self):
        census.LIMITS['files']=1
        with self.assertRaisesRegex(RuntimeError,'budget: files'):self.collect()

    def test_inventory_byte_bound_rejects(self):
        census.LIMITS['inventory_bytes']=1
        with self.assertRaisesRegex(RuntimeError,'budget: inventory_bytes'):self.collect()

    def test_single_inventory_file_bound_rejects(self):
        census.LIMITS['single_inventory_file']=1
        with self.assertRaisesRegex(RuntimeError,'single inventory'):self.collect()

    def test_directory_name_bounds_reject(self):
        for key in ('directory_entries','directory_name_bytes'):
            with self.subTest(key=key),patch.dict(census.LIMITS,{key:0}):
                with self.assertRaisesRegex(RuntimeError,'budget: '+key):self.collect()

    def test_guard_interrupt_propagates_without_commands(self):
        with patch.object(census,'run_command') as command:
            with self.assertRaisesRegex(RuntimeError,'owner lost'):
                census.collect(guard=lambda:(_ for _ in ()).throw(RuntimeError('owner lost')))
        command.assert_not_called()

    def test_explicit_guard_required(self):
        with self.assertRaisesRegex(RuntimeError,'Explicit caller'):census.collect(guard=None)

    def test_wall_bound_rejects(self):
        census.LIMITS['wall_seconds']=-1
        with self.assertRaisesRegex(RuntimeError,'wall bound'):self.collect()

    def test_symlink_root_rejected(self):
        link=self.root/'linked-site';link.symlink_to(self.site,target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError,'Canonical census root'):
            census.scan(link,'site',self.budget())

    def test_directory_mutation_rejected(self):
        original=os.scandir
        class MutatingScan:
            def __enter__(inner):
                inner.scan=original(self.site);return inner.scan
            def __exit__(inner,*args):
                inner.scan.close();self.write(self.site/'new.py',b'changed directory')
        with patch.object(census.os,'scandir',return_value=MutatingScan()):
            with self.assertRaisesRegex(RuntimeError,'Directory changed'):
                census.scan(self.site,'site',self.budget())

    def test_no_runtime_or_speedup_claims(self):
        result=self.collect()
        for key in ('lock_acquired_or_released','output_written','runtime_admitted_here',
                    'actual_process_exit_observed','native_ABI_qualified','full_model_qualified','new_speedup_claim'):
            self.assertFalse(result[key])
        self.assertEqual(result['native_or_model_payload_bytes_read'],0)
        self.assertEqual(len(result['skipped_commands']),1)
        self.assertIn('status',result['skipped_commands'][0]['argv'])

    def test_failed_git_is_unknown_not_clean(self):
        result=self.collect();results=self.command_results();results[1]['returncode']=1
        summary=census.summarize(result['entries'],result['metadata'],results)
        self.assertIn('git index unavailable',summary['issues'])
        self.assertIn(str(self.official/'tracked.py'),summary['unclaimed_names'])
        self.assertFalse(summary['working_tree_content_verified'])

    def test_missing_git_name_reported(self):
        result=self.collect();results=self.command_results()
        results[1]['stdout_stderr']+='100644 '+('b'*40)+' 0\tmissing.py\0'
        summary=census.summarize(result['entries'],result['metadata'],results)
        self.assertEqual(summary['missing_git_names'],[str(self.official/'missing.py')])

    def test_git_escape_and_conflict_rejected(self):
        row=dict(returncode=0,stdout_stderr='100644 '+('a'*40)+' 0\t../escape\0')
        with self.assertRaisesRegex(RuntimeError,'Git path'):census.git_index(row)
        row['stdout_stderr']='100644 '+('a'*40)+' 0\tname\0'
        row['stdout_stderr']*=2
        with self.assertRaisesRegex(RuntimeError,'duplicate/conflicted'):census.git_index(row)


class MetadataTests(SyntheticTree):
    def test_exact_allowed_metadata_hash_and_size(self):
        result=census.metadata_file(self.dist/'INSTALLER',self.budget())
        self.assertEqual(result['text'],'pip\n');self.assertEqual(result['bytes'],4)
        self.assertEqual(result['sha256'],hashlib.sha256(b'pip\n').hexdigest())
        self.assertFalse(result['independently_authenticated'])

    def test_missing_metadata(self):
        self.assertEqual(census.metadata_file(self.dist/'direct_url.json',self.budget())['status'],'missing')

    def test_payload_allowlist_rejects_before_open(self):
        with patch.object(census.os,'open') as opener:
            with self.assertRaisesRegex(RuntimeError,'allowlist'):
                census.metadata_file(self.site/'torch/native.so',self.budget())
        opener.assert_not_called()

    def test_metadata_symlink_hardlink_fifo_rejected_before_open(self):
        target=self.dist/'direct_url.json'
        for kind in ('symlink','hardlink','fifo'):
            with self.subTest(kind=kind):
                if kind=='symlink':target.symlink_to(self.site/'torch/native.so')
                elif kind=='hardlink':os.link(self.dist/'INSTALLER',target)
                else:os.mkfifo(target)
                with patch.object(census.os,'open') as opener:
                    result=census.metadata_file(target,self.budget())
                self.assertEqual(result['status'],'not_canonical_single_link_regular')
                opener.assert_not_called();target.unlink()

    def test_metadata_file_limit_before_open(self):
        census.LIMITS['metadata_file_bytes']=1
        with patch.object(census.os,'open') as opener:
            with self.assertRaisesRegex(RuntimeError,'Single metadata file cap'):
                census.metadata_file(self.dist/'INSTALLER',self.budget())
        opener.assert_not_called()

    def test_metadata_total_limit_before_open(self):
        census.LIMITS['metadata_total_bytes']=3
        with patch.object(census.os,'open') as opener:
            with self.assertRaisesRegex(RuntimeError,'metadata_total_bytes'):
                census.metadata_file(self.dist/'INSTALLER',self.budget())
        opener.assert_not_called()

    def test_metadata_changed_after_read_rejected(self):
        original=os.read;changed=False
        def read(fd,n):
            nonlocal changed
            result=original(fd,n)
            if not changed:
                changed=True;self.write(self.dist/'INSTALLER',b'new\n')
            return result
        with patch.object(census.os,'read',side_effect=read):
            with self.assertRaisesRegex(RuntimeError,'Stable complete metadata'):
                census.metadata_file(self.dist/'INSTALLER',self.budget())

    def test_empty_metadata_retained_and_hashed(self):
        self.write(self.dist/'INSTALLER',b'')
        row=census.metadata_file(self.dist/'INSTALLER',self.budget())
        self.assertEqual(row['bytes'],0);self.assertEqual(row['sha256'],hashlib.sha256(b'').hexdigest())


class FakePipe:
    def fileno(self):return 999
    def close(self):self.closed=True


class FakeProcess:
    def __init__(self):self.stdout=FakePipe();self.returncode=None;self.killed=False
    def wait(self,timeout=None):
        if self.returncode is None:self.returncode=0
        return self.returncode
    def poll(self):return self.returncode
    def kill(self):self.killed=True;self.returncode=-9


class FakeSelector:
    def register(self,fileobj,event):self.fileobj=fileobj;self.active=True
    def unregister(self,fileobj):self.active=False
    def get_map(self):return {1:1} if self.active else {}
    def select(self,timeout):return [(SimpleNamespace(fileobj=self.fileobj),1)]
    def close(self):pass


class CommandTests(unittest.TestCase):
    def run_fake(self,raw=b'head\n',guard=lambda:None):
        self.process=FakeProcess()
        with patch.object(census.subprocess,'Popen',return_value=self.process) as popen, \
             patch.object(census.selectors,'DefaultSelector',FakeSelector), \
             patch.object(census.os,'read',side_effect=[raw,b'']):
            result=census.run_command(census.commands()[0],census.Budget(guard))
        self.call=popen.call_args;return result

    def test_fixed_read_only_commands_no_status_or_shell(self):
        values=census.commands();self.assertEqual(len(values),6)
        self.assertEqual({x[0] for x in values},{'/usr/bin/git','/usr/bin/readlink','/usr/bin/dpkg-query'})
        self.assertFalse(any('status' in x for x in values))
        self.run_fake();self.assertNotIn('shell',self.call.kwargs)
        self.assertEqual(self.call.kwargs['env']['GIT_OPTIONAL_LOCKS'],'0')
        self.assertTrue(self.process.stdout.closed)

    def test_non_whitelisted_command_rejected_without_spawn(self):
        with patch.object(census.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError,'Only fixed'):
                census.run_command(['/bin/true'],census.Budget(lambda:None))
        popen.assert_not_called()

    def test_full_output_and_invalid_utf8_name_bytes_preserved(self):
        raw=b'100644 hash 0\tname-\xff\0'
        result=self.run_fake(raw)
        self.assertEqual(result['stdout_stderr'].encode('utf8','surrogateescape'),raw)
        self.assertEqual(result['returncode'],0)

    def test_command_byte_cap_kills_and_reaps(self):
        with patch.dict(census.LIMITS,command_bytes=2):
            with self.assertRaisesRegex(RuntimeError,'output cap'):self.run_fake(b'123')
        self.assertTrue(self.process.killed);self.assertIsNotNone(self.process.returncode)

    def test_total_command_byte_cap_kills_and_reaps(self):
        with patch.dict(census.LIMITS,command_total_bytes=2):
            with self.assertRaisesRegex(RuntimeError,'command_total_bytes'):self.run_fake(b'123')
        self.assertTrue(self.process.killed)

    def test_command_timeout_kills_and_reaps(self):
        with patch.dict(census.LIMITS,command_seconds=-1):
            with self.assertRaisesRegex(RuntimeError,'command timeout'):self.run_fake()
        self.assertTrue(self.process.killed)

    def test_guard_interrupt_kills_and_reaps(self):
        calls=0
        def guard():
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('guard interrupted')
        with self.assertRaisesRegex(RuntimeError,'guard interrupted'):self.run_fake(guard=guard)
        self.assertTrue(self.process.killed)


AUDIT={'blocked':[],'fixture_reads':0}


def synthetic_audit(event,args):
    if event=='subprocess.Popen':
        AUDIT['blocked'].append('subprocess');raise RuntimeError('No actual subprocess in synthetic checks')
    if event=='import' and args[0].split('.')[0] in {'torch','vllm','numpy','transformers','tokenizers','neuronx_distributed'}:
        AUDIT['blocked'].append('native_import');raise RuntimeError('No native ML import in synthetic checks')
    if event not in {'open','os.scandir','os.listdir'} or isinstance(args[0],int):return
    path=os.fsdecode(args[0])
    if path.startswith('/mnt/nvme/'):
        AUDIT['blocked'].append(path);raise RuntimeError('No real environment access in synthetic checks')
    if event=='open':
        flags=args[2]
        writing=bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if not writing and '/dory-census-synthetic-v1-' in path:
            if path.endswith(('.so','.safetensors','.pyc')):
                AUDIT['blocked'].append(path);raise RuntimeError('No synthetic payload read')
            AUDIT['fixture_reads']+=1


if __name__=='__main__':
    sys.addaudithook(synthetic_audit)
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),audit=AUDIT,
                          real_census_executed=False,native_model_imported=False),sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
