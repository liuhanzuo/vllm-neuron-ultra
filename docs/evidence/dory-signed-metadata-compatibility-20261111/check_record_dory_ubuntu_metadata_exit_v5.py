"""Pure-memory signed metadata replay and recorder boundary controls."""
import ast
import copy
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import record_dory_ubuntu_metadata_exit_v5 as r
import dory_ubuntu_metadata_saved_audit_v5 as audit
from check_dory_ubuntu_package_sources_v1 import Fixture, INPUTS
from check_dory_ubuntu_metadata_adapter_v6 import good_status


def fixture():
    out=Path('/mock/metadata')
    raw_files={};refs={};events=[];transferred=0;calls=0
    def save(name,raw):
        raw_files[name]=raw
        refs[name]=dict(path=str(out/name),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        return refs[name]
    f=Fixture()
    authority=dict(keyring=dict(bytes=3,sha256=hashlib.sha256(b'key').hexdigest()),
                   primary_fingerprints=audit.binding.TRUST['primary_fingerprints'],gpgv=dict(path=audit.adapter.GPGV,bytes=3,sha256='d'*64))
    trust=copy.deepcopy(audit.binding.TRUST)
    trust['keyring_sha256']=authority['keyring']['sha256']
    save('signed-metadata/ubuntu-archive-keyring.gpg',b'key')
    auth_ref=save('signed-metadata/key-authority.json',json.dumps(authority).encode())
    trust['authority_ref']=auth_ref
    def fetch(url,cap):
        nonlocal transferred
        raw=f.urls[url];transferred+=len(raw)
        ref=save('signed-metadata/'+audit.adapter.URL_FILES[url],raw)
        events.append(dict(event='ubuntu_metadata_download',url=url,saved=ref,total_transfer_bytes=transferred))
        return dict(body=raw,final_url=url)
    def verified(release,signature,current_trust):
        nonlocal calls
        calls+=1;n=calls
        save('signed-metadata/verify-'+str(n)+'.Release',release)
        save('signed-metadata/verify-'+str(n)+'.gpg',signature)
        checker=audit.Events(out,refs)
        events.append(dict(event='ubuntu_gpgv_start',invocation=n,argv=checker.argv(n)))
        status=save('signed-metadata/verify-'+str(n)+'-status.txt',good_status())
        stderr=save('signed-metadata/verify-'+str(n)+'-stderr.txt',b'diagnostic')
        events.append(dict(event='ubuntu_gpgv_exit',invocation=n,returncode=0,status=status,stderr=stderr))
        signatures=audit.adapter.parse_status(good_status(),0,current_trust,dt.datetime.now(dt.timezone.utc))
        return dict(verified=True,**signatures[0],verified_signatures=signatures,release_sha256=hashlib.sha256(release).hexdigest(),
                    signature_sha256=hashlib.sha256(signature).hexdigest(),keyring_sha256=current_trust['keyring_sha256'],
                    process_returncode=0,status_reference=status,stderr_reference=stderr)
    result=audit.discovery.discover(INPUTS,fetch,verified,trust,dt.datetime.now(dt.timezone.utc),lambda:None)
    result['adapter']=dict(metadata_files=list(refs.values()),gpgv_reference=authority['gpgv'],requests=12,signature_invocations=3,
        adapter_output_allocated_bound=4096+sum(((v['bytes']+4095)//4096)*4096 for v in refs.values()),
        actual_cryptographic_verifier='gpgv',runtime_admission_checked_here=False,caller_preconditions_required=True,process_exit_claimed=False)
    result_ref=save('signed-metadata/sources-result.json',json.dumps(result).encode())
    events.append(dict(event='ubuntu_metadata_discovery_complete',result=result_ref,status=result['status'],packages=len(result['packages']),missing=result['missing'],package_content_authenticated=False))
    raw=b''.join((json.dumps(e)+'\n').encode()for e in events)
    zipped=gzip.compress(raw,mtime=0);journal_ref=save('events.jsonl.gz',zipped)
    values={'signed-metadata/key-authority.json':authority,'signed-metadata/sources-result.json':result,
            'worker-result.json':{'events':dict(journal_ref,raw_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),events=19)}}
    return out,refs,values,events,raw_files,trust,auth_ref


def run_replay(f):
    out,refs,values,events,files,trust,authority_ref=f
    checker=audit.Events(out,refs)
    for event in events:checker.accept(event)
    with patch.object(audit.binding,'TRUST',trust),patch.object(audit.binding,'AUTHORITY',authority_ref):
        return audit.replay_saved(checker,values,refs,INPUTS,lambda name,cap:files[name],lambda:None)


class Tests(unittest.TestCase):
    def test_entire_realistic_chain_replay_and_all19events(self):
        f=fixture();result=run_replay(f)
        self.assertEqual(result['download_events'],12)
        self.assertTrue(result['entire_discovery_reproduced'])
        self.assertTrue(result['actual_signature_invocations_bound'])
        self.assertFalse(result['signature_verification_rerun'])
        self.assertFalse(result['package_content_authenticated'])

    def test_wrong_order_fixed_command_and_missing_event_denied(self):
        for kind in ('order','argv','missing','extra','count'):
            out,refs,v,events,files,trust,auth=fixture()
            if kind=='order':events[0],events[1]=events[1],events[0]
            elif kind=='argv':events[2]['argv'][0]='/wrong/gpgv'
            elif kind=='missing':events.pop(1)
            elif kind=='extra':events.append(events[-1])
            else:events[0]['total_transfer_bytes']+=1
            checker=audit.Events(out,refs)
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):
                for event in events:checker.accept(event)

    def test_failed_verifier_prefix_preserved_never_promoted(self):
        out,refs,v,events,files,trust,auth=fixture()
        events[3]['returncode']=2
        checker=audit.Events(out,refs)
        for e in events[:4]:checker.accept(e)
        self.assertTrue(checker.failed_verifier)
        self.assertEqual(checker.index,4)
        with self.assertRaises(RuntimeError):checker.accept(events[4])
        with self.assertRaises(RuntimeError):audit.replay_saved(checker,v,refs,INPUTS,lambda n,c:files[n],lambda:None)

    def test_complete_saved_result_mutations_and_extra_files_denied(self):
        for kind in ('package','status','bookkeeping','key','extra','signature'):
            f=fixture();out,refs,v,events,files,trust,auth=f
            if kind=='package':v['signed-metadata/sources-result.json']['packages'][0]['sha256']='0'*64
            elif kind=='status':events[-1]['packages']=0
            elif kind=='bookkeeping':v['signed-metadata/sources-result.json']['adapter']['adapter_output_allocated_bound']+=4096
            elif kind=='key':files['signed-metadata/ubuntu-archive-keyring.gpg']=b'bad'
            elif kind=='extra':refs['signed-metadata/evil.deb']=dict(path=str(out/'signed-metadata/evil.deb'),bytes=1,sha256='a'*64)
            else:files['signed-metadata/verify-1-status.txt']=good_status().replace(b' 0 4 0 1 10 00 ',b' 0 4 0 1 2 00 ')
            with self.subTest(kind=kind),self.assertRaises((RuntimeError,ValueError)):run_replay(f)

    def test_complete_gzip_replay_and_partial_failure_prefix(self):
        out,refs,v,events,files,trust,auth=fixture()
        ref=refs['events.jsonl.gz'];zipped=files['events.jsonl.gz']
        def open_saved(path,cap):
            return io.BytesIO(files[str(Path(path).relative_to(out))]),None
        def input_ref(path,cap):
            key=next(k for k,row in audit.binding.INPUTS.items()if row['path']==path)
            return audit.binding.INPUTS[key],INPUTS[key]
        with patch.object(r,'OUT',out),patch.object(r,'stable_open',side_effect=open_saved),patch.object(r,'unchanged'), \
             patch.object(r.c,'reference',side_effect=input_ref),patch.object(audit.binding,'TRUST',trust),patch.object(audit.binding,'AUTHORITY',auth):
            result=r.event_audit(ref,lambda:None,v,refs,True)
        self.assertTrue(result['complete']);self.assertEqual(result['events'],19)
        partial=zipped[:-8];partial_ref=dict(path=ref['path'],bytes=len(partial),sha256=hashlib.sha256(partial).hexdigest())
        with patch.object(r,'OUT',out),patch.object(r,'stable_open',return_value=(io.BytesIO(partial),None)),patch.object(r,'unchanged'):
            result=r.event_audit(partial_ref,lambda:None,v,refs,False)
        self.assertFalse(result['complete']);self.assertFalse(result['complete_workflow_claimed'])

    def test_post_terminal_observation_without_invented_live_or_success(self):
        refs={k:dict(path='/mock/'+k+'.json',sha256=str(i)*64)for i,k in enumerate(('plan','review','admitted','terminal'),1)}
        parent=dict(pid=10,start_ticks='100');child=dict(pid=11,start_ticks='101',ppid=10)
        tool=dict(kind='tool_session',session_id=999,terminal_chunk='def456',exit_code=1)
        terminal=dict(schema='dory_ubuntu_metadata_root_tool_terminal_v1',plan=refs['plan'],review=refs['review'],output_root='/mock/out',live_owner_observed_by_root=False,actual_tool=tool,owner=parent,child_owner=child)
        admitted=dict(schema='dory_ubuntu_metadata_root_admission_observation_v1',observed_after_tool_terminal=True,live_owner_observed_by_root=False,terminal=refs['terminal'],plan=refs['plan'],review=refs['review'],parent_owner=parent,child_owner=child)
        loaded=dict(plan={'output_root':'/mock/out'},review={},terminal=terminal,admitted=admitted)
        with patch.object(audit.binding,'validate'):
            self.assertEqual(audit.terminal_bindings(refs,loaded),(tool,parent,child))
            terminal['live_owner_observed_by_root']=True
            with self.assertRaises(RuntimeError):audit.terminal_bindings(refs,loaded)

    def test_original_guard_reads_resources_and_final_close_preserved(self):
        old=ast.parse((r.Q/'record_dory_system_package_metadata_exit_v1.py').read_bytes())
        new=ast.parse(Path(r.__file__).read_bytes())
        a={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)};b={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
        for name in ('decode','same_ref','stream_hash','stable_open','unchanged','saved_history','saved_descriptor','read_events','validate_bindings'):
            self.assertEqual(ast.dump(a[name]),ast.dump(b[name]),name)
        find=lambda node:next(n for n in ast.walk(node)if isinstance(n,ast.FunctionDef)and n.name=='guard')
        self.assertEqual(ast.dump(find(a['main'])),ast.dump(find(b['main'])))
        calls=lambda node:[ast.dump(n)for n in ast.walk(node)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in ('setrlimit','sched_setaffinity','nice')]
        self.assertEqual(calls(a['main']),calls(b['main']))
        text=Path(r.__file__).read_text();self.assertNotIn('Popen(',text);self.assertNotIn('urlopen(',text)
        self.assertEqual(ast.unparse(b['main']).count('os.close(fd)'),1)
        self.assertIn('28*c.GIB+32*c.MIB',text)
        self.assertNotIn("28*c.GIB+budget['output_bytes']",text)


if __name__=='__main__':unittest.main(verbosity=2)
