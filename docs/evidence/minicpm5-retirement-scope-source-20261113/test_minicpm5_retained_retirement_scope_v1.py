"""In-memory foreground lifecycle tests. No process, lock or proc access."""
import ast
import copy
import errno
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import minicpm5_retained_retirement_scope_v1 as scope
import minicpm5_retained_retirement_contract_v2 as c
import minicpm5_retained_retirement_holders_v2 as holders
import test_minicpm5_retained_retirement_caller_v2 as lifecycle


class Fixture:
    def __init__(self):
        self.events=[];self.live=set();self.nonce='a'*64
        self.permit=dict(path='/synthetic/permit.json',sha256='b'*64)
        self.argv=scope.command(self.permit,self.nonce)
        def owner(pid,ppid,argv):return dict(pid=pid,ppid=ppid,start_ticks=str(pid*100),argv=argv,cgroup='0::/synthetic')
        self.owner=owner(100,99,['synthetic-retirement-caller'])
        self.chain=[owner(103,102,self.argv[2:]),owner(102,101,self.argv),owner(101,100,self.argv),self.owner]
        self.ack=dict(schema=scope.ACK_SCHEMA,permit=self.permit,nonce=self.nonce,chain=self.chain,helper_spawning_forbidden=True)
        self.stdout=(json.dumps(self.ack)+'\n'+json.dumps(dict(complete=True))+'\n').encode()
        self.failures=[];self.waits=0;self.exit_code=0
        fixture=self
        class Lock:
            owner=fixture.owner
            def check(self):fixture.events.append('lock-check')
            def release(self,drained):
                assert drained is True and not fixture.live
                fixture.events.append('lock-release')
        class Process:
            pid=101;returncode=None
            def communicate(self,timeout):
                fixture.events.append(('wait',timeout));fixture.waits+=1
                if fixture.failures:raise fixture.failures.pop(0)
                self.returncode=fixture.exit_code;return fixture.stdout,b''
        self.lock=Lock();self.process=Process()
        def gone(owner):
            self.events.append(('gone',owner['pid']))
            if owner['pid']in self.live:raise ValueError('Captured identity remains live')
        self.io=SimpleNamespace(process_api=lambda:SimpleNamespace(gone=gone))
        self.scopes=scope.ForegroundScopes(self.io,self.lock)
    def popen(self,argv,**kwargs):
        assert argv==self.argv
        assert kwargs==dict(stdin=scope.subprocess.DEVNULL,stdout=scope.subprocess.PIPE,
            stderr=scope.subprocess.PIPE,close_fds=True,start_new_session=True)
        self.events.append('spawn-registered')
        assert self.scopes.active is not None
        return self.process
    def active(self):
        self.process.returncode=self.exit_code
        self.scopes.active=dict(permit=self.permit,nonce=self.nonce,process=self.process,
            output=(self.stdout,b''),spawn_failed=False)


class ForegroundLifetime(unittest.TestCase):
    def test_success_captures_and_independently_drains_helper_monitor_and_sudo(self):
        f=Fixture()
        with patch.object(scope.secrets,'token_hex',return_value=f.nonce),patch.object(scope.subprocess,'Popen',side_effect=f.popen):
            value,proof=f.scopes.invoke(f.permit)
        self.assertTrue(value['complete']);self.assertTrue(proof['physically_drained'])
        self.assertEqual(proof['last_scope']['owned_processes'],f.chain[:-1])
        self.assertEqual([x for x in f.events if isinstance(x,tuple)and x[0]=='gone'],[('gone',103),('gone',102),('gone',101)])
        self.assertEqual(proof['completed_scopes'],1);c.validate_scope_drain(proof,f.owner)

    def test_timeout_retains_lock_through_live_sudo_and_live_helper_then_raises_original(self):
        f=Fixture();original=scope.subprocess.TimeoutExpired(f.argv,90)
        f.failures=[original,scope.subprocess.TimeoutExpired(f.argv,1)]
        f.live={102,103};retry=[0]
        def retry_wait(_):
            self.assertNotIn('lock-release',f.events);retry[0]+=1
            # First retry leaves a live helper after the direct sudo has exited.
            if retry[0]>=2:f.live.clear()
        with patch.object(scope.secrets,'token_hex',return_value=f.nonce),patch.object(scope.subprocess,'Popen',side_effect=f.popen),\
                patch.object(scope.time,'sleep',side_effect=retry_wait),patch('builtins.print'):
            with self.assertRaises(scope.subprocess.TimeoutExpired)as caught:f.scopes.invoke(f.permit)
        self.assertIs(caught.exception,original);self.assertGreaterEqual(retry[0],2)
        self.assertEqual(f.process.returncode,0)
        f.lock.release(f.scopes.drain()['physically_drained']);self.assertEqual(f.events[-1],'lock-release')

    def test_keyboard_interrupt_in_wait_still_drains_before_propagation(self):
        f=Fixture();error=KeyboardInterrupt('synthetic interrupt');f.failures=[error]
        with patch.object(scope.secrets,'token_hex',return_value=f.nonce),patch.object(scope.subprocess,'Popen',side_effect=f.popen):
            with self.assertRaises(KeyboardInterrupt)as caught:f.scopes.invoke(f.permit)
        self.assertIs(caught.exception,error);self.assertIsNone(f.scopes.active)
        self.assertIn(('gone',103),f.events)

    def test_cleanup_notice_retry_exceptions_cannot_unwind_ownership(self):
        f=Fixture();f.active();f.scopes.active['output']=None
        f.failures=[KeyboardInterrupt('cleanup interrupted'),OSError('pipe temporarily unavailable')]
        retries=[0]
        def retry(_):
            retries[0]+=1;self.assertNotIn('lock-release',f.events)
            raise KeyboardInterrupt('retry interrupted')
        with patch.object(scope.time,'sleep',side_effect=retry),patch('builtins.print',side_effect=OSError('notice sink unavailable')):
            proof=f.scopes.drain()
        self.assertTrue(proof['physically_drained']);self.assertEqual(retries[0],2)

    def test_complete_ack_but_live_monitor_cannot_produce_drain_proof(self):
        f=Fixture();f.active();f.live={102}
        with self.assertRaisesRegex(ValueError,'remains live'):f.scopes._verify()
        self.assertIsNotNone(f.scopes.active);self.assertEqual(f.scopes.count,0)
        f.live.clear();self.assertTrue(f.scopes._verify()['physically_drained'])

    def test_proven_preexec_failure_preserves_failure_and_has_no_owned_child(self):
        f=Fixture();error=PermissionError(errno.EACCES,'synthetic executable denied','sudo')
        with patch.object(scope.secrets,'token_hex',return_value=f.nonce),patch.object(scope.subprocess,'Popen',side_effect=error):
            with self.assertRaises(PermissionError)as caught:f.scopes.invoke(f.permit)
        self.assertIs(caught.exception,error);proof=f.scopes.summary()
        self.assertTrue(proof['last_scope']['spawn_failed_before_exec']);self.assertEqual(proof['last_scope']['owned_processes'],[])

    def test_unknown_constructor_error_is_registered_and_has_no_drain_proof(self):
        for error in (KeyboardInterrupt('spawn interrupted'),OSError(errno.EIO,'pipe error')):
            f=Fixture()
            # Replace only the indefinite wait with a sentinel, never its proof.
            with patch.object(scope.secrets,'token_hex',return_value=f.nonce),patch.object(scope.subprocess,'Popen',side_effect=error),\
                    patch.object(f.scopes,'drain',side_effect=RuntimeError('synthetic pending observer')):
                with self.assertRaisesRegex(RuntimeError,'pending observer'):f.scopes.invoke(f.permit)
            self.assertIsNotNone(f.scopes.active);self.assertFalse(f.scopes.active['spawn_failed'])
            with self.assertRaises(ValueError):f.scopes._verify()
            self.assertNotIn('lock-release',f.events)

    def test_missing_ack_or_signal_killed_sudo_cannot_pass_by_direct_exit(self):
        for mode in ('no_ack','signal_exit'):
            f=Fixture()
            if mode=='no_ack':f.stdout=b''
            else:f.exit_code=-9
            f.active()
            with self.subTest(mode=mode),self.assertRaises(ValueError):f.scopes._verify()
            self.assertIsNotNone(f.scopes.active);self.assertEqual(f.scopes.count,0)

    def test_authenticated_helper_error_drains_but_never_returns_an_inspection(self):
        f=Fixture();f.stdout=(json.dumps(f.ack)+'\n').encode();f.exit_code=1
        with patch.object(scope.secrets,'token_hex',return_value=f.nonce),patch.object(scope.subprocess,'Popen',side_effect=f.popen):
            with self.assertRaises(scope.subprocess.CalledProcessError):f.scopes.invoke(f.permit)
        self.assertIsNone(f.scopes.active);self.assertTrue(f.scopes.summary()['physically_drained'])

    def test_transcript_limits_fail_closed_without_discarding_registration(self):
        f=Fixture();f.active();f.scopes.active['output']=(f.stdout,b'x'*65537)
        with self.assertRaisesRegex(ValueError,'Bounded complete'):f.scopes._verify()
        self.assertIsNotNone(f.scopes.active)

    def test_rolling_evidence_is_bounded_and_includes_every_completed_scope(self):
        f=Fixture()
        for _ in range(150):f.active();proof=f.scopes._verify()
        self.assertEqual(proof['completed_scopes'],150)
        self.assertLess(len(c.canonical(proof)),4096);self.assertNotEqual(proof['scope_chain_sha256'],'0'*64)


class ScopeAuthentication(unittest.TestCase):
    def test_direct_sudo_exec_helper_has_a_complete_two_identity_chain(self):
        f=Fixture();ack=copy.deepcopy(f.ack)
        ack['chain']=[dict(ack['chain'][0],pid=101,ppid=100),f.owner]
        self.assertEqual(scope.validate_start(ack,f.permit,f.nonce,f.owner,101),ack['chain'][:-1])

    def test_nonce_owner_direct_pid_and_every_ancestor_are_required(self):
        for mode in ('nonce','owner','sudo','helper','parent','duplicate','child_creation','direct_pid'):
            f=Fixture();ack=copy.deepcopy(f.ack);pid=101
            if mode=='nonce':ack['nonce']='f'*64
            elif mode=='owner':ack['chain'][-1]['start_ticks']='99999'
            elif mode=='sudo':ack['chain'][1]['argv']=['unrelated-claude']
            elif mode=='helper':ack['chain'][0]['argv']=['unrelated-xiaogou']
            elif mode=='parent':ack['chain'][0]['ppid']=100
            elif mode=='duplicate':ack['chain'][1]['pid']=103
            elif mode=='child_creation':ack['helper_spawning_forbidden']=False
            elif mode=='direct_pid':pid=999
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                scope.validate_start(ack,f.permit,f.nonce,f.owner,pid)

    def test_helper_ack_rechecks_all_actual_identities_and_retained_descriptor(self):
        f=Fixture();seen=[];owners={row['pid']:row for row in f.chain}
        def proc(pid):seen.append(pid);return copy.deepcopy(owners.get(pid))
        proof=[];io=SimpleNamespace(process_api=lambda:SimpleNamespace(proc=proc),descriptor=lambda owner,fd:proof.append((owner,fd)))
        ack=scope.start_ack(io,f.permit,dict(owner=f.owner,retained_fd=42),f.nonce,103)
        self.assertEqual(ack,f.ack);self.assertEqual(seen,[103,102,101,100]*2);self.assertEqual(proof,[(f.owner,42)])

    def test_helper_deadline_and_child_creation_block_are_real_code_with_mocked_effects(self):
        for event in scope.SPAWN_EVENTS:
            with self.subTest(event=event),self.assertRaises(RuntimeError):scope.forbid_helper_children(event,())
        scope.forbid_helper_children('open',('/synthetic',))
        with self.assertRaises(TimeoutError):scope.deadline_expired()
        io=SimpleNamespace(read=lambda *args:(_ for _ in ()).throw(ValueError('synthetic permit read stop')))
        with patch.object(holders.os,'geteuid',return_value=0),patch.object(holders.sys,'addaudithook')as audit,\
                patch.object(holders.signal,'signal')as handler,patch.object(holders.signal,'setitimer')as timer,\
                patch.dict('sys.modules',{'qwen35_sampler_runtime_io_v12':io}):
            with self.assertRaisesRegex(ValueError,'synthetic permit read stop'):
                holders.main(['--permit','/synthetic/permit.json','--permit-sha256','b'*64,'--scope-nonce','a'*64])
        audit.assert_called_once_with(scope.forbid_helper_children)
        handler.assert_called_once_with(holders.signal.SIGALRM,scope.deadline_expired)
        timer.assert_called_once_with(holders.signal.ITIMER_REAL,90)


class IntegrationBoundaries(unittest.TestCase):
    def test_failed_timeout_cannot_reach_deletion_or_success_publication(self):
        result=lifecycle.run_probe('holder_timeout');events=result['events']
        self.assertIn('timed out',result['error'])
        self.assertNotIn('prepare-exact-original-intent',events)
        self.assertNotIn('frozen-public-transaction-retire',events)
        self.assertLess(events.index('helper-scope-drain'),events.index('write-terminal.json'))
        terminal=result['values'][str(Path(result['fixture'].spec['paths']['control'])/'terminal.json')]
        self.assertTrue(terminal['failed']);self.assertEqual(terminal['failure']['type'],'TimeoutExpired')
        result=lifecycle.final_probe('holder_timeout');events=result['events']
        self.assertIn('timed out',result['error']);self.assertNotIn('write-actual-exit.json',events)
        self.assertLess(events.index('helper-scope-drain'),events.index('lock-release'))

    def test_both_actual_entry_paths_drain_before_any_terminal_or_lock_release(self):
        for run in (lifecycle.run_probe,lifecycle.final_probe):
            result=run();self.assertIsNone(result['error'])
            events=result['events'];self.assertIn('helper-scope-drain',events)
            self.assertEqual(result['events'][-1],'lock-release')
            if run==lifecycle.run_probe:
                self.assertLess(events.index('helper-scope-drain'),events.index('write-terminal.json'))
                terminal=result['values'][result['result']['path']]
                c.validate_scope_drain(terminal['helper_scope_drain'])
            else:
                self.assertLess(events.index('helper-scope-drain'),events.index('write-actual-exit.json'))
                self.assertEqual(events[-4:],['helper-scope-drain','lock-check','write-finalizer-helper-drain.json','lock-release'])
                self.assertLess(events.index('write-finalizer-helper-drain.json'),events.index('lock-release'))
                f=result['fixture'];receipt=f.values[f.spec['paths']['actual_exit']]
                c.validate_scope_drain(receipt['final_proof']['helper_scope_drain'])
                c.validate_scope_drain(f.values[str(Path(f.spec['paths']['control'])/'finalizer-helper-drain.json')])

    def test_no_timeout_kill_or_unconditional_lock_release_in_successors(self):
        text=Path(c.SOURCES[1]).read_text();self.assertNotIn('release(True)',text)
        self.assertNotIn('subprocess.run',text)
        tree=ast.parse(Path(c.SOURCES[3]).read_text())
        calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
        self.assertFalse(any(x.split('.')[-1]in ('kill','killpg','terminate','send_signal')for x in calls))
        self.assertNotIn('subprocess.run',calls)

    def test_original_frozen_v1_sources_and_limits_are_unchanged(self):
        pins={
            'contract':'0261bdc257b387218fac827714f7e262d9105ac39dda02bdbf91c44f4181812c',
            'caller':'2bb6a44a2723764b46e16ae1110123360f8c5b64f420c7b1ed354e6e5ff33d70',
            'holders':'8f10a504ccb453ca3a6a23d95b2f71b408034f30075c20c72da34baba695620a'}
        for key,h in pins.items():self.assertEqual(hashlib.sha256((Path(c.Q)/('minicpm5_retained_retirement_'+key+'_v1.py')).read_bytes()).hexdigest(),h)
        for path,h in c.base.FROZEN.items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),h)
        self.assertEqual(c.LIMITS,dict(seconds=21600,ack_seconds=180,root_floor_bytes=28<<30,nvme_floor_bytes=5<<30,
            control_bytes=128<<20,journal_bytes=128<<20,control_entries=128,journal_entries=4706,
            file_bytes=32<<20,source_bytes=32<<20,memory_address_bytes=6<<30))


if __name__=='__main__':unittest.main(verbosity=2)
