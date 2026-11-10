"""Durable capture and exact helper API tests using stdlib fixture processes.

Reads the installed helper's small Python source and library stat metadata,
never executes the installed SDK helper or reads the native library payload.
"""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import nemotron4b_process_handshake_v1 as handshake
import nemotron4b_gated_child_supervisor_v2 as supervisor
import nemotron4b_worker_receipts_v1 as receipts
import nemotron4b_gated_sdk_helper_v1 as sdk

NONCE='e'*32
PLAN=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research/nemotron4b-native-prerequisite-plan-20261106-v2.json')


def fixture():
    directory=Path(sys.argv[2]);mode=sys.argv[3]
    ready=int(sys.argv[sys.argv.index('--ready-fd')+1]);release=int(sys.argv[sys.argv.index('--release-fd')+1])
    if mode=='slow':time.sleep(.65)
    proof=handshake.child_wait(ready,release,nonce=NONCE,role='helper-0',deadline=time.monotonic()+5)
    recorded=receipts.unpack((directory/'helper-0-capture.json').read_bytes())
    assert recorded['first_actual_capture']==proof['ready']['owner']
    (directory/'body-started.json').write_text(json.dumps(proof))
    if mode=='slow':
        time.sleep(.8);(directory/'body-finished').touch()
    if mode=='error':raise SystemExit(17)
    if mode=='bad-stdout':os.write(1,b'wrong\xff\n')
    else:print(sdk.LIBRARY,flush=True)


def sdk_worker():
    directory=Path(sys.argv[2]);mode=sys.argv[3]
    writer=receipts.ReceiptWriter(directory,allowed_names=['helper-0-capture.json','helper-0-supervision.json'],guard=lambda:None)
    policy=json.loads(PLAN.read_text())['import_helper']
    if mode=='wrong-library':policy['library_identity']['size']+=1
    if mode=='wrong-source':policy['helper_sha256']='0'*64
    helper=sdk.GatedSDKHelper(policy,wrapper_prefix=[str(Path(sys.executable).resolve()),'-B','-S',
        str(Path(__file__).resolve()),'--fixture',str(directory),mode],
        environment=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1'),
        cwd=str(directory),nonce=NONCE,role='helper-0',expected_cgroup=handshake.owner(os.getpid())['cgroup'],
        guard=lambda:None,writer=writer)
    value=dict(actual_SDK_helper_executed=False)
    try:
        result=helper.run(['libneuronpjrt-path'],stdout=subprocess.PIPE)
        value.update(returncode=result.returncode,stdout=result.stdout.decode(),proof=helper.proof)
        if mode=='twice':
            try:helper.run(['libneuronpjrt-path'],stdout=subprocess.PIPE)
            except ValueError:value['second_call_rejected']=True
        if mode=='extra-spawn':
            try:subprocess.run(['/bin/true'],check=True)
            except ValueError:value['extra_spawn_rejected']=True
    except BaseException as error:value['error']=dict(type=type(error).__name__,message=str(error))
    value.update(calls=helper.calls,proof_present=helper.proof is not None)
    (directory/'worker-observation.json').write_text(json.dumps(value))


class ReceiptValues(unittest.TestCase):
    def test_exact_bytes_none_keys_and_nested_values_round_trip(self):
        value={'output':[b'',b'\x00\xff\n',None,{'base64':None,'path':b'bytes'}],
               'text':'中文','bool':True,'number':3.5}
        self.assertEqual(receipts.unpack(receipts.pack(value)),value)
        self.assertEqual(receipts.unpack(receipts.pack(b'root')),b'root')

    def test_duplicate_binary_paths_and_wrong_placeholder_rejected(self):
        value=json.loads(receipts.pack({'x':b'a'}))
        value['binary'].append(value['binary'][0])
        with self.assertRaises(ValueError):receipts.unpack(json.dumps(value).encode())
        value['binary'].pop();value['value']['x']='changed'
        with self.assertRaises(ValueError):receipts.unpack(json.dumps(value).encode())

    def test_noncanonical_encoding_and_duplicate_json_fields_rejected(self):
        raw=receipts.pack({'x':b'a'})
        with self.assertRaises(ValueError):receipts.unpack(raw.replace(b'YQ==',b'YR=='))
        with self.assertRaises(ValueError):receipts.unpack(b'{"schema":1,"schema":2}')

    def test_nonfinite_unknown_deep_and_oversized_values_rejected(self):
        deep=[];value=deep
        for _ in range(34):new=[];value.append(new);value=new
        for value in [float('nan'),object(),deep,b'x'*(5*2**20+1)]:
            with self.subTest(kind=type(value).__name__),self.assertRaises(ValueError):receipts.pack(value)


class ReceiptFiles(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='nano-durable-receipt-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()

    def writer(self,**kwargs):
        return receipts.ReceiptWriter(self.root,allowed_names=['capture.json','result.json'],guard=lambda:None,**kwargs)

    def test_written_file_hash_readback_and_no_overwrite(self):
        writer=self.writer();value={'raw':b'\xff\0'};reference=writer.write('capture.json',value)
        raw=Path(reference['path']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),reference['sha256'])
        self.assertEqual(receipts.unpack(raw),value)
        with self.assertRaises(FileExistsError):writer.write('capture.json',{'changed':True})
        self.assertEqual(Path(reference['path']).read_bytes(),raw)

    def test_file_and_directory_are_fsynced_before_reference(self):
        with patch.object(receipts.os,'fsync',wraps=os.fsync)as sync:
            self.writer().write('capture.json',{'captured':True})
            self.assertEqual(sync.call_count,2)

    def test_failed_sync_keeps_partial_evidence_without_success_reference(self):
        with patch.object(receipts.os,'fsync',side_effect=OSError('synthetic disk error')):
            with self.assertRaises(OSError):self.writer().write('capture.json',{'captured':True})
        self.assertTrue((self.root/'capture.json').is_file())
        self.assertEqual(receipts.unpack((self.root/'capture.json').read_bytes()),{'captured':True})

    def test_terminal_reservation_rejects_output_before_creation(self):
        writer=self.writer(work_cap=16384,reserve=4096)
        writer.write('capture.json',{'small':True})
        with self.assertRaisesRegex(ValueError,'reservation'):writer.write('result.json',{'small':True})
        self.assertFalse((self.root/'result.json').exists())

    def test_unknown_name_and_symlink_tree_rejected(self):
        writer=self.writer()
        with self.assertRaises(ValueError):writer.write('../outside.json',{})
        (self.root/'link').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError):writer.write('capture.json',{})

    def test_guard_denial_prevents_receipt_and_lock_creation(self):
        writer=self.writer()
        def deny():raise RuntimeError('resource denied')
        writer.guard=deny
        with self.assertRaises(RuntimeError):writer.write('capture.json',{})
        self.assertEqual(list(self.root.iterdir()),[])


class RecordedRelease(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='nano-recorded-release-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.writer=receipts.ReceiptWriter(self.root,
            allowed_names=['helper-0-capture.json'],guard=lambda:None)

    def run_child(self,record=None,mode='normal',**changes):
        def command(ready,release):
            return [str(Path(sys.executable).resolve()),'-B','-S',str(Path(__file__).resolve()),
                    '--fixture',str(self.root),mode,'--ready-fd',str(ready),'--release-fd',str(release)]
        args=dict(environment=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1'),
            cwd=str(self.root),nonce=NONCE,role='helper-0',expected_cgroup=handshake.owner(os.getpid())['cgroup'],
            guard=lambda:None,record_capture=record or(lambda value:self.writer.write('helper-0-capture.json',value)),
            handshake_seconds=2,execution_seconds=2,stdout_cap=4096,stderr_cap=16384)
        args.update(changes)
        result=supervisor.run(command,**args);self.assertTrue(result['child_reaped']);self.assertEqual(result['cleanup_errors'],[])
        return result

    def test_actual_capture_is_durable_before_child_body_and_grant(self):
        result=self.run_child();self.assertTrue(result['successful'],result['failure'])
        proof=result['handshake'];body=json.loads((self.root/'body-started.json').read_text())
        self.assertLessEqual(proof['capture_recorded_ns'],proof['grant']['guard_complete_ns'])
        self.assertLessEqual(proof['grant']['grant_ns'],body['child_release_ns'])
        self.assertEqual(body['ready']['owner'],receipts.unpack(Path(proof['capture_record']['path']).read_bytes())['first_actual_capture'])

    def test_persistence_failure_sends_no_grant_and_reaps_child(self):
        def fail(value):raise OSError('capture disk failure')
        result=self.run_child(record=fail)
        self.assertFalse(result['successful']);self.assertIsNone(result['handshake'])
        self.assertFalse((self.root/'body-started.json').exists())

    def test_invalid_record_reference_sends_no_grant(self):
        result=self.run_child(record=lambda value:{'path':'relative','sha256':'0'*64})
        self.assertFalse(result['successful']);self.assertFalse((self.root/'body-started.json').exists())

    def test_original_single_total_budget_includes_handshake(self):
        result=self.run_child(mode='slow',total_seconds=1)
        self.assertFalse(result['successful']);self.assertTrue((self.root/'body-started.json').exists())
        self.assertFalse((self.root/'body-finished').exists())
        self.assertTrue(result['cleanup_signals'])

    def test_nonzero_helper_exit_remains_failure_with_capture_evidence(self):
        result=self.run_child(mode='error')
        self.assertFalse(result['successful']);self.assertEqual(result['actual_exit_code'],17)
        self.assertIsNotNone(result['handshake']['capture_record'])


class SDKRequest(unittest.TestCase):
    def invoke(self,mode):
        temporary=tempfile.TemporaryDirectory(prefix='nano-sdk-request-fixture-');self.addCleanup(temporary.cleanup)
        directory=Path(temporary.name)
        result=subprocess.run([str(Path(sys.executable).resolve()),'-B','-S',str(Path(__file__).resolve()),
                               '--sdk-worker',str(directory),mode],capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr.decode())
        return directory,json.loads((directory/'worker-observation.json').read_text())

    def test_exact_sdk_request_records_real_wrapper_and_preserves_stdout(self):
        directory,value=self.invoke('normal')
        self.assertNotIn('error',value);self.assertEqual(value['stdout'],sdk.LIBRARY+'\n')
        self.assertEqual(value['calls'],1);self.assertTrue(value['proof_present'])
        self.assertNotEqual(value['proof']['actual_wrapper_argv'],[sdk.HELPER])
        original=receipts.unpack((directory/'helper-0-supervision.json').read_bytes())
        self.assertTrue(original['successful']);self.assertEqual(original['output']['stdout'],(sdk.LIBRARY+'\n').encode())
        self.assertFalse(value['actual_SDK_helper_executed'])

    def test_failed_helper_keeps_actual_exit_and_binary_output(self):
        for mode in ['error','bad-stdout']:
            with self.subTest(mode=mode):
                directory,value=self.invoke(mode);self.assertIn('error',value);self.assertFalse(value['proof_present'])
                result=receipts.unpack((directory/'helper-0-supervision.json').read_bytes())
                if mode=='error':self.assertEqual(result['actual_exit_code'],17)
                else:self.assertEqual(result['output']['stdout'],b'wrong\xff\n')

    def test_second_request_and_unrelated_spawn_rejected(self):
        for mode,field in [('twice','second_call_rejected'),('extra-spawn','extra_spawn_rejected')]:
            with self.subTest(mode=mode):
                directory,value=self.invoke(mode);self.assertTrue(value[field]);self.assertEqual(value['calls'],1)

    def test_original_source_or_library_metadata_mismatch_prevents_helper_body(self):
        for mode in ['wrong-source','wrong-library']:
            with self.subTest(mode=mode):
                directory,value=self.invoke(mode);self.assertIn('error',value)
                self.assertFalse((directory/'body-started.json').exists())
                self.assertFalse((directory/'helper-0-capture.json').exists())


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fixture':fixture()
    elif len(sys.argv)>1 and sys.argv[1]=='--sdk-worker':sdk_worker()
    else:unittest.main(verbosity=2)
