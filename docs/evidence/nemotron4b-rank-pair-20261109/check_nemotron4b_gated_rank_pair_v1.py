"""Real stdlib-only paired fixtures; never load SDK, native model or weights."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nemotron4b_process_handshake_v1 as handshake
import nemotron4b_gated_rank_pair_v1 as pair

NONCE = '6'*32


def fixture():
    ready, release = int(sys.argv[2]), int(sys.argv[3])
    role, mode, directory = sys.argv[4:7]
    root = Path(directory)
    if mode == 'early-fail':
        raise SystemExit(17)
    if mode == 'stubborn':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    actual_role = 'helper-0' if mode == 'wrong-role' else role
    proof = handshake.child_wait(ready, release, nonce=NONCE, role=actual_role,
                                deadline=time.monotonic()+5)
    marker = dict(role=role, release_ns=proof['child_release_ns'],
                  grant_ns=proof['grant']['grant_ns'], owner=proof['ready']['owner'])
    (root/(role+'.json')).write_text(json.dumps(marker))
    if mode == 'fail':
        raise SystemExit(19)
    if mode in ('wait', 'stubborn'):
        time.sleep(30)
    if mode == 'oversize':
        os.write(1, b'X'*8192)
        time.sleep(30)
    if mode != 'fast-success':
        peer = root/('rank-1.json' if role == 'rank-0' else 'rank-0.json')
        deadline = time.monotonic()+3
        while not peer.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('Peer was not released concurrently')
            time.sleep(.01)
    if mode == 'flood':
        for _ in range(32):
            os.write(1, b'O'*2048)
            os.write(2, b'E'*2048)
        os.write(1, b'\n')
    print(json.dumps(marker), flush=True)


class RankPair(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nano4b-pair-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recorded = []
        self.env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8',
                        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')

    def record(self, value):
        self.assertEqual(set(value['ranks']), {'rank-0', 'rank-1'})
        self.assertFalse((self.root/'rank-0.json').exists())
        self.assertFalse((self.root/'rank-1.json').exists())
        for role in pair.ROLES:
            row = value['ranks'][role]
            self.assertEqual(handshake.owner(row['first_actual_capture']['pid']),
                             row['first_actual_capture'])
            self.assertEqual(row['first_actual_capture'], row['second_actual_capture'])
        self.recorded.append(value)
        target = self.root/'captured.json'
        with target.open('x') as f:
            json.dump(value, f)
            f.flush()
            os.fsync(f.fileno())
        return dict(path=str(target), recorded_ns=time.monotonic_ns())

    def invoke(self, modes=('normal', 'normal'), **changes):
        commands = {}
        for role, mode in zip(pair.ROLES, modes):
            commands[role] = lambda ready, release, role=role, mode=mode: [
                str(Path(sys.executable).resolve()), '-B', '-S', str(Path(__file__).resolve()),
                '--fixture', str(ready), str(release), role, mode, str(self.root)]
        args = dict(environments={role:dict(self.env) for role in pair.ROLES},
            cwd=str(self.root), nonce=NONCE,
            expected_cgroup=handshake.owner(os.getpid())['cgroup'],
            guard=lambda: None, record_pair=self.record,
            handshake_seconds=2, execution_seconds=4, stdout_cap=131072, stderr_cap=131072)
        args.update(changes)
        self.last = pair.run(commands, **args)
        return self.last

    def assert_reaped(self, result):
        self.assertEqual(result['cleanup_errors'], [])
        for row in result['ranks'].values():
            if row['launched']:
                self.assertTrue(row['child_reaped'])
                self.assertTrue(row['actual_exit_observed'])
                self.assertIsNotNone(row['actual_exit_code'])
                self.assertFalse(Path('/proc', str(row['launched_pid'])).exists())
        self.assertFalse(result['runtime_admitted'])
        self.assertFalse(result['owned_group_drained_proven'])
        self.assertFalse(result['full_model_qualified'])

    def test_both_captured_before_either_release_then_peer_rendezvous(self):
        result = self.invoke()
        self.assertTrue(result['direct_children_successful'], result['failure'])
        self.assert_reaped(result)
        self.assertEqual(len(self.recorded), 1)
        receipt = result['pair_capture_receipt']
        for row in result['ranks'].values():
            value = json.loads(row['output']['stdout'])
            self.assertEqual(value['owner'], row['first_actual_capture'])
            self.assertLess(receipt['recorded_ns'], value['grant_ns'])
            self.assertLessEqual(value['grant_ns'], value['release_ns'])
            self.assertEqual(row['actual_exit_code'], 0)
            self.assertEqual(row['cleanup_signals'], [])

    def test_fast_first_success_does_not_require_released_peer_still_live(self):
        result = self.invoke(('fast-success', 'normal'))
        self.assertTrue(result['direct_children_successful'], result['failure'])
        self.assert_reaped(result)

    def test_both_output_pipes_collected_without_deadlock(self):
        result = self.invoke(('flood', 'flood'))
        self.assertTrue(result['direct_children_successful'], result['failure'])
        self.assert_reaped(result)
        for row in result['ranks'].values():
            self.assertEqual(row['output']['stderr'], b'E'*65536)
            self.assertTrue(row['output']['stdout'].startswith(b'O'*65536+b'\n'))

    def test_one_nonzero_exit_stops_and_reaps_live_sibling(self):
        result = self.invoke(('fail', 'wait'))
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertEqual(result['ranks']['rank-0']['actual_exit_code'], 19)
        self.assertIn('SIGTERM', result['ranks']['rank-1']['cleanup_signals'])
        self.assertEqual(result['ranks']['rank-1']['actual_exit_code'], -signal.SIGTERM)

    def test_stubborn_sibling_is_killed_and_failure_retained(self):
        result = self.invoke(('fail', 'stubborn'))
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertEqual(result['ranks']['rank-0']['actual_exit_code'], 19)
        self.assertEqual(result['ranks']['rank-1']['cleanup_signals'], ['SIGTERM', 'SIGKILL'])
        self.assertEqual(result['ranks']['rank-1']['actual_exit_code'], -signal.SIGKILL)

    def test_pre_ready_exit_releases_neither_rank(self):
        result = self.invoke(('normal', 'early-fail'))
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertEqual(self.recorded, [])
        self.assertEqual(result['ranks']['rank-1']['actual_exit_code'], 17)
        self.assertTrue(all(r['grant'] is None for r in result['ranks'].values()))
        self.assertFalse((self.root/'rank-0.json').exists())

    def test_wrong_ready_role_rejected_before_any_grant(self):
        result = self.invoke(('wrong-role', 'normal'))
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertEqual(self.recorded, [])
        self.assertTrue(all(r['grant'] is None for r in result['ranks'].values()))

    def test_durable_record_failure_releases_neither_rank(self):
        def denied(value):
            raise OSError('Synthetic durable record failure')
        result = self.invoke(record_pair=denied)
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertTrue(all(r['grant'] is None for r in result['ranks'].values()))
        self.assertFalse((self.root/'rank-0.json').exists())

    def test_guard_failure_after_record_retains_pair_and_releases_neither(self):
        def guard():
            if self.recorded:
                raise RuntimeError('Synthetic post-record guard failure')
        result = self.invoke(guard=guard)
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertIsNotNone(result['pair_capture'])
        self.assertIsNotNone(result['pair_capture_receipt'])
        self.assertTrue(all(r['grant'] is None for r in result['ranks'].values()))

    def test_timeout_reaps_both(self):
        result = self.invoke(('wait', 'wait'), execution_seconds=1)
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertIn('deadline', result['failure']['message'])
        self.assertTrue(all('SIGTERM' in r['cleanup_signals'] for r in result['ranks'].values()))

    def test_output_cap_failure_reaps_both(self):
        result = self.invoke(('oversize', 'wait'), stdout_cap=512)
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertIn('cap exceeded', result['failure']['message'])
        self.assertEqual(len(result['ranks']['rank-0']['output']['stdout']), 513)

    def test_second_spawn_failure_cleans_first(self):
        original = pair.subprocess.Popen
        calls = 0
        def spawn(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('Synthetic second rank spawn failure')
            return original(*args, **kwargs)
        with patch.object(pair.subprocess, 'Popen', side_effect=spawn):
            result = self.invoke()
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertTrue(result['ranks']['rank-0']['launched'])
        self.assertFalse(result['ranks']['rank-1']['launched'])
        self.assertIsNone(result['ranks']['rank-1']['actual_exit_code'])

    def test_original_cgroup_required_before_grant(self):
        result = self.invoke(expected_cgroup='0::/not-the-actual-fixture-group')
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertEqual(self.recorded, [])


    def test_missing_capture_receipt_releases_neither(self):
        result = self.invoke(record_pair=lambda value: {})
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertIsNotNone(result['pair_capture'])
        self.assertIsNone(result['pair_capture_receipt'])
        self.assertTrue(all(r['grant'] is None for r in result['ranks'].values()))

    def test_callback_cannot_mutate_actual_capture_or_argv(self):
        def record(value):
            receipt = self.record(value)
            value['ranks']['rank-0']['argv'][0] = '/not/the/interpreter'
            value['ranks']['rank-0']['first_actual_capture']['pid'] = -1
            return receipt
        result = self.invoke(record_pair=record)
        self.assertTrue(result['direct_children_successful'], result['failure'])
        self.assert_reaped(result)
        self.assertGreater(result['pair_capture']['ranks']['rank-0']['first_actual_capture']['pid'], 0)
        self.assertNotEqual(result['ranks']['rank-0']['argv'][0], '/not/the/interpreter')

    def test_second_grant_failure_retains_first_grant_and_pair_record(self):
        original = pair.send
        def send(fd, value, deadline):
            if value['role'] == 'rank-1':
                raise OSError('Synthetic second grant failure')
            return original(fd, value, deadline)
        with patch.object(pair, 'send', side_effect=send):
            result = self.invoke(('wait', 'normal'))
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertIsNotNone(result['pair_capture'])
        self.assertIsNotNone(result['pair_capture_receipt'])
        self.assertIsNotNone(result['ranks']['rank-0']['grant'])
        self.assertIsNone(result['ranks']['rank-1']['grant'])
        self.assertFalse((self.root/'rank-1.json').exists())

    def test_guard_failure_during_collection_reaps_both(self):
        def guard():
            if (self.root/'rank-0.json').exists() and (self.root/'rank-1.json').exists():
                raise RuntimeError('Synthetic guard failure after both releases')
        result = self.invoke(('wait', 'wait'), guard=guard)
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertIn('after both releases', result['failure']['message'])
        self.assertTrue(all(r['grant'] is not None for r in result['ranks'].values()))

    def test_no_bytecode_written(self):
        result = self.invoke()
        self.assertTrue(result['direct_children_successful'], result['failure'])
        self.assertEqual(list(self.root.rglob('*.pyc')), [])


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--fixture':
        fixture()
    else:
        unittest.main(verbosity=2)
