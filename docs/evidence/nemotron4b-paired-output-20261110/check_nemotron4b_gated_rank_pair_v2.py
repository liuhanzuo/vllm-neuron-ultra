"""Generated paired processes only: regression and original lifecycle coverage.

No model, SDK, original site restoration, native payload or systemd service.
Both original interpreters are used; generated rank bodies only write bounded
bytes and participate in the existing actual-process handshake protocol.
"""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import check_nemotron4b_gated_rank_pair_v1 as legacy_tests
import nemotron4b_gated_rank_pair_v1 as previous
import nemotron4b_gated_rank_pair_v2 as pair
import nemotron4b_owned_group_drain_v3 as old_group
import nemotron4b_owned_group_drain_v4 as group
import nemotron4b_process_handshake_v1 as handshake

NONCE = 'e'*32


def early(role, stream):
    return (role+':pre-gate:'+stream).encode()+b'\x00\xff\n'


def late(role, stream):
    return (role+':post-gate:'+stream).encode()+b'\xfe\x00\n'


def both(root, suffix):
    deadline = time.monotonic()+3
    while not all((root/(role+suffix)).exists() for role in pair.ROLES):
        if time.monotonic() >= deadline:
            raise RuntimeError('Generated peer did not reach byte barrier')
        time.sleep(.005)


def fixture():
    role, mode, folder, ready, release = sys.argv[2:]
    root = Path(folder)
    ready, release = int(ready), int(release)
    for name, fd in (('stdout', 1), ('stderr', 2)):
        os.write(fd, early(role, name))
        if mode == 'early-overflow':
            os.write(fd, b'X'*512)
    (root/(role+'-pre-written')).touch()
    both(root, '-pre-written')
    if mode in ('early-exit', 'early-overflow'):
        os._exit(37)
    if mode == 'malformed':
        os.write(ready, b'invalid');os.close(ready);os.close(release);os._exit(39)
    handshake.child_wait(ready, release, nonce=NONCE, role=role,
                         deadline=time.monotonic()+5)
    (root/(role+'-released')).touch()
    for name, fd in (('stdout', 1), ('stderr', 2)):
        os.write(fd, late(role, name))
    (root/(role+'-post-written')).touch()
    both(root, '-post-written')
    if mode == 'late-overflow':
        os.write(2, b'Y'*512)
    if mode == 'late-exit':
        os._exit(41)
    if mode == 'wait':
        time.sleep(30)
    os._exit(0)


class OriginalLifecycle(legacy_tests.RankPair):
    """Rerun the original assertions against v2, not a copy of the tests."""
    def setUp(self):
        super().setUp()
        replacement = patch.object(legacy_tests, 'pair', pair)
        replacement.start();self.addCleanup(replacement.stop)

    def test_output_cap_failure_reaps_both(self):
        # v1 kept its over-cap sentinel; v2 reports truncation separately.
        result = self.invoke(('oversize', 'wait'), stdout_cap=512)
        self.assertFalse(result['direct_children_successful'])
        self.assert_reaped(result)
        self.assertIn('cap exceeded', result['failure']['message'])
        self.assertEqual(result['ranks']['rank-0']['output']['stdout'], b'X'*512)
        self.assertTrue(result['ranks']['rank-0']['output_capture']['truncated']['stdout'])
        self.assertFalse(result['ranks']['rank-0']['output_pipe_capture_complete'])


class PairedOutput(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='nano-pair-output-fixture-')
        self.addCleanup(temporary.cleanup);self.root = Path(temporary.name)

    def invoke(self, modes=('normal', 'normal'), implementation=pair, **changes):
        def command(role, mode):
            return lambda ready, release: [
                sys.executable, '-B', '-S', str(Path(__file__).resolve()),
                '--fixture', role, mode, str(self.root), str(ready), str(release)]
        commands = {role:command(role, mode) for role, mode in zip(pair.ROLES, modes)}
        def record(value):
            self.assertFalse(any((self.root/(role+'-released')).exists() for role in pair.ROLES))
            target = self.root/'capture.json'
            with target.open('x') as f:
                json.dump(value, f);f.flush();os.fsync(f.fileno())
            return dict(path=str(target), recorded_ns=time.monotonic_ns())
        kwargs = dict(
            environments={role:dict(PATH='/usr/bin:/bin', LANG='C.UTF-8',
                                    PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
                          for role in pair.ROLES},
            cwd=str(self.root), nonce=NONCE,
            expected_cgroup=handshake.owner(os.getpid())['cgroup'],
            guard=lambda:None, record_pair=record,
            handshake_seconds=3, execution_seconds=3,
            stdout_cap=4096, stderr_cap=4096)
        kwargs.update(changes)
        result = implementation.run(commands, **kwargs)
        self.assertEqual(result['cleanup_errors'], [])
        for row in result['ranks'].values():
            self.assertTrue(row['child_reaped'])
            self.assertTrue(row['actual_exit_observed'])
            self.assertFalse(Path('/proc', str(row['launched_pid'])).exists())
        for name in ('runtime_admitted', 'owned_group_drained_proven',
                     'model_or_native_runtime_qualified', 'full_model_qualified',
                     'new_speedup_claim'):
            self.assertFalse(result[name])
        if implementation is pair:
            # Exercise the new adapter with every real generated-process result.
            summary = group.supervisor_summary(result, NONCE)
            self.assertEqual(summary['successful'], result['direct_children_successful'])
            self.assertEqual(set(summary['rank_output_capture']), set(pair.ROLES))
        return result

    def test_old_pair_loses_bytes_from_both_early_exits(self):
        result = self.invoke(('early-exit', 'early-exit'), implementation=previous)
        self.assertFalse(result['direct_children_successful'])
        self.assertTrue(all(row['output'] == {'stdout':b'', 'stderr':b''}
                            for row in result['ranks'].values()))

    def test_both_early_failures_preserve_all_four_binary_streams(self):
        result = self.invoke(('early-exit', 'early-exit'))
        self.assertFalse(result['direct_children_successful'])
        self.assertIsNone(result['pair_capture'])
        for role, row in result['ranks'].items():
            self.assertEqual(row['output'], {name:early(role, name) for name in ('stdout', 'stderr')})
            self.assertTrue(row['output_pipe_capture_complete'])
            self.assertTrue(row['output_may_be_partial'])
            self.assertIsNone(row['grant'])

    def test_malformed_ready_preserves_both_rank_diagnostic_prefixes(self):
        result = self.invoke(('malformed', 'normal'))
        self.assertFalse(result['direct_children_successful'])
        for role, row in result['ranks'].items():
            self.assertTrue(row['output']['stderr'].startswith(early(role, 'stderr')))
            self.assertIsNone(row['grant'])

    def test_capture_refusal_keeps_pre_grant_output_without_releasing_ranks(self):
        def refuse(value):
            raise OSError('generated durable capture refused')
        result = self.invoke(record_pair=refuse)
        self.assertFalse(result['direct_children_successful'])
        self.assertIn('capture refused', result['failure']['message'])
        for role, row in result['ranks'].items():
            self.assertTrue(row['output']['stdout'].startswith(early(role, 'stdout')))
            self.assertTrue(row['output']['stderr'].startswith(early(role, 'stderr')))
            self.assertIsNone(row['grant'])

    def test_normal_pair_has_complete_exact_output_and_validated_adapter(self):
        result = self.invoke()
        self.assertTrue(result['direct_children_successful'], result['failure'])
        for role, row in result['ranks'].items():
            self.assertEqual(row['output'], {
                name:early(role, name)+late(role, name) for name in ('stdout', 'stderr')})
            self.assertTrue(row['output_pipe_capture_complete'])
            self.assertFalse(row['output_may_be_partial'])
            self.assertFalse(row['output_capture']['drain_timed_out'])

    def test_nonzero_rank_preserves_both_outputs_and_reaps_waiting_sibling(self):
        result = self.invoke(('late-exit', 'wait'))
        self.assertFalse(result['direct_children_successful'])
        self.assertEqual(result['ranks']['rank-0']['actual_exit_code'], 41)
        self.assertTrue(result['ranks']['rank-1']['cleanup_signals'])
        for role, row in result['ranks'].items():
            self.assertEqual(row['output'], {
                name:early(role, name)+late(role, name) for name in ('stdout', 'stderr')})
            self.assertTrue(row['output_pipe_capture_complete'])
            self.assertTrue(row['output_may_be_partial'])

    def test_early_overflow_keeps_exact_prefix_and_flags_each_rank_and_stream(self):
        result = self.invoke(('early-overflow', 'early-overflow'), stdout_cap=64, stderr_cap=64)
        self.assertFalse(result['direct_children_successful'])
        for role, row in result['ranks'].items():
            for name in ('stdout', 'stderr'):
                self.assertEqual(row['output'][name], (early(role, name)+b'X'*512)[:64])
                self.assertTrue(row['output_capture']['truncated'][name])
                self.assertFalse(row['output_capture']['eof'][name])
            self.assertFalse(row['output_pipe_capture_complete'])

    def test_post_release_overflow_does_not_discard_other_rank_or_stream(self):
        result = self.invoke(('late-overflow', 'normal'), stderr_cap=64)
        self.assertFalse(result['direct_children_successful'])
        for role, row in result['ranks'].items():
            self.assertEqual(row['output']['stdout'], early(role, 'stdout')+late(role, 'stdout'))
        self.assertEqual(result['ranks']['rank-0']['output']['stderr'],
                         (early('rank-0', 'stderr')+late('rank-0', 'stderr')+b'Y'*512)[:64])
        self.assertTrue(result['ranks']['rank-0']['output_capture']['truncated']['stderr'])
        self.assertEqual(result['ranks']['rank-1']['output']['stderr'],
                         early('rank-1', 'stderr')+late('rank-1', 'stderr'))

    def test_guard_failure_after_both_writes_still_drains_all_streams(self):
        def guard():
            if all((self.root/(role+'-post-written')).exists() for role in pair.ROLES):
                raise ValueError('generated post-release guard refusal')
        result = self.invoke(('wait', 'wait'), guard=guard)
        self.assertFalse(result['direct_children_successful'])
        self.assertIn('guard refusal', result['failure']['message'])
        for role, row in result['ranks'].items():
            self.assertEqual(row['output'], {
                name:early(role, name)+late(role, name) for name in ('stdout', 'stderr')})

    def test_timeout_keeps_both_outputs_and_original_failure(self):
        result = self.invoke(('wait', 'wait'), execution_seconds=1)
        self.assertFalse(result['direct_children_successful'])
        self.assertIn('deadline', result['failure']['message'])
        for role, row in result['ranks'].items():
            self.assertEqual(row['output'], {
                name:early(role, name)+late(role, name) for name in ('stdout', 'stderr')})
            self.assertTrue(row['cleanup_signals'])

    def test_adapter_rejects_inconsistent_and_incomplete_capture_on_success(self):
        result = self.invoke()
        mutations = (
            lambda r:r['ranks']['rank-0']['output_capture']['eof'].update(stdout=False),
            lambda r:r['ranks']['rank-1']['output_capture']['truncated'].update(stderr=True),
            lambda r:r['ranks']['rank-0'].update(output_may_be_partial=True),
            lambda r:r['ranks']['rank-1']['output_capture'].update(drain_timed_out=True),
            lambda r:r['ranks']['rank-0']['output_capture'].update(drain_seconds=1.0),
        )
        for mutate in mutations:
            bad = copy.deepcopy(result);mutate(bad)
            with self.assertRaises((AssertionError, ValueError)):
                group.supervisor_summary(bad, NONCE)

    def test_original_result_is_persisted_before_owned_phase_finish(self):
        result = self.invoke()
        events = []
        reference = {'path':str(self.root/'original-supervision.json'), 'sha256':'d'*64}
        class Phase(group.OwnedPhase):
            def __init__(self):
                self.nonce = NONCE
            def finish(self, summary):
                events.append(('finish', summary));return summary
        def persist(value):
            self.assertIs(value, result);events.append(('persist', value));return reference
        summary = Phase().finish_supervisor(result, persist_result=persist)
        self.assertEqual([x[0] for x in events], ['persist', 'finish'])
        self.assertEqual(summary['original_supervisor_result'], reference)
        self.assertEqual(summary['supervisor_schema'], 'nemotron4b_gated_rank_pair_v2')

    def test_legacy_rank_schema_still_uses_unchanged_original_adapter(self):
        result = self.invoke(implementation=previous)
        self.assertEqual(group.supervisor_summary(result, NONCE),
                         old_group.supervisor_summary(result, NONCE))


class DrainAndSource(unittest.TestCase):
    def row(self, content, hold_open):
        readfd, writefd = os.pipe()
        stream = os.fdopen(readfd, 'rb');self.addCleanup(stream.close)
        os.write(writefd, content)
        if hold_open:
            self.addCleanup(os.close, writefd)
        else:
            os.close(writefd)
        return dict(child=types.SimpleNamespace(stdout=stream, stderr=None),
                    output={'stdout':bytearray(), 'stderr':bytearray()},
                    output_capture=dict(eof={'stdout':False, 'stderr':True},
                        truncated={'stdout':False, 'stderr':False},
                        drain_timed_out=False, drain_seconds=.25))

    def test_two_held_open_rank_pipes_share_one_deadline_and_keep_available_bytes(self):
        rows = {role:self.row(role.encode(), True) for role in pair.ROLES}
        start = time.monotonic()
        pair.drain_after_exit(rows, deadline=start+.08, caps={'stdout':64, 'stderr':64})
        self.assertLess(time.monotonic()-start, .5)
        for role, row in rows.items():
            self.assertEqual(bytes(row['output']['stdout']), role.encode())
            self.assertTrue(row['output_capture']['drain_timed_out'])
            self.assertFalse(row['output_capture']['eof']['stdout'])

    def test_exact_cap_eof_is_complete_for_both_ranks(self):
        rows = {role:self.row(b'X'*64, False) for role in pair.ROLES}
        pair.drain_after_exit(rows, deadline=time.monotonic()+.1, caps={'stdout':64, 'stderr':64})
        for row in rows.values():
            self.assertEqual(bytes(row['output']['stdout']), b'X'*64)
            self.assertTrue(row['output_capture']['eof']['stdout'])
            self.assertFalse(row['output_capture']['truncated']['stdout'])

    def test_capture_before_either_grant_and_owned_signaling_are_unchanged(self):
        def definitions(module):
            return {x.name:ast.dump(x, include_attributes=False)
                    for x in ast.parse(Path(module.__file__).read_bytes()).body
                    if isinstance(x, ast.FunctionDef)}
        old, new = definitions(previous), definitions(pair)
        for name in ('validate_ready', 'capture_and_release', 'signal_owned'):
            self.assertEqual(new[name], old[name], name)


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fixture':
        fixture()
    else:
        unittest.main(verbosity=2)
