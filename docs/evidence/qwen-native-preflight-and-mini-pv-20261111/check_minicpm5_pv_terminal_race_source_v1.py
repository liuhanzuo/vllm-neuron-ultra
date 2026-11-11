"""Source-only mock proof for PV finish's poll-to-proc terminal race.

The actual frozen method is AST-extracted without importing its module.
The candidate changes only the live-loop observation branch. It is a design
prototype, not an admitted worker/adapter. No real proc/process/lock/native IO.
"""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


SOURCE = Path(__file__).with_name('minicpm5_layer1_fixed_input_pv_adapter_v2.py')
SOURCE_SHA = 'd6991457ccc831103fb7e71d6c416c37d114473c96bc30a97b944c4ac2243523'


def require(value, message):
    if not value:
        raise ValueError(message)


def method(candidate, namespace):
    raw = SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_SHA, 'Frozen adapter source changed')
    tree = ast.parse(raw, filename=str(SOURCE))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Adapter')
    node = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'finish'))
    if candidate:
        loops = [item for item in node.body if isinstance(item, ast.While)]
        require(len(loops) == 1, 'Exact single original finite live loop')
        replacement = ast.parse('''
while self.child.poll() is None:
    self.guard()
    current = r.p.proc(self.child.pid)
    if current != owner:
        require(current is None or current == dict(owner, argv=[]),
                'Unexpected child identity is never downgraded to terminal')
        require(self.child.poll() is not None, 'Exact child identity while live')
        break
    require(time.monotonic() < deadline, 'Original whole-phase wall bound')
    time.sleep(.1)
''').body[0]
        node.body[node.body.index(loops[0])] = replacement
    ast.fix_missing_locations(node)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(SOURCE) + ('#candidate-design' if candidate else '#frozen'), 'exec'), namespace)
    return namespace['finish']


OWNER = dict(pid=101, start_ticks='200', ppid=99, argv=['original-python', '-B', 'worker'], cgroup='original-cgroup')


class FakeChild:
    def __init__(self, polls, exit_code, log):
        self.pid = OWNER['pid']
        self.polls, self.exit_code, self.log = iter(polls), exit_code, log
    def poll(self):
        value = next(self.polls)
        self.log.append(('poll', value))
        return value
    def wait(self, timeout):
        self.log.append(('wait', timeout))
        return self.exit_code


def fixture(candidate, polls, observed, exit_code=0, *, guard_error=False, gone_error=False, members=False):
    log, writes = [], []
    def guard():
        log.append(('guard',))
        require(not guard_error, 'Guard failed')
    def proc(pid):
        log.append(('proc', pid))
        return copy.deepcopy(observed)
    def gone(owner):
        log.append(('gone',))
        require(not gone_error, 'Owner still present')
    def group_members(pid):
        log.append(('group_members',))
        return ['descendant'] if members else []
    def write(name, value):
        log.append(('write', name))
        writes.append((name, copy.deepcopy(value)))
    child = FakeChild(polls, exit_code, log)
    r = NS(p=NS(proc=proc, gone=gone), drain_group=lambda *args: log.append(('drain',)), group_members=group_members)
    namespace = dict(require=require, r=r, prior=NS(LIMITS={'wall_seconds': 900}),
                     time=NS(monotonic=lambda: 1, sleep=lambda seconds: log.append(('sleep', seconds))))
    call = method(candidate, namespace)
    ports = NS(index=1, owner=copy.deepcopy(OWNER), child=child, guard=guard, store=NS(json=write), last_exit=0)
    return lambda: call(ports, 1, copy.deepcopy(OWNER), copy.deepcopy(OWNER['argv'])), ports, log, writes


class TerminalRaceTests(unittest.TestCase):
    def test_01_original_exact_success(self):
        call, _, _, writes = fixture(False, [None, 0], OWNER)
        self.assertEqual(call()['returncode'], 0)
        self.assertEqual(writes[0][0], 'phase-1-exit.json')

    def test_02_original_zombie_shaped_race_reproduced(self):
        call, _, log, writes = fixture(False, [None, 1], dict(OWNER, argv=[]), 1)
        with self.assertRaisesRegex(ValueError, 'Exact child identity while live'):
            call()
        self.assertEqual(writes, [])
        self.assertFalse(any(event[0] == 'wait' for event in log))

    def test_03_original_disappearance_race_reproduced(self):
        call, _, _, writes = fixture(False, [None, 1], None, 1)
        with self.assertRaisesRegex(ValueError, 'Exact child identity while live'):
            call()
        self.assertEqual(writes, [])

    def test_04_candidate_exact_success(self):
        call, ports, _, writes = fixture(True, [None, 0], OWNER)
        self.assertEqual(call()['returncode'], 0)
        self.assertEqual(len(writes), 1)
        self.assertIsNone(ports.child)

    def test_05_candidate_terminal_nonzero_recorded_without_promotion(self):
        call, _, _, writes = fixture(True, [1], None, 1)
        value = call()
        self.assertEqual(value['returncode'], 1)
        self.assertEqual(writes[0][1]['returncode'], 1)
        self.assertFalse(value['self_exit_claimed'])

    def test_06_candidate_zombie_shaped_race_records_actual_nonzero(self):
        call, _, _, writes = fixture(True, [None, 1], dict(OWNER, argv=[]), 1)
        self.assertEqual(call()['returncode'], 1)
        self.assertEqual(writes[0][1]['owner'], OWNER)

    def test_07_candidate_disappearance_race_records_actual_nonzero(self):
        call, _, _, writes = fixture(True, [None, 1], None, 1)
        self.assertEqual(call()['returncode'], 1)
        self.assertEqual(len(writes), 1)

    def test_08_nonempty_changed_argv_never_accepted(self):
        call, _, _, writes = fixture(True, [None, 1], dict(OWNER, argv=['different']), 1)
        with self.assertRaisesRegex(ValueError, 'Unexpected child identity'):
            call()
        self.assertEqual(writes, [])

    def test_09_changed_pid_never_accepted(self):
        call, _, _, _ = fixture(True, [None, 1], dict(OWNER, pid=102, argv=[]), 1)
        with self.assertRaisesRegex(ValueError, 'Unexpected child identity'):
            call()

    def test_10_changed_start_never_accepted(self):
        call, _, _, _ = fixture(True, [None, 1], dict(OWNER, start_ticks='201', argv=[]), 1)
        with self.assertRaisesRegex(ValueError, 'Unexpected child identity'):
            call()

    def test_11_changed_parent_never_accepted(self):
        call, _, _, _ = fixture(True, [None, 1], dict(OWNER, ppid=98, argv=[]), 1)
        with self.assertRaisesRegex(ValueError, 'Unexpected child identity'):
            call()

    def test_12_changed_cgroup_never_accepted(self):
        call, _, _, _ = fixture(True, [None, 1], dict(OWNER, cgroup='changed', argv=[]), 1)
        with self.assertRaisesRegex(ValueError, 'Unexpected child identity'):
            call()

    def test_13_empty_argv_while_still_live_rejected(self):
        call, _, _, writes = fixture(True, [None, None], dict(OWNER, argv=[]))
        with self.assertRaisesRegex(ValueError, 'Exact child identity while live'):
            call()
        self.assertEqual(writes, [])

    def test_14_owner_must_be_gone_before_receipt(self):
        call, _, _, writes = fixture(True, [1], None, 1, gone_error=True)
        with self.assertRaisesRegex(ValueError, 'Owner still present'):
            call()
        self.assertEqual(writes, [])

    def test_15_descendants_block_receipt(self):
        call, _, _, writes = fixture(True, [1], None, 1, members=True)
        with self.assertRaisesRegex(ValueError, 'Entire original owned process group empty'):
            call()
        self.assertEqual(writes, [])

    def test_16_guard_failure_preserved(self):
        call, _, _, writes = fixture(True, [None, 1], None, 1, guard_error=True)
        with self.assertRaisesRegex(ValueError, 'Guard failed'):
            call()
        self.assertEqual(writes, [])

    def test_17_wait_drain_gone_group_then_receipt(self):
        call, _, log, _ = fixture(True, [None, 1], dict(OWNER, argv=[]), 1)
        call()
        stages = [event[0] for event in log if event[0] in ('wait', 'drain', 'gone', 'group_members', 'write')]
        self.assertEqual(stages, ['wait', 'drain', 'gone', 'group_members', 'write'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
