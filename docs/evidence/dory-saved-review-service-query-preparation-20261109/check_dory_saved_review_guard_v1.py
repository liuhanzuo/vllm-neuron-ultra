"""Synthetic full-guard parity; does not touch actual locks or service state."""
from types import SimpleNamespace
import unittest

from dory_saved_review_guard_v1 import CompleteReviewGuard


class Query:
    startup_file_count = 41
    startup_bytes = 4326816
    requests = 0
    closed = poisoned = False

    def __init__(self, trace):
        self.trace = trace

    def audit(self):
        self.trace.append('audit')

    def check(self):
        self.requests += 1
        self.trace.append('fresh_service_request')

    def close(self):
        self.closed = True
        self.trace.append('query_close')


class Guard(unittest.TestCase):
    def make(self):
        trace, state = [], dict(now=2, root=100, nvme=100, memory=100)
        def lock(owner, value):
            self.assertEqual((owner, value), ('owner', 'lock'))
            trace.append('lock')
        low = SimpleNamespace(lock_proof=lock,
            clear_window=lambda: trace.append('window'),
            active_heavy=lambda: trace.append('original_service_request'))
        def free(path):
            trace.append(path)
            return state['root' if path == '/' else 'nvme']
        def memory():
            trace.append('memory')
            return state['memory']
        guard = CompleteReviewGuard(low=low, owner='owner', lock='lock',
            budget=dict(wall_seconds=10, root_floor=30, nvme_floor=20,
                        nvme_new_bytes=15, memory_available_floor_bytes=40),
            started=1, initial_nvme=100, free=free, memory_free=memory,
            clock=lambda: state['now'])
        return guard, trace, state

    def test_original_startup_order_and_all_resource_checks(self):
        guard, trace, _ = self.make()
        guard()
        self.assertEqual(trace, ['lock', 'window', 'original_service_request', '/', '/mnt/nvme', 'memory'])

    def test_persistent_transport_preserves_every_other_check_and_cadence(self):
        guard, trace, _ = self.make()
        guard()
        query = Query(trace)
        guard.attach_authenticated_query(query)
        trace.clear()
        for _ in range(3):
            guard()
        self.assertEqual(trace, ['lock', 'window', 'fresh_service_request', '/', '/mnt/nvme', 'memory'] * 3)
        self.assertEqual(query.requests, 3)
        self.assertEqual(guard.accounting()['successful_guards'], 4)

    def test_no_attach_before_original_startup(self):
        guard, trace, _ = self.make()
        with self.assertRaises(RuntimeError):
            guard.attach_authenticated_query(Query(trace))

    def test_second_attachment_rejected(self):
        guard, trace, _ = self.make()
        guard()
        guard.attach_authenticated_query(Query(trace))
        with self.assertRaises(RuntimeError):
            guard.attach_authenticated_query(Query(trace))

    def test_incomplete_dependency_authentication_rejected(self):
        guard, trace, _ = self.make()
        guard()
        query = Query(trace)
        query.startup_file_count = 19
        with self.assertRaises(RuntimeError):
            guard.attach_authenticated_query(query)

    def test_resource_failures_remain_failures_after_transport_change(self):
        for changed in [dict(now=11), dict(root=29), dict(nvme=84), dict(memory=39)]:
            guard, trace, state = self.make()
            guard()
            query = Query(trace)
            guard.attach_authenticated_query(query)
            state.update(changed)
            with self.subTest(changed=changed), self.assertRaisesRegex(RuntimeError, 'Full original resource bounds'):
                guard()
            self.assertTrue(guard.poisoned)
            with self.assertRaises(RuntimeError):
                guard()
            self.assertEqual(query.requests, 1)

    def test_lock_failure_prevents_service_request(self):
        guard, trace, _ = self.make()
        def failed(*args):
            raise RuntimeError('lost lock')
        guard.low.lock_proof = failed
        with self.assertRaisesRegex(RuntimeError, 'lost lock'):
            guard()
        self.assertEqual(trace, [])

    def test_window_failure_prevents_service_request(self):
        guard, trace, _ = self.make()
        def failed():
            raise RuntimeError('window conflict')
        guard.low.clear_window = failed
        with self.assertRaisesRegex(RuntimeError, 'window conflict'):
            guard()
        self.assertEqual(trace, ['lock'])

    def test_transport_failure_is_not_retried(self):
        guard, trace, _ = self.make()
        guard()
        query = Query(trace)
        guard.attach_authenticated_query(query)
        def failed():
            query.requests += 1
            raise RuntimeError('connection lost')
        query.check = failed
        with self.assertRaisesRegex(RuntimeError, 'connection lost'):
            guard()
        with self.assertRaises(RuntimeError):
            guard()
        self.assertEqual(query.requests, 1)

    def test_close_is_once_and_prohibits_more_checks(self):
        guard, trace, _ = self.make()
        guard()
        query = Query(trace)
        guard.attach_authenticated_query(query)
        guard.close()
        guard.close()
        self.assertEqual(trace.count('query_close'), 1)
        with self.assertRaises(RuntimeError):
            guard()

    def test_missing_request_cannot_pass_accounting(self):
        guard, trace, _ = self.make()
        guard()
        query = Query(trace)
        guard.attach_authenticated_query(query)
        query.check = lambda: None
        guard()
        with self.assertRaises(RuntimeError):
            guard.accounting()

    def test_failed_final_dependency_audit_is_retained(self):
        guard, trace, _ = self.make()
        guard()
        query = Query(trace)
        guard.attach_authenticated_query(query)
        def failed():
            raise RuntimeError('dependency changed at close')
        query.close = failed
        with self.assertRaisesRegex(RuntimeError, 'dependency changed'):
            guard.close()
        self.assertTrue(guard.closed and guard.poisoned)


if __name__ == '__main__':
    unittest.main(verbosity=2)
