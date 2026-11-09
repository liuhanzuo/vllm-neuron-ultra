"""Synthetic responses only: no new service, process or bus connection."""
import unittest

import dory_saved_review_service_query_v1 as s


def unit(name='trn1-example.service', active='active', sub='running'):
    return (name, 'description', 'loaded', active, sub, '',
            '/org/freedesktop/systemd1/unit/example', 0, '', '/')


class Query(unittest.TestCase):
    def make(self, responses):
        calls, closes = [], []
        responses = iter(responses)

        def request(states):
            calls.append(states)
            item = next(responses)
            if isinstance(item, BaseException):
                raise item
            return item, s.SIGNATURE

        return s.PersistentServiceQuery(request, lambda: closes.append(True)), calls, closes

    def test_fresh_query_on_every_check_including_unchanged_empty_results(self):
        q, calls, _ = self.make([[], [], []])
        for _ in range(3):
            self.assertEqual(q.check(), [])
        self.assertEqual(calls, [s.STATES] * 3)
        self.assertEqual(q.requests, 3)

    def test_new_service_after_success_is_not_hidden_by_cache(self):
        q, calls, _ = self.make([[], [unit()]])
        q.check()
        with self.assertRaisesRegex(RuntimeError, 'Competing'):
            q.check()
        self.assertEqual(len(calls), 2)
        self.assertTrue(q.poisoned)

    def test_running_and_both_transitions_are_detected(self):
        for active, sub in [('active', 'running'), ('activating', 'start'),
                            ('deactivating', 'stop')]:
            q, _, _ = self.make([[unit(active=active, sub=sub)]])
            with self.subTest(active=active), self.assertRaisesRegex(RuntimeError, 'Competing'):
                q.check()

    def test_original_exclusion_returns_full_matching_list(self):
        q, _, _ = self.make([[unit()]])
        self.assertEqual(q.check(excluded=('trn1-example',)), ['trn1-example'])

    def test_exclusion_does_not_hide_another_service(self):
        q, _, _ = self.make([[unit(), unit('trn1-other.service')]])
        with self.assertRaisesRegex(RuntimeError, 'Competing'):
            q.check(excluded=('trn1-example',))

    def test_nonservice_and_other_prefix_have_original_selection_semantics(self):
        q, _, _ = self.make([[unit('trn1-example.scope'), unit('other.service')]])
        self.assertEqual(q.check(), [])

    def test_query_error_poisons_connection_and_prevents_retry(self):
        q, calls, _ = self.make([OSError('bus disconnected'), []])
        with self.assertRaises(OSError):
            q.check()
        with self.assertRaises(RuntimeError):
            q.check()
        self.assertEqual(len(calls), 1)

    def test_malformed_or_duplicate_row_is_rejected(self):
        for rows in [[unit()[:-1]], [unit(), unit()], [tuple([*unit()[:7], True, '', '/'])],
                     [unit(active='inactive', sub='dead')]]:
            q, _, _ = self.make([rows])
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                q.check()
            self.assertTrue(q.poisoned)

    def test_wrong_signature_is_rejected(self):
        q = s.PersistentServiceQuery(lambda states: ([], 's'), lambda: None)
        with self.assertRaises(RuntimeError):
            q.check()
        self.assertTrue(q.poisoned)

    def test_oversized_complete_response_is_rejected(self):
        rows = [unit(f'other-{i}.service') for i in range(s.MAX_ROWS + 1)]
        with self.assertRaises(RuntimeError):
            s.service_names(rows, s.SIGNATURE)
        row = list(unit())
        row[1] = 'x' * s.MAX_TEXT_BYTES
        with self.assertRaises(RuntimeError):
            s.service_names([tuple(row)], s.SIGNATURE)

    def test_close_is_once_and_subsequent_checks_are_rejected(self):
        q, calls, closes = self.make([[]])
        q.close()
        q.close()
        with self.assertRaises(RuntimeError):
            q.check()
        self.assertEqual((calls, closes), ([], [True]))

    def test_invalid_exclusion_cannot_trigger_a_bus_request(self):
        for excluded in ['trn1-example', ('trn1-example.service',), ('other',)]:
            q, calls, _ = self.make([[]])
            with self.subTest(excluded=excluded), self.assertRaises(RuntimeError):
                q.check(excluded)
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
