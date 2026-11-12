"""SOURCE/STDLIB MOCK: tiny synthetic files; no SDK/native/process/lock calls."""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from test_source import Storage, io, b


class CompilerStorage(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            spec, reviewed, control = Storage().fixture(temp)
            with patch.object(io, 'control_root', return_value=control), \
                    patch.object(b, 'control_root', return_value=control), \
                    patch.object(b, 'reviewed_budget', return_value=reviewed), \
                    patch.object(b, 'unchanged_inputs', side_effect=lambda spec, value: value), \
                    patch.object(io.time, 'sleep') as sleep:
                yield spec, reviewed, control, sleep

    def cache(self, spec):
        directory = Path(spec['case']['compiler_cache']) / 'neuronxcc-2.23.6484.0+3b612583' / (
            'MODULE_' + 'a' * 20 + '+' + 'b' * 8)
        directory.mkdir(parents=True)
        return directory

    def work(self, spec):
        directory = Path(spec['case']['compiler_workdir']) / 'context_encoding_model' / '_tp0_bk0'
        directory.mkdir(parents=True)
        return directory

    def vanish_on_stat(self, victim, operation):
        original = io.observed_stat
        changed = []
        def observe(path):
            if path == victim and not changed:
                changed.append(True); operation()
            return original(path)
        return patch.object(io, 'observed_stat', side_effect=observe)

    def test_sdk_atomic_cache_publish_recounts_final_file(self):
        for name in ('model.hlo_module.pb', 'model.neff', 'wrapped_neff.hlo'):
            with self.subTest(name=name), self.fixture() as (spec, reviewed, control, sleep):
                directory = self.cache(spec)
                final = directory / name
                temporary = directory / (name + '-' + os.uname().nodename + '-12345')
                temporary.write_bytes(b'synthetic' * 700)
                with self.vanish_on_stat(temporary, lambda: temporary.replace(final)):
                    result = io.storage(spec)
                self.assertTrue(final.is_file()); self.assertFalse(temporary.exists())
                expected, inventory = io.storage_scan(spec)
                self.assertEqual(result, expected)
                self.assertEqual(result['allocated_bytes'], sum(row[6] * 512 for row in inventory))
                sleep.assert_called_once_with(.01)

    def test_sdk_cache_unlock_recounts_complete_tree(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            directory = self.cache(spec)
            lock = directory / 'model.hlo_module.pb.lock'; lock.touch()
            (directory / 'model.neff').write_bytes(b'synthetic' * 700)
            with self.vanish_on_stat(lock, lock.unlink): result = io.storage(spec)
            self.assertEqual(result, io.storage_scan(spec)[0])
            sleep.assert_called_once_with(.01)

    def test_bucket_replacement_does_not_reuse_partial_usage(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            directory = self.work(spec); victim = directory / 'model' / 'graph.hlo'
            victim.parent.mkdir(); victim.write_bytes(b'old synthetic')
            def reset():
                shutil.rmtree(directory); directory.mkdir()
                (directory / 'model.synthetic.neff').write_bytes(b'new synthetic' * 900)
            with self.vanish_on_stat(victim, reset): result = io.storage(spec)
            self.assertEqual(result, io.storage_scan(spec)[0])
            self.assertGreater(result['classes']['compiler_workdir'], 9000)
            sleep.assert_called_once_with(.01)

    def test_layout_directory_disappears_during_enumeration(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            directory = Path(spec['case']['compiler_workdir']) / 'layout_opt'
            directory.mkdir(parents=True); original = Path.iterdir; changed = []
            def children(path):
                if path == directory and not changed:
                    changed.append(True); directory.rmdir(); directory.mkdir()
                    (directory / 'metaneff').write_bytes(b'synthetic')
                    raise FileNotFoundError('SDK directory reset during enumeration')
                return original(path)
            with patch.object(Path, 'iterdir', children): result = io.storage(spec)
            self.assertEqual(result, io.storage_scan(spec)[0]); sleep.assert_called_once_with(.01)

    def test_unknown_cache_missing_and_wrong_sdk_or_hostname_fail_immediately(self):
        for name in ('model.neff', 'arbitrary.lock', 'model.neff-wrong-host-12345',
                     'compile_flags.json', 'model.neff-' + os.uname().nodename + '-0'):
            with self.subTest(name=name), self.fixture() as (spec, reviewed, control, sleep):
                victim = self.cache(spec) / name; victim.write_bytes(b'synthetic')
                with self.vanish_on_stat(victim, victim.unlink):
                    with self.assertRaises(FileNotFoundError): io.storage(spec)
                sleep.assert_not_called()
        with self.fixture() as (spec, reviewed, control, sleep):
            wrong = Path(spec['case']['compiler_cache']) / 'neuronxcc-other' / ('MODULE_' + 'a'*20 + '+' + 'b'*8)
            wrong.mkdir(parents=True); victim = wrong / 'model.hlo_module.pb.lock'; victim.touch()
            with self.vanish_on_stat(victim, victim.unlink):
                with self.assertRaises(FileNotFoundError): io.storage(spec)
            sleep.assert_not_called()

    def test_unknown_workdir_branch_and_other_tree_missing_fail(self):
        for key, relative in (('compiler_workdir', 'unexpected/file.bin'),
                              ('compiler_workdir', 'context_encoding_model/_tp0_bk6/graph.neff'),
                              ('artifact', 'model.neff'), ('model_path', 'model.neff')):
            with self.subTest(key=key, relative=relative), self.fixture() as (spec, reviewed, control, sleep):
                victim = Path(spec['case'][key]) / relative
                victim.parent.mkdir(parents=True); victim.write_bytes(b'synthetic')
                with self.vanish_on_stat(victim, victim.unlink):
                    with self.assertRaises(FileNotFoundError): io.storage(spec)
                sleep.assert_not_called()

    def test_sealed_build_has_no_new_compiler_mutation_exception(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            root = Path(spec['output_root']); root.mkdir()
            (root / 'build-immutable-identities.json').write_text('{}')
            lock = self.cache(spec) / 'model.hlo_module.pb.lock'; lock.touch()
            with self.vanish_on_stat(lock, lock.unlink):
                with self.assertRaises(FileNotFoundError): io.storage(spec)
            sleep.assert_not_called()

    def test_symlink_and_hardlink_in_compiler_trees_still_fail(self):
        for tree in ('compiler_cache', 'compiler_workdir'):
            with self.subTest(tree=tree), self.fixture() as (spec, reviewed, control, sleep):
                directory = self.cache(spec) if tree == 'compiler_cache' else self.work(spec)
                target = Path(spec['output_root']).parent / 'outside'; target.write_bytes(b'synthetic')
                entry = directory / 'model.neff'; entry.symlink_to(target)
                with self.assertRaises(ValueError): io.storage(spec)
                entry.unlink(); os.link(target, entry)
                with self.assertRaises(ValueError): io.storage(spec)
                sleep.assert_not_called()

    def test_each_cap_checked_again_after_topology_race(self):
        for field in ('single_file_bytes', 'allocated_bytes', 'entries'):
            with self.subTest(field=field), self.fixture() as (spec, reviewed, control, sleep):
                directory = self.work(spec); victim = directory / 'old'; victim.write_bytes(b'synthetic')
                def reset():
                    victim.unlink(); (directory / 'new').write_bytes(b'x' * 9000)
                    reviewed['classes']['compiler_workdir'][field] = 1
                with self.vanish_on_stat(victim, reset):
                    with self.assertRaisesRegex(ValueError, 'cap'): io.storage(spec)
                sleep.assert_called_once_with(.01)

    def test_live_log_and_regular_compiler_file_growth_converges(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            output = self.work(spec) / 'graph.neff'; output.write_bytes(b'synthetic')
            log = control / 'shared-build' / 'phase.log'; log.parent.mkdir(parents=True)
            log.write_bytes(b'synthetic log')
            original = io.storage_scan; scans = []
            def scan(value):
                if not scans:
                    scans.append(None); raise io.CompilerTreeRace('injected prior SDK reset')
                for path in (output, log):
                    with path.open('ab') as stream: stream.write(b'x' * 5000)
                observed = original(value); scans.append(observed); return observed
            with patch.object(io, 'storage_scan', side_effect=scan): result = io.storage(spec)
            self.assertEqual(len(scans), 3); self.assertEqual(result, scans[-1][0])
            for kind in ('compiler_workdir', 'build_auxiliary'):
                self.assertGreater(scans[-1][0]['classes'][kind], scans[-2][0]['classes'][kind])
            sleep.assert_called_once_with(.01)

    def test_growth_past_cap_on_second_complete_scan_is_rejected(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            output = self.work(spec) / 'graph.neff'; output.write_bytes(b'synthetic')
            original = io.storage_scan; calls = []
            def scan(value):
                calls.append(True)
                if len(calls) == 1: raise io.CompilerTreeRace('injected prior SDK reset')
                if len(calls) == 3:
                    output.write_bytes(b'x' * 10000)
                    reviewed['classes']['compiler_workdir']['single_file_bytes'] = 8192
                return original(value)
            with patch.object(io, 'storage_scan', side_effect=scan):
                with self.assertRaisesRegex(ValueError, 'single.*cap'): io.storage(spec)
            self.assertEqual(len(calls), 3)

    def test_topology_churn_between_complete_scans_restarts_both(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            lock = self.cache(spec) / 'model.hlo_module.pb.lock'; lock.touch()
            original = io.storage_scan; calls = []
            def scan(value):
                calls.append(True)
                if len(calls) == 1: raise io.CompilerTreeRace('injected prior SDK reset')
                if len(calls) == 3: lock.unlink()
                return original(value)
            with patch.object(io, 'storage_scan', side_effect=scan): result = io.storage(spec)
            self.assertEqual(len(calls), 5); self.assertEqual(sleep.call_count, 2)
            self.assertEqual(result, original(spec)[0])

    def test_unknown_disappearance_between_complete_scans_is_not_swallowed(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            directory = Path(spec['case']['artifact']); directory.mkdir()
            victim = directory / 'model.neff'; victim.write_bytes(b'synthetic')
            original = io.storage_scan; calls = []
            def scan(value):
                calls.append(True)
                if len(calls) == 1: raise io.CompilerTreeRace('injected prior SDK reset')
                if len(calls) == 3: victim.unlink()
                return original(value)
            with patch.object(io, 'storage_scan', side_effect=scan):
                with self.assertRaisesRegex(ValueError, 'unknown storage topology'): io.storage(spec)
            self.assertEqual(len(calls), 3); sleep.assert_called_once_with(.01)

    def test_unknown_inode_replacement_between_complete_scans_fails(self):
        with self.fixture() as (spec, reviewed, control, sleep):
            directory = Path(spec['case']['artifact']); directory.mkdir()
            victim = directory / 'model.neff'; victim.write_bytes(b'synthetic')
            original = io.storage_scan; calls = []
            def scan(value):
                calls.append(True)
                if len(calls) == 1: raise io.CompilerTreeRace('injected prior SDK reset')
                if len(calls) == 3:
                    replacement = directory / 'replacement'
                    replacement.write_bytes(b'synthetic'); replacement.replace(victim)
                return original(value)
            with patch.object(io, 'storage_scan', side_effect=scan):
                with self.assertRaisesRegex(ValueError, 'unknown storage topology'): io.storage(spec)
            self.assertEqual(len(calls), 3); sleep.assert_called_once_with(.01)

    def test_eight_attempt_bound_and_terminal_exact_inventory_preserved(self):
        with patch.object(io, 'storage_scan', side_effect=io.CompilerTreeRace('never settles')) as scan, \
                patch.object(io.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ValueError, 'finite storage retries'): io.storage({})
            self.assertEqual(scan.call_count, 8); self.assertEqual(sleep.call_count, 7)
        with patch.object(io, 'storage_scan', side_effect=[({'bytes': 1}, []), ({'bytes': 2}, [])]), \
                patch.object(io.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ValueError, 'terminal output inventory'): io.storage({}, terminal=True)
            sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
