"""Source/stdlib/mock only: warning accounting and preserved final-exit gate."""
import ast
from contextlib import nullcontext
import copy
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

# The existing suite installs a deny-native import finder before production
# modules. No model, tensor, RECORD, process, service, or lock operations occur.
import check_lfm3_nki05_full_cpu_successor_v3 as integration
known = integration.known_warnings
science = integration.science
runner = integration.runner
ROOT = integration.ROOT


def emit(journal, signature, **changes):
    values = dict(message=signature['message'], category=SyntaxWarning,
                  filename=signature['filename'], lineno=signature['lineno'])
    values.update(changes)
    return journal.showwarning(**values)


class Tests(unittest.TestCase):
    def journal(self, mode, matcher=lambda row:True):
        calls = []
        return known.KnownWarningLog(mode, lambda *a, **k:calls.append((a, k)), matcher), calls

    def test_01_worker_retains_first_exact_original_warning_and_all_counts(self):
        journal, calls = self.journal('worker')
        for signature in known.KNOWN:
            for _ in range(5): emit(journal, signature)
        self.assertEqual(len(calls), 4)
        for call, signature in zip(calls, known.KNOWN):
            self.assertEqual(call, ((signature['message'], SyntaxWarning, signature['filename'], signature['lineno']),
                                    dict(file=None, line=None)))
        self.assertTrue(all((row['observed'], row['forwarded'], row['deduplicated']) == (5, 1, 4)
                            for row in journal.snapshot()['known']))

    def test_02_batch_records_complete_signatures_and_counts_without_repeated_stderr(self):
        journal, calls = self.journal('batch')
        for signature in known.KNOWN:
            for _ in range(7): emit(journal, signature)
        self.assertEqual(calls, [])
        receipt = journal.snapshot()
        for row, signature in zip(receipt['known'], known.KNOWN):
            self.assertEqual({key:row[key] for key in signature}, signature)
            self.assertEqual((row['observed'], row['forwarded'], row['deduplicated']), (7, 0, 7))
        self.assertLess(len(runner.encode(receipt)), 4096)

    def test_03_changed_category_message_filename_lineno_line_and_explicit_file_forward(self):
        class OtherSyntax(SyntaxWarning): pass
        signature = known.KNOWN[0]
        changes = [dict(category=RuntimeWarning), dict(category=OtherSyntax),
                   dict(message=signature['message'] + ' changed'),
                   dict(filename='/different/source.py'), dict(lineno=signature['lineno'] + 1),
                   dict(lineno=True), dict(line='changed source text'), dict(file=io.StringIO())]
        for change in changes:
            journal, calls = self.journal('batch')
            emit(journal, signature, **change)
            self.assertEqual(len(calls), 1)
            self.assertEqual(journal.snapshot()['other_forwarded'], 1)
            self.assertTrue(all(row['observed'] == 0 for row in journal.snapshot()['known']))

    def test_04_real_source_hashes_match_and_changed_identity_is_never_suppressed(self):
        for signature in known.KNOWN:
            self.assertTrue(known.source_matches(signature))
            self.assertFalse(known.source_matches(dict(signature, source_sha256='0' * 64)))
            self.assertFalse(known.source_matches(dict(signature, source_line='changed source')))
        journal, calls = self.journal('batch', lambda row:False)
        emit(journal, known.KNOWN[0])
        self.assertEqual(len(calls), 1)
        self.assertEqual(journal.snapshot()['known'][0]['observed'], 0)

    def test_05_unknown_warnings_and_forwarding_errors_preserve_original_call_and_exception(self):
        journal, calls = self.journal('batch')
        message = RuntimeWarning('unrelated warning')
        journal.showwarning(message, RuntimeWarning, '/unknown.py', 19, line='original source')
        self.assertIs(calls[0][0][0], message)
        self.assertEqual(calls[0][1], dict(file=None, line='original source'))
        failure = OSError('synthetic output write failure')
        def forward(*args, **kwargs): raise failure
        journal = known.KnownWarningLog('batch', forward, lambda row:False)
        with self.assertRaises(OSError) as raised: emit(journal, known.KNOWN[0])
        self.assertIs(raised.exception, failure)

    def test_06_bootstrap_precedes_local_closure_and_imports_are_inert(self):
        before, filters = warnings.showwarning, list(warnings.filters)
        for name, argv in [('imported_module', ['x', '--mode', 'batch']),
                           ('__main__', ['x', '--mode', 'outer']),
                           ('__main__', ['x', '--mode', 'unknown']),
                           ('__main__', ['x'])]:
            self.assertIsNone(known.bootstrap(name, argv))
            self.assertIs(warnings.showwarning, before)
        tree = ast.parse(Path(runner.__file__).read_text())
        hook = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
                    and ast.unparse(n.targets[0]) == 'WARNING_LOG')
        local = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Import)
                     and n.names[0].name == 'minicpm5_layer0_execution_protocol_v4')
        self.assertLess(hook, local)
        self.assertEqual(warnings.filters, filters)
        self.assertIsNone(runner.WARNING_LOG)

    def test_07_actual_stdlib_warning_dispatch_and_hook_replacement_rejected(self):
        signature = known.KNOWN[0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', SyntaxWarning)
            previous = warnings.showwarning
            journal = known.bootstrap('__main__', ['source-mock.py', '--mode', 'batch'])
            try:
                warnings.warn_explicit(signature['message'], SyntaxWarning, signature['filename'], signature['lineno'])
                self.assertEqual(journal.snapshot()['known'][0]['observed'], 1)
                warnings.showwarning = previous
                with self.assertRaisesRegex(RuntimeError, 'must remain active'): journal.snapshot()
            finally:
                warnings.showwarning = previous

    def test_08_receipt_rejects_omissions_forged_signature_counts_role_and_boolean_count(self):
        journal, _ = self.journal('batch')
        emit(journal, known.KNOWN[0])
        original = journal.snapshot()
        mutations = [lambda x:x['known'].pop(), lambda x:x['known'][0].update(source_sha256='0'*64),
                     lambda x:x['known'][0].update(message='other'), lambda x:x['known'][0].update(source_line='other'),
                     lambda x:x['known'][0].update(deduplicated=0), lambda x:x['known'][0].update(observed=True),
                     lambda x:x['known'][0].update(observed=known.MAX_COUNT + 1),
                     lambda x:x['known'][0].update(forwarded=1), lambda x:x.update(mode='worker'),
                     lambda x:x.update(other_forwarded=-1), lambda x:x.update(scope='all_python_warnings')]
        for mutate in mutations:
            value = copy.deepcopy(original); mutate(value)
            with self.assertRaises(RuntimeError): known.validate_receipt(value, 'batch')

    def test_09_874_fresh_batch_journals_keep_log_repetition_bounded_without_extra_files(self):
        repeated_calls = []
        for _ in range(874):
            journal = known.KnownWarningLog('batch', lambda *a, **k:repeated_calls.append(a), lambda row:True)
            for signature in known.KNOWN: emit(journal, signature)
            self.assertEqual(sum(row['observed'] for row in journal.snapshot()['known']), 4)
        self.assertEqual(repeated_calls, [])
        worker, calls = self.journal('worker')
        for signature in known.KNOWN: emit(worker, signature)
        rendered = ''.join(warnings.formatwarning(*args, line=known.KNOWN[i]['source_line'])
                           for i,(args,kwargs) in enumerate(calls))
        self.assertLess(len(rendered.encode()), 4096)
        self.assertEqual((runner.FILE_CAP, runner.OUTPUT_CAP, runner.MAX_FILES), (256*2**10, 2**30, 2048))
        self.assertEqual(science.protocol_manifest()['total_children'], 874)

    def test_10_passed_saved_report_does_not_mask_actual_exit120(self):
        with tempfile.TemporaryDirectory(prefix='lfm-warning-pure-') as folder:
            root = Path(folder); host = integration.Host(root)
            target = root/'new_fixture'/'extended_cpu'; target.mkdir(parents=True)
            path = science.result_path(root, 'new_fixture', 'extended_cpu', 1319, 1320)
            host.once(path, dict(complete=True, passed=True, scientific_status='synthetic completed report'))
            child = SimpleNamespace(pid=123, returncode=120, wait=lambda **kwargs:120)
            owner = dict(pid=123, start_ticks='pure-mock')
            ref = dict(path=str(root/'admission.json'), sha256='0'*64)
            with patch.object(science, 'legacy_child_binding', lambda *args:nullcontext()), \
                 patch.object(science.subprocess, 'Popen', return_value=child), \
                 patch.object(science.legacy, 'child_identity', return_value=owner), \
                 patch.object(science.legacy, 'stop_child') as stop, \
                 patch.object(science, 'validate_report') as validate:
                outcome, error = science.run_one_batch(host, {}, ref, host.admission, {}, root,
                    'new_fixture', 'extended_cpu', 1319, 1320, lambda:True, 1800)
            self.assertIsInstance(error, RuntimeError)
            self.assertEqual(outcome['returncode'], 120)
            self.assertFalse(outcome['passed'])
            self.assertEqual(outcome['report'], host.reference(path))
            self.assertTrue(host.read(path)['passed'])
            validate.assert_not_called(); stop.assert_called_once_with(child, owner)

    def test_11_original_io_audit_accepts_named_source_reads_but_rejects_numeric_wrapper(self):
        tree = ast.parse(Path(science.legacy.p.__file__).read_text())
        outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit_child')
        audit = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == 'audit')
        env = dict(Path=Path, os=os, stat=stat, DEST=Path('/source-only-output'), expected={}, allowed_writes=set())
        exec(compile(ast.Module(body=[audit], type_ignores=[]), '<isolated-existing-audit>', 'exec'), env)
        filename = known.KNOWN[0]['filename']
        env['audit']('open', (filename, 'r', os.O_RDONLY | os.O_CLOEXEC))
        env['audit']('open', (filename, None, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW))
        with self.assertRaises(AssertionError): env['audit']('open', (55, 'r', os.O_RDONLY))
        source = Path(known.__file__).read_text()
        self.assertNotIn('os.fdopen(', source)
        self.assertTrue(known.source_matches(known.KNOWN[0]))

    def test_12_full_gate_rejects_missing_or_bad_worker_warning_receipt(self):
        with tempfile.TemporaryDirectory(prefix='lfm-warning-pure-') as folder:
            root = Path(folder); host, ref, calls = integration.mock_coordinator(root)
            result = host.read(root/'diagnostic-result.json')
            science.require_full_success(result)
            result['warning_log']['known'][0]['source_sha256'] = '0'*64
            with self.assertRaises(RuntimeError): science.require_full_success(result)
            result.pop('warning_log')
            with self.assertRaises(KeyError): science.require_full_success(result)
        self.assertFalse(any(name.split('.')[0] in {'numpy','ml_dtypes','torch','nki','neuronxcc'} for name in sys.modules))


if __name__ == '__main__': unittest.main(verbosity=2)
