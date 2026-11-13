"""SOURCE/STDLIB MOCK: extracted original verifier and synthetic partitions only."""
import ast
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from test_source import HERE, Q, c, io
from qwen35_package_serial_runtime_v9 import source_text as subject

ORIGINAL = Q / 'minicpm5_layer1_operator_sdk_sources_v1.py'
SOURCE_MODULE = 'research.minicpm5_layer1_operator_sdk_sources_v1'


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


class Fixture:
    def __init__(self):
        self.events = []
        self.special = subject.HOST_SOURCE_ENVELOPE[0]
        self.payloads = {self.special: b'synthetic exact source', '/mock/ordinary.py': b'ordinary',
                         '/mock/literal-mini-catalog.json': b'mini catalog'}
        self.alias = '/mock/alias.json'
        self.alias_sha = hashlib.sha256(b'logical alias').hexdigest()
        self.python = {f'/mock/python-{i}.pyc': str(i) * 64 for i in range(1, 4)}
        self.native = {f'/mock/native-{i}.so': '4' * 64 for i in range(9)}
        self.historical = {f'/mock/historical-{i}.pt': '5' * 64 for i in range(20)}
        self.sdk = types.ModuleType(SOURCE_MODULE)
        self.sdk.require = require
        self.sdk.text = Mock(side_effect=self.text)
        self.sdk.aliases = types.SimpleNamespace(
            validate_spec=Mock(side_effect=lambda value: self.events.append(('alias_validate', value))),
            source_sha=Mock(side_effect=self.alias_hash))
        self.sdk.metadata = types.SimpleNamespace(
            python_source_bytes=Mock(side_effect=self.python_bytes),
            query_sources=types.SimpleNamespace(OBSERVATION={'mock': 'observation'},
                native_pins=Mock(side_effect=lambda value: self.native.copy())))
        constants = dict(digest=digest, require=require, aliases=self.sdk.aliases, metadata=self.sdk.metadata,
            PARTITION_SCHEMA='synthetic-partition', R12_SPEC={'mock': 'r12'},
            CATEGORIES=('text_sha256', 'historical_payload_sha256', 'python_bytes_sha256', 'query_native_binary_sha256'))
        tree = ast.Module(body=[function(ORIGINAL, 'check_partition'), function(ORIGINAL, 'verify_text_sources')], type_ignores=[])
        constants.update(hashlib=hashlib, text=self.sdk.text)
        exec(compile(tree, str(ORIGINAL), 'exec'), constants)
        self.original_verify = constants['verify_text_sources']
        self.sdk.check_partition = Mock(side_effect=constants['check_partition'])
        self.partition = dict(schema='synthetic-partition', r12_spec={'mock': 'r12'},
            query_observation_ref={'mock': 'observation'}, query_observation={'mock': 'native pins'},
            metadata_aliases={self.alias: {'mock': 'alias binding'}}, categories=dict(
                text_sha256={**{p: hashlib.sha256(raw).hexdigest() for p, raw in self.payloads.items()}, self.alias: self.alias_sha},
                historical_payload_sha256=self.historical.copy(), python_bytes_sha256=self.python.copy(),
                query_native_binary_sha256=self.native.copy()))
        self.reseal()
        self.guard = Mock(side_effect=lambda: self.events.append(('guard',)))
        self.host = Mock(side_effect=self.host_bytes)

    def reseal(self):
        categories = self.partition['categories']
        self.partition['partition_sha256'] = digest(categories)
        self.partition['source_map_sha256'] = digest({p: sha for rows in categories.values() for p, sha in rows.items()})

    def text(self, path, expected=None):
        self.events.append(('text', path, expected))
        assert path not in self.native and path not in self.historical and path not in self.python
        return self.payloads[path]

    def host_bytes(self, path, expected):
        self.events.append(('host', path, expected))
        assert path == self.special
        return self.payloads[path]

    def alias_hash(self, path, aliases, reader):
        self.events.append(('alias', path))
        assert reader is self.sdk.text and aliases is self.partition['metadata_aliases']
        reader('/mock/literal-mini-catalog.json', hashlib.sha256(self.payloads['/mock/literal-mini-catalog.json']).hexdigest())
        return self.alias_sha

    def python_bytes(self, path, sha):
        self.events.append(('python', path, sha))
        assert self.python[path] == sha

    @contextmanager
    def installed(self):
        parent = types.ModuleType('research')
        parent.minicpm5_layer1_operator_sdk_sources_v1 = self.sdk
        with patch.dict(sys.modules, {'research': parent, SOURCE_MODULE: self.sdk}), \
             patch.object(subject, 'host_source_bytes', self.host), \
             patch('builtins.open', side_effect=AssertionError('unexpected file IO')), \
             patch('os.open', side_effect=AssertionError('unexpected fd IO')), \
             patch('subprocess.Popen', side_effect=AssertionError('unexpected process')), \
             patch('fcntl.flock', side_effect=AssertionError('unexpected lock')):
            yield

    def run(self):
        with self.installed():
            return subject.verify_text_sources(self.partition, self.guard)


class CompleteVerifier(unittest.TestCase):
    def test_full_partition_and_exact_receipt_preserved(self):
        f = Fixture(); before = copy.deepcopy(f.partition); result = f.run()
        f.sdk.check_partition.assert_called_once_with(f.partition)
        self.assertEqual(f.partition, before)
        self.assertEqual(result, dict(complete_text_verification=True,
            source_map_sha256=f.partition['source_map_sha256'], partition_sha256=f.partition['partition_sha256'],
            historical_payload_bytes_read=0, native_bytes_read=0))
        self.assertEqual(len(f.historical), 20); self.assertEqual(len(f.native), 9)
        self.assertEqual(f.guard.call_count, 2 + len(f.payloads) + 1 + 3)
        f.sdk.aliases.validate_spec.assert_called_once()

    def test_only_exact_direct_source_uses_helper_all_other_text_and_aliases_original(self):
        f = Fixture(); original_reader = f.sdk.text; f.run()
        f.host.assert_called_once_with(f.special, f.partition['categories']['text_sha256'][f.special])
        self.assertIs(f.sdk.text, original_reader)
        self.assertNotIn(f.special, [call.args[0] for call in f.sdk.text.call_args_list])
        self.assertEqual([call.args[0] for call in f.sdk.text.call_args_list],
            ['/mock/ordinary.py', '/mock/literal-mini-catalog.json', '/mock/literal-mini-catalog.json'])
        f.sdk.aliases.source_sha.assert_called_once_with(f.alias, f.partition['metadata_aliases'], original_reader)

    def test_three_original_python_bytes_all_checked_after_text_with_guards(self):
        f = Fixture(); f.run()
        self.assertEqual([call.args for call in f.sdk.metadata.python_source_bytes.call_args_list], list(f.python.items()))
        for i, event in enumerate(f.events):
            if event[0] in ('host', 'python'):
                self.assertEqual(f.events[i - 1], ('guard',))
        self.assertEqual(f.events[-1], ('guard',))

    def test_original_and_adapter_have_same_guard_order_calls_and_result_except_transport(self):
        f = Fixture(); actual = f.run(); events = f.events.copy()
        f.events.clear()
        expected = f.original_verify(f.partition, f.guard)
        normalized = [('text', *e[1:]) if e[0] == 'host' else e for e in events]
        self.assertEqual(actual, expected); self.assertEqual(normalized, f.events)

    def test_full_original_partition_rejects_digest_overlap_schema_and_native_binding(self):
        for mode in ('digest', 'overlap', 'schema', 'native'):
            f = Fixture()
            if mode == 'digest': f.partition['source_map_sha256'] = '0' * 64
            elif mode == 'overlap':
                f.partition['categories']['historical_payload_sha256'][f.special] = '1' * 64; f.reseal()
            elif mode == 'schema': f.partition['schema'] = 'bad'
            else:
                f.partition['categories']['query_native_binary_sha256'].pop(next(iter(f.native))); f.reseal()
            with self.subTest(mode=mode), self.assertRaises(ValueError): f.run()
            f.host.assert_not_called(); f.sdk.text.assert_not_called(); f.sdk.metadata.python_source_bytes.assert_not_called()

    def test_original_partition_alias_validation_failure_prevents_any_text(self):
        f = Fixture(); f.sdk.aliases.validate_spec.side_effect = ValueError('alias binding')
        with self.assertRaisesRegex(ValueError, 'alias binding'): f.run()
        f.host.assert_not_called(); f.sdk.text.assert_not_called()

    def test_both_direct_hash_mismatch_and_alias_hash_mismatch_rejected(self):
        for mode in ('direct', 'special', 'alias'):
            f = Fixture()
            if mode == 'direct': f.payloads['/mock/ordinary.py'] = b'changed'
            elif mode == 'special': f.host.side_effect = None; f.host.return_value = b'changed'
            else: f.sdk.aliases.source_sha.side_effect = None; f.sdk.aliases.source_sha.return_value = '0' * 64
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'Unchanged exact text source'): f.run()
            f.sdk.metadata.python_source_bytes.assert_not_called()

    def test_original_reader_or_exact_helper_failures_propagate(self):
        for mode in ('original', 'helper'):
            f = Fixture(); target = f.sdk.text if mode == 'original' else f.host
            target.side_effect = ValueError('reader failed')
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'reader failed'): f.run()

    def test_python_fixture_failures_propagate(self):
        f = Fixture(); f.sdk.metadata.python_source_bytes.side_effect = ValueError('python fixture')
        with self.assertRaisesRegex(ValueError, 'python fixture'): f.run()

    def test_guard_failure_prevents_reads_and_invalid_guard_rejected(self):
        f = Fixture(); f.guard.side_effect = ValueError('guard failed')
        with self.assertRaisesRegex(ValueError, 'guard failed'): f.run()
        f.sdk.check_partition.assert_not_called(); f.host.assert_not_called(); f.sdk.text.assert_not_called()
        f.guard = None
        with self.assertRaisesRegex(ValueError, 'Explicit observer'): f.run()

    def test_similar_or_noncanonical_paths_delegate_to_unchanged_original_reader(self):
        f = Fixture()
        for path in (f.special + '.other', f.special.replace('-r10-', '-r11-'),
                     str(Path(f.special).parent) + '/nested/../' + Path(f.special).name):
            f.payloads[path] = b'ordinary'
            self.assertEqual(subject.text_bytes(f.sdk, path, 'a' * 64), b'ordinary')
        f.host.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
