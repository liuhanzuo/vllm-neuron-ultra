"""Independent source/memory controls: synthetic inputs, no runtime operations."""
import ast
import copy
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import types
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
OPENED_SOURCE = set()


def restrict(event, args):
    if event in ('subprocess.Popen', 'os.system', 'fcntl.flock', 'fcntl.lockf') or event.startswith('socket.'):
        raise AssertionError('Forbidden actual runtime operation: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = Path(args[0])
        if path.suffix not in ('.py', '.pyc', '.so'):
            raise AssertionError('Only source/module reads allowed: ' + str(path))
        OPENED_SOURCE.add(str(path))


sys.addaudithook(restrict)
import dory_ubuntu_package_sources_v4 as discovery


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def synthetic_inputs():
    refs = {path: dict(path=path, bytes=1, sha256='e' * 64)
            for targets in discovery.TARGETS.values() for path, _ in targets
            if path != discovery.PREFIX + 'libgomp.so.1.0.0'}
    gaps = dict(system_package_files=[dict(path=path, observed_reference=ref) for path, ref in refs.items()])
    ended = dict(stage_success_observed=True, actual_tool=dict(exit_code=0),
                 parent_owner_gone=True, child_owner_gone=True, child_group_empty=True,
                 complete_events_audit=dict(complete=True),
                 evidence={'metadata-result.json': dict(sha256='e' * 64)})
    raw = dict(gaps=encoded(gaps), exit=encoded(ended))
    recorder = dict(actual_tool=dict(exit_code=0), owner_gone=True, independent_saved_audit_complete=True,
                    result=dict(sha256=discovery.sha(raw['exit'])))
    raw['recorder'] = encoded(recorder)
    nxt = dict(independent_saved_audit_pending=False,
               independent_saved_audit=dict(sha256=discovery.sha(raw['exit'])),
               independent_saved_audit_recorder_exit=dict(sha256=discovery.sha(raw['recorder'])),
               source=dict(sha256='e' * 64), packages=[dict(package=name, version=version,
                   architecture='amd64', observed_targets=[path for path, _ in discovery.TARGETS[name]],
                   saved_package_names=['/' + member for _, member in discovery.TARGETS[name]])
                   for name, version in discovery.PACKAGE_VERSIONS.items()])
    raw['next'] = encoded(nxt)
    return raw


INPUTS = synthetic_inputs()
ORIGINAL_INPUT_PINS = copy.deepcopy(discovery.INPUT_PINS)
discovery.INPUT_PINS = {key: (name, discovery.sha(INPUTS[key]))
                        for key, (name, _) in ORIGINAL_INPUT_PINS.items()}

# The author's fixture module normally reads four original JSON files during
# import. Compile its unchanged source definitions while replacing ONLY that
# top-level INPUTS assignment with the synthetic authenticated byte mapping.
name = 'check_dory_ubuntu_package_sources_v1'
path = Q / (name + '.py')
tree = ast.parse(path.read_bytes(), filename=str(path))
tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == 'INPUTS' for target in node.targets))]
for node in ast.walk(tree):
    if isinstance(node, ast.alias) and node.name == 'dory_ubuntu_package_sources_v1':
        node.name = 'dory_ubuntu_package_sources_v4'
fixture_module = types.ModuleType(name)
fixture_module.__file__ = str(path)
fixture_module.INPUTS = INPUTS
sys.modules[name] = fixture_module
exec(compile(tree, str(path), 'exec'), fixture_module.__dict__)

import check_record_dory_ubuntu_metadata_exit_v5 as authored
import record_dory_ubuntu_metadata_exit_v5 as recorder
import dory_ubuntu_metadata_saved_audit_v5 as audit


class FixedDateTime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        value = cls(2026, 11, 11, 12, tzinfo=dt.timezone.utc)
        return value if tz is not None else value.replace(tzinfo=None)


def event_result(fixture, events, success=False, truncate=False):
    out, refs, values, _, files, trust, authority = fixture
    raw = b''.join(encoded(row) + b'\n' for row in events)
    zipped = gzip.compress(raw, mtime=0)
    if truncate:
        zipped = zipped[:-8]
    ref = dict(path=str(out / 'events.jsonl.gz'), bytes=len(zipped), sha256=discovery.sha(zipped))
    values['worker-result.json']['events'] = dict(ref, raw_bytes=len(raw), raw_sha256=discovery.sha(raw), events=len(events))
    files['events.jsonl.gz'] = zipped
    with patch.object(recorder, 'OUT', out), patch.object(recorder, 'stable_open', return_value=(io.BytesIO(zipped), None)), \
         patch.object(recorder, 'unchanged'):
        return recorder.event_audit(ref, lambda: None, values, refs, success)


class Independent(unittest.TestCase):
    def test_full_replay_reuses_actual_parser_and_original_discovery(self):
        fixture = authored.fixture()
        with patch.object(audit.adapter, 'parse_status', wraps=audit.adapter.parse_status) as parsed, \
             patch.object(audit.discovery, 'discover', wraps=audit.discovery.discover) as replayed:
            result = authored.run_replay(fixture)
        self.assertEqual(len(audit.NESTED), 27)
        self.assertEqual(len(audit.schedule()), 19)
        self.assertEqual(parsed.call_count, 3)
        self.assertEqual(replayed.call_count, 1)
        self.assertTrue(result['entire_discovery_reproduced'])
        self.assertFalse(result['signature_verification_rerun'])
        self.assertFalse(result['package_content_authenticated'])
        self.assertEqual(result['unresolved_closed'], 0)

    def test_rebound_bad_signature_status_and_signed_index_bytes_rejected(self):
        variants = ('bad_signature', 'unknown_primary', 'weak_hash', 'missing_lf', 'signed_index')
        for kind in variants:
            with self.subTest(kind=kind):
                f = authored.fixture()
                _, refs, values, _, files, _, _ = f
                name = 'signed-metadata/verify-1-status.txt'
                raw = files[name]
                if kind == 'bad_signature':
                    raw += b'[GNUPG:] BADSIG 0000000000000000 invalid\n'
                elif kind == 'unknown_primary':
                    raw = raw.replace(audit.binding.TRUST['primary_fingerprints'][0].encode(), b'F' * 40)
                elif kind == 'weak_hash':
                    raw = raw.replace(b' 0 4 0 1 10 00 ', b' 0 4 0 1 2 00 ')
                elif kind == 'missing_lf':
                    raw = raw[:-1]
                else:
                    name = next(n for n in files if n.endswith('main-Packages.xz'))
                    raw = files[name][:-1] + bytes([files[name][-1] ^ 1])
                files[name] = raw
                refs[name].update(bytes=len(raw), sha256=discovery.sha(raw))
                adapter = values['signed-metadata/sources-result.json']['adapter']
                adapter['adapter_output_allocated_bound'] = 4096 + sum(((r['bytes'] + 4095) // 4096) * 4096 for r in adapter['metadata_files'])
                with self.assertRaises((ValueError, RuntimeError)):
                    authored.run_replay(f)

    def test_every_prefix_failure_and_truncated_success_are_not_promoted(self):
        for count in range(20):
            with self.subTest(count=count):
                f = authored.fixture()
                result = event_result(f, f[3][:count])
                self.assertEqual(result['accepted_event_prefix'], count)
                self.assertFalse(result['stage_success_claimed'])
                self.assertFalse(result['complete_workflow_claimed'])
                if count < 19:
                    checker = audit.Events(f[0], f[1])
                    for row in f[3][:count]:
                        checker.accept(row)
                    with self.assertRaises(RuntimeError):
                        audit.replay_saved(checker, f[2], f[1], INPUTS, lambda n, c: f[4][n], lambda: None)
        f = authored.fixture()
        f[3][3]['returncode'] = 2
        result = event_result(f, f[3][:4])
        self.assertTrue(result['negative_or_partial_verifier_retained'])
        self.assertFalse(result['stage_success_claimed'])
        f = authored.fixture()
        result = event_result(f, f[3], truncate=True)
        self.assertFalse(result['complete'])
        self.assertFalse(result['complete_workflow_claimed'])
        with self.assertRaises(RuntimeError):
            event_result(f, f[3], success=True, truncate=True)

    def test_complete_inventory_counts_directories_and_rejects_extra_names(self):
        out = Path('/mock/inventory')
        nested = out / 'signed-metadata'
        data = {out / name: b'{}' if name.endswith('.json') else b'x' for name in audit.NESTED}
        data[out / 'worker.log'] = b'x'
        directories = {out: [nested, out / 'worker.log'], nested: sorted(p for p in data if p.parent == nested)}
        def lstat(path):
            return types.SimpleNamespace(st_mode=stat.S_IFDIR if path in directories else stat.S_IFREG,
                st_size=8192 if path == out else 4096 if path == nested else len(data[path]),
                st_blocks=16 if path == out else 8)
        def run():
            with patch.object(recorder, 'OUT', out), patch.object(Path, 'resolve', lambda p: p), \
                 patch.object(Path, 'lstat', lstat), patch.object(Path, 'iterdir', lambda p: iter(directories[p])), \
                 patch.object(recorder, 'stable_open', side_effect=lambda p, c: (io.BytesIO(data[p]), None)), \
                 patch.object(recorder, 'unchanged'):
                return recorder.output_inventory(lambda: None)
        refs, _, allocated, errors = run()
        self.assertEqual(len(refs), 28)
        self.assertEqual(allocated, 8192 + 4096 + 28 * 4096)
        self.assertEqual(errors, {})
        with patch.dict(recorder.c.INVENTORY, output_bytes=allocated - 1), self.assertRaises(RuntimeError):
            run()
        extra = nested / 'unreviewed.deb'
        data[extra] = b'x'
        directories[nested].append(extra)
        with self.assertRaisesRegex(RuntimeError, 'Known finite'):
            run()

    def test_actual_exit_one_cannot_be_promoted_by_success_shaped_output(self):
        owner = dict(pid=10, start_ticks='100')
        values = {'owner.json': dict(owner=owner), 'failure.json': dict(owner=owner, actual_process_exit_claimed=False),
                  'result.json': dict(complete=True), 'signed-metadata/sources-result.json': dict(status='exact_addresses_ready')}
        self.assertIs(audit.successful_outputs({'failure.json': {}}, values, 1, {}, {}), False)
        with self.assertRaises(RuntimeError):
            audit.successful_outputs({}, values, 1, {}, {})
        for code in (-9, True, 2):
            with self.subTest(code=code), self.assertRaises(RuntimeError):
                audit.successful_outputs({}, values, code, {}, {})


if __name__ == '__main__':
    suite = unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromTestCase(authored.Tests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(Independent)))
    with patch.object(dt, 'datetime', FixedDateTime):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(author_methods=7, independent_methods=5,
        synthetic_input_bytes=sum(map(len, INPUTS.values())),
        original_input_reads=False, actual_network=False, actual_lock=False,
        actual_subprocess=False, actual_keyring_or_deb_reads=False,
        successful_event_count=19, nested_files=27, passed=result.wasSuccessful())))
    raise SystemExit(0 if result.wasSuccessful() else 1)
