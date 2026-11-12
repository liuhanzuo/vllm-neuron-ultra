"""Stdlib/mocked source checks only; no real native/provider/worker execution."""
import ast
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import spark26_native_provider_v2 as provider
import spark26_native_serial_worker_v2 as worker


def fake_manifest():
    """Synthetic RECORD metadata; never an admission or installed runtime claim."""
    sources, records, rows = {}, {}, []
    for name, version in provider.REQUIRED.items():
        path = str(provider.SITE/(name+'-'+version+'.dist-info')/'RECORD')
        text = name+'-'+version+'.dist-info/RECORD,,\n'
        sha = hashlib.sha256(text.encode()).hexdigest()
        sources[path] = sha; records[path] = text
        rows.append(dict(name=name, version=version, record=dict(path=path, sha256=sha)))
    sources.update(provider.contract.NATIVE_SOURCES)
    files = {name: dict(sha256=sha, bytes=1) for name, sha in provider.contract.NATIVE_SOURCES.items()}
    manifest = dict(schema='spark26_native_provider_manifest_v1', distributions=rows,
        python_overrides={}, additional_files=files, linked_native_files={},
        payload_count=len(files), payload_bytes=len(files), payload_sha256=provider.contract.digest(files))
    return manifest, sources, lambda ref: records[ref['path']]


class SourceChecks(unittest.TestCase):
    def mapping_fixture(self, inode=123):
        path = '/home/ubuntu/fixture/libfixture.so'
        device = os.makedev(259, 2)
        identity = [device, inode, 4096, 10, 20]
        raw = f'1000-2000 r-xp 00000000 103:02 {inode} {path}\n'
        return path, identity, raw

    def test_native_mapping_matches_actual_device_and_inode(self):
        path, identity, raw = self.mapping_fixture()
        result = provider.mapped_native_files(raw, {path: identity}, lambda p: identity)
        self.assertEqual(result[path]['inode'], identity[1])
        self.assertEqual(result[path]['device'], identity[0])

    def test_native_mapping_rejects_old_loaded_inode_despite_current_disk_match(self):
        path, identity, raw = self.mapping_fixture()
        raw = raw.replace(' 123 ', ' 122 ')
        with self.assertRaises(ValueError):
            provider.mapped_native_files(raw, {path: identity}, lambda p: identity)

    def test_native_mapping_rejects_wrong_device_and_deleted_object(self):
        path, identity, raw = self.mapping_fixture()
        for invalid in (raw.replace('103:02', '103:03'), raw.rstrip()+' (deleted)\n'):
            with self.assertRaises(ValueError):
                provider.mapped_native_files(invalid, {path: identity}, lambda p: identity)

    def test_native_mapping_rejects_changed_disk_after_authentication(self):
        path, identity, raw = self.mapping_fixture()
        replaced = [identity[0], 999, *identity[2:]]
        with self.assertRaises(ValueError):
            provider.mapped_native_files(raw, {path: identity}, lambda p: replaced)

    def test_native_mapping_rejects_unknown_zero_or_invalid_identity(self):
        path, identity, raw = self.mapping_fixture()
        for invalid in (raw.replace(' 123 ', ' 0 '), raw.replace('103:02', 'invalid')):
            with self.assertRaises(ValueError):
                provider.mapped_native_files(invalid, {path: identity}, lambda p: identity)
        with self.assertRaises(ValueError): provider.mapped_native_files(raw, {}, lambda p: identity)

    def test_runtime_authentication_budget_does_not_claim_total_io(self):
        self.assertEqual(worker.BUDGET['max_runtime_authentication_read_bytes'], 16*2**30)
        self.assertEqual(provider.MAX_RUNTIME_AUTHENTICATION_READ, 16*2**30)
        self.assertNotIn('max_runtime_read_bytes', worker.BUDGET)

    def test_imports_do_not_import_native(self):
        self.assertFalse({'torch', 'numpy', 'safetensors', 'ctypes', 'torch_neuronx', 'vllm'} & set(provider.sys.modules))

    def test_record_keeps_package_data_and_native(self):
        digest = base64.urlsafe_b64encode(b'a'*32).decode().rstrip('=')
        text = f'pkg/a.py,sha256={digest},3\npkg/lib.so,sha256={digest},4\npkg/data.json,sha256={digest},5\n'
        result = provider.record_payloads(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(sum(x['bytes'] for x in result.values()), 12)

    def test_record_excludes_external_script_and_bytecode(self):
        text = '../../../bin/torchrun,,\npkg/__pycache__/a.pyc,,\npkg-1.dist-info/RECORD,,\n'
        self.assertEqual(provider.record_payloads(text), {})

    def test_unhashed_package_rejected(self):
        with self.assertRaises(ValueError): provider.record_payloads('pkg/a.py,,\n')

    def test_conflicting_or_duplicate_record_payload_rejected(self):
        digest = base64.urlsafe_b64encode(b'a'*32).decode()
        row = f'pkg/a.py,sha256={digest},3\n'
        with self.assertRaises(ValueError): provider.record_payloads(row+row)

    def test_manifest_requires_every_actual_direct_package(self):
        manifest, sources, read = fake_manifest()
        self.assertEqual(len(provider.merge_manifest(manifest, sources, read)), len(provider.contract.NATIVE_SOURCES))
        manifest['distributions'].pop()
        with self.assertRaises(ValueError): provider.merge_manifest(manifest, sources, read)

    def test_manifest_rejects_binary_record_override(self):
        manifest, sources, read = fake_manifest()
        path = str(provider.SITE/'torch/lib/libtorch.so')
        manifest['python_overrides'][path] = dict(sha256='a'*64, bytes=1)
        sources[path] = 'a'*64
        with self.assertRaises(ValueError): provider.merge_manifest(manifest, sources, read)

    def test_manifest_rejects_dropped_fixed_host_source(self):
        manifest, sources, read = fake_manifest()
        manifest['additional_files'].pop(provider.contract.HOST_LOADER)
        files = manifest['additional_files']
        manifest.update(payload_count=len(files), payload_bytes=len(files), payload_sha256=provider.contract.digest(files))
        with self.assertRaises(ValueError): provider.merge_manifest(manifest, sources, read)

    def test_manifest_rejects_forged_count_and_budget(self):
        manifest, sources, read = fake_manifest()
        manifest['payload_bytes'] += 1
        with self.assertRaises(ValueError): provider.merge_manifest(manifest, sources, read)

    def test_real_hash_authenticator_checks_complete_fixture(self):
        # Only a newly-created tiny text fixture is read, never installed binaries/weights.
        with tempfile.TemporaryDirectory(prefix='spark-source-check-') as directory:
            path = Path(directory)/'fixture.txt'; path.write_text('abc')
            calls = []
            expected = dict(sha256=hashlib.sha256(b'abc').hexdigest(), bytes=3)
            actual = provider.authenticate_file(path, expected, lambda: calls.append(True))
            self.assertEqual(actual[2], 3); self.assertGreaterEqual(len(calls), 2)
            expected['sha256'] = 'f'*64
            with self.assertRaises(ValueError): provider.authenticate_file(path, expected, lambda: None)

    def test_authority_dict_cannot_instantiate_runtime(self):
        with self.assertRaises(ValueError): provider.Runtime({'authenticated': True}, {})

    def test_runtime_rejects_unknown_origin(self):
        runtime = object.__new__(provider.Runtime)
        runtime.context = SimpleNamespace(check=lambda: True)
        runtime.stdlib = Path('/usr/lib/python3.10')
        runtime.identities = {}; runtime.payloads = {}
        with self.assertRaises(ValueError): runtime.check_origin('/home/ubuntu/unknown-package.py')

    def test_origin_finder_checks_before_returning_loader(self):
        sequence = []
        spec = SimpleNamespace(origin='/declared/module.py', submodule_search_locations=None)
        finder = SimpleNamespace(find_spec=lambda *args: sequence.append('find') or spec)
        runtime = SimpleNamespace(check_origin=lambda *args: sequence.append('authenticate'))
        self.assertIs(provider.OriginFinder(runtime, [finder]).find_spec('fixture'), spec)
        self.assertEqual(sequence, ['find', 'authenticate'])

    def test_runtime_factory_has_real_import_targets_without_ast_execution(self):
        tree = ast.parse(Path(provider.__file__).read_text())
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'native_providers')
        imports = {n.module for n in ast.walk(factory) if isinstance(n, ast.ImportFrom)}
        self.assertIn('neuronx_distributed.trace.trace', imports)
        self.assertIn('vllm_neuron.worker.neuronx_distributed_model_loader', imports)
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                             and n.func.id in ('exec', 'eval', 'compile') for n in ast.walk(tree)))

    def test_exact_event_names_and_counts_match_frozen_core(self):
        plan = worker.contract.plan()
        self.assertEqual(len(plan['cache_trace']), 534)
        self.assertEqual(worker.COUNTS['cache_host_reorder'], 4*sum(x['op']=='decode' for x in plan['cache_trace']))
        self.assertEqual(sum(worker.COUNTS.values()), 4844)
        tree = ast.parse((worker.ROOT/'spark26_native_port_diagnostic_v3.py').read_text())
        emitted = {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == 'emit'}
        self.assertEqual(emitted, set(worker.COUNTS)-{'qkv_projection'})

    def test_event_sink_rejects_unknown_excess_and_oversize(self):
        with tempfile.TemporaryDirectory(prefix='spark-source-events-') as root:
            sink = worker.EventSink(root)
            with self.assertRaises(ValueError): sink.emit('unknown', {})
            with self.assertRaises(ValueError): sink.emit('cache_step', {'x':'x'*worker.BUDGET['event_row_bytes']})
            sink.emit('cache_step', {'step':0})
            sink.counts['cache_step'] = worker.COUNTS['cache_step']
            with self.assertRaises(ValueError): sink.emit('cache_step', {})
            receipt = sink.close()
            self.assertFalse(receipt['complete'])

    def test_incomplete_event_receipt_cannot_be_success(self):
        with tempfile.TemporaryDirectory(prefix='spark-source-events-') as root:
            sink = worker.EventSink(root)
            with self.assertRaises(ValueError): sink.close(complete=True)

    def test_service_command_retains_exact_clean_python_and_budget(self):
        spec = dict(unit='trn1-spark26-native-diagnostic-fixture', output_root='/mnt/nvme/spark26-native-diagnostic-fixture')
        ref = dict(path=spec['output_root']+'/admission.json', sha256='1'*64)
        command = worker.service_command(spec, ref)
        self.assertEqual(command[-len(worker.argv(ref)):], worker.argv(ref))
        self.assertIn('-S', command); self.assertIn('-B', command)
        self.assertIn('PJRT_DEVICE=CPU', command)
        self.assertIn('MemorySwapMax=0', command)
        self.assertIn('LimitAS='+str(24*2**30), command)

    def test_worker_context_fails_before_proc_for_wrong_interpreter(self):
        with mock.patch.object(worker.sys, 'executable', '/invalid/python'):
            with self.assertRaises(ValueError): worker.WorkerContext({})

    def test_no_owner_admission_hash_cycle_and_no_launch_or_lock_calls(self):
        source = Path(worker.__file__).read_text(); tree = ast.parse(source)
        self.assertNotIn('worker_owner_sha256', source)
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse({'flock', 'Popen', 'spawn', 'kill', 'setrlimit'} & calls)
        self.assertIn('worker_exit_or_outer_drain_proven=False', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
