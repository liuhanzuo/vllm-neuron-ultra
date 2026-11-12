"""Bounded saved JSON, source and synthetic in-memory streams only."""
import ast
import copy
import hashlib
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

import minicpm5_sdk310_stdlib_extra_members_v1 as m
import dory_deb_tar_members_v2 as tar
from check_dory_deb_tar_members_v1 import archive, entry


def definitions(names):
    """Execute only reviewed transport definitions; never import its SSL/I/O graph."""
    path = Path(m.c.Q/'dory_deb_transport_v3.py')
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == m.SOURCE_PINS[str(path)]
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    env = dict(Path=Path, hashlib=hashlib, os=os, stat=stat, MIB=2**20,
               identity=lambda x:tuple(getattr(x, 'st_' + k) for k in ('dev','ino','mode','nlink','size','mtime_ns','ctime_ns')))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


def synthetic(request):
    """Generated ASCII member bodies; no real archive or installed binary read."""
    request = copy.deepcopy(request)
    targets = {t['observed_path']:t['archive_member'] for t in request['package']['targets']}
    aliases = {name.lstrip('/'):name for name in request['package']['saved_package_names']}
    bodies = {path:('synthetic member ' + path).encode() for path in targets}
    raw = archive(*(entry(name, bodies[path]) for path,name in targets.items()))
    parsed = tar.verify(io.BytesIO(raw).read, lambda:None, targets, aliases)
    old = request['old_package']['inspection']
    old['tar'] = copy.deepcopy(parsed)
    for key in m.PATHS:
        del old['tar']['targets'][key]
        del old['tar']['aliases'][key.lstrip('/')]
    old['decompressor'].update(stdout_bytes=len(raw))
    inspection = copy.deepcopy(old); inspection['tar'] = parsed
    return request, inspection, raw, bodies


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = m.prepare()
        cls.documents = {key:m.c.load(ref, m.METADATA_CAP) for key,ref in m.REFS.items()}

    def loader(self, changes):
        documents = copy.deepcopy(self.documents)
        changes(documents)
        def load(ref, cap):
            self.assertEqual(cap, m.METADATA_CAP)
            key = next(key for key,value in m.REFS.items() if value == ref)
            return documents[key]
        return load

    def fake_transport(self, request, inspection, *, current_change=lambda row:None):
        calls = []
        def inspect(row, archive, tool, guard, emit):
            self.assertEqual(row, request['package']); self.assertEqual(str(archive), request['archive']['path'])
            self.assertEqual(tool, request['tool']); calls.append(('inspect', str(archive))); guard()
            return copy.deepcopy(inspection)
        def file_ref(path, guard, expected, cap=64*2**20, expected_nlink=1):
            self.assertIn(path, m.PATHS); self.assertEqual(expected_nlink, 1)
            target = next(t for t in request['targets'] if t['observed_path'] == path)
            mapping = target['saved_mappings']['pre_ctypes'][0]
            observed = dict(path=path, **expected)
            identity = dict(dev=os.makedev(*mapping['device']), ino=mapping['inode'], mode=stat.S_IFREG|0o644,
                            nlink=1, size=expected['bytes'], mtime_ns=1, ctime_ns=1)
            current_change(identity); guard(); calls.append(('file_ref', path))
            return observed, identity
        return SimpleNamespace(inspect=inspect, file_ref=file_ref), calls

    def test_01_actual_saved_metadata_only_preparation_is_exact_and_unadmitted(self):
        value = self.saved
        self.assertFalse(value['runtime_admitted'])
        self.assertEqual(len(value['package']['targets']), 6)
        self.assertEqual(value['archive']['bytes'], 1850286)
        self.assertEqual(value['archive']['sha256'], 'ee09afbc981a4aaa00053f019eab6b45dcc4dbeb4ca4fcc4277eff539e09f228')
        self.assertEqual(value['old_package']['inspection']['tar']['stream_bytes'], 8386560)
        self.assertEqual(value['old_package']['inspection']['tar']['stream_sha256'], '1d5f105c3ffa421ee3a996332f7f4d396ee918b69f3c43090cfb0b0d7dc3856d')
        self.assertEqual(value['mini_scientific_constraints'], m.CONSTRAINTS)
        self.assertEqual(value['archive_stage_budget'], m.c.INVENTORY)
        self.assertEqual(self.documents['replay']['target_count'], 12)
        self.assertEqual(len(self.documents['replay']['packages']), 8)

    def test_02_mutated_saved_provenance_or_target_never_reaches_payload_work(self):
        mutations = [lambda d:d['completed'].update(passed=False),
                     lambda d:d['replay_exit']['actual_tool'].update(exit_code=1),
                     lambda d:d['download_exit'].update(owners_gone=False),
                     lambda d:d['bootstrap_exit'].update(physically_drained=False),
                     lambda d:d['replay']['packages'].pop(),
                     lambda d:d['mapping']['targets'][0].update(proposed_archive_member='usr/lib/wrong.so'),
                     lambda d:d['mapping']['targets'][0].update(archive_member_sha256='0'*64),
                     lambda d:d['mapping']['targets'][0]['saved_mappings']['pre_ctypes'][0].update(inode=1),
                     lambda d:d['mapping']['scientific_constraints'].update(scientific_descriptor_rows=1),
                     lambda d:d['mapping']['proposed_archive_reuse']['archive'].update(sha256='0'*64),
                     lambda d:d['mapping']['proposed_archive_reuse'].update(prior_tar_stream_bytes=1),
                     lambda d:d['mapping']['proposed_archive_reuse']['signed_sources'].pop()]
        for mutate in mutations:
            with self.assertRaises((RuntimeError, KeyError)): m.prepare(self.loader(mutate))

    def test_03_metadata_pin_change_and_truncation_rejected_by_existing_reader(self):
        with tempfile.TemporaryDirectory(prefix='mini-extra-pure-') as folder:
            p = Path(folder)/'fixture.json'; p.write_text('{"synthetic":true}\n')
            ref, raw = m.c.reference(p)
            self.assertTrue(m.c.load(ref)['synthetic'])
            with self.assertRaises(RuntimeError): m.c.load(dict(ref, sha256='0'*64))
            p.write_bytes(raw[:-2])
            with self.assertRaises(RuntimeError): m.c.load(ref)

    def test_04_actual_existing_tar_parser_accepts_only_complete_six_member_stream(self):
        request, inspection, raw, bodies = synthetic(self.saved)
        self.assertIs(m.validate_inspection(inspection, request), inspection)
        self.assertEqual(len(inspection['tar']['targets']), 6)
        for path in m.PATHS:
            self.assertEqual(inspection['tar']['targets'][path]['sha256'], hashlib.sha256(bodies[path]).hexdigest())
        targets = {path:path.lstrip('/') for path in m.PATHS}
        for broken in (raw[:-1], raw[:-512], raw[:700], raw[:-1024] + b'X' + raw[-1023:]):
            with self.assertRaises(ValueError): tar.verify(io.BytesIO(broken).read, lambda:None, targets)

    def test_05_wrong_or_duplicate_member_and_member_type_rejected(self):
        targets = {path:path.lstrip('/') for path in m.PATHS}
        one, two = [path.lstrip('/') for path in m.PATHS]
        for raw in (archive(entry(one, b'synthetic'), entry('usr/lib/wrong.so', b'synthetic')),
                    archive(entry(one, b'a'), entry(one, b'b'), entry(two, b'c')),
                    archive(entry(one, kind=b'5'), entry(two, b'c'))):
            with self.assertRaises(ValueError): tar.verify(io.BytesIO(raw).read, lambda:None, targets)
        request, inspection, raw, bodies = synthetic(self.saved)
        inspection['tar']['targets'][m.PATHS[0]]['terminal'] = 'usr/lib/wrong.so'
        with self.assertRaises(RuntimeError): m.validate_inspection(inspection, request)

    def test_06_original_stream_digest_size_old_member_and_real_exit_must_match(self):
        request, original, raw, bodies = synthetic(self.saved)
        mutations = [lambda x:x['tar'].update(stream_sha256='0'*64),
                     lambda x:x['tar'].update(stream_bytes=len(raw)-512),
                     lambda x:x['tar'].update(complete=False),
                     lambda x:x['decompressor'].update(returncode=120),
                     lambda x:x['archive'].update(sha256='0'*64),
                     lambda x:x['archive_identity'].update(ino=1),
                     lambda x:x['tar']['targets'][next(iter(request['old_package']['inspection']['tar']['targets']))].update(sha256='0'*64),
                     lambda x:x['tar']['targets'].pop(m.PATHS[1]),
                     lambda x:x['tar']['aliases'].pop(m.PATHS[0].lstrip('/'))]
        for mutate in mutations:
            value = copy.deepcopy(original); mutate(value)
            with self.assertRaises(RuntimeError): m.validate_inspection(value, request)

    def test_07_existing_file_ref_rejects_wrong_digest_and_truncated_synthetic_archive(self):
        env = definitions({'need', 'file_ref'})
        with tempfile.TemporaryDirectory(prefix='mini-extra-pure-') as folder:
            path = Path(folder)/'synthetic.json'; raw = b'{"synthetic_archive":"not a real package"}\n'; path.write_bytes(raw)
            ref, identity = env['file_ref'](path, lambda:None)
            with self.assertRaisesRegex(ValueError, 'size/hash'):
                env['file_ref'](path, lambda:None, dict(ref, sha256='0'*64))
            path.write_bytes(raw[:-1])
            with self.assertRaisesRegex(ValueError, 'size/hash'):
                env['file_ref'](path, lambda:None, ref)

    def test_08_existing_transport_inspect_uses_whole_tar_and_real_success_receipt(self):
        request, inspection, raw, bodies = synthetic(self.saved)
        env = definitions({'need', 'inspect'})
        class Pipe:
            def __init__(self):
                self.read = io.BytesIO(raw).read; self.closed = 0
            def receipt(self): return copy.deepcopy(inspection['decompressor'])
            def close(self): self.closed += 1
        pipe = Pipe()
        def file_ref(path, guard, expected, cap=64*2**20):
            if str(path) == request['tool']['path']: return request['tool'], {'synthetic_tool':True}
            self.assertEqual(expected, request['package'])
            return request['archive'], inspection['archive_identity']
        env.update(file_ref=file_ref, TarPipe=lambda *args:pipe)
        value = env['inspect'](request['package'], Path(request['archive']['path']), request['tool'], lambda:None, lambda row:None)
        self.assertEqual(value, inspection); self.assertEqual(pipe.closed, 1)

    def test_09_collect_is_one_inspection_two_reads_and_preserves_nonpromotion(self):
        request, inspection, raw, bodies = synthetic(self.saved)
        transport, calls = self.fake_transport(request, inspection)
        with patch.object(m, 'prepare', return_value=request), \
             patch.dict(sys.modules, {'dory_deb_transport_v3':transport}):
            value = m.collect(lambda:None, lambda row:None)
        self.assertEqual(calls, [('inspect', request['archive']['path'])] + [('file_ref', p) for p in m.PATHS])
        self.assertEqual(value['current_match_count'], 2)
        self.assertEqual(value['prior_target_count'], 12)
        self.assertEqual(value['prior_package_count'], 8)
        for key in ('actual_process_exit_claimed','accepted_native_authority_created','native_ABI_qualified',
                    'complete_native_dependency_closure_established','full_model_qualified','new_speedup_claim',
                    'download_performed','installed_anything'): self.assertFalse(value[key])
        self.assertTrue(all(x['historical_full_file_content_proven'] is False for x in value['comparisons']))
        self.assertTrue(all(x['original_matches'] is None for x in value['comparisons']))
        self.assertLess(len(json.dumps(value).encode()), m.METADATA_CAP)

    def test_10_current_mapping_identity_change_stops_supplement(self):
        request, inspection, raw, bodies = synthetic(self.saved)
        for mutate in (lambda x:x.update(ino=x['ino']+1), lambda x:x.update(dev=os.makedev(1,1)),
                       lambda x:x.update(nlink=2), lambda x:x.update(size=1)):
            transport, calls = self.fake_transport(request, inspection, current_change=mutate)
            with patch.object(m, 'prepare', return_value=request), \
                 patch.dict(sys.modules, {'dory_deb_transport_v3':transport}):
                with self.assertRaises(RuntimeError): m.collect(lambda:None, lambda row:None)

    def test_11_guard_failure_precedes_any_payload_work(self):
        def denied(): raise RuntimeError('synthetic unadmitted owner')
        with patch.object(m, 'prepare') as prepare:
            with self.assertRaisesRegex(RuntimeError, 'synthetic unadmitted'): m.collect(denied, lambda row:None)
            prepare.assert_not_called()

    def test_12_inert_sources_unchanged_accepted_parsers_and_no_extra_framework(self):
        m.c.verify_sources(m.SOURCE_PINS)
        tree = ast.parse(Path(m.__file__).read_text())
        self.assertFalse(any(isinstance(n, ast.If) and '__main__' in ast.unparse(n.test) for n in tree.body))
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        for forbidden in ('subprocess.Popen','io.download','os.system','runtime.acquire','c.save','tarfile.open'):
            self.assertNotIn(forbidden, calls)
        self.assertEqual(calls.count('io.inspect'), 1)
        self.assertFalse(any(name.split('.')[0] in {'torch','numpy','nki','neuronxcc','ctypes'} for name in sys.modules))


if __name__ == '__main__': unittest.main(verbosity=2)
