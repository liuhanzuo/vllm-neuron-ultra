"""Source/AST and pure mocked integration checks. No runtime observations."""
import ast
import copy
import hashlib
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import dory_ubuntu_metadata_input_binding_v3 as b
import run_dory_ubuntu_metadata_root_v3 as parent
import dory_ubuntu_metadata_worker_v3 as worker


class Tests(unittest.TestCase):
    def test_exact_work_inputs_trust_and_adapter_version(self):
        plan = dict(work_kind=b.WORK_KIND, metadata_inputs=copy.deepcopy(b.INPUTS), ubuntu_trust=copy.deepcopy(b.TRUST),
                    source_sha256=dict(b.SOURCE_PINS), external_authorities=None)
        review = {k: copy.deepcopy(plan[k]) for k in ('work_kind', 'metadata_inputs', 'ubuntu_trust')}
        with patch.object(b.runtime, 'validate') as original:
            self.assertIs(b.validate(plan, review, {}), plan)
            original.assert_called_once_with(plan, review, {})
            for field in ('work_kind', 'metadata_inputs', 'ubuntu_trust', 'source_sha256', 'external_authorities'):
                bad = copy.deepcopy(plan)
                bad[field] = {} if field != 'work_kind' else 'wrong'
                with self.subTest(field=field), self.assertRaises(RuntimeError):
                    b.validate(bad, review, {})

    def test_authenticate_five_pinned_small_inputs_only(self):
        authority = dict(schema='dory_ubuntu_archive_key_authority_v1', independent_authority_verified=True,
                         keyring={'sha256': b.TRUST['keyring_sha256']}, primary_fingerprints=b.TRUST['primary_fingerprints'])
        observed = [(dict(sha256=ref['sha256']), b'mock-json') for ref in b.INPUTS.values()]
        with patch.object(b.c, 'reference', side_effect=observed) as read, patch.object(b.c, 'load', return_value=authority), \
             patch.object(b.discovery, 'bind_inputs') as bind:
            result = b.authenticate(Mock())
        self.assertEqual(read.call_count, 5)
        self.assertEqual(set(bind.call_args.args[0]), set(b.discovery.INPUT_PINS))
        self.assertFalse(result['package_content_authenticated'])
        b.validate_admission(dict(work_kind=b.WORK_KIND, metadata_input_binding=result), result)
        with self.assertRaises(RuntimeError):
            b.validate_admission(dict(work_kind='wrong', metadata_input_binding=result), result)

    def test_complete_allocated_scan_includes_directories_and_extra_gpgv_files(self):
        root = Path('/mock/tree')
        nested, file, extra = root/'signed-metadata', root/'signed-metadata'/'Release', root/'extra-gpgv-output'
        modes = {root: (stat.S_IFDIR, 4096, 8), nested: (stat.S_IFDIR, 4096, 8),
                 file: (stat.S_IFREG, 17, 8), extra: (stat.S_IFREG, 5000, 16)}
        def info(path):
            mode, size, blocks = modes[path]
            return SimpleNamespace(st_mode=mode, st_size=size, st_blocks=blocks)
        def children(path):
            return iter({root: [nested, extra], nested: [file]}[path])
        with patch.object(Path, 'lstat', info), patch.object(Path, 'iterdir', children):
            self.assertEqual(b.storage_usage(root), 20480)
            modes[extra] = (stat.S_IFLNK, 1, 1)
            with self.assertRaises(RuntimeError):
                b.storage_usage(root)

    def test_receipt_binds_complete_request_signature_counts_and_no_payload_claim(self):
        out = Path('/mock')
        value = dict(result=dict(path=str(out/'signed-metadata'/'sources-result.json'), sha256='a'*64), status='exact_addresses_ready', packages=6)
        result = dict(schema='dory_ubuntu_exact_package_sources_v1', status=value['status'], trust_anchor=b.TRUST,
                      packages=[{}]*6, adapter=dict(requests=12, signature_invocations=3), package_content_authenticated=False,
                      actual_archive_download=False, unresolved_closed=0, native_ABI_qualified=False)
        with patch.object(b.c, 'load', return_value=result), patch.object(b.c, 'save', return_value={'path': 'receipt'}) as save, \
             patch.object(b, 'storage_usage', return_value=4096):
            self.assertEqual(b.save_collected(out, value, {}, {}, {}, Mock()), {'path': 'receipt'})
            self.assertEqual(save.call_args.args[2], 8*b.c.MIB)
            for key in ('package_content_authenticated', 'actual_archive_download', 'native_ABI_qualified'):
                result[key] = True
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    b.save_collected(out, value, {}, {}, {}, Mock())
                result[key] = False

    def test_worker_actual_dispatch_ack_binding_then_real_adapter(self):
        tree = ast.parse(Path(worker.__file__).read_bytes())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        body = next(n for n in main.body if isinstance(n, ast.Try)).body
        start = next(i for i, n in enumerate(body) if isinstance(n, ast.ImportFrom) and n.module == 'dory_ubuntu_metadata_adapter_v4')
        stop = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'result_ref' for t in n.targets))
        block = ast.fix_missing_locations(ast.Module(body=body[start:stop+1], type_ignores=[]))
        order, binding, journal = [], {'bound': True}, Mock()
        fakeinputs = SimpleNamespace(TRUST=b.TRUST,
            authenticate=Mock(side_effect=lambda _: order.append('authenticate') or binding),
            validate_admission=Mock(side_effect=lambda *_: order.append('admission')),
            save_collected=Mock(side_effect=lambda *_: order.append('save') or {'path': 'receipt'}))
        adapter, jmod = ModuleType('dory_ubuntu_metadata_adapter_v4'), ModuleType('dory_serial_stage_journal_v2')
        adapter.run = Mock(side_effect=lambda *_: order.append('actual-adapter') or {'mock': True})
        jmod.Journal = Mock(return_value=journal)
        ns = dict(c=SimpleNamespace(verify_loaded=Mock()), plan={'source_sha256': {}}, inputs=fakeinputs,
                  guard=Mock(), admission={}, out=Path('/mock'), plan_ref={},
                  a=SimpleNamespace(admission='/mock/admission', admission_sha256='a'*64))
        with patch.dict(sys.modules, {'dory_ubuntu_metadata_adapter_v4': adapter, 'dory_serial_stage_journal_v2': jmod}):
            exec(compile(block, '<actual dispatch>', 'exec'), ns)
        self.assertEqual(order, ['authenticate', 'admission', 'actual-adapter', 'save'])
        adapter.run.assert_called_once_with(ns['guard'], journal.emit, b.TRUST, Path('/mock/signed-metadata'))

    def test_original_process_resources_ack_and_drain_are_preserved(self):
        for previous, current in [('run_dory_system_package_metadata_root_v1.py', parent.__file__),
                                  ('dory_system_package_metadata_worker_v1.py', worker.__file__)]:
            def selected(path):
                tree = ast.parse(Path(path).read_bytes())
                return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr in ('setrlimit', 'sched_setaffinity', 'nice', 'Popen', 'acquire', 'validate_admission')]
            self.assertEqual(selected(Q/previous), selected(current))
        for name in ('free', 'memory_free', 'drain', 'release_after_drain', 'preflight_history'):
            self.assertIs(getattr(parent, name), getattr(b.runtime.lifecycle, name))
        ps, ws = Path(parent.__file__).read_text(), Path(worker.__file__).read_text()
        self.assertIn('pass_fds=(readfd,)', ps)
        self.assertLess(ps.index('metadata_binding=inputs.authenticate(guard)'), ps.index('subprocess.Popen('))
        self.assertLess(ws.index("os.read(a.ready_fd,1)==b'R'"), ws.index('metadata_binding=inputs.authenticate(guard)'))
        self.assertIn("inputs.storage_usage(out)<=budget['output_bytes']", ws)
        self.assertIn("free(ROOT)>=28*c.GIB", ps)
        self.assertNotIn('28*c.GIB+budget', ps)
        self.assertIn("initial>=30*c.GIB+budget['output_bytes']", ps)

    def test_v2_only_changes_gpgv_group_and_all_pins_match(self):
        v1 = ast.parse((Q/'dory_ubuntu_metadata_adapter_v1.py').read_bytes())
        v2 = ast.parse((Q/'dory_ubuntu_metadata_adapter_v2.py').read_bytes())
        # Ignore only module docstring and the one reviewed process-group change.
        v1.body.pop(0); v2.body.pop(0)
        count = 0
        for n in ast.walk(v2):
            if isinstance(n, ast.keyword) and n.arg == 'start_new_session':
                self.assertIs(n.value.value, False)
                n.value.value = True
                count += 1
        self.assertEqual(count, 1)
        self.assertEqual(ast.dump(v1), ast.dump(v2))
        for name, expected in b.SOURCE_PINS.items():
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
