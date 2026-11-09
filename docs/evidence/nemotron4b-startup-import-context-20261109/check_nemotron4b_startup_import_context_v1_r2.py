"""Actual synthetic venv startup plus existing helper entry and parent gate.

No installed SDK code, model or original native library is executed. The
new dispatch context is actually installed during startup/helper execution.
Origin policy here covers generated fixtures only, not the real SDK closure.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nemotron4b_process_handshake_v1 as handshake
import nemotron4b_gated_child_supervisor_v1 as supervisor
import nemotron4b_guarded_helper_entry_v2 as helper
import nemotron4b_gated_site_restore_v1 as restore
import nemotron4b_helper_dependency_catalog_v1 as catalog
import importlib.machinery as machinery
import nemotron4b_import_router_v2 as routing
import nemotron4b_import_context_v1 as dispatch

NONCE = '7'*32


def fixture():
    ready, release = int(sys.argv[2]), int(sys.argv[3])
    root, mode = Path(sys.argv[4]), sys.argv[5]
    package = root/('lib/python%d.%d/site-packages' % sys.version_info[:2])
    source = root/'bin/synthetic-helper'
    site_source = Path(sys.base_prefix)/('lib/python%d.%d/site.py' % sys.version_info[:2])
    # Remove only the duplicate introduced by the fixture's local module imports.
    sys.path[:] = list(dict.fromkeys(sys.path))
    pre_path = list(sys.path)
    expected = [str(root/'bin'), *pre_path[1:], str(package)]
    expected = list(dict.fromkeys(str(Path(p).absolute()) for p in expected))
    paths = [root/'pyvenv.cfg', site_source, package/'synthetic.pth',
             package/'_nano_site_restore_marker.py']
    files = {}
    for p in paths:
        _, row = catalog.source(p)
        files[str(p)] = {k:row[k] for k in ('sha256', 'bytes', 'fullstat')}
    policy = dict(helper_script=str(source), site_module=str(site_source),
        venv_config=str(root/'pyvenv.cfg'), site_packages=str(package),
        files=files, pth_files=[str(package/'synthetic.pth')],
        pre_path=pre_path, post_path=expected, prefix=str(root), base_prefix=sys.base_prefix)
    if mode == 'wrong-path':
        policy['post_path'].append(str(root/'unexpected'))
    provenance, state = {}, {}
    @contextmanager
    def origins():
        state['origin_context_enter_ns'] = time.monotonic_ns()
        if mode == 'denied-context':
            raise RuntimeError('Synthetic origin context denied startup')
        modules = {}
        fixture_sources = [site_source, site_source.parent/'_sitebuiltins.py',
                           *package.glob('_nano_site_*.py')]
        for path in fixture_sources:
            _, record = catalog.source(path)
            modules[path.stem] = dict(record, kind='source', package=False)
        approved_builtins = {}
        preloaded = dict(sys.modules)
        original_finders = tuple(sys.meta_path)
        original_list = sys.meta_path
        def authorize_finder(finder):
            if any(finder is f for f in original_finders):
                return
            marker_module = sys.modules.get('_nano_site_restore_marker')
            require = handshake.require
            require(marker_module is not None and
                    finder is vars(marker_module).get('FINDER'),
                    'Only pinned synthetic startup finder')
        def deny(*args):
            raise RuntimeError('Fixture has no custom or native provider permission')
        def authorize_builtin(name, spec, finder):
            handshake.require(name == 'pwd' and spec.origin == 'built-in'
                              and spec.loader is machinery.BuiltinImporter
                              and finder is machinery.BuiltinImporter,
                              'Only observed original pwd interpreter builtin')
            approved_builtins[name] = spec
        router = routing.ImportRouter(finders=original_finders, modules=modules,
            namespaces={}, guard=lambda:None, authorize_finder=authorize_finder,
            authorize_builtin=authorize_builtin, authorize_native=deny, authorize_custom=deny)
        def audit_modules():
            for name, module in tuple(sys.modules.items()):
                if name in preloaded:
                    handshake.require(module is preloaded[name], 'Fixture preloaded module unchanged')
                    continue
                if name in approved_builtins:
                    handshake.require(module.__spec__ is approved_builtins[name]
                                      and module.__loader__ is machinery.BuiltinImporter,
                                      'Exact observed interpreter builtin origin')
                    continue
                row = modules.get(name)
                handshake.require(row is not None and getattr(module, '__file__', None)==row['path'],
                                  'New fixture module has pinned origin: '+name)
                handshake.require(isinstance(module.__spec__.loader,routing.PinnedSourceLoader)
                                  and module.__spec__.loader.router is router,
                                  'New fixture module uses authenticated source loader')
        def audit_environment():
            handshake.require(sys.meta_path is original_list, 'Fixture retains original finder list')
        scope = dispatch.DispatchContext(router,
            authenticate_bootstrap=lambda original:handshake.require(
                original is dispatch.bootstrap._find_spec, 'Exact observed fixture bootstrap'),
            audit_modules=audit_modules, audit_environment=audit_environment,
            approve_existing_spec=lambda *a:deny())
        with scope.installed()as imports:
            yield imports
        state['origin_context_exit_ns'] = time.monotonic_ns()
    def wait_parent():
        proof = handshake.child_wait(ready, release, nonce=NONCE, role='helper-0',
                                     deadline=time.monotonic()+5)
        state['parent_release'] = proof
        if mode == 'changed-pth':
            (package/'synthetic.pth').write_text('import _nano_site_restore_marker; print("changed")\n')
        if mode == 'extra-pth':
            (package/'extra.pth').write_text('import _nano_site_restore_marker\n')
        return proof
    raw = source.read_bytes()
    source_ref = dict(path=str(source), sha256=hashlib.sha256(raw).hexdigest())
    try:
        result = helper.run(source_reference=source_ref, read_source=lambda ref:source.read_bytes(),
            expected_stdout=b'synthetic-library\n', library_identity=lambda:{'synthetic':True},
            expected_library_identity={'synthetic':True}, guard=lambda: None,
            wait_for_parent=wait_parent, import_context=lambda:restore.restore(
                policy, guard=lambda:None, origin_context=origins, provenance=provenance))
        state['helper_stdout'] = result['stdout'].decode()
        if mode == 'second-startup':
            try:
                with restore.restore(policy, guard=lambda:None, origin_context=origins, provenance={}):
                    pass
            except ValueError as error:
                state['second_startup_rejected'] = 'one-use' in str(error)
            else:
                raise AssertionError('Second startup unexpectedly permitted')
        code = 0
    except BaseException as error:
        state['error'] = dict(type=type(error).__name__, message=str(error))
        code = error.code if isinstance(error, SystemExit) and type(error.code) is int else 13
    state.update(provenance=provenance, no_site=sys.flags.no_site,
        actual_prefix=sys.prefix, actual_sdk_executed=False)
    marker = root/'startup-marker.json'
    state['marker'] = json.loads(marker.read_text()) if marker.exists() else None
    print(json.dumps(state), flush=True)
    raise SystemExit(code)


class SiteRestoration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nano4b-restoration-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'bin').mkdir()
        self.python = self.root/'bin/python'
        self.python.symlink_to(Path(sys.executable).resolve())
        (self.root/'pyvenv.cfg').write_text('home = '+str(Path(sys.executable).resolve().parent)
            +'\ninclude-system-site-packages = false\n')
        self.package = self.root/('lib/python%d.%d/site-packages' % sys.version_info[:2])
        self.package.mkdir(parents=True)
        self.marker = self.root/'startup-marker.json'
        (self.package/'_nano_site_restore_marker.py').write_text(
            'import json,time,sys\nfrom pathlib import Path\n'
            +'Path('+repr(str(self.marker))+').write_text(json.dumps({"time_ns":time.monotonic_ns(),"prefix":sys.prefix}))\n')
        (self.package/'synthetic.pth').write_text('import _nano_site_restore_marker\n')
        (self.root/'bin/synthetic-helper').write_text('print("synthetic-library")\n')
        self.env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8',
                        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')

    def invoke(self, mode='normal', guard=lambda:None):
        def command(ready, release):
            return [str(self.python), '-B', '-S', str(Path(__file__).resolve()), '--fixture',
                    str(ready), str(release), str(self.root), mode]
        result = supervisor.run(command, environment=self.env, cwd=str(self.root), nonce=NONCE,
            role='helper-0', expected_cgroup=handshake.owner(os.getpid())['cgroup'],
            guard=guard, handshake_seconds=2, execution_seconds=3, stdout_cap=65536, stderr_cap=8192)
        self.assertTrue(result['child_reaped'])
        self.assertEqual(result['cleanup_errors'], [])
        self.assertFalse(result['runtime_admitted'])
        value = json.loads(result['output']['stdout']) if result['output']['stdout'] else None
        return result, value

    def test_actual_parent_gate_then_site_then_original_helper(self):
        result, value = self.invoke()
        self.assertTrue(result['successful'], result['failure'])
        self.assertEqual(value['helper_stdout'], 'synthetic-library\n')
        self.assertTrue(value['provenance']['complete'])
        self.assertEqual(value['actual_prefix'], str(self.root))
        self.assertEqual(value['marker']['prefix'], str(self.root))
        self.assertEqual(value['provenance']['actual_path'][0], str(self.root/'bin'))
        self.assertLess(value['parent_release']['child_release_ns'], value['origin_context_enter_ns'])
        self.assertLess(value['origin_context_enter_ns'], value['marker']['time_ns'])
        self.assertEqual(value['no_site'], 1)
        self.assertFalse(value['provenance']['no_site_flag_equivalence_proven'])

    def test_pth_error_swallowed_by_site_is_still_failure(self):
        (self.package/'synthetic.pth').write_text('import definitely_missing_nano_fixture_module\n')
        result, value = self.invoke()
        self.assertFalse(result['successful'])
        self.assertIn('Silent startup required', value['error']['message'])
        self.assertIn('Remainder of file ignored', value['provenance']['startup_stderr'])
        self.assertIsNone(value['marker'])
        self.assertNotIn('helper_stdout', value)

    def test_startup_stdout_does_not_escape_as_successful_helper_output(self):
        (self.package/'synthetic.pth').write_text('import _nano_site_restore_marker; print("unexpected startup text")\n')
        result, value = self.invoke()
        self.assertFalse(result['successful'])
        self.assertIn('unexpected startup text', value['provenance']['startup_stdout'])
        self.assertNotIn('helper_stdout', value)

    def test_pth_change_after_parent_gate_prevents_initialization(self):
        result, value = self.invoke('changed-pth')
        self.assertFalse(result['successful'])
        self.assertIsNone(value['marker'])
        self.assertNotIn('origin_context_enter_ns', value)

    def test_new_pth_file_prevents_initialization(self):
        result, value = self.invoke('extra-pth')
        self.assertFalse(result['successful'])
        self.assertIn('complete startup .pth set', value['error']['message'])
        self.assertIsNone(value['marker'])

    def test_origin_context_denial_prevents_startup(self):
        result, value = self.invoke('denied-context')
        self.assertFalse(result['successful'])
        self.assertIsNone(value['marker'])
        self.assertIn('origin context denied', value['error']['message'])

    def test_wrong_restored_import_path_is_rejected(self):
        result, value = self.invoke('wrong-path')
        self.assertFalse(result['successful'])
        self.assertIn('restored helper import path', value['error']['message'])
        self.assertIsNotNone(value['marker'])
        self.assertNotIn('helper_stdout', value)

    def test_parent_denial_prevents_all_startup(self):
        count = 0
        def guard():
            nonlocal count
            count += 1
            if count == 4:
                raise RuntimeError('Synthetic parent denied release')
        result, value = self.invoke(guard=guard)
        self.assertFalse(result['successful'])
        self.assertFalse(self.marker.exists())
        self.assertIsNone(result['handshake'])

    def test_second_site_restore_cannot_retry_in_same_process(self):
        result, value = self.invoke('second-startup')
        self.assertTrue(result['successful'], result['failure'])
        self.assertTrue(value['second_startup_rejected'])

    def test_original_nonzero_helper_exit_is_preserved(self):
        (self.root/'bin/synthetic-helper').write_text('raise SystemExit(23)\n')
        result, value = self.invoke()
        self.assertFalse(result['successful'])
        self.assertEqual(result['actual_exit_code'], 23)
        self.assertEqual(value['error']['type'], 'SystemExit')
        self.assertFalse(value['provenance']['complete'])
        self.assertEqual(value['provenance']['failure']['type'], 'SystemExit')


    def test_helper_cannot_change_restored_execution_prefix(self):
        (self.root/'bin/synthetic-helper').write_text(
            'import sys\nsys.exec_prefix = "/unexpected-prefix"\nprint("synthetic-library")\n')
        result, value = self.invoke()
        self.assertFalse(result['successful'])
        self.assertIn('environment unchanged after helper', value['error']['message'])
        self.assertFalse(value['provenance']['complete'])

    def test_startup_output_overflow_stays_a_failure(self):
        (self.package/'synthetic.pth').write_text(
            'import _nano_site_restore_marker; print("X" * 5000)\n')
        result, value = self.invoke()
        self.assertFalse(result['successful'])
        self.assertTrue(value['provenance']['startup_output_overflow'])
        self.assertFalse(value['provenance']['complete'])
        self.assertNotIn('helper_stdout', value)

    def test_bytecode_is_not_written(self):
        result, value = self.invoke()
        self.assertTrue(result['successful'], result['failure'])
        self.assertEqual(list(self.root.rglob('*.pyc')), [])

    def test_pth_registers_finder_then_helper_preserves_child_optional_fallback(self):
        original=(self.package/'_nano_site_restore_marker.py').read_text()
        (self.package/'_nano_site_restore_marker.py').write_text(original+
            'from importlib.machinery import PathFinder\n'
            'class Finder:\n'
            ' def find_spec(self,name,path=None,target=None):\n'
            '  return PathFinder.find_spec(name,path,target)\n'
            'FINDER=Finder()\n'
            'sys.meta_path.insert(0,FINDER)\n')
        (self.package/'_nano_site_optional_child.py').write_text('raise ImportError("optional backend absent")\n')
        (self.root/'bin/synthetic-helper').write_text(
            'try:\n import _nano_site_optional_child\nexcept ImportError:\n print("synthetic-library")\n')
        result,value=self.invoke()
        self.assertTrue(result['successful'],value)
        self.assertTrue(value['provenance']['import_context']['dispatch_context_completed'])
        self.assertFalse(value['provenance']['import_context']['full_runtime_dependency_closure_proven'])
        self.assertEqual(value['helper_stdout'],'synthetic-library\n')
        self.assertLess(value['parent_release']['child_release_ns'],value['marker']['time_ns'])

    def test_helper_catch_all_cannot_hide_unpinned_module_origin(self):
        (self.package/'_unreviewed_nano_dependency.py').write_text('VALUE=1\n')
        (self.root/'bin/synthetic-helper').write_text(
            'try:\n import _unreviewed_nano_dependency\nexcept Exception:\n pass\nprint("synthetic-library")\n')
        result,value=self.invoke()
        self.assertFalse(result['successful'])
        self.assertFalse(value['provenance']['complete'])
        self.assertEqual(value['error']['type'],'ProvenanceError')


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fixture':
        fixture()
    else:
        unittest.main(verbosity=2)
