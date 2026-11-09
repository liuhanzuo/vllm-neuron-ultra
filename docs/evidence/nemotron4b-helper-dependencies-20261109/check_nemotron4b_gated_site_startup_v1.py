"""Synthetic venv startup experiment; no installed SDK or package execution.

The fixture restores site.main only after the real parent/child handshake.
This is evidence for a future bootstrap design, not a runtime authorization
or a complete authenticated import context for the real SDK environment.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nemotron4b_process_handshake_v1 as handshake
import nemotron4b_gated_child_supervisor_v1 as supervisor

NONCE = '4'*32
MARKER_MODULE = '_nano4b_synthetic_startup_marker'


def fixture():
    if sys.argv[1] == '--normal-startup-probe':
        print(json.dumps(dict(marker_preceded_script=MARKER_MODULE in sys.modules,
                              no_site=sys.flags.no_site, prefix=sys.prefix)))
        return
    ready, release = int(sys.argv[2]), int(sys.argv[3])
    marker = Path(sys.argv[4])
    handshake.require(sys.flags.no_site == 1 and MARKER_MODULE not in sys.modules
                      and not marker.exists(), 'No fixture startup before parent release')
    proof = handshake.child_wait(ready, release, nonce=NONCE, role='helper-0',
                                 deadline=time.monotonic()+5)
    restore_ns = time.monotonic_ns()
    import site
    site.main()
    observed = json.loads(marker.read_text())
    print(json.dumps(dict(marker_preceded_script=False, marker_after_site=MARKER_MODULE in sys.modules,
        parent_grant_ns=proof['grant']['grant_ns'], release_ns=proof['child_release_ns'],
        restore_ns=restore_ns, marker_ns=observed['marker_ns'], prefix=sys.prefix,
        no_site_flag_remains=sys.flags.no_site, actual_SDK_helper_executed=False)))


class SiteStartup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nano4b-site-fixture-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'bin').mkdir()
        self.python = self.root/'bin/python3'
        self.python.symlink_to(Path(sys.executable).resolve())
        (self.root/'pyvenv.cfg').write_text('home = '+str(Path(sys.executable).resolve().parent)
            +'\ninclude-system-site-packages = false\n')
        site = self.root/('lib/python%d.%d/site-packages' % sys.version_info[:2])
        site.mkdir(parents=True)
        self.marker = self.root/'startup.json'
        (site/(MARKER_MODULE+'.py')).write_text('import json, time\n'
            +'from pathlib import Path\n'
            +'Path('+repr(str(self.marker))+').write_text(json.dumps({"marker_ns": time.monotonic_ns()}))\n')
        (site/'synthetic.pth').write_text('import '+MARKER_MODULE+'\n')
        self.env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', PYTHONNOUSERSITE='1',
                        PYTHONDONTWRITEBYTECODE='1')

    def invoke(self, guard=lambda: None):
        def command(ready, release):
            return [str(self.python), '-B', '-S', str(Path(__file__).resolve()), '--gated-site-fixture',
                    str(ready), str(release), str(self.marker)]
        return supervisor.run(command, environment=self.env, cwd=str(self.root), nonce=NONCE,
            role='helper-0', expected_cgroup=handshake.owner(os.getpid())['cgroup'], guard=guard,
            handshake_seconds=2, execution_seconds=2)

    def test_normal_python_executes_pth_before_script(self):
        result = subprocess.run([str(self.python), '-B', str(Path(__file__).resolve()),
            '--normal-startup-probe'], env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=5, check=True)
        value = json.loads(result.stdout)
        self.assertTrue(value['marker_preceded_script']); self.assertTrue(self.marker.exists())
        self.assertEqual(value['no_site'], 0); self.assertEqual(value['prefix'], str(self.root))

    def test_no_site_then_actual_parent_release_then_startup(self):
        result = self.invoke()
        self.assertTrue(result['successful'], result['failure'])
        self.assertTrue(result['child_reaped']); self.assertEqual(result['actual_exit_code'], 0)
        value = json.loads(result['output']['stdout'])
        self.assertFalse(value['marker_preceded_script']); self.assertTrue(value['marker_after_site'])
        self.assertEqual(value['prefix'], str(self.root))
        self.assertLessEqual(value['parent_grant_ns'], value['release_ns'])
        self.assertLess(value['release_ns'], value['restore_ns'])
        self.assertLessEqual(value['restore_ns'], value['marker_ns'])
        self.assertEqual(value['no_site_flag_remains'], 1)
        self.assertFalse(value['actual_SDK_helper_executed'])

    def test_parent_denial_never_executes_startup(self):
        calls = 0
        def guard():
            nonlocal calls
            calls += 1
            if calls == 4: raise RuntimeError('Synthetic parent denied actual release')
        result = self.invoke(guard)
        self.assertFalse(result['successful']); self.assertTrue(result['child_reaped'])
        self.assertIsNone(result['handshake']); self.assertFalse(self.marker.exists())
        self.assertEqual(result['cleanup_errors'], [])

    def test_fixture_does_not_create_bytecode(self):
        result = self.invoke()
        self.assertTrue(result['successful'], result['failure'])
        self.assertEqual(list(self.root.rglob('*.pyc')), [])
        self.assertFalse(result['runtime_admitted'])


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('--gated-site-fixture', '--normal-startup-probe'): fixture()
    else: unittest.main(verbosity=2)
