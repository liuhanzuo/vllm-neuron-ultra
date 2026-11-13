"""AST and pure-block tests only: never import helpers or execute actual main."""
import ast
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import unittest

REQUEST = Path('/home/ubuntu/prepare_qwen35_writer_geometry_r3_request_20261113_v1.py')
FINAL = Path('/home/ubuntu/finalize_qwen35_writer_geometry_r3_20261113_v2.py')
OLD = Path('/home/ubuntu/finalize_qwen35_writer_geometry_r2_20261112_v1.py')
PINS = {REQUEST: '9b8a47c30753d6b1ea2712a5a68f20aac05fe7fada90ee3f3596400302848807',
        FINAL: 'f79d3c9379a62993559116bf51a3b79cbbbf05860021aba325f3867f499f2a04',
        OLD: 'a142c013ee978d5d270eb1b8ee7f99083b15a2aff94498b980fe52b95b5da0c2'}
SOURCE = {}
for path, sha in PINS.items():
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha
    SOURCE[path] = data.decode()


def main_nodes(path):
    return next(n for n in ast.parse(SOURCE[path]).body if isinstance(n, ast.FunctionDef) and n.name == 'main').body


def assign_name(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def execute(nodes, namespace):
    # Only explicit selected assertion/construction blocks; main/imports absent.
    module = ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for n in ast.walk(module))
    exec(compile(ast.fix_missing_locations(module), '<selected-pure-block>', 'exec'), namespace)


class FakePath:
    def __init__(self, value, state): self.value, self.state = value, state
    def __truediv__(self, value): return FakePath(self.value + '/' + value, self.state)
    def exists(self): return self.state.get('exists', False)
    def is_symlink(self): return self.state.get('symlink', False)


class Qualification(unittest.TestCase):
    def fixture(self, qualified=True, exists=None):
        state = dict(exists=qualified if exists is None else exists, symlink=False)
        exit_ref = dict(path='/mock/exit.json', sha256='a'*64)
        reports = {'conv': dict(path='/mock/report.json', sha256='b'*64)}
        sources = {'/mock/source.py': 'c'*64}
        proof = dict(schema='qwen35_writer_geometry_independent_qualification_v1', reviewer='/root',
                     passed=True, blockers=[], actual_exit=exit_ref, reports=reports,
                     source_sha256=sources, full_model_qualified=False)
        digest = ['d'*64]
        def ref(path):
            if not state['exists']: raise FileNotFoundError('mock qualification absent')
            return dict(path=path.value, sha256=digest[0])
        ns = dict(Q=FakePath('/mock/research', state), qualified=qualified, ended=dict(reports=reports, source_sha256=sources),
                  exit_ref=exit_ref, ref=ref, read=lambda row: copy.deepcopy(proof))
        return ns, proof, state, digest

    def initial(self, ns):
        nodes = main_nodes(FINAL)
        start = next(i for i, n in enumerate(nodes) if assign_name(n, 'qualification_path'))
        end = next(i for i, n in enumerate(nodes) if assign_name(n, 'running_ref'))
        execute(nodes[start:end], ns)

    def republish(self, ns):
        block = next(n for n in main_nodes(FINAL) if isinstance(n, ast.Try))
        node = next(n for n in block.body if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'qualification_ref is not None')
        execute([node], ns)

    def test_valid_success_binds_exact_proof_and_rechecks(self):
        ns, _, _, _ = self.fixture(); self.initial(ns); self.republish(ns)
        self.assertEqual(ns['qualification_ref']['path'], '/mock/research/qwen35-writer-geometry-writer-r3-20261113-root-qualification.json')

    def test_missing_success_proof_rejected(self):
        ns, _, _, _ = self.fixture(exists=False)
        with self.assertRaises(FileNotFoundError): self.initial(ns)

    def test_every_qualification_field_is_checked(self):
        changes = dict(schema='wrong', reviewer='/other', passed=False, blockers=['not finished'],
                       actual_exit={'wrong': True}, reports={}, source_sha256={}, full_model_qualified=True)
        for key, value in changes.items():
            ns, proof, _, _ = self.fixture(); proof[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError): self.initial(ns)

    def test_failed_audit_has_no_qualification(self):
        ns, _, _, _ = self.fixture(qualified=False); self.initial(ns); self.republish(ns)
        self.assertIsNone(ns['qualification_ref'])

    def test_failed_audit_rejects_existing_or_dangling_qualification(self):
        for field in ('exists', 'symlink'):
            ns, _, state, _ = self.fixture(qualified=False); state[field] = True
            with self.subTest(field=field), self.assertRaises(AssertionError): self.initial(ns)

    def test_success_proof_changes_before_publication_rejected(self):
        ns, _, _, digest = self.fixture(); self.initial(ns); digest[0] = 'e'*64
        with self.assertRaises(AssertionError): self.republish(ns)

    def test_failed_proof_appears_before_publication_rejected(self):
        ns, _, state, _ = self.fixture(qualified=False); self.initial(ns); state['exists'] = True
        with self.assertRaises(AssertionError): self.republish(ns)


class PreservedControl(unittest.TestCase):
    def test_two_drain_loops_match_frozen_predecessor_AST(self):
        def drain(path):
            block = next(n for n in main_nodes(path) if isinstance(n, ast.Try))
            return next(n for n in block.body if isinstance(n, ast.For)
                        and ast.unparse(n.iter) == 'range(2)')
        self.assertEqual(ast.dump(drain(FINAL), include_attributes=False), ast.dump(drain(OLD), include_attributes=False))

    def test_source_delta_and_complete_catalog_append_preserved(self):
        nodes = main_nodes(FINAL); prior = main_nodes(OLD)
        def delta(items):
            return next(n for n in items if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                        and isinstance(n.value.func, ast.Name) and n.value.func.id == 'source_delta')
        self.assertEqual(ast.dump(delta(nodes), include_attributes=False), ast.dump(delta(prior), include_attributes=False))
        text = SOURCE[FINAL]
        self.assertIn("assert catalog['runs'][:-1] == old['runs']; catalog_shape(catalog)", text)
        self.assertIn("runs=old['runs'] + [dict(", text)
        self.assertIn("len(old['runs']) == 164", text)
        self.assertIn("total_rows=165", text)
        self.assertIn("actual_tool'] == actual", text)
        self.assertIn("ended['spec'] == spec_ref and ended['review'] == review_ref", text)
        self.assertIn("running_ref == live['running_handoff']", text)

    def test_publication_precomputes_all_caps_before_any_write(self):
        block = next(n for n in main_nodes(FINAL) if isinstance(n, ast.Try))
        calls = [(i, n.func.id) for i, statement in enumerate(block.body) for n in ast.walk(statement)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('predicted', 'once')]
        self.assertEqual([name for _, name in calls], ['predicted']*3 + ['once']*3)
        self.assertEqual(len({i for i, name in calls if name == 'predicted'}), 3)

    def test_request_seed_and_paths_are_metadata_only(self):
        tree = ast.parse(SOURCE[REQUEST])
        imports = {n.names[0].name for n in tree.body if isinstance(n, ast.Import)}
        self.assertEqual(imports, {'hashlib', 'json', 'sys'})
        self.assertNotIn('History', ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')))
        text = SOURCE[REQUEST]
        for fragment in ("session_id=72363, terminal_chunk='129c0c', exit_code=1", "== 9791", "== 164", "== (1, 7865, 7)", "with output.open('xb')", "full_preflight_pending=True", "runtime_launched=False"):
            self.assertIn(fragment, text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
