"""Exact source/identity preservation; no actual runtime operations."""
import ast
import hashlib
import json
from pathlib import Path
import unittest
from test_source import HERE,Q,c

class Preservation(unittest.TestCase):
    def test_v8_byte_content_and_identity_unchanged(self):
        rows=json.loads((HERE/'predecessor_pins.json').read_bytes())['files']
        for path,row in rows.items():
            value=Path(path);self.assertEqual(hashlib.sha256(value.read_bytes()).hexdigest(),row['sha256'])
            s=value.stat();self.assertEqual({k:getattr(s,k) for k in row['identity']},row['identity'])

    def test_all_original_runtime_functions_preserved_except_exact_target_mapping(self):
        old=Q/'qwen35_package_serial_runtime_v8'
        for name in (*c.RUNTIME_FILES,'source_text.py','input_pins.json'):
            if name in ('compiler_assets.py','compiler_assets.json'):continue
            before=(old/name).read_text().replace('qwen35_package_serial_runtime_v8','qwen35_package_serial_runtime_v9')
            if name=='io.py':before=before.replace('target = str(compiler_assets.ASSET_ROOT / path.name)','target = str(compiler_assets.target_path(path.name))')
            if name=='budget.py':before=before.replace('installed SDK PWP assets','installed SDK PWP/DVE assets')
            self.assertEqual((HERE/name).read_text(),before,name)

    def test_asset_nofollow_hash_identity_and_ancestry_bodies_unchanged(self):
        def nodes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        old=nodes(Q/'qwen35_package_serial_runtime_v8/compiler_assets.py');new=nodes(HERE/'compiler_assets.py')
        for name in ('attrs','directory','parent_fd'):self.assertEqual(old[name],new[name])
        before=(Q/'qwen35_package_serial_runtime_v8/compiler_assets.py').read_text()
        after=(HERE/'compiler_assets.py').read_text()
        for name in ('asset','compiler_sdk_link'):
            source=ast.get_source_segment(before,next(n for n in ast.parse(before).body if isinstance(n,ast.FunctionDef) and n.name==name))
            source=source.replace('path.parent == ASSET_ROOT and path.name in NAMES','path.name in NAMES and path == target_path(path.name)').replace('target = ASSET_ROOT / parts[4]','target = target_path(parts[4])')
            self.assertEqual(ast.dump(ast.parse(source).body[0],include_attributes=False),new[name])

if __name__=='__main__':unittest.main()
