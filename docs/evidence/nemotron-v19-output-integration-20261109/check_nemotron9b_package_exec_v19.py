"""Source integration with synthetic generated payloads; no native/compiler run.

Reconstruct all original r8 links plus the 32 r9 names. Both current storage
inventory and final evidence scanning must route the same exact reader.
The v19 execution draft still needs actual post-r9/Dory history integration.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

R = Path(__file__).resolve().parents[1]
Q = R/'research'
sys.path[:0] = [str(R),str(Q)]
import nemotron9b_package_protocol_v18 as old
import nemotron9b_package_protocol_v19 as p
import nemotron9b_package_evidence_v19 as evidence
import check_nemotron9b_package_exec_v17 as previous


class Source(unittest.TestCase):
    def test_original_v18_sources_unchanged(self):
        origin=json.loads((Q/'nemotron9b-v19-draft-origin-20261109-v1.json').read_text())
        self.assertEqual(len(origin['original_v18_sha256']),16)
        for path,digest in origin['original_v18_sha256'].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),digest)

    def test_fourteen_execution_bodies_are_version_renames_only(self):
        count=0
        for path in R.glob('nemotron9b_package_*v18.py'):
            if path.stem in ('nemotron9b_package_protocol_v18','nemotron9b_package_metadata_v18'):
                continue
            self.assertEqual(path.read_text().replace('_v18','_v19'),
                             path.with_name(path.name.replace('_v18','_v19')).read_text())
            count+=1
        self.assertEqual(count,14)

    def test_protocol_function_bodies_only_change_version_and_source_list(self):
        def functions(source):
            return {n.name:ast.dump(n)for n in ast.parse(source).body
                    if isinstance(n,ast.FunctionDef)and n.name!='required_sources'}
        self.assertEqual(functions(Path(old.__file__).read_text().replace('_v18','_v19')),
                         functions(Path(p.__file__).read_text()))

    def test_all52_original_stages_and_resource_limits_retained(self):
        for name in ('STAGES','SCIENTIFIC_STAGES','ROOT_FLOOR','NVME_FLOOR','ROOT_CAP','NVME_CAP',
                     'CAPS','BUCKETS','SPLIT_FLAGS','ALIAS_FLAGS','PACKAGE','COMMIT'):
            self.assertEqual(getattr(p,name),getattr(old,name),name)
        self.assertEqual(len(p.STAGES),52)
        for stage in (*p.STAGES,'controller'):
            self.assertEqual(p.resources(stage),old.resources(stage))

    def test_full_paths_use_one_distinct_fixed_reader(self):
        self.assertIs(evidence.p,p)
        self.assertIs(p.compiler_links,p.output_inventory.compiler_links)
        self.assertEqual(Path(p.compiler_links.__file__),R/'nemotron9b_compiler_tiled_links_v1.py')
        self.assertEqual(Path(p.output_inventory.__file__),R/'nemotron9b_compiler_tiled_inventory_v1.py')
        self.assertEqual(Path(p.compiler_links.generated.__file__),Q/'nemotron9b_compiler_dynamic_links_v2.py')
        self.assertIs(p.compiler_links.generated._read.__globals__['target_for'],
                      p.compiler_links.generated.target_for)

    def test_complete_new_sources_include_all_previous(self):
        required=set(p.required_sources())
        self.assertTrue(set(old.required_sources())<required)
        self.assertTrue({str(path)for path in R.glob('nemotron9b_package_*v19.py')}<=required)
        self.assertTrue(all(Path(path).is_file()for path in required))
        for path in (p.compiler_links.__file__,p.output_inventory.__file__,
                     p.compiler_links.generated.__file__,__file__):
            self.assertIn(str(Path(path).resolve()),required)

    def test_original_metadata_schemas_and_bounds_retained(self):
        self.assertEqual(p.metadata.LIMITS,old.metadata.LIMITS)
        for name,shape in old.metadata.ENVELOPE_SCHEMAS.items():
            self.assertEqual(p.metadata.ENVELOPE_SCHEMAS[name],shape)
        self.assertEqual(set(p.metadata.ENVELOPE_SCHEMAS)-set(old.metadata.ENVELOPE_SCHEMAS),
            {'nemotron9b_package_runtime_v19','nemotron9b_package_runtime_source_review_v19'})

    def test_unchanged_historical_gate_refuses_actual_new_dory_directories(self):
        self.assertIs(p.attention_completion,old.attention_completion)
        with self.assertRaisesRegex(RuntimeError,'No unaccounted new Dory admission'):
            p.attention_completion.completed()

    def test_no_runtime_spec_or_native_libraries(self):
        self.assertFalse(list(Q.glob('nemotron9b-package-runtime-v19-*-spec.json')))
        self.assertFalse(any(x.split('.')[0]in{'torch','numpy','vllm','neuronxcc','torch_neuronx'}
                             for x in sys.modules))


class FullPaths(previous.FullPaths):
    def setUp(self):
        for name,value in [('p',p),('evidence',evidence)]:
            change=patch.object(previous,name,value)
            change.start();self.addCleanup(change.stop)
        super().setUp()
        result=json.loads((Q/'nemotron9b-r9-link-source-check-20261109-v1/result.json').read_text())
        self.assertEqual(result['real_rejected_links'],32)
        original=Path('/mnt/nvme/nemotron9b-package-r9-20261109/compile-work')
        self.tiled=[]
        for row in result['accepted_real_metadata']:
            link=self.compiler/Path(row['path']).relative_to(original)
            target=self.compiler/Path(row['symlink']).relative_to(original)
            link.parent.mkdir(parents=True,exist_ok=True)
            target.parent.mkdir(parents=True,exist_ok=True)
            data=b'SYNTHETIC r9 tiled transpose '+str(target.relative_to(self.compiler)).encode()
            if target.exists():self.assertEqual(target.read_bytes(),data)
            else:target.write_bytes(data)
            link.symlink_to(target);self.tiled.append(link)
        self.assertEqual(len(self.tiled),32)

    def test_original_failure_is_reproduced_and_new_complete_paths_accept(self):
        with self.assertRaisesRegex(RuntimeError,'Reviewed generated numeric basename'):
            old.usage(self.work,compiler_root=self.compiler,compiler_assets=self.policy)
        actual=p.usage(self.work,compiler_root=self.compiler,compiler_assets=self.policy)
        expected=sum(max(f.lstat().st_size,f.lstat().st_blocks*512)
                     for f in self.work.rglob('*')if not f.is_dir())
        self.assertEqual(actual,expected)
        with patch.object(p.os,'statvfs',return_value=types.SimpleNamespace(f_bavail=128*2**30,f_frsize=1)):
            self.assertEqual(p.storage(self.spec,reserve=True)['nvme_used'],expected)
        rows=evidence.tree(self.work,lambda:None,spec=self.spec)
        kinds={}
        for row in rows.values():
            if row.get('compiler_asset'):
                kind=row['compiler_asset_kind'];kinds[kind]=kinds.get(kind,0)+1
                if row.get('compiler_generated'):
                    self.assertTrue(row['target']['content_hashed'])
                    target=Path(row['target']['path'])
                    self.assertEqual(row['target']['sha256'],hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertEqual(kinds,dict(generated_debug=17,installed_sglnk=17,
                                   generated_numeric=10,generated_tiled_transpose_numeric=32))

    def reject_both(self):
        for call in (lambda:p.usage(self.work,compiler_root=self.compiler,compiler_assets=self.policy),
                     lambda:evidence.tree(self.work,lambda:None,spec=self.spec)):
            with self.assertRaises(RuntimeError):call()

    def test_tiled_cross_unit_target_refused_in_both_paths(self):
        link=self.tiled[0];target=link.readlink()
        rogue=target.parent.parent.with_name('neuronxcc-rogue123')/target.parent.name/target.name
        rogue.parent.mkdir(parents=True);rogue.write_bytes(target.read_bytes())
        link.unlink();link.symlink_to(rogue)
        self.reject_both()

    def test_repeated_subgraph_id_mismatch_refused_in_both_paths(self):
        link=self.tiled[0]
        before=link.name
        parts=before.split('_')
        indexes=[i for i,value in enumerate(parts)if value.startswith('sg')]
        self.assertEqual(len(indexes),2)
        parts[indexes[1]]='sg0099'if parts[indexes[0]]!='sg0099'else'sg0098'
        link.rename(link.with_name('_'.join(parts)))
        self.reject_both()

    def test_generated_hardlink_target_refused_in_both_paths(self):
        link=self.tiled[0];target=link.readlink()
        alias=target.with_name('untrusted-hardlink.bin')
        alias.hardlink_to(target)
        self.reject_both()

    def test_new_tiled_target_bytes_are_actually_hashed(self):
        link=self.tiled[0]
        before=evidence.tree(self.work,lambda:None,spec=self.spec)[str(link)]
        target=link.readlink();target.write_bytes(target.read_bytes()+b' changed')
        after=evidence.tree(self.work,lambda:None,spec=self.spec)[str(link)]
        self.assertNotEqual(before['target']['sha256'],after['target']['sha256'])
        self.assertEqual(after['target']['sha256'],hashlib.sha256(target.read_bytes()).hexdigest())


if __name__=='__main__':
    unittest.main(verbosity=2)
