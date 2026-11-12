"""Pure source/metadata fixtures. Never import native stacks or inspect runtime."""
import builtins
import sys

BLOCKED={'torch','numpy','safetensors','transformers','ctypes','torch_xla','torch_neuronx',
         'neuronx_distributed','neuronx_distributed_inference','vllm','vllm_neuron',
         'tokenizers','ml_dtypes','huggingface_hub','packaging'}
original_import=builtins.__import__
def blocked_import(name,*args,**kwargs):
    if name.split('.')[0]in BLOCKED:raise AssertionError('Forbidden native/installed import: '+name)
    return original_import(name,*args,**kwargs)
builtins.__import__=blocked_import

import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import spark26_manifest_requirements_v2 as req
import spark26_native_manifest_generator_v2 as gen


def row(requires=(),extras=(),version='1.0',python='>=3.10'):
    return dict(version=version,requires=list(requires),extras=list(extras),requires_python=python)


class Requirements(unittest.TestCase):
    def test_normalized_names_extras_spec_and_marker(self):
        result=req.requirement('Some.Pkg[a-b,C_D]>=1.2; python_version < "3.11"')
        self.assertEqual(result,dict(name='some_pkg',extras=['a_b','c_d'],specifiers='>=1.2',
            marker='python_version < "3.11"',url=None))
        for name in ['-bad','bad-','a b']:
            with self.assertRaises(req.Unresolved):req.name(name)

    def test_release_padding_epoch_and_local(self):
        for actual,operator,wanted in [('2.8','==','2.8.0'),('2.8','==','2.8.0.*'),
            ('1!0.1','>','9999'),('2.8.0+CPU','==','2.8.0'),('1.0+CPU.01','==','1.0+cpu.1')]:
            with self.subTest(actual=actual,wanted=wanted):self.assertTrue(req.compare(actual,operator,wanted))
        self.assertFalse(req.compare('2.8.1','==','2.8.0.*'))

    def test_compatible_and_comma_specifiers(self):
        self.assertTrue(req.satisfies('2.8.4','(~=2.8.0,!=2.8.3)'))
        self.assertFalse(req.satisfies('2.9.0','~=2.8.0'))
        self.assertTrue(req.satisfies('2.9.0','~=2.8'))
        with self.assertRaises(req.Unresolved):req.satisfies('2.8','~=2')

    def test_installed_metadata_pep440_separator_spellings(self):
        self.assertTrue(req.satisfies('0.44.0','<0.45,>=0.44.0dev0'))
        self.assertTrue(req.satisfies('1.41.1','<2.0a.0,>=1.37.4'))
        self.assertEqual(req.version('2.0a.0')['key'],req.version('2.0a0')['key'])
        self.assertEqual(req.version('0.44.0dev0')['key'],req.version('0.44.0.dev0')['key'])
        for invalid in ['2.0aa0','2.0a..0','1.0+cpu..1','1.0+cpu.']:
            with self.assertRaises(req.Unresolved):req.version(invalid)

    def test_prerelease_dev_post_order_and_strict_exclusions(self):
        ordered=['1.0.dev1','1.0a1','1.0b1','1.0rc1','1.0','1.0.post1.dev1','1.0.post1']
        self.assertEqual(sorted(ordered,key=lambda v:req.version(v)['key']),ordered)
        self.assertFalse(req.compare('1.0rc1','<','1.0'))
        self.assertFalse(req.compare('1.0.post1','>','1.0'))
        self.assertTrue(req.compare('1.0.post2','>','1.0.post1'))

    def test_unsupported_syntax_fails_closed(self):
        for actual,spec in [('1.0','garbage'),('branch','>=1.0'),('1.0','>=1.*'),('1.0','>1.0+cpu')]:
            with self.subTest(actual=actual,spec=spec):
                with self.assertRaises(req.Unresolved):req.satisfies(actual,spec)

    def test_marker_precedence_parentheses_and_reversed_version(self):
        self.assertTrue(req.marker('sys_platform == "linux" or python_version < "3.0" and os_name == "nt"',gen.ENV,[]))
        self.assertFalse(req.marker('(sys_platform == "linux" or python_version < "3.0") and os_name == "nt"',gen.ENV,[]))
        self.assertTrue(req.marker('"3.9" < python_version',gen.ENV,[]))

    def test_marker_extra_and_string_membership(self):
        self.assertTrue(req.marker('extra == "a_B"',gen.ENV,['a_b']))
        self.assertFalse(req.marker('extra == "audio"',gen.ENV,[]))
        self.assertTrue(req.marker('sys_platform not in "win32 darwin"',gen.ENV,[]))

    def test_unknown_marker_three_valued_logic_and_no_version_string_fallback(self):
        self.assertIsNone(req.marker('future_variable == "yes"',gen.ENV,[]))
        self.assertFalse(req.marker('future_variable == "yes" and sys_platform == "darwin"',gen.ENV,[]))
        self.assertTrue(req.marker('future_variable == "yes" or sys_platform == "linux"',gen.ENV,[]))
        with self.assertRaises(req.Unresolved):req.marker('python_version > "unknown"',gen.ENV,[])

    def test_marker_cannot_execute_expression(self):
        with self.assertRaises(req.Unresolved):req.marker('__import__("os")',gen.ENV,[])


class DependencyGraph(unittest.TestCase):
    def graph(self,inventory):return gen.dependency_graph(inventory,{'a':'1.0'},gen.ENV,{})
    def test_cycles_and_extras_reach_fixed_point(self):
        graph=self.graph({'a':row(['b','c']), 'b':row(['d; extra == "fast"'],['fast']),
                          'c':row(['b[fast]','a']), 'd':row()})
        self.assertTrue(graph['complete_declared_dependency_resolution'])
        self.assertEqual(graph['selected'],{'a':[],'b':['fast'],'c':[],'d':[]})
        self.assertTrue(next(e for e in graph['edges']if e['parent']=='b')['active'])

    def test_inactive_platform_edges_retained_without_selecting_package(self):
        graph=self.graph({'a':row(['windows-only; sys_platform == "win32"','cuda-lib; platform_machine == "x86_64"']),
                          'cuda_lib':row()})
        self.assertEqual(set(graph['selected']),{'a','cuda_lib'})
        self.assertEqual(len(graph['edges']),2)

    def test_missing_and_unknown_env_are_unresolved(self):
        graph=self.graph({'a':row(['unknown; future_env == "yes"'])})
        self.assertEqual({r['kind']for r in graph['unresolved']},
                         {'missing_installed_distribution','unknown_marker_environment'})
        self.assertFalse(graph['complete_declared_dependency_resolution'])

    def test_mismatched_versions_requires_python_and_unknown_extra(self):
        graph=self.graph({'a':row(['b[missing]>=2.0']), 'b':row(version='1.0',python='>=3.12')})
        self.assertEqual({r['kind']for r in graph['unresolved']},
                         {'installed_version_mismatch','requires_python_mismatch','undeclared_requested_extras'})

    def test_direct_url_and_unknown_installed_version_are_not_silent(self):
        graph=self.graph({'a':row(['b @ https://example.invalid/b.whl']), 'b':row(version='branch')})
        self.assertEqual({r['kind']for r in graph['unresolved']},
                         {'direct_requirement_URL_not_resolved','unsupported_installed_version'})


class SourceMetadata(unittest.TestCase):
    def test_text_reader_stable_reference(self):
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'a.py';path.write_text('x=1\n');reader=gen.TextReader()
            raw=reader.raw(path);self.assertEqual(reader.references[str(path)]['sha256'],gen.digest(raw))
            path.write_text('x=2\n')
            with self.assertRaises(ValueError):reader.raw(path)

    def test_text_reader_refuses_binary_weight_before_open(self):
        with mock.patch.object(gen.os,'open',side_effect=AssertionError('Must never open')):
            for path in ['/tmp/a.so','/tmp/model.safetensors','/tmp/a.bin','/tmp/a.pyc']:
                with self.subTest(path=path):
                    with self.assertRaises(ValueError):gen.TextReader().raw(path)

    def test_text_reader_refuses_symlink(self):
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'a.py';path.write_text('x=1\n');link=Path(folder)/'link.py';link.symlink_to(path)
            with self.assertRaises(ValueError):gen.TextReader().raw(link)

    def test_independent_record_parser_agrees_with_frozen_provider(self):
        sha=base64.urlsafe_b64encode(b'a'*32).decode().rstrip('=')
        raw=f'pkg/a.py,sha256={sha},3\npkg/lib.so,sha256={sha},4\npkg/data.txt,sha256={sha},5\n'
        raw+='../../../bin/tool,,\npkg/__pycache__/a.pyc,,\npkg-1.dist-info/RECORD,,\n'
        self.assertEqual(gen.record_entries(raw),gen.provider.record_payloads(raw))
        self.assertEqual(sum(r['bytes']for r in gen.record_entries(raw).values()),12)

    def test_record_rejects_unhashed_bad_hash_duplicate_and_bad_csv(self):
        sha=base64.urlsafe_b64encode(b'a'*32).decode()
        valid=f'pkg/a.py,sha256={sha},3\n'
        for raw in ['pkg/a.py,,\n','pkg/a.py,sha256=YQ,1\n',valid+valid,'pkg/a.py,extra,fields,wrong\n']:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):gen.record_entries(raw)

    def test_python_override_requires_exact_preexisting_review_pin(self):
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'a.py';path.write_text('new=1\n');raw=path.read_bytes()
            payload={str(path):dict(sha256='a'*64,bytes=1)}
            changed,overrides,pins,issues,checked=gen.reviewed_python_payloads(payload,gen.TextReader(),{})
            self.assertEqual(changed,payload);self.assertFalse(overrides);self.assertEqual(checked,1)
            self.assertEqual(issues[0]['kind'],'unreviewed_modified_Python_payload')
            changed,overrides,pins,issues,checked=gen.reviewed_python_payloads(payload,gen.TextReader(),
                {str(path):{gen.digest(raw):[{'independent_fixture':True}]}})
            self.assertEqual(changed,overrides);self.assertFalse(issues);self.assertEqual(pins[str(path)],gen.digest(raw))

    def test_binary_payload_is_not_opened_or_overridden(self):
        payload={'/tmp/a.so':dict(sha256='a'*64,bytes=1)}
        with mock.patch.object(gen.TextReader,'raw',side_effect=AssertionError('Binary read')):
            changed,overrides,pins,issues,checked=gen.reviewed_python_payloads(payload,gen.TextReader(),{})
        self.assertEqual(changed,payload);self.assertFalse(overrides);self.assertEqual(checked,0)

    def test_unselected_wheel_cannot_hide_conflicting_record_ownership(self):
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'other-1.dist-info';path.mkdir()
            encoded=base64.urlsafe_b64encode(b'b'*32).decode().rstrip('=')
            (path/'RECORD').write_text('cv2/lib.so,sha256='+encoded+',3\n')
            target=str(gen.SITE/'cv2/lib.so');selected={target:dict(sha256='a'*64,bytes=3)}
            result=gen.ambient_record_conflicts({'other':dict(directory=str(path))},set(),selected,gen.TextReader())
            self.assertEqual(result['issues'][0]['kind'],'ambient_conflicting_RECORD_owner')
            self.assertEqual(result['issues'][0]['path'],target)
            selected[target]['sha256']=(b'b'*32).hex()
            result=gen.ambient_record_conflicts({'other':dict(directory=str(path))},set(),selected,gen.TextReader())
            self.assertFalse(result['issues']);self.assertEqual(len(result['identical_duplicate_owners']),1)

    def test_editable_literal_mapping_read_without_execution(self):
        with tempfile.TemporaryDirectory()as folder:
            base=Path(folder);site=base/'site';site.mkdir();root=base/'editable';package=root/'vllm_neuron';package.mkdir(parents=True)
            (package/'__init__.py').write_text('raise RuntimeError("MUST NOT EXECUTE")\n')
            info=site/'vllm_neuron-1.dist-info';info.mkdir()
            (info/'direct_url.json').write_text(json.dumps(dict(dir_info=dict(editable=True),url='file://'+str(root))))
            finder=site/'__editable___fixture_finder.py'
            finder.write_text('MAPPING: dict = '+repr({'vllm_neuron':str(package),'test':str(root/'test')})+'\nraise RuntimeError("MUST NOT EXECUTE")\n')
            (info/'RECORD').write_text(finder.name+',,\n')
            reader=gen.TextReader()
            with mock.patch.object(gen,'SITE',site),mock.patch.object(gen.provider,'EDITABLE',root):
                files,origin,issues=gen.editable_files(dict(directory=str(info),name='vllm_neuron'),reader)
            self.assertEqual(len(files),1);self.assertFalse(origin['pth_executed']);self.assertFalse(issues)
            self.assertIn(str(info/'RECORD'),reader.references)
            with self.assertRaises(ValueError):gen.editable_files(dict(directory=str(info),name='wrong'),gen.TextReader())


class SavedElf(unittest.TestCase):
    def fixture(self):
        path=str(gen.SITE/'pkg/lib.so');entry=dict(sha256='a'*64,bytes=3)
        return path,entry,{path:dict(**entry,elf=dict(needed=[]),evidence={'saved':True})}

    def test_exact_path_hash_size_evidence_only(self):
        path,entry,saved=self.fixture();linked,result=gen.elf_closure({path:entry},saved)
        self.assertTrue(result['complete']);self.assertEqual(result['exact_saved_objects'],1)
        for bad in [{'/other/lib.so':saved[path]}, {path:dict(saved[path],sha256='b'*64)}, {path:dict(saved[path],bytes=4)}]:
            linked,result=gen.elf_closure({path:entry},bad)
            self.assertFalse(result['complete']);self.assertFalse(linked)

    def test_missing_needed_metadata_not_empty_dependency_proof(self):
        path,entry,saved=self.fixture();saved[path]['elf']={}
        _,result=gen.elf_closure({path:entry},saved)
        self.assertEqual(result['unresolved'][0]['kind'],'incomplete_saved_ELF_metadata')

    def test_other_venv_soname_candidate_never_transplanted(self):
        path,entry,saved=self.fixture();saved[path]['elf']['needed']=['libdep.so']
        target='/other/venv/lib/python3.12/site-packages/pkg/libdep.so'
        saved[target]=dict(**entry,elf=dict(needed=[]),evidence={'saved':True})
        linked,result=gen.elf_closure({path:entry},saved)
        self.assertFalse(linked);self.assertFalse(result['complete'])
        self.assertIn('other_environment_ELF_candidate_rejected',{r['kind']for r in result['unresolved']})

    def test_ambiguous_and_unknown_loader_selection_stay_unresolved(self):
        path,entry,saved=self.fixture();saved[path]['elf']['needed']=['libdep.so']
        for target in ['/opt/one/libdep.so','/opt/two/libdep.so']:
            saved[target]=dict(**entry,elf=dict(needed=[]),evidence={'saved':True})
        linked,result=gen.elf_closure({path:entry},saved)
        self.assertFalse(linked);self.assertEqual(result['unresolved'][0]['kind'],'unresolved_or_ambiguous_saved_ELF_dependency')
        del saved['/opt/two/libdep.so']
        linked,result=gen.elf_closure({path:entry},saved)
        self.assertEqual(set(linked),{'/opt/one/libdep.so'});self.assertFalse(result['complete'])
        self.assertEqual(result['unresolved'][0]['kind'],'saved_ELF_candidate_selection_not_proven')

    def test_system_candidate_boundary_does_not_prove_actual_selection(self):
        path,entry,saved=self.fixture();saved[path]['elf']['needed']=['libc.so.6']
        saved['/usr/lib/libc.so.6']=dict(**entry,elf=dict(needed=[]),evidence={'saved':True})
        linked,result=gen.elf_closure({path:entry},saved)
        self.assertFalse(linked);self.assertFalse(result['complete']);self.assertTrue(result['edges'][0]['host_system_ELF_boundary'])


class FrozenRuntimeManifest(unittest.TestCase):
    def test_exact_schema_and_independent_reconstruction(self):
        path=str(gen.SITE/'pkg/a.py');entry=dict(sha256='a'*64,bytes=3)
        encoded=base64.urlsafe_b64encode(bytes.fromhex(entry['sha256'])).decode().rstrip('=')
        raw='pkg/a.py,sha256='+encoded+',3\n';record=str(gen.SITE/'pkg-1.0.dist-info/RECORD')
        pins={record:gen.digest(raw.encode())};payloads=gen.record_entries(raw)
        manifest=dict(schema='spark26_native_provider_manifest_v1',distributions=[dict(name='pkg',version='1.0',
            record=dict(path=record,sha256=pins[record]))],python_overrides={},additional_files={},linked_native_files={},
            payload_count=1,payload_bytes=3,payload_sha256=gen.provider.contract.digest(payloads))
        with mock.patch.object(gen.provider,'REQUIRED',{'pkg':'1.0'}),mock.patch.object(gen.provider.contract,'NATIVE_SOURCES',{}):
            self.assertEqual(gen.provider.merge_manifest(manifest,pins,lambda ref:raw),payloads)
            manifest['payload_bytes']+=1
            with self.assertRaises(ValueError):gen.provider.merge_manifest(manifest,pins,lambda ref:raw)

    def test_no_forbidden_imports(self):self.assertFalse(BLOCKED&set(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
