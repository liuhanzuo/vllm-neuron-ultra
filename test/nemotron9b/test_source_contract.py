# SPDX-License-Identifier: Apache-2.0
"""Real source equivalence and tokenizer contracts, stdlib mocks only."""
import ast
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from vllm_neuron.models.nemotron9b import identity as i, prepare_view as v, register
P=Path(i.__file__).parent


class UndoPackaging(ast.NodeTransformer):
    def visit_ClassDef(self,node):
        self.generic_visit(node)
        if node.name=='NeuronNemotron9BForCausalLM':
            assert len(node.bases)==1 and ast.unparse(node.bases[0])=='ExplicitPositionForCausalLM'
            node.bases=[ast.Name(id='NeuronQwen35ForCausalLM',ctx=ast.Load())]
        return node
    def visit_ImportFrom(self,node):
        relocations={'nki_read_torch':'slot_cache_nki_read_torch',
                     'nki_write_torch':'slot_cache_nki_predicated_torch'}
        if node.level==1 and node.module in relocations:
            node.module=relocations[node.module]
            node.level=0
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='split_last_dim':
            assert len(node.args)==2 and not node.keywords
            return ast.Call(func=ast.Attribute(value=node.args[0],attr='split',ctx=ast.Load()),args=[node.args[1]],keywords=[ast.keyword(arg='dim',value=ast.UnaryOp(op=ast.USub(),operand=ast.Constant(value=1)))])
        return node


class SourceTests(unittest.TestCase):
    def test_every_frozen_math_class_and_function_matches(self):
        contract=json.loads((P/'core_contract.json').read_text())
        for row in contract['entries']:
            tree=ast.parse((P/row['target']).read_text())
            container=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name==row['parent'])if 'parent'in row else tree
            node=next(x for x in container.body if isinstance(x,(ast.ClassDef,ast.FunctionDef))and x.name==row['name'])
            normalized=node if row.get('exact')else UndoPackaging().visit(node)
            got=hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()
            with self.subTest(row=row):self.assertEqual(got,row['ast_sha256'])

    def test_frozen_ast_rejects_math_index_dtype_and_import_mutations(self):
        contract=json.loads((P/'core_contract.json').read_text())
        for filename,name,old,new in (
                ('cache.py','mamba_forward','self.groups * self.sd','self.groups + self.sd'),
                ('cache.py','mamba_forward','self.conv1d.weight[:, 0]','self.conv1d.weight[:, 1]'),
                ('modeling.py','NeuronNemotron9BModel','dtype=torch.float32','dtype=torch.bfloat16'),
                ('cache.py','read_cache','from .nki_read_torch','from .wrong_module')):
            source=(P/filename).read_text();self.assertIn(old,source)
            tree=ast.parse(source.replace(old,new,1))
            node=next(x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef))and x.name==name)
            got=hashlib.sha256(ast.dump(UndoPackaging().visit(node),include_attributes=False).encode()).hexdigest()
            expected=next(x['ast_sha256']for x in contract['entries']if x['target']==filename and x['name']==name)
            with self.subTest(mutation=old):self.assertNotEqual(got,expected)

    def test_only_exact_split_and_parent_relocations_are_normalized(self):
        for source in ('split_last_dim(x, [1, 2], dim=0)',
                       'split_last_dim(x, [1, 2], extra)',
                       'class NeuronNemotron9BForCausalLM(WrongParent): pass'):
            with self.subTest(source=source),self.assertRaises(AssertionError):
                UndoPackaging().visit(ast.parse(source))

    def test_four_split_replacements_and_no_runtime_ast_rewrite(self):
        calls=[]
        for name in ('text_core.py','cache.py'):
            tree=ast.parse((P/name).read_text())
            calls += [x for x in ast.walk(tree)if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='split_last_dim']
            self.assertFalse(any(isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id in ('exec','eval')for x in ast.walk(tree)))
        self.assertEqual(len(calls),4)

    def test_disabled_plugin_imports_no_native_or_model(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            self.assertFalse(name.startswith(('torch','transformers','neuron','vllm.model')))
            return original(name,*args,**kwargs)
        with patch.dict(os.environ,{},clear=True),patch('builtins.__import__',side_effect=guarded):register()

    def test_model_compatibility_transform_is_narrow(self):
        from vllm_neuron.models.nemotron9b.provenance import ORIGINAL_CONFIG
        source=copy.deepcopy(ORIGINAL_CONFIG);out=v.compatible_model_config(source)
        self.assertEqual(source,ORIGINAL_CONFIG)
        self.assertEqual({k for k in set(out)|set(source)if out.get(k)!=source.get(k)}, {'auto_map','model_type','architectures','time_step_limit'})
        self.assertEqual(out['time_step_limit'],[0.,float('inf')])
        wrong=copy.deepcopy(source);wrong['hidden_size']=1
        with self.assertRaises(i.AdmissionError):v.compatible_model_config(wrong)

    def test_tokenizer_transform_only_class(self):
        x={'tokenizer_class':'PreTrainedTokenizer','chat_template':'unchanged','extra_special_tokens':[]}
        y=v.compatible_tokenizer_config(x)
        self.assertEqual(y,dict(x,tokenizer_class='PreTrainedTokenizerFast'));self.assertEqual(x['tokenizer_class'],'PreTrainedTokenizer')

    def test_complete_fixture_protocol_and_thinking_behavior(self):
        fixture=json.loads((P/'tokenizer_fixtures.json').read_text());prompts={x['prompt']:x for x in fixture['fixtures']};raw={(x['text'],x['add_special_tokens']):x for x in fixture['plain']};decode={(tuple(x['input_ids']),skip):x['decode_skip_special'if skip else'decode']for x in fixture['plain']for skip in (False,True)}
        class Tok:
            all_special_ids=fixture['special_token_ids']
            def apply_chat_template(self,messages,tokenize,**kw):
                text=messages[0]['content']
                return prompts[text]['input_ids']if tokenize else '<think></think>'if text.endswith('/no_think')else'<think>\n'
            def encode(self,text,add_special_tokens):return raw[text,add_special_tokens]['input_ids']
            def decode(self,ids,skip_special_tokens):return decode[tuple(ids),skip_special_tokens]
            def get_vocab(self):return {'inert':1}
        with patch.object(v.hashlib,'sha256',return_value=SimpleNamespace(hexdigest=lambda:fixture['vocabulary_sha256'])):
            self.assertEqual(v.validate_tokenizer(Tok(),fixture),{'quality_inputs':96,'raw_checks':22,'thinking_controls':2})
            bad=copy.deepcopy(fixture);bad['fixtures'].pop()
            with self.assertRaises(i.AdmissionError):v.validate_tokenizer(Tok(),bad)
            with patch.object(Tok,'apply_chat_template',return_value=[]):
                with self.assertRaises(i.AdmissionError):v.validate_tokenizer(Tok(),fixture)

    def test_existing_view_rejected_before_payload_or_tokenizer(self):
        with tempfile.TemporaryDirectory()as path,patch.object(v,'verify_files',side_effect=AssertionError('No payload reads')),patch.object(v,'load_tokenizer',side_effect=AssertionError('No tokenizer')):
            with self.assertRaises(i.AdmissionError):v.prepare(path,path)

    def test_registered_model_routes_before_generic_fallback(self):
        loader=P.parents[1]/'worker/neuronx_distributed_model_loader.py';s=loader.read_text()
        self.assertLess(s.index('if getattr(config, "model_type", None) == "nemotron9b_neuron_text"'),s.index('# If fused speculation'))
        self.assertIn('vllm.general_plugins',(P.parents[2]/'setup.py').read_text())


if __name__=='__main__':unittest.main()
