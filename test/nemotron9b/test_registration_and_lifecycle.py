# SPDX-License-Identifier: Apache-2.0
"""Execute lifecycle/control code with inert stdlib objects, never native libraries."""
import ast
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from vllm_neuron.models.nemotron9b import identity as i, prepare_view as v
from vllm_neuron.models.nemotron9b.provenance import ORIGINAL_CONFIG

P = Path(i.__file__).parent
ENV = {'NEURON_NEMOTRON9B': '1', 'XLA_DISABLE_FUNCTIONALIZATION': '1'}


@contextmanager
def registration_fixture(collision=None):
    events = []
    modules = {}
    def module(name, **values):
        obj = ModuleType(name)
        obj.__dict__.update(values)
        modules[name] = obj
        return obj
    package = module('_nemotron_test', __path__=[str(P)])
    module('_nemotron_test.identity', require=i.require, selected_variant=i.selected_variant)
    class Config: pass
    class CausalLM:
        def __init__(self, config): self.config = config
        def forward(self, *args, **kwargs): return args, kwargs
    class Model: pass
    original = lambda *a: None
    mamba = SimpleNamespace(forward=original)
    types = {}
    configs = {}
    archs = []
    def install_nxdi():
        events.append('nxdi')
        types['nemotron9bneurontext'] = {'causal-lm': Model}
    def install_cache():
        events.append('cache')
        mamba.forward = lambda *a: 'installed'
    core = module('_nemotron_test.text_core', Mamba=mamba)
    cache = module('_nemotron_test.cache', ORIGINAL_FORWARD=original, install=install_cache)
    package.text_core, package.cache = core, cache
    module('_nemotron_test.modeling', install_nxdi=install_nxdi, NeuronNemotron9BForCausalLM=Model)
    module('transformers', PretrainedConfig=Config, AutoConfig=SimpleNamespace(
        register=lambda name, cls: (events.append('config'), configs.update({name: cls}))))
    module('transformers.models.auto.configuration_auto', CONFIG_MAPPING=configs)
    module('vllm.model_executor.models.registry', ModelRegistry=SimpleNamespace(
        get_supported_archs=lambda: archs,
        register_model=lambda name, cls: (events.append('vllm'), archs.append(name))))
    module('vllm_neuron.worker.neuronx_distributed_model_loader', NeuronCausalLM=CausalLM)
    module('neuronx_distributed_inference.utils.constants', MODEL_TYPES=types)
    if collision == 'config': configs['nemotron9b_neuron_text'] = object()
    if collision == 'vllm': archs.append('Nemotron9BNeuronTextForCausalLM')
    if collision == 'nxdi': types['nemotron9bneurontext'] = {'causal-lm': object()}
    if collision == 'cache': mamba.forward = object()
    with patch.dict(sys.modules, modules), patch.dict(os.environ, ENV, clear=True):
        spec = importlib.util.spec_from_file_location('_nemotron_test.registration', P/'registration.py')
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        yield loaded, events


class RegistrationTests(unittest.TestCase):
    def test_registration_order_idempotence_and_scheduler_flags(self):
        with registration_fixture() as (module, events):
            module.install()
            module.install()
            self.assertEqual(events, ['cache', 'nxdi', 'config', 'vllm'])
            cls = module.Nemotron9BNeuronTextForCausalLM
            self.assertFalse(cls.has_inner_state)
            self.assertFalse(cls.is_hybrid)
            model = cls(vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config='config')))
            self.assertEqual(model.forward('ids', 'positions', 'slots'),
                             (('ids', 'slots'), {'position_ids': 'positions'}))

    def test_every_collision_fails_before_any_registration_mutation(self):
        for collision in ('config', 'vllm', 'nxdi', 'cache'):
            with self.subTest(collision=collision), registration_fixture(collision) as (module, events):
                with self.assertRaises(i.AdmissionError): module.install()
                self.assertEqual(events, [])
                self.assertFalse(module._INSTALLED)

    def test_opt_in_checked_even_after_registration(self):
        with registration_fixture() as (module, events):
            module.install()
            with patch.dict(os.environ, {'NEURON_CPU_SAMPLER_FAST': '1'}):
                with self.assertRaises(i.AdmissionError): module.install()


class PreparationTests(unittest.TestCase):
    def test_failed_tokenizer_validation_retains_unadmitted_partial_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/'source'; source.mkdir()
            dest = Path(temporary)/'view'
            (source/'config.json').write_text(json.dumps(ORIGINAL_CONFIG))
            (source/'tokenizer_config.json').write_text(json.dumps({'tokenizer_class':'PreTrainedTokenizer'}))
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            def digest(path):
                return i.CONFIG_VIEW_SHA256 if path.name=='config.json' else i.TOKENIZER_VIEW_SHA256
            with patch.object(v,'verify_files'), patch.object(v,'digest_file',side_effect=digest), \
                    patch.object(v,'load_tokenizer',side_effect=i.AdmissionError('mock validation failure')):
                with self.assertRaises(i.AdmissionError): v.prepare(source,dest)
                self.assertTrue(dest.is_dir())
                self.assertTrue((dest/'config.json').is_file())
                self.assertFalse((dest/i.VIEW_MANIFEST).exists())
                with self.assertRaisesRegex(i.AdmissionError,'Fresh destination'): v.prepare(source,dest)
            self.assertEqual(before,{p.name:p.read_bytes()for p in source.iterdir()})

    def test_source_fingerprint_includes_shipped_contracts(self):
        sources = i.source_identity()
        for name in ('vllm_neuron/models/nemotron9b/core_contract.json',
                     'vllm_neuron/models/nemotron9b/neuron_config_contract.json',
                     'vllm_neuron/models/nemotron9b/tokenizer_fixtures.json',
                     'vllm_neuron/models/nemotron9b/licenses/GSM8K-MIT.txt'):
            self.assertIn(name,sources)
        self.assertEqual(i.SOFTWARE['torch-xla'],'2.8.1')


class ForwardingTests(unittest.TestCase):
    def test_one_token_prefill_and_decode_keep_original_routing(self):
        class Tensor:
            dtype = 'int64'
            def __init__(self,values): self.values=values; self.shape=(len(values),len(values[0]))
            def __eq__(self,value): return SimpleNamespace(all=lambda:all(x==value for row in self.values for x in row))
            def expand(self,batch,width): return Tensor(self.values*batch)
        def pad(tensor,amount,value):
            self.assertEqual(amount,(0,1))
            return Tensor([row+[value]for row in tensor.values])
        class Base:
            def forward(self,**kwargs): return kwargs
        tree=ast.parse((P/'helpers.py').read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='ExplicitPositionForCausalLM')
        namespace={'NeuronBaseForCausalLM':Base,'F':SimpleNamespace(pad=pad),
                   'torch':SimpleNamespace(tensor=lambda values,dtype:Tensor([values]))}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'<pinned forwarding>','exec'),namespace)
        model=namespace['ExplicitPositionForCausalLM'](); model.config=SimpleNamespace(pad_token_id=7)
        ids=Tensor([[5],[6]]); positions=Tensor([[0],[0]])
        got=model.forward(input_ids=ids,position_ids=positions,seq_ids='slots')
        self.assertEqual(got['input_ids'].values,[[5,7],[6,7]])
        self.assertEqual(got['attention_mask'].values,[[1,0],[1,0]])
        self.assertEqual(got['seq_ids'],'slots')
        for values in ([[1],[2]], [[0],[3]]):
            positions=Tensor(values)
            got=model.forward(input_ids=ids,position_ids=positions,attention_mask='mask')
            self.assertIs(got['input_ids'],ids)
            self.assertEqual(got['attention_mask'],'mask')
        with self.assertRaises(ValueError): model.forward(input_ids=ids)


if __name__=='__main__': unittest.main()
