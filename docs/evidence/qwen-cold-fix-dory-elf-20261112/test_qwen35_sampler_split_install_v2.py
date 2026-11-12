"""Actual frozen body/installer AST with synthetic imports, no native or payload IO."""
import ast
import builtins
import copy
import hashlib
import inspect
from pathlib import Path
import sys
from types import FunctionType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
ROOT = Q.parent
sys.path[:0] = [str(Q), str(ROOT)]
REAL_IMPORT = builtins.__import__


def deny_native(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','numpy','nki','neuronxcc','vllm','vllm_neuron','transformers',
                              'torch_neuronx','neuronx_distributed_inference'}:
        raise AssertionError('No native imports in source-only checks: ' + name)
    return REAL_IMPORT(name, *args, **kwargs)


with patch.object(builtins, '__import__', deny_native):
    import qwen35_sampler_native_leaves_v1 as old
    import qwen35_sampler_native_leaves_v2 as new

PINS = {
    'research/qwen35_sampler_native_leaves_v1.py': '7c4aca1a47627d35c5f43ae64144763c24a5b69a3e8683a8794bdbcc051cfe4b',
    'frontend_split_correction_v2.py': '1845febc4e220001cebc562b075538f7d0233118dd33de8b695b210e78ff735d',
    'diagnose_qwen35_state_reset_cold_v1.py': '339cc077ae3235522e0454a187df09f4600139d8dcec9b2e52444f895b38ad8f',
    'probe_qwen35_state_copy_v1.py': '99a6c46cfd98a99e17540937143713288cf1c6a166ae8d18104470e8df081051',
    'validate_real_logits.py': '1f77e72f1cfcd8d6ee7253daea78e8d598cb7c07d9c98acc7ece9e1e4b41161f',
    'validate_qwen35_long_decode.py': 'b9f4bf6d1db700c0b56c57ec3b9917a03487a665256087df1fd2bbc8aa8cab9f',
}


def source(name):
    value = (ROOT/name).read_bytes()
    assert hashlib.sha256(value).hexdigest() == PINS[name]
    return value


def function(name, function_name, namespace):
    tree = ast.parse(source(name))
    node, = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function_name]
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/name),'exec'),namespace)
    return namespace[function_name]


class StopBeforePayload(RuntimeError):
    pass


class Fixture:
    def __init__(self, phase, *, cache=True, preinstalled=False):
        self.events=[];self.phase=phase;self.cache=cache;self.directory=Path('/source-only/no-output')
        self.case={'model_path':'/source-only/no-model','artifact':'/source-only/no-artifact'}
        namespace={};exec(compile('def forward(self): return None\n',str(ROOT/'qwen35_text_port.py'),'exec'),namespace)
        self.cls=type('LinearAttention',(),{'forward':namespace['forward']})
        def admission(family): self.events.append('primitive_admission')
        def clone(forward,count):
            assert count==1
            self.events.append('split_install')
            replacement={};exec(compile('def forward(self): return None\n',str(ROOT/'frontend_split_correction_v2.py'),'exec'),replacement)
            return replacement['forward'],['one-original-split']
        globals_=dict(Path=Path,ROOT=ROOT,inspect=inspect,primitive_admission=admission,clone_forward=clone)
        self.install=function('frontend_split_correction_v2.py','install',globals_)
        def before_payload(*args,**kwargs):
            self.events.append('original_body_boundary')
            raise StopBeforePayload('Actual body reached before payload/native operations')
        self.modules={
            'frontend_split_correction_v2':SimpleNamespace(install=self.install),
            'qwen35_text_port':SimpleNamespace(LinearAttention=self.cls),
            'orchestrate':SimpleNamespace(validate_cache=self.validate_cache),
            'qwen35_split_runtime_v4':SimpleNamespace(install_runner_hooks=lambda runner:self.events.append('runner_hooks')),
            'torch':SimpleNamespace(set_num_threads=before_payload),
            'server_with_bias_fix':SimpleNamespace(),
            'qwen35_vllm_registration':SimpleNamespace(install=lambda:self.events.append('model_registration')),
            'qwen35_neuron_adapter':SimpleNamespace(NeuronQwen35ForCausalLM=object()),
            'vllm_neuron.worker.neuronx_distributed_model_loader':SimpleNamespace(NeuronModelBase=object()),
            'validate_real_logits':SimpleNamespace(metrics=object(),neuron=before_payload),
            'validate_qwen35_long_decode':SimpleNamespace(passed=object(),CRITERIA={},neuron=before_payload),
            'qwen35_split_state_checks_v4':SimpleNamespace(raw_zero_states=object(),graph_routes=object()),
            'qwen35_state_inplace_reset_v1':SimpleNamespace(clear_states=object(),verify_cold_reproduction=object()),
        }
        cold=ModuleType('diagnose_qwen35_state_reset_cold_v1')
        cold.DEST=Path('/original/frozen-destination')
        function('diagnose_qwen35_state_reset_cold_v1.py','main',vars(cold))
        state=ModuleType('probe_qwen35_state_copy_v1')
        state.DEST=Path('/original/frozen-destination');state.verify=before_payload
        self.originals={'diagnose_qwen35_state_reset_cold_v1':cold,'probe_qwen35_state_copy_v1':state}
        self.cold_code=cold.main.__code__
        self.preinstalled=preinstalled

    def importer(self,name,*args,**kwargs):
        if name in self.modules:return self.modules[name]
        return deny_native(name,*args,**kwargs)

    def validate_cache(self,case):
        assert case==self.case;self.events.append('cache_check');return self.cache

    def read(self,path):
        assert path==self.directory/'case.json','No payload or result IO expected'
        self.events.append('case_read');return copy.deepcopy(self.case)

    def run(self,leaf):
        def clone(code,globals_,name):
            assert code is self.cold_code
            assert globals_['DEST']==self.directory
            before=vars(self.originals['diagnose_qwen35_state_reset_cold_v1'])
            assert {k:v for k,v in globals_.items() if k!='DEST'}=={k:v for k,v in before.items() if k!='DEST'}
            self.events.append('unchanged_cold_code_DEST_only')
            return FunctionType(code,globals_,name)
        with patch.object(builtins,'__import__',self.importer),\
             patch.object(leaf,'nvme_destination',side_effect=lambda path:Path(path)),\
             patch.object(leaf,'expected_case',return_value=self.case),\
             patch.object(leaf,'read_small',side_effect=self.read),\
             patch.object(leaf,'checked_source',side_effect=source),\
             patch.object(leaf.importlib,'import_module',side_effect=lambda name:self.originals[name]),\
             patch.object(leaf,'FunctionType',side_effect=clone):
            if self.preinstalled:self.install('qwen35')
            leaf.run_model_stage(self.phase,self.directory,lambda:{'admitted':True,'exclusive':True})


class Tests(unittest.TestCase):
    def test_exact_delta_only_moves_wrapper_install_under_not_cold(self):
        before=ast.parse(source('research/qwen35_sampler_native_leaves_v1.py'))
        after=ast.parse(Path(new.__file__).read_bytes())
        fn=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='run_model_stage')
        conditional,=[n for n in fn.body if isinstance(n,ast.If) and ast.unparse(n.test)=="phase != 'cold'"]
        self.assertEqual(ast.unparse(conditional.body[0]),'from frontend_split_correction_v2 import install')
        self.assertEqual(ast.unparse(conditional.body[1]),"install('qwen35')")
        index=fn.body.index(conditional);fn.body[index:index+1]=conditional.body
        self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))
        for path in PINS:source(path)

    def test_frozen_installer_rejects_old_wrapper_cold_double_install(self):
        fixture=Fixture('cold')
        with self.assertRaises(AssertionError):fixture.run(old)
        self.assertEqual(fixture.events.count('split_install'),1)
        self.assertEqual(fixture.events.count('primitive_admission'),2)
        self.assertNotIn('original_body_boundary',fixture.events)

    def test_cold_original_full_main_installs_once_after_cache_check(self):
        fixture=Fixture('cold')
        with self.assertRaises(StopBeforePayload):fixture.run(new)
        self.assertEqual(fixture.events.count('split_install'),1)
        self.assertEqual(fixture.events.count('primitive_admission'),1)
        self.assertLess(fixture.events.index('cache_check'),fixture.events.index('split_install'))
        self.assertIn('unchanged_cold_code_DEST_only',fixture.events)
        self.assertEqual(fixture.originals['diagnose_qwen35_state_reset_cold_v1'].DEST,Path('/original/frozen-destination'))

    def test_other_three_phases_keep_wrapper_install_once(self):
        for phase in ('state_copy','numeric','long'):
            fixture=Fixture(phase)
            with self.subTest(phase=phase),self.assertRaises(StopBeforePayload):fixture.run(new)
            self.assertEqual(fixture.events.count('split_install'),1)
            self.assertLess(fixture.events.index('split_install'),fixture.events.index('cache_check'))

    def test_preinstalled_cold_still_fails_strict_installer(self):
        fixture=Fixture('cold',preinstalled=True)
        with self.assertRaises(AssertionError):fixture.run(new)
        self.assertEqual(fixture.events.count('primitive_admission'),2)
        self.assertNotIn('original_body_boundary',fixture.events)

    def test_cold_bad_cache_stops_before_original_body_or_install(self):
        fixture=Fixture('cold',cache=False)
        with self.assertRaisesRegex(ValueError,'frozen graph cache invalid'):fixture.run(new)
        self.assertNotIn('split_install',fixture.events)
        self.assertNotIn('unchanged_cold_code_DEST_only',fixture.events)

    def test_missing_admission_and_unknown_phase_rejected_before_install(self):
        with patch.object(new,'read_small') as read:
            with self.assertRaises(ValueError):new.run_model_stage('cold','/unused',lambda:{'admitted':False})
            read.assert_not_called()
        fixture=Fixture('unknown')
        with self.assertRaisesRegex(ValueError,'invalid/reused model stage'):fixture.run(new)
        self.assertNotIn('split_install',fixture.events)

    def test_scientific_body_and_other_phase_install_ownership_static(self):
        cold=ast.parse(source('diagnose_qwen35_state_reset_cold_v1.py'))
        main=next(n for n in cold.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        calls=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='correct_split']
        self.assertEqual(len(calls),1);self.assertEqual(ast.literal_eval(calls[0].args[0]),'qwen35')
        for name in ('probe_qwen35_state_copy_v1.py','validate_real_logits.py','validate_qwen35_long_decode.py'):
            tree=ast.parse(source(name))
            self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module=='frontend_split_correction_v2' for n in ast.walk(tree)))
        assertions=[ast.unparse(n) for n in ast.walk(main) if isinstance(n,ast.Assert)]
        self.assertIn('assert len(rows) == len(captured) == 84',assertions)
        self.assertFalse(any(name in sys.modules for name in ('torch','numpy','vllm','nki','neuronxcc')))


if __name__=='__main__':unittest.main(verbosity=2)
