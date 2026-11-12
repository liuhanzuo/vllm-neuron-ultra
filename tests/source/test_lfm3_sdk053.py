# SPDX-License-Identifier: Apache-2.0
"""stdlib-only, full-module/mock source tests; NOT native or model validation."""
import ast
import builtins
import contextlib
import copy
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vllm_neuron.models import lfm3
from vllm_neuron.models.lfm3 import identity as ident, serving, artifacts, runtime

ENV = {"NEURON_LFM3": "1", "NEURON_CPU_SAMPLER_FAST": "1",
       "NEURON_LFM3_CAPACITY": "4", "NEURON_LFM3_CONTEXT": "2048",
       "NEURON_LFM3_NKI_WRITER": "0", "NEURON_LFM3_EXPERIMENTAL_RUNTIME": "1",
       "NEURON_PLATFORM_TARGET_OVERRIDE": "trn1", "NEURON_LOGICAL_NC_CONFIG": "1"}
NS = types.SimpleNamespace


def vconfig(batch=4, context=2048):
    hf = NS(**copy.deepcopy(ident.EXPECTED_MODEL))
    return NS(model_config=NS(hf_config=hf, dtype="bfloat16", max_model_len=context,
                             quantization=None, get_vocab_size=lambda: 128000),
              cache_config=NS(block_size=context, num_gpu_blocks_override=batch+1,
                              enable_prefix_caching=False, cache_dtype="auto"),
              scheduler_config=NS(max_num_seqs=batch, enable_chunked_prefill=False,
                                  async_scheduling=False, max_num_batched_tokens=131072),
              parallel_config=NS(tensor_parallel_size=2), lora_config=None,
              speculative_config=None, kv_transfer_config=None, load_config=None,
              observability_config=None, device_config=None)


def new(req_id="a", block=1, tokens=None):
    return NS(req_id=req_id, prompt_token_ids=[1,2] if tokens is None else tokens,
              block_ids=([block],), num_computed_tokens=0, mm_features=[],
              lora_request=None, pooling_params=None, sampling_params=NS())


def output(news=(), cached=(), finished=()):
    counts = {r.req_id: len(r.prompt_token_ids) for r in news}
    counts.update({r:1 for r in cached})
    return NS(scheduled_new_reqs=list(news), scheduled_cached_reqs=NS(
        req_ids=list(cached), resumed_req_ids=set(), new_block_ids=[None]*len(cached),
        num_computed_tokens=[2]*len(cached)), num_scheduled_tokens=counts,
        total_num_scheduled_tokens=sum(counts.values()), finished_req_ids=set(finished),
        free_encoder_mm_hashes=[], scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={}, num_common_prefix_blocks=[0], preempted_req_ids=None)


def runner_state():
    return NS(max_num_reqs=4, max_model_len=2048, max_prompt_length=2048,
              vllm_req_to_neuron_seq_id_mapping={}, free_seq_ids=set(range(4)), requests={})


def admit_fake_step(runner, out):
    serving.preflight_step(runner, out)
    for req_id in out.finished_req_ids:
        slot = runner.vllm_req_to_neuron_seq_id_mapping.pop(req_id, None)
        if slot is not None:
            runner.free_seq_ids.add(slot)
        runner.requests.pop(req_id, None)
    for req in out.scheduled_new_reqs:
        runner.vllm_req_to_neuron_seq_id_mapping[req.req_id] = runner.free_seq_ids.pop()
        runner.requests[req.req_id] = NS(**vars(req), output_token_ids=[9])


class Node:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = type(name, (Node,), {})
        setattr(self, name, value)
        return value


@contextlib.contextmanager
def external_stubs():
    """Full loader/runner/sampler modules load with no installed native import."""
    importlib.import_module("vllm_neuron.worker")
    saved = dict(sys.modules)
    parents = {}
    def module(name):
        if name in sys.modules:
            return sys.modules[name]
        result = Stub(name)
        result.__path__ = []
        sys.modules[name] = result
        if "." in name:
            parent_name, _, child = name.rpartition(".")
            parent = module(parent_name)
            parents.setdefault((parent, child), parent.__dict__.get(child))
            setattr(parent, child, result)
        return result
    try:
        targets = [ROOT/"vllm_neuron/worker"/name for name in
                   ("neuronx_distributed_model_loader.py", "neuronx_distributed_model_runner.py", "cpu_sampler.py")]
        for p in targets:
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in (
                        "torch", "vllm", "transformers", "neuronx_distributed_inference"):
                    mod = module(node.module)
                    for alias in node.names:
                        getattr(mod, alias.name)
        torch = module("torch"); nn = module("torch.nn")
        torch.nn = nn; nn.Module = Node
        torch.types = NS(Device=Node)
        for name in ("bfloat16", "float16", "float32", "float64", "int32", "int64", "long", "strided"):
            setattr(torch, name, name)
        torch.inference_mode = lambda *a, **k: lambda f:f
        torch.no_grad = lambda *a, **k: lambda f:f
        torch.Tensor = type("Tensor", (Node,), {})
        sys.modules["regex"] = re
        module("vllm_neuron.worker.utils").get_num_layers_from_hf_config = lambda hf:24
        module("vllm.distributed.kv_transfer").has_kv_transfer_group = lambda:False
        module("vllm.forward_context").set_forward_context = lambda *a, **k:contextlib.nullcontext()
        module("vllm.v1.worker.kv_connector_model_runner_mixin").KVConnectorModelRunnerMixin = type("KV", (), {})
        module("vllm.v1.worker.lora_model_runner_mixin").LoRAModelRunnerMixin = type("LoRA", (), {})
        class Sampler(Node):
            def __init__(self):
                self.topk_topp_sampler = Node(logprobs_mode="raw_logits")
        module("vllm.v1.sample.sampler").Sampler = Sampler
        yield module
    finally:
        for key in set(sys.modules) - set(saved):
            del sys.modules[key]
        sys.modules.update(saved)
        for (parent, child), previous in parents.items():
            if previous is None:
                parent.__dict__.pop(child, None)
            else:
                setattr(parent, child, previous)
        # Remove real package attributes from freshly mocked imports as well.
        worker = sys.modules.get("vllm_neuron.worker")
        if worker:
            for name in ("neuronx_distributed_model_loader", "neuronx_distributed_model_runner", "cpu_sampler"):
                worker.__dict__.pop(name, None)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, ENV, clear=True)
        self.env.start()
    def tearDown(self):
        self.env.stop()

    def test_disabled_plugin_imports_nothing_native(self):
        seen=[]; original=builtins.__import__
        def blocked(name, *a, **k):
            seen.append(name)
            if name.split(".")[0] in ("torch", "nki", "transformers", "neuronx_distributed", "vllm"):
                raise AssertionError(name)
            return original(name,*a,**k)
        with patch.dict(os.environ, {"NEURON_LFM3":"0"}), patch("builtins.__import__", blocked):
            lfm3.register()
        self.assertFalse(any("registration" in name for name in seen))

    def test_distinct_identity_and_model_view(self):
        self.assertNotEqual(ident.SCHEMA, ident.VIEW_SCHEMA)
        self.assertEqual(ident.MODEL_ID, "LiquidAI/LFM3-1.7B")
        self.assertEqual(ident.SOFTWARE["vllm"], "0.16.0+empty")
        self.assertEqual(ident.selected_variant(), "reference_sdk053_v1")
        with patch.dict(os.environ, {"NEURON_LFM3_NKI_WRITER":"1"}):
            self.assertEqual(ident.selected_variant(), "nki05_typed_decode_v1")

    def test_scope_all_six_candidates(self):
        for b in (4,8,16):
            for c in (2048,4096):
                with self.subTest(b=b,c=c), patch.dict(os.environ, {"NEURON_LFM3_CAPACITY":str(b),"NEURON_LFM3_CONTEXT":str(c)}):
                    self.assertEqual(serving.validate_serving_config(vconfig(b,c))["context"], c)
                    self.assertEqual(len(serving.state_shapes(b,c)),35)

    def test_scope_rejects_unsupported_modes(self):
        cases=[("scheduler_config","async_scheduling",True),
               ("scheduler_config","enable_chunked_prefill",True),
               ("cache_config","enable_prefix_caching",True),
               ("cache_config","block_size",16),
               ("cache_config","num_gpu_blocks_override",4),
               ("parallel_config","tensor_parallel_size",4),
               ("parallel_config","data_parallel_size",2),
               ("model_config","quantization","int8")]
        for part,key,value in cases:
            with self.subTest(part=part,key=key):
                c=vconfig();setattr(getattr(c,part),key,value)
                with self.assertRaises(ident.AdmissionError):serving.validate_serving_config(c)
        for key in ("kv_transfer_config","speculative_config","lora_config"):
            c=vconfig();setattr(c,key,NS())
            with self.assertRaises(ident.AdmissionError):serving.validate_serving_config(c)

    def test_platform_null_block_budget(self):
        c=vconfig();c.cache_config.num_gpu_blocks_override=None
        serving.validate_serving_config(c,before_platform=True)
        self.assertEqual(c.cache_config.num_gpu_blocks_override,4)
        c.cache_config.num_gpu_blocks_override+=1;c.cache_config._neuron_null_block_adjusted=True
        serving.validate_serving_config(c,before_platform=True)
        self.assertEqual(serving.proxy_memory_bytes(c),5*24*2*2048*2*128*2)

    def test_canonical_nonfinite_and_type_distinction(self):
        self.assertNotEqual(ident.canonical_configuration({"a":True}),ident.canonical_configuration({"a":1}))
        ident.canonical_configuration({"neuron_config":{"quantize_clamp_bound":float("inf")}})
        for value in (float("inf"),float("nan")):
            with self.assertRaises(ident.AdmissionError):ident.canonical_configuration({"a":value})

    def test_slot_finish_abort_same_id_and_reuse(self):
        r=runner_state();admit_fake_step(r,output([new()]))
        old=r.vllm_req_to_neuron_seq_id_mapping["a"]
        admit_fake_step(r,output())
        self.assertEqual(r.vllm_req_to_neuron_seq_id_mapping["a"],old)
        admit_fake_step(r,output(cached=["a"]))
        admit_fake_step(r,output([new(tokens=[5,6,7])],finished=["a"]))
        self.assertEqual(r.requests["a"].prompt_token_ids,[5,6,7])
        admit_fake_step(r,output(finished=["a"]))
        self.assertEqual(r.free_seq_ids,set(range(4)))
        admit_fake_step(r,output([new("b",tokens=[8])]))
        self.assertEqual(set(r.requests),{"b"})

    def test_slot_errors_leave_ledger_unchanged(self):
        for mutate in (lambda o:setattr(o.scheduled_new_reqs[0],"prompt_token_ids",[1]*2049),
                       lambda o:setattr(o.scheduled_new_reqs[0],"num_computed_tokens",1),
                       lambda o:setattr(o.scheduled_new_reqs[0],"block_ids",([0],)),
                       lambda o:setattr(o,"preempted_req_ids",{"a"}),
                       lambda o:o.scheduled_cached_reqs.resumed_req_ids.add("a")):
            r=runner_state();o=output([new()]);mutate(o);before=copy.deepcopy(vars(r))
            with self.assertRaises(ident.AdmissionError):serving.preflight_step(r,o)
            self.assertEqual(vars(r),before)

    def test_exhausted_slots_and_cached_context_limit(self):
        r=runner_state();admit_fake_step(r,output([new(str(i),i+1) for i in range(4)]))
        with self.assertRaisesRegex(ident.AdmissionError,"exhausted"):
            serving.preflight_step(r,output([new("extra")]))
        r.requests["0"].output_token_ids=[9]*2047
        with self.assertRaisesRegex(ident.AdmissionError,"context"):
            serving.preflight_step(r,output(cached=["0"]))

    def test_loaded_state_ledger_and_rejections(self):
        shapes=serving.state_shapes(4,2048)
        states=[{f"kv_mgr.past_key_values.{i}":NS(shape=s,dtype="bfloat16") for i,s in enumerate(shapes)} for _ in range(2)]
        serving.validate_loaded_states(states,4,2048)
        states[0]["kv_mgr.past_key_values.0"].shape=(4,1024,6)
        with self.assertRaises(ident.AdmissionError):serving.validate_loaded_states(states,4,2048)
        del states[0]["kv_mgr.past_key_values.0"]
        with self.assertRaises(ident.AdmissionError):serving.validate_loaded_states(states,4,2048)

    def test_runtime_capability_cleared_after_error(self):
        identity={"schema":ident.SCHEMA,"variant":ident.selected_variant(),"scope":ident.selected_scope()}
        self.assertFalse(runtime.admitted())
        with self.assertRaisesRegex(RuntimeError,"deliberate"):
            with runtime.admission(identity):
                self.assertTrue(runtime.admitted());raise RuntimeError("deliberate")
        self.assertFalse(runtime.admitted())
        with patch.dict(os.environ,{"NKI_SIMULATOR":"1"}):
            with self.assertRaises(ident.AdmissionError):
                with runtime.admission(identity):pass

    def test_typed_dispatch_retains_one_row(self):
        from vllm_neuron.models.lfm3.lfm3_nki05_typed_dispatch_v1 import make_writer
        original=Mock(return_value="original");typed=Mock(return_value="typed")
        writer=make_writer(4,original,typed)
        self.assertEqual(writer(NS(shape=(4,128)),NS(shape=(1,128)),"slots"),"original")
        typed.assert_not_called()
        self.assertEqual(writer(NS(shape=(4,128)),NS(shape=(2,128)),"slots"),"typed")

    def test_device_factory_rejects_before_native_import(self):
        from vllm_neuron.models.lfm3.slot_cache_nki05_typed_jit_v1 import make_device_kernel
        with patch("builtins.__import__", side_effect=AssertionError("must not import")):
            with self.assertRaisesRegex(RuntimeError,"admission"):make_device_kernel(lambda:False)

    def test_artifact_missing_rejects_without_compile(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"candidate"; conf=NS(to_json_string=lambda:'{"metadata":null}')
            identity={"schema":ident.SCHEMA,"variant":ident.selected_variant(),"scope":ident.selected_scope()}
            with patch.dict(os.environ,{"NEURON_COMPILED_ARTIFACTS":str(p),"BASE_COMPILE_WORK_DIR":str(p)+".work", "NEURON_COMPILE_CACHE_URL":str(p)+".cache"}), patch.object(artifacts,"build_identity",return_value=identity):
                model=Mock()
                with self.assertRaisesRegex(ident.AdmissionError,"Artifact missing"):
                    artifacts.load_or_compile(NS(),"fake-model",model,conf,{})
                model.assert_not_called();self.assertFalse(p.exists())

    def test_artifact_existing_incomplete_never_recompiled(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"candidate";p.mkdir();(p/"keep").write_text("untouched")
            identity={"schema":ident.SCHEMA,"variant":ident.selected_variant(),"scope":ident.selected_scope()}
            with patch.dict(os.environ,{"NEURON_LFM3_ALLOW_COMPILE":"1","NEURON_COMPILED_ARTIFACTS":str(p),"BASE_COMPILE_WORK_DIR":str(p)+".work","NEURON_COMPILE_CACHE_URL":str(p)+".cache"}),patch.object(artifacts,"build_identity",return_value=identity):
                model=Mock()
                with self.assertRaises(ident.AdmissionError):
                    artifacts.load_or_compile(NS(),"fake",model,NS(to_json_string=lambda:'{"metadata":null}'),{})
                model.assert_not_called();self.assertEqual((p/"keep").read_text(),"untouched")

    def test_artifact_saved_body_and_bytes_rejections(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);identity={"configuration":{"metadata":None,"x":1}}
            (p/"model.pt").write_bytes(b"fake-model")
            (p/"neuron_config.json").write_text(json.dumps({"metadata":{"lfm3_sdk053":identity},"x":1}))
            manifest={"complete":True,"identity":identity,"files":artifacts.artifact_files(p)}
            (p/ident.ARTIFACT_MANIFEST).write_text(json.dumps(manifest))
            artifacts.admit_existing(p,identity)
            with self.assertRaises(ident.AdmissionError):artifacts.admit_existing(p,{"configuration":{}})
            (p/"model.pt").write_bytes(b"changed")
            with self.assertRaises(ident.AdmissionError):artifacts.admit_existing(p,identity)
            (p/"neuron_config.json").write_text(json.dumps({"metadata":{"lfm3_sdk053":identity},"x":True}))
            with self.assertRaises(ident.AdmissionError):artifacts.admit_saved_configuration(p,identity)

    def test_artifact_fresh_success_and_failed_load_no_completion(self):
        for fail_load in (False, True):
            with self.subTest(fail_load=fail_load), tempfile.TemporaryDirectory() as td:
                p=Path(td)/"candidate";events=[]
                config=NS(metadata=None)
                config.to_json_string=lambda:json.dumps({"metadata":config.metadata,"x":1})
                identity={"schema":ident.SCHEMA,"variant":ident.selected_variant(),
                          "scope":ident.selected_scope(),"configuration":{"metadata":None,"x":1}}
                class FakeModel:
                    def __init__(self,*args):events.append("construct")
                    def compile(self,path):
                        events.append("compile")
                        (Path(path)/"model.pt").write_bytes(b"fake")
                        (Path(path)/"neuron_config.json").write_text(config.to_json_string())
                    def load(self,path):
                        events.append("load")
                        self_test.assertFalse((Path(path)/ident.ARTIFACT_MANIFEST).exists())
                        if fail_load:raise ValueError("fake load failed")
                self_test=self
                with patch.dict(os.environ,{"NEURON_LFM3_ALLOW_COMPILE":"1",
                        "NEURON_COMPILED_ARTIFACTS":str(p),"BASE_COMPILE_WORK_DIR":str(p)+".work",
                        "NEURON_COMPILE_CACHE_URL":str(p)+".cache"}), \
                        patch.object(artifacts,"build_identity",return_value=identity), \
                        patch.object(artifacts,"serial_compiler",contextlib.nullcontext):
                    if fail_load:
                        with self.assertRaisesRegex(ValueError,"fake load failed"):
                            artifacts.load_or_compile(NS(),"fake",FakeModel,config,{})
                        self.assertFalse((p/ident.ARTIFACT_MANIFEST).exists())
                        with self.assertRaises(ident.AdmissionError):
                            artifacts.load_or_compile(NS(),"fake",FakeModel,config,{})
                    else:
                        artifacts.load_or_compile(NS(),"fake",FakeModel,config,{})
                        artifacts.admit_existing(p,identity)
                    self.assertEqual(events,["construct","compile","load"])

    def test_neuron_scope_overrides_rejected(self):
        n={"tp_degree":2,"batch_size":4,"max_batch_size":4,"tkg_batch_size":4,"ctx_batch_size":1,
           "seq_len":2048,"max_length":2048,"max_context_length":2048,"padding_side":"right",
           "apply_seq_ids_mask":True,"is_continuous_batching":True,"enable_bucketing":True,
           "save_sharded_checkpoint":False,"skip_sharding":False,"skip_warmup":False,
           "logical_nc_config":1,"start_rank_id":0,"local_ranks_size":2,"ep_degree":1,
           "world_size":2,"buckets":[2048],"allow_input_truncation":False,"torch_dtype":"bfloat16"}
        ident.validate_scope(ident.EXPECTED_MODEL,n)
        for key,value in (("tkg_batch_size",8),("apply_seq_ids_mask",False),("async_mode",True),
                          ("allow_input_truncation",True),("on_device_sampling_config",{}),
                          ("max_length",4096),("buckets",[128,2048]),("cp_degree",2)):
            with self.subTest(key=key),self.assertRaises(ident.AdmissionError):
                ident.validate_scope(ident.EXPECTED_MODEL,dict(n,**{key:value}))
        model=copy.deepcopy(ident.EXPECTED_MODEL);model["conv_L_cache"]=8
        with self.assertRaises(ident.AdmissionError):ident.validate_scope(model,n)

    def test_full_runner_allocates_only_valid_new_request(self):
        with external_stubs():
            m=importlib.import_module("vllm_neuron.worker.neuronx_distributed_model_runner")
            r=m.NeuronxDistributedModelRunner(vconfig(),"cpu");r.max_prompt_length=2048
            data=m.IntermediateInputData()
            r._process_new_request_for_continuous_batching(new(),data)
            self.assertEqual(len(r.free_seq_ids),3)
            self.assertEqual(data.input_tokens,[[1,2]])
            self.assertEqual(data.input_block_ids,[r.vllm_req_to_neuron_seq_id_mapping["a"]])

    def test_work_cache_unique_and_no_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"a"
            with self.assertRaises(ident.AdmissionError):artifacts.compile_directories(p,fresh=True)
            with patch.dict(os.environ,{"BASE_COMPILE_WORK_DIR":str(p)+".work","NEURON_COMPILE_CACHE_URL":str(p)+".cache"}):
                work,cache=artifacts.compile_directories(p,fresh=True)
                work.mkdir()
                with self.assertRaises(ident.AdmissionError):artifacts.compile_directories(p,fresh=True)

    def test_loader_full_body_branches_before_generic(self):
        with external_stubs() as mod:
            loader=importlib.import_module("vllm_neuron.worker.neuronx_distributed_model_loader")
            config=NS(model_type="lfm3")
            model_cls=NS(get_neuron_config_cls=lambda:lambda **k:NS(**k))
            owner=NS(_get_compiled_model_path=Mock(side_effect=AssertionError("generic path")))
            with patch.object(artifacts,"load_or_compile",return_value=(True,"isolated",config)) as strict:
                result=loader.NeuronModelBase._load_weights_common(owner,"fake",model_cls,
                        neuron_config={},config=config)
                self.assertEqual(result[1],"isolated");strict.assert_called_once()
                owner._get_compiled_model_path.assert_not_called()
                # Even a malformed explicit config cannot route an LFM3 class
                # into generic path deletion or fallback.
                model_cls.__module__="vllm_neuron.models.lfm3.modeling"
                loader.NeuronModelBase._load_weights_common(owner,"fake",model_cls,
                        neuron_config={},config=NS(model_type="malformed"))
                self.assertEqual(strict.call_count,2)
            with patch.object(artifacts,"load_or_compile",side_effect=ValueError("do not fallback")):
                with self.assertRaisesRegex(ValueError,"do not fallback"):
                    loader.NeuronModelBase._load_weights_common(owner,"fake",model_cls,neuron_config={},config=config)

    def test_sampler_class_and_per_instance_full_constructor(self):
        with external_stubs() as mod:
            loader=importlib.import_module("vllm_neuron.worker.neuronx_distributed_model_loader")
            with patch.dict(os.environ,{"NEURON_ON_DEVICE_SAMPLING_DISABLED":"1"}):
                obj=loader.NeuronModelBase(NS(get_text_config=lambda:NS(vocab_size=128000)))
                self.assertIsInstance(obj.sampler,mod("vllm.v1.sample.sampler").Sampler)
            runner_mod=importlib.import_module("vllm_neuron.worker.neuronx_distributed_model_runner")
            sampler_mod=importlib.import_module("vllm_neuron.worker.cpu_sampler")
            r=runner_mod.NeuronxDistributedModelRunner(vconfig(),"cpu")
            self.assertIsInstance(r.cpu_sampler.topk_topp_sampler,sampler_mod.BoundedTopKTopPSampler)
            first=r.cpu_sampler.topk_topp_sampler
            sampler_mod.maybe_enable_cpu_sampler(r.cpu_sampler)
            self.assertIs(r.cpu_sampler.topk_topp_sampler,first)
            other=mod("vllm.v1.sample.sampler").Sampler()
            self.assertNotIsInstance(other.topk_topp_sampler,sampler_mod.BoundedTopKTopPSampler)

    def test_full_runner_rejects_before_mutation(self):
        with external_stubs():
            m=importlib.import_module("vllm_neuron.worker.neuronx_distributed_model_runner")
            r=m.NeuronxDistributedModelRunner(vconfig(),"cpu");r.max_prompt_length=2048
            update=Mock(side_effect=AssertionError("mutated"));r._update_states=update
            invalid=output([new(tokens=[1]*2049)])
            with self.assertRaises(ident.AdmissionError):r.execute_model(invalid)
            update.assert_not_called();self.assertEqual(r.free_seq_ids,set(range(4)))
            with self.assertRaises(ident.AdmissionError):
                r._process_new_request_for_continuous_batching(invalid.scheduled_new_reqs[0],NS())
            self.assertEqual(r.free_seq_ids,set(range(4)))

    def test_full_runner_execute_deferred_reorder_grammar_sampler_order(self):
        with external_stubs():
            m=importlib.import_module("vllm_neuron.worker.neuronx_distributed_model_runner")
            r=m.NeuronxDistributedModelRunner(vconfig(),"cpu");r.max_prompt_length=2048
            events=[]
            r._update_states=lambda o:events.append("states")
            r._prepare_model_input=lambda o:NS(request_ids=["a"])
            r.maybe_get_kv_connector_output=lambda o:contextlib.nullcontext(None)
            r.model=NS(architecture="Lfm3ForCausalLM",neuron_config=NS(on_device_sampling_config=None))
            r._execute_model_for_text=lambda *a:events.append("forward") or ["logits"]
            r._prepare_logits_for_sampling=lambda *a:events.append("reorder") or "ordered"
            r._apply_grammar_bitmask_from_output=lambda *a:events.append("grammar") or "masked"
            r.cpu_sampler=lambda *a:events.append("sample") or "sampled"
            r.input_batch.sampling_metadata=NS()
            r._generate_model_runner_output=lambda o:o
            self.assertIsNone(r.execute_model(output([new()])))
            self.assertEqual(events,["states","forward"])
            self.assertEqual(r.sample_tokens(NS(grammar_bitmask="mask")),"sampled")
            self.assertEqual(events,["states","forward","reorder","grammar","sample"])
            self.assertIsNone(r._cached_logits)
            with self.assertRaises(RuntimeError):r.sample_tokens(None)

    def test_registration_full_module_conflicts_and_writer_switch(self):
        with external_stubs() as mod:
            mapping={}; registry=NS(get_supported_archs=lambda:set(),register_model=Mock())
            mod("transformers.models.auto.configuration_auto").CONFIG_MAPPING=mapping
            mod("transformers").AutoConfig=NS(register=Mock())
            mod("vllm.model_executor.models.registry").ModelRegistry=registry
            nxdi=mod("neuronx_distributed_inference.utils.constants");nxdi.MODEL_TYPES={}
            modeling=Stub("vllm_neuron.models.lfm3.modeling")
            modeling.NeuronLfm3ForCausalLM=type("Lfm",(),{})
            modeling.install_nxdi=lambda:nxdi.MODEL_TYPES.update(lfm3={"causal-lm":modeling.NeuronLfm3ForCausalLM})
            helpers=Stub("vllm_neuron.models.lfm3.helpers");helpers.store_slots=lambda *a:None
            core=Stub("vllm_neuron.models.lfm3.text_core");core.store_slots=helpers.store_slots
            with patch.dict(sys.modules,{modeling.__name__:modeling,helpers.__name__:helpers,core.__name__:core}):
                registration=importlib.import_module("vllm_neuron.models.lfm3.registration")
                for conflict in ("transformers","vllm","nxdi","writer"):
                    with self.subTest(conflict=conflict):
                        mapping.clear();registry.get_supported_archs=lambda:set();nxdi.MODEL_TYPES={};core.store_slots=helpers.store_slots
                        if conflict=="transformers":mapping["lfm3"]=object()
                        if conflict=="vllm":registry.get_supported_archs=lambda:{"Lfm3ForCausalLM"}
                        if conflict=="nxdi":nxdi.MODEL_TYPES={"lfm3":{"causal-lm":object()}}
                        if conflict=="writer":core.store_slots=lambda *a:None
                        with self.assertRaises(ident.AdmissionError):registration.install()
                        registry.register_model.assert_not_called()
                core.store_slots=helpers.store_slots;nxdi.MODEL_TYPES={}
                registration.install();registration.install()
                registry.register_model.assert_called_once()
                self.assertIn("lfm3",nxdi.MODEL_TYPES)
                for env in ({"NEURON_LFM3_NKI_WRITER":"1"},{"NEURON_LFM3_CAPACITY":"8"},{"NEURON_LFM3_CONTEXT":"4096"}):
                    with patch.dict(os.environ,env),self.assertRaises(ident.AdmissionError):registration.install()


if __name__ == "__main__":
    unittest.main(verbosity=2)
