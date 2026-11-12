"""Check real vLLM configuration construction and unchanged tokenization."""
import json
from pathlib import Path
from qwen35_vllm_registration import install
install()
from transformers import AutoConfig, AutoTokenizer
from vllm.config import ModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.models.registry import ModelRegistry

ROOT = Path(__file__).resolve().parent
case = json.loads((ROOT/'qwen35-2b-baseline.json').read_text())
path, context = case['model_path'], case['max_model_len']
config = AutoConfig.from_pretrained(path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
references = json.loads((ROOT/'qwen35-2b-hf-reference.json').read_text())['cases']
checks = []
for row in references:
    ids = tokenizer.apply_chat_template([{'role':'user','content':row['text']}],
                                       tokenize=True,add_generation_prompt=True,enable_thinking=False)
    assert ids == row['input_ids'], row['id']
    checks.append({'fixture':row['id'],'tokens':len(ids),'exact':True})
model_config = ModelConfig(model=path, max_model_len=context, dtype='bfloat16', enforce_eager=True)
info, architecture = ModelRegistry.inspect_model_cls(config.architectures, model_config)
assert info.is_text_generation_model and not info.supports_multimodal
assert info.has_inner_state and info.is_hybrid
engine_config = EngineArgs(model=path,max_model_len=context,max_num_seqs=4,
    tensor_parallel_size=2,enable_prefix_caching=False,
    additional_config={'override_neuron_config':{'on_device_sampling_config':None,
                                                'apply_seq_ids_mask':True}}).create_engine_config()
assert not engine_config.scheduler_config.chunked_prefill_enabled
assert not engine_config.cache_config.enable_prefix_caching
assert engine_config.model_config.is_hybrid
report = {'passed':True,'architecture':architecture,'model_type':config.model_type,
          'tokenizer_checks':checks,'model_info':vars(info),'scope':'Real vLLM ModelConfig and EngineArgs.create_engine_config, registry, and six pinned reference token sequences; no model compile or qualification',
          'engine_config':{'block_size':engine_config.cache_config.block_size,
                           'max_model_len':engine_config.model_config.max_model_len,
                           'chunked_prefill':engine_config.scheduler_config.chunked_prefill_enabled,
                           'prefix_cache':engine_config.cache_config.enable_prefix_caching,
                           'is_hybrid':engine_config.model_config.is_hybrid},
          'tokenizer_warning':'Installed Transformers emits generic large-vocabulary Mistral regex warning. No tokenizer mutation applied; all six frozen original-HF token sequences are exact.'}
(ROOT/'qwen35-registration-probe.json').write_text(json.dumps(report,indent=2,default=str))
print(json.dumps(report,default=str),flush=True)
