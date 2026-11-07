"""Generate per-channel INT8 checkpoint using BF16 loading, without SDK edits."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from neuronx_distributed_inference.utils.constants import MODEL_TYPES
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-path',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    model_type=AutoConfig.from_pretrained(args.model_path).model_type
    parent=MODEL_TYPES[model_type]['causal-lm']
    class Bf16Loader(parent):
        @staticmethod
        def load_hf_model(model_path,**kwargs):
            return AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,**kwargs)
    out=Path(args.output)
    if (out/'model.safetensors').is_file():
        print('Checkpoint already exists; no overwrite',flush=True)
        return
    out.mkdir(parents=True,exist_ok=True)
    nc=Bf16Loader.get_neuron_config_cls()(tp_degree=2,batch_size=4,seq_len=2048,torch_dtype=torch.bfloat16,
            quantized=True,quantization_type='per_channel_symmetric',quantization_dtype='int8',
            quantized_checkpoints_path=str(out),modules_to_not_convert=['lm_head'])
    cfg=Bf16Loader.get_config_cls()(nc,load_config=load_pretrained_config(args.model_path))
    start=time.perf_counter()
    torch.set_num_threads(2)
    Bf16Loader.save_quantized_state_dict(args.model_path,cfg)
    report={'model_path':args.model_path,'output':str(out),'seconds':time.perf_counter()-start,
            'bytes':sum(p.stat().st_size for p in out.glob('*.safetensors')),'load_dtype':'bfloat16','quantization':'per_channel_symmetric_int8','excluded_modules':['lm_head']}
    (out/'quantization-provenance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
