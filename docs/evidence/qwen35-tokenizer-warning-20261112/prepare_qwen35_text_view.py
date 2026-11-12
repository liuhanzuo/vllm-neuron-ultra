"""Create a text configuration view; all official weights/tokenizer stay unchanged."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = Path('/mnt/nvme/models/Qwen3.5-2B')
dest = Path('/mnt/nvme/models/Qwen3.5-2B-neuron-text-exp20261031')
dest.mkdir(exist_ok=True)
config = json.loads((source/'config.json').read_text())['text_config']
config.update(model_type='qwen35_neuron_text', architectures=['Qwen35NeuronTextForCausalLM'],
              torch_dtype='bfloat16', pad_token_id=248044, bos_token_id=None,
              rope_theta=config['rope_parameters']['rope_theta'])
(dest/'config.json').write_text(json.dumps(config, indent=2))
files = {}
for p in source.iterdir():
    if p.name == 'config.json' or not p.is_file():
        continue
    target = dest/p.name
    if target.is_symlink():
        assert target.resolve() == p.resolve()
    elif target.exists():
        raise RuntimeError('Refusing to replace ' + str(target))
    else:
        target.symlink_to(p)
    files[p.name] = {'source':str(p), 'bytes':p.stat().st_size}
manifest = {'source_model':'Qwen/Qwen3.5-2B',
    'source_revision':'15852e8c16360a2fea060d615a32b45270f8a8fc',
    'source':str(source), 'view':str(dest), 'files':files,
    'source_config_sha256':hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),
    'view_config_sha256':hashlib.sha256((dest/'config.json').read_bytes()).hexdigest(),
    'scope':'Original unchanged tensors and tokenizer via symlinks. Only text config/architecture identifier changed; adapter explicitly filters visual weights. Vision is unsupported; no substitution with plain Qwen3.'}
(ROOT/'qwen35-text-view-manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest))
