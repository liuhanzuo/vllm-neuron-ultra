"""Original MiniCPM5 BF16 input/reference work; runtime entries require root guard.

No CLI, queue, native import, network access or payload read occurs at import.
"""
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import tempfile
from minicpm5_loading_info_v4 import normalize_loading_info

ROOT = Path(__file__).resolve().parent
READINESS = ROOT / 'research/minicpm5-2b-readiness-20261104-v1'
MODEL_ID = 'openbmb/MiniCPM5-2B'
REVISION = 'f97400052a43d642bbc6e9975e2397e3ae6a6b52'
MODEL_PATH = Path('/mnt/nvme/models/MiniCPM5-2B')
VOCAB = 130560
PARAMETERS = 2516756480
WEIGHT = 'model-00000-of-00001.safetensors'
WEIGHT_BYTES = 5033557096
WEIGHT_SHA256 = '14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c'
TOKENIZER_BLOB = 'ec0bf9a34dc884738494ed1f74d0c5327b06d690'
FILES = ('config.json', 'generation_config.json', 'model.safetensors.index.json', 'special_tokens_map.json',
         'tokenizer_config.json', 'chat_template.jinja', 'tokenizer.json', WEIGHT)
PREFIXES = (32, 128, 256, 512, 1024, 2048, 2052)
BUCKETS = (128, 256, 512, 1024, 2048, 4096)
LENGTH = 3072
STEPS = 2041
MARKS = (0,22,30,31,32,33,63,64,65,126,127,128,129,255,256,257,
         510,511,512,513,1022,1023,1024,1025,2042,2043,2047,2048,2049,
         2168,2169,3063,3064,3065)
CRITERIA = {'cosine_min': .995, 'kl_max_nats': .02, 'max_abs_error_max': 2.,
            'argmax_match_fraction_min': .9, 'all_reference_margins_gt_0.5_must_match': True}
NATIVE_ROOT = ROOT / 'research/transformers-reference/src/transformers'
NATIVE_LLAMA = NATIVE_ROOT / 'models/llama/modeling_llama.py'
RUNTIME_ROOT = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/transformers')
PHASES = ('download', 'tokenizer-native', 'tokenizer-runtime', 'reference32', 'reference-long', 'reference-cold')


def read(path):
    path = Path(path)
    assert path.suffix == '.json' and path.stat().st_size < 4 * 1024**2
    return json.loads(path.read_text())


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def text_sha(path):
    path = Path(path)
    assert path.suffix in ('.py', '.json', '.jinja', '.md', '.pth') and path.stat().st_size < 4 * 1024**2
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    path = Path(path); stat = path.stat()
    return dict(realpath=str(path.resolve()), device=stat.st_dev, inode=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def once(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='.' + path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication; existing target is never replaced.
    finally:
        if temporary is not None: temporary.unlink()


def progress(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_paths():
    paths = [Path(__file__), READINESS / 'official-metadata.json', READINESS / 'inventory.json',
        READINESS / 'fetch-provenance.json', READINESS / 'safetensors-header.json']
    paths += [READINESS / name for name in FILES if name not in ('tokenizer.json', WEIGHT)]
    paths += [ROOT / name for name in ('validate_real_logits.py', 'validate_qwen3_4b_full_cold_v3.py',
        'validate_qwen3_4b_long_decode.py', 'qwen3_4b_cached_json_v2_protocol.py', 'quality-suite.json',
        'evaluate_quality.py', 'benchmark_workloads.py', 'bench_http.py', 'orchestrate.py')]
    common = ('__init__.py', 'models/llama/modeling_llama.py', 'models/llama/configuration_llama.py', 'modeling_utils.py',
        'configuration_utils.py', 'modeling_rope_utils.py', 'cache_utils.py', 'tokenization_utils_base.py',
        'models/auto/tokenization_auto.py')
    paths += [base / name for base in (NATIVE_ROOT, RUNTIME_ROOT) for name in common]
    paths += [ROOT / 'minicpm5_loading_info_v4.py', ROOT / 'minicpm5-baseline-full-v3/download.json',
              NATIVE_ROOT / 'utils/loading_report.py']
    paths += [NATIVE_ROOT / 'tokenization_utils_tokenizers.py', NATIVE_ROOT / 'masking_utils.py',
              RUNTIME_ROOT / 'tokenization_utils_fast.py']
    assert all(path.is_file() for path in paths)
    return sorted(set(paths))


def normalize_ids(value):
    if hasattr(value, 'keys'):
        assert 'input_ids' in value; value = value['input_ids']
    if isinstance(value, tuple): value = list(value)
    if value and isinstance(value[0], (list, tuple)):
        assert len(value) == 1; value = list(value[0])
    assert isinstance(value, list) and value and all(type(x) is int and 0 <= x < VOCAB for x in value)
    return value


class TokenIds:
    """Keep the real tokenizer; normalize only its documented token-ID container."""
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def __getattr__(self, name): return getattr(self.tokenizer, name)
    def apply_chat_template(self, *args, **kwargs):
        value = self.tokenizer.apply_chat_template(*args, **kwargs)
        return value if isinstance(value, str) else normalize_ids(value)


def _function(path, name, globals_=None):
    tree = ast.parse(Path(path).read_text())
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    namespace = {} if globals_ is None else dict(globals_)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def canonical_fixtures(tokenizer):
    rows = _function(ROOT / 'validate_real_logits.py', 'fixtures')(TokenIds(tokenizer))
    assert len(rows) == 8 and [x['id'] for x in rows] == list(range(8))
    return rows


def long_fixtures(tokenizer):
    path = ROOT / 'validate_qwen3_4b_long_decode.py'; tree = ast.parse(path.read_text())
    reference, = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'reference']
    values = {}
    for node in reference.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('prefixes', 'paragraphs'): values[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(values) == {'prefixes', 'paragraphs'} and len(values['prefixes']) == len(values['paragraphs']) == 2
    result = []
    for i, (prompt, paragraph) in enumerate(zip(values['prefixes'], values['paragraphs'])):
        prefix = normalize_ids(tokenizer.apply_chat_template([dict(role='user', content=prompt)],
            tokenize=True, add_generation_prompt=True, enable_thinking=False))
        body = normalize_ids(tokenizer.encode(paragraph, add_special_tokens=False))
        ids = (prefix + body * (LENGTH // len(body) + 1))[:LENGTH]
        assert len(ids) == LENGTH
        result.append(dict(id=i, text=prompt, paragraph=paragraph, input_ids=ids))
    return result


def build_protocol(fixtures):
    """Same full Qwen call schedule, with MiniCPM-generated tokens only."""
    assert [x['id'] for x in fixtures] == [0, 1] and all(len(x['input_ids']) == LENGTH for x in fixtures)
    assert all(type(token) is int and 0 <= token < VOCAB for x in fixtures for token in x['input_ids'])
    trajectories = [dict(id=i, fixture=f, prefix=n, slot=s, last_position=n+STEPS-1+(2 if s != 3 else 0))
        for i, (f, n, s) in enumerate(((0,23,3), (1,127,1), (0,1023,0), (1,1,2)))]
    trajectories.append(dict(id=4, fixture=1, prefix=31, slot=3, last_position=32))
    def request(t, position, record=True):
        return dict(trajectory=t['id'], fixture=t['fixture'], slot=t['slot'], position=position, record=record)
    operations = [dict(phase='prefill', requests=[request(t, t['prefix']-1)]) for t in trajectories[:4]]
    operations += [dict(phase='long_decode', requests=[request(t, t['prefix']+step, t['prefix']+step in MARKS)
        for t in trajectories[:4]]) for step in range(STEPS)]
    operations += [dict(phase='slot_reuse_prefill', requests=[request(trajectories[4],30)]),
        dict(phase='decode_after_slot_reuse', requests=[request(trajectories[4],31)] + [request(t,t['prefix']+STEPS) for t in trajectories[1:4]]),
        dict(phase='partial_batch', requests=[request(t,t['prefix']+STEPS+1) for t in trajectories[1:4]]),
        dict(phase='resume_inactive', requests=[request(trajectories[4],32)])]
    checks = [dict(**{k: r[k] for k in ('trajectory','fixture','position','slot')}, phase=op['phase'])
        for op in operations for r in op['requests'] if r['record']]
    protocol = dict(fixtures=copy.deepcopy(fixtures), trajectories=trajectories, operations=operations, checks=checks,
        criteria=CRITERIA, revision=REVISION, native_source_sha256=text_sha(NATIVE_LLAMA),
        enable_thinking=False, sequence_length=LENGTH, teacher_forced_decode_steps=STEPS)
    from qwen3_4b_cached_json_v2_protocol import validate
    validate(protocol)  # Executes the original Qwen loop AST with inert arrays; never imports Torch.
    assert len(operations) == 2049 and len(checks) == 93
    assert sum(t['last_position']-t['prefix']+1 for t in trajectories) == 8172
    return protocol


def calls_for(protocol):
    from qwen3_4b_cached_json_v2_protocol import calls_for as original_calls
    return original_calls(protocol)


def download_manifest():
    metadata = read(READINESS / 'official-metadata.json')
    assert metadata['id'] == MODEL_ID and metadata['sha'] == REVISION and not metadata['gated']
    siblings = {row['rfilename']: row for row in metadata['siblings']}
    result = []
    for name in FILES:
        row = siblings[name]; item = dict(name=name, size=row['size'], git_blob=row['blobId'],
            url=f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}')
        if name == WEIGHT:
            assert row['size'] == WEIGHT_BYTES and row['lfs']['sha256'] == WEIGHT_SHA256
            item['sha256'] = WEIGHT_SHA256; item['git_blob'] = None
        elif name == 'tokenizer.json':
            assert row['size'] == 9894271 and row['blobId'] == TOKENIZER_BLOB
        else:
            path = READINESS / name
            assert path.stat().st_size == row['size']
            blob = hashlib.sha1(b'blob ' + str(row['size']).encode() + b'\0' + path.read_bytes()).hexdigest()
            assert blob == row['blobId'], 'Pinned small file no longer matches official Git blob: ' + name
            item['sha256'] = text_sha(path)
        result.append(item)
    return result


def _enter(context, phase):
    context['guard'](phase)  # Must be the first action before any native/network/payload work.
    assert Path(context['ROOT']).resolve() == ROOT and context['revision'] == REVISION
    assert Path(context['model_path']) == MODEL_PATH and context['case']['model_path'] == str(MODEL_PATH)
    assert Path(context['DEST']).resolve().parent == ROOT and context['case']['max_model_len'] == 4096
    assert context['case']['batch'] == 4 and context['case']['tp'] == 2 and context['case']['cpu'] is True
    assert not context['case'].get('int8') and not context['case'].get('cpu_fast_sampler')
    pins = context['plan']['source_sha256']
    for path in source_paths(): assert pins[str(path)] == text_sha(path), str(path)
    return dict(context, ROOT=ROOT, DEST=Path(context['DEST']), model_path=MODEL_PATH, phase=phase)


def _file_hashes(path, expected):
    sha256 = hashlib.sha256(); blob = hashlib.sha1(b'blob ' + str(expected['size']).encode() + b'\0')
    size = 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            size += len(chunk); assert size <= expected['size']; sha256.update(chunk); blob.update(chunk)
    assert size == expected['size']
    if expected.get('sha256'): assert sha256.hexdigest() == expected['sha256']
    if expected.get('git_blob'): assert blob.hexdigest() == expected['git_blob']
    return dict(bytes=size, sha256=sha256.hexdigest(), git_blob=blob.hexdigest() if expected.get('git_blob') else None)


def _download(context):
    # The successful v3 official files are reused only after the exclusive v4
    # worker rehashes every byte. No network fallback and no second download.
    destination = context['DEST']; model = context['model_path']; manifest = download_manifest()
    previous_path = ROOT / 'minicpm5-baseline-full-v3/download.json'
    previous = read(previous_path)
    assert previous['complete'] and previous['passed'] and previous['model'] == MODEL_ID and previous['revision'] == REVISION
    assert previous['weight_sha256_verified'] and previous['tokenizer_git_blob_verified']
    previous_files = {row['name']: row for row in previous['files']}
    assert len(previous['files']) == len(previous_files) == len(FILES) and set(previous_files) == set(FILES)
    assert not (destination / 'download.json').exists()
    assert model.is_dir() and not model.is_symlink()
    assert shutil.disk_usage(model).free >= 5 * 1024**3
    records = []
    for item in manifest:
        path = model / item['name']; old = previous_files[item['name']]
        assert path.is_file() and not path.is_symlink(), 'Successful v3 file missing; no automatic redownload'
        assert old['path'] == str(path) and old['identity'] == identity(path)
        assert all(old.get(key) == value for key, value in item.items())
        checked = _file_hashes(path, item)
        assert all(old.get(key) == value for key, value in checked.items())
        records.append({**item, **checked, 'path': str(path), 'identity': identity(path),
                        'downloaded': False, 'final_host': None})
    with (model / WEIGHT).open('rb') as stream:
        length = int.from_bytes(stream.read(8), 'little'); assert length == 44128
        header = stream.read(length)
    assert header == (READINESS / 'safetensors-header.json').read_bytes()
    _check_config(read(model / 'config.json'))
    assert read(model / 'generation_config.json')['eos_token_id'] == [1, 130073]
    report = dict(complete=True, passed=True, model=MODEL_ID, revision=REVISION, files=records,
        tensor_count=381, parameters=PARAMETERS, original_file_bytes_preserved=True,
        weight_sha256_verified=True, tokenizer_git_blob_verified=True, license='apache-2.0',
        reused_successful_download=str(previous_path), reused_manifest_sha256=text_sha(previous_path),
        worker_rehashed_all_files=True, network_access=False)
    once(destination / 'download.json', report)
    return dict(outputs={str(destination / 'download.json'):text_sha(destination / 'download.json')})


def _verify_download(context):
    report = read(context['DEST'] / 'download.json')
    assert report['complete'] and report['passed'] and report['revision'] == REVISION and report['model'] == MODEL_ID
    assert len(report['files']) == len(FILES) and {x['name'] for x in report['files']} == set(FILES)
    assert not context['model_path'].is_symlink()
    expected = {x['name']:x for x in download_manifest()}
    for row in report['files']:
        assert row['identity'] == identity(row['path']) and Path(row['path']) == context['model_path'] / row['name']
        spec = expected[row['name']]; assert row['bytes'] == spec['size']
        if spec.get('sha256'): assert row['sha256'] == spec['sha256']
        if spec.get('git_blob'): assert row['git_blob'] == spec['git_blob']
    assert report['weight_sha256_verified'] and report['tokenizer_git_blob_verified']
    return report


def _check_config(config):
    expected = dict(architectures=['LlamaForCausalLM'], model_type='llama', bos_token_id=0,
        eos_token_id=[1,130073], pad_token_id=1, hidden_size=2048, intermediate_size=6144,
        num_attention_heads=16, num_key_value_heads=2, num_hidden_layers=42, head_dim=128,
        vocab_size=VOCAB, rms_norm_eps=1e-6, max_position_embeddings=131072,
        tie_word_embeddings=False, hidden_act='silu', rope_theta=5000000, rope_scaling=None,
        torch_dtype='bfloat16', use_cache=True)
    assert all(config[key] == value for key,value in expected.items())


def _tokenizer(context):
    _verify_download(context)
    import transformers
    from transformers import AutoTokenizer
    native = context['phase'] == 'tokenizer-native'
    expected_root = NATIVE_ROOT if native else RUNTIME_ROOT
    assert Path(transformers.__file__).resolve() == (expected_root / '__init__.py').resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(context['model_path']), local_files_only=True, trust_remote_code=False)
    tokenizer_source = expected_root / ('tokenization_utils_tokenizers.py' if native else 'tokenization_utils_fast.py')
    assert Path(sys.modules[type(tokenizer).__module__].__file__).resolve() == tokenizer_source.resolve()
    assert tokenizer.is_fast and len(tokenizer) == VOCAB
    assert tokenizer.bos_token_id == 0 and tokenizer.pad_token_id == 1 and tokenizer.eos_token_id == 1
    assert tokenizer.convert_tokens_to_ids('<|im_end|>') == 130073
    assert tokenizer.chat_template == (context['model_path'] / 'chat_template.jinja').read_text()
    return TokenIds(tokenizer)


def runtime_tokenizer(context):
    """Shared by serving: guarded original tokenizer, normalized ID container only."""
    current = _enter(context, context.get('phase', 'tokenizer-runtime'))
    return _tokenizer(current)


def _workload_text():
    path = ROOT / 'benchmark_workloads.py'; tree = ast.parse(path.read_text())
    function, = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'inputs']
    statement, = [x for x in function.body if isinstance(x, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'text' for t in x.targets)]
    namespace = {'__builtins__': {'range':range}}
    exec(compile(ast.Module(body=[statement], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['text']


def _tokenizer_phase(context):
    mode = 'native' if context['phase'] == 'tokenizer-native' else 'runtime'
    tokenizer = _tokenizer(context)
    directory = context['DEST'] / 'tokenizers'; directory.mkdir(exist_ok=True)
    output = directory / (mode + '.json'); assert not output.exists()
    canonical = canonical_fixtures(tokenizer)
    suite = read(ROOT / 'quality-suite.json'); assert len(suite['cases']) == 96
    quality = []
    for item in suite['cases']:
        ids = normalize_ids(tokenizer.apply_chat_template([dict(role='user',content=item['prompt'])],
            tokenize=True,add_generation_prompt=True,enable_thinking=False))
        assert len(ids)+item['max_tokens'] <= 4096, 'Never truncate quality prompts'
        quality.append(dict(id=item['id'], input_ids=ids, max_tokens=item['max_tokens']))
    full = normalize_ids(tokenizer.apply_chat_template([dict(role='user',content=_workload_text())],
        tokenize=True,add_generation_prompt=True,enable_thinking=False))
    initial_http_text = '请详细介绍一下法国的首都。'
    initial_http_ids = normalize_ids(tokenizer.apply_chat_template([dict(role='user',content=initial_http_text)],
        tokenize=True,add_generation_prompt=True,enable_thinking=False))
    workloads = {str(length):full[:length-16]+full[-16:] for length in (32,128,512,1536)}
    assert all(len(ids)==int(length) for length,ids in workloads.items())
    fixtures = long_fixtures(tokenizer); protocol = build_protocol(fixtures)
    if mode == 'runtime':
        import orchestrate
        from evaluate_quality import prepare_inputs
        from benchmark_workloads import inputs
        from unittest.mock import patch
        # Only prompt container adaptation; neither the harness nor math is changed.
        with patch.object(orchestrate, 'prompt_tokenizer', return_value=tokenizer):
            actual_quality = prepare_inputs(context['case'], suite)
            assert [dict(id=item['id'],input_ids=ids,max_tokens=item['max_tokens']) for item,ids in actual_quality] == quality
            assert {str(k):v for k,v in inputs(context['case']).items()} == workloads
            assert orchestrate.prompt_ids(context['case'], initial_http_text) == initial_http_ids
            assert all(orchestrate.prompt_ids(context['case'], row['text']) == row['input_ids'] for row in canonical)
    vocabulary = tokenizer.get_vocab()
    assert len(vocabulary) == VOCAB and set(vocabulary.values()) == set(range(VOCAB))
    by_id = {value:key for key,value in vocabulary.items()}; parts = []; output_files = []
    for start in range(0, VOCAB, 2048):
        ids = list(range(start, min(start+2048,VOCAB))); rows = []
        for token_id in ids:
            token = by_id[token_id]
            assert tokenizer.convert_tokens_to_ids(token) == token_id
            rows.append(dict(id=token_id, token=token,
                decoded=tokenizer.decode([token_id], skip_special_tokens=False),
                decoded_skip_special=tokenizer.decode([token_id], skip_special_tokens=True),
                encoded_surface_ids=tokenizer.encode(token, add_special_tokens=False)))
        path = directory / f'{mode}-vocab-{start//2048:03d}.json'
        once(path, dict(start=start, count=len(rows), rows=rows))
        parts.append(dict(start=start, count=len(rows), semantic_sha256=json_digest(rows)))
        output_files.append(path)
    special = dict(bos=tokenizer.bos_token_id, pad=tokenizer.pad_token_id, eos=tokenizer.eos_token_id,
        generation_eos=[1,130073], think=tokenizer.convert_tokens_to_ids('<think>'),
        end_think=tokenizer.convert_tokens_to_ids('</think>'), im_start=tokenizer.convert_tokens_to_ids('<|im_start|>'),
        im_end=tokenizer.convert_tokens_to_ids('<|im_end|>'), all_special_ids=tokenizer.all_special_ids,
        clean_up_tokenization_spaces=tokenizer.clean_up_tokenization_spaces)
    assert special['think'] == 8 and special['end_think'] == 9 and special['im_start'] == 130072
    value = dict(mode=mode, complete=True, passed=True, revision=REVISION, enable_thinking=False,
        canonical=canonical, quality=quality, workloads=workloads, initial_http_input_ids=initial_http_ids,
        long_fixtures=fixtures,
        special_tokens=special, vocab_parts=parts, vocabulary_entries=VOCAB, full_vocabulary_serialized=True,
        full_vocabulary_compared=(mode == 'runtime'),
        suite_sha256=text_sha(ROOT / 'quality-suite.json'), protocol_sha256=json_digest(protocol),
        template_sha256=text_sha(context['model_path'] / 'chat_template.jinja'),
        tokenizer_json_sha256=next(row['sha256'] for row in read(context['DEST']/'download.json')['files'] if row['name']=='tokenizer.json'))
    if mode == 'native':
        once(context['DEST'] / 'input-protocol.json', protocol); output_files.append(context['DEST'] / 'input-protocol.json')
    else:
        native = read(directory / 'native.json')
        assert native['full_vocabulary_compared'] is False
        assert all(value[key] == native[key] for key in value if key not in ('mode','full_vocabulary_compared')), 'Runtime/native tokenizer divergence'
        assert protocol == read(context['DEST'] / 'input-protocol.json')
        for index in range(len(parts)):
            assert read(directory / f'native-vocab-{index:03d}.json') == read(directory / f'runtime-vocab-{index:03d}.json')
    once(output, value); output_files.append(output)
    return dict(outputs={str(path):text_sha(path) for path in output_files}, vocabulary_entries=VOCAB,
        canonical_prompts=8, quality_prompts=96, workload_input_lengths=[32,128,512,1536], long_fixtures=2,
        native_runtime_parity=(mode == 'runtime'))


def _load_native(context):
    _verify_download(context)
    import torch
    import transformers.models.llama.modeling_llama as native
    actual = Path(native.__file__).resolve()
    assert actual == NATIVE_LLAMA.resolve() and text_sha(actual) == context['plan']['source_sha256'][str(NATIVE_LLAMA)]
    torch.set_num_threads(2)
    model, loading = native.LlamaForCausalLM.from_pretrained(str(context['model_path']),
        dtype=torch.bfloat16, attn_implementation='eager', local_files_only=True,
        trust_remote_code=False, output_loading_info=True)
    loading = normalize_loading_info(loading)
    assert not any(loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')), loading
    model.eval()
    header = read(READINESS / 'safetensors-header.json'); header.pop('__metadata__',None)
    state = model.state_dict()
    assert set(state) == set(header) and len(state) == 381
    for name, value in state.items():
        assert list(value.shape) == header[name]['shape'] and value.dtype == torch.bfloat16 and value.device.type == 'cpu'
    assert sum(value.numel() for value in model.parameters()) == PARAMETERS
    assert not model.config.tie_word_embeddings and model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    assert model.config.rope_parameters == dict(rope_type='default', rope_theta=5000000)
    assert model.config.eos_token_id == [1,130073] and model.config.pad_token_id == 1
    assert model.config._attn_implementation == 'eager'
    _check_config(read(context['model_path'] / 'config.json'))
    provenance = dict(model_class='transformers.models.llama.modeling_llama.LlamaForCausalLM',
        source=str(actual), source_sha256=text_sha(actual), revision=REVISION,
        dtype='torch.bfloat16', attention='eager', parameters=PARAMETERS, tensors=381,
        independent_embedding_and_lm_head=True, loading_info=loading, native_threads=2,
        device='cpu', math_implementation_modified=False)
    return torch, model, provenance


def _cache(cache, expected):
    assert int(cache.get_seq_length()) == expected and len(cache.layers) == 42
    for layer in cache.layers:
        assert list(layer.keys.shape) == list(layer.values.shape) == [1,2,expected,128]
        assert str(layer.keys.dtype) == str(layer.values.dtype) == 'torch.bfloat16'


def _vector(output, context, label):
    import torch
    value = output.logits[0,-1].detach().float().cpu().clone()
    assert value.shape == (VOCAB,)
    if not bool(torch.isfinite(value).all()):
        path = context['DEST'] / (context['phase']+'-nonfinite.pt')
        with path.open('xb') as stream: torch.save(dict(label=label,logits=value), stream)
        once(path.with_suffix('.json'), dict(passed=False,label=label,nan_elements=int(torch.isnan(value).sum()),
            inf_elements=int(torch.isinf(value).sum()), tensor=str(path)))
        raise AssertionError('Original native reference returned nonfinite logits; evidence preserved')
    return value


def _payload(context, stem, value, manifest):
    import torch
    path = context['DEST'] / (stem+'.pt'); metadata = path.with_suffix('.json')
    assert not path.exists() and not metadata.exists()
    with path.open('xb') as stream: torch.save(value, stream)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''): digest.update(block)
    document = dict(complete=True,passed=True,revision=REVISION,criteria=CRITERIA,**manifest,
        tensor=str(path),tensor_sha256=digest.hexdigest(),tensor_identity=identity(path),
        full_model_qualified=False,new_speedup_claim=False)
    once(metadata,document)
    return metadata


def _reference32(context):
    tokenizers = read(context['DEST'] / 'tokenizers/native.json')
    rows = []; torch, model, provenance = _load_native(context)
    with torch.inference_mode():
        for fixture in tokenizers['canonical']:
            started=time.monotonic(); ids=fixture['input_ids']; length=len(ids)
            output=model(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,length),dtype=torch.long),
                position_ids=torch.arange(length)[None],use_cache=True,logits_to_keep=1)
            _cache(output.past_key_values,length)
            frames=[_vector(output,context,dict(fixture=fixture['id'],step=0))]; continuation=[]
            for step in range(3):
                token=int(frames[-1].argmax());continuation.append(token)
                output=model(input_ids=torch.tensor([[token]]),attention_mask=torch.ones((1,length+step+1),dtype=torch.long),
                    position_ids=torch.tensor([[length+step]]),past_key_values=output.past_key_values,use_cache=True,logits_to_keep=1)
                _cache(output.past_key_values,length+step+1)
                frames.append(_vector(output,context,dict(fixture=fixture['id'],step=step+1)))
            rows.append(dict(**fixture,continuation_ids=continuation,logits=torch.stack(frames),seconds=time.monotonic()-started))
            del output
            progress(context['DEST']/'reference32-progress.json',dict(completed_fixtures=len(rows),frames=4*len(rows)))
    assert len(rows)==8 and all(row['logits'].shape==(4,VOCAB) for row in rows)
    metadata=_payload(context,'hf-reference',rows,dict(frames=32,all_finite=True,provenance=provenance,
        native_source_sha256=provenance['source_sha256'],fixtures=[{k:v for k,v in row.items() if k!='logits'} for row in rows],
        enable_thinking=False,native_prefills=8,native_cached_steps=24,native_forward_calls=32,
        full_vocabulary_size=VOCAB,scope='Original native Llama BF16/eager eight fixtures, prefill and three cached greedy teacher-forced steps.'))
    return dict(outputs={str(metadata):text_sha(metadata)},frames=32)


def _reference_long(context):
    protocol=read(context['DEST']/'input-protocol.json')
    assert protocol==build_protocol(protocol['fixtures'])
    torch,model,provenance=_load_native(context)
    captured={}; updates=0; forwards=0; metadata_files=[]
    with torch.inference_mode():
        for trajectory in protocol['trajectories']:
            fixture=protocol['fixtures'][trajectory['fixture']]; ids=fixture['input_ids']; prefix=trajectory['prefix']
            checks={r['position']:r for r in protocol['checks'] if r['trajectory']==trajectory['id']}
            rows=[]
            output=model(input_ids=torch.tensor([ids[:prefix]]),attention_mask=torch.ones((1,prefix),dtype=torch.long),
                position_ids=torch.arange(prefix)[None],use_cache=True,logits_to_keep=1)
            _cache(output.past_key_values,prefix); forwards+=1
            value=_vector(output,context,dict(trajectory=trajectory['id'],position=prefix-1))
            if prefix-1 in checks:
                row=dict(**checks[prefix-1],logits=value);rows.append(row);captured[(trajectory['id'],prefix-1)]=row
            for position in range(prefix,trajectory['last_position']+1):
                output=model(input_ids=torch.tensor([[ids[position]]]),attention_mask=torch.ones((1,position+1),dtype=torch.long),
                    position_ids=torch.tensor([[position]]),past_key_values=output.past_key_values,use_cache=True,logits_to_keep=1)
                _cache(output.past_key_values,position+1);updates+=1;forwards+=1
                value=_vector(output,context,dict(trajectory=trajectory['id'],position=position))
                if position in checks:
                    row=dict(**checks[position],logits=value);rows.append(row);captured[(trajectory['id'],position)]=row
                if position%128==0:
                    progress(context['DEST']/'reference-long-progress.json',dict(trajectory=trajectory['id'],position=position,
                        native_cached_steps=updates,native_forward_calls=forwards,retained_frames=len(captured)))
            assert len(rows)==len(checks);del output
            metadata_files.append(_payload(context,'cached-reference-trajectory-'+str(trajectory['id']),rows,
                dict(trajectory=trajectory,frames=len(rows),all_finite=True,provenance=provenance)))
    rows=[captured[(r['trajectory'],r['position'])] for r in protocol['checks']]
    assert len(rows)==93 and updates==8172 and forwards==8177
    metadata_files.append(_payload(context,'cached-reference',rows,dict(frames=93,all_finite=True,
        all_forward_calls_finite=True,native_cached_steps=updates,native_forward_calls=forwards,
        native_prefills=5,device_operations=2049,device_decode_calls=2044,teacher_forced_decode_steps=2041,
        full_vocabulary_size=VOCAB,slot_reuse_reference_trajectory=4,
        lifecycle_phases=['slot_reuse_prefill','decode_after_slot_reuse','partial_batch','resume_inactive'],
        protocol_sha256=json_digest(protocol),operations_sha256=json_digest(protocol['operations']),
        fixture_ids_sha256=json_digest([x['input_ids'] for x in protocol['fixtures']]),
        trajectories=protocol['trajectories'],provenance=provenance,native_source_sha256=provenance['source_sha256'],
        scope='Five complete original native Llama BF16 cached trajectories, all 8172 single-token updates. All 93 requested marks retained; no full-causal oracle substitution.')))
    return dict(outputs={str(path):text_sha(path) for path in metadata_files},frames=93,native_cached_steps=updates)


def _reference_cold(context):
    protocol=read(context['DEST']/'input-protocol.json');assert protocol==build_protocol(protocol['fixtures'])
    torch,model,provenance=_load_native(context);rows=[]
    with torch.inference_mode():
        for prefix in PREFIXES:
            for fixture in protocol['fixtures']:
                ids=fixture['input_ids'];positions=[prefix-1,prefix,prefix+1]
                output=model(input_ids=torch.tensor([ids[:prefix]]),attention_mask=torch.ones((1,prefix),dtype=torch.long),
                    position_ids=torch.arange(prefix)[None],use_cache=True,logits_to_keep=1)
                _cache(output.past_key_values,prefix)
                values=[_vector(output,context,dict(prefix=prefix,fixture=fixture['id'],position=prefix-1))]
                for position in (prefix,prefix+1):
                    output=model(input_ids=torch.tensor([[ids[position]]]),attention_mask=torch.ones((1,position+1),dtype=torch.long),
                        position_ids=torch.tensor([[position]]),past_key_values=output.past_key_values,use_cache=True,logits_to_keep=1)
                    _cache(output.past_key_values,position+1)
                    values.append(_vector(output,context,dict(prefix=prefix,fixture=fixture['id'],position=position)))
                rows.append(dict(prefix=prefix,fixture=fixture['id'],input_ids=ids,positions=positions,logits=torch.stack(values)))
                del output
                progress(context['DEST']/'reference-cold-progress.json',dict(records=len(rows),frames=3*len(rows)))
    assert len(rows)==14 and all(row['logits'].shape==(3,VOCAB) for row in rows)
    metadata=_payload(context,'cold-reference',rows,dict(frames=42,all_finite=True,prefixes=list(PREFIXES),
        native_prefills=14,native_cached_steps=28,native_forward_calls=42,full_vocabulary_size=VOCAB,
        independent_fresh_cache_per_prefix_and_fixture=True,
        protocol_sha256=json_digest(protocol),provenance=provenance,native_source_sha256=provenance['source_sha256'],
        scope='Seven exact prefill prefixes x two MiniCPM fixtures x prefill plus two cached updates, 42 full-vocabulary BF16 native frames.'))
    return dict(outputs={str(metadata):text_sha(metadata)},frames=42)


def run(phase, context):
    current=_enter(context,phase)
    assert phase in PHASES
    function={'download':_download,'tokenizer-native':_tokenizer_phase,'tokenizer-runtime':_tokenizer_phase,
        'reference32':_reference32,'reference-long':_reference_long,'reference-cold':_reference_cold}[phase]
    return function(current)


def static_preflight():
    """Read bounded metadata/source and replay inert controls, with no runtime work.

    Synthetic IDs exercise only the preserved request schedule. They are never
    persisted as model input or used to claim tokenizer/numerical correctness.
    """
    imported_before = set(sys.modules)
    manifest = download_manifest()
    assert len(manifest) == 8 and sum(row['size'] for row in manifest) == 5043584039
    _check_config(read(READINESS / 'config.json'))
    inventory = read(READINESS / 'inventory.json')
    assert inventory['revision'] == REVISION and inventory['total_parameters'] == PARAMETERS
    assert inventory['tensor_payload_bytes_downloaded'] == 0
    assert inventory['full_weight_digest_verified'] is False and not inventory['tokenizer_json']['downloaded']
    fixtures = [dict(id=i, text='stdlib schedule control', paragraph='synthetic IDs only',
        input_ids=[(j + i * 17) % VOCAB for j in range(LENGTH)]) for i in range(2)]
    protocol = build_protocol(fixtures)
    from qwen3_4b_cached_json_v2_protocol import validate
    controls = ['Eight immutable official download specifications', 'Six cached small files match official Git blobs',
        'Original 2049-call schedule equals existing Qwen loop AST including every token/position/slot']
    for label, change in [
        ('Dropped decode call rejected', lambda p:p['operations'].pop(100)),
        ('Changed cache slot rejected', lambda p:p['operations'][100]['requests'][0].update(slot=2)),
        ('Changed decode position rejected', lambda p:p['operations'][100]['requests'][0].update(position=0)),
        ('Dropped slot reuse checkpoint rejected', lambda p:p['checks'].pop(-1)),
        ('Relaxed numerical threshold rejected', lambda p:p['criteria'].update(max_abs_error_max=3.)),
    ]:
        changed = copy.deepcopy(protocol); change(changed)
        try: validate(changed)
        except AssertionError: controls.append(label)
        else: raise AssertionError(label)
    class Denied(Exception): pass
    guarded = []
    def deny(phase):
        guarded.append(phase)
        raise Denied(phase)
    for phase in PHASES:
        try: run(phase, {'guard':deny})
        except Denied as error: assert error.args == (phase,)
        else: raise AssertionError('Admission guard was bypassed: ' + phase)
    try: runtime_tokenizer({'guard':deny})
    except Denied as error: assert error.args == ('tokenizer-runtime',)
    else: raise AssertionError('Serving tokenizer admission guard was bypassed')
    assert guarded == list(PHASES) + ['tokenizer-runtime']
    controls.append('All six phase entry points and serving tokenizer reject before native/network/payload access')
    assert normalize_ids({'input_ids':[[0,8,9,130072,130073]]}) == [0,8,9,130072,130073]
    for value in ([], {'input_ids':[[0],[1]]}, [VOCAB], [-1], [True]):
        try: normalize_ids(value)
        except AssertionError: pass
        else: raise AssertionError('Invalid tokenizer ID container accepted')
    controls.append('Native/runtime ID container normalization preserves IDs and rejects invalid inputs')
    phases = {}
    for row in protocol['checks']: phases[row['phase']] = phases.get(row['phase'],0) + 1
    assert {k:phases[k] for k in ('slot_reuse_prefill','decode_after_slot_reuse','partial_batch','resume_inactive')} == {
        'slot_reuse_prefill':1,'decode_after_slot_reuse':4,'partial_batch':3,'resume_inactive':1}
    assert not any(name.split('.')[0] in {'torch','transformers','tokenizers','numpy','neuronx_distributed_inference'}
        for name in set(sys.modules) - imported_before)
    paths = source_paths()
    return dict(passed=True, scope='stdlib-only metadata/source and inert protocol/guard controls',
        model=MODEL_ID, revision=REVISION, controls=controls, source_files=len(paths),
        native_module_sha256=text_sha(Path(__file__)), download_files=len(manifest),
        future_download_bytes=sum(row['size'] for row in manifest), phases=list(PHASES),
        required_native_reference_frames=dict(original32=32,cold=42,long_and_reuse=93),
        native_long_cached_steps=8172,native_long_forward_calls=8177,device_operations=2049,
        teacher_forced_decode_steps=STEPS,retained_long_rows_by_phase=phases,
        cold_prefixes=list(PREFIXES),full_vocabulary=VOCAB,synthetic_schedule_only=True,
        native_imports=False,network_access=False,weight_payload_bytes_read=0,
        actual_tokenizer_execution=False,actual_numerical_execution=False,full_model_qualified=False)
