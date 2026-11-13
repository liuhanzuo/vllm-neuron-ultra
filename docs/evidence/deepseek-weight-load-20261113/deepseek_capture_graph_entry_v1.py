"""Build/load leaves for an externally admitted DeepSeek diagnostic graph.

Importing this module uses only the standard library and the frozen pure bridge.
There is no CLI, launcher, admission, cleanup, lock or qualification shortcut.
The owning serial worker must authenticate the complete source/weight closure,
fresh-process ownership, resource reservation and predecessor history. Build
and load belong in separate fresh processes. Failed outputs are retained.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import deepseek_cpu_capture_bridge_v1 as bridge

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path('/mnt/nvme/models/DeepSeek-R1-0528-Qwen3-8B-neuron-yarn-exp20261031')
OFFICIAL_MODEL = Path('/mnt/nvme/models/DeepSeek-R1-0528-Qwen3-8B')
ORIGINAL_ARTIFACT = Path('/mnt/nvme/neuron-artifacts/latest-20261031-deepseek-r1-qwen3-8b-bf16-cpu-tp2-b4-ctx4096')
ORIGINAL_CONFIG = ORIGINAL_ARTIFACT / 'neuron_config.json'
ORIGINAL_CONFIG_SHA256 = '26b0382de78b251aaa45fcd2c70e632140bba9f1120579e3888fed5685860a18'
BUCKETS = (128, 256, 512, 1024, 2048, 4096)
TAGS = ('context_encoding_model', 'token_generation_model')
SCHEMA = 'deepseek_capture_graph_build_v1'
REPORT_NAME = 'capture-build.json'
require = bridge.require


def json_bytes(value):
    # Original SDK JSON includes Infinity (quantize_clamp_bound), unchanged.
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def ordinary(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink(),
            'ordinary absolute path required: ' + str(path))
    return path


def file_record(path):
    """Runtime-only streaming fingerprint; never invoked on actual payloads in source tests."""
    path = ordinary(path)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size > 0,
            'nonempty unshared ordinary file required: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(chunk)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, key) == getattr(after, key) for key in fields), 'file changed while hashing')
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def check_paths(artifact, cache, workdir, *, fresh):
    paths = tuple(ordinary(p) for p in (artifact, cache, workdir))
    protected = (ORIGINAL_ARTIFACT, MODEL_PATH, OFFICIAL_MODEL,
                 ROOT / 'deepseek-cold-kv-diagnostic-v2')
    for path in paths:
        require(path.parent.is_dir() and path.name.startswith('deepseek-capture-'),
                'dedicated deepseek-capture-* child of an existing parent required')
        for other in (*paths, *protected):
            if other == path and other not in protected:
                continue
            require(path != other and path not in other.parents and other not in path.parents,
                    'artifact/cache/workdir and original evidence must be disjoint')
        if fresh:
            # SDK trace unconditionally rmtree's an existing compiler_workdir.
            require(not path.exists() and not path.is_symlink(), 'refuse an existing output/workdir')
        else:
            require(path.is_dir(), 'retained build directory required')
    require(len(set(paths)) == 3, 'three distinct output directories required')
    return paths


def check_environment(cache, workdir):
    expected = dict(BASE_COMPILE_WORK_DIR=str(workdir), NEURON_COMPILE_CACHE_URL=str(cache),
                    TRN1_COMPILE_WORKERS='1', NEURON_RT_VISIBLE_CORES='0-1',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    TORCHINDUCTOR_COMPILE_THREADS='1', PYTHONNOUSERSITE='1', PYTHONOPTIMIZE='0')
    require(__debug__ and all(os.environ.get(k) == v for k, v in expected.items()),
            'owning runtime must supply the exact bounded build/load environment')
    # Full environment authentication remains the outer guard's responsibility.
    for name in ('TRN1_CPU_SAMPLER_FAST', 'TRN1_SAMPLER_DIAGNOSTIC_DIR',
                 'NXD_INFERENCE_CAPTURE_SNAPSHOT', 'NXD_LAYOUT_TRANSFORMATION_OPTIONS',
                 'NEURON_COMBINE_KV_ALLOCATIONS', 'NEURON_RT_NUM_CORES'):
        require(not os.environ.get(name), 'unexpected runtime override: ' + name)
    return expected


def original_config():
    require(ORIGINAL_CONFIG.stat().st_size == 8378, 'original serialized config size')
    raw = ORIGINAL_CONFIG.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ORIGINAL_CONFIG_SHA256, 'original serialized config SHA256')
    value = json.loads(raw)
    require(value['_name_or_path'] == str(MODEL_PATH) and MODEL_PATH.is_dir(),
            'original checkpoint-loader view required; no fallback or source redirection')
    require(value['neuron_config']['save_sharded_checkpoint'] is False and
            value['neuron_config']['skip_sharding'] is False and
            value['neuron_config']['skip_warmup'] is False, 'original sharding/warmup config required')
    return bridge.instrument_config_dict(value)


def check_config(actual, expected):
    require(json_bytes(actual) == json_bytes(expected), 'complete original config plus capture-only delta required')


def check_wrapper_configs(model, expected):
    require([wrapper.tag for wrapper in model.models] == list(TAGS), 'exact original two wrappers required')
    check_config(json.loads(model.config.to_json_string()), expected)
    result = {}
    for wrapper in model.models:
        value = json.loads(wrapper.config.to_json_string())
        n = value['neuron_config']
        require(n['buckets'] == list(BUCKETS) and n['batch_size'] == (1 if wrapper.tag == TAGS[0] else 4),
                'all original CTE B1/TKG B4 buckets required')
        require(n['on_device_sampling_config'] is None and n['output_logits'] is False and
                n['tensor_capture_config'] == bridge.capture_config() and
                n['deepseek_cpu_capture_version'] == bridge.VERSION, 'CPU capture wrapper ABI required')
        result[wrapper.tag] = value
    return result


def capture_shape(name, batch):
    boundary = name.rsplit('.', 1)[1]
    if boundary == 'q_normalized': return [batch, 1, 16, 128, 2]
    if boundary == 'k_normalized': return [batch, 1, 4, 128, 2]
    if boundary == 'query': return [batch, 16, 1, 128, 2]
    if boundary in ('key', 'value'): return [batch, 4, 1, 128, 2]
    return [batch, 1, 2048 if boundary == 'attention_heads' else 4096, 2]


def check_metaneff(meta, tag, index):
    """Validate actual future protobuf fields; fake objects suffice for source tests.

The codes below come from the pinned metaneff.proto: FLOAT=1, BFLOAT16=14,
USER_INPUT=1, INPUT_STATE=2, INPUT_WEIGHT=3. num_states/num_weights are not
populated by this SDK writer, so count the actual typed input records.
"""
    require(tag in TAGS and type(index) is int and 0 <= index < 6, 'original graph identity required')
    batch = 1 if tag == TAGS[0] else 4
    require(meta.model_config.return_aliases is False, 'device aliases must be hidden')
    outputs, inputs = list(meta.output_tensors), list(meta.input_tensors)
    require(len(outputs) == 541, 'one logits head + 468 captures + 72 aliased KV outputs required')
    for i, tensor in enumerate(outputs):
        require(tensor.name == f'output{i}'.encode(), 'original ordered output names required')
    require(list(outputs[0].shape) == [batch, 1, 151936] and outputs[0].data_type == 1,
            'complete FP32 vocabulary logits required')
    for name, tensor in zip(bridge.NAMES, outputs[1:469]):
        require(list(tensor.shape) == capture_shape(name, batch) and tensor.data_type == 14,
                name + ': exact BF16 geometry and both TP ranks required')
    aliases = dict(meta.output_aliases_to)
    bridge.check_alias_indices(sorted(aliases))
    require(len(set(aliases.values())) == 72, 'every KV state must have a distinct alias')
    states, weights, user_inputs = {}, [], []
    for i, tensor in enumerate(inputs):
        require(tensor.name == f'input{i}'.encode(), 'original ordered input names required')
        require(tensor.type in (1, 2, 3), 'untyped graph input rejected')
        if tensor.type == 2: states[i] = tensor
        elif tensor.type == 3:
            key = tensor.checkpoint_key.decode('utf8')
            require(bool(key), 'named weight inputs required')
            weights.append(dict(key=key, shape=list(tensor.shape), dtype=tensor.data_type))
        else: user_inputs.append(tensor)
    require(len(states) == 72 and set(aliases.values()) == set(states), 'all and only the 72 original KV inputs aliased')
    state_rows = []
    for kv_index, output_index in enumerate(range(469, 541)):
        source = states[aliases[output_index]]
        key = f'kv_mgr.past_key_values.{kv_index}'
        require(source.checkpoint_key == key.encode(), 'alias targets must preserve exact KV order')
        for tensor in (source, outputs[output_index]):
            require(list(tensor.shape) == list(bridge.KV_SHAPE) and tensor.data_type == 14,
                    'full original BF16 KV state geometry required')
        state_rows.append(dict(output_index=output_index, input_index=aliases[output_index], key=key))
    require(len(user_inputs) >= 2 and list(user_inputs[0].shape) == [batch, BUCKETS[index] if batch == 1 else 1]
            and user_inputs[0].data_type == user_inputs[1].data_type == 2 and
            list(user_inputs[1].shape) == [batch, BUCKETS[index]],
            'actual original bucket input/mask geometry required')
    require(weights and len({row['key'] for row in weights}) == len(weights), 'unique complete observed weight-key inventory')
    return dict(tag=tag, index=index, bucket=BUCKETS[index], outputs=541, device_visible_outputs=469,
                state_aliases=state_rows, weight_inputs=weights,
                baseline_weight_layout_equality_proven=False, output_semantics_neutrality_proven=False)


def _native_dependencies():
    import torch
    import server_with_bias_fix as wrapper
    from neuronx_distributed.trace import model_builder
    from torch_neuronx.proto import metaneff_pb2
    from deepseek_qwen3_capture_adapter_v1 import DeepSeekCaptureInferenceConfig, DeepSeekCaptureForCausalLM
    require(wrapper.workers == 1 and
            model_builder.concurrent.futures.ThreadPoolExecutor is wrapper.LimitedCompilerPool,
            'existing process-local one-worker compiler cap must actually be installed')
    torch.set_num_threads(1)
    return DeepSeekCaptureInferenceConfig, DeepSeekCaptureForCausalLM, metaneff_pb2.MetaNeff


def _begin(artifact, cache, workdir, execution_guard, *, fresh):
    require(callable(execution_guard), 'external authenticated serial execution guard required')
    execution_guard()
    require(not any(name.split('.')[0] in ('torch', 'torch_neuronx', 'neuronx_distributed',
                                          'neuronx_distributed_inference') for name in sys.modules),
            'fresh build/load process before native imports required')
    paths = check_paths(artifact, cache, workdir, fresh=fresh)
    environment = check_environment(paths[1], paths[2])
    expected = original_config()
    execution_guard()
    return paths, environment, expected


def _inspect(model, artifact, workdir, expected, meta_cls, execution_guard):
    wrappers = check_wrapper_configs(model, expected)
    require(not (artifact / 'weights').exists() and not (artifact / 'weights').is_symlink(),
            'original configuration must not save new weight shards')
    check_config(json.loads((artifact / 'neuron_config.json').read_text()), expected)
    check_config(json.loads((workdir / 'neuron_config.json').read_text()), expected)
    files = {str(p): file_record(p) for p in (artifact / 'model.pt', artifact / 'neuron_config.json',
                                              workdir / 'neuron_config.json')}
    rows = []
    require(sorted(str(p.relative_to(workdir)) for p in workdir.glob('*/_tp0_bk*/metaneff.pb')) ==
            sorted(f'{tag}/_tp0_bk{i}/metaneff.pb' for tag in TAGS for i in range(6)),
            'exactly twelve retained graph metadata files required')
    for tag in TAGS:
        for i, bucket in enumerate(BUCKETS):
            execution_guard()
            folder = workdir / tag / f'_tp0_bk{i}'
            config = deepcopy(wrappers[tag])
            config['neuron_config']['buckets'] = [bucket]
            config['neuron_config']['context_encoding_buckets' if tag == TAGS[0] else 'token_generation_buckets'] = [bucket]
            check_config(json.loads((folder / 'neuron_config.json').read_text()), config)
            for name in ('neuron_config.json', 'metaneff.pb', 'graph.neff'):
                path = folder / name
                files[str(path)] = file_record(path)
            raw = (folder / 'metaneff.pb').read_bytes()
            require(hashlib.sha256(raw).hexdigest() == files[str(folder / 'metaneff.pb')]['sha256'],
                    'metadata changed before protobuf decoding')
            meta = meta_cls(); meta.ParseFromString(raw)
            rows.append(check_metaneff(meta, tag, i))
    execution_guard()
    return files, rows


def build(artifact, cache, workdir, execution_guard):
    """Compile once in fresh namespaces; return a report to the owning worker.

All native operations occur only after the external guard. The guard must
reserve compile/cache/artifact/shared-memory/weight peaks in addition to the
47,150,344,192-byte two-arm diagnostic output budget. This leaf grants none.
"""
    (artifact, cache, workdir), environment, expected = _begin(
        artifact, cache, workdir, execution_guard, fresh=True)
    config_cls, model_cls, meta_cls = _native_dependencies()
    cfg = config_cls.from_json_string(json.dumps(expected))
    check_config(json.loads(cfg.to_json_string()), expected)
    execution_guard()
    model = model_cls(str(MODEL_PATH), config=cfg)
    check_wrapper_configs(model, expected)
    builder = model.get_builder()
    require(Path(builder.compiler_workdir) == workdir, 'dedicated actual compiler workdir required')
    require(list(builder.model_collection) == list(TAGS) and
            all(len(builder.model_collection[tag].example_inputs) == 6 for tag in TAGS), 'all twelve graphs required')
    execution_guard()
    check_environment(cache, workdir)
    check_paths(artifact, cache, workdir, fresh=True)
    model.compile(str(artifact), debug=False, pre_shard_weights_hook=None, dry_run=False)
    execution_guard()
    files, graphs = _inspect(model, artifact, workdir, expected, meta_cls, execution_guard)
    report = dict(schema=SCHEMA, complete=True, artifact=str(artifact), cache=str(cache), workdir=str(workdir),
                  model_path=str(MODEL_PATH), original_config_sha256=ORIGINAL_CONFIG_SHA256,
                  config_sha256=hashlib.sha256(json_bytes(expected)).hexdigest(), environment=environment,
                  files=files, graphs=graphs, full_model_qualified=False, new_speedup_claim=False,
                  output_semantics_neutrality_proven=False, baseline_weight_layout_equality_proven=False)
    execution_guard()
    with (artifact / REPORT_NAME).open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    return report


def load(artifact, cache, workdir, expected_report_sha256, execution_guard):
    """Authenticate completed build evidence, load without warmup, return model.

expected_report_sha256 comes from the owning worker's independent actual-exit
audit, never from hashing an untrusted report immediately before this call.
Full observer-off/on raw neutrality, lifecycle144 and the original failed cold
gate remain mandatory downstream. Loading is not inference qualification.
"""
    (artifact, cache, workdir), environment, expected = _begin(
        artifact, cache, workdir, execution_guard, fresh=False)
    require(isinstance(expected_report_sha256, str) and len(expected_report_sha256) == 64 and
            all(c in '0123456789abcdef' for c in expected_report_sha256), 'externally authenticated build-report digest required')
    report_path = ordinary(artifact / REPORT_NAME)
    require(report_path.stat().st_size <= 4 * 1024**2, 'bounded build report required')
    raw = report_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_report_sha256, 'authenticated build-report SHA256 mismatch')
    report = json.loads(raw)
    require(report['schema'] == SCHEMA and report['complete'] is True and
            (report['artifact'], report['cache'], report['workdir']) == tuple(map(str, (artifact, cache, workdir))) and
            report['environment'] == environment and report['model_path'] == str(MODEL_PATH) and
            report['original_config_sha256'] == ORIGINAL_CONFIG_SHA256 and
            report['config_sha256'] == hashlib.sha256(json_bytes(expected)).hexdigest(), 'exact build identity required')
    require(all(report[key] is False for key in ('full_model_qualified', 'new_speedup_claim',
            'output_semantics_neutrality_proven', 'baseline_weight_layout_equality_proven')),
            'build evidence cannot grant model/neutrality/weight-layout qualification')
    # Require every file, not an arbitrary subset from a report. Reject any
    # untrusted path before hashing and before SDK or torch.jit deserialization.
    expected_files = {str(p) for p in (artifact / 'model.pt', artifact / 'neuron_config.json', workdir / 'neuron_config.json')}
    expected_files.update(str(workdir / tag / f'_tp0_bk{i}' / name)
                          for tag in TAGS for i in range(6) for name in ('neuron_config.json', 'metaneff.pb', 'graph.neff'))
    require(set(report['files']) == expected_files, 'complete exact 39-file build identity required')
    for path, identity in report['files'].items():
        execution_guard()
        require(file_record(path) == identity, 'compiled artifact/config/graph changed: ' + path)
    execution_guard()
    config_cls, model_cls, meta_cls = _native_dependencies()
    cfg = config_cls.from_json_string(json.dumps(expected))
    check_config(json.loads(cfg.to_json_string()), expected)
    model = model_cls(str(MODEL_PATH), config=cfg)
    files, graphs = _inspect(model, artifact, workdir, expected, meta_cls, execution_guard)
    require(files == report['files'] and graphs == report['graphs'], 'actual graph metadata must match authenticated build')
    execution_guard()
    check_environment(cache, workdir)
    model.load(str(artifact), skip_warmup=True)
    execution_guard()
    check_wrapper_configs(model, expected)
    require(model.is_loaded_to_neuron is True, 'completed SDK load required')
    from qwen35_split_state_checks_v4 import graph_routes
    from deepseek_prior_mask_state_v1 import bindings
    routes, states = graph_routes(model), bindings(model)
    require(routes['complete'] and len(routes['graphs']) == 12 and len(states) == 144,
            'all loaded graph routes and TP state objects required')
    execution_guard()
    return model
