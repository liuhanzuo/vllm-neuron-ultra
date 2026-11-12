# SPDX-License-Identifier: Apache-2.0
"""Exact original tokenizer protocols for future bounded preflight; native imports lazy."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
from contextlib import contextmanager

TOKENIZER_NAMES = ('tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt', 'chat_template.jinja')
TEMPLATE_ARGS = dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
ORIGINAL_ROOT = '/mnt/nvme/models/Qwen3.5-2B'
HISTORICAL_VIEW = '/mnt/nvme/models/Qwen3.5-2B-neuron-text-exp20261031'


def require(value, message):
    if not value: raise RuntimeError(message)


def tokenizer_paths(work):
    roots = [Path(ORIGINAL_ROOT), Path(HISTORICAL_VIEW), Path(work) / 'shared-hf-config']
    return {str(root / name) for root in roots for name in ('config.json', *TOKENIZER_NAMES)}


def verify_payload(path, descriptor):
    """Read only one exact pinned tokenizer file, streamed; never weights."""
    path = Path(path)
    require(path.name in (*TOKENIZER_NAMES, 'config.json'), 'Not an allowed tokenizer/config filename')
    require(descriptor['size'] <= 16 * 2**20, 'Tokenizer per-file byte ceiling')
    with path.open('rb') as stream:
        before = path.stat(); require(before.st_size == descriptor['size'], 'Pinned tokenizer size changed')
        algorithm = descriptor['algorithm']
        require(algorithm in ('git-blob-sha1', 'sha256'), 'Unknown tokenizer digest scheme')
        digest = hashlib.sha1() if algorithm == 'git-blob-sha1' else hashlib.sha256()
        if algorithm == 'git-blob-sha1': digest.update(('blob ' + str(before.st_size) + '\0').encode())
        count = 0
        while block := stream.read(1024 * 1024):
            count += len(block); require(count <= descriptor['size'], 'Tokenizer grew during verification')
            digest.update(block)
        after = path.stat()
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(fields(before) == fields(after) and count == descriptor['size'] and digest.hexdigest() == descriptor['digest'],
            'Pinned original tokenizer bytes changed: ' + path.name)
    return dict(file=path.name, bytes=count, algorithm=algorithm, digest=digest.hexdigest())


def lineage(bundle, work):
    inputs = bundle['tokenizer']; original = Path(ORIGINAL_ROOT); historical = Path(HISTORICAL_VIEW)
    view = Path(work) / 'shared-hf-config'
    require(set(inputs['files']) == {'config.json', *TOKENIZER_NAMES}, 'Exact tokenizer-only file inventory required')
    # Config was authenticated and constructed by the preserved v1 config checks.
    config = (view / 'config.json').read_bytes()
    require(hashlib.sha256(config).hexdigest() == bundle['view_config_sha256'], 'Packaged text config changed')
    require((historical / 'config.json').stat().st_size == len(config), 'Historical config size changed')
    require((historical / 'config.json').read_bytes() == config, 'Historical/package view config bytes differ')
    receipts = []
    for name, descriptor in inputs['files'].items():
        path = original / name
        require(path.resolve() == path, 'Original tokenizer must remain at its pinned ordinary path')
        receipts.append(verify_payload(path, descriptor))
        if name == 'config.json': continue
        old_link = historical / name
        require(old_link.is_symlink() and old_link.resolve() == path, 'Historical tokenizer lineage changed: ' + name)
        new_link = view / name
        if new_link.is_symlink():
            require(new_link.resolve() == path, 'Prepared tokenizer lineage changed: ' + name)
        else:
            require(not new_link.exists(), 'Cannot overwrite prepared tokenizer file')
            new_link.symlink_to(path)
    return dict(original=str(original), historical_view=str(historical), packaged_view=str(view),
                pinned_files=receipts, unchanged_original_symlinks=True, model_weights_read=False)


def checked_protocol(inputs, key):
    row = inputs['sources'][key]; p = Path(row['path'])
    require(p.is_absolute() and p.resolve() == p and p.suffix in ('.py', '.json'), 'Ordinary protocol source/metadata required')
    require(p.stat().st_size == row['bytes'] <= 4 * 2**20, 'Protocol source/metadata byte ceiling')
    raw = p.read_bytes()
    require(len(raw) == row['bytes'] <= 4 * 2**20 and hashlib.sha256(raw).hexdigest() == row['sha256'], 'Tokenization protocol/source changed: ' + key)
    return raw


def function_node(source, name, parent=None):
    tree = ast.parse(source)
    if parent: tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parent)
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def original_function(source, name, namespace):
    node = function_node(source, name)
    scope = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<pinned-original-tokenizer-protocol>', 'exec'), scope)
    return scope[name]


@contextmanager
def tokenizer_provider(tokenizer):
    """Inject only the original functions' tokenizer dependency, never HTTP/model work."""
    original = sys.modules.get('orchestrate')
    module = ModuleType('orchestrate'); module.prompt_tokenizer = lambda case: tokenizer
    sys.modules['orchestrate'] = module
    try: yield module
    finally:
        if original is None: sys.modules.pop('orchestrate', None)
        else: sys.modules['orchestrate'] = original


class RecordingTokenizer:
    def __init__(self, real): self.real, self.calls = real, []
    def apply_chat_template(self, messages, **kwargs):
        require(kwargs == TEMPLATE_ARGS, 'Original chat-template arguments changed')
        require(len(messages) == 1 and messages[0].get('role') == 'user', 'Original single user message changed')
        ids = self.real.apply_chat_template(messages, **kwargs)
        require(type(ids) is list and ids and all(type(n) is int for n in ids), 'Expected full unmodified list of token IDs')
        self.calls.append(dict(messages=copy.deepcopy(messages), kwargs=dict(kwargs), input_ids=list(ids)))
        return ids
    def encode(self, text, **kwargs):
        require(kwargs == {'add_special_tokens': False}, 'Long-reference body encoding arguments changed')
        ids = self.real.encode(text, **kwargs)
        require(type(ids) is list and ids and all(type(n) is int for n in ids), 'Expected original body token IDs')
        return ids


def projected_prompt(source, method, namespace):
    """Execute the exact prompt expression from the real HTTP client; no request body."""
    fn = function_node(source, method, 'NativeClients')
    assignments = [n for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prompt' for t in n.targets)]
    require(len(assignments) == 1, 'Unique actual HTTP prompt assignment required')
    scope = dict(namespace)
    exec(compile(ast.Module(body=assignments, type_ignores=[]), '<original-http-prompt-expression>', 'exec'), scope)
    return scope['prompt']


def long_reconstruction(source, tokenizer):
    """Project unchanged source statements before its first tensor/model operation."""
    tree = ast.parse(source); fn = function_node(source, 'reference')
    length = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LENGTH' for t in n.targets))
    initial = [n for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('prefixes', 'paragraphs') for t in n.targets)]
    require(len(initial) == 2, 'Original long text inputs unavailable')
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and ast.unparse(n.target) == '(i, (prompt, paragraph))')
    statements = []
    for item in loop.body:
        if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'tensor' for t in item.targets): break
        statements.append(copy.deepcopy(item))
    require(statements and isinstance(statements[-1], ast.Assert), 'Long tokenizer slice boundary changed')
    require(not any(isinstance(n, ast.Name) and n.id in ('torch', 'model') for item in statements for n in ast.walk(item)), 'Native/model work entered tokenizer projection')
    projected = copy.deepcopy(loop)
    projected.body = statements + [ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='captured', ctx=ast.Load()), attr='append', ctx=ast.Load()), args=[ast.Name(id='ids', ctx=ast.Load())], keywords=[]))]
    scope = dict(tok=tokenizer, LENGTH=length, captured=[])
    module = ast.fix_missing_locations(ast.Module(body=initial + [projected], type_ignores=[]))
    exec(compile(module, '<original-long-tokenizer-statements>', 'exec'), scope)
    require(len(scope['captured']) == 2, 'Original long sequence count changed')
    return scope['captured']


def encode_protocol(tokenizer, inputs):
    recorded = RecordingTokenizer(tokenizer)
    case = json.loads(checked_protocol(inputs, 'case'))
    suite = json.loads(checked_protocol(inputs, 'quality_suite'))
    require(len(suite['cases']) == len({x['id'] for x in suite['cases']}) == 96, 'Complete unique quality96 required')
    quality_source = checked_protocol(inputs, 'quality_source').decode()
    workload_source = checked_protocol(inputs, 'workload_source').decode()
    orchestrate_source = checked_protocol(inputs, 'orchestrate_source').decode()
    driver_source = checked_protocol(inputs, 'driver_source').decode()
    hf_reference = json.loads(checked_protocol(inputs, 'hf_reference'))
    long_reference = json.loads(checked_protocol(inputs, 'long_reference'))
    historical_quality = json.loads(checked_protocol(inputs, 'historical_quality'))
    historical_workloads = json.loads(checked_protocol(inputs, 'historical_workloads'))
    with tokenizer_provider(recorded) as dependency:
        quality = original_function(quality_source, 'prepare_inputs',
            {'ROOT': Path(inputs['sources']['quality_source']['path']).parent, 'json': json, 'hashlib': hashlib})
        quality_rows = quality(case, suite)
        workloads = original_function(workload_source, 'inputs', {})
        workload_ids = workloads(case)
        prompt_ids = original_function(orchestrate_source, 'prompt_ids', {'prompt_tokenizer': dependency.prompt_tokenizer})
        initial = projected_prompt(driver_source, 'initial', {'prompt_ids': prompt_ids, 'case': case})
        diagnostic = projected_prompt(driver_source, 'diagnostic', {'inputs': workloads, 'case': case})
        # Actual smoke literal checklist; its HTTP calls are not evaluated.
        smoke_fn = function_node(orchestrate_source, 'quality')
        checks = next(ast.literal_eval(n.value) for n in smoke_fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'checks' for t in n.targets))
        smoke = [prompt_ids(case, text) for text, expected in checks]
        six = [prompt_ids(case, row['text']) for row in hf_reference['cases']]
        long = long_reconstruction(checked_protocol(inputs, 'long_source').decode(), recorded)
    require(len(six) == 6 and len(long) == 2, 'Saved HF/long fixture coverage changed')
    require(set(workload_ids) == {32, 128, 512, 1536} and diagnostic == workload_ids[128], 'Actual HTTP input protocol changed')
    require(len(smoke) == 4 and len(historical_quality['rows']) == 96
        and len({r['id'] for r in historical_quality['rows']}) == 96, 'Historical/smoke coverage changed')
    return dict(quality=[dict(id=row['id'], input_ids=ids) for row, ids in quality_rows],
        workloads={str(k): v for k, v in workload_ids.items()}, initial=initial, diagnostic=diagnostic,
        smoke=smoke, six_hf=six, long=long, template_calls=recorded.calls,
        saved_six=[r['input_ids'] for r in hf_reference['cases']],
        saved_long=[r['input_ids'] for r in long_reference['fixtures']],
        saved_quality={r['id']: r['prompt_ids'] for r in historical_quality['rows']},
        saved_workloads=historical_workloads['inputs'])


def compare_encodings(original, view):
    for key in ('quality', 'workloads', 'initial', 'diagnostic', 'smoke', 'six_hf', 'long', 'template_calls'):
        require(original[key] == view[key], 'Complete original/view token IDs differ: ' + key)
    require(len(original['quality']) == 96, 'Missing quality cases')
    for result in (original, view):
        require({r['id']: r['input_ids'] for r in result['quality']} == result['saved_quality'],
                'Current quality96 token IDs differ from historical paired baseline')
        require(result['workloads'] == result['saved_workloads'],
                'Current HTTP workload IDs differ from historical baseline')
        require(result['six_hf'] == result['saved_six'], 'Current tokenizer differs from saved six original HF sequences')
        require(result['long'] == result['saved_long'], 'Reconstructed long sequence differs from saved original IDs')
    return dict(all96_quality_ids_equal=True, exact_workload_lengths=[32, 128, 512, 1536],
        initial_http_equal=True, diagnostic128_equal=True, smoke_unique_prompts=4,
        historical_quality96_equal=True, historical_workload_inputs_equal=True,
        saved_six_hf_equal=True, reconstructed_long_sequences_equal=2,
        same_template_arguments=TEMPLATE_ARGS,
        unavailable_reference_evidence=[
            'No saved original HF quality96, workload, initial or diagnostic tokenization is pinned; current original/view equality is required, with historical served-view IDs also required for quality96/workloads.',
            'Only the first smoke prompt also has a saved original HF sequence; all four current original/view smoke inputs are compared.'],
        execution_scope=dict(complete_original_functions=['evaluate_quality.prepare_inputs', 'benchmark_workloads.inputs', 'orchestrate.prompt_ids'],
            source_projections=['NativeClients.initial prompt assignment', 'NativeClients.diagnostic prompt assignment',
                'orchestrate.quality literal checklist', 'validate_qwen35_long_decode.reference pre-tensor tokenizer statements'],
            http_or_model_execution=False),
        scope='Input token IDs only; no output generation, model, quality score or performance result')


def run(bundle, work):
    evidence = lineage(bundle, work)
    from transformers import AutoTokenizer
    # Preserve warning and original fixed tokenization: no Mistral fix or suppression.
    original = AutoTokenizer.from_pretrained(evidence['original'], local_files_only=True)
    view = AutoTokenizer.from_pretrained(evidence['packaged_view'], local_files_only=True)
    left = encode_protocol(original, bundle['tokenizer'])
    right = encode_protocol(view, bundle['tokenizer'])
    proof = compare_encodings(left, right)
    return dict(passed=True, lineage=evidence, comparison=proof, original=left, view=right,
                tokenizer_payload_read=True, model_weights_read=False, native_tokenizer_executed=True)
