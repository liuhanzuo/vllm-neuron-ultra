"""Source-only additive candidate; no production imports, history or admission.

Reads only bounded .py/.json text. Reuses the exact mature source collector
definitions in isolation. --write creates two new research reports exclusively.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT / 'research'
COLLECTOR = Q / 'prepare_dory_residual_spec_v3.py'
COLLECTOR_SHA = '8e4ef981c363c7136ffa90ee842dab8dfc5b754aced23f5ccb7d2e9ad7772d8e'
ORIGINAL = Q / 'lfm3-nki05-execution-source-closure-20261111-v1.json'
ORIGINAL_SHA = 'ed390bea54408f841402b5adbbf18d332edef77b78d1bd092a06c18b3701bada'
TERMINAL = Q / 'lfm3-nki05-transport-r5-explicit-control-root-actual-exit-20261111-v1.json'
ROOT_REVIEW = Q / 'lfm3-full-cpu-integration-root-source-review-20261111-v1.json'
RUNNER = ROOT / 'run_lfm3_nki05_full_cpu_serial_v2.py'
PRIOR = Q / 'lfm3-full-cpu-execution-source-candidate-20261111-v1.json'
PRIOR_SHA = '94d7b02cf6948e2e77cdde9f8ed166e6dba0996ad7956d00de6d930c238f185b'
RUNTIME_PLAN = ROOT / 'lfm3-movement-device-diagnostic-v3/plan.json'
RUNTIME_PLAN_SHA = '846251d079aa459edf699c30b2a2a35688e13b4fe2a4b5b398be402563065374'
ANALYSIS = Q / 'lfm3-full-cpu-execution-source-analysis-20261111-v2.json'
OUTPUT = Q / 'lfm3-full-cpu-execution-source-candidate-20261111-v2.json'
NEW_SEEDS = tuple(ROOT / name for name in (
    'run_lfm3_nki05_full_cpu_serial_v2.py', 'lfm3_nki05_full_cpu_leaf_v2.py',
    'run_lfm3_typed_writer_cpu_pilot_v3.py', 'lfm3_typed_writer_cpu_pilot_protocol_v3.py',
    'lfm3_typed_fixture_ledger_leaf_v1.py', 'lfm3_nki05_compact_writeback_v1.py',
    'slot_cache_nki05_typed_selector_v1.py'))
EXTENDED = ROOT / 'lfm3_batch_slots_bits_extended_v4.py'


def require(value, message):
    if not value:
        raise AssertionError(message)


def text_bytes(path):
    path = Path(path)
    require(path.is_absolute() and path.suffix in ('.py', '.json'), 'Only Python/JSON text')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= 8 * 2**20, 'Bounded ordinary text')
    data = path.read_bytes()
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(len(data) == before.st_size and all(getattr(before, k) == getattr(after, k) for k in fields),
            'Stable source/metadata text')
    data.decode('utf-8')
    return data


def sha(path):
    return hashlib.sha256(text_bytes(path)).hexdigest()


def reference(path):
    return dict(path=str(path), sha256=sha(path))


def read(path, expected=None):
    data = text_bytes(path)
    require(Path(path).suffix == '.json', 'JSON metadata only')
    require(expected is None or hashlib.sha256(data).hexdigest() == expected, 'Metadata pin: ' + str(path))
    return json.loads(data)


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def merge(pins, additions):
    for path, expected in additions.items():
        require(path not in pins or pins[path] == expected, 'Original source pin conflict: ' + path)
        pins[path] = expected


def collector():
    data = text_bytes(COLLECTOR)
    require(hashlib.sha256(data).hexdigest() == COLLECTOR_SHA, 'Exact mature collector')
    nodes = [n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef)
             and n.name in ('python_bytes', 'local_imports')]
    require(len(nodes) == 2, 'Two original definitions only')
    namespace = dict(ast=ast, hashlib=hashlib, Path=Path, stat=stat, require=require,
                     ENTRIES=(), ROOTS=(Q, ROOT, Q / 'dory-native-reference-source-20261107-v1'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(COLLECTOR), 'exec', dont_inherit=True), namespace)
    return namespace['local_imports']


def new_dynamic_calls(paths):
    rows = []
    class Visitor(ast.NodeVisitor):
        def __init__(self, path):
            self.path, self.scope = path, []
        def visit_FunctionDef(self, node):
            self.scope.append(node.name); self.generic_visit(node); self.scope.pop()
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_Call(self, node):
            function = node.func
            names = {'exec', 'eval', 'compile', '__import__'}
            attributes = {'import_module', 'spec_from_file_location', 'module_from_spec',
                          'exec_module', 'load_module', 'run_module', 'run_path'}
            if (isinstance(function, ast.Name) and function.id in names
                    or isinstance(function, ast.Attribute) and function.attr in attributes):
                rows.append(dict(path=self.path, line=node.lineno, function='.'.join(self.scope),
                                 call=ast.unparse(node)))
            self.generic_visit(node)
    for path in sorted(paths):
        Visitor(path).visit(ast.parse(text_bytes(path), filename=path))
    return rows


def pure_verifier_checks(pins, original_pins, graph):
    """Unchanged runner definitions with text IO and synthetic local modules."""
    tree = ast.parse(text_bytes(RUNNER))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('verify_sources', 'execution_source_paths')]
    require(len(nodes) == 2, 'Exact two frozen verifier definitions')
    process = ROOT / 'minicpm5_layer0_execution_protocol_v4.py'
    descriptor = Q / 'descriptor_serial_lock_proof_v1.py'
    modules = {f'synthetic_local_{i}': SimpleNamespace(__file__=path)
               for i, path in enumerate(sorted(graph['source_sha256']))}
    namespace = dict(Path=Path, ROOT=ROOT, SELF=str(RUNNER),
                     p=SimpleNamespace(require=require, sha=sha), sys=SimpleNamespace(modules=modules),
                     namespace_sources=SimpleNamespace(known_namespace=lambda *args: False),
                     ORIGINAL_HELPERS={str(process): '81e7d7dd913c6b85f0add7b946692d3270f80cc8f2fe980ecad543251657f185',
                                       str(descriptor): 'f3faa10356e16dce6b689298ceebf11441f3821ce391f54896cbffed1c82b003'},
                     diagnostic=SimpleNamespace(__file__=str(ROOT / 'lfm3_nki05_transport_diagnostic_v3.py')),
                     history=SimpleNamespace(__file__=str(ROOT / 'lfm3_nki05_serial_history_v1.py')),
                     science=SimpleNamespace(__file__=str(ROOT / 'lfm3_nki05_full_cpu_leaf_v2.py')))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNNER), 'exec', dont_inherit=True), namespace)
    verifier = namespace['execution_source_paths']
    require(verifier(pins) == sorted(map(Path, pins)), 'Full explicit source map')
    outcomes = {'complete_map_with_all_local_graph_sources': True}
    def rejects(label, candidate, message):
        try:
            verifier(candidate)
        except AssertionError as error:
            require(message in str(error), 'Specific expected verifier rejection')
            outcomes[label] = str(error)
        else:
            raise AssertionError('Mutation unexpectedly passed: ' + label)
    rejects('previous590_map_rejected_for_full_cpu_v2', original_pins, 'Every loaded local helper pinned:')
    missing = dict(pins); missing.pop(str(EXTENDED))
    rejects('missing_AST_loaded_extended_generator_rejected', missing, str(EXTENDED))
    changed = dict(pins); changed[str(ROOT / 'lfm3_movement_device_protocol_v3.py')] = '0' * 64
    rejects('changed_runtime_verifier_source_rejected', changed, 'Exact execution dependency source:')
    namespace['sys'].modules = {}
    require(verifier(pins) == sorted(map(Path, pins)), 'Explicit closure independent of initial sys.modules')
    outcomes['empty_initial_modules_keeps_complete_explicit_map'] = True
    try:
        merge(dict(original_pins), {next(iter(original_pins)): '0' * 64})
    except AssertionError:
        outcomes['old_pin_rebinding_rejected'] = True
    else:
        raise AssertionError('Old pin rebinding unexpectedly passed')
    outcomes['actual_runtime_namespace_authentication_exercised'] = False
    outcomes['production_modules_imported'] = False
    return outcomes


def build():
    require(not any(name.split('.')[0] in {'torch', 'numpy', 'nki', 'neuronxcc', 'transformers', 'vllm'}
                    for name in sys.modules), 'No framework imports in source-only builder')
    terminal = read(TERMINAL)
    spec_ref = terminal['spec']
    spec = read(Path(spec_ref['path']), spec_ref['sha256'])
    r5 = spec['source_sha256']
    prior = read(PRIOR, PRIOR_SHA)
    prior_analysis = read(Path(prior['analysis']['path']), prior['analysis']['sha256'])
    previous = prior['source_sha256']
    original = read(ORIGINAL, ORIGINAL_SHA)
    require(len(r5) == 570 and len(previous) == 590 and len(original['source_sha256']) == 558, 'Exact previous source counts')
    require(all(previous.get(p) == h for p,h in r5.items()), 'Complete r5 preserved in fullCPU v1 candidate')
    require(all(previous.get(p) == s for p, s in original['source_sha256'].items()), 'All original558 retained')
    review = read(ROOT_REVIEW)
    for path, expected in review['source_sha256'].items():
        if Path(path).suffix == '.py':
            require(sha(path) == expected, 'Frozen source review Python pin')
    plan = read(RUNTIME_PLAN, RUNTIME_PLAN_SHA)
    require(len(plan['runtime']['distributions']) == 7, 'Original seven runtime distributions')
    finite = copy.deepcopy(prior_analysis['finite_dynamic_edges'])
    seeds = list(NEW_SEEDS) + [Path(p) for p in previous if Path(p).is_relative_to(ROOT)]
    seeds += [Path(row['target']) for row in finite]
    graph = collector()(entries=seeds)
    pins = dict(previous); merge(pins, graph['source_sha256'])
    require(558 <= len(pins) <= 1024, 'Unchanged runner source-map count cap')
    require(all(sha(path) == expected for path, expected in pins.items()), 'Every Python source matches')
    require(all(target in pins for targets in graph['edges'].values() for target in targets), 'All literal edges closed')
    require(all(row['target'] in pins for row in finite), 'All finite dynamic targets retained')
    require(pins[str(EXTENDED)] == 'fdd5396b9309335800fda28fe529daaea6b70af39316bfbbd6fbc152fee4b4e8',
            'Exact extended fixture source pin')
    additions = {p: s for p, s in sorted(pins.items()) if p not in previous}
    new_rows = new_dynamic_calls(additions)
    reasons = {}
    for path in additions:
        reasons[path] = dict(explicit_new_seed=Path(path) in NEW_SEEDS,
                             literal_callers=[caller for caller, targets in graph['edges'].items() if path in targets],
                             finite_dynamic_callers=[row for row in finite if row['target'] == path])
        require(any(reasons[path].values()), 'Every addition has a source derivation')
    checks = pure_verifier_checks(pins, previous, graph)
    runtime_issue = dict(status='v2_source_fix_pending_independent_root_review_and_actual_runtime',
                        finding='v2 sets PYTHONPYCACHEPREFIX in the clean pre-start environment and calls the unchanged original isolation predicate before the seven-distribution verifier and every batch native import.',
                        condition='Original pinned NumPy RECORD hashed-bytecode branch remains fully required; actual RECORD/native bytes not read here. Frozen v1 remains unchanged and unsuitable for launch.',
                        sources=[str(ROOT / name) for name in ('run_lfm3_nki05_full_cpu_serial_v2.py',
                                 'lfm3_nki05_full_cpu_leaf_v2.py', 'lfm3_movement_device_protocol_v3.py',
                                 'smollm3_record_bytecode_v1.py')])
    analysis = dict(schema='lfm3_full_cpu_execution_source_analysis_v2', source_only=True,
                    collector=dict(**reference(COLLECTOR), definitions=['python_bytes', 'local_imports'],
                                   isolated_definition_execution_only=True, roots=[str(Q), str(ROOT), str(Q / 'dory-native-reference-source-20261107-v1')],
                                   max_source_files=2048, max_parsed_source_bytes=64*2**20),
                    graph=graph, seed_paths=sorted(set(map(str, seeds))), addition_derivation=reasons,
                    finite_dynamic_edges=finite, inherited_dynamic_call_inventory=(prior_analysis['inherited_dynamic_call_inventory']
                        + prior_analysis['added_source_dynamic_or_code_auth_calls']),
                    previous590_candidate=reference(PRIOR),
                    added_source_dynamic_or_code_auth_calls=new_rows,
                    dynamic_scope=dict(all_literal_local_import_branches_scanned=True,
                                       all_known_finite_local_dynamic_targets_closed=True,
                                       unknown_dynamic_imports_resolved=False,
                                       complete_external_framework_closure_claimed=False,
                                       inherited_inventory_preserved_verbatim=True,
                                       new_local_exec_target=str(EXTENDED),
                                       new_compile_only_sources=[str(ROOT/'run_lfm3_typed_writer_cpu_pilot_v3.py'),
                                                                '/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/nki/_simulator.py'],
                                       remaining_unknowns=['Inherited runtime-derived import_module/class modules and external vLLM/NxDI/Transformers/rotary imports remain catalogued but unresolved.',
                                                           'NumPy/ml_dtypes old/new environments and NKI lazy Torch imports are external runtime boundaries; existing12 SDK/Torch text pins are preserved, not promoted to a full package closure.',
                                                           'The original seven-distribution RECORD verifier remains required after owned admission; no RECORD/native/script payload was read by this task.']),
                    runtime_plan=dict(**reference(RUNTIME_PLAN), distributions=[dict(name=d['name'],version=d['version']) for d in plan['runtime']['distributions']],
                                      native_bytes_verified=False, source_map_role='Historical JSON runtime input, deliberately not a .py execution-map entry'),
                    pure_verifier_checks=checks, runtime_review_issue=runtime_issue,
                    current_history_or_catalog_read=False, actual_history_resolved=False,
                    actual_framework_imports=False, actual_process_service_lock_operations=False,
                    model_or_tensor_payload_bytes_read=0, native_binary_bytes_read=0,
                    frozen_sources_changed=False, runtime_admitted=False)
    analysis_bytes = encode(analysis)
    candidate = dict(schema='lfm3_full_cpu_execution_source_candidate_v2', source_only=True,
                     author='/root/lfm_after_writeback_design_20261111',
                     candidate_only=True, execution_entrypoint=str(RUNNER),
                     previous_spec=spec_ref, previous590_candidate=reference(PRIOR), locator_terminal=reference(TERMINAL),
                     original558_closure=reference(ORIGINAL), frozen_integration_review=reference(ROOT_REVIEW),
                     builder=reference(Path(__file__).resolve()),
                     analysis=dict(path=str(ANALYSIS), sha256=hashlib.sha256(analysis_bytes).hexdigest()),
                     original_source_count=558, r5_source_count=570, previous_source_count=len(previous), source_count=len(pins),
                     additive_source_count=len(additions), local_graph_files=graph['files'], local_graph_bytes=graph['bytes_read'],
                     original558_retained=True, all_previous570_retained=True, all_previous590_retained=True, old_hashes_rebound=0,
                     all_python_source_hashes_verified=True, finite_dynamic_target_count=len(finite),
                     source_sha256=dict(sorted(pins.items())), additive_source_sha256=additions,
                     source_map_pretty_bytes=len(encode(pins)), original_source_cap=1024,
                     pure_verifier_checks=checks, runtime_review_issue=runtime_issue,
                     interpretation='Complete conservative local Python literal closure plus enumerated finite local dynamic targets, preserving all previous590 pins including original558, r5 full570 and existing12 external SDK/Torch pins. External runtime-dependent framework resolution remains unproved.',
                     new_source_map_is_not_spec_or_admission=True, current_runtime_binding_created=False,
                     full874_execution_not_started=True, native_import_compatibility_verified=False,
                     whole_resource_fit_verified=False, runtime_admitted=False, full_model_qualified=False, new_speedup_claim=False)
    require(candidate['source_map_pretty_bytes'] < 256*2**10, 'Source map fits unchanged per-file budget; root must also budget its future full spec')
    return candidate, analysis_bytes


if __name__ == '__main__':
    require(sys.argv[1:] in ([], ['--write']), 'Only optional --write for new research reports')
    candidate, analysis_bytes = build()
    if sys.argv[1:]:
        require(not ANALYSIS.exists() and not OUTPUT.exists(), 'Never replace an existing research report')
        with ANALYSIS.open('xb') as stream: stream.write(analysis_bytes)
        with OUTPUT.open('xb') as stream: stream.write(encode(candidate))
    print(json.dumps({key:candidate[key] for key in ('source_count','additive_source_count','local_graph_files',
                     'local_graph_bytes','source_map_pretty_bytes','finite_dynamic_target_count','pure_verifier_checks')},indent=2))
    if sys.argv[1:]: print(json.dumps({'candidate':reference(OUTPUT),'analysis':reference(ANALYSIS)},indent=2))
