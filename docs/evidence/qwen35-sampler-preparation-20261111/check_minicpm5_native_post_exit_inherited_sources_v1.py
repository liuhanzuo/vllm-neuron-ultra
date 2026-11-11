"""Source-only compatibility tests using real r3 overlay/catalog metadata.

Only extracted pure functions execute. No producer, runtime module import,
process/lock observation, package source, tensor, native library or payload IO.
The real overlay source extension is an inherited-map fixture; this test does
not reconstruct or qualify the complete historical 44,503-pin scientific plan.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

Q = Path(__file__).resolve().parent
ROOT = Q.parent
checks = []


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pin_shape(value):
    require(type(value) is dict and len(value) <= 100000, 'Bounded complete inherited map')
    require(all(type(p) is str and Path(p).is_absolute() and type(s) is str
                and re.fullmatch('[a-f0-9]{64}', s) for p, s in value.items()), 'Exact inherited pins')


def tree(name):
    path = Q / name if (Q / name).is_file() else ROOT / name
    value = ast.parse(path.read_bytes(), filename=str(path))
    compile(value, str(path), 'exec')
    return value


def definition(value, name):
    return next(n for n in value.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def execute_pure(name, names, namespace):
    value = tree(name)
    nodes = [definition(value, function) for function in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(Q / name), 'exec'), namespace)
    return namespace


def reject(label, thunk, expected):
    try:
        thunk()
    except RuntimeError as error:
        assert expected in str(error), (label, str(error))
        checks.append(label)
    else:
        raise AssertionError('Unexpected acceptance: ' + label)


life = execute_pure('minicpm5_layer1_operator_native_lifecycle_core_v2.py', ['ref'],
                    dict(require=require, Path=Path, re=re))
r = SimpleNamespace(life=SimpleNamespace(ref=life['ref']), digest=digest,
                    sources=SimpleNamespace(metadata=SimpleNamespace(pin_shape=pin_shape)))
new_audit = str(Q / 'audit_minicpm5_layer1_operator_native_result_v4.py')
new_recorder = str(Q / 'record_minicpm5_layer1_operator_native_model_exit_v3.py')
core_path = str(Q / 'minicpm5_layer1_operator_native_post_exit_core_v5.py')
limits = dict(address_space_bytes=2 * 2**30, cpu_seconds=600, hard_cpu_seconds=605,
              output_file_bytes=2 * 2**20, wall_seconds=900, cpu=6, nice=19)
namespace = dict(require=require, Path=Path, re=re, copy=copy, ROOT=ROOT, Q=Q,
                 SELF=new_audit, core=SimpleNamespace(__file__=core_path), r=r, LIMITS=limits)
core_checks = execute_pure('minicpm5_layer1_operator_native_post_exit_core_v4.py', ['actual_tool'],
                          dict(require=require, re=re))
namespace['core'].actual_tool = core_checks['actual_tool']
audit = execute_pure('audit_minicpm5_layer1_operator_native_result_v4.py',
                     ['validate_spec', 'merged_plan', 'source_binding', 'validate_source_binding',
                      'validate_recorded_source_binding'], namespace)

overlay_path = Q / 'minicpm5-layer1-operator-native-r3-sourceclosure-20261111-overlay.json'
catalog_path = Q / 'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261111-v50.json'
overlay_bytes = overlay_path.read_bytes(); overlay = json.loads(overlay_bytes)
catalog_bytes = catalog_path.read_bytes(); catalog = json.loads(catalog_bytes)
assert hashlib.sha256(catalog_bytes).hexdigest() == '55db2de4ecd4232c3d7e232d5d34ef148a6df785b6a5a9bbf5fe3f4b3de5bc7a'
assert len(catalog['runs']) == 92 and len(catalog_bytes) == 2001785
inherited = copy.deepcopy(overlay['source_extension'])
assert len(inherited) == 3291
old_names = ('record_minicpm5_layer1_operator_native_model_exit_v2.py',
             'audit_minicpm5_layer1_operator_native_result_v3.py',
             'minicpm5_layer1_operator_native_post_exit_core_v5.py')
assert all(str(Q / n) in inherited for n in old_names)
assert all(catalog['runs'][91]['source_sha256'][str(Q / n)] == inherited[str(Q / n)] for n in old_names)
checks.append('actual r3 overlay and complete catalog already inherit all three old post-exit entry/core pins')

plan = dict(source_sha256=inherited, fixture='real overlay subset; full base intentionally not read')
before = copy.deepcopy(plan)
extension = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (new_audit, new_recorder)}
model_spec = dict(path=str(Q / 'minicpm5-layer1-operator-native-r3-sourceclosure-20261111-spec.json'),
                  sha256=hashlib.sha256((Q / 'minicpm5-layer1-operator-native-r3-sourceclosure-20261111-spec.json').read_bytes()).hexdigest())
actual_spec = json.loads(Path(model_spec['path']).read_text())
model_review_path = Q / 'minicpm5-layer1-operator-native-r3-sourceclosure-20261111-source-review.json'
model_review = dict(path=str(model_review_path), sha256=hashlib.sha256(model_review_path.read_bytes()).hexdigest())
model_overlay = actual_spec['overlay']
assert model_overlay == dict(path=str(overlay_path), sha256=hashlib.sha256(overlay_bytes).hexdigest())
args = (model_spec, model_review, model_overlay, new_recorder)
merged, proof = audit['source_binding'](plan, extension, *args)
assert plan == before and merged['source_sha256'] == dict(inherited, **extension)
assert proof['inherited_source_files'] == 3291 and proof['extension_source_files'] == 2
assert proof['complete_source_files'] == 3293 and len(proof['entry_sources']) == 4
assert audit['validate_source_binding'](proof, plan, extension, *args) == merged
checks.append('real overlay subset is retained exactly with only two new source pins and complete reference/digest binding')

duplicate = {str(Q / old_names[0]): inherited[str(Q / old_names[0])]}
reject('old required-entry extension conflicts even with an identical inherited hash',
       lambda: audit['merged_plan'](plan, duplicate), 'cannot replace or reclassify')
conflict = dict(duplicate); conflict[next(iter(conflict))] = '0' * 64
reject('changed inherited pin is never replaceable', lambda: audit['merged_plan'](plan, conflict), 'cannot replace or reclassify')
reject('copying all 3291 inherited overlay pins into additive extension fails',
       lambda: audit['source_binding'](plan, inherited, *args), 'only-new')
bad = dict(extension); bad['/tmp/unreviewed.py'] = '0' * 64
reject('unknown external source path rejected', lambda: audit['source_binding'](plan, bad, *args), 'new local text')
bad = dict(extension); bad[str(ROOT / 'native.so')] = '0' * 64
reject('native file cannot masquerade as new text', lambda: audit['source_binding'](plan, bad, *args), 'new local text')
bad = dict(extension); bad[str(ROOT / 'research' / '..' / 'escape.py')] = '0' * 64
reject('noncanonical new source rejected', lambda: audit['source_binding'](plan, bad, *args), 'new local text')
missing = copy.deepcopy(plan); del missing['source_sha256'][core_path]
reject('missing inherited core pin rejected', lambda: audit['source_binding'](missing, extension, *args), 'entry/core retained')
missing_new = dict(extension); del missing_new[new_audit]
reject('missing new auditor entry rejected', lambda: audit['source_binding'](plan, missing_new, *args), 'entry/core retained')
reject('unreviewed future producer rejected',
       lambda: audit['source_binding'](plan, extension, *args[:3], str(Q / 'record_future_v999.py')), 'current post-exit producer')
for key, value in [('inherited_source_files', 3290), ('complete_source_map_sha256', 'f' * 64),
                   ('complete_source_reconstruction_required', False), ('unknown', True)]:
    bad = copy.deepcopy(proof); bad[key] = value
    reject('source receipt mutation ' + key, lambda bad=bad: audit['validate_source_binding'](bad, plan, extension, *args), 'Exact complete inherited-source')
for key in ('model_spec', 'model_review', 'model_overlay'):
    bad = copy.deepcopy(proof); bad[key]['sha256'] = 'e' * 64
    reject('model reference rebound ' + key, lambda bad=bad: audit['validate_source_binding'](bad, plan, extension, *args), 'Exact complete inherited-source')
dropped = copy.deepcopy(plan); key = next(p for p in inherited if p not in proof['entry_sources']); del dropped['source_sha256'][key]
reject('dropping an unrelated inherited package/helper pin invalidates original receipt',
       lambda: audit['validate_source_binding'](proof, dropped, extension, *args), 'Exact complete inherited-source')
_, empty_proof = audit['source_binding'](merged, {}, *args)
assert empty_proof['extension_source_files'] == 0
checks.append('previously inherited authenticated entries need no duplicate extension on a later use')

exit_ref = dict(path=str(Q / 'synthetic-model-exit.json'), sha256='a' * 64)
rec_spec_ref = dict(path=str(Q / 'synthetic-recorder-spec.json'), sha256='b' * 64)
rec_review_ref = dict(path=str(Q / 'synthetic-recorder-review.json'), sha256='c' * 64)
live_ref = dict(path=str(Q / 'synthetic-live.json'), sha256='d' * 64)
tool = dict(kind='tool_session', session_id=8333, terminal_chunk='abcdef', exit_code=0)
ended_fixture = dict(recorder_spec=rec_spec_ref, recorder_review=rec_review_ref,
    recorder_source=dict(path=new_recorder, sha256=extension[new_recorder]),
    source_binding=proof, live=live_ref, spec=model_spec, review=model_review, actual_tool=tool)
rec_spec = dict(schema='minicpm5_native_model_exit_recorder_spec_v1', live=live_ref,
    model_spec=model_spec, model_review=model_review, actual_tool=tool,
    source_extension=extension, output=exit_ref['path'], limits=limits)
rec_review = dict(schema='minicpm5_native_model_exit_recorder_source_review_v1', reviewer='/root',
    passed=True, blockers=[], spec=rec_spec_ref, source_extension=extension)
consumer = audit['validate_recorded_source_binding']
consumer_args = (exit_ref, rec_spec, rec_review, plan, model_overlay, merged)
observed = consumer(ended_fixture, *consumer_args)
assert observed['verified'] is True and observed['complete_recorder_sources'] == 3293
checks.append('actual-next-consumer helper verifies recorder spec/review and complete inherited source receipt')
bad_ended = copy.deepcopy(ended_fixture); del bad_ended['source_binding']
reject('next consumer requires the recorder source receipt', lambda: consumer(bad_ended, *consumer_args), 'receipt required')
bad_ended = copy.deepcopy(ended_fixture); bad_ended['source_binding']['inherited_source_map_sha256'] = 'f' * 64
reject('next consumer rejects recorder receipt digest tampering', lambda: consumer(bad_ended, *consumer_args), 'Exact complete inherited-source')
bad_ended = copy.deepcopy(ended_fixture); bad_ended['recorder_source']['sha256'] = 'e' * 64
reject('next consumer rejects actual recorder entry hash tampering', lambda: consumer(bad_ended, *consumer_args), 'Entire actual recorder source map')
bad_spec = copy.deepcopy(rec_spec); bad_spec['source_extension'][new_recorder] = 'e' * 64
bad_review = copy.deepcopy(rec_review); bad_review['source_extension'] = bad_spec['source_extension']
reject('next consumer rejects a substituted but internally matching spec/review extension',
       lambda: consumer(ended_fixture, exit_ref, bad_spec, bad_review, plan, model_overlay, merged), 'Exact complete inherited-source')
bad_review = copy.deepcopy(rec_review); bad_review['passed'] = False
reject('next consumer rejects unaccepted recorder review',
       lambda: consumer(ended_fixture, exit_ref, rec_spec, bad_review, plan, model_overlay, merged), 'Actual recorder reviewed extension')
bad_review = copy.deepcopy(rec_review); bad_review['spec']['sha256'] = 'f' * 64
reject('next consumer rejects recorder-review spec rebound',
       lambda: consumer(ended_fixture, exit_ref, rec_spec, bad_review, plan, model_overlay, merged), 'Actual recorder reviewed extension')
bad_spec = copy.deepcopy(rec_spec); bad_spec['output'] = str(Q / 'another-model-exit.json')
reject('next consumer binds exact model terminal output',
       lambda: consumer(ended_fixture, exit_ref, bad_spec, rec_review, plan, model_overlay, merged), 'Actual recorder spec binds')
bad_spec = copy.deepcopy(rec_spec); bad_spec['actual_tool']['exit_code'] = False
reject('next consumer rejects boolean successful exit despite Python equality',
       lambda: consumer(ended_fixture, exit_ref, bad_spec, rec_review, plan, model_overlay, merged), 'successful tool terminal')
bad_audit_map = copy.deepcopy(merged); del bad_audit_map['source_sha256'][key]
reject('next consumer rejects dropped non-entry package pin from auditor closure',
       lambda: consumer(ended_fixture, exit_ref, rec_spec, rec_review, plan, model_overlay, bad_audit_map), 'Entire actual recorder source map')
bad_spec = copy.deepcopy(rec_spec); bad_spec['unknown'] = True
reject('next consumer rejects unknown recorder spec fields',
       lambda: consumer(ended_fixture, exit_ref, bad_spec, rec_review, plan, model_overlay, merged), 'Actual recorder spec binds')
runtime_audit = definition(tree('audit_minicpm5_layer1_operator_native_result_v4.py'), 'audit')
calls = [n for n in ast.walk(runtime_audit) if isinstance(n, ast.Call)]
consumers = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'validate_recorded_source_binding']
assert len(consumers) == 1 and ast.unparse(consumers[0].args[4]) == "resolved['plan']"
actual_reads = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'read'
                and isinstance(n.func.value, ast.Name) and n.func.value.id == 'metadata'
                and n.args and ast.unparse(n.args[0]) in ("ended['recorder_spec']", "ended['recorder_review']")]
assert len(actual_reads) == 2 and all(n.lineno < consumers[0].lineno for n in actual_reads)
raw_audit = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'verify_saved_driver')
assert consumers[0].lineno < raw_audit.lineno
checks.append('audit runtime consumes both actual recorder metadata references and validates receipt before raw audit')

spec_ref = dict(path=str(Q / 'synthetic-spec.json'), sha256='a' * 64)
review_ref = dict(path=str(Q / 'synthetic-review.json'), sha256='b' * 64)
spec = dict(schema='minicpm5_native_post_exit_audit_spec_v1', model_exit=spec_ref,
            recorder_exit=review_ref, current_serial_catalog=model_overlay, source_extension=extension,
            output=str(Q / 'minicpm5-layer1-operator-native-synthetic-post-exit-audit-synthetic.json'), limits=limits)
review = dict(schema='minicpm5_native_post_exit_audit_source_review_v1', reviewer='/root',
              passed=True, blockers=[], spec=spec_ref, source_extension=extension)
audit['validate_spec'](spec_ref, review_ref, spec, review)
checks.append('unchanged audit spec/review schema accepts only-new pins with inherited core')
bad = copy.deepcopy(spec); bad['unknown'] = True
reject('unknown audit spec field rejected', lambda: audit['validate_spec'](spec_ref, review_ref, bad, review), 'exact independent')

rec_namespace = dict(namespace, SELF=new_recorder)
recorder = execute_pure('record_minicpm5_layer1_operator_native_model_exit_v3.py',
                        ['actual_tool', 'validate_spec'], rec_namespace)
for exit_code in (0, 1, 137, -9):
    assert recorder['actual_tool'](dict(kind='tool_session', session_id=8333, terminal_chunk='abcdef', exit_code=exit_code))['exit_code'] == exit_code
checks.append('recorder retains true zero and nonzero tool exits unchanged')
reject('boolean exit rejected', lambda: recorder['actual_tool'](dict(kind='tool_session', session_id=8333, terminal_chunk='abcdef', exit_code=True)), 'true nonzero')

for old, new, names in [
    ('record_minicpm5_layer1_operator_native_model_exit_v2.py', 'record_minicpm5_layer1_operator_native_model_exit_v3.py',
     ['actual_tool', 'Guard', 'read_receipts', 'terminal_handoff', 'run']),
    ('audit_minicpm5_layer1_operator_native_result_v3.py', 'audit_minicpm5_layer1_operator_native_result_v4.py',
     ['merged_plan', 'pair', 'verify_saved_driver', 'AuditGuard', 'run'])]:
    a, b = tree(old), tree(new)
    assert all(ast.dump(definition(a, n), include_attributes=False) == ast.dump(definition(b, n), include_attributes=False) for n in names)
    checks.append('unchanged protected scientific/exit/lease/IO functions: ' + new)
for old, new, name in [('record_minicpm5_layer1_operator_native_model_exit_v2.py', 'record_minicpm5_layer1_operator_native_model_exit_v3.py', 'record'),
                       ('audit_minicpm5_layer1_operator_native_result_v3.py', 'audit_minicpm5_layer1_operator_native_result_v4.py', 'audit')]:
    a, b = definition(tree(old), name), definition(tree(new), name)
    def required_calls(fn):
        return [ast.dump(n, include_attributes=False) for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'require']
    assert required_calls(a) == required_calls(b)
    checks.append('all original runtime validation predicates retained: ' + name)
life_old, life_new = tree('minicpm5_layer1_operator_native_lifecycle_core_v2.py'), tree('minicpm5_layer1_operator_native_lifecycle_core_v3.py')
for name in ('build_contract', 'binding', 'command', 'process', 'coordinator_process',
             'check_driver_result', 'wrap_native_result', 'unwrap_native_result'):
    assert ast.dump(definition(life_old, name), include_attributes=False) == ast.dump(definition(life_new, name), include_attributes=False)
checks.append('old post-exit core accepts current lifecycle contracts and unchanged driver_v4 result schema')
outer_old, outer_new = tree('minicpm5_layer1_operator_native_outer_runner_v4.py'), tree('minicpm5_layer1_operator_native_outer_runner_v5.py')
def method(value, name):
    return next(n for n in definition(value, 'OwnedService').body if isinstance(n, ast.FunctionDef) and n.name == name)
assert ast.dump(method(outer_old, 'verify_drained'), include_attributes=False) == ast.dump(method(outer_new, 'verify_drained'), include_attributes=False)
checks.append('old outer post-exit drain verifier is identical to current outer_v5')

assert len(catalog_bytes) < 2 * 2**20 and 2 * 2**20 - len(catalog_bytes) == 95367
overlay_map_bytes = len(json.dumps(inherited, indent=2).encode())
assert len(catalog_bytes) + overlay_map_bytes > 2 * 2**20
assert len(catalog_bytes) + 2 * overlay_map_bytes < 8 * 2**20
checks.append('full future catalog growth crosses 2MiB output bound but remains within current 8MiB input transport')
assert len(catalog['runs']) == 92 and json.loads(catalog_bytes) == catalog
checks.append('all 92 existing catalog rows remain unchanged; no live catalog or runtime files written')

sources = {str(Q / n): hashlib.sha256((Q / n).read_bytes()).hexdigest() for n in
           ('record_minicpm5_layer1_operator_native_model_exit_v3.py',
            'audit_minicpm5_layer1_operator_native_result_v4.py', Path(__file__).name)}
print(json.dumps(dict(schema='minicpm5_native_post_exit_inherited_source_checks_v1', passed=True,
    check_count=len(checks), checks=checks, real_overlay_source_subset_entries=len(inherited),
    full_base_plan_reconstructed=False, complete_existing_catalog_rows=92,
    catalog_bytes=len(catalog_bytes), catalog_2mib_headroom_bytes=95367,
    one_pretty_overlay_source_map_bytes=overlay_map_bytes,
    source_binding_receipt_bytes=len(json.dumps(proof, indent=2).encode()), sources=sources,
    target_packages_imported=False, runtime_entrypoints_called=False,
    process_lock_service_inspection=False, raw_native_or_archive_payload_read=False,
    actual_exit_created=False, runtime_admitted=False, full_model_qualified=False,
    new_speedup_claim=False), indent=2))
