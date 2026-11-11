"""Pure synthetic selection/parser tests; never run either collector main."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

Q = Path(__file__).resolve().parent


def own_module(name, filename):
    path = Q / filename
    source = path.read_bytes()
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), 'exec')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, tree


before = set(sys.modules)
m, _ = own_module('dory_source_selector_synthetic_v2', 'dory_loader_route_selection_v2.py')
c, new_tree = own_module('dory_source_collector_synthetic_v2', 'collect_dory_loader_route_sources_v2.py')
old_tree = ast.parse((Q / 'collect_dory_loader_route_sources_v1.py').read_bytes())
checks = []


def row(name, size=16):
    identity = dict.fromkeys(m.IDENTITY_FIELDS, 0)
    identity.update(st_size=size, st_nlink=1)
    return dict(event='SITE_inventory',
                reference=dict(path=m.SITE + '/' + name, bytes=size, sha256='a' * 64),
                identity=identity,
                provenance=dict(kind='upstream_wheel_member', evidence=dict(
                    package=name.split('/')[0], member=name, archive_sha256='a' * 64)))


required = [row(name) for name in m.REQUIRED]
candidate_names = ['vllm/distributed/device_communicators/shm_broadcast.py',
                   'vllm/v1/kv_offload/example.py',
                   'vllm/example/file_system.py']
rows = required + [row(name) for name in candidate_names]


def select(values):
    return m.select_targets(iter(values), expected_events=len(values))


def reject(label, thunk, message):
    try:
        thunk()
    except RuntimeError as error:
        assert message in str(error), (label, str(error))
        checks.append(label)
    else:
        raise AssertionError('Unexpected acceptance: ' + label)


reached = []


def success_tail():
    yield from copy.deepcopy(rows)
    yield dict(event='finishing_complete')
    reached.append(True)


got = m.select_targets(success_tail(), expected_events=len(rows) + 1)
assert len(got) == 8 and reached == [True]
assert [x['reference']['path'] for x in got] == sorted(x['reference']['path'] for x in rows)
checks.append('authenticated EOF is exhausted and selection order is deterministic')


def failed_tail():
    yield from rows
    raise RuntimeError('synthetic EOF digest mismatch')


reject('failed digest after last selected member',
       lambda: m.select_targets(failed_tail(), expected_events=len(rows) + 1), 'EOF digest mismatch')
reject('missing required source', lambda: select(rows[1:]), 'All five')
reject('duplicate selected source', lambda: select(rows + [rows[0]]), 'Duplicate')
reject('short event stream', lambda: m.select_targets(iter(rows), expected_events=len(rows) + 1), 'Exact event count')
reject('long event stream', lambda: m.select_targets(iter(rows), expected_events=len(rows) - 1), 'Bounded complete')
reject('non-dictionary event', lambda: select(rows + [None]), 'Bounded complete')
reject('boolean event count', lambda: m.select_targets(iter(rows), expected_events=True), 'Finite exact')

for field, value in [('package', 'other'), ('member', 'torch/other.py'), ('archive_sha256', 'b' * 64)]:
    bad = copy.deepcopy(rows); bad[0]['provenance']['evidence'][field] = value
    reject('provenance mismatch ' + field, lambda bad=bad: select(bad), 'wheel member provenance')
bad = copy.deepcopy(rows); bad[0]['provenance']['kind'] = 'current_hash_only'
reject('self-reported hash is not prior authority', lambda: select(bad), 'wheel member provenance')
for field, value in [('bytes', True), ('bytes', -1), ('sha256', 'not-a-sha256')]:
    bad = copy.deepcopy(rows); bad[0]['reference'][field] = value
    reject('invalid reference ' + field + ' ' + str(value), lambda bad=bad: select(bad), 'bounded source reference')
for field, value in [('st_size', 17), ('st_nlink', 2), ('st_ino', True)]:
    bad = copy.deepcopy(rows); bad[0]['identity'][field] = value
    reject('identity mismatch ' + field, lambda bad=bad: select(bad), 'saved source identity')
bad = rows + [row('vllm/distributed/device_communicators/../../escape.py')]
reject('source path traversal', lambda: select(bad), 'Canonical selected source path')
bad = required + [row('vllm/v1/kv_offload/large.py', m.POLICY['per_source_bytes'] + 1)]
reject('single source limit', lambda: select(bad), 'bounded source reference')
bad = required + [row(f'vllm/v1/kv_offload/item{i}.py') for i in range(188)]
reject('source count limit before payload reads', lambda: select(bad), 'source count cap')
bad = required + [row(f'vllm/v1/kv_offload/big{i}.py', m.POLICY['per_source_bytes']) for i in range(16)]
reject('aggregate byte limit before payload reads', lambda: select(bad), 'source byte cap')

empty = copy.deepcopy(required); empty[3] = row(m.REQUIRED[3], 0)
assert len(select(empty)) == 5
checks.append('authenticated empty package init source is accepted')
ignored = [row('vllm/_C_AVX512.abi3.so'), row('vllm/model_executor/models/other.py'),
           row('torch/distributed/other.py')]
foreign = row(candidate_names[0]); foreign['reference']['path'] = '/another/environment/' + candidate_names[0]
assert len(select(rows + ignored + [foreign])) == len(rows)
checks.append('native files unrelated sources and other environments are never selected')
assert m.POLICY['whole_package_search'] is False and m.POLICY['missing_match_proves_absence'] is False
assert m.selection_reason(m.SITE + '/vllm/arbitrary/hidden_caller.py') is None
checks.append('a caller outside the finite candidate scopes can be missed and is not denied')

code = b'''raise RuntimeError("MUST NOT EXECUTE")
from vllm import fs_io_C
def deferred():
    import vllm.spinloop
    return importlib.import_module("vllm.fs_io_C")
'''
finding = m.inspect_source(candidate_names[0], code)
assert [x['line'] for x in finding['direct_import_statements']] == [2, 4]
assert finding['textual_mentions'][1]['enclosing_scopes'] == ['FunctionDef:deferred']
assert finding['actual_import_or_call_observed'] is False
checks.append('AST locates direct and deferred imports without executing target source')
finding = m.inspect_source(candidate_names[0], b'# fs_io_C\nx = "spinloop"\n')
assert not finding['direct_import_statements'] and len(finding['textual_mentions']) == 2
checks.append('comments and strings remain textual evidence not proven imports')
finding = m.inspect_source(candidate_names[0], b'# fs_io_C ' + b'x' * 3000)
assert finding['textual_mentions'][0]['text_truncated'] is True
assert len(finding['textual_mentions'][0]['text']) == 2048
checks.append('long finding text is explicitly marked truncated')
reject('source parser finding count limit',
       lambda: m.inspect_source(candidate_names[0], b'# spinloop\n' * 513), 'findings cap')


def find_function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


for name in ('require', 'validate_reference', 'write_source', 'guard', 'pinned', 'current_handoff', 'interrupted'):
    assert ast.dump(find_function(old_tree, name), include_attributes=False) == ast.dump(find_function(new_tree, name), include_attributes=False), name
checks.append('V1 guard pinned-read source-write and handoff helpers are structurally unchanged')
old_main = find_function(old_tree, 'main'); new_main = find_function(new_tree, 'main')
old_final = next(n.finalbody for n in old_main.body if isinstance(n, ast.Try) and n.finalbody)
new_final = next(n.finalbody for n in new_main.body if isinstance(n, ast.Try) and n.finalbody)
assert ast.dump(ast.Module(body=old_final, type_ignores=[]), include_attributes=False) == ast.dump(ast.Module(body=new_final, type_ignores=[]), include_attributes=False)
checks.append('V1 failure recording and lock-release finally path is structurally unchanged')

ref = dict(path='/synthetic/plan.json', sha256='a' * 64)
history = dict(path='/synthetic/history.json', sha256='b' * 64)
pins = dict(c.ENTRYPOINT_SOURCES)
pins[str(Q / 'collect_dory_loader_route_sources_v2.py')] = 'c' * 64
pins[str(Q / 'dory_loader_route_selection_v2.py')] = 'd' * 64
plan = dict(schema='dory_loader_route_source_plan_v2', runtime_bound=True,
            automatic_successor_launch=False, output_root='/mnt/nvme/dory-loader-route-sources-synthetic',
            budget=c.BUDGET, lock_policy={'synthetic': True}, source_sha256=pins,
            source_selection_policy=m.POLICY, current_serial_catalog=history,
            current_handoff=history, predecessor_exit=history, predecessor_expected_exit_code=0,
            predecessor_owners=[{'synthetic': True}], predecessor_services=[])
review = {key: plan[key] for key in ('source_sha256', 'source_selection_policy', 'lock_policy',
                                  'current_serial_catalog', 'current_handoff', 'predecessor_exit')}
review.update(schema='dory_loader_route_source_review_v2', reviewer='/root', passed=True,
              blockers=[], plan=ref, latest_handoff_accepted=True, current_physical_history_accepted=True)
assert c.validate_plan(plan, review, ref, plan['lock_policy'], m.POLICY).name.endswith('synthetic')
checks.append('new finite selection policy is explicitly bound in synthetic plan and review')
bad = copy.deepcopy(review); bad['source_selection_policy']['source_file_cap'] += 1
reject('review cannot expand source scope', lambda: c.validate_plan(plan, bad, ref, plan['lock_policy'], m.POLICY), 'finite source selection')
bad = copy.deepcopy(plan); bad['source_sha256'][next(iter(c.ENTRYPOINT_SOURCES))] = 'f' * 64
badreview = copy.deepcopy(review); badreview['source_sha256'] = bad['source_sha256']
reject('original entrypoint pins cannot change', lambda: c.validate_plan(bad, badreview, ref, bad['lock_policy'], m.POLICY), 'source pins retained')
assert not any(name.split('.')[0] in ('torch', 'vllm', 'numpy', 'transformers', 'tokenizers') for name in set(sys.modules) - before)
print(json.dumps(dict(schema='dory_loader_route_source_followup_synthetic_v2', passed=True,
    checks=checks, check_count=len(checks), collector_main_called=False,
    real_event_or_installed_source_or_native_payload_read=False,
    process_lock_service_inspection=False,
    sources={str(Q / name): hashlib.sha256((Q / name).read_bytes()).hexdigest()
             for name in ('collect_dory_loader_route_sources_v1.py', 'collect_dory_loader_route_sources_v2.py',
                          'dory_loader_route_selection_v2.py', 'check_dory_loader_route_sources_synthetic_v2.py')},
    runtime_admitted=False, loader_selection_established=False,
    native_ABI_qualified=False, full_model_qualified=False, new_speedup_claim=False), indent=2))
