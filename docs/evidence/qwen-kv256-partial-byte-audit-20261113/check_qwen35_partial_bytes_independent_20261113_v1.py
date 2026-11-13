"""Source-only AST probes; all data/process/lock state is synthetic."""
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import types

SOURCE = Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research/audit_qwen35_kv256_partial_bytes_root_20261113_v1.py')
source = SOURCE.read_bytes()
tree = ast.parse(source)
main = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'main')

def require(ok, msg):
    if not ok:
        raise ValueError(msg)

def extract(name, scope):
    node = next(x for x in ast.walk(main) if isinstance(x, ast.FunctionDef) and x.name == name)
    code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<source-only-extract>', 'exec')
    exec(code, scope)
    return scope[name]

results = {}
clock = [10.0]
expected = {'path':'/synthetic/idle','identity':{'inode':1}}
latest = [expected]
counters = dict(lock=0, latest=0, heavy=0)
def check(): counters['lock'] += 1
def latest_handoff():
    counters['latest'] += 1
    return latest[0]
def active_heavy(): counters['heavy'] += 1
scope = dict(require=require, lock=types.SimpleNamespace(check=check),
    time=types.SimpleNamespace(monotonic=lambda:clock[0]), started=0.0, last=[-1.0],
    identity=lambda p:expected['identity'], idle_ref={'path':'/synthetic/idle'},
    expected=expected, latest_handoff=latest_handoff,
    process_api=lambda:types.SimpleNamespace(active_heavy=active_heavy))
guard = extract('guard', scope)
guard()
latest[0] = {'path':'/synthetic/unknown-successor','identity':{'inode':2}}
clock[0] = 10.1
guard()
results['unknown_successor_inside_throttle_accepted'] = counters['latest'] == 1
clock[0] = 10.6
try:
    guard()
except ValueError as e:
    results['outside_throttle_negative_control_rejected'] = str(e) == 'Unknown successor during raw analysis'
else:
    raise AssertionError('outside-throttle control failed')

def identity(path):
    s = Path(path).lstat()
    return dict(realpath=str(path), device=s.st_dev, inode=s.st_ino, size=s.st_size,
        mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)

with tempfile.TemporaryDirectory(prefix='qwen35-partial-source-review-') as tmp:
    root = Path(tmp)
    (root/'inputs').mkdir()
    path = root/'inputs'/'oracle.raw'
    data = b'\x12\x34'*8
    path.write_bytes(data)
    reference = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    seen = {}
    capture = extract('capture', dict(require=require, guard=lambda:None, STATE_BYTES=8*2**20,
        Path=Path, root=root, identity=identity, os=os, stat=stat, hashlib=hashlib, seen=seen))
    assert capture(reference, 'inputs') == data
    first = seen[str(path)]['identity']
    replacement = root/'inputs'/'replacement.raw'
    replacement.write_bytes(data)
    os.replace(replacement, path)
    assert capture(reference, 'inputs') == data
    results['same_bytes_identity_rebinding_accepted'] = seen[str(path)]['identity'] != first
    results['final_seen_identity_check_accepts_rebinding'] = all(identity(p) == r['identity'] for p,r in seen.items())
    path.write_bytes(b'x'*len(data))
    try:
        capture(reference, 'inputs')
    except ValueError as e:
        results['changed_bytes_negative_control_rejected'] = str(e) == 'Full raw contents and identity unchanged'
    else:
        raise AssertionError('changed-byte control failed')

host_assert = next(x for x in ast.walk(main) if isinstance(x, ast.Call)
    and isinstance(x.func, ast.Name) and x.func.id == 'require'
    and len(x.args)>1 and isinstance(x.args[1], ast.Constant)
    and x.args[1].value == 'This prefix records unchanged host inputs')
expr = compile(ast.fix_missing_locations(ast.Expression(host_assert.args[0])), '<host-assert>', 'eval')
results['empty_host_checks_accepted'] = eval(expr, {'row':dict(call_returned=True, host_inputs_unchanged=[])})
results['nonboolean_host_checks_accepted'] = eval(expr, {'row':dict(call_returned=True, host_inputs_unchanged=[1,1])})
assert all(results.values()), results
print(json.dumps(dict(schema='qwen35_partial_bytes_independent_source_probe_v1',
    source=dict(path=str(SOURCE),sha256=hashlib.sha256(source).hexdigest()),
    real_history_lock_raw_report_exit_idle_reads=False, real_module_imports=False,
    findings_reproduced=results), indent=2))
