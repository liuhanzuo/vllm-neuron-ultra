"""Assemble a reviewable residual specification from actual completed inputs.

No writes, native imports, source-binary reads, admission or automatic launch.
Literal local Python imports are discovered statically; dynamic imports and
external packages still rely on the original fixed source/environment inputs.
The root must review and save the returned concrete specification separately.
"""
import ast
import copy
import hashlib
from pathlib import Path
import stat

import dory_residual_serial_contract_v4 as contract
import dory_after_nemotron_r9_v2 as history

Q = Path(__file__).resolve().parent
ROOTS = (Q, Q.parent, Q/'dory-native-reference-source-20261107-v1')
ENTRIES = tuple(Q/name for name in (
    'prepare_dory_residual_spec_v3.py', 'dory_after_nemotron_r9_v2.py',
    'dory_residual_serial_contract_v4.py', 'dory_residual_worker_v4.py',
    'run_dory_residual_serial_v4.py', 'dory_residual_saved_complete_v2.py'))
require = contract.require


def merge(pins, additions):
    for path, digest in additions.items():
        require(isinstance(path, str) and Path(path).is_absolute()
                and isinstance(digest, str) and len(digest) == 64
                and set(digest) <= set('0123456789abcdef'), 'Absolute source path and SHA256')
        require(path not in pins or pins[path] == digest, 'Never rebind an original input pin: '+path)
        pins[path] = digest


def python_bytes(path):
    path = Path(path)
    before = path.stat()
    require(path.is_absolute() and path.suffix == '.py' and stat.S_ISREG(before.st_mode)
            and before.st_size <= 8*2**20, 'Bounded ordinary Python source')
    raw = path.read_bytes()
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(len(raw) == before.st_size and all(getattr(before, k) == getattr(after, k) for k in fields),
            'Stable local Python source')
    return raw


def local_imports(entries=ENTRIES, roots=ROOTS):
    roots = tuple(Path(x) for x in roots)
    pending, pins, sizes, edges, external = list(map(Path, entries)), {}, {}, {}, set()
    def resolve(name):
        for root in roots:
            base = root/name.replace('.', '/')
            for path in (base.with_suffix('.py'), base/'__init__.py'):
                if path.is_file():
                    return path
    def within(path):
        return any(path.resolve().is_relative_to(root.resolve()) for root in roots)
    while pending:
        path = pending.pop()
        name = str(path)
        if name in pins:
            continue
        require(within(path) and len(pins) < 2048, 'Bounded local source closure')
        raw = python_bytes(path)
        require(sum(sizes.values())+len(raw) <= 64*2**20, 'Bounded source-only parse budget')
        pins[name], sizes[name] = hashlib.sha256(raw).hexdigest(), len(raw)
        found = set()
        for node in ast.walk(ast.parse(raw, filename=name)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dep = resolve(alias.name)
                    if dep is None:
                        external.add(alias.name.split('.')[0])
                    else:
                        found.add(dep)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parent
                    for _ in range(node.level-1):
                        base = base.parent
                    require(within(base), 'Relative imports stay within local roots')
                    target = base/(node.module or '').replace('.', '/')
                    candidates = [target.with_suffix('.py'), target/'__init__.py'] if node.module else []
                    candidates += [target/(a.name+'.py') for a in node.names if a.name != '*']
                    found.update(x for x in candidates if x.is_file())
                else:
                    dep = resolve(node.module) if node.module else None
                    children = [resolve(node.module+'.'+a.name) for a in node.names
                                if node.module and a.name != '*']
                    if dep is not None:
                        found.add(dep)
                    found.update(x for x in children if x is not None)
                    if dep is None and not any(x is not None for x in children) and node.module:
                        external.add(node.module.split('.')[0])
        edges[name] = sorted(str(x) for x in found)
        pending.extend(sorted(found))
    return dict(source_sha256=pins, bytes_read=sum(sizes.values()), files=len(pins), edges=edges,
        external_import_roots=sorted(external), dynamic_import_resolution_verified=False,
        runtime_import_identity_verified=False, native_executed=False)


def assemble(mode, *, r9_actual_exit=None, r9_additional_actual_exits=(), r9_recovery_operations=None,
             metadata_exit=None, producer_spec=None, native_exit=None, extra_metadata=()):
    """Return a concrete candidate only after prerequisite receipts validate."""
    require(mode in contract.MODES, 'Explicit metadata/native/review preparation')
    require((mode == 'metadata' and r9_actual_exit is not None and metadata_exit is None
             and producer_spec is None and native_exit is None)
            or (mode == 'native' and metadata_exit is not None and producer_spec is None and native_exit is None)
            or (mode == 'review' and metadata_exit is not None and producer_spec is not None and native_exit is not None),
            'Actual predecessor inputs for the selected mode')
    pins, references = {}, []
    def read(ref):
        require(ref is not None, 'Actual completed metadata reference required')
        value = contract.load(ref)
        merge(pins, {ref['path']: ref['sha256']})
        references.append(copy.deepcopy(ref))
        return value
    old, proposal = read(contract.ORIGINAL), read(contract.PROPOSAL)
    read(contract.PLAN)
    r9 = read(history.SPEC)
    read(history.REVIEW)
    merge(pins, old['source_sha256'])
    if mode == 'metadata':
        # Missing/live r9 cannot yield a candidate. No invented terminal data.
        read(r9_actual_exit)
        for ref in r9_additional_actual_exits:
            read(ref)
        read(r9_recovery_operations)
        h = history.reconstruct(r9_actual_exit, r9_additional_actual_exits, r9_recovery_operations)
    else:
        receipt = read(metadata_exit)
        prep = read(receipt['spec'])
        from dory_residual_serial_contract_v2 import validate as validate_metadata_v2
        validate_metadata_v2(prep, 'metadata')
        merge(pins, prep['source_sha256'])
        for key in ('admission', 'result', 'source_files'):
            read(receipt[key])
        h = read(receipt['predecessor_history'])
        r9_actual_exit = prep['r9_actual_exit']
        r9_additional_actual_exits = prep['r9_additional_actual_exits']
        r9_recovery_operations = prep['r9_recovery_operations']
    require(h['schema'] == 'dory_post_nemotron_r9_history_v1' and len(h['rows']) == h['complete_count'] == 1149
            and len(h['r9_stages']) == 53, 'Complete actual post-r9 history')
    merge(pins, h['predecessor_source_sha256'])
    if mode == 'review':
        producer = read(producer_spec)
        contract.validate(producer, 'native')
        merge(pins, producer['source_sha256'])
        read(native_exit)
        for name in ('admission.json', 'child.json', 'child-owner.json', 'child-exit.json',
                     'result.json', 'predecessor-history.json'):
            read(history.terminal.ref(Path(producer['output'])/name))
    if mode != 'metadata':
        failed = read(contract.FAILED_NATIVE)
        failed_spec = read(failed['spec'])
        merge(pins, failed_spec['source_sha256'])
        for key in ('admission', 'child', 'child_owner_record', 'child_exit', 'failure'):
            read(failed[key])
        merge(pins, {failed['worker_log']['path']: failed['worker_log']['sha256']})
        for ref in failed['diagnostic_files'].values():
            read(ref)
    for ref in extra_metadata:
        read(ref)
    graph = local_imports()
    merge(pins, {name: r9['source_sha256'][name] for name in graph['source_sha256']
                 if name in r9['source_sha256']})
    merge(pins, graph['source_sha256'])
    spec = copy.deepcopy(old)
    budget = (contract.METADATA_BUDGET if mode == 'metadata' else
              proposal['proposed_native_budget' if mode == 'native' else 'proposed_saved_review_budget'])
    spec.update(schema='dory_residual_complete_replay_diagnostic_v1', execution_mode=mode,
        diagnostic_only=True, runtime_admitted=True, review_only_spec=mode == 'review',
        selection_plan=contract.PLAN, resource_proposal=contract.PROPOSAL,
        metadata_output=str(Path(old['base'])/'residual-metadata-r1'),
        output=str(Path(old['base'])/'residual-replay-r2'),
        review_output=str(Path(old['base'])/'residual-replay-review-r2'),
        worker=str(Q/'dory_residual_worker_v4.py'), executor=str(Q/'run_dory_residual_serial_v4.py'),
        reviewer=str(Q/'dory_residual_saved_complete_v2.py'), budget=copy.deepcopy(budget),
        expected_original_raw_bytes=contract.RAW_BYTES,
        payload_review_budget=copy.deepcopy(proposal['proposed_payload_review_budget']),
        r9_actual_exit=r9_actual_exit, r9_additional_actual_exits=list(r9_additional_actual_exits),
        r9_recovery_operations=r9_recovery_operations,
        source_sha256=pins, essential_sources=sorted(pins),
        full_model_qualified=False, new_speedup_claim=False, automatic_launch=False)
    if mode != 'metadata':
        spec['metadata_preparation_exit'] = metadata_exit
        spec['failed_native_exit'] = contract.FAILED_NATIVE
    if mode == 'review':
        spec.update(producer_spec=producer_spec, native_exit=native_exit)
    sizes = {name: Path(name).stat().st_size for name in pins}
    spec['source_read_cap_bytes'] = sum(sizes.values())+4*2**20
    contract.validate(spec, mode)
    if mode != 'metadata':
        contract.metadata_history(spec, history.p.low.gone)
    return spec, dict(schema='dory_residual_spec_preparation_v1', mode=mode,
        actual_input_references=references, local_import_graph=graph,
        actual_predecessor_services=len(h['rows']), actual_predecessor_processes=len(h['processes']),
        r9_outcome=h['r9_outcome'],
        source_file_sizes=sizes, source_bytes_planned=sum(sizes.values()),
        source_binary_bytes_authenticated_here=0, source_binary_authentication_pending=True,
        candidate_specification_only=True, source_review_required=True,
        no_specification_written=True, serial_lock_acquired=False, native_started=False,
        automatic_launch=False, full_model_qualified=False, new_speedup_claim=False)


if __name__ == '__main__':
    raise RuntimeError('Call assemble with real completed receipts; root must review/save separately, never auto-launch')
