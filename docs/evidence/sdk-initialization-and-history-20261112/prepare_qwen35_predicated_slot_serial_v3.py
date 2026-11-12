"""Source-only additive inventory; never bind a catalog or authorize execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import qwen35_predicated_slot_serial_contract_v3 as contract
from prepare_qwen35_package_preflight_serial_v2 import reference

NEW = ('qwen35_predicated_slot_serial_contract_v3.py', 'qwen35_predicated_slot_serial_v3.py',
    'prepare_qwen35_predicated_slot_serial_v3.py', 'check_qwen35_predicated_slot_serial_v3.py',
    'qwen35_predicated_slot_sdk_environment_v1.py')


def static_local_sources():
    """Original AST import-closure method, including dynamic known leaf/helpers."""
    pending = [Q/name for name in NEW]
    pending += [Q.parent/'qwen35_predicated_slot_device_leaf_v1.py', Q/'qwen35_package_preflight_history_v2.py',
        Q.parent/'minicpm5_layer0_execution_protocol_v4.py',Q.parent/'test_qwen35_predicated_slot_source_v1.py']
    found = {}
    while pending:
        path = pending.pop()
        if str(path) in found: continue
        if path.resolve() != path or not path.is_relative_to(Q.parent): raise ValueError('Ordinary local source')
        found[str(path)] = reference(path)['sha256']
        if len(found) > 2048: raise ValueError('Bounded static source closure')
        modules = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import): modules.update(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module: modules.add(node.module)
                modules.update((node.module+'.' if node.module else '')+item.name for item in node.names if item.name != '*')
        for name in modules:
            relative = Path(*name.split('.'))
            for base in (Q.parent,Q):
                for candidate in (base/relative.with_suffix('.py'),base/relative/'__init__.py'):
                    if candidate.is_file(): pending.append(candidate)
    return dict(sorted(found.items()))


def prepare(destination):
    manifest = reference(contract.MANIFEST['path'])
    if manifest != contract.MANIFEST: raise ValueError('Frozen primitive manifest changed')
    source = json.loads(Path(manifest['path']).read_text())
    for row in source['source_files']:
        if reference(row['path']) != row: raise ValueError('Frozen primitive source changed')
    sdk = {}
    for name in contract.SDK_ROOTS:
        for path in (contract.SITE/name).rglob('*.py'):
            if path.resolve() != path: raise ValueError('Ordinary SDK text source')
            sdk[str(path)] = reference(path)['sha256']
    value = dict(schema='qwen35_predicated_slot_serial_source_delivery_v3', source_only=True,
        native_executed=False, compiler_executed=False, lock_acquired=False, services_launched=False,
        current_history_selected=False, runtime_input_created=False, launch_authorized_by_this_inventory=False,
        primitive_manifest=manifest, new_sources=[reference(Q/name) for name in NEW],
        static_local_source_sha256=static_local_sources(), sdk_source_sha256=dict(sorted(sdk.items())),
        workload=contract.protocol.workload(), limits=contract.LIMITS,
        scope='Reuse accepted package serial-v2 History and original RetainedLock/Service/ACK/drain; root supplies complete then-current pins and metadata later')
    with Path(destination).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True); stream.write('\n')
    return reference(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('destination')
    print(json.dumps(prepare(Path(parser.parse_args().destination).absolute()),indent=2))


if __name__ == '__main__': main()
