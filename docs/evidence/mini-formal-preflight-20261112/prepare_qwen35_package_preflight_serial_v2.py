"""Write only a fresh source inventory; never choose current runtime bindings."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

Q = Path(__file__).resolve().parent
ROOT = Q.parent
NEW = ('qwen35_package_preflight_serial_v2.py', 'qwen35_package_preflight_history_v2.py',
       'check_qwen35_package_preflight_serial_v2.py', 'prepare_qwen35_package_preflight_serial_v2.py')
REUSED = ('qwen35_sampler_runtime_contract_v12.py', 'qwen35_sampler_runtime_io_v12.py',
    'qwen35_sampler_runtime_service_v12.py', 'qwen35_sampler_runtime_history_v12.py',
    'qwen35_sampler_history_supplement_v1.py', 'descriptor_serial_lock_proof_v1.py',
    'minicpm5_layer1_operator_recovered_completion_v2.py', 'minicpm5_layer1_operator_native_overlay_v3.py',
    'minicpm5_layer1_operator_sdk_sources_v1.py', 'prepare_minicpm5_layer1_operator_native_v7.py',
    'minicpm5_native_current_physical_v1.py', 'minicpm5_layer1_drain_hold_v1.py')
WORKER_BUNDLE = Q / 'qwen35-package-native-preflight-source-bundle-20261112-v3.json'
WORKER_SHA = '6c45ed5293eca2cd7ee48488ebd64f79e26c12136b2aec8b46af40c28238a080'


def reference(path):
    path = Path(path)
    if path.suffix not in ('.py', '.json') or path.stat().st_size > 4*2**20:
        raise ValueError('Bounded source/metadata only')
    payload = path.read_bytes()
    return dict(path=str(path), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def static_local_sources():
    """AST imports only; no import execution or model/native payload discovery."""
    pending = [Q / name for name in (*NEW, *REUSED)]
    found = {}
    while pending:
        path = pending.pop()
        if str(path) in found: continue
        if path.resolve() != path or not path.is_relative_to(ROOT): raise ValueError('Ordinary local source path')
        found[str(path)] = reference(path)['sha256']
        if len(found) > 2048: raise ValueError('Finite source closure')
        tree = ast.parse(path.read_text())
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): modules.update(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module: modules.add(node.module)
                modules.update((node.module + '.' if node.module else '') + item.name for item in node.names if item.name != '*')
        for name in modules:
            relative = Path(*name.split('.'))
            for base in (ROOT, Q):
                for candidate in (base / relative.with_suffix('.py'), base / relative / '__init__.py'):
                    if candidate.is_file() and candidate.suffix == '.py': pending.append(candidate)
    return dict(sorted(found.items()))


def prepare(destination):
    frozen = reference(WORKER_BUNDLE)
    if frozen['sha256'] != WORKER_SHA: raise ValueError('Immutable native v3 bundle changed')
    worker = json.loads(WORKER_BUNDLE.read_text())
    for row in worker['worker_sources']:
        if reference(row['path']) != row: raise ValueError('Immutable native v3 source changed')
    value = dict(schema='qwen35_package_preflight_serial_source_delivery_v2', source_only=True,
        native_executed=False, tokenizer_or_model_payload_read=False, runtime_input_created=False,
        current_history_selected=False, launch_authorized_by_this_inventory=False,
        new_sources=[reference(Q / name) for name in NEW], direct_reused_sources=[reference(Q / name) for name in REUSED],
        static_local_source_sha256=static_local_sources(), native_worker_bundle=frozen,
        scope='One existing-lifecycle native preflight stage; root must retain every then-current historical source and metadata binding')
    with Path(destination).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n')
    return reference(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('destination')
    args = parser.parse_args(); print(json.dumps(prepare(Path(args.destination).absolute()), indent=2))


if __name__ == '__main__': main()
