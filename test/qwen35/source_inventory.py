# SPDX-License-Identifier: Apache-2.0
"""AST-only direct import inventory and exact source-byte fingerprints."""
import ast
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / 'vllm_neuron/models/qwen35'


def inventory():
    files = {}
    for path in sorted(PACKAGE.glob('*.py')):
        source = path.read_bytes()
        imports = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imports += [{'line': node.lineno, 'module': name.name, 'relative': False, 'names': []} for name in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports.append({'line': node.lineno, 'module': '.' * node.level + (node.module or ''),
                                'relative': bool(node.level), 'names': [name.name for name in node.names]})
        files[str(path.relative_to(REPO))] = {'sha256': hashlib.sha256(source).hexdigest(), 'imports': sorted(imports, key=lambda x: (x['line'], x['module']))}
    existing = ['vllm_neuron/__init__.py', 'vllm_neuron/models/__init__.py',
        'vllm_neuron/worker/neuronx_distributed_model_loader.py',
        'vllm_neuron/worker/neuronx_distributed_model_runner.py', 'vllm_neuron/worker/cpu_sampler.py']
    provenance_tree = ast.parse((PACKAGE / 'provenance.py').read_text())
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in provenance_tree.body if isinstance(n, ast.Assign)}
    return {'scope': 'Exact direct Python import scan; SDK sources are pinned separately, not a dynamic native-library dependency scan.',
            'native_modules_imported_by_scan': [], 'package_sources': files,
            'external_direct_imports': sorted({row['module'] for info in files.values() for row in info['imports'] if not row['relative']}),
            'existing_repo_dependencies': {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in existing},
            'pinned_runtime_versions': constants['SOFTWARE'], 'pinned_sdk_source_sha256': constants['SDK_EXPECTED_SOURCE_SHA256'],
            'original_experiment_source_sha256': constants['EXPERIMENT_SOURCES']}


if __name__ == '__main__':
    result = inventory()
    output = Path(__file__).with_name('dependency_inventory.json')
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'file': str(output), 'package_python_files': len(result['package_sources']),
                      'external_direct_imports': result['external_direct_imports'], 'native_imports': []}, indent=2))
