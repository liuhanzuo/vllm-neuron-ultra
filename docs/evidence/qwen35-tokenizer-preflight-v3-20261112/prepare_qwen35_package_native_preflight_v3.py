# SPDX-License-Identifier: Apache-2.0
"""Source-only preparation; writes an inventory bundle, never a runtime input."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107')
WORKER = ROOT / 'qwen35_package_native_preflight_v3.py'
CHECKER = ROOT / 'check_qwen35_package_native_preflight_v3.py'


def literal_assignments(path):
    tree = ast.parse(path.read_text())
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try: result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError): pass
    return result


def reference(path):
    if path.suffix not in ('.py', '.json') or path.stat().st_size > 4 * 2**20:
        raise ValueError('Source/metadata only')
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def prepare(destination):
    constants = literal_assignments(WORKER)
    provenance = literal_assignments(REPO / 'vllm_neuron/models/qwen35/provenance.py')
    delivery = REPO / 'test/qwen35/source_delivery_v2.json'
    if reference(delivery)['sha256'] != constants['SOURCE_DELIVERY_SHA256']:
        raise ValueError('Frozen package v2 snapshot changed')
    for relative, digest in json.loads(delivery.read_text())['files'].items():
        if reference(REPO / relative)['sha256'] != digest:
            raise ValueError('Frozen package v2 file changed: ' + relative)
    official = ROOT / 'Qwen__Qwen3.5-2B.config.json'
    if json.loads(official.read_text()) != provenance['ORIGINAL_CONFIG']:
        raise ValueError('Pinned original config fixture differs')
    view = provenance['EXPECTED_MODEL']
    if hashlib.sha256(json.dumps(view, indent=2).encode()).hexdigest() != provenance['CONFIG_VIEW_SHA256']:
        raise ValueError('Exact text-view hash differs')
    # Extract the literal kwargs call as pure AST data, without running worker code.
    node = next(n.value for n in ast.parse(WORKER.read_text()).body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'NEURON_KWARGS')
    kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    bundle = dict(schema=constants['SCHEMA'] + '-source-bundle', source_only=True, native_executed=False,
        runtime_input_created=False, source_commit=constants['SOURCE_COMMIT'], checkout=str(REPO),
        python=constants['PYTHON'], site_packages=constants['SITE_PACKAGES'],
        source_delivery=reference(delivery),
        package_sources={str(p.relative_to(REPO)): reference(p)['sha256'] for p in sorted((REPO/'vllm_neuron').rglob('*.py'))},
        worker_sources=[reference(p) for p in (WORKER, CHECKER, Path(__file__).resolve(), ROOT / 'qwen35_package_tokenizer_checks_v3.py')],
        software=provenance['SOFTWARE'], sdk_sources=provenance['SDK_EXPECTED_SOURCE_SHA256'],
        official_config=reference(official), view_config=view,
        view_config_sha256=provenance['CONFIG_VIEW_SHA256'], neuron_kwargs=kwargs)
    # Only pinned source/metadata is read in preparation; tokenizer payloads are deferred.
    experiment = ROOT.parent
    protocol_files = dict(case='qwen35-state-reset-full-v1/case.json', quality_suite='quality-suite.json',
        quality_source='evaluate_quality.py', workload_source='benchmark_workloads.py',
        orchestrate_source='orchestrate.py', driver_source='research/qwen35_sampler_pair_driver_v3.py',
        hf_reference='qwen35-2b-hf-reference.json', long_reference='qwen35-baseline/long-decode-hf-reference.json',
        long_source='validate_qwen35_long_decode.py', historical_quality='qwen35-state-reset-full-v1/quality96.json',
        historical_workloads='qwen35-state-reset-full-v1/workloads.json')
    names = {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt', 'chat_template.jinja'}
    bundle['tokenizer'] = dict(files={name: provenance['SOURCE_FILES'][name] for name in sorted(names)},
        sources={name: reference(experiment / relative) for name, relative in protocol_files.items()},
        runtime_payload_read_required=True, tokenizer_payload_read_during_preparation=False)
    with Path(destination).open('x') as stream:
        json.dump(bundle, stream, indent=2, sort_keys=True); stream.write('\n')
    return reference(Path(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(prepare(Path(args.destination).absolute()), indent=2))


if __name__ == '__main__':
    main()
