"""Inventory exact installed helper dependencies without importing them."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import nemotron4b_helper_dependency_catalog_v1 as c
from analyze_nemotron4b_helper_imports_v1 import read

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT/'research'
SITE = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
PLAN = Q/'nemotron4b-native-prerequisite-plan-20261106-v2.json'
PLAN_SHA = '7e5f16c07ad07c56b5bd3e6f5122ac6dd13b5e09d387da18bf116e035d81a2fd'
OUT = Q/'nemotron4b-helper-external-dependency-inventory-20261109-v1.json'
DISTRIBUTIONS = {
    'libneuronxla-2.2.15515.0+50c26cbd': ['libneuronxla'],
    'boto3-1.42.58': ['boto3'], 'botocore-1.42.58': ['botocore'],
    's3transfer-0.16.0': ['s3transfer'], 'jmespath-1.1.0': ['jmespath'],
    'python_dateutil-2.9.0.post0': ['dateutil'], 'urllib3-2.6.3': ['urllib3'],
    'protobuf-7.34.0': ['google/protobuf', 'google/_upb'], 'six-1.17.0': ['six.py'],
}


def main():
    c.require(sys.dont_write_bytecode and __debug__, 'Source-only Python -B without optimization')
    raw, plan_ref = c.source(PLAN, PLAN_SHA); plan = json.loads(raw)
    previous = plan['source_sha256']
    prior = {}
    for path, digest in previous.items():
        _, prior[path] = read(path, digest)
    modules, sources, namespaces, natives, distributions = {}, {}, {}, {}, {}
    for name, roots in DISTRIBUTIONS.items():
        folder = SITE/(name+'.dist-info')
        record_raw, record_ref = c.source(folder/'RECORD', previous.get(str(folder/'RECORD')))
        records = c.parse_record(record_raw)
        metadata_raw, metadata_ref = c.verified_source(folder/'METADATA', SITE, records,
                                                      previous.get(str(folder/'METADATA')))
        selected = []
        for root in roots:
            path = SITE/root
            c.require(path.exists(), 'Exact expected distribution source root')
            paths = [path] if path.is_file() else sorted(path.rglob('*.py'))
            for path in paths:
                raw, ref = c.verified_source(path, SITE, records, previous.get(str(path)))
                relative = path.relative_to(SITE).as_posix()
                module, is_package = c.module_name(relative)
                c.require(module not in modules, 'Disjoint installed source modules')
                modules[module] = dict(source=raw.decode('utf-8'), is_package=is_package)
                sources[module] = dict(**ref, distribution=name, already_in_original_pins=str(path) in previous)
                selected.append(module)
        # Native payloads are only stat'ed; RECORD digest is a declared value.
        for relative in records:
            if relative.endswith('.so') and any(relative == r or relative.startswith(r+'/') for r in roots):
                path = SITE/relative
                declared_digest, declared_size = c.record_digest(records, relative)
                fullstat = c.identity(path)
                c.require(fullstat['st_size'] == declared_size, 'Native RECORD size agrees with stat')
                stem = path.name.split('.', 1)[0]
                module = '.'.join((*Path(relative).parts[:-1], stem))
                natives[module] = dict(path=str(path), fullstat=fullstat, distribution=name,
                    installed_RECORD_declared_sha256=declared_digest, payload_digest_recomputed=False)
        distributions[name] = dict(RECORD=record_ref, METADATA=metadata_ref,
            metadata_requirements=[line for line in metadata_raw.decode().splitlines() if line.startswith('Requires-Dist:')],
            selected_python_modules=selected, all_selected_python_sources_RECORD_verified=True)
    for module in set(modules) | set(natives):
        parts = module.split('.')
        for n in range(1, len(parts)):
            parent = '.'.join(parts[:n])
            if parent in modules or parent in namespaces: continue
            directory = SITE.joinpath(*parts[:n])
            c.require(directory.is_dir() and directory.resolve() == directory
                      and not any(p.is_symlink() for p in (directory, *directory.parents))
                      and not (directory/'__init__.py').exists(), 'Actual ordinary namespace parent')
            namespaces[parent] = dict(path=str(directory), init_py_absent=True)
    helper = Path(plan['import_helper']['argv'][0])
    # The installed extensionless console script is authenticated by its
    # original source pin using the existing bounded source-only reader.
    entry, entry_ref = read(helper, plan['import_helper']['helper_sha256'])
    graph = c.dependency_graph(entry.decode(), modules, namespaces, natives)
    startup = {}
    pth_paths = sorted(SITE.glob('*.pth'))
    c.require(len(pth_paths) == 2 and all(str(p) in previous for p in pth_paths), 'Original exact two startup .pth files')
    for path in pth_paths:
        raw, ref = c.source(path, previous[str(path)])
        startup[str(path)] = dict(**ref, executable_lines=[line for line in raw.decode().splitlines()
            if line.startswith(('import ', 'import\t'))])
    # Its startup call is not a package import in the helper graph. Bind these
    # additional Python sources independently without executing add_shim().
    setuptools = SITE/'setuptools-82.0.0.dist-info'
    raw, startup_record_ref = c.source(setuptools/'RECORD')
    startup_records = c.parse_record(raw)
    startup_sources = {}
    for path in sorted((SITE/'_distutils_hack').glob('*.py')):
        _, ref = c.verified_source(path, SITE, startup_records, previous.get(str(path)))
        startup_sources[str(path)] = ref
    finder = SITE/'__editable___vllm_neuron_0_2_2_lts_finder.py'
    _, startup_sources[str(finder)] = c.source(finder, previous[str(finder)])
    boundary_roots = sorted({row['target'].split('.')[0] for row in graph['unresolved_import_boundaries']})
    boundary_presence = {}
    for name in boundary_roots:
        # Directory/file presence only: importing find_spec can execute parent packages.
        candidates = [SITE/name, SITE/(name+'.py')]
        boundary_presence[name] = dict(installed_candidates=[str(p) for p in candidates if p.exists()],
            system_interpreter_stdlib_name=name in sys.stdlib_module_names,
            resolved_runtime_origin=False)
    result = dict(schema='nemotron4b_helper_external_dependency_inventory_v1', reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), original_plan=plan_ref,
        original119pins_revalidated=len(prior), original_entrypoint=entry_ref,
        distributions=distributions, python_sources=sources, namespace_packages=namespaces,
        native_payload_stat_only=natives, graph=graph, unresolved_boundary_presence=boundary_presence,
        startup_pth=startup, startup_extra_sources=startup_sources, startup_RECORD=startup_record_ref,
        python_source_count=len(sources), python_source_bytes=sum(r['bytes'] for r in sources.values()),
        nonstdlib_known_remaining_boundaries=[n for n in boundary_roots if n not in sys.stdlib_module_names],
        static_only=True, actual_SDK_helper_executed=False, installed_package_code_executed=False,
        native_payload_bytes_read=0, native_or_model_executed=False, runtime_admitted=False,
        full_model_qualified=False, new_speedup_claim=False, complete_runtime_dependency_closure_proven=False,
        bootstrap_design=dict(use_no_site_before_parent_gate=True, proposed_interpreter_flags=['-B', '-S'],
            original_startup_must_be_authenticated_then_restored_after_gate=True,
            original_no_site_flag_semantics_not_yet_runtime_qualified=True),
        limitations=[
            'Installed RECORD agreement is local provenance, not independent publisher authenticity.',
            'The graph follows conservative module/class-body imports across selected packages, not function execution or a real trace.',
            'Native digests are declared in RECORD only; payloads were not read or loaded.',
            'Python startup .pth files execute before a normal script; a future bootstrap must wait before restoring site initialization.',
            'Stdlib/interpreter sources, native dependencies, dynamic loaders and remaining external boundaries still require a reviewed runtime context.',
        ], source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__).resolve(), Q/'nemotron4b_helper_dependency_catalog_v1.py',
            Q/'nemotron4b_helper_import_graph_v1.py', Q/'analyze_nemotron4b_helper_imports_v1.py')})
    data = (json.dumps(result, indent=2, allow_nan=False)+'\n').encode()
    c.require(len(data) <= 8*2**20, 'Bounded dependency inventory')
    with OUT.open('xb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(result=dict(path=str(OUT), sha256=hashlib.sha256(data).hexdigest()),
        distributions=len(distributions), python_sources=len(sources), source_bytes=result['python_source_bytes'],
        reachable_python_modules=len(graph['reachable_python_modules']), native_boundaries=graph['reachable_native_boundaries'],
        remaining_nonstdlib_boundaries=result['nonstdlib_known_remaining_boundaries'], startup_pth_count=len(startup),
        native_payload_bytes_read=0, actual_SDK_helper_executed=False)))


if __name__ == '__main__': main()
