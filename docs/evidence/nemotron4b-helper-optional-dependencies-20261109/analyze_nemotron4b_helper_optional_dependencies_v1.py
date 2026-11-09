"""Extend the pinned helper catalog with installed optional dependencies.

Read Python and package metadata only. Native files are stat-only and retain
the RECORD-declared digest; no installed code or native payload is executed.
This is an inventory, not a runtime import allowlist or complete closure.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import nemotron4b_helper_dependency_catalog_v1 as c
from analyze_nemotron4b_helper_dependencies_v2 import original_source
from analyze_nemotron4b_helper_imports_v1 import read

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT/'research'
SITE = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
BASE = Q/'nemotron4b-helper-external-dependency-inventory-20261109-v1.json'
BASE_SHA = '6c10f83abcb474ac1660114b250895d5bc07e38939307497e9cd1e4faf39cdb5'
OUT = Q/'nemotron4b-helper-optional-dependency-inventory-20261109-v1.json'
DISTRIBUTIONS = {
    'pyopenssl-25.3.0': ['OpenSSL'],
    'cryptography-46.0.5': ['cryptography'],
    'certifi-2026.2.25': ['certifi'],
    'typing_extensions-4.15.0': ['typing_extensions.py'],
    'cffi-2.0.0': ['cffi', '_cffi_backend.cpython-310-x86_64-linux-gnu.so'],
    'pycparser-3.0': ['pycparser'],
}


def selected(relative, roots):
    return any(relative == root or relative.startswith(root+'/') for root in roots)


def inventory_distribution(site, name, roots):
    """Authenticate every selected Python file and stat selected native leaves."""
    site = Path(site)
    c.require(roots and all(not Path(root).is_absolute() and '..' not in Path(root).parts
                           for root in roots), 'Explicit relative distribution roots')
    folder = site/(name+'.dist-info')
    record_raw, record_ref = c.source(folder/'RECORD')
    records = c.parse_record(record_raw)
    metadata_raw, metadata_ref = c.verified_source(folder/'METADATA', site, records)
    sources, modules, native = {}, {}, {}
    for root in roots:
        path = site/root
        c.require(path.exists(), 'Required installed dependency root')
        if path.is_file():
            c.require(path.suffix in ('.py', '.so'), 'Explicit Python or native top-level root')
            paths = [path] if path.suffix == '.py' else []
        else:
            paths = sorted(path.rglob('*.py'))
        for path in paths:
            raw, ref = c.verified_source(path, site, records)
            module, package = c.module_name(path.relative_to(site).as_posix())
            c.require(module not in modules, 'No duplicate selected Python module')
            sources[module] = dict(**ref, distribution=name)
            modules[module] = dict(source=raw.decode(), is_package=package)
    for relative in records:
        if relative.endswith('.so') and selected(relative, roots):
            digest, size = c.record_digest(records, relative)
            path = site/relative
            fullstat = c.identity(path)
            c.require(fullstat['st_size'] == size, 'Native RECORD size agrees with stat')
            module = '.'.join((*Path(relative).parts[:-1], path.name.split('.', 1)[0]))
            c.require(module not in native and module not in modules, 'Disjoint native module')
            native[module] = dict(path=str(path), fullstat=fullstat, distribution=name,
                installed_RECORD_declared_sha256=digest, payload_digest_recomputed=False)
    # A top-level native root must not disappear merely because RECORD omitted it.
    for root in roots:
        if root.endswith('.so'):
            c.require(any(row['path'] == str(site/root) for row in native.values()),
                      'Required native root has a complete RECORD row')
    distribution = dict(RECORD=record_ref, METADATA=metadata_ref,
        metadata_requirements=[line for line in metadata_raw.decode().splitlines()
                               if line.startswith('Requires-Dist:')],
        selected_python_modules=sorted(modules), selected_native_modules=sorted(native),
        all_selected_python_sources_RECORD_verified=True)
    return distribution, sources, modules, native


def namespaces_for(site, modules, natives):
    result = {}
    for module in set(modules) | set(natives):
        parts = module.split('.')
        for n in range(1, len(parts)):
            parent = '.'.join(parts[:n])
            if parent in modules or parent in natives or parent in result:
                continue
            directory = Path(site).joinpath(*parts[:n])
            c.require(directory.is_dir() and directory.resolve() == directory
                      and not any(p.is_symlink() for p in (directory, *directory.parents))
                      and not (directory/'__init__.py').exists(), 'Actual namespace parent')
            result[parent] = dict(path=str(directory), init_py_absent=True)
    return result


def main():
    c.require(sys.dont_write_bytecode and __debug__, 'Python -B without optimization')
    raw, base_ref = c.source(BASE, BASE_SHA)
    base = json.loads(raw)
    original_plan = base['original_plan']
    plan_raw, _ = c.source(Path(original_plan['path']), original_plan['sha256'])
    previous = json.loads(plan_raw)['source_sha256']
    for path, digest in previous.items():
        original_source(path, digest)
    sources = dict(base['python_sources'])
    modules = {}
    for name, row in sources.items():
        raw, current = c.source(row['path'], row['sha256'])
        c.require(current['fullstat'] == row['fullstat'], 'Original installed source identity unchanged')
        modules[name] = dict(source=raw.decode(), is_package=Path(row['path']).name == '__init__.py')
    for group in ('startup_pth', 'startup_extra_sources'):
        for row in base[group].values():
            _, current = c.source(row['path'], row['sha256'])
            c.require(current['fullstat'] == row['fullstat'], 'Original startup identity unchanged')
    distributions = dict(base['distributions'])
    for row in distributions.values():
        for key in ('RECORD', 'METADATA'):
            _, current = c.source(row[key]['path'], row[key]['sha256'])
            c.require(current['fullstat'] == row[key]['fullstat'], 'Original distribution metadata unchanged')
    natives = dict(base['native_payload_stat_only'])
    for row in natives.values():
        c.require(c.identity(row['path']) == row['fullstat'], 'Original native identity unchanged')
    added = []
    for name, roots in DISTRIBUTIONS.items():
        distribution, extra, descriptions, native = inventory_distribution(SITE, name, roots)
        c.require(name not in distributions and not (set(extra) & set(sources))
                  and not (set(native) & (set(natives) | set(sources)))
                  and not (set(extra) & set(natives)), 'Disjoint extended distribution origins')
        distributions[name] = distribution
        sources.update(extra)
        modules.update(descriptions)
        natives.update(native)
        added.extend(extra)
    namespaces = namespaces_for(SITE, modules, natives)
    entry, _ = read(Path(base['original_entrypoint']['path']), base['original_entrypoint']['sha256'])
    graph = c.dependency_graph(entry.decode(), modules, namespaces, natives)
    roots = sorted({row['target'].split('.')[0] for row in graph['unresolved_import_boundaries']})
    presence = {}
    for name in roots:
        presence[name] = dict(installed_candidates=[str(p) for p in (SITE/name, SITE/(name+'.py')) if p.exists()],
            system_interpreter_stdlib_name=name in sys.stdlib_module_names,
            resolved_runtime_origin=False)
    result = dict(schema='nemotron4b_helper_optional_dependency_inventory_v1', reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), base_inventory=base_ref,
        original119pins_revalidated=len(previous), original263sources_unchanged=True,
        original_startup_and_distribution_metadata_unchanged=True,
        original_entrypoint=base['original_entrypoint'],
        added_distributions=list(DISTRIBUTIONS), added_python_modules=sorted(added),
        added_python_source_count=len(added), added_python_bytes=sum(sources[k]['bytes'] for k in added),
        distributions=distributions, python_sources=sources, namespace_packages=namespaces,
        native_payload_stat_only=natives, graph=graph, unresolved_boundary_presence=presence,
        python_source_count=len(sources), python_source_bytes=sum(row['bytes'] for row in sources.values()),
        known_remaining_boundaries=[name for name in roots if name not in sys.stdlib_module_names],
        static_only=True, installed_package_code_executed=False, actual_SDK_helper_executed=False,
        native_payload_bytes_read=0, model_or_native_runtime_executed=False, runtime_admitted=False,
        full_model_qualified=False, new_speedup_claim=False, complete_runtime_dependency_closure_proven=False,
        limitations=[
            'Presence in a conservative module-body graph does not prove this optional package executes for the real helper.',
            'Native hashes are RECORD declarations only; native payloads were not read or loaded.',
            'six.moves and native-created submodules cannot be classified by missing filesystem paths alone.',
            'Function bodies, dynamic imports, custom finders, optional/platform paths and native linked libraries remain runtime context work.',
            'Interpreter/stdlib/startup authentication and actual helper/root/worker integration remain required.',
        ], source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__).resolve(), Q/'nemotron4b_helper_dependency_catalog_v1.py',
            Q/'analyze_nemotron4b_helper_dependencies_v2.py',
            Q/'analyze_nemotron4b_helper_imports_v1.py', Q/'nemotron4b_helper_import_graph_v1.py')})
    data = (json.dumps(result, indent=2, allow_nan=False)+'\n').encode()
    c.require(len(data) <= 8*2**20, 'Bounded extended inventory')
    with OUT.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(path=str(OUT), sha256=hashlib.sha256(data).hexdigest(),
        distributions=len(distributions), added_python_sources=len(added),
        added_python_bytes=result['added_python_bytes'], all_python_sources=len(sources),
        all_python_bytes=result['python_source_bytes'],
        reachable_python_modules=len(graph['reachable_python_modules']),
        native_stat_only_modules=sorted(natives), reachable_native_boundaries=graph['reachable_native_boundaries'],
        remaining_boundaries=result['known_remaining_boundaries'],
        original119pins_unchanged=True, original263sources_unchanged=True,
        native_payload_bytes_read=0, actual_SDK_helper_executed=False)))


if __name__ == '__main__':
    main()
