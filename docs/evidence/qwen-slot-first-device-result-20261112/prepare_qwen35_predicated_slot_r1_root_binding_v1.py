"""Thin source-only slot binding candidate after the exact package r4 terminal.

Explicit future catalog/handoff/terminal refs only. Reuse the already verified
r4 source/input partitions and accepted slot-v3 contract. No History.resolve,
native import, physical collector, process launch, lock, or running ACK here.
The root must review the draft and run the full original History precheck.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import qwen35_predicated_slot_serial_contract_v3 as slot
import prepare_qwen35_sampler_v9_candidate_v3 as objects
from prepare_qwen35_sampler_r6_candidate_v1 import JsonMetadata
from qwen35_sampler_runtime_contract_v12 import merge_sources, require, reference
from qwen35_sampler_runtime_io_v12 import identity, ref

FIXED = {
    'seed_spec': {'path': str(Q / 'qwen35-package-preflight-r4-20261112-spec-v5.json'),
                  'sha256': '5d11ef9a549cebf16c8434dc0dbf297d4a3ccb97c8826ad882aa67c0cf597ecf'},
    'seed_review': {'path': str(Q / 'qwen35-package-preflight-r4-20261112-review-v5.json'),
                    'sha256': '7cce54029c113c1828785cf12434ae8acfc70a23e28244c93cab99e8fecf12d1'},
    'seed_overlay': {'path': str(Q / 'qwen35-package-preflight-r4-20261112-history-overlay-v5.json'),
                     'sha256': '57d607e754229a93c35fece91f50852eb9d4997493bbbf30a2febeef5790af02'},
    'seed_binding': {'path': str(Q / 'qwen35-package-preflight-r4-20261112-root-binding-v5.json'),
                     'sha256': 'afc08e66df9d0b3954fa9d6aecd7de7136844cf28ee28dd7273dcabf9eeea6e1'},
    'delivery': {'path': str(Q / 'qwen35-predicated-slot-serial-source-delivery-20261112-v3.json'),
                 'sha256': '571d052ac0b633721ccea078717910723da42d7870205da76d95ace62b31ffcd'},
    'acceptance': {'path': str(Q / 'qwen35-predicated-slot-serial-root-source-review-20261112-v3.json'),
                   'sha256': '8034727dfe0be10c5e4b3e2c486a2b133f9aa0453fe09f659dc9619db1a8dfcf'},
}
R4_SESSION = 30240


def paths(run_name, running_handoff):
    require(type(run_name) is str and re.fullmatch('[a-z0-9][a-z0-9-]{0,39}', run_name), 'Exact bounded slot run name')
    running = Path(running_handoff)
    require(running.parent == Q and str(running) == running_handoff
            and re.fullmatch(r'continuation-handoff-[a-z0-9-]+\.json', running.name), 'Original running handoff namespace')
    prefix = 'qwen35-predicated-slot-' + run_name
    return dict(overlay=str(Q / (prefix + '-history-overlay.json')),
        spec=str(Q / (prefix + '-spec.json')), review_draft=str(Q / (prefix + '-review-draft.json')),
        review=str(Q / (prefix + '-review.json')), candidate=str(Q / (prefix + '-binding-candidate.json')),
        output='/mnt/nvme/neuron-work/' + prefix,
        overlay_output='/mnt/nvme/minicpm5-layer1-operator-native-slot-' + run_name,
        running=running_handoff)


def candidate(catalog_ref, handoff_ref, terminal_ref, run_name, running_handoff, *, read, observed, preparer_ref):
    """Pure in-memory composition using explicit authenticated read callbacks."""
    for value in (catalog_ref, handoff_ref, terminal_ref, preparer_ref): reference(value)
    destination = paths(run_name, running_handoff)
    seed, old_review, old_overlay, binding, delivery, acceptance = [read(FIXED[key]) for key in FIXED]
    require(binding['passed'] is binding['full_source_history_resolution_passed'] is True
        and binding['spec'] == FIXED['seed_spec'] and binding['review'] == FIXED['seed_review']
        and binding['history_overlay'] == FIXED['seed_overlay'] and binding['lfm_children'] == 874
        and binding['catalog_rows'] == 149 and binding['source_files'] == len(seed['source_sha256']) == 48194,
        'Exact independently verified package r4 seed; no reduced reconstruction claim')
    require(seed['history_overlay'] == FIXED['seed_overlay']
        and seed['current_serial_catalog'] == old_overlay['current_serial_catalog'] == binding['catalog']
        and old_review['spec'] == FIXED['seed_spec'] and old_review['source_sha256'] == seed['source_sha256']
        and old_review['passed'] is old_review['launch_authorized'] is True and old_review['blockers'] == [],
        'Exact original seed spec/review/overlay/catalog edges')
    require(acceptance['reviewer'] == '/root' and acceptance['source_accepted'] is True
        and acceptance['blockers'] == [] and acceptance['delivery'] == FIXED['delivery']
        and delivery['schema'] == 'qwen35_predicated_slot_serial_source_delivery_v3'
        and delivery['primitive_manifest'] == slot.MANIFEST
        and delivery['workload'] == slot.protocol.workload() and delivery['limits'] == slot.LIMITS,
        'Existing accepted complete slot v3 implementation, workload and limits')
    old_catalog = read(seed['current_serial_catalog']); catalog = read(catalog_ref)
    objects.catalog_shape(old_catalog); objects.catalog_shape(catalog)
    require(len(old_catalog['runs']) == 149 and len(catalog['runs']) == 150
        and catalog['runs'][:-1] == old_catalog['runs']
        and catalog['completed_sdk'] == old_catalog['completed_sdk']
        and catalog['current_handoff'] == handoff_ref, 'All149 exact seed rows plus the one actual r4 row')
    handoff, terminal = read(handoff_ref), read(terminal_ref)
    require(handoff['serial_execution'] is True and handoff['live_tool_sessions'] == []
        and handoff['active_heavy_units'] == [], 'Explicit complete idle post-r4 handoff')
    row = catalog['runs'][-1]; tool = terminal['actual_tool']
    require(row['terminal'] == terminal_ref and row['tool'] == tool and row['owners'] == terminal['owners']
        and row['units'] == terminal['units'] and terminal['schema'] == 'qwen35_package_preflight_root_actual_exit_v1'
        and terminal['reviewer'] == '/root' and terminal['spec'] == FIXED['seed_spec']
        and terminal['review'] == FIXED['seed_review'] and tool['kind'] == 'tool_session'
        and tool['session_id'] == R4_SESSION and type(tool['exit_code']) is int and 0 <= tool['exit_code'] <= 255
        and re.fullmatch('[a-f0-9]{3,64}', tool['terminal_chunk'])
        and terminal['actual_tool_exit_observed'] is terminal['actual_outer_exit_observed'] is True
        and terminal['physically_drained'] is terminal['owners_gone'] is True
        and terminal['original_lock_reacquired_and_released'] is True
        and terminal['lock_identity'] == slot.LOCK_IDENTITY
        and terminal['full_model_qualified'] is terminal['new_speedup_claim'] is False,
        'Actual root-observed r4 tool terminal and complete drain, preserving its real success or failure')
    require(all(row['source_sha256'].get(path) == sha for path, sha in seed['source_sha256'].items()),
            'All actual package r4 sources retained in its terminal catalog row')
    base = read(old_overlay['base_plan'])
    require(len(base['inputs']) == 8791, 'Complete original base input partition')
    recovered = {path: sha for path, sha in seed['source_sha256'].items() if path not in old_overlay['source_extension']}
    require(len(recovered) == 44532 and not set(recovered) & set(old_overlay['source_extension'])
        and merge_sources(recovered, old_overlay['source_extension']) == seed['source_sha256'],
        'Exact original recovered44532 plus complete disjoint seed extension')
    read(seed['lfm_r2_scope'])
    read({key: slot.MANIFEST[key] for key in ('path', 'sha256')})
    metadata_refs = [FIXED[key] for key in ('seed_binding', 'delivery', 'acceptance')] + [catalog_ref, handoff_ref, terminal_ref]
    require(all(type(row['source_sha256']) is dict for row in catalog['runs']), 'Every full catalog row has its source mapping')
    # Some immutable historical rows have an empty map. They stay in the full
    # catalog; the original merge_sources accepts only nonempty operands.
    pins = merge_sources(seed['source_sha256'], *(row['source_sha256'] for row in catalog['runs'] if row['source_sha256']),
        delivery['static_local_source_sha256'], delivery['sdk_source_sha256'],
        {item['path']: item['sha256'] for item in metadata_refs}, {preparer_ref['path']: preparer_ref['sha256']})
    inherited = copy.deepcopy(base['inputs'])
    objects.merge_inputs(inherited, seed['input_extension'])
    objects.merge_inputs(inherited, old_overlay['input_extension'])
    objects.merge_inputs(inherited, observed)
    ordinary = {path: copy.deepcopy(value) for path, value in observed.items() if value['identity']['size'] <= 8 * 2**20}
    preparation_only = {path: copy.deepcopy(value) for path, value in observed.items() if path not in ordinary}
    require(FIXED['seed_spec']['path'] in preparation_only and FIXED['seed_review']['path'] in preparation_only
        and old_overlay['base_plan']['path'] in preparation_only, 'Large seed envelopes stay preparation-only')
    overlay = copy.deepcopy(old_overlay)
    overlay.update(current_serial_catalog=copy.deepcopy(catalog_ref), output_root=destination['overlay_output'],
        source_extension={path: sha for path, sha in pins.items() if path not in recovered})
    objects.merge_inputs(overlay['input_extension'], {path: value for path, value in ordinary.items() if path not in base['inputs']})
    require(not set(overlay['input_extension']) & set(base['inputs']), 'Original disjoint overlay input partition')
    overlay_ref = objects.object_ref(destination['overlay'], overlay, ordinary=True)
    inputs = copy.deepcopy(seed['input_extension']); objects.merge_inputs(inputs, ordinary)
    require(all(value['identity']['size'] <= 8 * 2**20 for value in inputs.values()), 'Ordinary runtime input cap unchanged')
    spec = dict(schema='qwen35_predicated_slot_serial_spec_v3', run_name=run_name, output_root=destination['output'],
        history_overlay=overlay_ref, current_serial_catalog=copy.deepcopy(catalog_ref), source_sha256=pins,
        input_extension=inputs, running_handoff_path=running_handoff, lfm_r2_scope=copy.deepcopy(seed['lfm_r2_scope']),
        source_delivery=copy.deepcopy(FIXED['delivery']), primitive_manifest=copy.deepcopy(slot.MANIFEST),
        workload=slot.protocol.workload(), limits=copy.deepcopy(slot.LIMITS), lock_identity=copy.deepcopy(slot.LOCK_IDENTITY),
        native_python=slot.PYTHON, automatic_successor=False)
    spec_ref = objects.object_ref(destination['spec'], spec)
    draft = dict(schema='qwen35_sampler_runtime_source_review_v1', reviewer='/root', passed=False,
        blockers=['Concrete root review and complete original History precheck pending'], spec=spec_ref,
        source_sha256=copy.deepcopy(pins), launch_authorized=False)
    # Validate the candidate shape without publishing or claiming approval.
    slot.validate(spec, dict(draft, passed=True, blockers=[], launch_authorized=True), spec_ref)
    return dict(paths=destination, overlay=overlay, spec=spec, review_draft=draft,
        overlay_ref=overlay_ref, spec_ref=spec_ref,
        review_draft_ref=objects.object_ref(destination['review_draft'], draft),
        preparation_only_inputs=preparation_only, catalog_rows=150, source_files=len(pins),
        root_review_pending=True, full_history_precheck_pending=True, physical_pair_performed=False,
        lock_acquired=False, native_executed=False, compiler_executed=False, runtime_launched=False)


def publish(value, unchanged):
    """Exclusive metadata publication only; never create output or handoff."""
    selected = value['paths']
    require(len(set(selected.values())) == len(selected), 'Every future path distinct')
    for path in selected.values():
        require(not Path(path).exists() and not Path(path).is_symlink(), 'Every future path must be fresh: ' + path)
    unchanged()
    for key in ('overlay', 'spec', 'review_draft'):
        path = selected[key]; raw = objects.encode(value[key])
        with Path(path).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        require(ref(path, 64 * 2**20) == value[key + '_ref'], 'Exact original encoded candidate bytes')
    unchanged()
    manifest = {key: copy.deepcopy(item) for key, item in value.items() if key not in ('overlay', 'spec', 'review_draft')}
    manifest['schema'] = 'qwen35_predicated_slot_root_binding_candidate_v1'
    with Path(selected['candidate']).open('xb') as stream:
        stream.write(objects.encode(manifest)); stream.flush(); os.fsync(stream.fileno())
    return ref(selected['candidate'], 64 * 2**20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('catalog', 'catalog-sha256', 'handoff', 'handoff-sha256', 'terminal', 'terminal-sha256', 'run-name', 'running-handoff'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    require(sys.dont_write_bytecode and __debug__, 'Original -B and assertions')
    reader = JsonMetadata(lambda: None)
    def read(reference): return reader.read(reference)['value']
    value = candidate(dict(path=args.catalog, sha256=args.catalog_sha256),
        dict(path=args.handoff, sha256=args.handoff_sha256), dict(path=args.terminal, sha256=args.terminal_sha256),
        args.run_name, args.running_handoff, read=read, observed=reader.observed,
        preparer_ref=ref(str(Path(__file__).resolve())))
    slot.verify_delivery(value['spec'])
    slot.resources(value['spec'], reserve=True)  # Original40GiB initial NVMe requirement; no new ceiling.
    def unchanged():
        require(all(identity(path) == row['identity'] for path, row in reader.observed.items()), 'Every read metadata identity unchanged')
    print(json.dumps(publish(value, unchanged), indent=2))


if __name__ == '__main__': main()
