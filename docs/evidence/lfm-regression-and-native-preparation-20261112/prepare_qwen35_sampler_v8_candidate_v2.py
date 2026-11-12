"""Finite r6-to-v8 metadata candidate; no launch, latest lookup or file writes.

All future refs and output names are explicit. Complete recovered/witness
resolution, source authentication and physical freshness remain root work.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

from qwen35_sampler_runtime_contract_v8 import (LIMITS, LOCK_IDENTITY, PYTHON, Q,
    digest, merge_sources, reference, require, units)

FIXED = {
    'spec': dict(path=str(Q/'qwen35-sampler-r6-20261112-spec-v3.json'),
                 sha256='fccff37e79995b084dcf7929a2ea1b0967bf945b037f39332f4934e63850761e'),
    'overlay': dict(path=str(Q/'qwen35-sampler-r6-20261112-history-overlay-v3.json'),
                    sha256='fc1a36c3bead2e9cffb48e04cfcc31b2e671f878ee8866eb582231084f8db652'),
    'catalog': dict(path=str(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v75.json'),
                    sha256='867cf38455632f091a57ecf9eb5a1cedf25b60294668dd8cc6c3a67f12cf6b8d'),
    'completed_catalog': dict(path=str(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v76.json'),
                    sha256='d9b86ae53918266626c5ce6ee63f6909d359bcaa32012e7d5b05f5b3280bed3f'),
    'r6_exit': dict(path=str(Q/'qwen35-sampler-r6-root-actual-exit-20261112-v1.json'),
                    sha256='c8db5a0260afdf4d3ec6a0ee6e6db1beaa12f5ae6ff646ee2e323fba8b3f973f'),
    'base_plan': dict(path=str(Q/'minicpm5-layer1-operator-sdk-r2-20261111-plan.json'),
                    sha256='2e5441f4b2c2ad2efd76192ae92c068c35d3ae9fc51c532062d4284692910632'),
    'ancestor_spec': dict(path=str(Q/'qwen35-sampler-r5-20261112-spec.json'),
                    sha256='0cc84d4ee409e2bf26f6e46dcdc746c1e793be0b7ef5811fcc939dd3d849540e'),
    'ancestor_overlay': dict(path=str(Q/'qwen35-sampler-r5-20261112-history-overlay.json'),
                    sha256='0eca9211dc27e65de9369f853efd51cde8a39777e115044eed1ea29d3ef1f429'),
    'binding_review': dict(path=str(Q/'qwen35-sampler-r6-root-binding-review-20261112-v1.json'),
                    sha256='dd59441c984bef2d7faf5c54250003188fb99e65347a77e2a730bfa0fcf92254'),
    'runtime_review': dict(path=str(Q/'qwen35-sampler-api-failure-drain-v8-root-source-review-20261112-v1.json'),
                    sha256='851eb35b3c0620cf93f48924480e87d82e4ed60230c1307a980a591d05cf91ea'),
    'runtime_delivery': dict(path=str(Q/'qwen35-sampler-api-failure-drain-v8-source-delivery-20261112-v1.json'),
                    sha256='f5486539821ea5a73d3940aae35902cd3b82b70ea28fcdc2ad2e96374bb0db90'),
    'leaf_review': dict(path=str(Q/'qwen35-sampler-cold-install-root-source-review-20261112-v2.json'),
                    sha256='bf8345e4ec4e2771e7a29937d15e66a1be3d1521a7c9058e75e212449c303b7a'),
    'lfm_spec': dict(path=str(Q/'lfm3-full-cpu-r2-20261112-spec.json'),
                    sha256='955c2267d1ac27bb1af5f680b4e0acefa4355c05bdfa874cf4f46f4381116630'),
}
# Exact historical large envelopes are preparation evidence only. They are
# never ordinary runtime metadata inputs, even if an injected reader accepts them.
PREPARATION_ONLY = frozenset(FIXED[key]['path'] for key in ('spec','base_plan','ancestor_spec'))

LEAF = dict(path=str(Q/'qwen35_sampler_native_leaves_v2.py'),
            sha256='efdca076af1eafbc423f0b23f3c5604204efc84c2b3ee7f9d6d200809d160db7')
V8_NAMES = tuple('qwen35_sampler_runtime_'+name+'_v8.py' for name in ('contract','io','history','service','adapter')) + ('run_qwen35_sampler_runtime_v8.py',)
IDENTITY_KEYS = {'realpath','device','inode','size','mtime_ns','ctime_ns'}


def encode(value):
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    require(len(raw) <= LIMITS['spec_envelope_bytes'], 'Original finite metadata envelope')
    return raw


def object_ref(path,value,*,ordinary=False):
    data=encode(value)
    require(not ordinary or len(data)<=LIMITS['metadata_bytes'], 'Candidate overlay exceeds original ordinary metadata envelope')
    return dict(path=path,sha256=hashlib.sha256(data).hexdigest())


class JsonMetadata:
    """Future explicit JSON-only stable reader; no reads during construction."""
    def __init__(self, guard): self.guard, self.observed = guard, {}
    def read(self, ref):
        from qwen35_sampler_runtime_io_v8 import identity, raw
        reference(ref)
        require(Path(ref['path']).suffix == '.json', 'Only explicit saved JSON metadata')
        self.guard(); before = identity(ref['path'])
        cap = LIMITS['spec_envelope_bytes'] if ref in [FIXED[k] for k in ('spec','base_plan','ancestor_spec')] else LIMITS['metadata_bytes']
        payload = raw(ref['path'], cap)
        require(identity(ref['path']) == before and hashlib.sha256(payload).hexdigest() == ref['sha256'],
                'Complete stable metadata identity and hash')
        row = dict(sha256=ref['sha256'], identity=before)
        require(ref['path'] not in self.observed or self.observed[ref['path']] == row,
                'Previously read metadata identity cannot be refreshed')
        def unique(pairs):
            result = {}
            for key, value in pairs:
                require(key not in result, 'Unique JSON keys'); result[key] = value
            return result
        value = json.loads(payload, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Finite JSON required')))
        self.observed[ref['path']] = row; self.guard()
        return dict(value=value, identity=copy.deepcopy(before))


def merge_inputs(target, addition):
    for path, row in addition.items():
        reference(dict(path=path, sha256=row['sha256']))
        require(type(row) is dict and set(row) == {'sha256','identity'} and type(row['identity']) is dict,
                'Complete original input identity')
        require(path not in target or target[path] == row, 'Old input identity or digest cannot be replaced: '+path)
        target[path] = copy.deepcopy(row)


def catalog_shape(value):
    require(type(value) is dict and set(value) == {'schema','reviewer','passed','blockers','completed_sdk',
            'current_handoff','all_later_serial_runs_included','runs'}
            and value['schema'] == 'minicpm5_operator_later_serial_metadata_catalog_v1'
            and value['reviewer'] == '/root' and value['passed'] is True and value['blockers'] == []
            and value['all_later_serial_runs_included'] is True and type(value['runs']) is list
            and 0 < len(value['runs']) <= 256, 'Original complete root serial catalog contract')
    reference(value['current_handoff'])
    result = {}
    for row in value['runs']:
        require(type(row) is dict and set(row) == {'terminal','tool','owners','units','source_sha256'},
                'Complete original serial row')
        reference(row['terminal']); path = row['terminal']['path']
        require(path not in result, 'No duplicate serial terminal')
        result[path] = row
    return result


def future_shape(future, old_spec, old_overlay):
    require(type(future) is dict and set(future) == {'terminal','prerequisite_exit','catalog','run_name','output_root',
        'overlay_output_root','running_handoff_path','overlay_path','spec_path','review_candidate_path'},
        'Every future terminal, catalog, run and output binding must be explicit')
    for key in ('terminal','prerequisite_exit','catalog'): reference(future[key])
    units(future['run_name'])
    require(future['run_name'] != old_spec['run_name']
            and future['output_root'] == '/mnt/nvme/neuron-work/qwen35-sampler-pair-'+future['run_name']
            and future['output_root'] != old_spec['output_root'], 'New exact Qwen output namespace')
    root = Path(future['overlay_output_root'])
    require(root.parent == Path('/mnt/nvme') and re.fullmatch('minicpm5-layer1-operator-native-[a-z0-9-]{1,64}',root.name)
            and str(root) == future['overlay_output_root'] and future['overlay_output_root'] != old_overlay['output_root'],
            'Explicit fresh metadata-only overlay output namespace')
    for key in ('running_handoff_path','overlay_path','spec_path','review_candidate_path'):
        path = Path(future[key]); require(path.parent == Q and str(path) == future[key]
            and re.fullmatch('[a-z0-9][a-z0-9-]*\.json',path.name), 'Finite explicit research JSON output')
    require(Path(future['running_handoff_path']).name.startswith('continuation-handoff-'), 'Original future handoff namespace')
    paths = [future[key] for key in ('running_handoff_path','overlay_path','spec_path','review_candidate_path')]
    forbidden = {ref['path'] for ref in FIXED.values()} | {future[key]['path'] for key in ('terminal','prerequisite_exit','catalog')}
    forbidden.add(old_spec['running_handoff_path'])
    require(len(set(paths)) == 4 and not set(paths) & forbidden, 'Candidate paths cannot overwrite bound or historical metadata')


def contains_reference(value,ref):
    """Saved finite JSON only, not a claim that idle evidence is physically fresh."""
    if type(value) is dict:
        return value==ref or any(contains_reference(v,ref) for v in value.values())
    if type(value) is list:return any(contains_reference(v,ref) for v in value)
    return False


def prepare(future, read_metadata, guard):
    """Return candidate objects and exact bytes, without writes or authority.

    read_metadata(ref) returns {value, identity} after full stable JSON hashing.
    Real use should pass JsonMetadata(guard).read; mocks must identify themselves.
    terminal is the explicit idle handoff; prerequisite_exit is the separately
    root-observed completed workload (e.g. the future LFM terminal).
    """
    require(callable(read_metadata) and callable(guard), 'Explicit stable metadata reader and guard')
    observed = {}; values = {}
    def read(ref):
        reference(ref); require(Path(ref['path']).suffix == '.json', 'Metadata-only reference')
        guard(); saved = read_metadata(copy.deepcopy(ref))
        require(type(saved) is dict and set(saved) == {'value','identity'} and type(saved['identity']) is dict
                and set(saved['identity']) == IDENTITY_KEYS and saved['identity']['realpath'] == ref['path'],
                'Stable complete metadata stat identity supplied by reader')
        for key in IDENTITY_KEYS - {'realpath'}:
            require(type(saved['identity'][key]) is int and saved['identity'][key] >= 0, 'Finite numeric metadata identity')
        cap=LIMITS['spec_envelope_bytes'] if ref['path'] in PREPARATION_ONLY else LIMITS['metadata_bytes']
        require(saved['identity']['size']<=cap, 'Exact bounded metadata class; no large future runtime input')
        row = dict(sha256=ref['sha256'], identity=copy.deepcopy(saved['identity']))
        merge_inputs(observed, {ref['path']: row}); values[ref['path']] = copy.deepcopy(saved['value'])
        guard(); return saved['value']
    fixed = {key:read(ref) for key,ref in FIXED.items()}
    old_spec,old_overlay,old_catalog = (fixed[key] for key in ('spec','overlay','catalog'))
    review,delivery,leaf_review = (fixed[key] for key in ('runtime_review','runtime_delivery','leaf_review'))
    binding_review=fixed['binding_review'];base=fixed['base_plan']
    ancestor,ancestor_overlay=fixed['ancestor_spec'],fixed['ancestor_overlay']
    require(binding_review['reviewer']=='/root' and binding_review['passed'] is True
            and binding_review['spec']==FIXED['spec'] and binding_review['history_overlay']==FIXED['overlay']
            and binding_review['source_count']==47959 and binding_review['original36_spec_inputs_and137_overlay_inputs_retained'] is True,
            'Exact independently resolved r6-v3 source and input partition')
    require(len(old_spec['source_sha256'])==47959 and len(old_overlay['source_extension'])==3427
            and old_spec['history_overlay']==FIXED['overlay']
            and old_spec['current_serial_catalog']==old_overlay['current_serial_catalog']==FIXED['catalog']
            and old_overlay['base_plan']==FIXED['base_plan'] and len(base['inputs'])==8791,
            'Exact executed r6 47959 sources, 3427 extension and original 8791 base inputs')
    require(len(ancestor['input_extension'])==36 and len(ancestor_overlay['input_extension'])==137
            and all(old_spec['input_extension'].get(p)==row for p,row in ancestor['input_extension'].items())
            and all(old_overlay['input_extension'].get(p)==row for p,row in ancestor_overlay['input_extension'].items())
            and all(old_spec['source_sha256'].get(p)==sha for p,sha in ancestor['source_sha256'].items()),
            'Every original36/137 identity and all original r5 sources retained')
    require(not set(base['inputs']) & set(old_overlay['input_extension']), 'Frozen corrected overlay has no duplicate base input')
    require(old_spec['limits']==LIMITS and old_spec['native_python']==PYTHON
            and old_spec['lock_identity']==LOCK_IDENTITY and old_spec['automatic_successor'] is False,
            'Original complete scientific runtime/resources unchanged')
    future_shape(future,old_spec,old_overlay)
    require(review['schema']=='qwen35_sampler_api_failure_drain_v8_root_source_review_v1' and review['reviewer']=='/root'
            and review['passed'] is True and review['source_only'] is True and review['runtime_admitted'] is False
            and review['source_delivery']==FIXED['runtime_delivery']
            and set(review['source_files'])==set(V8_NAMES) and review['source_files']==delivery['source_files']
            and review['original_science_and_resources_unchanged'] is True
            and review['original_cleanup_on_wait_failure_preserved'] is True,
            'Exact independently accepted six-file v8 source review')
    require(leaf_review['passed'] is True and leaf_review['reviewer']=='/root' and leaf_review['source']==LEAF
            and old_spec['source_sha256'].get(LEAF['path'])==LEAF['sha256'], 'Accepted original leaf v2 retained')
    v8={row['path']:row['sha256'] for row in review['source_files'].values()}
    require(set(v8)=={str(Q/name) for name in V8_NAMES}, 'Exact v8 paths; historical v6 and original science retained')
    catalog=read(future['catalog']);old_rows=catalog_shape(old_catalog);rows=catalog_shape(catalog)
    completed_rows=catalog_shape(fixed['completed_catalog'])
    require(len(old_rows)==120 and len(completed_rows)==121
            and all(completed_rows.get(path)==row for path,row in old_rows.items())
            and all(rows.get(path)==row for path,row in completed_rows.items())
            and fixed['completed_catalog']['runs'][:120]==old_catalog['runs']
            and catalog['runs'][:121]==fixed['completed_catalog']['runs'],
            'All121 exact completed serial rows retained, including all120 original r6 predecessor rows')
    require(catalog['completed_sdk']==old_catalog['completed_sdk']==fixed['completed_catalog']['completed_sdk']
            and catalog['current_handoff']==future['terminal'], 'Original SDK binding and explicit future idle terminal')
    r6=fixed['r6_exit'];r6row=rows.get(FIXED['r6_exit']['path'],{})
    require(r6row.get('terminal')==FIXED['r6_exit'] and r6row.get('tool')==r6['actual_tool']
            and r6['actual_tool_exit_observed'] is True and r6['actual_tool']['exit_code']==1
            and r6['spec']==FIXED['spec'] and r6['owners_gone'] is True and r6['physically_drained'] is True,
            'Actual failed r6 exit retained verbatim, never recast as successful')
    terminal=read(future['terminal'])
    require(terminal['serial_execution'] is True and terminal['live_tool_sessions']==[] and terminal['active_heavy_units']==[]
            and terminal.get('running') is False, 'Future catalog must bind a root-observed idle terminal')
    prerequisite=read(future['prerequisite_exit']);prerequisite_row=rows.get(future['prerequisite_exit']['path'],{})
    lfm=fixed['lfm_spec']
    require(future['prerequisite_exit']['path'] not in completed_rows
            and prerequisite_row.get('terminal')==future['prerequisite_exit']
            and prerequisite.get('reviewer')=='/root' and prerequisite.get('actual_tool_exit_observed') is True
            and prerequisite.get('actual_tool')==prerequisite_row.get('tool')
            and prerequisite.get('spec')==FIXED['lfm_spec'] and prerequisite.get('output_root')==lfm['output_root']
            and prerequisite.get('source_sha256')==lfm['source_sha256'] and len(lfm['source_sha256'])==595
            and all(prerequisite_row.get('source_sha256',{}).get(path)==sha for path,sha in lfm['source_sha256'].items())
            and prerequisite.get('owners_gone') is True and prerequisite.get('physically_drained') is True
            and prerequisite.get('original_lock_observed_free') is True and prerequisite.get('lock_identity')==LOCK_IDENTITY
            and prerequisite.get('owners')==prerequisite_row.get('owners')
            and lfm['unit'] in prerequisite_row.get('units',[]),
            'Explicit future LFM r2 actual exit, original595 sources, actual tool and drained lifecycle required')
    require(contains_reference(terminal,future['prerequisite_exit']), 'Idle handoff must retain the exact future LFM actual exit')
    row_pins = {}
    known_units = set()
    for row in catalog['runs']:
        guard(); ended = read(row['terminal']); require(type(ended) is dict and ended, 'Complete saved serial terminal metadata')
        tool = row['tool']
        require(type(tool) is dict and ((tool.get('kind') == 'tool_session' and set(tool) == {'kind','session_id','exit_code','terminal_chunk'}
                and type(tool['session_id']) is int and tool['session_id'] > 0)
                or (tool.get('kind') == 'tool_call' and set(tool) == {'kind','exit_code','terminal_chunk'}))
                and type(tool['exit_code']) is int and 0 <= tool['exit_code'] <= 255
                and type(tool['terminal_chunk']) is str and re.fullmatch('[a-f0-9]{3,64}',tool['terminal_chunk']),
                'Retain actual success or failure tool fields')
        require(type(row['owners']) is list and len(row['owners']) <= 256
                and type(row['units']) is list and len(row['units']) <= 32, 'Complete bounded row identities')
        for owner in row['owners']:
            require(type(owner) is dict and type(owner.get('pid')) is int and owner['pid'] > 0
                    and str(owner.get('start_ticks','')).isdigit(), 'Original recorded owner identity')
        for unit in row['units']:
            require(type(unit) is str and re.fullmatch('trn1-[A-Za-z0-9_-]{1,240}',unit), 'Exact saved service unit name')
            known_units.add(unit)
        require(type(row['source_sha256']) is dict, 'Complete row source map')
        for path, sha in row['source_sha256'].items():
            reference(dict(path=path,sha256=sha))
            require(path not in row_pins or row_pins[path] == sha, 'Conflicting source pins across complete future catalog rows')
            row_pins[path] = sha
    pins = merge_sources(old_spec['source_sha256'],v8,row_pins) if row_pins else merge_sources(old_spec['source_sha256'],v8)
    require(not set(units(future['run_name']).values()) & known_units, 'Future Qwen units disjoint from saved rows')
    require(all(pins.get(path) == sha for path,sha in old_overlay['source_extension'].items()), 'All old overlay source pins retained')
    recovered_pins = {path:sha for path,sha in old_spec['source_sha256'].items() if path not in old_overlay['source_extension']}
    require(len(recovered_pins) == 44532, 'Exact original recovered partition retained without rebuilding')
    # Compare all identities before partitioning. Even preparation-only reads
    # cannot silently refresh an identity that appeared in inherited evidence.
    inherited={};merge_inputs(inherited,base['inputs']);merge_inputs(inherited,old_spec['input_extension'])
    merge_inputs(inherited,old_overlay['input_extension']);merge_inputs(inherited,observed)
    runtime_observed={path:row for path,row in observed.items() if path not in PREPARATION_ONLY}
    require(all(row['identity']['size']<=LIMITS['metadata_bytes'] for row in runtime_observed.values()),
            'Only ordinary bounded JSON can enter runtime metadata inputs')
    overlay=copy.deepcopy(old_overlay)
    overlay.update(current_serial_catalog=copy.deepcopy(future['catalog']),output_root=future['overlay_output_root'],
        source_extension={path:sha for path,sha in pins.items() if path not in recovered_pins})
    overlay_inputs=copy.deepcopy(old_overlay['input_extension'])
    merge_inputs(overlay_inputs,{path:row for path,row in runtime_observed.items() if path not in base['inputs']})
    require(not set(overlay_inputs)&set(base['inputs']), 'No duplicate original base input in successor overlay')
    overlay['input_extension']=overlay_inputs
    overlay_ref = object_ref(future['overlay_path'],overlay,ordinary=True)
    spec = copy.deepcopy(old_spec)
    spec.update(run_name=future['run_name'], output_root=future['output_root'], running_handoff_path=future['running_handoff_path'],
                history_overlay=overlay_ref, current_serial_catalog=copy.deepcopy(future['catalog']),source_sha256=pins)
    spec_inputs = copy.deepcopy(old_spec['input_extension']);merge_inputs(spec_inputs,runtime_observed);spec['input_extension']=spec_inputs
    spec_ref = object_ref(future['spec_path'],spec)
    candidate = dict(schema='qwen35_sampler_runtime_source_review_v1',reviewer='/root/candidate-preparation',
        passed=False,blockers=['Independent root Bundle.resolve, source witness closure and runtime admission pending'],
        spec=spec_ref,source_sha256=copy.deepcopy(pins),launch_authorized=False)
    artifacts = {future['overlay_path']:overlay,future['spec_path']:spec,future['review_candidate_path']:candidate}
    guard()
    return dict(schema='qwen35_sampler_v8_metadata_candidate_v2', artifacts=artifacts,
        references={key:object_ref(future[key],artifacts[future[key]]) for key in ('overlay_path','spec_path','review_candidate_path')},
        evidence=dict(fixed_inputs=copy.deepcopy(FIXED),future=copy.deepcopy(future),original_sources=47959,
            full_candidate_sources=len(pins),original_serial_rows=120,completed_serial_rows=121,candidate_serial_rows=len(rows),
            all_original_sources_retained=True,all_original_rows_retained=True,
            original_spec_inputs=len(old_spec['input_extension']),original_overlay_inputs=len(old_overlay['input_extension']),
            all_original_input_identities_retained=True,new_metadata_observed=len(observed),
            original36_spec_inputs_retained=True,original137_overlay_inputs_retained=True,
            original_base_inputs=8791,overlay_base_input_intersection=0,
            preparation_only_inputs={p:observed[p] for p in sorted(PREPARATION_ONLY)},
            future_LFM_actual_exit_and_idle_explicit=True,r6_actual_failed_exit_preserved=True,
            full_recovered_witness_closure_verified=False,Bundle_resolve_performed=False,
            old_source_payloads_rehashed=False,physical_checks_performed=False,output_freshness_observed=False,
            runtime_admitted=False,launch_authorized=False,new_speedup_claim=False))


if __name__ == '__main__':
    raise RuntimeError('Source-only candidate API: explicit future refs and independent root review required; no default bindings')
