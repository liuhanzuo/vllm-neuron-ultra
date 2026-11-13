"""Pure Mini completed-writer source renderer; no IO, History, lock or CLI.

Root supplies actual authenticated completed catalog/terminal/finalization bytes.
Synthetic fixtures may call this function only in memory and cannot grant actual
admission. Returned successors must receive independent source review before use.
"""
import ast
import copy
import hashlib
import json
from pathlib import PurePosixPath as Path
import re


def require(value, message):
    if not value:
        raise ValueError(message)


def reference(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Exact supplied metadata reference')
    path = Path(value['path'])
    require(path.parent == Q and str(path) == value['path'] and '..' not in path.parts
            and path.suffix == '.json' and re.fullmatch('[a-f0-9]{64}', value['sha256']),
            'Canonical research JSON and actual SHA-256')


def metadata(row):
    require(type(row) is dict and set(row) == {'reference', 'payload'}, 'Explicit authenticated byte envelope')
    reference(row['reference']); raw = row['payload']
    require(type(raw) is bytes and 0 < len(raw) <= 8 * 2**20
            and hashlib.sha256(raw).hexdigest() == row['reference']['sha256'],
            'Full actual metadata digest and original 8 MiB cap')
    return json.loads(raw)

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT / 'research'
BASE_CATALOG = dict(path=str(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v116.json'),
    sha256='3dbe69187b90bd160fea38e235bc871061755596011d70ecd61b01c9867d48f5')
ACTUAL_OUTER_TOOL = dict(kind='tool_session',session_id=72363,terminal_chunk='129c0c',exit_code=1)
LOCK_IDENTITY = dict(device=66306,inode=8294421)
COMPOSITION = dict(recovered_sources=44532,retained_prior_sources=48560,runtime_sources=48632,
    post_audit_sources=48635,new_row_sources=75,all_previous_rows_retained=True,exact_full_mapping_reconstructed=True)
PREDECESSORS = {'accept_minicpm5_formal_v7_metadata_root_v3.py': 'b801cf9c0105d849b702efb5f1eb0b95a7d8704fdbf3040f48d3ec3287f563a2',
 'minicpm5_fixed_input_pv_catalog_auth_v4.py': 'bcb9458866a5e3b9cb9ac41bea9e930877a8d5d13a6bb4d28795aad9bdb88edd',
 'minicpm5_sdk_mapping_outer_source_entry_v13.py': 'ac91d686caa92567144ff449b5c79b4b746a7c3b9e896a83a40bf5f81a94ff1c',
 'prepare_minicpm5_catalog_formal_binding_v6.py': 'e0242abd589034d8d63577dbf7f28a4967b848fddd7ee57059a3c214d54e9c46',
 'prepare_minicpm5_formal_v7_metadata_root_v3.py': '56d259fe91e541a45f83581695298798b8349d2b240334eeb28a16f35a2eef86',
 'prepare_minicpm5_pv_sdk_import_catalog_v4.py': '7ae556692d9635ec929f33d5a786996b23d7c29e05d83ede4eb88c02398a7deb'}
EXTERNAL_TEXT = {'/home/ubuntu/accept_qwen35_writer_geometry_root_r2_20261112_v1.py': 'e41da08cea4eacb6d048a61df00ff2443c82b243242d762791da0733e1303fc5',
 '/home/ubuntu/ack_qwen35_writer_geometry_root_r2_20261112_v1.py': 'da7241408a6cdd972123aed5d1ab6d813e87061b5990cae9e579277db9a7120b',
 '/home/ubuntu/finalize_qwen35_writer_geometry_r2_20261112_v1.py': 'a142c013ee978d5d270eb1b8ee7f99083b15a2aff94498b980fe52b95b5da0c2',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/__init__.py': '70d4db750a8213758205101d73186af02a14d9fee883303023ae5efd37208fec',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/__init__.py': 'f4b98116f32de98fc7ac1103dbb9e8ec10805ea3466d4c8fed2a42e2fb9fe68f',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/lfm3/artifacts.py': 'ad799b2a14f26d9e53aa86d73d12437eebb0f89323b6260ca5e3160547bd6286',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/nemotron9b/__init__.py': 'e2435f7fae27ed68fda952ebffc854204a320e720eb7eca9bdd623283dcd5959',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/nemotron9b/artifacts.py': '23e9512442c235b971dd04c19c9ad2103d699630f3895ae81136050f2c491187',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/nemotron9b/identity.py': 'b95e4e7535cad63e6df707aaa5422ac4ad26879386f9b2c9d0777667636699e4',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/nemotron9b/modeling.py': 'b86ca3656e048d38958d535db5c01de387b2a6897048d9a3f24c354630288665',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/nemotron9b/state.py': '634063828f9799823a78a56ad16a65ac955e9878ee9f18fd1c504ca24368df19',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/__init__.py': 'a07686e18368bfded1b040270c2e335ece72db3aefa15a79020c2ff067a4c28f',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/artifacts.py': 'f32a7bffb8752514aa65ca7e0599b3781557bf7974e7b3bec311df473b656091',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/build.py': 'e449f71f5759cc861cb02fd3b40d6d3186f64d8d33cf275da1f901d9641514c8',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/cache_writer.py': '7ad18c9dff829ea3649f6968bf51b7696902d19d6ac035e9a63f4752dd19b66a',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/helpers.py': 'b77c567096ba8869b04ebe9971d2418939752f506a5fce6ccebbc5ea1d9b787d',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/identity.py': '900e021075bb919a2966d938ea26692b304ccfc76e5fa7e9071ddce40011030c',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/modeling.py': '791e4e8fb46edd7062e7d0349f7df25dd0a8f74ee4a67d32a74d123b91e425ac',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/prepare_view.py': 'c326d48eee241dc60a1f7a7b779f8bd75182809fb7b1991f7a7d177ad91f566b',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/primitives.py': '64db0f2a04dcf41f3747694419d576fa4fb865038fc273c5030bb7604ade81fa',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/provenance.py': '682313f7b2e71b6b5bbeb3a11e82190bfa3c77ddd9a209bec36ae85835e77e89',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/registration.py': 'bb5a3a97efa7ea253549697c0386c7a3e98d26a76dcf9f47faa79ba259a43ff7',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/serve.py': '7a16b34b805075b5eef9771057cbcc3ebb0fb207814072d782fd3aa9212a9265',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/slot_cache_nki_predicated.py': 'b2f0ef3a727409c9816dc9c5e77cfd13df3a1777f4cdc31c8ce3e8cd373eca7e',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/state.py': '095e46f6d02dfc4b43cd77ebaa20a7a25a4879d2786b2c1a2ed5a6e73ffbabc7',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/text_core.py': '3f0a966af288d87c55e72f5dd41b8f11f343b43d3c1f45385fa360d774aacdfb',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/qwen35/validation.py': '294c81d07aee0396c960afc1d6f78a0da98d838ab05cd970cbe55a8536ab6dff',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/smollm3/artifacts.py': '4bc4892f971fa9fabed049e13aff0e34a95056e6dcbb8cc4bb3107189670d2f5',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/smollm3/identity.py': '31cc6a3993ec0bf1be61cb07f431ed682ecb4d4c678a03d1d3e9af30067645e2',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/models/smollm3/modeling.py': '8501ad8016a3a7f6396825e9410ed7878083791ef4e0b4542081e9d08243a6a5',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/platform.py': '4387dfe509f055d72b93d955c1fb103142e000c452898680bb71e3e0886f8f26',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/platform_overrides.py': '01e81c6ec753ced45dd57bded421d0909ec930b0cd61ec25382b7dfb711aba6d',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/worker/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/worker/constants.py': '8342a9cfcd077b3c3b5d8ad23254fa3080a1f30c02331a7889c21965ad9726a3',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/worker/cpu_sampler.py': '9638cd6189a819684efe02fb25b959065c9c5ea5cfc3c45ea3a9153b9ce4289a',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/worker/neuron_worker.py': '3f3c2195e1fc399e56b701d63a6d1f67f46d6bcaef38b42ffce095fe1a1201ba',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/worker/neuronx_distributed_model_loader.py': 'c1881ab64a296bb6a98657b8a8135ee8facc954c5e684803f2624dd2e2dcbb31',
 '/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112/vllm_neuron/worker/neuronx_distributed_model_runner.py': '4484edd883d199310e1646969417e05a8162148e404f840dc66cad6c6233a77e'}
ROOT_HELPERS = {'/home/ubuntu/accept_qwen35_writer_geometry_root_r2_20261112_v1.py': 'e41da08cea4eacb6d048a61df00ff2443c82b243242d762791da0733e1303fc5',
 '/home/ubuntu/ack_qwen35_writer_geometry_root_r2_20261112_v1.py': 'da7241408a6cdd972123aed5d1ab6d813e87061b5990cae9e579277db9a7120b',
 '/home/ubuntu/finalize_qwen35_writer_geometry_r2_20261112_v1.py': 'a142c013ee978d5d270eb1b8ee7f99083b15a2aff94498b980fe52b95b5da0c2'}


def completed_inputs(inputs):
    """Validate caller bytes; never discover or read a prospective output."""
    require(type(inputs) is dict and set(inputs) == {'base', 'catalog', 'ended', 'finalization'},
            'Actual four completed metadata inputs required')
    values = {key: metadata(row) for key, row in inputs.items()}
    refs = {key: row['reference'] for key, row in inputs.items()}
    base, catalog, ended, final = (values[k] for k in ('base', 'catalog', 'ended', 'finalization'))
    require(refs['base'] == BASE_CATALOG, 'Exact frozen v116 baseline, not a synthetic historical prefix')
    for value in (base, catalog):
        require(value['schema'] == 'minicpm5_operator_later_serial_metadata_catalog_v1'
                and value['reviewer'] == '/root' and value['passed'] is True and value['blockers'] == [],
                'Original complete catalog envelope')
    require(len(base['runs']) == 163 and len(catalog['runs']) == 164
            and catalog['runs'][:163] == base['runs']
            and catalog['completed_sdk'] == base['completed_sdk'], 'Every exact v116 row and SDK history retained')
    row = catalog['runs'][163]
    require(set(row) == {'terminal', 'tool', 'owners', 'units', 'source_sha256'}
            and row['terminal'] == refs['ended'] and row['tool'] == ACTUAL_OUTER_TOOL
            and len(row['source_sha256']) == 75
            and all(row['source_sha256'].get(path) == sha for path, sha in EXTERNAL_TEXT.items()),
            'Actual r2 row and exact 38 exceptional Python pins; no incomplete or arbitrary future row')
    require(ended['schema'] == 'qwen35_writer_geometry_root_actual_exit_v1'
            and ended['reviewer'] == '/root' and ended['actual_tool'] == ACTUAL_OUTER_TOOL
            and ended['actual_tool_exit_observed'] is ended['actual_outer_exit_observed'] is True
            and ended['owners'] == row['owners'] and ended['units'] == row['units']
            and ended['owners_gone'] is ended['physically_drained'] is True
            and ended['original_lock_held_during_audit'] is True and ended['lock_identity'] == LOCK_IDENTITY
            and (ended['completed_layouts'], ended['completed_calls'], ended['compiled_graphs']) == (1, 7865, 7)
            and ended['geometry_qualified'] is ended['full_model_qualified'] is ended['new_speedup_claim'] is False
            and ended['reconciliation']['passed'] is False,
            'Actual independent partial failed geometry audit; never model success')
    require(final['schema'] == 'qwen35_writer_geometry_root_finalization_v1'
            and final['reviewer'] == '/root' and final['actual_exit'] == refs['ended']
            and final['actual_outer_tool'] == ACTUAL_OUTER_TOOL
            and final['complete_old_catalog'] == BASE_CATALOG and final['catalog'] == refs['catalog']
            and final['idle_handoff'] == catalog['current_handoff']
            and final['source_composition'] == COMPOSITION
            and final['all_prior163_rows_unchanged'] is final['original_8MiB_cap_preserved'] is True
            and final['total_rows'] == 164 and final['original_lock_identity'] == LOCK_IDENTITY
            and final['geometry_qualified'] is final['full_model_qualified'] is final['new_speedup_claim'] is False,
            'Actual complete root finalization and exact source composition')
    audit = final['actual_independent_auditor_tool']
    require(set(audit) == {'kind', 'session_id', 'terminal_chunk', 'exit_code'}
            and audit['kind'] == 'tool_session' and audit['session_id'] == 9791 and audit['exit_code'] == 0
            and type(audit['terminal_chunk']) is str and re.fullmatch('[a-f0-9]{3,64}', audit['terminal_chunk']),
            'Actual independent auditor terminal must already be known')
    require({r['path']: r['sha256'] for r in final['external_sources']} == ROOT_HELPERS
            and len(final['external_sources']) == 3, 'Exact three externally reviewed root helpers')
    drains = final['two_current_drain_observations']
    require(type(drains) is list and len(drains) == 2 and drains[0] == drains[1]
            and set(drains[0]) == set(row['units']), 'Two complete finalizer drain observations')
    return values, copy.deepcopy(refs)


AUTHENTICATOR_BODY = '''

def authenticate_completed_writer(catalog_ref, read, guard):
    previous.previous.previous.codec.reference(catalog_ref); guard()
    historical = read(HISTORICAL_CATALOG)
    base = read(previous.HISTORICAL_CATALOG)
    current = historical if catalog_ref == HISTORICAL_CATALOG else read(catalog_ref)
    require(historical['schema'] == 'minicpm5_operator_later_serial_metadata_catalog_v1'
        and historical['reviewer'] == '/root' and historical['passed'] is True and historical['blockers'] == []
        and len(base['runs']) == 163 and len(historical['runs']) == 164
        and historical['runs'][:163] == base['runs'] and historical['completed_sdk'] == base['completed_sdk']
        and type(current['runs']) is list and len(current['runs']) >= 164
        and current['runs'][:164] == historical['runs'], 'Entire exact164 completed historical prefix retained')
    require(historical['runs'][163] == ENDED_ROW, 'Whole exact ended row including all75 source pins')
    ended = read(ACTUAL_EXIT)
    require(ended['schema'] == 'qwen35_writer_geometry_root_actual_exit_v1' and ended['reviewer'] == '/root'
        and ended['actual_tool'] == ENDED_ROW['tool']
        and ended['owners'] == ENDED_ROW['owners'] and ended['units'] == ENDED_ROW['units']
        and all(ended.get(k) == v for k, v in ENDED_STATUS.items()), 'Entire saved failed status and actual independent audit')
    final = read(FINALIZATION)
    require(all(final.get(k) == v for k, v in FINALIZATION_STATUS.items())
        and final['catalog'] == HISTORICAL_CATALOG and final['actual_exit'] == ACTUAL_EXIT
        and final['idle_handoff'] == historical['current_handoff'], 'Exact original finalizer and physical completion evidence')
    require(all(ENDED_ROW['source_sha256'].get(p) == s for p, s in COMPLETED_WRITER_TEXT.items()),
            'All38 exact exceptional Python paths tied to completed row')
    guard(); return dict(COMPLETED_WRITER_TEXT)


def extension_shape(extension, *, completed_lfm_text, catalog_ref, read, guard=lambda: None):
    require(type(extension) is dict and 0 < len(extension) <= 2048, 'Original bounded additive extension')
    selected = {p: s for p, s in extension.items() if p in COMPLETED_WRITER_TEXT}
    ordinary = {p: s for p, s in extension.items() if p not in COMPLETED_WRITER_TEXT}
    previous.extension_shape(ordinary, completed_lfm_text=completed_lfm_text,
        catalog_ref=catalog_ref, read=read, guard=guard)
    if selected:
        require(selected == COMPLETED_WRITER_TEXT, 'All38 exact completed writer pins, no subset or changed SHA')
        require(authenticate_completed_writer(catalog_ref, read, guard) == selected,
                'Same complete ended writer domain in prepare and actual authentication')
    return extension
'''


def domain_source(values, refs):
    ended = values['ended']; final = values['finalization']
    ended_keys = ('actual_tool_exit_observed', 'actual_outer_exit_observed', 'owners_gone', 'physically_drained',
        'original_lock_held_during_audit', 'lock_identity', 'completed_layouts', 'completed_calls', 'compiled_graphs',
        'geometry_qualified', 'full_model_qualified', 'new_speedup_claim', 'reconciliation')
    final_keys = tuple(k for k in final if k not in ('catalog', 'actual_exit', 'idle_handoff'))
    constants = dict(HISTORICAL_CATALOG=refs['catalog'], ACTUAL_EXIT=refs['ended'], FINALIZATION=refs['finalization'],
        COMPLETED_WRITER_TEXT=EXTERNAL_TEXT, ENDED_ROW=values['catalog']['runs'][163],
        ENDED_STATUS={k: ended[k] for k in ended_keys}, FINALIZATION_STATUS={k: final[k] for k in final_keys})
    header = ('"""Exact38 ended writer text pins; no arbitrary future-catalog domain permission."""\n'
              'from research import minicpm5_completed_qwen_package_text_v3 as previous\n'
              'ROOT, require = previous.ROOT, previous.require\n')
    return (header + ''.join(k + ' = ' + repr(v) + '\n' for k, v in constants.items()) + AUTHENTICATOR_BODY).encode()


def render(predecessors, inputs):
    """Return seven source byte strings with exact reversible mechanical wiring.

    Actual authenticity, path freshness and independent review are caller duties.
    No file, process, current handoff, History or model operation occurs here.
    """
    require(type(predecessors) is dict and set(predecessors) == set(PREDECESSORS), 'All six frozen predecessor sources')
    for name, sha in PREDECESSORS.items():
        require(type(predecessors[name]) is bytes and len(predecessors[name]) < 4 * 2**20
                and hashlib.sha256(predecessors[name]).hexdigest() == sha, 'Exact frozen predecessor: ' + name)
    values, refs = completed_inputs(inputs)
    recipes = {
      'prepare_minicpm5_pv_sdk_import_catalog_v5.py': ('prepare_minicpm5_pv_sdk_import_catalog_v4.py', [
        ('minicpm5_completed_qwen_package_text_v3', 'minicpm5_completed_qwen_package_text_v4')]),
      'minicpm5_fixed_input_pv_catalog_auth_v5.py': ('minicpm5_fixed_input_pv_catalog_auth_v4.py', [
        ('minicpm5_completed_qwen_package_text_v3', 'minicpm5_completed_qwen_package_text_v4')]),
      'minicpm5_sdk_mapping_outer_source_entry_v14.py': ('minicpm5_sdk_mapping_outer_source_entry_v13.py', [
        ('minicpm5_fixed_input_pv_catalog_auth_v4', 'minicpm5_fixed_input_pv_catalog_auth_v5')]),
      'prepare_minicpm5_catalog_formal_binding_v7.py': ('prepare_minicpm5_catalog_formal_binding_v6.py', [
        ('from research import prepare_minicpm5_pv_sdk_import_catalog_v4 as catalog_prepare',
         'from research import prepare_minicpm5_pv_sdk_import_catalog_v5 as catalog_prepare'),
        ("           'minicpm5_sdk_mapping_worker_source_entry_v8.py')",
         "           'minicpm5_sdk_mapping_worker_source_entry_v8.py',\n           'minicpm5_completed_qwen_package_text_v4.py',\n           'minicpm5_fixed_input_pv_catalog_auth_v5.py',\n           'minicpm5_sdk_mapping_outer_source_entry_v14.py')"),
        ("                     'prepare_minicpm5_catalog_formal_binding_v6.py')",
         "                     'prepare_minicpm5_catalog_formal_binding_v6.py',\n                     'prepare_minicpm5_pv_sdk_import_catalog_v5.py',\n                     'prepare_minicpm5_catalog_formal_binding_v7.py')"),
        ('Exactly twenty-six explicit', 'Exactly thirty-one explicit'),
        ("entry_module='research.minicpm5_sdk_mapping_outer_source_entry_v13'", "entry_module='research.minicpm5_sdk_mapping_outer_source_entry_v14'"),
        ("entry_path=str(Q / 'minicpm5_sdk_mapping_outer_source_entry_v13.py')", "entry_path=str(Q / 'minicpm5_sdk_mapping_outer_source_entry_v14.py')")]),
      'prepare_minicpm5_formal_v7_metadata_root_v4.py': ('prepare_minicpm5_formal_v7_metadata_root_v3.py', [
        ('prepare_minicpm5_catalog_formal_binding_v6', 'prepare_minicpm5_catalog_formal_binding_v7'),
        ('prepare_minicpm5_formal_v7_metadata_root_v3', 'prepare_minicpm5_formal_v7_metadata_root_v4'),
        ('accept_minicpm5_formal_v7_metadata_root_v3', 'accept_minicpm5_formal_v7_metadata_root_v4'),
        ("           'accept_minicpm5_formal_v7_metadata_root_v2.py')",
         "           'accept_minicpm5_formal_v7_metadata_root_v2.py',\n           'prepare_minicpm5_formal_v7_metadata_root_v3.py',\n           'accept_minicpm5_formal_v7_metadata_root_v3.py')")]),
      'accept_minicpm5_formal_v7_metadata_root_v4.py': ('accept_minicpm5_formal_v7_metadata_root_v3.py', [
        ('prepare_minicpm5_formal_v7_metadata_root_v3', 'prepare_minicpm5_formal_v7_metadata_root_v4'),
        ('accept_minicpm5_formal_v7_metadata_root_v3', 'accept_minicpm5_formal_v7_metadata_root_v4'),
        ('minicpm5_fixed_input_pv_catalog_auth_v4', 'minicpm5_fixed_input_pv_catalog_auth_v5'),
        ('minicpm5_sdk_mapping_outer_source_entry_v13', 'minicpm5_sdk_mapping_outer_source_entry_v14'),
        ("and len(runtime) == 191, 'Complete retained186 plus five exact transport sources on both interpreters'",
         "and len(runtime) == 194, 'Complete retained191 plus three exact completed-writer sources on both interpreters'")])}
    files = {str(Q/'minicpm5_completed_qwen_package_text_v4.py'): domain_source(values, refs)}
    ledgers = {}
    for name, (old, replacements) in recipes.items():
        original = predecessors[old].decode(); current = original; ledger = []
        for before, after in replacements:
            count = current.count(before); require(count > 0, 'Exact predecessor replacement must exist')
            current = current.replace(before, after); ledger.append(dict(before=before, after=after, count=count))
        restored = current
        for change in reversed(ledger):
            require(restored.count(change['after']) == change['count'], 'Unambiguous exact reversal')
            restored = restored.replace(change['after'], change['before'])
        require(restored == original, 'Entire original source exactly recoverable')
        files[str(Q/name)] = current.encode(); ledgers[str(Q/name)] = dict(predecessor=old, edits=ledger)
    for raw in files.values():
        require(len(raw) < 4 * 2**20, 'Original source text cap'); ast.parse(raw)
    return dict(schema='minicpm5_completed_writer_source_candidate_v1', source_only=True, source_accepted=False,
        launch_authorized=False, actual_inputs_independently_verified=False, history_executed=False,
        native_executed=False, files=files, exact_predecessor_reversal=ledgers, completed_refs=refs,
        sdk_worker_unchanged='minicpm5_sdk_mapping_worker_source_entry_v8.py',
        reconstruction_unchanged='minicpm5_saved_operator_reconstruction_v2.py',
        expected_bootstrap_sources=194, expected_catalog_source_refs=31, expected_helper_source_refs=9,
        accepted_new_text_domain=copy.deepcopy(EXTERNAL_TEXT),
        later_unknown_root_helper_paths_accepted=False)
