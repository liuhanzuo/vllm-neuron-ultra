"""Exact twelve-call select diagnostic; all checks are stdlib and import-inert."""
from pathlib import Path
import struct

from research import qwen35_slot_select_bitcast_proposal_v1 as proposal
from research import qwen35_slot_reference_canonicalization_evidence_v1 as original
from research.qwen35_slot_special_value_protocol_v2 import sha, file_ref, checked, write_raw, differences
from research.qwen35_slot_select_probe_hlo_v1 import hlo_proof
from qwen35_sampler_runtime_contract_v12 import require
from qwen35_sampler_runtime_io_v12 import identity, raw
import qwen35_predicated_slot_protocol_v1 as frozen

Q = Path(__file__).resolve().parent
NAME = 'conv-select-probes'
ARMS, PROFILES, MAPPINGS = proposal.ARMS, proposal.PROFILES, proposal.MAPPINGS
SHAPE = list(proposal.SHAPE)
BYTES = 98304
PAYLOAD_CAP = 8 * 2**20
ARTIFACT_CAP = 128 * 2**20
INPUTS = (
    dict(path=str(Q/'qwen35-slot-special-value-r2-root-actual-exit-20261112-v1.json'),
         sha256='24536a19e7fcef6a8d13e17ab1f10749916b2e4ca057076a729f94b4904e09fd'),
    dict(path=str(Q/'qwen35-slot-special-value-r2-independent-raw-review-20261112-v1.json'),
         sha256='1cb3e38eb2d0dec4fad72f4f027d34c2e8437814fa15f0f7806bf5ec41094b23'),
    dict(path=str(Q/'qwen35-slot-select-hlo-mechanism-source-review-20261112-v1.json'),
         sha256='7336600d22f3f9b9235bb26fd08ea8af2502bb3c62cb198e91fc6d78b091829d'))


def dtype(arm):
    require(arm in ARMS, 'Exact diagnostic arm')
    return 'torch.int16' if arm == 'external_s16_select' else 'torch.bfloat16'


def input_data(profile):
    require(profile in PROFILES, 'Only two original raw profiles')
    return dict(cache=original.input_bytes(0, negative_nan=profile == 'negative_nan_raw'),
                updates=original.input_bytes(17, negative_nan=profile == 'negative_nan_raw'))


def slots(mapping):
    values = dict(MAPPINGS)
    require(mapping in values, 'Only idle and single active mapping')
    return struct.pack('<4q', *values[mapping])


def inputs(profile, mapping):
    return dict(input_data(profile), slots=slots(mapping))


def expected(profile, mapping):
    data = input_data(profile)
    return proposal.byte_oracle(data['cache'], data['updates'], dict(MAPPINGS)[mapping])


def schedule():
    return proposal.schedule()


def workload():
    return dict(schema='qwen35_slot_select_probe_workload_v1', name=NAME, shape=SHAPE,
        arms=list(ARMS), dtypes={arm: dtype(arm) for arm in ARMS}, profiles=list(PROFILES),
        mappings={name: list(values) for name, values in MAPPINGS}, schedule=schedule(),
        calls=12, graphs=3, all_graphs_compiled_fresh=True, existing_graphs_called=0,
        compiler_args=list(frozen.FLAGS), diagnostic_inputs=list(INPUTS), raw_payload_cap=PAYLOAD_CAP,
        max_live_graphs=1, original_eight_layout_acceptance_unchanged=True,
        qualification_matrix_completed=False, full_model_qualified=False, model_speedup_claim=False)


def verify_inputs():
    for reference in INPUTS:
        require(sha(raw(reference['path'], 16*2**20)) == reference['sha256'], 'Frozen prior diagnostic metadata unchanged')


def prefix(root, arm, profile, mapping):
    require(arm in ARMS and profile in PROFILES and mapping in dict(MAPPINGS), 'Fixed call identity')
    return Path(root)/(profile+'-'+mapping+'-'+arm)


def record_call(root, arm, profile, mapping, actual, actual_shape, actual_dtype, after, guard):
    require(actual_shape == SHAPE and actual_dtype == dtype(arm), 'Actual exact arm output shape/dtype')
    before = inputs(profile, mapping)
    require(set(after) == set(before), 'All three actual input storages checked')
    positions, offsets = differences(actual, expected(profile, mapping))
    guard('before_capture:'+arm+':'+profile+':'+mapping)
    base = prefix(root, arm, profile, mapping)
    output = write_raw(str(base)+'-output.raw', actual, BYTES)
    mismatch = write_raw(str(base)+'-differences.u32', offsets, 4*BYTES)
    row = dict(arm=arm, profile=profile, mapping=mapping, slots=list(dict(MAPPINGS)[mapping]),
        shape=actual_shape, dtype=actual_dtype, actual_call_completed=True, output=output, differences=mismatch,
        expected_sha256=sha(expected(profile, mapping)), different_bytes=len(positions),
        first16_different_bytes=positions[:16], bitwise_equal=not positions,
        input_unchanged={key: after[key] == before[key] for key in before},
        input_after_sha256={key: sha(after[key]) for key in before})
    guard('after_capture:'+arm+':'+profile+':'+mapping)
    return row


def validate_graph(graph, root):
    arm = graph['arm']
    require(arm in ARMS, 'Three declared fresh graph arms')
    directory = Path(root)/'artifacts'/arm
    artifact, hlo = graph['artifact'], graph['hlo']
    require(artifact['path'] == str(directory/'model.pt') and hlo['path'] == str(directory/'model/graph.hlo'), 'Exact fresh graph namespace')
    checked(artifact)
    require(graph['hlo_proof'] == hlo_proof(checked(hlo, 16*2**20), arm), 'Full actual graph input/call/output HLO dataflow')
    require(graph['load_before'] == graph['load_after'] == graph['post_calls'] == artifact,
            'Same complete graph bytes before/after load and all four calls')
    require(identity(artifact['path']) == graph['load_identity'], 'Same actual loaded artifact identity')
    require(graph['device_core'] == 0 and graph['moved_to_device'] is True, 'Actual fixed core move')
    code_ref = graph['forward_code']
    require(code_ref['path'] == str(Path(root)/(arm+'-forward.txt')), 'Exact actual forward evidence')
    code = checked(code_ref, 65536)
    require(b'ops.neuron.forward_v2' in code or b'neuron::forward_v2' in code, 'Loaded compiled Neuron invocation; no host fallback')


def validate_report(report, root):
    root = Path(root)
    require(report['schema'] == 'qwen35_slot_select_probe_result_v1' and report['complete'] is True
            and report['error'] is None and report['workload'] == workload() and report['runtime_calls'] == 12
            and all(report[key] is False for key in ('qualification_matrix_completed', 'full_model_qualified', 'model_speedup_claim')),
            'Complete twelve-call diagnostic only; no primitive/model qualification')
    require(set(report['inputs']) == set(PROFILES) and set(report['slots']) == set(dict(MAPPINGS)), 'All whole raw inputs present')
    total = 0
    for profile in PROFILES:
        require(set(report['inputs'][profile]) == {'cache', 'updates'}, 'Both full payloads')
        for key, payload in input_data(profile).items():
            reference = report['inputs'][profile][key]
            require(reference['path'] == str(root/(profile+'-'+key+'.raw'))
                    and checked(reference, BYTES) == payload, 'Actual full saved deterministic input bytes')
            total += reference['bytes']
    for mapping, _ in MAPPINGS:
        reference = report['slots'][mapping]
        require(reference['path'] == str(root/(mapping+'-slots.raw')) and checked(reference, 32) == slots(mapping), 'Actual whole slot bytes')
        total += reference['bytes']
    require([row['arm'] for row in report['graphs']] == list(ARMS), 'Exactly three newly compiled called graphs')
    for graph in report['graphs']:
        validate_graph(graph, root)
    require([{key: row[key] for key in ('arm', 'profile', 'mapping', 'slots')} for row in report['calls']] == schedule(),
            'Complete ordered 3x2x2 actual calls; mismatches cannot skip later arms')
    for row in report['calls']:
        arm, profile, mapping = (row[key] for key in ('arm', 'profile', 'mapping'))
        require(row['actual_call_completed'] is True and row['shape'] == SHAPE and row['dtype'] == dtype(arm), 'Exact actual output')
        base = prefix(root, arm, profile, mapping)
        require(row['output']['path'] == str(base)+'-output.raw'
                and row['differences']['path'] == str(base)+'-differences.u32', 'Exact unique full output/offset paths')
        actual = checked(row['output'], BYTES)
        positions, offsets = differences(actual, expected(profile, mapping))
        require(checked(row['differences'], 4*BYTES) == offsets and row['different_bytes'] == len(positions)
                and row['first16_different_bytes'] == positions[:16] and row['bitwise_equal'] == (not positions)
                and row['expected_sha256'] == sha(expected(profile, mapping)), 'All differing byte positions independently recomputed')
        before = inputs(profile, mapping)
        require(row['input_unchanged'] == dict.fromkeys(before, True)
                and row['input_after_sha256'] == {key: sha(payload) for key, payload in before.items()}, 'All actual inputs remain bitwise identical')
        total += row['output']['bytes'] + row['differences']['bytes']
    require(total <= PAYLOAD_CAP, 'Whole raw and complete offset payload budget')
    return dict(layout=NAME, diagnostic_complete=True, calls=12, graphs=3,
        bitwise_matches={arm: {profile: {mapping: next(row['bitwise_equal'] for row in report['calls']
            if (row['arm'],row['profile'],row['mapping']) == (arm,profile,mapping))
            for mapping, _ in MAPPINGS} for profile in PROFILES} for arm in ARMS},
        qualification_matrix_completed=False, full_model_qualified=False, model_speedup_claim=False)
