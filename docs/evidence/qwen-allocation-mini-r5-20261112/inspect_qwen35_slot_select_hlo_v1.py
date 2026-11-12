"""Inspect one explicitly authorized reference HLO and pinned text sources only."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
from qwen35_slot_special_value_protocol_v2 import _protobuf_fields, _field_values, _varint
from qwen35_slot_select_bitcast_proposal_v1 import schedule

SITE = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
HLO = Path('/mnt/nvme/neuron-work/qwen35-predicated-slot-r1-v3-20261112/conv/artifacts/standalone-reference/model/graph.hlo')
HLO_SHA = 'be083f7f728dcdb6a744fbacadf840c647ab06b5eff0b1355004cdc573c67d5e'


def reference(path):
    raw = Path(path).read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def integers(fields, number):
    result = []
    for field, wire, value in fields:
        if field != number:
            continue
        assert wire in (0, 2)
        if wire == 0:
            result.append(value)
        else:
            position = 0
            while position < len(value):
                item, position = _varint(value, position)
                result.append(item)
    return result


def descriptor(path):
    nodes = [node for node in ast.walk(ast.parse(path.read_bytes())) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'AddSerializedFile']
    assert len(nodes) == 1
    return _protobuf_fields(ast.literal_eval(nodes[0].args[0]))


def evidence():
    hlo_ref = reference(HLO)
    assert hlo_ref['bytes'] == 4084 and hlo_ref['sha256'] == HLO_SHA
    # Verify field numbering and primitive names from installed descriptor text,
    # without importing generated protobuf modules or SDK packages.
    definitions = {}
    for raw in _field_values(descriptor(SITE / 'torch_neuronx/pyhlo/service/hlo_pb2.py'), 4, 2):
        message = _protobuf_fields(raw)
        name = _field_values(message, 1, 2)[0].decode()
        definitions[name] = {}
        for item in _field_values(message, 2, 2):
            fields = _protobuf_fields(item)
            definitions[name][_field_values(fields, 1, 2)[0].decode()] = integers(fields, 3)[0]
    assert definitions['HloModuleProto']['computations'] == 3
    assert definitions['HloComputationProto']['instructions'] == 2
    assert {key: definitions['HloInstructionProto'][key] for key in ('name', 'opcode', 'shape', 'id', 'operand_ids')} == dict(name=1, opcode=2, shape=3, id=35, operand_ids=36)
    primitive_types = {}
    for raw in _field_values(descriptor(SITE / 'torch_neuronx/pyhlo/xla_data_pb2.py'), 5, 2):
        enum = _protobuf_fields(raw)
        if _field_values(enum, 1, 2) != [b'PrimitiveType']:
            continue
        for item in _field_values(enum, 2, 2):
            fields = _protobuf_fields(item)
            primitive_types[integers(fields, 2)[0]] = _field_values(fields, 1, 2)[0].decode()
    computations = _field_values(_protobuf_fields(HLO.read_bytes()), 3, 2)
    assert len(computations) == 1
    comp = _protobuf_fields(computations[0])
    rows = []
    for raw in _field_values(comp, 2, 2):
        fields = _protobuf_fields(raw)
        shape = _protobuf_fields(_field_values(fields, 3, 2)[0])
        rows.append(dict(id=integers(fields, 35)[0], name=_field_values(fields, 1, 2)[0].decode(),
            opcode=_field_values(fields, 2, 2)[0].decode(), dtype=primitive_types[integers(shape, 2)[0]],
            shape=integers(shape, 3), operands=integers(fields, 36),
            comparison_direction=[x.decode() for x in _field_values(fields, 63, 2)],
            comparison_type=[x.decode() for x in _field_values(fields, 72, 2)]))
    counts = dict(Counter(row['opcode'] for row in rows))
    assert len(rows) == 61 and counts == dict(parameter=3, broadcast=24, slice=8, reshape=16,
        constant=1, compare=4, select=4, tuple=1)
    selected = [row for row in rows if row['opcode'] == 'select']
    assert [row['id'] for row in selected] == [18, 32, 46, 60]
    assert all(row['dtype'] == 'BF16' and row['shape'] == [4, 3072, 4] for row in selected)
    assert all(row['comparison_direction'] == ['EQ'] and row['comparison_type'] == ['SIGNED']
               for row in rows if row['opcode'] == 'compare')
    sources = [Q.parent / name for name in ('qwen35_text_port.py', 'qwen35_predicated_slot_protocol_v1.py',
        'qwen35_predicated_slot_device_leaf_v1.py', 'slot_cache_nki_predicated.py', 'slot_cache_nki_predicated_torch.py')]
    sources += [SITE / name for name in ('torch_neuronx/pyhlo/service/hlo_pb2.py', 'torch_neuronx/pyhlo/xla_data_pb2.py',
        'torch_neuronx/pyhlo/scribe.py', 'torch_neuronx/xla_impl/base.py', 'torch_neuronx/xla_impl/hlo_conversion.py',
        'torch_neuronx/xla_impl/trace.py', 'torch_xla/core/xla_builder.py', 'torch_xla/core/xla_op_registry.py',
        'torch_xla/__init__.py')]
    sources += [Q / name for name in ('qwen35_slot_special_value_protocol_v2.py',
        'qwen35_slot_select_bitcast_proposal_v1.py', 'check_qwen35_slot_select_bitcast_proposal_v1.py',
        'inspect_qwen35_slot_select_hlo_v1.py')]
    prior = Q / 'qwen35-slot-special-value-r2-independent-raw-review-20261112-v1.json'
    protocol = ast.parse((Q.parent / 'qwen35_predicated_slot_protocol_v1.py').read_text())
    flags = next(ast.literal_eval(node.value) for node in protocol.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'FLAGS' for target in node.targets))
    return dict(schema='qwen35_slot_select_hlo_mechanism_source_review_v1',
        reviewer='/root/qwen_slot_diagnostic_source_review_20261112', source_only=True,
        native_executed=False, compiler_executed=False, payloads_read='Only the authorized 4084-byte HLO metadata; no weights/model.pt/NEFF/tensor payload.',
        current_bindings_history_locks_or_processes_inspected=False, reference_hlo=hlo_ref,
        prior_actual_six_call_raw_audit=reference(prior), source_files=[reference(path) for path in sources],
        hlo=dict(computations=1, instructions=61, opcode_counts=counts, full_instruction_summary=rows,
            bf16_select_ids=[18, 32, 46, 60], convert_instructions=0, bitcast_convert_instructions=0,
            floating_point_arithmetic_instructions=0, input_parameters=[dict(parameter=0,dtype='BF16',shape=[4,3072,4]),
                dict(parameter=1,dtype='BF16',shape=[4,3072,4]),dict(parameter=2,dtype='S64',shape=[4])],
            scope='SDK setup_compiler_dirs serializes this HLO before hlo_compile invokes neuronx-cc; not the compiler final machine IR.'),
        mechanism_findings=dict(
            established='The frozen torch.where loop lowers to four BF16 selects with BF16 branch storage; no explicit conversion or floating arithmetic appears in the saved compiler-input HLO.',
            excluded_at_this_boundary='An explicit frontend BF16-to-FP32-to-BF16 convert chain is absent.',
            still_hypothesis='BF16 select lowering, optimization or its graph-specific runtime implementation normalizes signed zero and NaN; no individual compiler pass is identified.',
            fast_math=dict(original_compiler_args=list(flags),cpu_xla_flag='--xla_cpu_enable_fast_math=false',
                cpu_flag_is_neuron_backend_math_contract=False,
                observed_args_have_explicit_fast_math_switch=False,
                auto_cast_none_proves_bitwise_select_semantics=False,
                supported_neuron_fast_math_disable_flag_established_by_read_python_sources=False,
                recommendation='Keep all original compiler flags in the next probe. Do not invent a backend fast-math flag or claim -O2 is the cause.')),
        sdk_source_support=dict(xla_call='torch_neuronx.xla_impl.base.xla_call -> xla_op_registry.register',
            reinterpretation='torch_xla.core.xla_builder.Op.bitcast -> mkop(BitcastConvert)',
            numeric_conversion_distinct='Op.cast -> mkop(Convert)',signed16_type_maps_present=True,
            prototype_uses_torch_dtype_view=False,backend_compilation_and_bitwise_fidelity_unproven=True),
        proposal=dict(source=str(Q/'qwen35_slot_select_bitcast_proposal_v1.py'),graphs=3,calls=12,schedule=schedule(),
            compiler_flags='Exactly original FLAGS, no flag sweep',profiles='Original fixed original_raw and negative_nan_raw, unchanged salts and complete bytes',
            oracle='Idle: exact cache; one_active: cache slots0..2 unchanged, output slot3 equals update row0; all 98304 bytes checked.',
            external_integer_io='Construct cache/updates from the same raw bytes using uint8 storage viewed as torch.int16. No numeric cast. Slots remain int64. Save output as raw int16 bytes with explicit dtype.',
            internal_bitcast_hlo_gate='Trace both BF16 inputs through BitcastConvert to S16, all four selects in S16, final BitcastConvert to BF16. Follow called computations; reject numerical Convert or a rewritten BF16-select path.',
            single_select_hlo_gate='Exactly one BF16 select with runtime slot predicate; no floating arithmetic/Convert.',
            external_integer_hlo_gate='S16 input/output payload and four S16 selects; no BF16 select or numeric conversion.',
            invocation_gate='Actual loaded forward_v2 invocation, same fixed core, stable saved artifact hash and complete output; reject a host passthrough.',
            capture='Save full inputs, every output, all mismatch u32 offsets, input-after hashes and whole HLO. Continue other calls after output mismatch; exceptions abort, no fallback/retry.',
            lifecycle='New explicitly bound successor only, original owner/SDKEnvironment/resources/drain, max one live graph; no launch implemented here.',
            full_raw_offset_cap_bytes=8*2**20, measured_calls=0,
            independent_diagnostic_not_baseline_replacement=True,
            followup_qualification='Even a passing integer variant must separately pass all original eight layouts, both modes, all625 mappings, raw+finite profiles, persistent state/old-read/input checks and original timing schedule. The failed original reference remains failed.'),
        source_mock_tests=dict(command='/home/ubuntu/venvs/vllm_neuron/bin/python -I -B '+str(Q/'check_qwen35_slot_select_bitcast_proposal_v1.py'),
            tests=5,terminal_chunk='822bde',exit_code=0,scope='12 fake-SDK calls through actual proposal and frozen writer AST; all65536 host integer encodings; no native validation.'),
        qualification_matrix_completed=False, primitive_qualified=False, full_model_qualified=False, performance_claimed=False)


if __name__ == '__main__':
    print(json.dumps(evidence(), indent=2, sort_keys=True, allow_nan=False))
