"""Strict stdlib dataflow proof for the three fixed select-probe HLO modules."""
import hashlib
import json
import math

from qwen35_sampler_runtime_contract_v12 import require
from research.qwen35_slot_special_value_protocol_v2 import _protobuf_fields as fields, _field_values as values, _varint

TYPES = {1: 'PRED', 3: 'S16', 5: 'S64', 13: 'TUPLE', 16: 'BF16'}
SHAPE = (4, 3072, 4)
ARMS = ('one_bf16_select', 'internal_s16_select', 'external_s16_select')
ALLOWED = {'parameter', 'constant', 'slice', 'reshape', 'broadcast', 'copy',
           'compare', 'select', 'bitcast-convert', 'call', 'tuple', 'get-tuple-element'}


def ints(fs, number):
    result = []
    for field, wire, value in fs:
        if field != number:
            continue
        require(wire in (0, 2), 'Integer HLO field wire type')
        if wire == 0:
            result.append(value)
        else:
            position = 0
            while position < len(value):
                item, position = _varint(value, position)
                result.append(item)
    return result


def scalar(fs, number, default=None):
    selected = values(fs, number, 0)
    require(len(selected) <= 1, 'Unique scalar HLO field')
    require(bool(selected) or default is not None, 'Required scalar HLO field')
    return selected[0] if selected else default


def blob(fs, number):
    selected = values(fs, number, 2)
    require(len(selected) == 1, 'One required length-delimited HLO field')
    return selected[0]


def shape(raw):
    fs = fields(raw)
    kind = scalar(fs, 2)
    require(kind in TYPES, 'Only declared probe primitive types')
    dims = tuple(ints(fs, 3))
    require(len(dims) <= 4 and all(0 < n <= 49152 for n in dims), 'Bounded positive HLO shape')
    children = tuple(shape(value) for value in values(fs, 4, 2))
    dynamic = ints(fs, 6)
    require(not any(dynamic) and (not dynamic or len(dynamic) == len(dims)), 'Static shapes only')
    require((not dims and bool(children)) if kind == 13 else not children, 'Exact tuple/array shape')
    return (TYPES[kind], dims, children)


def value(kind, dims, semantic):
    return ((kind, tuple(dims), ()), semantic)


def cast_semantic(semantic, target):
    if semantic[0] == 'leaf':
        return semantic[:3] + (semantic[3] + (target,),)
    require(semantic[0] == 'select', 'Bitcast only on payload data')
    return semantic[:3] + (cast_semantic(semantic[3], target), cast_semantic(semantic[4], target))


def hlo_proof(payload, arm):
    require(arm in ARMS, 'One declared select-probe arm')
    module = fields(payload)
    entry = scalar(module, 6)
    computations = {}
    total = 0
    for raw in values(module, 3, 2):
        fs = fields(raw)
        cid, root = scalar(fs, 5), scalar(fs, 6)
        require(cid not in computations, 'Unique computation identity')
        instructions = {}
        parameters = {}
        for raw_instruction in values(fs, 2, 2):
            item = fields(raw_instruction)
            identifier = scalar(item, 35)
            opcode = blob(item, 2).decode('ascii')
            require(identifier not in instructions and opcode in ALLOWED,
                    'Unique instruction and no numeric Convert/custom-call/arithmetic')
            require(not values(item, 28, 2) and not values(item, 43, 2), 'No detached backend target/config')
            require(not ints(item, 37), 'No control-only payload paths')
            instructions[identifier] = dict(op=opcode, shape=shape(blob(item, 3)),
                operands=ints(item, 36), calls=ints(item, 38), fields=item)
            if opcode == 'parameter':
                number = scalar(item, 9, 0)
                require(number not in parameters, 'Unique computation parameters')
                parameters[number] = identifier
            total += 1
            require(total <= 65536, 'Bounded complete HLO instructions')
        require(root in instructions and set(parameters) == set(range(len(parameters))), 'Complete root and parameter order')
        computations[cid] = (root, instructions, parameters)
    require(entry in computations and 0 < len(computations) <= 64, 'Bounded actual entry computation')
    kind = 'S16' if arm == 'external_s16_select' else 'BF16'
    arguments = [value(kind, SHAPE, ('leaf', 'cache', None, (kind,))),
                 value(kind, SHAPE, ('leaf', 'updates', None, (kind,))),
                 value('S64', (4,), ('slots',))]
    visited = set()
    select_count = 0
    bitcast_count = 0
    invocations = 0

    def computation(cid, args, stack):
        nonlocal select_count, bitcast_count, invocations
        require(cid in computations and cid not in stack and len(stack) < 16, 'Acyclic bounded called-computation path')
        invocations += 1
        require(invocations <= 128, 'Bounded dynamic computation expansion')
        root, instructions, parameters = computations[cid]
        require(len(args) == len(parameters), 'Actual call parameter count')
        memo = {}
        active = set()

        def evaluate(identifier):
            nonlocal select_count, bitcast_count
            if identifier in memo:
                return memo[identifier]
            require(identifier in instructions and identifier not in active, 'Complete acyclic operand dataflow')
            active.add(identifier)
            item = instructions[identifier]
            op, declared, fs = item['op'], item['shape'], item['fields']
            operands = [evaluate(index) for index in item['operands']]
            require(op == 'call' or not item['calls'], 'Called computation belongs only to actual call')
            if op == 'parameter':
                require(not operands, 'Parameter has no operands')
                result = args[scalar(fs, 9, 0)]
            elif op == 'constant':
                require(not operands and declared == ('S64', (4,), ()), 'Only original S64 destination indices constant')
                literal = fields(blob(fs, 8))
                # Fixed SDK LiteralProto.s64s=5; no floating-point constants.
                require(shape(blob(literal, 1)) == declared and ints(literal, 5) == [0, 1, 2, 3],
                        'Exact literal shape and destination indices [0,1,2,3]')
                result = value('S64', (4,), ('indices',))
            elif op == 'call':
                require(len(item['calls']) == 1, 'Exactly one actual callee')
                result = computation(item['calls'][0], operands, stack + (cid,))
            elif op == 'tuple':
                require(declared == ('TUPLE', (), tuple(item[0] for item in operands)), 'Exact tuple member shapes')
                result = (declared, ('tuple', tuple(operands)))
            elif op == 'get-tuple-element':
                require(len(operands) == 1 and operands[0][1][0] == 'tuple', 'Actual tuple projection')
                index = scalar(fs, 13, 0)
                require(index < len(operands[0][1][1]), 'Tuple projection in range')
                result = operands[0][1][1][index]
            elif op == 'bitcast-convert':
                require(len(operands) == 1, 'One bitcast operand')
                source, semantic = operands[0]
                require({source[0], declared[0]} == {'BF16', 'S16'} and source[1:] == declared[1:],
                        'Equal-width BF16/S16 reinterpretation only')
                bitcast_count += 1
                result = (declared, cast_semantic(semantic, declared[0]))
            elif op in ('reshape', 'broadcast', 'copy', 'slice'):
                require(len(operands) == 1, 'One shape/storage operand')
                source, semantic = operands[0]
                require(source[0] == declared[0] and not source[2] and not declared[2], 'Storage transforms preserve dtype')
                if op == 'reshape':
                    require(math.prod(source[1]) == math.prod(declared[1]), 'Reshape preserves all elements')
                elif op == 'copy':
                    require(source == declared, 'Copy preserves shape')
                elif op == 'broadcast':
                    dimensions = ints(fs, 14)
                    require(len(dimensions) == len(source[1]) and dimensions == sorted(set(dimensions))
                            and all(i < len(declared[1]) for i in dimensions)
                            and all(source[1][j] == declared[1][i] for j, i in enumerate(dimensions)), 'Exact broadcast dimensions')
                    if math.prod(source[1]) != math.prod(declared[1]):
                        # Shape equality alone does not prove element correspondence:
                        # the trailing payload axis also has length four. Accept only
                        # the three original expansions; same-size broadcasts merely
                        # insert singleton axes and preserve the flat element order.
                        scalar_slot = (semantic[0] == 'slot' and math.prod(source[1]) == 1
                                       and declared[1] == (4,))
                        row_payload = (semantic[:2] == ('leaf', 'update_row')
                                       and source[1] == (3072, 4) and declared[1] == SHAPE
                                       and dimensions == [1, 2])
                        slot_predicate = (semantic[0] == 'condition' and source[1] == (4,)
                                          and declared[1] == SHAPE and dimensions == [0])
                        require(scalar_slot or row_payload or slot_predicate,
                                'Exact scalar-slot, whole update-row or destination-axis predicate expansion')
                else:
                    ranges = []
                    for encoded in values(fs, 17, 2):
                        selection = fields(encoded)
                        ranges.append((scalar(selection, 1, 0), scalar(selection, 2), scalar(selection, 3)))
                    require(len(ranges) == len(source[1]) and bool(ranges), 'Complete exact slice dimensions')
                    row = ranges[0][0]
                    require(0 <= row < 4 and ranges[0] == (row, row+1, 1)
                            and ranges[1:] == [(0, n, 1) for n in source[1][1:]]
                            and declared[1] == (1,) + source[1][1:], 'Exactly one complete source row selected')
                    if semantic == ('slots',):
                        require(source[1] == (4,), 'Original four slot inputs')
                        semantic = ('slot', row)
                    else:
                        require(source[1] == SHAPE and semantic[:3] == ('leaf', 'updates', None), 'Slice comes from actual updates input')
                        semantic = ('leaf', 'update_row', row, semantic[3])
                result = (declared, semantic)
            elif op == 'compare':
                require(len(operands) == 2 and all(item[0] == ('S64', (4,), ()) for item in operands)
                        and declared == ('PRED', (4,), ())
                        and values(fs, 63, 2) == [b'EQ'] and values(fs, 72, 2) == [b'SIGNED'], 'Original signed slot equality')
                left, right = (item[1] for item in operands)
                if right == ('indices',):
                    left, right = right, left
                require(left == ('indices',) and right[0] == 'slot', 'Predicate depends on destination indices and actual slot input')
                result = (declared, ('condition', right[1]))
            else:
                require(op == 'select' and len(operands) == 3, 'Exactly three select operands')
                predicate, yes, no = operands
                require(predicate[0] == ('PRED', SHAPE, ()) and predicate[1][0] == 'condition'
                        and yes[0] == no[0] == declared and declared[1] == SHAPE
                        and declared[0] in ('BF16', 'S16'), 'Whole-payload select has exact dtype/shape')
                select_count += 1
                result = (declared, ('select', predicate[1][1], declared[0], yes[1], no[1]))
            require(result[0] == declared, 'Actual operand/callee result matches declared instruction shape')
            active.remove(identifier)
            visited.add((cid, identifier))
            memo[identifier] = result
            return result

        return evaluate(root)

    output = computation(entry, arguments, ())
    if output[1][0] == 'tuple':
        require(len(output[1][1]) == 1, 'Exactly one model output')
        output = output[1][1][0]
    require(output[0] == (kind, SHAPE, ()), 'Actual full output type')
    trace = ('BF16', 'S16', 'BF16') if arm == 'internal_s16_select' else (kind,)
    selection_type = 'BF16' if arm == 'one_bf16_select' else 'S16'
    expected = ('leaf', 'cache', None, trace)
    count = 1 if arm == 'one_bf16_select' else 4
    for row in range(count):
        expected = ('select', row, selection_type, ('leaf', 'update_row', row, trace), expected)
    require(output[1] == expected and select_count == count, 'Exact reachable selection chain and per-input bitcast paths')
    require(bitcast_count == (3 if arm == 'internal_s16_select' else 0), 'Exact expanded input/output bitcast operations')
    require(visited == {(cid, identifier) for cid, (_, items, _) in computations.items() for identifier in items},
            'Every saved computation/instruction belongs to verified output dataflow')
    return dict(schema='qwen35_slot_select_hlo_proof_v1', arm=arm,
        hlo_sha256=hashlib.sha256(payload).hexdigest(), computations=len(computations), instructions=total,
        expanded_computation_invocations=invocations, reachable_selects=select_count,
        expanded_bitcast_converts=bitcast_count, select_dtype=selection_type, input_output_dtype=kind,
        exact_actual_input_and_slot_dataflow=True, numeric_converts=0, custom_calls=0,
        output_semantics_sha256=hashlib.sha256(json.dumps(expected, separators=(',', ':')).encode()).hexdigest())
