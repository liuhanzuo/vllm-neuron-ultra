"""Fixed KV256 assembly HLO proof; stdlib-only, no IO or native imports."""
import base64
import hashlib
import json
import math

from qwen35_kv256_alias_diagnostic_bytes_v1 import require

def _varint(payload,position):
    """Bounded canonical uint64; reject truncation, overflow and overlong forms."""
    value=0
    for index in range(10):
        require(position<len(payload),'Complete protobuf varint')
        byte=payload[position];position+=1
        require(index<9 or byte<=1,'Protobuf uint64 does not overflow')
        value|=(byte&127)<<(7*index)
        if byte<128:
            require(index==0 or byte!=0,'Canonical protobuf varint')
            return value,position
    raise ValueError('Complete protobuf uint64')

def _protobuf_fields(payload):
    """Read complete wire fields only; never scan across encoded boundaries."""
    require(type(payload) is bytes and len(payload)<=16*2**20,'Bounded HLO protobuf bytes')
    position=0;fields=[]
    while position<len(payload):
        require(len(fields)<65536,'Bounded HLO protobuf field count')
        tag,position=_varint(payload,position);number,wire=tag>>3,tag&7
        require(0<number<2**29 and wire in (0,1,2,5),'Valid non-group HLO protobuf field')
        if wire==0:value,position=_varint(payload,position)
        else:
            if wire==2:size,position=_varint(payload,position)
            else:size=8 if wire==1 else 4
            require(size<=len(payload)-position,'Complete length-delimited/fixed HLO field')
            value=payload[position:position+size];position+=size
        fields.append((number,wire,value))
    return fields

def _field_values(fields,number,wire):
    selected=[(kind,value) for field,kind,value in fields if field==number]
    require(all(kind==wire for kind,value in selected),'Exact HLO field wire type')
    return [value for kind,value in selected]


def ints(fs, number):
    result = []
    for field, wire, value in fs:
        if field != number: continue
        require(wire in (0, 2), 'Integer HLO field')
        if wire == 0: result.append(value)
        else:
            position = 0
            while position < len(value):
                item, position = _varint(value, position); result.append(item)
    return result


def scalar(fs, number, default=None):
    items = _field_values(fs, number, 0)
    require(len(items) <= 1 and (items or default is not None), 'Unique scalar HLO field')
    return items[0] if items else default


def blob(fs, number):
    items = _field_values(fs, number, 2)
    require(len(items) == 1, 'Unique required HLO bytes field')
    return items[0]


def shape(raw):
    fs = _protobuf_fields(raw); kind = scalar(fs, 2); dims = tuple(ints(fs, 3))
    types = {1:'pred', 3:'s16', 4:'s32', 5:'s64', 13:'tuple', 16:'bf16'}
    require(kind in types and len(dims) <= 8 and all(0 < x <= 4194304 for x in dims), 'Bounded supported shape')
    children = tuple(shape(x) for x in _field_values(fs, 4, 2))
    dynamic = ints(fs, 6)
    require(not any(dynamic) and (not dynamic or len(dynamic) == len(dims)), 'Static HLO dimensions')
    require((not dims and bool(children)) if kind == 13 else not children, 'Array or tuple shape')
    return types[kind], dims, children


STATE = ('bf16', (4,1,4096,256), ())
PREFIX = ('bf16', (4,1,256,256), ())
TAIL = ('bf16', (4,1,3840,256), ())
SLOTS = ('s32', (4,), ())
ARMS = ('bf16_cat', 's16_bitcast_cat')


def assembly_hlo_proof(payload, arm):
    """Prove output-1 assembly and its exact state/prefix/writer input paths.

    NKI implementation is bound separately by frozen source plus canonical
    backend func_literal identity. This proof does not interpret kernel IR or
    prove old-gather semantics; full old/new/state raw checks remain mandatory.
    """
    require(arm in ARMS, 'Exact matched assembly arm')
    module = _protobuf_fields(payload); entry = scalar(module, 6)
    computations = {}; instruction_count = 0
    for raw in _field_values(module, 3, 2):
        fs = _protobuf_fields(raw); cid = scalar(fs, 5); root = scalar(fs, 6)
        require(cid not in computations, 'Unique computation')
        nodes = {}; params = {}
        for iraw in _field_values(fs, 2, 2):
            item = _protobuf_fields(iraw); identifier = scalar(item, 35)
            require(identifier not in nodes, 'Unique instruction ID')
            n = dict(op=blob(item,2).decode('ascii'), shape=shape(blob(item,3)),
                     operands=ints(item,36), calls=ints(item,38), fields=item, id=(cid,identifier))
            nodes[identifier] = n
            if n['op'] == 'parameter':
                index = scalar(item,9,0); require(index not in params, 'Unique parameter number'); params[index] = identifier
            instruction_count += 1; require(instruction_count <= 65536, 'Bounded full HLO instruction count')
        require(root in nodes and set(params)==set(range(len(params))), 'Complete computation root/parameters')
        computations[cid] = root, nodes, params
    require(entry in computations and 0 < len(computations) <= 128, 'Bounded entry computation')
    _, entry_nodes, params = computations[entry]
    require(len(params)==3, 'Original state, updates and int32 slots only')
    by_shape = {entry_nodes[i]['shape']:p for p,i in params.items()}
    require(set(by_shape)=={STATE,PREFIX,SLOTS}, 'Exact original BF16 state/update and int32 slots')
    cache_number, update_number, slot_number = (by_shape[s] for s in (STATE,PREFIX,SLOTS))
    entry_args = [dict(op='input',shape=entry_nodes[params[i]]['shape'],number=i,inputs=()) for i in range(3)]
    expanded = [0]

    def comp(cid, args, stack):
        require(cid in computations and cid not in stack and len(stack)<16, 'Bounded acyclic calls')
        root,nodes,parameters = computations[cid]
        require(len(args)==len(parameters), 'Exact call arity')
        memo={};active=set()
        def walk(identifier):
            if identifier in memo:return memo[identifier]
            require(identifier in nodes and identifier not in active, 'Acyclic complete operand references')
            active.add(identifier); n=nodes[identifier]; expanded[0]+=1
            require(expanded[0]<=262144 and not ints(n['fields'],37), 'Bounded expansion without control-only dependencies')
            operands=[walk(i)for i in n['operands']]
            require(n['op']=='call' or not n['calls'], 'Calls only on call instructions')
            if n['op']=='parameter':
                require(not operands,'No parameter operands');out=args[scalar(n['fields'],9,0)]
            elif n['op']=='call':
                require(len(n['calls'])==1,'Single exact callee');out=comp(n['calls'][0],operands,stack+(cid,))
            elif n['op']=='get-tuple-element':
                require(len(operands)==1 and operands[0]['op']=='tuple','Projection of actual tuple')
                index=scalar(n['fields'],13,0);require(index<len(operands[0]['inputs']),'Tuple index')
                out=operands[0]['inputs'][index]
            else: out=dict(n,inputs=tuple(operands))
            require(out['shape']==n['shape'],'Callee/operand shape matches declared result')
            memo[identifier]=out;active.remove(identifier);return out
        return walk(root)

    output=comp(entry,entry_args,())
    require(output['op']=='tuple' and len(output['inputs'])==2
            and output['shape']==('tuple',(),(STATE,STATE)), 'Original old/new physical outputs')
    alias=_protobuf_fields(blob(module,8)); entries=_field_values(alias,1,2)
    require(len(entries)==1,'Exactly one state alias')
    ae=_protobuf_fields(entries[0])
    require(ints(ae,1)==[1] and scalar(ae,2,0)==cache_number and not ints(ae,3)
            and scalar(ae,4)==2, 'Output1 MUST_ALIAS original full BF16 state')
    path=[]
    def clean(n):
        while n['op']=='copy':
            require(len(n['inputs'])==1 and n['shape']==n['inputs'][0]['shape'],'Exact-shape copy only')
            path.append(n['id']);n=n['inputs'][0]
        return n
    def expect(n,opcode,declared,count):
        n=clean(n);require(n['op']==opcode and n['shape']==declared and len(n['inputs'])==count,
                           'Exact '+opcode+' assembly dataflow; no numeric conversion or arithmetic')
        path.append(n.get('id'));return n
    new=clean(output['inputs'][1]);bitcasts=0
    if arm=='s16_bitcast_cat':
        new=expect(new,'bitcast-convert',STATE,1);new=new['inputs'][0];bitcasts+=1
    join_type='s16' if arm=='s16_bitcast_cat' else 'bf16'
    join=expect(new,'concatenate',(join_type,STATE[1],()),2)
    require(ints(join['fields'],14)==[2],'Concatenate token axis2 only')
    prefix,tail=join['inputs']
    if arm=='s16_bitcast_cat':
        prefix=expect(prefix,'bitcast-convert',('s16',PREFIX[1],()),1)['inputs'][0]
        tail=expect(tail,'bitcast-convert',('s16',TAIL[1],()),1)['inputs'][0];bitcasts+=2
    require(prefix['shape']==PREFIX and tail['shape']==TAIL,'BF16 assembly operands unchanged')
    def input_is(n,number,declared):
        n=clean(n);require(n['op']=='input' and n['number']==number and n['shape']==declared,'Exact original graph input')
    def state_slice(n,start,end,declared):
        n=expect(n,'slice',declared,1);input_is(n['inputs'][0],cache_number,STATE)
        ranges=[]
        for encoded in _field_values(n['fields'],17,2):
            f=_protobuf_fields(encoded);ranges.append((scalar(f,1,0),scalar(f,2),scalar(f,3)))
        require(ranges==[(0,4,1),(0,1,1),(start,end,1),(0,256,1)],'Exact full-row state slice bounds')
    state_slice(tail,256,4096,TAIL)
    def flatten_path(n,target):
        n=clean(n)
        while n['op']=='reshape':
            require(len(n['inputs'])==1 and n['shape'][0]==n['inputs'][0]['shape'][0]=='bf16'
                    and math.prod(n['shape'][1])==math.prod(n['inputs'][0]['shape'][1]),'Equal-element BF16 reshape only')
            path.append(n['id']);n=clean(n['inputs'][0])
        require(n['shape']==target,'Original unflattened writer operand');return n
    writer=clean(prefix)
    while writer['op']=='reshape':
        require(len(writer['inputs'])==1 and writer['inputs'][0]['shape'][0]=='bf16'
                and math.prod(writer['shape'][1])==math.prod(writer['inputs'][0]['shape'][1]),'Writer output reshape only')
        path.append(writer['id']);writer=clean(writer['inputs'][0])
    require(writer['op']=='custom-call' and writer['shape']==('bf16',(4,65536),())
            and len(writer['inputs'])==3,'Original flat NKI prefix writer output and three inputs')
    require(_field_values(writer['fields'],28,2)==[b'AwsNeuronCustomNativeKernel'],'Original NKI target')
    cache_in,updates_in,slots_in=writer['inputs']
    require(cache_in['shape']==updates_in['shape']==('bf16',(4,65536),()),'Original flat writer payloads')
    state_slice(flatten_path(cache_in,PREFIX),0,256,PREFIX)
    input_is(flatten_path(updates_in,PREFIX),update_number,PREFIX)
    input_is(slots_in,slot_number,SLOTS)
    backend=blob(writer['fields'],43);decoded=base64.b64decode(backend,validate=True)
    require(base64.b64encode(decoded)==backend,'Canonical NKI backend encoding')
    def unique(pairs):
        result={}
        for k,v in pairs:require(k not in result,'Unique backend JSON key');result[k]=v
        return result
    def reject_constant(value):raise ValueError('Nonfinite backend JSON')
    config=json.loads(decoded,object_pairs_hook=unique,parse_constant=reject_constant)
    require(type(config)is dict and config.get('func_name')=='slot_cache_gather_predicated'
            and config.get('platform_target')=='trn1' and type(config.get('func_literal'))is str
            and bool(config['func_literal']), 'Actual original NKI writer function and target')
    return dict(schema='qwen35_kv256_s16_pair_hlo_proof_v1',arm=arm,
        hlo_sha256=hashlib.sha256(payload).hexdigest(),hlo_bytes=len(payload),
        computations=len(computations),instructions=instruction_count,entry_parameter_numbers=dict(cache=cache_number,updates=update_number,slots=slot_number),
        assembly_path_instruction_ids=[list(x)for x in path if x is not None],
        concat_axis=2,concat_dtype=join_type,bitcast_converts=bitcasts,numeric_converts_in_assembly=0,
        exact_tail_slice=True,exact_writer_prefix_slice=True,original_writer_inputs=True,
        physical_output_order=['old','new'],state_alias_output=1,state_alias_kind='MUST_ALIAS',
        public_payload_dtype='bf16',writer_func_literal_sha256=hashlib.sha256(config['func_literal'].encode()).hexdigest(),
        writer_backend_sha256=hashlib.sha256(backend).hexdigest(),old_gather_semantics_proven=False,
        optimized_backend_instructions_proven=False,full_geometry_qualified=False,full_model_qualified=False)
