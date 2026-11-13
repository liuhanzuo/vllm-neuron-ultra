"""Exact SOURCE-ONLY v1->v2 preparer derivation; --check never writes.

No current metadata, History, locks, native imports or runtime admission.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

Q=Path(__file__).resolve().parent
PREDECESSOR=dict(path=str(Q/'prepare_qwen35_kv256_alias_diagnostic_binding_v1.py'),
    sha256='d903881e0a66e0d9e5baec131c7974d08d16935ebdf69439ac2576032f6c5c79')
TARGET=Q/'prepare_qwen35_kv256_alias_diagnostic_binding_v2.py'

CONSTANTS='''
# One unchanged inherited plan; the frozen runtime already has its typed reader.
PREDECESSOR_PREPARER = dict(path=str(Q/'prepare_qwen35_kv256_alias_diagnostic_binding_v1.py'),
    sha256='d903881e0a66e0d9e5baec131c7974d08d16935ebdf69439ac2576032f6c5c79')
INHERITED_PLAN = dict(path=str(Q/'minicpm5-layer1-operator-sdk-r2-20261111-plan.json'),
    sha256='2e5441f4b2c2ad2efd76192ae92c068c35d3ae9fc51c532062d4284692910632',
    size_bytes=12803326, schema='minicpm5_layer0_runtime_inputs_v1', original_typed_cap_bytes=32*MIB)
'''

HELPERS='''
def inherited_plan_shape(value):
    """Exact plan branch of the existing operator SDK envelope shape checks."""
    require(type(value)is dict and value.get('schema')==INHERITED_PLAN['schema'], 'Original typed plan schema')
    pins=value['source_sha256']
    require(type(pins)is dict and len(pins)<=100000, 'Original bounded complete plan source map')
    for name,digest in pins.items():
        require(type(name)is str and Path(name).is_absolute()and str(Path(name))==name and '..'not in Path(name).parts
            and type(digest)is str and re.fullmatch('[0-9a-f]{64}',digest), 'Original literal absolute source pin')
    reference=value['source_contract']
    require(type(reference)is dict and set(reference)in({'path','sha256'},{'path','sha256','bytes'})
        and type(reference['path'])is str and Path(reference['path']).is_absolute()
        and type(reference['sha256'])is str and re.fullmatch('[a-f0-9]{64}',reference['sha256'])
        and ('bytes'not in reference or type(reference['bytes'])is int and reference['bytes']>=0),
        'Original exact source-contract reference')
    require(type(value['latest_completion'])is dict and type(value['output_root'])is str
        and type(value['inputs'])is dict and len(value['inputs'])<=100000, 'Original complete typed plan fields')


def retained_input_caps(inputs,seed_inputs,observed,base_ref,base):
    """Ordinary8MiB plus exactly one already-inherited, unchanged typed input."""
    require(all(inputs.get(path)==row for path,row in seed_inputs.items()), 'Every complete seed input retained unchanged')
    retained={}
    for path,row in inputs.items():
        size=row['identity']['size']
        require(type(size)is int and size>=0, 'Finite original input size')
        if path!=INHERITED_PLAN['path']:
            require(size<=8*MIB, 'Ordinary runtime metadata retains original8MiB cap')
            continue
        require(base_ref=={k:INHERITED_PLAN[k]for k in('path','sha256')}
            and row['sha256']==INHERITED_PLAN['sha256']and size==INHERITED_PLAN['size_bytes']
            and size<=INHERITED_PLAN['original_typed_cap_bytes'], 'Only the exact original typed plan exception')
        require(seed_inputs.get(path)==row and observed.get(path)==row,
            'Same full inherited and freshly authenticated plan identity; never refresh or introduce an input')
        ident=row['identity']
        require(set(ident)==IDENTITY_KEYS and ident['realpath']==path
            and all(type(ident[k])is int and ident[k]>=0 for k in IDENTITY_KEYS-{'realpath'}),
            'Exact complete original plan identity')
        inherited_plan_shape(base);retained[path]=copy.deepcopy(row)
    return retained


def verify_executed_preparer(helper):
    """CLI source check before any candidate preparation or output publication."""
    require(helper['schema']=='qwen35_kv256_alias_diagnostic_binding_source_delivery_v2'
        and helper['source_sha256'].get(str(SELF))==hashlib.sha256(SELF.read_bytes()).hexdigest()
        and helper['source_sha256'].get(PREDECESSOR_PREPARER['path'])==PREDECESSOR_PREPARER['sha256'],
        'Exact executed v2 preparer and retained frozen v1 helper must be pre-enrolled')


'''


def derive():
    data=Path(PREDECESSOR['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==PREDECESSOR['sha256'],'Frozen preparer predecessor changed'
    source=data.decode();edits=[]
    def edit(old,new):
        nonlocal source
        assert source.count(old)==1,(old,source.count(old))
        source=source.replace(old,new);edits.append(dict(before=old,after=new))
    edit("SELF = Q / 'prepare_qwen35_kv256_alias_diagnostic_binding_v1.py'",
        "SELF = Q / 'prepare_qwen35_kv256_alias_diagnostic_binding_v2.py'")
    edit('\n\ndef require(value, message):','\n'+CONSTANTS+'\n\ndef require(value, message):')
    edit('def api():',HELPERS+'def api():')
    edit("helper['schema'] == 'qwen35_kv256_alias_diagnostic_binding_source_delivery_v1'",
        "helper['schema'] == 'qwen35_kv256_alias_diagnostic_binding_source_delivery_v2'\n"
        "            and helper['predecessor_preparer'] == PREDECESSOR_PREPARER\n"
        "            and helper['retained_input_exception'] == INHERITED_PLAN\n"
        "            and helper['source_sha256'].get(PREDECESSOR_PREPARER['path']) == PREDECESSOR_PREPARER['sha256']")
    edit("    require(all(v['identity']['size'] <= 8*MIB for v in inputs.values()), 'No large preparation envelope becomes a runtime input')",
        "    retained_large_inputs = retained_input_caps(inputs, seed['input_extension'], observed, overlay['base_plan'], base)\n"
        "    for path in retained_large_inputs: preparation_only.pop(path, None)")
    edit("            full_history_children_recheck_pending=True, original_resource_limits_preserved=True, exact_diagnostic_workload=True, full_geometry_qualified=False),",
        "            full_history_children_recheck_pending=True, original_resource_limits_preserved=True, exact_diagnostic_workload=True, full_geometry_qualified=False,\n"
        "            exact_retained_typed_inputs=retained_large_inputs, ordinary_runtime_metadata_cap_bytes=8*MIB),")
    edit('    result = prepare(future, reader.read, fresh_path, lambda: None)',
        "    verify_executed_preparer(reader.read(future['preparer_delivery'])['value'])\n"
        '    result = prepare(future, reader.read, fresh_path, lambda: None)')
    restored=source
    for row in reversed(edits):
        assert restored.count(row['after'])==1,row['after'][:100]
        restored=restored.replace(row['after'],row['before'])
    assert restored.encode()==data,'Exact source reversal failed'
    ast.parse(source)
    return source,edits


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();source,edits=derive()
    if args.check:assert TARGET.read_text()==source
    else:
        with TARGET.open('x')as stream:stream.write(source)
    print(json.dumps(dict(source_only=True,exact_edits=len(edits),runtime_launched=False,checked=args.check)))


if __name__=='__main__':main()
