"""Extract six1.17 Python3.10/Linux alias metadata without executing package code."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

from nemotron4b_helper_dependency_catalog_v1 import source
from nemotron4b_process_handshake_v1 import require

Q=Path(__file__).resolve().parent
INVENTORY=Q/'nemotron4b-helper-stdlib-origin-inventory-20261109-v1.json'
INVENTORY_SHA='fed3b20fe2305238f4c3e03a22ad8ef5d2b9aa496b71dffdc69b99c740d050d3'
OUT=Q/'nemotron4b-six-alias-source-policy-20261109-v1.json'


def value(node):
    if isinstance(node,ast.Constant)and type(node.value)in(str,int,bool,type(None)):
        return node.value
    if isinstance(node,ast.Tuple):return tuple(value(x)for x in node.elts)
    if isinstance(node,ast.Name)and node.id in {'__name__','PY34','PY3'}:
        return {'__name__':'six','PY34':True,'PY3':True}[node.id]
    if isinstance(node,ast.Attribute)and isinstance(node.value,ast.Name)and node.value.id=='sys':
        require(node.attr in ('platform','version_info'),'Known static six platform input')
        return 'linux'if node.attr=='platform'else(3,10,12)
    if isinstance(node,ast.BinOp)and isinstance(node.op,ast.Add):
        a,b=value(node.left),value(node.right)
        require(type(a)is str and type(b)is str,'Literal string addition only')
        return a+b
    if isinstance(node,ast.Compare)and len(node.ops)==len(node.comparators)==1:
        a,b=value(node.left),value(node.comparators[0]);op=node.ops[0]
        if isinstance(op,ast.Eq):return a==b
        if isinstance(op,ast.Lt):return a<b
        if isinstance(op,ast.GtE):return a>=b
    if isinstance(node,ast.IfExp):return value(node.body if value(node.test)else node.orelse)
    raise ValueError('Unreviewed six policy expression: '+ast.dump(node))


def extract(raw):
    tree=ast.parse(raw)
    declarations=[n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name)and t.id=='_moved_attributes'for t in n.targets)]
    require(len(declarations)==1 and isinstance(declarations[0].value,ast.List),
            'One explicit six moved declaration table')
    aliases={};overrides=[]
    for call in declarations[0].value.elts:
        require(isinstance(call,ast.Call)and isinstance(call.func,ast.Name)
                and not call.keywords and call.func.id in ('MovedModule','MovedAttribute'),
                'Known six moved declaration')
        if call.func.id!='MovedModule':continue
        require(2<=len(call.args)<=3,'Known MovedModule positional signature')
        name=value(call.args[0]);target=value(call.args[2])if len(call.args)==3 else name
        if target is None:target=name
        fullname='six.moves.'+name
        require(fullname not in aliases,'Unique initial moved module alias')
        aliases[fullname]=dict(kind='moved',name=name,target=target)
    registrations=[]
    for node in tree.body:
        if not(isinstance(node,ast.Expr)and isinstance(node.value,ast.Call)):continue
        call=node.value
        if not(isinstance(call.func,ast.Attribute)and isinstance(call.func.value,ast.Name)
               and call.func.value.id=='_importer'and call.func.attr=='_add_module'):continue
        require(not call.keywords and len(call.args)>=2,'Explicit top-level synthetic alias registration')
        obj=call.args[0]
        if isinstance(obj,ast.Name)and obj.id=='moves':
            cls_name='_MovedItems';module_name='six.moves'
        else:
            require(isinstance(obj,ast.Call)and isinstance(obj.func,ast.Name)
                    and obj.func.id.startswith('Module_six_moves_')and len(obj.args)==1
                    and not obj.keywords,'Known synthetic six module constructor')
            cls_name=obj.func.id;module_name=value(obj.args[0])
        names=['six.'+value(a)for a in call.args[1:]]
        row=dict(kind='synthetic',class_name=cls_name,module_name=module_name,aliases=names)
        for name in names:
            if name in aliases:overrides.append(dict(alias=name,original=aliases[name],replacement=row))
            aliases[name]=dict(row)
        registrations.append(row)
    all_add=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name)and n.func.value.id=='_importer'and n.func.attr=='_add_module']
    require(len(all_add)==len(registrations)+1,'All static alias registrations accounted for plus initial table loop')
    require(len(registrations)==7 and aliases['six.moves']['class_name']=='_MovedItems',
            'Original six1.17 synthetic registration groups')
    windows=[n for n in tree.body if isinstance(n,ast.If)and isinstance(n.test,ast.Compare)
        and ast.dump(n.test)==ast.dump(ast.parse("sys.platform == 'win32'",mode='eval').body)]
    require(len(windows)==1 and value(windows[0].test)is False,'Explicit Linux exclusion of winreg registration')
    require('six.moves.winreg'not in aliases,'No Windows-only alias on this machine')
    return dict(aliases=aliases,registration_groups=registrations,overridden_aliases=overrides)


def main():
    raw,inventory_ref=source(INVENTORY,INVENTORY_SHA);inventory=json.loads(raw)
    reference=inventory['merged_source_modules']['six']
    raw,actual=source(reference['path'],reference['sha256'])
    require(actual['fullstat']==reference['fullstat'],'Original installed six source identity')
    result=extract(raw)
    result.update(schema='nemotron4b_six_alias_source_policy_v1',reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_inventory=inventory_ref,original_six_source=actual,
        selected_python_version=[3,10,12],selected_platform='linux',
        alias_count=len(result['aliases']),
        moved_alias_count=sum(r['kind']=='moved'for r in result['aliases'].values()),
        synthetic_alias_count=sum(r['kind']=='synthetic'for r in result['aliases'].values()),
        analyzer_source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        installed_package_code_executed=False,actual_SDK_helper_executed=False,
        runtime_admitted=False,complete_runtime_dependency_closure_proven=False,new_speedup_claim=False)
    with OUT.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k]for k in ['alias_count','moved_alias_count','synthetic_alias_count']}))


if __name__=='__main__':main()
