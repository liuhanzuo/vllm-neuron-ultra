import ast,hashlib,json,sys
from pathlib import Path
Q=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research');ROOT=Q.parent
sys.path[:0]=[str(Q),str(ROOT)]
import minicpm5_retained_retirement_contract_v4 as p
import minicpm5_retained_retirement_contract_v3 as previous
prior_path=Q/'minicpm5-retirement-static-source-closure-20261113-v5.json'
prior_bytes=prior_path.read_bytes()
assert hashlib.sha256(prior_bytes).hexdigest()=='3182f2185fad329be0231f5add821ae6bcb1b9793a75a6193037fc768eefbe7d'
prior=json.loads(prior_bytes)
starts={Path(x)for x in (*p.SOURCES,*previous.SOURCES,*p.base.FROZEN,*prior['source_sha256'])}
seen={};pending=list(starts);edges={}
def candidates(base,name):
 rel=Path(*name.split('.')) if name else Path()
 return [base/rel.with_suffix('.py'),base/rel/'__init__.py'] if name else [base/'__init__.py']
def local_paths(node,path):
 output=set()
 if isinstance(node,ast.Import):
  names=[x.name for x in node.names];bases=[Q,ROOT]
 else:
  names=[node.module or '']
  if node.module:names.extend(node.module+'.'+x.name for x in node.names if x.name!='*')
  elif node.level:names.extend(x.name for x in node.names if x.name!='*')
  if node.level:
   base=path.parent
   for _ in range(node.level-1):base=base.parent
   bases=[base]
  else:bases=[Q,ROOT]
 for base in bases:
  for name in names:
   for item in candidates(base,name):
    if item.is_file()and (item.is_relative_to(Q)or item.is_relative_to(ROOT)):
     output.add(item)
     for parent in item.parents:
      if parent==ROOT.parent:break
      init=parent/'__init__.py'
      if init.is_file():output.add(init)
 return output
while pending:
 path=pending.pop()
 if path in seen:continue
 data=path.read_bytes();tree=ast.parse(data,str(path));seen[path]=hashlib.sha256(data).hexdigest();found=set()
 for node in ast.walk(tree):
  if isinstance(node,(ast.Import,ast.ImportFrom)):found.update(local_paths(node,path))
 edges[str(path)]=sorted(map(str,found));pending.extend(found-set(seen))
assert all(seen.get(Path(path))==sha for path,sha in prior['source_sha256'].items()),'Complete previous static source closure retained'
value=dict(schema='minicpm5_retained_retirement_static_local_import_closure_v1',root_sources=sorted(map(str,starts)),
 source_sha256={str(k):v for k,v in sorted(seen.items())},edges=edges,source_only=True,
 transitive_modules_executed_for_discovery=False,dynamic_imports_remain_in_complete_original_seed_closure=True)
output=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research/minicpm5-retirement-static-source-closure-20261113-v6.json')
with output.open('x')as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(dict(source_files=len(seen),python_bytes=sum(Path(x).stat().st_size for x in seen),manifest=str(output))))
