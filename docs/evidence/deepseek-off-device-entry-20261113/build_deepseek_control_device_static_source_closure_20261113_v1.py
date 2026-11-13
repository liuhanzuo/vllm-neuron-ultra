import ast,hashlib,json,sys
from pathlib import Path
Q=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research');ROOT=Q.parent
starts={ROOT/'deepseek_cold_control_device_entry_v1.py'}
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
value=dict(schema='deepseek_control_device_static_local_import_closure_v1',root_sources=sorted(map(str,starts)),
 source_sha256={str(k):v for k,v in sorted(seen.items())},edges=edges,source_only=True,
 transitive_modules_executed_for_discovery=False,actual_external_sdk_and_dynamic_sources_remain_outer_runtime_duty=True)
output=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research/deepseek-control-device-entry-static-source-closure-20261113-v1.json')
with output.open('x')as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(dict(source_files=len(seen),python_bytes=sum(Path(x).stat().st_size for x in seen),manifest=str(output))))
