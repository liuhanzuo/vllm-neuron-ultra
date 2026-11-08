"""Read-only process-reference scan for the fixed retirement selection."""
from pathlib import Path
import hashlib,json,os
PLAN=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research/nemotron9b-r7-space-retirement-plan-20261108-v1.json')
raw=PLAN.read_bytes();assert len(raw)<8*2**20 and hashlib.sha256(raw).hexdigest()=='b482eae9246668dbbe137ebf175e8e9031751588be09e09f76486b30c966abd6'
plan=json.loads(raw);targets={row['path']for row in plan['files']};roots=();hits=[];errors=[];processes=0
assert os.geteuid()==0

def matches(value):return value in targets or value.startswith(roots)
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:
  processes+=1
  argv=[x.decode(errors='replace')for x in (process/'cmdline').read_bytes().split(b'\0')if x]
  if any(matches(x)for x in argv):hits.append(dict(pid=int(process.name),kind='argv'))
  for item in [process/'cwd',process/'exe',*(process/'fd').iterdir()]:
   try:value=os.readlink(item).removesuffix(' (deleted)')
   except FileNotFoundError:continue
   if matches(value):hits.append(dict(pid=int(process.name),kind='open',path=value))
  for line in (process/'maps').read_text().splitlines():
   fields=line.split(maxsplit=5)
   if len(fields)==6 and matches(fields[-1]):hits.append(dict(pid=int(process.name),kind='mapped',path=fields[-1]))
 except (FileNotFoundError,ProcessLookupError):continue
 except PermissionError as error:errors.append(dict(pid=int(process.name),path=error.filename))
value=dict(processes=processes,hits=hits,permission_errors=errors,read_only=True)
print(json.dumps(value),flush=True)
assert not hits and not errors
