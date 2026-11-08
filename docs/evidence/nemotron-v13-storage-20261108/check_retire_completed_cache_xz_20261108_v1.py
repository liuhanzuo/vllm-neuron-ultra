"""Small synthetic destructive-operation and archive-integrity controls. No actual caches."""
import copy,hashlib,io,json,lzma,os,sys,tarfile,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from research import retire_completed_cache_xz_20261108_v1 as m
passed=[]
def reject(name,fn):
 try:fn()
 except (AssertionError,FileExistsError,FileNotFoundError,NotADirectoryError):passed.append(name);return
 raise AssertionError('Accepted invalid '+name)

def main():
 with tempfile.TemporaryDirectory(prefix='xz-retirement-source-controls-') as tmp:
  t=Path(tmp);root=Path('/mnt/nvme/neuron-cache/neuronxcc-2.23.6484.0+3b612583');name='MODULE_fixture+fixture';payload=b'completed cache bytes';digest=hashlib.sha256(payload).hexdigest()
  manifest=dict(directories={str(root/name):dict(mode=0o40700,uid=os.getuid(),gid=os.getgid())},files={name+'/model.neff':dict(mode=0o100600,uid=os.getuid(),gid=os.getgid(),size=len(payload),sha256=digest)})
  def make(kind):
   stream=io.BytesIO()
   with tarfile.open(fileobj=stream,mode='w') as archive:
    d=tarfile.TarInfo(name);d.type=tarfile.DIRTYPE;d.mode=0o700;d.uid=os.getuid();d.gid=os.getgid();archive.addfile(d)
    if kind!='missing':
     f=tarfile.TarInfo(name+'/model.neff');f.mode=0o600;f.uid=os.getuid();f.gid=os.getgid();f.size=len(payload)
     if kind=='mode':f.mode=0o644
     if kind=='escape':f.name='../model.neff'
     if kind=='symlink':f.type=tarfile.SYMTYPE;f.linkname='/etc/passwd';f.size=0
     archive.addfile(f,io.BytesIO(payload))
     if kind=='duplicate':archive.addfile(f,io.BytesIO(payload))
   raw=stream.getvalue();path=t/(kind+'.tar.xz');path.write_bytes(lzma.compress(raw,preset=0));return path,hashlib.sha256(raw).hexdigest(),len(raw)
  path,sha,size=make('valid');got=m.verified_archive(path,manifest,sha,size,lambda:None);assert got['files_rehashed']==1 and got['directories_verified']==1;passed.append('complete bytes and metadata accepted')
  for kind in ['missing','mode','escape','symlink','duplicate']:
   path,sha,size=make(kind);reject(kind,lambda:m.verified_archive(path,manifest,sha,size,lambda:None))
  path,sha,size=make('valid2');bad=copy.deepcopy(manifest);bad['files'][name+'/model.neff']['sha256']='0'*64
  reject('member payload mismatch',lambda:m.verified_archive(path,bad,sha,size,lambda:None))
  reject('whole tar mismatch',lambda:m.verified_archive(path,manifest,'0'*64,size,lambda:None))
  reject('whole tar size mismatch',lambda:m.verified_archive(path,manifest,sha,size+1,lambda:None))
  b=m.budget(5*m.GIB,2*m.GIB,0,45*m.GIB,3*m.GIB);assert b['projected_free_lower_bound']>=48*m.GIB;passed.append('sufficient real recovery')
  reject('insufficient net recovery',lambda:m.budget(4*m.GIB,3*m.GIB,0,50*m.GIB,3*m.GIB))
  reject('insufficient admission floor',lambda:m.budget(5*m.GIB,2*m.GIB,0,30*m.GIB,3*m.GIB))
  reject('metadata reservation required',lambda:m.budget(4*m.GIB,2*m.GIB,0,50*m.GIB,3*m.GIB))
  reject('negative accounting',lambda:m.budget(5*m.GIB,2*m.GIB,-1,45*m.GIB,3*m.GIB))
  def entry(label):
   d=t/label;d.mkdir();f=d/'model.neff';f.write_bytes(payload)
   return dict(path=str(d),directory_identity=m.original.identity(d),files=[m.original.identity(f)])
  row=entry('retire-ok');m.retire_directory(row);assert not Path(row['path']).exists();passed.append('only exact selected directory removed')
  row=entry('retire-change');f=Path(row['files'][0]['path']);f.write_bytes(b'changed');reject('changed file identity',lambda:m.retire_directory(row));assert f.exists()
  row=entry('retire-extra');d=Path(row['path']);(d/'extra').write_bytes(b'keep');reject('unexpected child',lambda:m.retire_directory(row));assert (d/'extra').exists()
  row=entry('retire-hardlink');f=Path(row['files'][0]['path']);os.link(f,t/'retained-link');reject('changed hardlink count',lambda:m.retire_directory(row));assert f.exists()
  row=entry('retire-replaced');d=Path(row['path']);saved=t/'moved';d.rename(saved);d.symlink_to(saved,target_is_directory=True);reject('replaced directory symlink',lambda:m.retire_directory(row));assert (saved/'model.neff').exists()
 result=dict(schema='completed_cache_xz_retirement_source_controls_v1',passed=True,checks=len(passed),controls=passed,source=m.ref(Path(m.__file__)),test_source=m.ref(Path(__file__)),actual_cache_retired=False,actual_archive_verified=False,native_started=False,new_speedup_claim=False)
 out=R/'research/completed-cache-xz-retirement-source-controls-20261108-v1.json';print(json.dumps(dict(report=m.once(out,result),checks=len(passed))))
if __name__=='__main__':main()
