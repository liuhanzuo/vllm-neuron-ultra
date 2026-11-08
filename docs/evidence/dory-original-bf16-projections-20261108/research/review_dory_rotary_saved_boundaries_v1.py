"""Independent stdlib-only replay of complete Dory RoPE saved boundaries.

Checks original fixtures, both output paths, exact values/bytes and native-style
multiply/add from saved cache coefficients. Does not recompute libtorch sin/cos,
execute official dispatch or qualify a full model.
"""
import hashlib,json,math,os,stat,struct
from pathlib import Path
POSITIONS=(0,1,1023,1024,2048,262143)
THETAS=(10000,10000000);DTYPES=('float32','bfloat16');FIXTURES=('signed_ramp','binary_scale_cancellation')
LABELS=('input_query','input_key','candidate_query','candidate_key','reference_query','reference_key')
PATTERN=(0.0,-0.0,1.0,-1.0,0.00390625,-0.00390625,0.5,-0.5,1.0078125,-1.0078125,128.0,-128.0,0.99609375,-0.99609375,0.03125,-0.03125)

def require(value,message):
 if not value:raise ValueError(message)

def stable(path,cap):
 path=Path(path);require(path.is_absolute()and path.resolve()==path,'Canonical evidence path')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd);require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=cap,'Bounded single evidence file')
  chunks=[];size=0
  while True:
   block=os.read(fd,min(65536,cap+1-size))
   if not block:break
   size+=len(block);require(size<=cap,'Evidence grew');chunks.append(block)
  after=os.fstat(fd);last=path.lstat();keys=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
  require(all(getattr(before,k)==getattr(after,k)==getattr(last,k)for k in keys),'Evidence changed during read')
  return b''.join(chunks)
 finally:os.close(fd)

def load_ref(ref):
 raw=stable(Path(ref['path']),2**20);require(len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256'],'Exact metadata reference');return json.loads(raw)

def pack(value,dtype):
 if dtype=='float32':return struct.pack('<f',value)
 require(dtype=='bfloat16','Original floating dtype')
 bits=struct.unpack('<I',struct.pack('<f',value))[0]
 # Finite values only. This is round-to-nearest, ties-to-even at BF16 precision.
 return struct.pack('<H',((bits+0x7fff+((bits>>16)&1))>>16)&0xffff)

def decode(raw,dtype):
 if dtype=='int64':return [v[0]for v in struct.iter_unpack('<q',raw)]
 if dtype=='float32':return [v[0]for v in struct.iter_unpack('<f',raw)]
 require(dtype=='bfloat16','Original saved dtype')
 return [struct.unpack('<f',b'\0\0'+raw[i:i+2])[0]for i in range(0,len(raw),2)]

def rounded(value,dtype):return decode(pack(value,dtype),dtype)[0]

def expected_rows(root):
 root=Path(root);rows={'positions':dict(shape=[6],dtype='int64')}
 for theta in THETAS:
  for dtype in DTYPES:
   prefix=f'theta{theta}_{dtype}';rows[prefix+'_cache_selected']=dict(shape=[6,256],dtype=dtype)
   for fixture in FIXTURES:
    for label in LABELS:rows[prefix+'_'+fixture+'_'+label]=dict(shape=[6,16 if label.endswith('query')else 4,256],dtype=dtype)
 for name,row in rows.items():row.update(path=str(root/(name+'.raw')),bytes=math.prod(row['shape'])*dict(int64=8,float32=4,bfloat16=2)[row['dtype']])
 require(len(rows)==53,'Complete fixed raw matrix');return rows

def fixture_bytes(fixture,heads,dtype):
 count=6*heads*256
 require(fixture in FIXTURES,'Original fixture')
 return b''.join(pack(((i%509)-254)/32 if fixture=='signed_ramp'else PATTERN[i%len(PATTERN)],dtype)for i in range(count))

def rotate_from_saved_coefficients(input_raw,cache_raw,heads,dtype):
 x=decode(input_raw,dtype);coeff=decode(cache_raw,dtype);output=[]
 for pos in range(6):
  for head in range(heads):
   first=[];second=[];offset=(pos*heads+head)*256
   for j in range(128):
    left,right=x[offset+j],x[offset+128+j];cosine,sine=coeff[pos*256+j],coeff[pos*256+128+j]
    first.append(pack(rounded(left*cosine,dtype)-rounded(right*sine,dtype),dtype))
    second.append(pack(rounded(right*cosine,dtype)+rounded(left*sine,dtype),dtype))
   output.extend(first+second)
 return b''.join(output)

def comparison(actual_raw,expected_raw,dtype):
 actual,expected=decode(actual_raw,dtype),decode(expected_raw,dtype);require(len(actual)==len(expected)>0,'Exact output size')
 finite=all(math.isfinite(x)for x in actual+expected)
 require(finite,'Finite complete saved outputs')
 delta=[abs(a-e)for a,e in zip(actual,expected)]
 return dict(all_finite=True,actual_sha256=hashlib.sha256(actual_raw).hexdigest(),expected_sha256=hashlib.sha256(expected_raw).hexdigest(),
  bitwise_equal=actual_raw==expected_raw,unequal_values=sum(a!=e for a,e in zip(actual,expected)),max_abs_error=max(delta),mean_abs_error=sum(delta)/len(delta),
  max_relative_nonzero=max((d/abs(e)for d,e in zip(delta,expected)if e!=0),default=0.0),
  nonzero_against_zero=sum(a!=0 and e==0 for a,e in zip(actual,expected)),passed=actual_raw==expected_raw and all(a==e for a,e in zip(actual,expected)))

def review(result_ref,guard):
 require(callable(guard),'Actual bounded external guard required');guard();result=load_ref(result_ref)
 require(result['schema']=='dory_rotary_control_capture_result_v1'and result['complete']is True,'Complete actual runtime result')
 control,capture,manifest=(load_ref(result[k])for k in ('control','capture','manifest'))
 require(result['observer_invariant']==(control==capture),'Actual observer equality classification')
 root=Path(result_ref['path']).parent/'raw';expected=expected_rows(root);require(set(manifest['rows'])==set(expected),'All53exact names')
 require(set(root.iterdir())=={Path(row['path'])for row in expected.values()},'No extra/missing raw boundary file')
 saved={}
 for name,want in expected.items():
  guard();row=manifest['rows'][name];require(all(row[k]==v for k,v in want.items()),'Exact original boundary shape/dtype/path')
  raw=stable(Path(row['path']),128*1024);require(len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'],'Saved boundary bytes changed');saved[name]=raw
  if row['dtype']!='int64':require(all(math.isfinite(v)for v in decode(raw,row['dtype'])),'Finite full tensor')
 require(decode(saved['positions'],'int64')==list(POSITIONS),'Original boundary positions')
 require(manifest['tensors']==result['saved_tensors']==53 and manifest['raw_bytes']==sum(map(len,saved.values())),'Raw counts derived from actual bytes')
 require(result['original_comparisons']==result['captured_comparisons']==result['unique_position_comparisons']==96 and result['executed_position_comparisons']==192,'Two complete original comparison rounds')
 derived=[];replay=[];inputs=[];caches=[]
 for theta in THETAS:
  for dtype in DTYPES:
   prefix=f'theta{theta}_{dtype}';cache=saved[prefix+'_cache_selected'];caches.append(dict(theta=theta,dtype=dtype,shape=[262144,256],selected_positions_sha256=hashlib.sha256(cache).hexdigest()))
   for fixture in FIXTURES:
    case=prefix+'_'+fixture;inputs.append(dict(theta=theta,dtype=dtype,fixture=fixture,input_sha256={kind:hashlib.sha256(saved[case+'_input_'+kind]).hexdigest()for kind in ('query','key')}))
    for kind,heads in [('query',16),('key',4)]:
     inp=saved[case+'_input_'+kind];require(inp==fixture_bytes(fixture,heads,dtype),'Original deterministic fixture bytes')
     candidate,reference=saved[case+'_candidate_'+kind],saved[case+'_reference_'+kind]
     replayed=rotate_from_saved_coefficients(inp,cache,heads,dtype)
     replay.append(dict(theta=theta,dtype=dtype,fixture=fixture,tensor=kind,reference_reproduced=replayed==reference,candidate_reproduced=replayed==candidate,
                        replay_sha256=hashlib.sha256(replayed).hexdigest()))
     width=heads*256*dict(float32=4,bfloat16=2)[dtype]
     for i,position in enumerate(POSITIONS):derived.append(dict(theta=theta,fixture=fixture,tensor=kind,position=position,shape=[heads,256],dtype='torch.'+dtype,**comparison(candidate[i*width:(i+1)*width],reference[i*width:(i+1)*width],dtype)))
 require(capture['input_cases']==inputs and capture['cache_proofs']==caches,'Input/cache hashes agree with retained raw bytes')
 require(len(control['comparisons'])==len(capture['comparisons'])==len(derived)==96,'Complete row counts')
 for row,claimed in zip(derived,capture['comparisons']):require(all(claimed[k]==v for k,v in row.items()),'Independent full saved-byte metrics')
 parity=all(row['passed']for row in derived)
 require(capture['passed']is parity and result['captured_arithmetic_passed']is parity,'Derived arithmetic pass classification')
 require(result['original_arithmetic_passed']is control['passed']and result['passed']==(result['observer_invariant']and control['passed']and parity),'Original control and overall classification retained')
 for value in [control,capture,result]:require(value['native_reference_qualified']is value['new_speedup_claim']is False,'No full-model or speedup inference')
 guard();return dict(schema='dory_rotary_independent_saved_boundary_review_v1',complete=True,raw_tensors=53,raw_bytes=sum(map(len,saved.values())),comparisons=derived,
  independent_native_style_replay=replay,all_reference_outputs_reproduced=all(row['reference_reproduced']for row in replay),all_candidate_outputs_reproduced=all(row['candidate_reproduced']for row in replay),
  observer_invariant=result['observer_invariant'],arithmetic_parity=parity,result=result_ref,full_cache_trigonometry_recomputed=False,official_dispatch_executed=False,original_weights_loaded=False,
  native_reference_qualified=False,neuron_baseline_qualified=False,new_speedup_claim=False)
