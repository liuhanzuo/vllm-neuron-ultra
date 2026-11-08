"""Independent complete saved-byte metrics and rounding review, no Torch/NumPy."""
import argparse,array,hashlib,json,math,os,resource,signal,struct,sys,time
from pathlib import Path
Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
import run_minicpm5_frame1_fixed_oproj_v3 as t


def bf16(value):
    bits=struct.unpack('<I',struct.pack('<f',value))[0]
    return ((bits+0x7fff+((bits>>16)&1))>>16)&0xffff


def compare(a,b,lookup,guard):
    assert len(a)==len(b)
    full=[0,0.,0.];real=[0,0.,0.];last=[0,0.,0.]
    for i,(x,y)in enumerate(zip(a,b)):
        if i%32768==0:guard()
        left,right=lookup[x],lookup[y];assert math.isfinite(left)and math.isfinite(right)
        delta=abs(left-right);different=int(x!=y)
        full[0]+=different;full[1]=max(full[1],delta);full[2]+=delta
        if i<1231*2048:real[0]+=different;real[1]=max(real[1],delta);real[2]+=delta
        if 1230*2048<=i<1231*2048:last[0]+=different;last[1]=max(last[1],delta);last[2]+=delta
    def result(row,count):return dict(compared_elements=count,different_elements=row[0],max_abs=row[1],mean_abs=row[2]/count)
    return dict(full=result(full,len(a)),real_tokens=result(real,min(len(a),1231*2048)),prediction=result(last,2048))


def metric_equal(actual,expected):
    for scope,row in actual.items():
        old=expected[scope];assert row['compared_elements']==old['compared_elements']and row['different_elements']==old['different_elements']
        assert row['max_abs']==old['max_abs']
        assert math.isclose(row['mean_abs'],old['mean_abs'],rel_tol=1e-12,abs_tol=1e-15)


def run(external_ref):
    t.d.m.p.user_bus();fd,owner,lock=t.d.m.old.post.external.acquire_lock();start=time.monotonic();rows={}
    try:
        t.d.m.p.clear_window();t.d.m.p.active_heavy();ex=t.d.load(external_ref);result=t.d.load(ex['result']);plan=t.d.load(result['plan'])
        assert ex['actual_tool_exit_observed']and ex['exact_owner_gone']and ex['owner']==result['owner'];t.d.actual_exit(ex['actual_exit']);t.d.m.p.gone(result['owner'])
        assert result['passed']and result['complete']and plan['sources']==t.sources();admission=t.d.load(result['admission']);assert admission['lock_identity']==lock
        os.sched_setaffinity(0,{6});os.nice(19);resource.setrlimit(resource.RLIMIT_AS,(2*2**30,)*2);resource.setrlimit(resource.RLIMIT_CPU,(600,605));resource.setrlimit(resource.RLIMIT_FSIZE,(2**20,)*2)
        def timeout(signum,frame):raise TimeoutError('Independent fixed-context review wall budget')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(900);last_guard=[0.]
        def guard():
            now=time.monotonic();assert now-start<900
            if now-last_guard[0]>.5:t.d.m.p.lock_proof(owner,lock);t.d.m.p.clear_window();t.usage();last_guard[0]=now
        def load(row,code):
            path=Path(row['path']);before=t.identity(path);raw=path.read_bytes();assert len(raw)==row['bytes']<=16*2**20 and hashlib.sha256(raw).hexdigest()==row['sha256'];assert t.identity(path)==before
            assert len(raw)==math.prod(row['metadata']['shape'])*(2 if code=='H'else 4)
            a=array.array(code);a.frombytes(raw);assert sys.byteorder=='little';rows[str(path)]=dict(path=str(path),sha256=row['sha256'],bytes=len(raw),identity=before);return a
        lookup=tuple(struct.unpack('<f',struct.pack('<I',i<<16))[0]for i in range(65536))
        # Predeclared RNE tie checks, independent of observed projection outcomes.
        for bits,expected in [(0x3f808000,0x3f80),(0x3f818000,0x3f82),(0xbf808000,0xbf80),(0xbf818000,0xbf82)]:assert bf16(struct.unpack('<f',struct.pack('<I',bits))[0])==expected
        observed={role:{name:load(row,'H')for name,row in values.items()}for role,values in plan['payloads'].items()}
        weight=load(result['weight'],'H');assert hashlib.sha256(weight.tobytes()).hexdigest()==plan['weight']['sha256']
        spec=plan['weight'];t.d.m.same(t.identity(spec['file']['path']),spec['file']['identity'],'Original checkpoint unchanged')
        with open(spec['file']['path'],'rb')as stream:
            header_length=struct.unpack('<Q',stream.read(8))[0];assert 0<header_length<2**20;header=json.loads(stream.read(header_length));assert header==t.d.load(spec['header'])
            begin,end=header[spec['name']]['data_offsets'];assert end-begin==8*2**20;stream.seek(8+header_length+begin);assert stream.read(end-begin)==weight.tobytes()
        controls=t.d.load(result['controls']);assert controls['all3_fullshape_exact']
        byname={c['name']:c for c in result['cases']};assert list(byname)==['native_context','cpu_context'];audited=[];outputs_by_case={}
        for name,case in byname.items():
            outputs={key:load(row,'H')for key,row in case['captures'].items()};assert tuple(outputs)==t.FORMULAS
            partials_b=[load(row,'H')for row in case['partials']['bf16_partial']];partials_f=[load(row,'f')for row in case['partials']['fp32_partial']]
            if name=='native_context':
                assert outputs['bf16_full']==observed['native']['projection_output'];assert case['threads']==2 and case['length']==1231
                actual_input=observed['native']['context_before_o']
            else:
                assert outputs['bf16_split']==observed['cpu-rank0']['projection_output']==observed['cpu-rank1']['projection_output'];assert case['threads']==1 and case['length']==2048
                actual_input=array.array('H')
                for row in range(2048):
                    actual_input.extend(observed['cpu-rank0']['context_before_o'][row*1024:(row+1)*1024]);actual_input.extend(observed['cpu-rank1']['context_before_o'][row*1024:(row+1)*1024])
            assert hashlib.sha256(actual_input.tobytes()).hexdigest()==case['input']['sha256'];assert case['input']['shape']==[1,case['length'],2048]
            expected_elements=case['length']*2048;assert all(len(x)==expected_elements for x in [*outputs.values(),*partials_b,*partials_f])
            for i,(b0,b1,f0,f1)in enumerate(zip(*partials_b,*partials_f)):
                if i%32768==0:guard()
                assert all(math.isfinite(x)for x in [lookup[b0],lookup[b1],f0,f1])
                assert bf16(lookup[b0]+lookup[b1])==outputs['bf16_split'][i]==outputs['bf16_split_wide_sum'][i]
                assert bf16(f0+f1)==outputs['fp32_split_final'][i]
                assert bf16(lookup[bf16(f0)]+lookup[bf16(f1)])==outputs['fp32_split_round_local'][i]
            metrics={}
            for label,expected in case['comparisons'].items():
                a,b=label.split('_vs_');metrics[label]=compare(outputs[a],outputs[b],lookup,guard);metric_equal(metrics[label],expected)
            local=[]
            for rank in (0,1):
                rounded=array.array('H',(bf16(x)for x in partials_f[rank]));metric=compare(partials_b[rank],rounded,lookup,guard);metric_equal(metric,case['local_bf16_vs_fp32_cast'][rank]);local.append(metric)
            audited.append(dict(name=name,metrics=metrics,local_bf16_vs_fp32_cast=local,all_partial_rounding_equations_exact=True,elements=expected_elements));outputs_by_case[name]=outputs
            print(json.dumps(dict(stage='independent_case_complete',name=name)),flush=True)
        cross={}
        native=observed['native'];cpu=observed['cpu-rank0'];cpu_input=array.array('H')
        for row in range(1231):
            cpu_input.extend(observed['cpu-rank0']['context_before_o'][row*1024:(row+1)*1024]);cpu_input.extend(observed['cpu-rank1']['context_before_o'][row*1024:(row+1)*1024])
        for name,a,b in [('input',native['context_before_o'],cpu_input),('original_output',native['projection_output'],cpu['projection_output'][:1231*2048]),('cpu_context_full_vs_native',native['projection_output'],outputs_by_case['cpu_context']['bf16_full'][:1231*2048])]:
            actual=compare(a,b,lookup,guard)['full'];metric_equal({'full':actual},{'full':result['observed_comparisons'][name]});cross[name]=actual
        for row in rows.values():assert t.identity(row['path'])==row['identity']
        assert {n:t.d.m.p.sha(t.R/n)for n in t.d.m.CANONICAL}==t.d.m.CANONICAL
        assert not {'torch','numpy'}&set(sys.modules)
        terminal=t.d.once(Q/'minicpm5-frame1-fixed-oproj-r2-20261108-v1-root-terminal.json',dict(schema='minicpm5_fixed_oproj_independent_terminal_v1',complete=True,passed=True,reviewer='/root',verifier=owner,native_owner=result['owner'],lock_identity=lock,actual_exit=external_ref,result=ex['result'],plan=result['plan'],verifier_source=t.d.ref(Path(__file__).resolve()),cases=audited,observed_comparisons=cross,raw_files=list(rows.values()),raw_bytes=sum(x['bytes']for x in rows.values()),all3_original_fullshape_outputs_exact=True,all_saved_raw_files_authenticated=True,all_reported_metrics_independently_recomputed=True,all_partial_rounding_equations_exact=True,original8MiB_weight_independently_reread=True,matrix_multiplication_independently_reexecuted=False,mean_metric_comparison_tolerance=dict(relative=1e-12,absolute=1e-15),native_owner_gone=True,actual_self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False,measured_resources=dict(elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)))
        print(json.dumps(dict(terminal=terminal,owner=owner)),flush=True)
    finally:signal.alarm(0);t.d.m.old.post.external.release_lock(fd)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exit',required=True);p.add_argument('--exit-sha256',required=True);a=p.parse_args();run(dict(path=a.exit,sha256=a.exit_sha256))
