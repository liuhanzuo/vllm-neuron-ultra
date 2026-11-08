"""Independent full saved-byte review of actual two-rank gather; no Torch."""
import argparse,hashlib,json,math,os,resource,signal,struct,sys,time
from pathlib import Path
Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
import run_minicpm5_full_bf16_gather_v1 as t

def run(external):
    d=t.d;d.m.p.user_bus();fd,owner,lock=d.m.old.post.external.acquire_lock();started=time.monotonic()
    try:
        d.m.p.clear_window();d.m.p.active_heavy();ex=d.load(external);r=d.load(ex['result']);plan=d.load(r['plan']);prior=d.load(plan['original_plan']);saved=d.load(plan['original_result'])
        assert ex['actual_tool_exit_observed'] and ex['exact_owner_gone'] and ex['all_workers_gone'];d.actual_exit(ex['actual_exit']);d.m.p.gone(ex['owner'])
        assert r['passed'] and r['complete'] and ex['owner']==r['owner'] and r['actual_child_exit_codes']==[0,0]
        assert lock==ex['lock_identity']==plan['lock_identity'];assert plan['source_sha256']==t.sources()
        os.sched_setaffinity(0,{6});os.nice(19);resource.setrlimit(resource.RLIMIT_AS,(2*2**30,)*2);resource.setrlimit(resource.RLIMIT_CPU,(120,125));resource.setrlimit(resource.RLIMIT_FSIZE,(2**20,)*2)
        rows={};cases=[]
        def read(row):
            path=Path(row['path']);before=t.old.identity(path);raw=path.read_bytes();assert len(raw)==row['bytes']<=16*2**20
            assert hashlib.sha256(raw).hexdigest()==row['sha256'];assert t.old.identity(path)==before
            if 'identity' in row:assert row['identity']==before
            assert row['metadata']['dtype']=='torch.bfloat16';assert len(raw)==2*math.prod(row['metadata']['shape'])
            rows[str(path)]=dict(path=str(path),bytes=len(raw),sha256=row['sha256'],identity=before);return raw
        native=read(prior['payloads']['native']['context_before_o']);cpu=[read(prior['payloads']['cpu-rank'+str(rank)]['context_before_o']) for rank in (0,1)]
        expected_inputs=dict(native_context=native,cpu_context=b''.join(cpu[rank][i*2048:(i+1)*2048] for i in range(2048) for rank in (0,1)))
        ranks=[d.load(ref) for ref in r['ranks']]
        for rank,report in enumerate(ranks):
            assert report['complete'] and report['passed'] and report['rank']==rank and report['parent']==r['owner'];d.m.p.gone(report['owner'])
            assert report['actual_backend']=='gloo' and report['actual_global_rank_order']==[0,1] and report['explicit_subgroup']
            for case,expected in zip(report['cases'],saved['cases']):
                assert case['name']==expected['name'] and case['threads']==expected['threads'] and case['all_exact']
                full_input=expected_inputs[case['name']]
                if case['name']=='native_context':local=b''.join(native[i*4096+rank*2048:i*4096+(rank+1)*2048] for i in range(1231))
                else:local=cpu[rank]
                assert hashlib.sha256(local).hexdigest()==case['local_input']['sha256']
                assert case['partial']==expected['partials']['bf16_partial'][rank]['metadata']
                actual={k:read(row) for k,row in case['payloads'].items()};assert set(actual)=={'input','full','split'}
                assert actual['input']==full_input;assert case['payloads']['input']['metadata']==expected['input']
                for key,formula in [('full','bf16_full'),('split','bf16_split')]:
                    assert actual[key]==read(expected['captures'][formula]);assert case['payloads'][key]['metadata']==expected['captures'][formula]['metadata']
                cases.append(dict(rank=rank,name=case['name'],all_complete_bytes_exact=True,input_bytes=len(full_input),output_bytes=len(actual['full'])))
                d.m.p.lock_proof(owner,lock);t.usage()
        weight=read(saved['weight']);spec=prior['weight'];assert hashlib.sha256(weight).hexdigest()==spec['sha256']
        assert t.old.identity(spec['file']['path'])==spec['file']['identity']
        with open(spec['file']['path'],'rb') as f:
            n=struct.unpack('<Q',f.read(8))[0];assert 0<n<2**20;header=json.loads(f.read(n));assert header==d.load(spec['header'])
            start,end=header[spec['name']]['data_offsets'];assert end-start==8*2**20;f.seek(8+n+start);assert f.read(end-start)==weight
        for report in ranks:assert report['original_weight']==saved['weight']['metadata'] and report['original_weight_unchanged']
        for row in rows.values():assert t.old.identity(row['path'])==row['identity']
        assert not {'torch','numpy'}&set(sys.modules);assert {n:d.m.p.sha(t.R/n) for n in d.m.CANONICAL}==d.m.CANONICAL
        terminal=d.once(Q/'minicpm5-full-bf16-gather-r1-20261108-v1-root-terminal.json',dict(schema='minicpm5_full_bf16_gather_terminal_v1',passed=True,complete=True,reviewer='/root',verifier=owner,native_owner=r['owner'],worker_owners=[x['owner'] for x in ranks],lock_identity=lock,
            result=ex['result'],actual_exit=external,verifier_source=d.ref(Path(__file__).resolve()),cases=cases,raw_files=list(rows.values()),raw_bytes=sum(x['bytes'] for x in rows.values()),
            all12_new_payloads_independently_compared=True,input_order_independently_reconstructed=True,original8MiB_weight_independently_reread=True,matrix_multiplication_independently_reexecuted=False,
            model_executed=False,device_executed=False,full_model_qualified=False,new_speedup_claim=False,actual_self_exit_claimed=False,
            measured_resources=dict(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)))
        print(json.dumps(dict(terminal=terminal,owner=owner)),flush=True)
    finally:d.m.old.post.external.release_lock(fd)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exit',required=True);p.add_argument('--exit-sha256',required=True);a=p.parse_args();run(dict(path=a.exit,sha256=a.exit_sha256))
