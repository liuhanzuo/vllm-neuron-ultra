"""Fixed six-call matched BF16/S16 assembly leaf; no execution, native imports or IO on import.

Future root admission must provide the existing serial owner, exact local/SDK
source closure, SDKEnvironment, child ACK and original resources/drain guards.
This result is diagnostic completion only, never geometry/model qualification.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

from qwen35_kv256_alias_diagnostic_bytes_v1 import (ROWS,TOKENS,WIDTH,STATE_BYTES,PROFILES,
    pattern,compare,digest,require)

Q=Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
PACKAGE=Path('/home/ubuntu/src/vllm-neuron-ultra-qwen-writer-20261112')
PYTHON='/home/ubuntu/venvs/vllm_neuron/bin/python'
FLAGS=('--target=trn1','--auto-cast=none','--model-type=transformer','-O2','--lnc=1',
    '--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=1 --vectorize-strided-dma')
FIXED_SOURCES={
    str(PACKAGE/'vllm_neuron/models/qwen35/cache_writer.py'):'7ad18c9dff829ea3649f6968bf51b7696902d19d6ac035e9a63f4752dd19b66a',
    str(PACKAGE/'vllm_neuron/models/qwen35/slot_cache_nki_predicated.py'):'b2f0ef3a727409c9816dc9c5e77cfd13df3a1777f4cdc31c8ce3e8cd373eca7e',
    str(PACKAGE/'vllm_neuron/models/qwen35/text_core.py'):'3f0a966af288d87c55e72f5dd41b8f11f343b43d3c1f45385fa360d774aacdfb',
}
FRESH=(
    dict(name='bf16_cat',bucket=256,alias=1,operation='bf16_cat'),
    dict(name='s16_bitcast_cat',bucket=256,alias=1,operation='s16_bitcast_cat'),
)
GRAPHS=FRESH
RAW_CAP=708*2**20
FRESH_ARTIFACT_CAP=32*2**20
OUTPUT_CLASS_CAPS=dict(raw_and_bitmaps=RAW_CAP,fresh_artifacts=4*FRESH_ARTIFACT_CAP,
    all_other_outputs=188*2**20)


def workload():
    return dict(schema='qwen35_kv256_s16_pair_workload_v1',
        graphs=[dict(x,physical_output_order=physical_output_order(x))for x in GRAPHS],
        profiles=list(PROFILES),slots=[-1]*4,slots_dtype='torch.int32',state_shape=[4,1,4096,256],
        dtype='torch.bfloat16',calls=6,saved_graphs=0,new_graphs=2,raw_capture_cap_bytes=RAW_CAP,
        new_artifact_cap_bytes=FRESH_ARTIFACT_CAP,complete_owned_output_cap_bytes=2**30,
        output_class_caps=dict(OUTPUT_CLASS_CAPS),max_live_graphs=1,fresh_processes=1,
        per_leaf_seconds=900,full_geometry_qualified=False,full_model_qualified=False,new_speedup_claim=False)


def physical_output_order(graph):
    require(graph['name']in{g['name']for g in GRAPHS},'Exact diagnostic graph name')
    return ['new','old']if graph['alias']==0 else['old','new']


def normalize_outputs(graph, outputs):
    """Same logical old/new semantics even when new is physical output zero."""
    require(type(outputs)is tuple and len(outputs)==2,'Exact physical output pair')
    return tuple(reversed(outputs))if physical_output_order(graph)==['new','old']else outputs


def expected_reference(refs, name):
    require(name in('old','new','state'),'Exact logical output/state role')
    return refs['old'if name=='old'else'cache']


def sources(source_pins):
    mandatory={str(Q/name)for name in ('qwen35_kv256_s16_pair_leaf_v1.py',
        'qwen35_kv256_alias_diagnostic_bytes_v1.py','qwen35_kv256_s16_concat_control_v1.py','qwen35_kv256_s16_pair_hlo_v1.py')}
    require(type(source_pins)is dict and mandatory<=set(source_pins)
        and all(source_pins.get(p)==s for p,s in FIXED_SOURCES.items()),'Actual complete source bindings required')
    require(Path(__file__).resolve()==Q/'qwen35_kv256_s16_pair_leaf_v1.py'
        and Path(sys.modules['qwen35_kv256_alias_diagnostic_bytes_v1'].__file__).resolve()==
        Q/'qwen35_kv256_alias_diagnostic_bytes_v1.py','Actual pure leaf imports at exact source paths')
    checked={}
    for name in sorted(mandatory|set(FIXED_SOURCES)):
        p=Path(name)
        require(p.resolve()==p and not p.is_symlink()and p.stat().st_size<=4*2**20,'Exact bounded source path')
        actual=digest(p.read_bytes());require(actual==source_pins[name],'Source changed: '+name);checked[name]=actual
    return checked


def _file_ref(path, guard, cap):
    guard('before_artifact_read');p=Path(path);before=p.stat()
    require(p.resolve()==p and not p.is_symlink()and before.st_nlink==1
            and 0<before.st_size<=cap,'Exact finite singleton artifact')
    h=hashlib.sha256()
    with p.open('rb')as f:
        while True:
            guard('artifact_read');b=f.read(1<<20)
            if not b:break
            h.update(b)
    after=p.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Artifact changed during read')
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=before.st_size)


def run_diagnostic(directory, *, guard, save_report, record_artifact, source_pins):
    require(all(callable(x)for x in(guard,save_report,record_artifact)),'Original admitted callbacks required')
    require(sys.dont_write_bytecode and __debug__ and str(Path(sys.executable).absolute())==PYTHON,
        'Original -B/assertion-enabled SDK interpreter')
    started=time.monotonic()
    def check(event):
        guard(event);require(time.monotonic()-started<900,'Original leaf deadline retained')
    check('before_diagnostic');pins=sources(source_pins);root=Path(directory)
    require(root.is_absolute()and root.parent.is_dir()and not root.exists()and not root.is_symlink(),
        'Fresh diagnostic namespace under admitted output')
    root.mkdir();(root/'inputs').mkdir();(root/'calls').mkdir();(root/'graphs').mkdir()
    report=dict(schema='qwen35_kv256_s16_pair_result_v1',workload=workload(),complete=False,output_root=str(root),
        source_sha256=pins,graphs=[],calls=[],started_calls=0,returned_calls=0,raw_bytes_written=0,
        full_geometry_qualified=False,full_model_qualified=False,new_speedup_claim=False)
    save_report(report)
    def save_raw(path,data):
        check('before_raw_capture')
        require(type(data)is bytes and len(data)<=STATE_BYTES and path.is_relative_to(root), 'Bounded raw capture')
        require(report['raw_bytes_written']+len(data)<=RAW_CAP,'All raw/bitmap/input/host captures fit fixed cap')
        with path.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(fd)
        finally:os.close(fd)
        report['raw_bytes_written']+=len(data)
        return dict(path=str(path),sha256=digest(data),bytes=len(data))
    try:
        # Original SDK latch expects its first post-import guard to be a compile
        # event so it can install the accepted context restoration hook.
        check('before_native_imports')
        import torch
        import torch_neuronx
        check('before_compile:diagnostic:source_import')
        from vllm_neuron.models.qwen35.cache_writer import store_slots_nki_decode
        import qwen35_kv256_s16_concat_control_v1 as assembly_control
        from qwen35_kv256_s16_pair_hlo_v1 import assembly_hlo_proof
        require(Path(sys.modules[store_slots_nki_decode.__module__].__file__).resolve()==
            PACKAGE/'vllm_neuron/models/qwen35/cache_writer.py','Exact writer checkout import')
        require(Path(assembly_control.__file__).resolve()==Q/'qwen35_kv256_s16_concat_control_v1.py',
            'Exact original S16 assembly control source')
        require(Path(sys.modules['qwen35_kv256_s16_pair_hlo_v1'].__file__).resolve()==Q/'qwen35_kv256_s16_pair_hlo_v1.py',
            'Exact actual HLO proof module source')
        torch.set_num_threads(1)
        def tensor(data,shape):return torch.frombuffer(bytearray(data),dtype=torch.uint8).view(torch.bfloat16).reshape(shape)
        def raw(value):return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        def state(compiled):
            states=dict(compiled.states._parameters);require(len(states)==1,'One actual alias state')
            name,value=next(iter(states.items()))
            require(list(value.shape)==[4,1,4096,256]and value.dtype==torch.bfloat16
                and value.device.type=='privateuseone','Full actual original BF16 state')
            return name,value
        inputs={}
        slots_bytes=struct.pack('<iiii',-1,-1,-1,-1)
        slots_ref=save_raw(root/'inputs'/'slots-int32.raw',slots_bytes)
        for profile in PROFILES:
            cache=pattern(profile,TOKENS*WIDTH,salt=17)
            old=cache[:STATE_BYTES//4]*4
            values=dict(cache=cache,old=old,updates={})
            refs=dict(cache=save_raw(root/'inputs'/(profile+'-cache.raw'),cache),
                old=save_raw(root/'inputs'/(profile+'-expected-old.raw'),old),updates={},slots=slots_ref)
            for bucket in(256,):
                update=pattern(profile,bucket*WIDTH,salt=31);values['updates'][bucket]=update
                refs['updates'][str(bucket)]=save_raw(root/'inputs'/(profile+'-updates'+str(bucket)+'.raw'),update)
            inputs[profile]=(values,refs)
        require(inputs['raw'][1]['cache']['sha256']=='585fbf90d9c212afe9be4e1620de1c21d0aaaee82186b012b926a80d34da71c4',
            'Original failing full raw payload reconstructed exactly')
        report['inputs']={p:v[1]for p,v in inputs.items()};save_report(report)
        def module(graph):
            assembler=(lambda prefix,tail:torch.cat((prefix,tail),dim=2)) if graph['operation']=='bf16_cat' else assembly_control.make_assembler()
            initial=tensor(inputs['finite'][0]['cache'],[4,1,4096,256])
            return assembly_control._build_control(torch,assembler,initial,store_slots_nki_decode)
        with torch.inference_mode():
            for graph in GRAPHS:
                check('before_compile:'+graph['name']);g=dict(graph,calls=0,
                    physical_output_order=physical_output_order(graph))
                work=root/'graphs'/graph['name'];work.mkdir()
                source=module(graph)
                example=(tensor(inputs['finite'][0]['updates'][256],[4,1,256,256]),torch.arange(4,dtype=torch.int32))
                compiled=torch_neuronx.trace(source,example,input_output_aliases=assembly_control.alias_map(source),
                    compiler_workdir=str(work),compiler_args=list(FLAGS))
                check('after_compile:'+graph['name']);p=work/'model.pt';compiled.save(str(p))
                artifact=_file_ref(p,check,FRESH_ARTIFACT_CAP);g['artifact']=artifact;g['origin']='fresh_control'
                # The pinned SDK setup_compiler_dirs writes the actual compiler
                # input module here. Retain it, never synthesize a proof-only HLO.
                hlo_path=work/'model'/'graph.hlo'
                hlo=_file_ref(hlo_path,check,16*2**20)
                hlo_bytes=hlo_path.read_bytes()
                require(digest(hlo_bytes)==hlo['sha256'],'Actual compiler input HLO stable')
                g['hlo']=hlo
                report['graphs'].append(g);save_report(report)
                g['hlo_proof']=assembly_hlo_proof(hlo_bytes,graph['name'])
                require(_file_ref(hlo_path,check,16*2**20)==hlo,'Compiler input HLO unchanged during proof')
                if len(report['graphs'])==2:
                    require(report['graphs'][0]['hlo_proof']['writer_func_literal_sha256']==g['hlo_proof']['writer_func_literal_sha256'],
                        'Matched arms have identical original NKI writer literal')
                del compiled,source;gc.collect()
                record_artifact(dict(name=graph['name'],artifact=g['artifact'],origin=g['origin'],bucket=graph['bucket'],alias=graph['alias'],hlo=g['hlo'],hlo_proof=g['hlo_proof']))
                save_report(report);check('before_device_load:'+graph['name'])
                compiled=torch.jit.load(g['artifact']['path']);torch_neuronx.move_trace_to_device(compiled,0)
                require(_file_ref(g['artifact']['path'],check,128*2**20)==g['artifact'],'Artifact stable through loading')
                persistent=graph['alias']is not None
                binding=None
                if persistent:
                    key,current=state(compiled);binding=(key,int(current._cdata),str(current.device))
                for profile in PROFILES:
                    check('before_call:'+graph['name']+':'+profile)
                    values,refs=inputs[profile];cache=tensor(values['cache'],[4,1,4096,256])
                    updates=tensor(values['updates'][graph['bucket']],[4,1,graph['bucket'],256])
                    slots=torch.tensor([-1]*4,dtype=torch.int32)
                    require(raw(slots)==slots_bytes and str(slots.dtype)=='torch.int32','Original int32 inactive slots')
                    row=dict(graph=graph['name'],profile=profile,inputs=refs,slots=[-1]*4,bucket=graph['bucket'],
                        alias=graph['alias'],call_returned=False,proofs={},host_inputs_unchanged=[])
                    row['physical_output_order']=physical_output_order(graph)
                    call_dir=root/'calls'/(graph['name']+'-'+profile);call_dir.mkdir()
                    report['calls'].append(row)
                    if persistent:
                        state(compiled)[1].copy_(cache);reset=raw(state(compiled)[1])
                        row['reset_exact']=reset==values['cache'];row['reset_sha256']=digest(reset)
                        if not row['reset_exact']:
                            row['reset_actual']=save_raw(call_dir/'reset-failed.raw',reset);save_report(report)
                            raise ValueError('Actual state reset changed raw bytes before diagnostic call')
                    args=(updates,slots)if persistent else(cache,updates,slots)
                    before=[raw(x)for x in args]
                    require(before==[values['updates'][256],slots_bytes],'Exact pre-call host payloads')
                    report['started_calls']+=1
                    save_report(report)
                    outputs=tuple(x.detach().cpu().clone()for x in compiled(*args))
                    report['returned_calls']+=1;row['call_returned']=True;g['calls']+=1
                    save_report(report)
                    require(len(outputs)==2 and all(list(x.shape)==[4,1,4096,256]and x.dtype==torch.bfloat16 for x in outputs),
                        'Complete original output arity/shape/precision')
                    outputs=normalize_outputs(graph,outputs)
                    actuals={'old':raw(outputs[0]),'new':raw(outputs[1])}
                    expected={'old':values['old'],'new':values['cache']}
                    if persistent:
                        key,current=state(compiled)
                        require((key,int(current._cdata),str(current.device))==binding,'Alias state binding changed')
                        actuals['state']=raw(current);expected['state']=values['cache']
                    for name,data in actuals.items():
                        actual_ref=save_raw(call_dir/(name+'.raw'),data)
                        proof,bitmap=compare(data,expected[name],graph['bucket'],guard=lambda:check('difference_scan'))
                        proof.update(actual=actual_ref,expected=expected_reference(refs,name),
                            difference_bitmap=save_raw(call_dir/(name+'.diffbits'),bitmap))
                        row['proofs'][name]=proof
                    row['host_inputs_unchanged']=[raw(x)==b for x,b in zip(args,before)]
                    row['host_inputs_after']=[save_raw(call_dir/('host-input-'+str(i)+'.raw'),raw(x))for i,x in enumerate(args)]
                    for i,(x,same)in enumerate(zip(args,row['host_inputs_unchanged'])):
                        if not same:row.setdefault('changed_host_inputs',{})[str(i)]=row['host_inputs_after'][i]
                    row['all_expected_bytes_equal']=all(x['identical']for x in row['proofs'].values())and all(row['host_inputs_unchanged'])
                    # A completed mismatch is evidence; retain it and run the
                    # remaining fixed controls. Native/resource errors still abort.
                    save_report(report);check('after_call:'+graph['name']+':'+profile)
                require(_file_ref(g['artifact']['path'],check,128*2**20)==g['artifact'],'Artifact stable after calls')
                del compiled;gc.collect()
        require(report['started_calls']==report['returned_calls']==6 and len(report['graphs'])==2
            and all(g['calls']==3 for g in report['graphs']),'All exact controls and profiles completed')
        require(sources(source_pins)==pins,'All actual leaf sources unchanged')
        report.update(complete=True,diagnostic_completed=True,
            s16_all_profiles_byte_exact=all(row['all_expected_bytes_equal']for row in report['calls']if row['graph']=='s16_bitcast_cat'),
            both_finite_profiles_byte_exact=all(row['all_expected_bytes_equal']for row in report['calls']if row['profile']=='finite'),
            original_missing_controls_replaced=False,
            mismatch_calls=sum(not row['all_expected_bytes_equal']for row in report['calls']),
            original_failed_r3_result_preserved=True,independent_actual_exit_audit_required=True)
    except BaseException as error:
        report['error']=repr(error);raise
    finally:save_report(report)
    return report


def validate_report(value):
    require(value['schema']=='qwen35_kv256_s16_pair_result_v1'and value['workload']==workload(),
        'Exact diagnostic identity; never original geometry schema')
    require(value['complete']is True and value['diagnostic_completed']is True
        and value['started_calls']==value['returned_calls']==6 and len(value['graphs'])==2,
        'Complete fixed diagnostic, including mismatching outcomes')
    require([g['name']for g in value['graphs']]==[g['name']for g in GRAPHS]
        and all(g['calls']==3 for g in value['graphs']),'Exact matched two-graph order')
    wanted=[(g['name'],p)for g in GRAPHS for p in PROFILES]
    require([(r['graph'],r['profile'])for r in value['calls']]==wanted,'No omitted or duplicated calls')
    require(set(value['inputs'])==set(PROFILES),'All exact input profiles')
    root=Path(value['output_root'])
    require(root.is_absolute()and '..'not in root.parts,'Canonical absolute capture namespace')
    require(type(value['mismatch_calls'])is int and value['mismatch_calls']==
        sum(not r['all_expected_bytes_equal']for r in value['calls']),'Exact mismatch count')
    require(value['s16_all_profiles_byte_exact']is all(r['all_expected_bytes_equal']for r in value['calls']if r['graph']=='s16_bitcast_cat')
        and value['both_finite_profiles_byte_exact']is all(r['all_expected_bytes_equal']for r in value['calls']if r['profile']=='finite')
        and value['original_missing_controls_replaced']is False,'No mismatch promoted or missing control replaced')
    for actual,graph in zip(value['graphs'],GRAPHS):
        require(all(actual.get(k)==v for k,v in graph.items()),'Exact declared graph controls')
        require(actual['physical_output_order']==physical_output_order(graph),'Exact physical graph output order')
        a=actual['artifact'];require(set(a)=={'path','sha256','size_bytes'},'Exact graph artifact fields')
        require(actual['origin']=='fresh_control'and a['path']==str(root/'graphs'/graph['name']/'model.pt')
            and type(a['size_bytes'])is int and 0<a['size_bytes']<=FRESH_ARTIFACT_CAP,'Bounded fresh control artifact')
        hlo=actual['hlo'];proof=actual['hlo_proof']
        require(set(hlo)=={'path','sha256','size_bytes'} and hlo['path']==str(root/'graphs'/graph['name']/'model'/'graph.hlo')
            and 0<hlo['size_bytes']<=16*2**20 and proof['arm']==graph['name'] and proof['hlo_sha256']==hlo['sha256']
            and proof['hlo_bytes']==hlo['size_bytes'] and proof['state_alias_output']==1
            and proof['concat_dtype']==('bf16'if graph['name']=='bf16_cat'else's16')
            and proof['numeric_converts_in_assembly']==0,'Actual bounded HLO and matching dataflow proof')
    require(value['graphs'][0]['hlo_proof']['writer_func_literal_sha256']==value['graphs'][1]['hlo_proof']['writer_func_literal_sha256'],
        'Same original NKI writer literal in both arms')
    for row in value['calls']:
        g=next(g for g in GRAPHS if g['name']==row['graph'])
        require(row['call_returned']is True and row['slots']==[-1]*4 and row['bucket']==g['bucket']
            and row['alias']==g['alias']and row['physical_output_order']==physical_output_order(g),
            'Exact actual completed call and physical output order')
        keys={'old','new'}|({'state'}if g['alias']is not None else set())
        require(set(row['proofs'])==keys and len(row['host_inputs_unchanged'])==(2 if g['alias']is not None else 3),
            'Complete old/new/actual-state and host input checks')
        refs=value['inputs'][row['profile']]
        require(row['inputs']==refs and all(type(x)is bool for x in row['host_inputs_unchanged']),
            'Exact profile inputs and actual boolean host checks')
        if g['alias']is not None:
            require(row['reset_exact']is True and row['reset_sha256']==refs['cache']['sha256'],'Exact reset before call')
        changed={str(i)for i,same in enumerate(row['host_inputs_unchanged'])if not same}
        require(set(row.get('changed_host_inputs',{}))==changed,'Every mutated host input retained')
        require(len(row['host_inputs_after'])==2,'Both post-call host inputs fully captured')
        for index,host in enumerate(row['host_inputs_after']):
            require(host['path']==str(root/'calls'/(g['name']+'-'+row['profile'])/('host-input-'+str(index)+'.raw'))
                and host['bytes']==(524288 if index==0 else 16),'Exact complete post-call host capture')
        for name,proof in row['proofs'].items():
            call_dir=root/'calls'/(g['name']+'-'+row['profile'])
            require(proof['actual']['path']==str(call_dir/(name+'.raw'))
                and proof['difference_bitmap']['path']==str(call_dir/(name+'.diffbits'))
                and proof['expected']==expected_reference(refs,name),
                'Exact expected bytes and all call capture paths')
            require(proof['bytes_compared']==STATE_BYTES and proof['actual']['bytes']==STATE_BYTES
                and proof['bitmap_bytes']==STATE_BYTES//8 and proof['difference_bitmap']['bytes']==STATE_BYTES//8,
                'Full raw and all-position bitset captures')
            require(proof['actual_sha256']==proof['actual']['sha256']
                and proof['expected_sha256']==proof['expected']['sha256']
                and proof['bitmap_sha256']==proof['difference_bitmap']['sha256'],'Exact capture references')
            require(sum(x['differing_bytes']for x in proof['partitions'])==proof['differing_bytes']
                and sum(x['differing_words']for x in proof['partitions'])==proof['differing_words'],
                'Complete per-row prefix/tail accounting')
            require(type(proof['differing_bytes'])is int and type(proof['differing_words'])is int
                and 0<=proof['differing_words']<=proof['differing_bytes']<=2*proof['differing_words']<=STATE_BYTES
                and proof['identical']is(proof['differing_bytes']==0)
                and proof['identical']is(proof['actual_sha256']==proof['expected_sha256']),
                'Exact finite counts and equality flags')
            require([(p['row'],p['partition'],p['start_byte'],p['end_byte_exclusive'])for p in proof['partitions']]==
                [(r,part,(r*TOKENS+lo)*WIDTH*2,(r*TOKENS+hi)*WIDTH*2)
                 for r in range(ROWS)for part,lo,hi in(('prefix',0,g['bucket']),('tail',g['bucket'],TOKENS))],
                'Every prefix and tail of every row has exact coordinates')
        require(row['all_expected_bytes_equal']is(all(p['identical']for p in row['proofs'].values())
            and all(row['host_inputs_unchanged'])),'Mismatch retained honestly')
    require(type(value['raw_bytes_written'])is int and 0<value['raw_bytes_written']<=RAW_CAP
        and value['original_failed_r3_result_preserved']is True
        and value['independent_actual_exit_audit_required']is True and value['full_geometry_qualified']is False
        and value['full_model_qualified']is False and value['new_speedup_claim']is False,'No diagnostic promotion')
    return dict(diagnostic_completed=True,graphs=2,calls=6,mismatch_calls=value['mismatch_calls'],
        full_geometry_qualified=False,full_model_qualified=False,new_speedup_claim=False)


def audit_saved_bytes(value, *, guard):
    """Admitted separate post-exit reader: all saved bytes and bitmaps rederived.

    This reads real raw captures only when explicitly called in the serial audit
    slot. Source tests use only bounded synthetic callback examples instead.
    """
    validate_report(value);root=Path(value['output_root'])
    from qwen35_kv256_s16_pair_hlo_v1 import assembly_hlo_proof
    for graph in value['graphs']:
        hlo=graph['hlo'];p=Path(hlo['path'])
        require(_file_ref(p,lambda _:guard(),16*2**20)==hlo,'Entire saved HLO identity')
        payload=p.read_bytes();require(digest(payload)==hlo['sha256'],'Stable saved HLO')
        require(assembly_hlo_proof(payload,graph['name'])==graph['hlo_proof'],'Saved HLO dataflow proof independently reproduced')
    require(root.is_absolute()and root.resolve()==root,'Exact admitted diagnostic capture root')
    def read(ref,role):
        require(set(ref)=={'path','sha256','bytes'}and 0<ref['bytes']<=STATE_BYTES,'Exact bounded raw reference')
        p=Path(ref['path']);require(p.is_relative_to(root/role),'Capture outside its declared root role')
        guard();info=_file_ref(p,lambda _:guard(),STATE_BYTES)
        require(info==dict(path=ref['path'],sha256=ref['sha256'],size_bytes=ref['bytes']),'Full capture digest')
        data=p.read_bytes();require(digest(data)==ref['sha256'],'Capture changed before byte comparison')
        guard();return data
    unique={}
    def capture(ref,role):
        data=read(ref,role)
        require(ref['path']not in unique or unique[ref['path']]==ref,'One exact identity per captured file')
        unique[ref['path']]=ref
        return data
    for profile,refs in value['inputs'].items():
        expected_cache=pattern(profile,TOKENS*WIDTH,salt=17)
        require(capture(refs['cache'],'inputs')==expected_cache,'Original or separately tagged cache pattern')
        require(capture(refs['old'],'inputs')==expected_cache[:STATE_BYTES//4]*4,'Clamped inactive old gather oracle')
        require(capture(refs['slots'],'inputs')==struct.pack('<iiii',-1,-1,-1,-1),'Original int32 inactive slots')
        require(set(refs['updates'])=={'256'},'Exact KV256 update shape')
        for bucket in(256,):
            require(capture(refs['updates'][str(bucket)],'inputs')==pattern(profile,bucket*WIDTH,salt=31),
                'Original or separately tagged update pattern')
    comparisons=0
    for row in value['calls']:
        for proof in row['proofs'].values():
            actual=capture(proof['actual'],'calls');expected=capture(proof['expected'],'inputs')
            bitmap=capture(proof['difference_bitmap'],'calls')
            observed,actual_bitmap=compare(actual,expected,row['bucket'],guard=guard)
            require(bitmap==actual_bitmap and observed=={k:v for k,v in proof.items()
                if k not in ('actual','expected','difference_bitmap')},'Complete raw difference evidence does not reproduce')
            comparisons+=1
        originals=([row['inputs']['cache']]if row['alias']is None else[])+[
            row['inputs']['updates'][str(row['bucket'])],row['inputs']['slots']]
        require(len(row['host_inputs_after'])==2,'Both complete post-call host captures')
        for index,ref in enumerate(row['host_inputs_after']):
            require(ref['path']==str(root/'calls'/(row['graph']+'-'+row['profile'])/('host-input-'+str(index)+'.raw')),'Exact host capture path')
            require((capture(ref,'calls')==capture(originals[index],'inputs'))is row['host_inputs_unchanged'][index],'Host-input equality independently reproduced')
        for index,ref in row.get('changed_host_inputs',{}).items():
            require(ref['path']==str(root/'calls'/(row['graph']+'-'+row['profile'])/('host-input-'+index+'.raw')),
                'Exact changed host capture path')
            require(capture(ref,'calls')!=capture(originals[int(index)],'inputs'),'Changed host evidence reproduces')
    require(comparisons==18,'All old/new/state comparisons independently reread')
    require(sum(ref['bytes']for ref in unique.values())==value['raw_bytes_written'],'Every raw byte accounted once')
    return dict(comparisons=18,all_saved_raw_and_difference_bits_reproduced=True,
        full_geometry_qualified=False,full_model_qualified=False,new_speedup_claim=False)
