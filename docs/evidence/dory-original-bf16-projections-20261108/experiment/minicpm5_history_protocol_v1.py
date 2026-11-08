"""MiniCPM5 original32 CPU continuation with authenticated frozen v2 control."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parent
SCHEMA = 'minicpm5-history-crossover-v1'
DEST = ROOT / SCHEMA
DATA = Path('/mnt/nvme') / SCHEMA
OLD = ROOT / 'minicpm5-baseline-full-v5'
V4 = ROOT / 'minicpm5-baseline-full-v4'
ARTIFACT = Path('/mnt/nvme/neuron-artifacts/latest-20261031-minicpm5-2b-bf16-cpu-tp2-b4-ctx4096-baseline-v4')
SHARDS = ARTIFACT / 'weights'
HF = V4 / 'hf-reference.pt'
PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
SCRIPT = ROOT / 'run_minicpm5_history_v1.py'
CPU_SCRIPT = ROOT / 'run_minicpm5_history_cpu_v1.py'
CONTROLLER_SCRIPT = ROOT / 'run_minicpm5_history_queue_v1.py'
QUEUE = 'trn1-minicpm5-history-crossover-v1-queue'
import minicpm5_history_contract_v1 as contract_spec
PHASES=contract_spec.PHASES
CPU_PHASES=contract_spec.CPU_PHASES
UNITS={phase:'trn1-latest-verify-'+SCHEMA+'-'+phase for phase in PHASES}
RESOURCES={phase:((26,24,8,[6,7],43200) if phase in CPU_PHASES else (26,24,8,[7],43200) if phase=='device_crossover' else (6,5,0,[7],43200)) for phase in PHASES}
VOCAB = 130560
BUCKETS = (128, 256, 512, 1024, 2048, 4096)
CRITERIA = {'cosine_min': .995, 'kl_max_nats': .02, 'max_abs_error_max': 2.,
            'argmax_match_fraction_min': .9, 'all_reference_margins_gt_0.5_must_match': True}
INPUT_NAMES = ('input_ids','attention_mask','position_ids','seq_ids','sampling_params','hidden_states','adapter_ids')
CACHE_BYTES = 4 * 1 * 4224 * 128 * 2
SNAPSHOT_BYTES = 168 * CACHE_BYTES
RAW_STATE_BYTES=contract_spec.RAW_POSTSTATE_BYTES
STATE_BUDGET=RAW_STATE_BYTES
EXTRA_BUDGET=contract_spec.EXTRA_BYTES
NVME_FLOOR=contract_spec.NVME_FLOOR
ROOT_FLOOR=contract_spec.ROOT_FLOOR
START_NVME=contract_spec.START_NVME


def no_symlink(path):
    path = Path(path).absolute()
    assert all(not x.is_symlink() for x in (path, *path.parents)), str(path)
    return path


def read(path):
    path = no_symlink(path); assert path.stat().st_size < 32 * 1024**2
    return json.loads(path.read_text())


def sha(path):
    path = no_symlink(path); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def text_sha(path):
    path = no_symlink(path); assert path.stat().st_size < 32 * 1024**2
    assert path.suffix in ('.py','.json','.txt','.md','.log','.pth','.jinja','.metadata') or path.name == 'RECORD'
    data = path.read_bytes(); data.decode('utf-8'); return hashlib.sha256(data).hexdigest()


def identity(path):
    path = no_symlink(path); s = path.stat()
    return dict(realpath=str(path.resolve()),device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def once(path, value):
    path = no_symlink(path)
    with path.open('x') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def save(path, value):
    path = no_symlink(path); tmp = no_symlink(path.with_suffix('.tmp'))
    with tmp.open('x') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def stages():
    from minicpm5_history_model_v1 import stages as names
    return names()


def layout():
    return [dict(rank=rank,name=f'kv_mgr.past_key_values.{2*layer+side}',shape=[4,1,4224,128],
        dtype='torch.bfloat16',layer=layer,kv='key' if side==0 else 'value')
        for rank in range(2) for layer in range(42) for side in range(2)]


def calls():
    fixtures = read(OLD/'hf-reference.json')['fixtures']
    assert [x['id'] for x in fixtures] == list(range(8))
    assert [len(x['input_ids']) for x in fixtures] == [24,28,28,27,27,27,35,1231]
    result=[]; frame=0; slots=[3,1,0,2]
    for group,order in enumerate(([0,7,2,5],[4,1,6,3])):
        for fixture,slot in zip(order,slots):
            result.append(dict(index=len(result),group=group,phase='prefill',frames=[dict(index=frame,fixture=fixture,slot=slot,step=0)],logical_positions=[len(fixtures[fixture]['input_ids'])-1])); frame+=1
        for step in range(1,4):
            result.append(dict(index=len(result),group=group,phase='decode',frames=[dict(index=frame+i,fixture=f,slot=s,step=step) for i,(f,s) in enumerate(zip(order,slots))],logical_positions=[len(fixtures[f]['input_ids'])+step-1 for f in order])); frame+=4
    assert len(result)==14 and frame==32
    return result


def check_input_metadata(rows, phase):
    assert phase in ('prefill','decode') and len(rows)==7
    batch=1 if phase=='prefill' else 4; bucket=rows[1]['shape'][1]
    assert bucket in BUCKETS
    active=bucket if phase=='prefill' else 1
    shapes=[[batch,active],[batch,bucket],[batch,active],[batch],[batch,3],[batch],[batch]]
    for i,(row,shape) in enumerate(zip(rows,shapes)):
        assert row['shape']==shape and row['dtype']==('torch.float32' if i==4 else 'torch.int32')
        assert len(row['sha256'])==64
    assert all(row['sha256']==hashlib.sha256(bytes(batch*4)).hexdigest() for row in rows[5:])
    return bucket


def failures(rows):
    return [x for x in rows if x.get('nonfinite',False) or x['cosine'] < .995 or x['kl_reference_to_neuron'] > .02 or x['max_abs_error'] > 2 or (x['reference_margin'] > .5 and not x['argmax_match'])]


def original():
    row=read(OLD/'real-logits-validation.json')
    assert row['complete'] and row['passed'] is False and row['frames']==32 and len(row['routes'])==14
    assert row['criteria']==CRITERIA and len(failures(row['rows']))==15
    expected=[(f['fixture'],f['slot'],f['step']) for c in calls() for f in c['frames']]
    assert [(x['fixture'],x['slot'],x['step']) for x in row['rows']]==expected
    assert sum(x['argmax_match'] for x in row['rows'])==31
    return row


def contract():
    return dict(schema=SCHEMA,phases=list(PHASES),resources={k:list(v) for k,v in RESOURCES.items()},
      calls=calls(),state_layout=layout(),stages=stages(),criteria=CRITERIA,scope=contract_spec.scope(),
      graph_input_names=list(INPUT_NAMES),graph_input_count=7,
      serial_controller=dict(script=str(CONTROLLER_SCRIPT),unit=QUEUE,argv=[str(CONTROLLER_SCRIPT),'--queue']),
      original_graphs_recompiled=False,full_model_qualified=False,new_speedup_claim=False,
      all_intervention_scoring_after_all14_device_control_restorations=True,
      cpu_environment_transition='Exact preimport cores0-1 to pinned SDK rank0/1; no environment repair')


def verify():
    plan=read(DEST/'plan.json'); expected=contract()
    assert all(plan.get(k)==v for k,v in expected.items())
    assert plan['prior_tail']['queue'] in plan['predecessors']
    assert plan['review']['passed'] and plan['review']['source_only']
    assert text_sha(plan['review']['path'])==plan['review']['sha256']
    review=read(plan['review']['path'])
    assert all(review.get(k) is True for k in ('passed','source_only','independent_review','suitable_for_enrollment_as_independent_review','original_gates_preserved'))
    assert all(review['source_sha256'].get(str(ROOT/name))==plan['source_sha256'][str(ROOT/name)] for name in source_names())
    assert review['source_sha256'][str(CONTROLLER_SCRIPT)]==plan['source_sha256'][str(CONTROLLER_SCRIPT)]
    from prepare_minicpm5_history_v1 import source_paths
    assert all(review['source_sha256'].get(str(path))==plan['source_sha256'].get(str(path)) for path in source_paths())
    for name,digest in plan['source_sha256'].items(): assert text_sha(name)==digest,name
    for name,row in plan['payloads'].items(): assert identity(name)==row['identity'],name
    for row in plan['legacy_source_aliases']:
        assert str(Path(row['path']).resolve())==row['target']
        assert plan['source_sha256'][row['target']]==row['sha256']
    from minicpm5_history_reuse_v1 import inputs as reused_inputs
    reused = reused_inputs()
    assert plan['reused_control']==reused['reused_control'] and plan['reused_cpu']==reused['reused_cpu']
    assert all(plan['source_sha256'].get(name) == value for name,value in reused['source_sha256'].items())
    assert all(plan['payloads'].get(name) == value for name,value in reused['payloads'].items())
    original(); return plan


def source_names():
    return ('minicpm5_history_contract_v1.py','minicpm5_history_protocol_v1.py',
      'minicpm5_history_reuse_v1.py','minicpm5_history_results_v1.py','minicpm5_history_model_v1.py','minicpm5_history_environment_v1.py',
      'minicpm5_history_tensors_v1.py','run_minicpm5_history_cpu_v1.py','minicpm5_history_device_v1.py',
      'analyze_minicpm5_history_v1.py','run_minicpm5_history_v1.py','check_minicpm5_history_v1.py')


def admitted(plan):
    row=read(DEST/'admission.json'); assert row['passed'] and row['plan_sha256']==text_sha(DEST/'plan.json')
    assert row['sha256']=={name:r['sha256'] for name,r in plan['payloads'].items()}


def receipt(phase):
    assert phase in PHASES
    row=read(DEST/(phase+'-result.json'))
    assert row['phase']==phase and row['passed'] and row['plan_sha256']==text_sha(DEST/'plan.json')
    for name,digest in row['outputs'].items(): assert text_sha(name)==digest
    return row


def prerequisites(phase):
    for earlier in PHASES[:PHASES.index(phase)]: receipt(earlier)


def finish(phase, paths):
    once(DEST/(phase+'-result.json'),dict(passed=True,phase=phase,plan_sha256=text_sha(DEST/'plan.json'),outputs={str(x):text_sha(x) for x in paths}))


def heavy_units():
    raw=subprocess.check_output(['systemctl','--user','list-units','--type=service','--state=running,activating,deactivating','--plain','--no-legend','--no-pager'],text=True)
    names=[line.split()[0] for line in raw.splitlines() if line.split()]
    return sorted(x for x in names if x.startswith(('trn1-bench-','trn1-latest-verify-','trn1-quant-','trn1-direct-')) or (x.startswith('trn1-') and x.endswith('-worker.service')))


def start(pid):
    try: return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError: return None


def guard(phase, child=False):
    assert __debug__ and phase in PHASES
    assert not any(x in sys.modules for x in ('torch','numpy','transformers')) or os.environ.get('TRN1_MINICPM5_HISTORY_PHASE')==phase
    prefix=DEST/('pycache-'+phase)
    assert os.environ.get('PYTHONPYCACHEPREFIX')==sys.pycache_prefix==str(prefix)
    assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1' and sys.dont_write_bytecode and not prefix.exists() and not prefix.is_symlink()
    assert os.environ.get('PYTHONOPTIMIZE')=='0' and os.environ.get('PYTHONNOUSERSITE')=='1'
    assert sys.flags.optimize==0 and not any(os.environ.get(k) for k in ('LD_PRELOAD','PYTHONHOME','PYTHONSTARTUP','PYTHONPATH'))
    assert Path(sys.executable).resolve()==Path(PYTHON).resolve() and Path(sys.prefix)==Path(PYTHON).parent.parent
    assert os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1'
    from minicpm5_history_environment_v1 import guard as environment_guard
    environment_guard(phase, child)
    assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    assert not any(os.environ.get(k) for k in ('TRN1_CPU_SAMPLER_FAST','TRN1_SMOLLM3_MIN_DECODE_BUCKET','TRN1_MINICPM5_NKI_CACHE'))
    assert (os.environ.get('NXD_CPU_MODE')=='1')==(phase in CPU_PHASES)
    unit=UNITS[phase]; cgroup=Path('/proc/self/cgroup').read_text().strip()
    assert cgroup.endswith('/'+unit+'.service') and heavy_units()==[unit+'.service']
    main=int(subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','--value']))
    main_start=start(main); assert main_start
    args=[x.decode() for x in Path('/proc',str(main),'cmdline').read_bytes().split(b'\0') if x]
    assert args[args.index(str(SCRIPT)):]==[str(SCRIPT),phase]
    own=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x]
    if child:
        assert phase in CPU_PHASES and os.getppid()==main and os.environ.get('WORLD_SIZE')=='2'
        assert int(os.environ['RANK']) in (0,1) and own[own.index(str(CPU_SCRIPT)):]==[str(CPU_SCRIPT),phase]
    else: assert main==os.getpid() and own[own.index(str(SCRIPT)):]==[str(SCRIPT),phase]
    maximum,high,swap,cpus,_=RESOURCES[phase]
    cg=Path('/sys/fs/cgroup')/cgroup.split(':',2)[2].lstrip('/')
    assert [(cg/k).read_text().strip() for k in ('memory.max','memory.high','memory.swap.max')]==[str(n*1024**3) for n in (maximum,high,swap)]
    assert os.sched_getaffinity(0)==set(cpus) and os.getpriority(os.PRIO_PROCESS,0)==19
    assert start(main)==main_start
    plan=verify(); prerequisites(phase)
    admission=read(DEST/'serial-admission.json')
    assert admission['passed'] and admission['plan_sha256']==text_sha(DEST/'plan.json')
    assert admission['predecessors']==plan['predecessors']
    owner=admission['owner']; assert read(DEST/'queue-identity.json')==owner
    assert owner['script']==str(CONTROLLER_SCRIPT) and owner['unit']==QUEUE
    assert start(owner['pid'])==str(owner['start_ticks'])
    owner_args=Path('/proc',str(owner['pid']),'cmdline').read_bytes().split(b'\0')
    assert owner_args[-3:]==[str(CONTROLLER_SCRIPT).encode(),b'--queue',b'']
    assert Path('/proc',str(owner['pid']),'cgroup').read_text().strip().endswith('/'+owner['unit'].removesuffix('.service')+'.service')
    assert int(subprocess.check_output(['systemctl','--user','show',owner['unit'],'-p','MainPID','--value']))==owner['pid']
    assert start(owner['pid'])==str(owner['start_ticks'])
    assert all(start(x['pid'])!=str(x['start_ticks']) for x in plan['predecessors'])
    if phase!='admit': admitted(plan)
    return plan


def storage_check(startup=False):
    no_symlink(DATA); no_symlink(DEST)
    assert shutil.disk_usage(DATA.parent).free >= (START_NVME if startup else NVME_FLOOR)
    assert shutil.disk_usage(ROOT).free >= ROOT_FLOOR + (EXTRA_BUDGET if startup else 0)


class StorageMonitor:
    def __enter__(self):
        storage_check(); self.stop=threading.Event()
        def watch():
            while not self.stop.wait(5):
                try: storage_check()
                except BaseException as error:
                    print('storage floor breached: '+repr(error),flush=True); os._exit(74)
        self.thread=threading.Thread(target=watch,daemon=True); self.thread.start(); return self
    def __exit__(self,*args):
        self.stop.set(); self.thread.join(); storage_check()


def check_control(r):
    assert r['complete'] and r['calls']==14 and r['frames']==32 and len(r['reports'])==14
    assert r['all_original32_logits_raw_bytes_equal'] and r['all_original32_metrics_exact'] and r['original15_failures_reproduced']
    assert r['same168_bindings'] and r['full_state_snapshots']==28 and r['cache_tensors_saved']==4704
    assert r['resets']==[dict(before_call=0),dict(before_call=0),dict(before_call=7)]
    assert r['graph_input_count']==7 and r['compressed_raw_graph_inputs']==98 and r['all_sdk_placeholder_values_preserved']
    assert not r['full_model_qualified'] and not r['new_speedup_claim']


def check_cpu(r):
    assert r['complete'] is True and r['rank'] in (0,1) and r['calls']==14 and r['frames']==32 and len(r['reports'])==14
    assert r['arm'] in contract_spec.CPU_ARMS and r['stage_tensors']==14*213 and r['prediction_vectors']==32*213
    assert r['state_imports_raw_exact']==r['state_input_immutability_checks']==14*84
    assert r['all_original_shard_tensors_loaded'] and r['graph_input_count']==7 and r['all_sdk_placeholder_values_preserved']
    from minicpm5_history_environment_v1 import check_receipt
    check_receipt(r['sdk_environment_transition'],r['rank'])
    from minicpm5_history_model_v1 import check_load_metadata
    assert [x['before_call'] for x in r['constructions']]==[0,4,7,11]
    for row in r['constructions']:
        check_load_metadata(row['proof'],r['rank'])
        assert row['proof']['phase']==('prefill' if row['before_call'] in (0,7) else 'decode')
        assert not row['proof']['compiled'] and not row['proof']['resharded'] and not row['proof']['sdk_math_modified']
    if r['arm']=='C_Hd_control':assert r['all_v3_outputs_stages_poststates_exact'] is True
    else:assert r['seed_device_call0_then_previous_cpu_post_only'] is True
    assert not r['full_model_qualified'] and not r['new_speedup_claim']
