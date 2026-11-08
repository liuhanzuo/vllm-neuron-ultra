"""Source-only fixed experiment contract; future root owns actual lifecycle.

No CLI, native import, service, payload read, serial lock or namespace creation.
RuntimeContext requires independently supplied live guard/writer/monitor.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import smollm3_token_capture_protocol_v7 as old
import smollm3_observer_metadata_source_v3 as source_metadata
import smollm3_observer_invariance_environment_v8 as environment_policy
import smollm3_enginecore_title_runtime_v11 as title_runtime

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'sio13'
BASE=ROOT/'research/smollm3-engine-token-capture-inputs-20261107-v7.json'
BASE_SHA='c4d2b9977379d698646e8485e470e15c766b90d72f1b43001f1e6559e8f22507'
BASE_CANONICAL_SHA='8076a269dbb035fd214f1a9a2415ff8c8609e187422ee29d5d71cdabc79cb1f0'
PYTHON=old.PYTHON
TARGET=old.TARGET
ROLES=old.ROLES
SPAWN='TRN1_SMOL_OBSERVER_INVARIANCE_V13'
MIB=2**20;GIB=2**30
OUTPUT_CAP=2*GIB
CONTROL_CAP=128*MIB
NATIVE_PREFIXES=('torch','numpy','transformers','vllm','vllm_neuron','neuronxcc','torch_neuronx','neuronx_distributed')
require=old.require
performance_guard=old.performance_guard


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def matrix():
    return [dict(run_id='r%02d'%(role_index*4+order),role=role,mode=mode,pair=order//2,order=order%2,
                 ordinal=role_index*4+order,requests=22,output_tokens=2808,fresh_server=True,
                 run_cap_bytes=(256 if mode=='on' else 64)*MIB,run_deadline_seconds=7200)
            for role_index,role in enumerate(ROLES) for order,mode in enumerate(('off','on','on','off'))]


def run_row(run_id):
    rows=[x for x in matrix() if x['run_id']==run_id];require(len(rows)==1,'Exact fixed run');return rows[0]


def directory(run_id):run_row(run_id);return DEST/run_id


def temporary(run_id):run_row(run_id);return DEST/('t'+run_id[1:])


def ipc_layout():
    rows={}
    for run in matrix():
        base=temporary(run['run_id']);lengths=dict(zmq=len(os.fsencode(base/'00000000-0000-0000-0000-000000000000')),
            multiprocessing=len(os.fsencode(base/'pymp-xxxxxxxx/listener-xxxxxxxx')))
        require(max(lengths.values())<=107,'Exact Linux IPC pathname bound')
        rows[run['run_id']]=dict(temporary_root=str(base),rpc_base=str(base),max_path_bytes=lengths)
    return rows


def unit(run_id,kind):
    run_row(run_id);require(kind in ('server','client'),'Exact service kind')
    return 'trn1-latest-verify-smollm3-observer-invariance-v14-'+run_id+'-'+kind


def resource(kind):
    require(kind in ('server','client'),'Resource kind');server=kind=='server'
    return dict(memory_max_bytes=(26 if server else 3)*GIB,memory_high_bytes=(25 if server else 2)*GIB,
        swap_max_bytes=(8 if server else 0)*GIB,allowed_cpus=list(range(8)),nice=19,tasks_max=2048,
        runtime_max_seconds=7200,limit_fsize_bytes=64*MIB,kill_mode='control-group',remain_after_exit=True)


def bounded_source(path,wanted=None):
    performance_guard()
    return source_metadata.source_bytes(path,wanted)


def load_base():return json.loads(bounded_source(BASE,BASE_SHA))


def template(base):
    """Scientific interface template only; no current predecessor identities."""
    require(digest(base)==BASE_CANONICAL_SHA and base['schema']=='smollm3_engine_token_capture_plan_v7' and base['roles']==list(ROLES) and base['groups']==old.groups(),'Original scientific base')
    return dict(schema='smollm3_observer_invariance_source_plan_v6',source_only=True,runtime_admitted=False,
        base_plan=dict(path=str(BASE),sha256=BASE_SHA),base_plan_canonical_sha256=digest(base),
        matrix=matrix(),groups=old.groups(),work=str(DEST),ipc_layout=ipc_layout(),
        roles=list(ROLES),requests_per_run=22,output_tokens_per_run=2808,total_requests=264,total_output_tokens=33696,
        fresh_servers=12,fresh_clients=12,resources={kind:resource(kind) for kind in ('server','client')},
        storage=dict(total_output_cap_bytes=OUTPUT_CAP,aggregate_control_included_bytes=CONTROL_CAP,
            run_caps_included_bytes=sum(x['run_cap_bytes'] for x in matrix()),root_floor_bytes=28*GIB,
            nvme_floor_bytes=5*GIB,root_admission_free_bytes=30*GIB,all_logs_and_exports_included=True,
            each_run_includes_its_temporary_IPC_directory=True),
        wall_time=dict(each_run_shared_server_client_seconds=7200,total_matrix_seconds=86400,external_final_audit_seconds=1800),
        estimate=dict(v7_on_allocated_bytes=[19513344,19501056,19558400],v7_first_to_last_response_seconds=[735.2084805965424,79.79382729530334,93.37182402610779],
            twelve_workloads_seconds_estimate=3633.496527671814,excludes_twelve_startups=True,not_runtime_admission=True),
        metadata_scope=dict(both='Original request bodies,raw HTTP/SSE responses,parsed22 token/text/usage records,group and completion reports,mode identity receipts,all service logs/owners/exit proofs',
            on='Frozen observer4 full seven graph input arrays plus engine/request/sample/slot observations; original exporter3 chunks after root drain',
            off='No observer module/install/files,graph inputs,slot observations or fabricated export',
            output_cap_behavior='Fail and retain partial evidence on overflow; never truncate history or silently discard differences'),
        observer=dict(on='smollm3_token_capture_observer_v4.install',off='no observer import or installation in parent/spawn',
            process_count_cap=8,file_cap_bytes=64*MIB,on_run_cap_bytes=256*MIB,off_run_cap_bytes=64*MIB),
        immutable_science=dict(cases=base['cases'],prompts=base['prompts'],historical=base['historical'],
            artifact_sha256=base['artifact_sha256'],payloads=base['payloads'],serialization_root_review=base['serialization_root_review'],
            implementation_commit=base['implementation_commit']),
        historical_match_is_result_not_admission=True,collect_all_twelve_despite_output_differences=True,
        api_identity_scope_only=True,graph_state_equivalence_claim=False,original76_requests_qualified=False,
        original_cold_long_reuse_quality_serving_gates_unchanged=True,compile_allowed=False,extra_cache_reset_calls=0,
        runtime_bindings=dict(final_catalog=None,root_source_review=None,actual_unit_owners=None,attempt_ids=None,serial_lock=None,canonical_sha256=None),
        source_sha256={})


def validate_template(value,base):
    expected=template(base);require({k:v for k,v in value.items() if k!='source_sha256'}=={k:v for k,v in expected.items() if k!='source_sha256'},'Exact source-only matrix/science/resource template')
    pins=value['source_sha256'];require(isinstance(pins,dict) and all(type(k)is str and re.fullmatch('[0-9a-f]{64}',v) for k,v in pins.items()),'Immutable source closure')
    require(all(pins.get(k)==v for k,v in base['source_sha256'].items()) and pins.get(str(BASE))==BASE_SHA,'Complete original source closure')
    for name in ('smollm3_observer_invariance_protocol_v18.py','smollm3_observer_invariance_worker_v18.py','smollm3_observer_invariance_analysis_v18.py',
                 'smollm3_observer_invariance_environment_v8.py','smollm3_spawn_environment_witness_v14.py','smollm3_ray_import_witness_v5.py',
                 'smollm3_enginecore_title_transition_v1.py','smollm3_enginecore_title_hook_v3.py','smollm3_enginecore_title_runtime_v11.py'):
        require(str(ROOT/name) in pins,'Callable implementation closure')
    for name in ('smollm3_observer_metadata_source_v3.py','smollm3_observer_metadata_codec_v2.py',
                 'smollm3_observer_metadata_store_v1.py','nemotron9b_package_metadata_v7.py',
                 'minicpm5_layer0_execution_protocol_v4.py','minicpm5_layer0_runtime_auth_v1.py','historical_b3_payload_partition_v1.py'):
        require(str(ROOT/name)in pins,'Complete metadata transport source closure')
    require(value['storage']['run_caps_included_bytes']+CONTROL_CAP==OUTPUT_CAP,'Control included in total2GiB')
    return value


def environment(base,run_id,kind,attempt):
    row=run_row(run_id);require(re.fullmatch('[0-9a-f]{32}',attempt) is not None,'Fresh attempt nonce');resource(kind)
    # Exact v7 values; only output/IPC namespace and authenticated spawn token change.
    case=base['cases'][row['role']];root=directory(run_id);tmp=temporary(run_id)
    return dict(HOME='/home/ubuntu',USER='ubuntu',LANG='C.UTF-8',PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',
        PYTHONOPTIMIZE='0',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPYCACHEPREFIX=str(root/('pycache-'+kind)),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NEURON_RT_VISIBLE_CORES='0-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        TORCHINDUCTOR_COMPILE_THREADS='1',TOKENIZERS_PARALLELISM='false',NEURON_SMOLLM3='1',NEURON_SMOLLM3_CACHE_VARIANT=row['role'],
        NEURON_SMOLLM3_ALLOW_COMPILE='0',NEURON_CPU_SAMPLER_FAST='0',VLLM_PLUGINS='neuron,neuron_smollm3',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        PYTHONPATH=str(TARGET),TMPDIR=str(tmp),VLLM_RPC_BASE_PATH=str(tmp),HF_HOME='/mnt/nvme/hf',
        PATH='/home/ubuntu/venvs/vllm_neuron/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin',NEURON_COMPILED_ARTIFACTS=case['artifact'],
        BASE_COMPILE_WORK_DIR=str(root/'compile-forbidden'),NEURON_COMPILE_CACHE_URL=str(root/'cache-forbidden'),
        XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus',**{SPAWN:run_id+':'+attempt})


def native_loaded():return any(n==x or n.startswith(x+'.') for n in sys.modules for x in NATIVE_PREFIXES)


class RuntimeContext:
    """Root-supplied authenticated guard, exclusive writer and whole-tree poll.

    No service operations here. Root must keep one serial lock through all24
    units, retain exits, independently stop/drain and prohibit successors.
    """
    def __init__(self,source_plan,base,run_id,kind,spawned,guard,writer,monitor):
        require(not native_loaded(),'Context before native import');validate_template(source_plan,base)
        self.plan=copy.deepcopy(source_plan);self.base=copy.deepcopy(base);self.row=run_row(run_id);self.kind=kind;self.spawned=spawned
        self.external_guard=guard;self.external_writer=writer;self.external_monitor=monitor;self.callback_lock=threading.RLock();self.permit=copy.deepcopy(guard())
        self.environment_stage='clean';self.runner_transition=None;self.ray_witness=None
        self.ray_state=None if kind=='client' else environment_policy.ray.absent_state()
        if spawned:self.ray_state=environment_policy.ray.inherited_state(self.permit['ray_inheritance'],self.permit,self.actual_environment())
        self.lineage=copy.deepcopy(self.permit.get('environment_lineage'))
        if kind=='server':environment_policy.validate_lineage(self.lineage,self.permit,spawned)
        else:require(self.lineage is None,'Client has no environment lineage')
        contract=json.loads(bounded_source(environment_policy.CONTRACT,self.plan['source_sha256'][str(environment_policy.CONTRACT)]))
        require(contract['schema']=='smollm3_exact_dependency_environment_contract_v4'and
            all(self.plan['source_sha256'].get(k)==v for k,v in contract['source_sha256'].items()),'Exact environment setter source closure')
        self.environment_contract=contract
        launch=environment(self.base,run_id,kind,self.permit['attempt_id'])
        self.system_environment=environment_policy.capture_system(os.environ,launch,self.permit)
        title_runtime.init_context(self)
        self.authenticate()
    @property
    def python_executable(self):return PYTHON
    @staticmethod
    def source_bytes(path,wanted):return bounded_source(path,wanted)
    @staticmethod
    def actual_environment():return os.environ
    def begin_runner_import(self):
        with self.callback_lock:
            self.poll();require(self.kind=='server' and self.environment_stage=='native_ready' and self.runner_transition is None,'One exact runner import transition')
            require(not any(x=='cv2' or x.startswith('cv2.') for x in sys.modules),'cv2 not imported before runner boundary')
            for name,wanted in self.environment_contract['source_sha256'].items():bounded_source(Path(name),wanted)
            row=dict(schema='smollm3_runner_import_before_v1',pid=self.permit['pid'],start_ticks=self.permit['start_ticks'],
                     permit_sha256=digest(self.permit),lineage_sha256=digest(self.lineage),stage='native_ready',
                     ld_source_state=environment_policy.ld_state(self.lineage,'native_ready'),cv2_loaded=False,exact_values_checked=True)
            self.record('runner-before-'+str(self.permit['pid'])+'.json',row)
            self.runner_transition=copy.deepcopy(row)
    def complete_runner_import(self):
        with self.callback_lock:
            require(self.runner_transition is not None and self.environment_stage=='native_ready','Matching before runner transition')
            before=self.runner_transition
            require(before['permit_sha256']==digest(self.permit) and before['lineage_sha256']==digest(self.lineage),'Exact unchanged runner-before binding')
            self.authenticate(check_environment=False)
            observed=self.external_monitor()
            require(observed['recursive']is True and observed['no_symlinks']is True and observed['foreign_files']==[],'Runner transition complete output observation')
            require(0<=observed['total_allocated_bytes']<=OUTPUT_CAP and 0<=observed['run_allocated_bytes']<=self.row['run_cap_bytes'] and
                    0<=observed['aggregate_control_allocated_bytes']<=CONTROL_CAP and observed['root_free_bytes']>=28*GIB and observed['nvme_free_bytes']>=5*GIB,'Runner transition original bounds')
            for name,wanted in self.environment_contract['source_sha256'].items():bounded_source(Path(name),wanted)
            launch=environment(self.base,self.row['run_id'],self.kind,self.permit['attempt_id'])
            environment_policy.check(os.environ,launch,self.kind,'runner_ready',self.spawned,self.system_environment,self.lineage,self.ray_state)
            require('cv2' in sys.modules and Path(getattr(sys.modules['cv2'],'__file__','')).resolve()==environment_policy.SITE/'cv2/__init__.py','Successful exact cv2 loader import')
            self.environment_stage='runner_ready';self.record_environment();self.poll()
            self.record('runner-after-'+str(self.permit['pid'])+'.json',dict(schema='smollm3_runner_import_after_v1',
                pid=self.permit['pid'],start_ticks=self.permit['start_ticks'],before_sha256=digest(before),
                permit_sha256=digest(self.permit),ld_source_state=environment_policy.ld_state(self.lineage,'runner_ready'),exact_values_checked=True))
    def advance_environment(self,stage):
        with self.callback_lock:
            stages=environment_policy.stages(self.kind,self.spawned)
            old_index=stages.index(self.environment_stage)
            require(old_index+1<len(stages)and stages[old_index+1]==stage and stage not in('runner_ready','ray_ready'),'Exact forward environment stage transition')
            self.authenticate();self.environment_stage=stage;self.authenticate()
            self.record_environment()
    def record_environment(self):
        launch=environment(self.base,self.row['run_id'],self.kind,self.permit['attempt_id'])
        proof=environment_policy.check(os.environ,launch,self.kind,self.environment_stage,self.spawned,self.system_environment,self.lineage,self.ray_state)
        proof.update(pid=self.permit['pid'],start_ticks=self.permit['start_ticks'],attempt_id=self.permit['attempt_id'],
            run_id=self.row['run_id'],mode=self.row['mode'],source_contract=dict(path=str(environment_policy.CONTRACT),sha256=self.plan['source_sha256'][str(environment_policy.CONTRACT)]))
        self.record('environment-'+self.environment_stage+'-'+str(self.permit['pid'])+'.json',proof)
    def authenticate(self,check_environment=True):
        performance_guard();value=self.external_guard();require(value==self.permit,'Runtime lease changed')
        require(value['schema']=='smollm3_observer_invariance_runtime_permit_v6' and value['run']==self.row and value['kind']==self.kind and value['spawned'] is self.spawned,'Exact run/role/mode/pair/worker binding')
        require(value['source_plan_sha256']==digest(self.plan) and value['base_plan_sha256']==BASE_SHA and value['source_plan_immutable'] is True
            and digest(self.base)==BASE_CANONICAL_SHA,'Scientific source binding')
        require(value['uid']==os.getuid()==1000 and value['pid']==os.getpid() and value['python']==sys.executable==PYTHON,'Actual native worker')
        statrow=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
        require(value['start_ticks']==statrow[19] and value['ppid']==int(statrow[1]),'Actual worker start/parent')
        title_runtime.worker_check(self)
        require(value['unit']==unit(self.row['run_id'],self.kind) and value['cgroup']=='0::'+old.GROUP_PREFIX+value['unit']+'.service','Owned service namespace')
        require(re.fullmatch('[0-9a-f]{32}',value['invocation_id']) and value['invocation_id']!='0'*32,'Actual root verified invocation')
        require(value['resources']==resource(self.kind) and os.sched_getaffinity(0)==set(range(8)) and os.getpriority(os.PRIO_PROCESS,0)==19,'Original service/CPU resources')
        require(sys.dont_write_bytecode and __debug__ and sys.flags.optimize==0,'Original Python flags')
        env=environment(self.base,self.row['run_id'],self.kind,value['attempt_id']);require(value['environment']==env,'Exact reviewed env launch')
        require(value.get('environment_lineage')==self.lineage,'Immutable exact environment lineage')
        if check_environment:
            environment_policy.check(os.environ,env,self.kind,self.environment_stage,self.spawned,self.system_environment,self.lineage,self.ray_state)
            if self.kind=='server'and self.ray_state['kind']=='local_import_zero':
                require(self.ray_witness is not None and digest(self.ray_witness.after)==self.ray_state['source_receipt_sha256'],'Same-process Ray completion receipt')
                self.ray_witness.check_committed()
        for key in ('independent_root_source_review','all_source_pins_verified','all_payloads_verified','all_final_predecessors_drained','serial_lock_proven','live_service_identity_verified','recursive_output_monitor_active','previous_run_fully_drained','fresh_run_namespace','actual_tiny_serialization_verified'):
            require(value[key] is True,'Missing independent root proof: '+key)
        require(value['canonical_unchanged'] is True and value['automatic_successor_launch'] is False,'Canonical/exclusive scope')
        require(value['output_root']==str(DEST) and value['run_directory']==str(directory(self.row['run_id'])) and value['temporary_root']==str(temporary(self.row['run_id'])),'Fresh exact roots')
        wrapper=Path(value['wrapper_source']);require(wrapper.is_absolute() and str(wrapper) in value['root_runtime_source_sha256'] and value['root_runtime_sources_verified'] is True,'Future independently reviewed main/spawn wrapper closure')
        require(value['shared_run_deadline_monotonic']==value['run_started_monotonic']+7200 and
            value['matrix_deadline_monotonic']==value['matrix_started_monotonic']+86400 and
            value['matrix_started_monotonic']<=value['run_started_monotonic']<=time.monotonic()<min(value['shared_run_deadline_monotonic'],value['matrix_deadline_monotonic']),
            'Shared server/client7200s and complete matrix24h deadline')
        return copy.deepcopy(value)
    def poll(self):
        with self.callback_lock:
            self.authenticate();observed=self.external_monitor()
            require(observed['recursive'] is True and observed['no_symlinks'] is True and observed['foreign_files']==[],'Whole output tree observation')
            require(0<=observed['total_allocated_bytes']<=OUTPUT_CAP and 0<=observed['run_allocated_bytes']<=self.row['run_cap_bytes'] and
                0<=observed['aggregate_control_allocated_bytes']<=CONTROL_CAP,'Inclusive total/run/control caps')
            require(observed['root_free_bytes']>=28*GIB and observed['nvme_free_bytes']>=5*GIB,'Original floors')
            require(observed['root']==str(DEST) and observed['run_id']==self.row['run_id'] and
                observed['run_scope_paths']==[str(directory(self.row['run_id'])),str(temporary(self.row['run_id']))],'Monitor root/run plus temporary scope');return observed
    def write(self,name,raw):
        with self.callback_lock:
            self.poll();require(re.fullmatch('[A-Za-z0-9_-]+\.(json|raw)',name) and type(raw)is bytes and len(raw)<=4*MIB,'Bounded exclusive control/request receipt')
            result=self.external_writer(name,raw)
            require(result['durable'] is True and result['exclusive'] is True and result['path']==str(directory(self.row['run_id'])/name)
                and result['sha256']==hashlib.sha256(raw).hexdigest() and result['bytes']==len(raw),'Actual exclusive durable receipt')
            self.poll();return result
    def record(self,name,value):return self.write(name,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())


    def install_title_witness(self):return title_runtime.install_hook(self)
    def begin_engine_title(self,hook,witness):return title_runtime.begin(self,hook,witness)
    def complete_engine_title(self,hook,token,result):return title_runtime.complete(self,hook,token,result)
    def fail_engine_title(self,hook,token,error_type):return title_runtime.fail(self,hook,token,error_type)
    def title_spawn_parent(self):return title_runtime.spawn_parent(self)

    def install_ray_witness(self):
        with self.callback_lock:
            self.poll();require(self.kind=='server'and self.ray_witness is None,'One admitted Ray witness')
            self.ray_witness=environment_policy.ray.RayImportWitness(self)
            self.ray_witness.install()
            self.record('ray-hook-'+str(self.permit['pid'])+'.json',dict(schema='smollm3_ray_hook_v1',
                process=environment_policy.identity(self.permit),permit_sha256=digest(self.permit),
                same_in_both_modes=True,source_sha256={str(environment_policy.ray.SITE/k):v for k,v in environment_policy.ray.PINS.items()}))
    def complete_ray_import(self):
        with self.callback_lock:
            witness=self.ray_witness
            require(self.kind=='server'and witness is not None and witness.before is not None and witness.after is None,'Only pending actual Ray writer can transition')
            require(self.spawned or self.environment_stage=='api_loading','Root transition only from API loading')
            self.authenticate(check_environment=False)
            observed=self.external_monitor()
            require(observed['recursive']is True and observed['no_symlinks']is True and observed['foreign_files']==[],'Ray transition recursive resource observation')
            require(0<=observed['total_allocated_bytes']<=OUTPUT_CAP and 0<=observed['run_allocated_bytes']<=self.row['run_cap_bytes'] and
                0<=observed['aggregate_control_allocated_bytes']<=CONTROL_CAP and observed['root_free_bytes']>=28*GIB and observed['nvme_free_bytes']>=5*GIB,'Ray transition original budgets')
            require(observed['root']==str(DEST) and observed['run_id']==self.row['run_id'] and
                observed['run_scope_paths']==[str(directory(self.row['run_id'])),str(temporary(self.row['run_id']))],'Ray transition exact output scopes')
            after=witness.ready()
            state=dict(kind='local_import_zero',source_receipt_sha256=digest(after),inherited_receipt_sha256=self.ray_state['inherited_receipt_sha256'])
            stage=self.environment_stage if self.spawned else 'ray_ready'
            launch=environment(self.base,self.row['run_id'],self.kind,self.permit['attempt_id'])
            # Exact single pending transition; ordinary poll never infers it.
            environment_policy.check(self.actual_environment(),launch,self.kind,stage,self.spawned,self.system_environment,self.lineage,state)
            witness.commit(after);self.ray_state=state;self.environment_stage=stage
            # Commit before record(), whose poll must now see the ready state.
            self.record('ray-after-'+str(self.permit['pid'])+'.json',after)
            if not self.spawned:
                self.record_environment();self.advance_environment('serve')
            self.poll()
            return after
    def complete_pending_ray_before_spawn(self):
        witness=self.ray_witness
        require(witness is not None,'Spawn requires installed Ray source witness')
        if witness.before is not None and witness.after is None:self.complete_ray_import()
        require(self.ray_state['kind']!='absent','Spawn requires proven Ray state')
