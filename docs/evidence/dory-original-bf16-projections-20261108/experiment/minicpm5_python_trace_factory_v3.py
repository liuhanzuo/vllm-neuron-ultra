"""Fixed 160-frame diagnostic partition factory. No root creation/admission."""
import copy
import os
from pathlib import Path
import stat
from minicpm5_layer0_runtime_auth_v1 import require,digest
from minicpm5_python_trace_writer_v1 import FrameWriter
from minicpm5_native_ablation_witness_v1 import VECTOR_BYTES
from minicpm5_python_trace_plan_v3 import validate_input
from minicpm5_python_trace_observer_v2 import SCHEMA
ARMS=('unobserved','previous213_hooks','noop_wrappers','noop_trace','full_trace')
MIB=2**20


def validate_partitions(plan,input_plan):
    validate_input(input_plan)
    require(plan['schema']=='minicpm5_python_trace_source_plan_v1','Exact160 plan schema')
    require(plan['scale_observation_schema']==SCHEMA,'Exact Python trace provenance schema')
    require(plan['arms']==list(ARMS) and plan['frames_per_arm']==32,'Original five arms32')
    root=Path(plan['output_root'])
    require(root.is_absolute() and str(root)==input_plan['output_root'] and root.name=='minicpm5-native-python-trace-v3','New output root')
    require(plan['original88_scope_preserved'] is True and input_plan['writer_contexts_executable_in_ablation'] is False,'Original88 remains separate')
    require(plan['output_cap_bytes']==2*2**30 and plan['control_cap_bytes']==32*MIB and plan['root_floor_bytes']==28*2**30,'Original disk bounds')
    require(plan['memory_max_bytes']==26*2**30 and plan['memory_high_bytes']==24*2**30 and plan['swap_max_bytes']==0,'Original memory bounds')
    require(digest(input_plan)==plan['input_context']['canonical_sha256'],'Frozen scientific input binding')
    frames=input_plan['frames'];require(len(frames)==32 and {f['index'] for f in frames}==set(range(32)),'Original32 frame identities')
    native={row['frame_index']:row for row in input_plan['writer_contexts'] if row['arm']=='HF_native'}
    require(len(native)==32,'Original observer32 reservations')
    expected={}
    for arm in ARMS:
        for frame in frames:
            calls=[c['index'] for c in input_plan['calls'] if frame in c['frames']];require(len(calls)==1,'Unique original frame call')
            cap=native[frame['index']]['raw_cap'] if arm=='full_trace' else 0
            row=dict(arm=arm,frame=frame,call=calls[0],directory=str(root/(arm.replace('_','-')+'-frame-%02d'%frame['index'])),raw_cap=VECTOR_BYTES+cap,
                observer_raw_cap=cap,tensor_cap=16*MIB,metadata_cap=2*MIB,max_observer_payloads=64 if cap else 0,
                max_payloads=68 if cap else 4,allocation_slack=72*4096)
            expected[arm,frame['index']]=row
    rows=plan['writer_contexts'];require(len(rows)==160,'All160 partitions')
    actual={(row['arm'],row['frame']['index']):row for row in rows}
    require(len(actual)==160 and actual==expected,'Exact160 geometry and per-frame caps')
    totals=dict(partition_raw_bytes=sum(row['raw_cap'] for row in rows),partition_metadata_bytes=sum(row['metadata_cap'] for row in rows),partition_allocation_slack=sum(row['allocation_slack'] for row in rows))
    totals['partition_allocated_bytes']=sum(totals.values())
    require(all(plan[k]==v for k,v in totals.items()) and totals['partition_allocated_bytes']<plan['output_cap_bytes'],'Aggregate reserved output budget')
    return copy.deepcopy(actual)


class WriterFactory:
    def __init__(self,plan,input_plan,plan_sha256,guard,whole_tree_monitor):
        require(digest(plan)==plan_sha256,'Exact source plan')
        self.plan=copy.deepcopy(plan);self.plan_sha256=plan_sha256;self.guard=guard;self.monitor=whole_tree_monitor
        self.rows=validate_partitions(self.plan,input_plan);self.used=set();self.root=Path(plan['output_root'])
        self.permit=copy.deepcopy(guard());require(self.permit['source_plan_sha256']==plan_sha256,'Worker plan binding')
        require(self.permit['arm'] in ARMS and self.permit['uid']==os.getuid()==1000,'Worker arm/owner')
        info=self.root.lstat();require(stat.S_ISDIR(info.st_mode) and info.st_uid==1000 and stat.S_IMODE(info.st_mode)==0o700 and not self.root.is_symlink(),'Owned output root')
        self.root_key=(info.st_dev,info.st_ino);require(self.root_key==(self.permit['output_root_identity']['device'],self.permit['output_root_identity']['inode']),'Admitted root inode')
        self._whole()
    def _whole(self):
        require(self.guard()==self.permit and digest(self.plan)==self.plan_sha256,'Worker/plan changed')
        info=self.root.lstat();require((info.st_dev,info.st_ino)==self.root_key and stat.S_ISDIR(info.st_mode),'Root replaced')
        observed=self.monitor()
        require(observed['plan_sha256']==self.plan_sha256 and observed['output_root']==str(self.root) and observed['attempt_id']==self.permit['attempt_id'],'Whole-tree monitor binding')
        require(observed['recursive'] is True and observed['no_symlinks'] is True and observed['unplanned_files']==[],'Whole-tree coverage')
        require(type(observed['allocated_bytes']) is int and 0<=observed['allocated_bytes']<=self.plan['output_cap_bytes'],'Actual whole-tree cap')
        require(observed['root_free_bytes']>=self.plan['root_floor_bytes'],'Root free-space floor')
    def __call__(self,arm,frame,call):
        self._whole();p=self.permit;key=arm,frame['index']
        require(arm==p['arm'] and key in self.rows and key not in self.used,'Wrong or reused arm/frame')
        row=self.rows[key];require(frame==row['frame'] and call==row['call'],'Original frame/call identity')
        base=dict(schema='minicpm5_layer0_writer_permit_v1',directory=row['directory'],pid=p['pid'],start_ticks=p['start_ticks'],arm='HF_native',rank=-1,
            call=call,frame_indices=[frame['index']],attempt_id=p['attempt_id'],plan_sha256=self.plan_sha256)
        binding=dict(schema='minicpm5_python_trace_writer_binding_v1',base_permit=base,frame=copy.deepcopy(frame),arm=arm,
            raw_cap=row['raw_cap'],observer_raw_cap=row['observer_raw_cap'],metadata_cap=row['metadata_cap'],max_payloads=row['max_payloads'])
        def check():self._whole();return copy.deepcopy(binding)
        self.used.add(key)
        return FrameWriter(row['directory'],binding,check)
