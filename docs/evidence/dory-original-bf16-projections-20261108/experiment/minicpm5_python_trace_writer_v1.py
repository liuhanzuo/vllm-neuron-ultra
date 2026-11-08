"""Bounded diagnostic writer over original exclusive raw IO library.

No service/root creation. Future root supplies exact per-frame permit and an
external guard which verifies process, source, output tree and original scope.
Failed diagnostic receipts never use the successful capture.json schema.
"""
import hashlib
import json
import os
from minicpm5_native_ablation_raw_io_v1 import ExclusiveRawWriter,require
from minicpm5_native_ablation_witness_v1 import VECTOR_BYTES
from minicpm5_python_trace_observer_v1 import SCHEMA,validate_capture_summary
MIB=2**20


class FrameWriter:
    def __init__(self,directory,binding,guard):
        require(binding['schema']=='minicpm5_python_trace_writer_binding_v1','Exact diagnostic binding')
        self.binding=dict(binding);self.guard=guard;self.names=[];self.receipts=[];self.json_bytes=0;self.closed=False;self.failed=False
        permit=binding['base_permit'];require(permit['arm']=='HF_native' and permit['rank']==-1 and permit['frame_indices']==[binding['frame']['index']],'Original native physical writer scope')
        require(permit['directory']==str(directory),'Exact writer path')
        require(binding['arm'] in ('unobserved','previous213_hooks','noop_wrappers','noop_trace','full_trace'),'Exact diagnostic arm')
        require(binding['raw_cap']==VECTOR_BYTES+(binding['observer_raw_cap'] if binding['arm']=='full_trace' else 0),'Explicit witness+observer raw partition')
        require(binding['metadata_cap']==2*MIB and binding['max_payloads']==(68 if binding['arm']=='full_trace' else 4),'Bound metadata/payloads')
        # Successor raw writer preserves original IO and permits64 observer
        # payloads plus4 endpoint witnesses; original scope is not reduced.
        self.base=ExclusiveRawWriter(directory,permit,self._guard,raw_cap=binding['raw_cap'],tensor_cap=16*MIB,metadata_cap=binding['metadata_cap'])
    def _guard(self):
        require(not self.closed and not self.failed and self.guard()==self.binding,'Changed diagnostic binding')
        return dict(self.binding['base_permit'])
    def tensor(self,name,value,metadata):
        self._guard();require(name not in self.names and name in ('actual-bf16','actual-fp32','expected-fp32','expected-bf16-derived'),'Exact unique witness name')
        frame=self.binding['frame'];context=dict(arm='HF_native',rank=-1,call=self.binding['base_permit']['call'],frames=[frame],ablation_arm=self.binding['arm'])
        require(metadata['identity']['arm']==self.binding['arm'] and metadata['identity']['frame']==frame,'Witness frame binding')
        raw=dict(metadata,selected_bytes=metadata['bytes'],context=context,witness_name=name)
        result=self.base.tensor(value,raw);self.names.append(name);return result
    def observer_tensor(self,value,metadata):
        self._guard();require(self.binding['arm']=='full_trace' and self.base.index<64,'Full observer payload reservation')
        return self.base.tensor(value,metadata)
    def record(self,name,report):
        self._guard();require(name in ('witness','observation') and name not in self.receipts,'Unique diagnostic receipt')
        require(self.names==['actual-bf16','actual-fp32','expected-fp32','expected-bf16-derived'],'All four witness payloads first')
        frame=self.binding['frame']
        identity=report['identity'] if name=='witness' else dict(arm=report['arm'],frame=report['frame'])
        require(identity['arm']==self.binding['arm'] and identity['frame']==frame,'Exact diagnostic receipt owner')
        require(type(report['passed']) is bool and report['raw_exact'] is report['passed'] and report['full_model_qualified'] is False,'Diagnostic equality distinct from model qualification')
        # Include every observer+endpoint payload in the final observation,
        # preserving failed partial captures for independent final auditing.
        value=dict(report,binding=self.binding)
        if name=='observation':
            require(self.receipts==['witness'],'Witness must be durable first')
            if self.binding['arm']=='full_trace':
                capture=report['observer']
                require(capture['scale_observation_schema']==SCHEMA,'Actual new trace schema; legacy aten metadata forbidden')
                validate_capture_summary(capture,capture['trace_contract'])
            else:require(report['observer'] is None,'Control arm cannot retain full observer data')
            value.update(payloads=self.base.records,raw_bytes=self.base.raw_bytes)
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        require(self.json_bytes+len(raw)+self.base.metadata_bytes<=self.binding['metadata_cap'],'All frame JSON/metadata budget')
        path=self.base._exclusive(name+'.json',raw);os.fsync(self.base.fd);self.json_bytes+=len(raw);self.receipts.append(name)
        return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),durable=True)
    def failure(self,error,stage,snapshots,events):
        self._guard();require('failure' not in self.receipts,'Unique failure receipt')
        require(type(stage) is str and len(stage)<64,'Bound failure stage')
        require(isinstance(snapshots,list) and len(snapshots)<=64 and isinstance(events,list) and len(events)<=64,'Bound partial observation')
        value=dict(schema='minicpm5_python_trace_frame_failure_v1',binding=self.binding,
            complete=False,passed=False,raw_exact=False,full_model_qualified=False,
            error_type=type(error).__name__,error=str(error)[:4096],stage=stage,
            completed_receipts=list(self.receipts),witness_names=list(self.names),
            payloads=self.base.records,raw_bytes=self.base.raw_bytes,
            partial_snapshots=snapshots,partial_events=events,
            missing_frames_fabricated=False,original_error_preserved=True)
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        require(self.json_bytes+len(raw)+self.base.metadata_bytes<=self.binding['metadata_cap'],'Failure plus prior metadata budget')
        path=self.base._exclusive('failure.json',raw);os.fsync(self.base.fd);self.failed=True;self.json_bytes+=len(raw);self.receipts.append('failure')
        return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),durable=True)
    def close(self):
        if self.closed:return
        self.closed=True;self.base.close()
