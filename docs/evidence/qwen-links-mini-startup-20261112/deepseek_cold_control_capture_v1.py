"""Read existing native caches or original Neuron state; never change model graphs."""
from pathlib import Path
import deepseek_cold_kv_contract_v2 as c
from deepseek_cold_kv_codec_v2 import sha,encode


class Capture:
    def __init__(self,backend,destination,data_root):
        assert backend in ('native','device')
        self.backend=backend
        self.directory=Path(destination)/'cache-captures'/backend
        self.raw_directory=Path(data_root)/backend
        for path in (self.directory,self.raw_directory):
            assert path.is_absolute() and path.resolve()==path
            assert 'deepseek-cold-kv-diagnostic-v2' not in path.parts
            assert not path.exists() and not path.is_symlink()
        self.directory.mkdir(parents=True)
        self.raw_directory.mkdir(parents=True)
        self.frames=[];self.previous={};self.bindings=None;self.prefixes={}

    def write(self,frame,values,inputs):
        import torch
        assert frame==c.frames(self.backend)[len(self.frames)]
        assert list(values)==c.stages()
        key=c.key(frame,self.backend);report_path=self.directory/(key+'.json');payload_path=self.raw_directory/(key+'.pt')
        assert not report_path.exists() and not payload_path.exists()
        lineage=(frame['prefix'],frame['slot'] if self.backend=='device' else frame['fixture'])
        previous=self.previous.get(lineage)
        assert (previous is None)==(frame['step']==0)
        rows=[];payload={};hashes={}
        for stage,value in values.items():
            assert value.shape==(8,frame['prefix']+frame['step'],128) and value.dtype==torch.bfloat16
            record,tensor=encode(value,None if previous is None else previous['sha256'][stage])
            record.update(stage=stage,previous_frame=None if previous is None else previous['key'])
            c.check_record(record,frame,self.backend,stage)
            rows.append(record);payload[stage]=tensor;hashes[stage]=record['full_sha256']
        torch.save(payload,payload_path)
        result=dict(complete=True,backend=self.backend,frame=frame,key=key,stages=c.stages(),rows=rows,
            captured_elements=sum(x.numel() for x in values.values()),saved_elements=sum(x.numel() for x in payload.values()),
            payload=str(payload_path),payload_file_sha256=c.old.sha(payload_path),inputs=inputs,
            all_cache_values_finite=True,read_only_observation=True,compiled_model_unchanged=True,
            full_model_qualified=False,new_speedup_claim=False)
        c.check_frame(result,self.backend,frame);c.old.save(report_path,result)
        self.frames.append(dict(frame=frame,key=key,path=str(report_path),sha256=c.old.sha(report_path)))
        self.previous[lineage]=dict(key=key,sha256=hashes)
        c.old.save(self.directory/'progress.json',dict(completed_frames=len(self.frames),required_frames=len(c.frames(self.backend))))

    def native(self,cache,prefix,fixture,step,input_ids):
        import torch
        assert self.backend=='native' and len(cache.layers)==36
        values={}
        for layer_index,layer in enumerate(cache.layers):
            for side,tensor in (('key',layer.keys),('value',layer.values)):
                assert tensor.dtype==torch.bfloat16 and tensor.shape==(1,8,prefix+step,128)
                values[f'layers.{layer_index}.{side}']=tensor[0].detach().cpu().clone().contiguous()
        self.write(dict(prefix=prefix,fixture=fixture,step=step),values,
            dict(input_ids=input_ids,position=prefix-1+step,native_cache_length=int(cache.get_seq_length())))

    def device(self,model,ids,positions,slots,prefill,route):
        import torch
        assert self.backend=='device'
        state=model.traced_model.nxd_model.state
        assert len(state)==2 and all(set(rank)=={f'kv_mgr.past_key_values.{i}' for i in range(72)} for rank in state)
        bindings=[(rank,name,int(tensor._cdata),list(tensor.shape),str(tensor.dtype))
            for rank,tensors in enumerate(state) for name,tensor in sorted(tensors.items())]
        if self.bindings is None:self.bindings=bindings
        assert bindings==self.bindings
        active=slots.tolist();assert len(active)==len(set(active)) and all(slot in range(4) for slot in active)
        if prefill:
            assert len(active)==1 and ids.shape[0]==1
            prefix=ids.shape[1];assert prefix in c.PREFIXES
            assert positions.tolist()==[list(range(prefix))]
            self.prefixes[active[0]]=prefix
            frames=[dict(prefix=prefix,slot=active[0],fixture=active[0]%2,step=0)]
        else:
            assert active==[0,1,2,3] and tuple(ids.shape)==tuple(positions.shape)==(4,1)
            assert len(set(self.prefixes.values()))==1
            frames=[dict(prefix=self.prefixes[slot],slot=slot,fixture=slot%2,
                         step=int(positions[row,0])-self.prefixes[slot]+1) for row,slot in enumerate(active)]
            assert all(frame['step'] in (1,2) for frame in frames)
        values=[{} for _ in frames]
        for layer in range(36):
            for side_index,side in enumerate(('key','value')):
                name=f'kv_mgr.past_key_values.{2*layer+side_index}'
                rank_tensors=[]
                for rank in (0,1):
                    tensor=state[rank][name]
                    assert tensor.shape==(4,4,4224,128) and tensor.dtype==torch.bfloat16
                    rank_tensors.append(tensor.detach().cpu())
                for index,frame in enumerate(frames):
                    slot,length=frame['slot'],frame['prefix']+frame['step']
                    full=torch.cat([value[slot,:,:length,:] for value in rank_tensors],dim=0).clone().contiguous()
                    assert full.shape==(8,length,128)
                    values[index][f'layers.{layer}.{side}']=full
                del rank_tensors
        assert bindings==[(rank,name,int(tensor._cdata),list(tensor.shape),str(tensor.dtype))
            for rank,tensors in enumerate(state) for name,tensor in sorted(tensors.items())]
        inputs=dict(input_ids=ids.tolist(),position_ids=positions.tolist(),slots=active,prefill=bool(prefill),route=dict(route))
        for frame,tensors in zip(frames,values):self.write(frame,tensors,inputs)

    def finish(self):
        value=dict(complete=True,backend=self.backend,scope=c.scope(),frames=self.frames,
            full_frame_count=len(self.frames),cache_records=72*len(self.frames),
            compiled_model_unchanged=True,full_model_qualified=False,new_speedup_claim=False)
        c.check_manifest(value,self.backend)
        c.old.save(self.directory/'manifest.json',value)
        return value
