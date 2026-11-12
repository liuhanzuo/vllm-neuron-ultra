"""One fixed six-call diagnostic leaf; no CLI, lock or service ownership.

Only the separately bound serial successor may call run_diagnostic. Importing
this module does not import the SDK or inspect device/process/lock state.
"""
import gc
from pathlib import Path

from research import qwen35_slot_special_value_protocol_v1 as protocol
import qwen35_predicated_slot_protocol_v1 as frozen


def run_diagnostic(root,*,guard,save_report,record_artifact):
    root=Path(root)
    protocol.require(all(callable(x) for x in (guard,save_report,record_artifact)),'Owned original callbacks required')
    protocol.require(root.is_dir() and root.resolve()==root,'Existing owned diagnostic directory')
    guard('before_diagnostic_inputs');protocol.verify_inputs();frozen.verify_sources()
    report=dict(schema='qwen35_slot_special_value_result_v1',workload=protocol.workload(),
        complete=False,error=None,inputs={},graphs=[],calls=[],runtime_calls=0,
        qualification_matrix_completed=False,full_model_qualified=False,model_speedup_claim=False)
    save_report(report)
    try:
        for profile in protocol.PROFILES:
            report['inputs'][profile]={key:protocol.write_raw(root/(profile+'-'+key+'.raw'),payload,protocol.BYTES)
                                      for key,payload in protocol.inputs(profile).items()}
        save_report(report)
        artifacts=root/'artifacts';artifacts.mkdir(exist_ok=False)
        identity_work=artifacts/'identity';identity_work.mkdir(exist_ok=False)
        guard('before_native_imports')
        import numpy as np
        import torch
        import torch_neuronx
        from research.qwen35_slot_identity_dma_nki_v1 import qwen35_conv_identity_dma
        torch.set_num_threads(1)

        def tensor(payload,key='cache'):
            array=np.frombuffer(payload,dtype=np.uint8).copy()
            return torch.from_numpy(array).view(torch.int64).reshape(4) if key=='slots' else torch.from_numpy(array).view(torch.bfloat16).reshape(protocol.SHAPE)
        def bytes_of(value):
            return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        class Identity(torch.nn.Module):
            def forward(self,cache):
                return qwen35_conv_identity_dma(cache.reshape(384,128)).reshape(4,3072,4)

        # Real identity compilation is the first post-import SDK boundary. The
        # original runner installs its pinned context fix at this exact guard.
        guard('before_compile:identity:diagnostic')
        with torch.inference_mode():
            traced=torch_neuronx.trace(Identity().eval(),(tensor(protocol.inputs('original_raw')['cache']),),
                compiler_workdir=str(identity_work),compiler_args=list(frozen.FLAGS),input_output_aliases={})
            guard('after_compile:identity:diagnostic')
            identity_path=identity_work/'model.pt';traced.save(str(identity_path))
            identity_ref=protocol.file_ref(identity_path)
            hlo_ref=protocol.file_ref(identity_work/'model/graph.hlo',16*2**20)
            hlo_proof=protocol.identity_hlo_proof(protocol.checked(hlo_ref,16*2**20))
            record_artifact(dict(arm='identity',artifact=identity_ref,hlo=hlo_ref,hlo_proof=hlo_proof))
            del traced;gc.collect()

            for arm in protocol.ARMS:
                artifact=identity_ref if arm=='identity' else protocol.ARTIFACTS[arm]
                guard('before_load:'+arm)
                before=protocol.identity(artifact['path']);protocol.checked(artifact)
                compiled=torch.jit.load(artifact['path'])
                protocol.require(protocol.identity(artifact['path'])==before,'Artifact identity stable through actual load')
                after=protocol.file_ref(artifact['path']);protocol.require(after==artifact,'Actual loaded bytes equal frozen artifact')
                code=str(compiled.code).encode()
                protocol.require(b'ops.neuron.forward_v2' in code or b'neuron::forward_v2' in code,
                    'Actual loaded graph calls Neuron; no host identity passthrough')
                code_ref=protocol.write_raw(root/(arm+'-forward.txt'),code,65536)
                torch_neuronx.move_trace_to_device(compiled,0)
                guard('after_device_load:'+arm)
                graph=dict(arm=arm,artifact=artifact,load_before=artifact,load_after=after,
                    load_identity=before,forward_code=code_ref,device_core=0,moved_to_device=True)
                if arm=='identity':graph.update(hlo=hlo_ref,hlo_proof=hlo_proof)
                report['graphs'].append(graph);save_report(report)
                for profile in protocol.PROFILES:
                    values={key:tensor(payload,key) for key,payload in protocol.inputs(profile).items()}
                    arguments=(values['cache'],) if arm=='identity' else (values['cache'],values['updates'],values['slots'])
                    guard('before_call:diagnostic:'+arm+':'+profile)
                    returned=compiled(*arguments)
                    protocol.require(isinstance(returned,torch.Tensor),'Exactly one actual tensor output')
                    actual=returned.detach().cpu().clone()
                    report['runtime_calls']+=1
                    keys=('cache',) if arm=='identity' else ('cache','updates','slots')
                    row=protocol.record_call(root,arm,profile,bytes_of(actual),list(actual.shape),str(actual.dtype),
                        {key:bytes_of(values[key]) for key in keys},guard)
                    report['calls'].append(row);save_report(report)
                    # Byte mismatch is recorded data. Every arm/profile still
                    # gets its independent invocation; exceptions still abort.
                    del returned,actual,arguments,values
                graph['post_calls']=protocol.file_ref(artifact['path'])
                protocol.require(graph['post_calls']==artifact and protocol.identity(artifact['path'])==before,
                    'Same artifact identity and all bytes after actual calls')
                del compiled;gc.collect();save_report(report)
                guard('after_graph:'+arm)
        protocol.verify_inputs();frozen.verify_sources()
        report['complete']=True
        protocol.validate_report(report,root)
        guard('before_diagnostic_completion');save_report(report)
        return report
    except BaseException as error:
        report['complete']=False;report['error']=repr(error)
        raise
    finally:
        save_report(report)
