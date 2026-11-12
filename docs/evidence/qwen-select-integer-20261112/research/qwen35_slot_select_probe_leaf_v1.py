"""Owned twelve-call leaf; no CLI, lock, service or native imports on import."""
import gc
from pathlib import Path

from research import qwen35_slot_select_probe_protocol_v1 as protocol
import qwen35_predicated_slot_protocol_v1 as frozen


def run_diagnostic(root, *, guard, save_report, record_artifact):
    root = Path(root)
    protocol.require(all(callable(value) for value in (guard, save_report, record_artifact)), 'Original owner callbacks required')
    protocol.require(root.is_dir() and root.resolve() == root, 'Existing owned fresh diagnostic directory')
    guard('before_diagnostic_inputs'); protocol.verify_inputs(); frozen.verify_sources()
    report = dict(schema='qwen35_slot_select_probe_result_v1', workload=protocol.workload(),
        complete=False, error=None, inputs={}, slots={}, graphs=[], calls=[], runtime_calls=0,
        qualification_matrix_completed=False, full_model_qualified=False, model_speedup_claim=False)
    save_report(report)
    try:
        for profile in protocol.PROFILES:
            report['inputs'][profile] = {key: protocol.write_raw(root/(profile+'-'+key+'.raw'), payload, protocol.BYTES)
                for key, payload in protocol.input_data(profile).items()}
        for mapping, _ in protocol.MAPPINGS:
            report['slots'][mapping] = protocol.write_raw(root/(mapping+'-slots.raw'), protocol.slots(mapping), 32)
        save_report(report)
        artifacts = root/'artifacts'; artifacts.mkdir(exist_ok=False)
        guard('before_native_imports')
        import numpy as np
        import torch
        import torch_neuronx
        from research.qwen35_slot_select_bitcast_proposal_v1 import make_modules
        torch.set_num_threads(1)
        modules = make_modules()
        protocol.require(tuple(modules) == protocol.ARMS, 'Exact frozen three-module proposal')

        def tensor(payload, arm, key):
            storage = torch.from_numpy(np.frombuffer(payload, dtype=np.uint8).copy())
            if key == 'slots':
                return storage.view(torch.int64).reshape(4)
            dtype = torch.int16 if arm == 'external_s16_select' else torch.bfloat16
            return storage.view(dtype).reshape(protocol.SHAPE)

        def bytes_of(value):
            return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

        with torch.inference_mode():
            for arm in protocol.ARMS:
                # This is also the exact first post-import SDK boundary.
                guard('before_compile:'+arm+':diagnostic')
                work = artifacts/arm; work.mkdir(exist_ok=False)
                example = protocol.inputs('original_raw', 'idle')
                examples = tuple(tensor(example[key], arm, key) for key in ('cache','updates','slots'))
                traced = torch_neuronx.trace(modules[arm], examples, compiler_workdir=str(work),
                    compiler_args=list(frozen.FLAGS), input_output_aliases={})
                guard('after_compile:'+arm+':diagnostic')
                artifact_path = work/'model.pt'; traced.save(str(artifact_path))
                artifact = protocol.file_ref(artifact_path)
                hlo = protocol.file_ref(work/'model/graph.hlo', 16*2**20)
                proof = protocol.hlo_proof(protocol.checked(hlo, 16*2**20), arm)
                record_artifact(dict(arm=arm, artifact=artifact, hlo=hlo, hlo_proof=proof))
                del traced, examples, example; gc.collect()
                guard('before_load:'+arm)
                before = protocol.identity(artifact['path']); protocol.checked(artifact)
                compiled = torch.jit.load(artifact['path'])
                protocol.require(protocol.identity(artifact['path']) == before, 'Artifact identity stable through actual load')
                after = protocol.file_ref(artifact['path'])
                protocol.require(after == artifact, 'Actually loaded whole graph bytes unchanged')
                code = str(compiled.code).encode()
                protocol.require(b'ops.neuron.forward_v2' in code or b'neuron::forward_v2' in code, 'Compiled Neuron invocation path required')
                code_ref = protocol.write_raw(root/(arm+'-forward.txt'), code, 65536)
                torch_neuronx.move_trace_to_device(compiled, 0)
                guard('after_device_load:'+arm)
                graph = dict(arm=arm, artifact=artifact, hlo=hlo, hlo_proof=proof,
                    load_before=artifact, load_after=after, load_identity=before, forward_code=code_ref,
                    device_core=0, moved_to_device=True)
                report['graphs'].append(graph); save_report(report)
                for profile in protocol.PROFILES:
                    for mapping, _ in protocol.MAPPINGS:
                        values = {key: tensor(payload, arm, key) for key, payload in protocol.inputs(profile, mapping).items()}
                        arguments = tuple(values[key] for key in ('cache','updates','slots'))
                        guard('before_call:diagnostic:'+arm+':'+profile+':'+mapping)
                        returned = compiled(*arguments)
                        protocol.require(isinstance(returned, torch.Tensor), 'Exactly one actual tensor output')
                        actual = returned.detach().cpu().clone()
                        report['runtime_calls'] += 1
                        row = protocol.record_call(root, arm, profile, mapping, bytes_of(actual),
                            list(actual.shape), str(actual.dtype), {key: bytes_of(item) for key, item in values.items()}, guard)
                        report['calls'].append(row); save_report(report)
                        # Byte mismatch is data; exceptions abort without retry/fallback.
                        del returned, actual, arguments, values
                graph['post_calls'] = protocol.file_ref(artifact['path'])
                protocol.require(graph['post_calls'] == artifact and protocol.identity(artifact['path']) == before, 'Same complete artifact after actual calls')
                del compiled; gc.collect(); save_report(report)
                guard('after_graph:'+arm)
        protocol.verify_inputs(); frozen.verify_sources()
        report['complete'] = True
        protocol.validate_report(report, root)
        guard('before_diagnostic_completion'); save_report(report)
        return report
    except BaseException as error:
        report['complete'] = False; report['error'] = repr(error)
        raise
    finally:
        save_report(report)
