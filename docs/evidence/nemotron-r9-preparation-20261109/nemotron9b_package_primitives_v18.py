"""Original raw alias and split-control bodies with package imports/new output roots."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import nemotron9b_package_protocol_v18 as p
from nemotron9b_package_primitive_oracles_v18 import array_for,select,compare,mappings,classify,classifier_controls,output_shapes,inspect_hlo
SPEC=DEST=None
sha=p.stream_sha
save=p.save

def configure(spec):
    global SPEC,DEST
    SPEC=spec;DEST=Path(spec['work_root'])/'primitives'

def alias(selection):
    plan={'source_sha256':SPEC['source_sha256']}
    job,=[row for row in p.alias_jobs()if row['name']==selection]
    (DEST/'device').mkdir(exist_ok=True)
    output=DEST/'device'/(selection+'.json');assert not output.exists()
    report=dict(complete=False,passed=False,job=job,checks=[],source_sha256=plan['source_sha256'],raw_failure_payloads=[])
    started=time.monotonic()
    try:
        import numpy as np
        import torch
        import torch_neuronx
        from vllm_neuron.models.nemotron9b.helpers import store_slots
        from vllm_neuron.models.nemotron9b.nki_read_torch import read_slots_nki
        from vllm_neuron.models.nemotron9b.nki_write_torch import store_slots_nki_predicated
        torch.set_num_threads(1)
        case=job['case'];tdtype=torch.bfloat16 if case['dtype']=='bfloat16' else torch.float32
        def values(shape,salt):
            if job['data']=='raw':return array_for(case['dtype'],shape,raw=True,salt=salt)
            # Exactly representable finite, nonzero values keep this control separate from exceptional bits.
            base=array_for(case['dtype'],shape,raw=False,salt=salt)
            n=np.arange(math.prod(shape),dtype=np.int32).reshape(shape)+salt
            return (((n%127)+1)*np.where(n%2,1,-1)).astype(base.dtype)
        initial=values(case['shape'],0)
        def tensor(a):return torch.from_numpy(a.view(np.uint16).copy()).view(tdtype)
        def array(t):return t.detach().cpu().contiguous().view(torch.uint16).numpy().copy().view(initial.dtype)
        writer=store_slots_nki_predicated if job['variant']=='read_predicated' else store_slots
        has_read=job['variant']!='writer_original';persistent=job['mode']=='persistent'
        def compute(cache,updates,slots):
            old=read_slots_nki(cache,slots) if has_read else None
            new=writer(cache,updates,slots)
            return (old,new) if has_read else new
        class Functional(torch.nn.Module):
            def forward(self,cache,updates,slots):return compute(cache,updates,slots)
        class Persistent(torch.nn.Module):
            def __init__(self):
                super().__init__();self.cache=torch.nn.Parameter(tensor(initial),requires_grad=False)
            def forward(self,updates,slots):return compute(self.cache,updates,slots)
        updates_shape=[case['rows']]+case['shape'][1:];slot_samples=mappings(case['rows'])
        source=(Persistent() if persistent else Functional()).eval()
        sample=(tensor(values(updates_shape,11)),torch.tensor(slot_samples[0],dtype=torch.int64))
        aliases={source.cache:(1 if has_read else 0)} if persistent else {}
        if not persistent:sample=(tensor(initial),*sample)
        work=DEST/'work'/selection;assert not work.exists();work.mkdir(parents=True)
        with torch.inference_mode():
            compiled=torch_neuronx.trace(source,sample,input_output_aliases=aliases,
                compiler_workdir=str(work),compiler_args=p.ALIAS_FLAGS)
            artifact=work/'model.pt';compiled.save(str(artifact))
            report.update(artifact=str(artifact),artifact_sha256=sha(artifact),compile_seconds=time.monotonic()-started)
            if persistent:
                states=dict(compiled.states._parameters.items());assert len(states)==1;state_name,=states
                assert compare(array(states[state_name]),initial)['passed'];del states
                torch_neuronx.move_trace_to_device(compiled,0)
                state=dict(compiled.states._parameters.items())[state_name]
                assert state.device.type=='privateuseone' and compare(array(state),initial)['passed'];del state
            expected=initial.copy();cache_tensor=tensor(initial)
            for step,slots in enumerate(slot_samples):
                before_expected=expected.copy();old_expected=select(expected,slots)
                updates=values(updates_shape,17+step);update_tensor=tensor(updates);slot_tensor=torch.tensor(slots,dtype=torch.int64)
                cache_before=array(cache_tensor) if not persistent else None
                result=compiled(update_tensor,slot_tensor) if persistent else compiled(cache_tensor,update_tensor,slot_tensor)
                old_actual=array(result[0]) if has_read else None
                new_actual=array(result[1] if has_read else result)
                for row,slot in enumerate(slots):
                    if slot>=0:expected[slot]=updates[row]
                comparisons={'updated_output':(expected,new_actual)}
                if has_read:comparisons['old_read']=(old_expected,old_actual)
                if persistent:
                    state=dict(compiled.states._parameters.items())[state_name]
                    assert state.device.type=='privateuseone'
                    comparisons['persistent_state']=(expected,array(state));del state
                checks={name:{**compare(actual,ideal),'classification':classify(ideal,actual)
                              if ideal.shape==actual.shape and ideal.dtype==actual.dtype else {'unavailable':'Shape or dtype differs'}}
                        for name,(ideal,actual) in comparisons.items()}
                unchanged=compare(array(update_tensor),updates)['passed'] and slot_tensor.tolist()==slots
                if not persistent:unchanged &= compare(array(cache_tensor),cache_before)['passed']
                check=dict(step=step,slots=slots,comparisons=checks,input_unchanged=unchanged,
                           passed=unchanged and all(row['passed'] for row in checks.values()))
                report['checks'].append(check)
                if not check['passed'] and not report['raw_failure_payloads']:
                    folder=DEST/'payloads'/selection;folder.mkdir(parents=True,exist_ok=False)
                    payloads={'expected_state_before':before_expected,'updates':updates}
                    if cache_before is not None:payloads['actual_functional_input']=cache_before
                    for name,(ideal,actual) in comparisons.items():payloads[name+'-expected']=ideal;payloads[name+'-actual']=actual
                    for name,data in payloads.items():
                        path=folder/(name+'.bin');path.write_bytes(data.tobytes())
                        report['raw_failure_payloads'].append(dict(name=name,path=str(path),sha256=sha(path),
                            bytes=path.stat().st_size,shape=list(data.shape),dtype=case['dtype'],byteorder=sys.byteorder,step=step))
                save(output,report)
                # Functional control carries its actual output into the next call, without device aliasing.
                if not persistent:cache_tensor=tensor(new_actual)
            report.update(complete=True,passed=all(row['passed'] for row in report['checks']))
            assert len(report['checks'])==len(slot_samples)
    except Exception as error:report['error']=repr(error);raise
    finally:
        report['seconds']=time.monotonic()-started
        report['scope']='Strict raw-word correctness and isolated write controls only. A completed failed control is retained; no model or speed qualification.'
        save(output,report);print(json.dumps({k:report[k] for k in ['complete','passed','seconds']}),flush=True)
    assert report['complete']and report['passed']
    return report

def split(selection):
    job, = [j for j in p.split_jobs() if j['name'] == selection]
    plan = {'source_sha256':SPEC['source_sha256']}
    directory = DEST / 'jobs' / selection
    assert not directory.exists()
    directory.mkdir(parents=True)
    report = dict(complete=False, primitive_passed=False, job=job, source_sha256=plan['source_sha256'],
                  cpu_checks=[], device_checks=[], compiler_flags=p.SPLIT_FLAGS,
                  functionalization_environment=os.environ['XLA_DISABLE_FUNCTIONALIZATION'],
                  scope='Exact split consumers only. No full model, quality, throughput or baseline qualification.')
    output = directory / 'result.json'
    save(output, report)
    started = time.monotonic()
    try:
        import torch
        import torch_neuronx
        from vllm_neuron.models.nemotron9b.helpers import split_last_dim
        torch.set_num_threads(1)
        case = job['case']
        b, t = case['batch'], case['length']

        class Consumers(torch.nn.Module):
            def __init__(self, variant):
                super().__init__()
                self.variant = variant

            def forward(self, value):
                projected = value if case['kind'] == 'nemotron_projection' else value.transpose(1, 2)
                if self.variant == 'tensor_split':
                    parts = split_last_dim(projected, case['sizes'])
                elif case['kind'] == 'qwen35_conv':
                    parts = torch.split(projected, case['sizes'], dim=-1)
                else:
                    parts = projected.split(case['sizes'], dim=-1)
                if case['kind'] == 'nemotron_projection':
                    return parts[0].float(), parts[1].transpose(1, 2).float(), parts[2].float()
                return tuple(part.reshape(shape).float() for part, shape in zip(parts, output_shapes(case)))

        def input_for(prime):
            codes = torch.arange(math.prod(case['input_shape']), dtype=torch.int32)
            return (codes.remainder(prime) - prime // 2).to(torch.bfloat16).reshape(case['input_shape'])

        def oracle(value):
            result, offset = [], 0
            for i, width in enumerate(case['sizes']):
                indices = torch.arange(offset, offset + width, dtype=torch.int64)
                axis = 2 if case['kind'] == 'nemotron_projection' else 1
                part = torch.index_select(value, axis, indices)
                if case['kind'] == 'nemotron_projection':
                    if i == 1:
                        part = part.permute(0, 2, 1)
                else:
                    part = part.permute(0, 2, 1).reshape(output_shapes(case)[i])
                result.append(part.float())
                offset += width
            return tuple(result)

        def fingerprint(value):
            import hashlib
            raw = value.detach().cpu().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
            return hashlib.sha256(raw).hexdigest()

        def compare(actual, expected):
            assert len(actual) == len(expected) == 3
            result = []
            for got, want in zip(actual, expected):
                same = got.shape == want.shape and got.dtype == want.dtype
                a, e = got.detach().cpu().reshape(-1).contiguous().view(torch.uint8), want.reshape(-1).contiguous().view(torch.uint8)
                exact = same and torch.equal(a, e)
                result.append(dict(passed=exact, shape=list(got.shape), expected_shape=list(want.shape),
                    dtype=str(got.dtype), expected_dtype=str(want.dtype), actual_sha256=fingerprint(got),
                    expected_sha256=fingerprint(want), bytes_compared=e.numel(),
                    different_bytes=0 if exact else (int(torch.count_nonzero(a != e)) if same else None)))
            return result

        with torch.inference_mode():
            for prime in (251, 241):
                value = input_for(prime)
                before = fingerprint(value)
                expected = oracle(value)
                for variant in ('original', 'tensor_split'):
                    actual = Consumers(variant)(value)
                    checks = compare(actual, expected)
                    report['cpu_checks'].append(dict(prime=prime, variant=variant, comparisons=checks,
                                                     input_unchanged=fingerprint(value) == before))
                    assert all(c['passed'] for c in checks) and report['cpu_checks'][-1]['input_unchanged']
                    del actual
                del expected, value
            report['cpu_passed'] = True
            report['stage'] = 'trace_compile'
            save(output, report)
            value = input_for(251)
            trace_start = time.monotonic()
            compiled = None
            try:
                compiled = torch_neuronx.trace(Consumers(job['variant']).eval(), (value,),
                    compiler_workdir=str(directory / 'compiler-work'), compiler_args=p.SPLIT_FLAGS)
                artifact = directory / 'model.pt'
                compiled.save(str(artifact))
                report.update(compile_passed=True, artifact=str(artifact), artifact_sha256=sha(artifact))
            except Exception as error:
                report.update(compile_passed=False, compile_error=repr(error))
            report['compile_seconds'] = time.monotonic() - trace_start
            del value
            hlo = directory / 'compiler-work/model/graph.hlo'
            try:
                report['hlo'] = inspect_hlo(hlo, case)
            except Exception as error:
                report['hlo_error'] = repr(error)
            if compiled is not None:
                report['stage'] = 'device'
                for prime in (251, 241):
                    value = input_for(prime)
                    before = fingerprint(value)
                    expected = oracle(value)
                    actual = compiled(value)
                    checks = compare(actual, expected)
                    unchanged = fingerprint(value) == before
                    report['device_checks'].append(dict(prime=prime, comparisons=checks, input_unchanged=unchanged))
                    if (not all(c['passed'] for c in checks) or not unchanged) and 'raw_failure' not in report:
                        payload = directory / 'first-failure.pt'
                        torch.save(dict(input=value, expected=expected, actual=actual, prime=prime), payload)
                        report['raw_failure'] = dict(path=str(payload), sha256=sha(payload), bytes=payload.stat().st_size)
                    save(output, report)
                    del actual, expected, value
            report['device_exact'] = (len(report['device_checks']) == 2 and
                all(row['input_unchanged'] and all(c['passed'] for c in row['comparisons'])
                    for row in report['device_checks']))
            hlo_pass = report.get('hlo', {}).get('sampled_mapping_passed', False)
            report['primitive_passed'] = bool(report['cpu_passed'] and report['device_exact'] and hlo_pass)
            compiler_codes = set(re.findall(r'NCC_[A-Z]+[0-9]+', report.get('compile_error', '')))
            for logfile in (directory / 'compiler-work').rglob('*'):
                if logfile.is_file() and logfile.suffix in ('.log', '.txt') and logfile.stat().st_size < 8 * 2**20:
                    compiler_codes.update(re.findall(r'NCC_[A-Z]+[0-9]+', logfile.read_text(errors='replace')))
            report['compiler_error_codes'] = sorted(compiler_codes)
            mapping_bad = report.get('hlo', {}).get('mismatching_sampled_addresses', 0) > 0
            if job['variant'] == 'original':
                report['control_outcome_confirmed'] = (report['primitive_passed'] if case['single_row_control'] else
                    bool(mapping_bad and ((report.get('compile_passed') and not report['device_exact']) or
                                          ('NCC_IBIR158' in compiler_codes))))
            report['stage'] = 'complete'
    except Exception as error:
        report['error'] = repr(error)
        report['stage'] = 'failed'
    finally:
        report['complete'] = True
        report['seconds'] = time.monotonic() - started
        group = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                     if line.startswith('0::'))
        assert group and group != '/'
        report['cgroup_memory'] = {name: (Path('/sys/fs/cgroup' + group) / name).read_text().strip()
                                  for name in ('memory.max', 'memory.high', 'memory.swap.max', 'memory.peak', 'memory.events')}
        save(output, report)
        print(json.dumps({k: report[k] for k in ('stage', 'primitive_passed', 'seconds')}), flush=True)
    assert report['complete']and report.get('cpu_passed')
    assert report['primitive_passed']if job['variant']=='tensor_split'else report.get('control_outcome_confirmed')
    return report

