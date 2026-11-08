"""Admitted first22 token capture workers; no service launcher or retry.

--check is source-only. --server/--client need a future independently reviewed
exclusive supervisor and authenticated live service/owner admission.
"""
import argparse
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time
import urllib.error
import urllib.request

import smollm3_token_capture_protocol_v7 as p


def parser_module():
    path = p.ROOT / 'research/smollm3_engine_token_capture_parse_20261107_v1.py'
    spec = importlib.util.spec_from_file_location('smol_capture_parser', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def response_bytes(response, retained=None, cap=4 * 2**20):
    value = retained if retained is not None else bytearray()
    while True:
        block = response.read(min(65536, cap + 1 - len(value)))
        if not block:
            break
        value.extend(block)
        p.require(len(value) <= cap, 'HTTP response cap exceeded')
    return bytes(value)


def request(plan, role, group, index, directory, opener=urllib.request.urlopen):
    name = group['id'] + '-request' + str(index)
    body = p.request_body(plan, group)
    raw_request = json.dumps(body).encode()
    p.once(directory / (name + '-request.json'), body)
    with (directory / (name + '-request.raw')).open('xb') as f:
        f.write(raw_request); f.flush(); os.fsync(f.fileno())
    req = urllib.request.Request('http://127.0.0.1:8010/v1/completions', data=raw_request,
                                 headers={'Content-Type': 'application/json'})
    raw = b''; partial = bytearray(); status = None
    try:
        with opener(req, timeout=group['timeout']) as response:
            status = response.status
            p.require(status == 200, 'Non-200 completion response')
            raw = response_bytes(response, partial)
        # Save wire data before parser/text checks. Partial failures also persist.
        with (directory / (name + '-response.raw')).open('xb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        parse = parser_module()
        kwargs = dict(prompt_ids=body['prompt'], output_tokens=body['max_tokens'], model=body['model'])
        result = parse.parse_sse(raw.splitlines(keepends=True), **kwargs) if group['stream'] else parse.parse_completion(raw, **kwargs)
        result.update(group=group['id'], index=index, request_body_sha256=hashlib.sha256(raw_request).hexdigest(),
                      response_status=status, observed_at=time.time())
        p.once(directory / (name + '-parsed.json'), result)
        return result
    except BaseException as error:
        raw = bytes(partial)
        if isinstance(error, urllib.error.HTTPError):
            status = error.code
            try:
                raw = response_bytes(error)
            except BaseException:
                pass
        response_path = directory / (name + '-response.raw')
        if not response_path.exists():
            with response_path.open('xb') as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
        p.once(directory / (name + '-failure.json'), dict(passed=False, kind='response_protocol_or_transport',
            error=repr(error), group=group['id'], index=index, response_status=status,
            raw_sha256=hashlib.sha256(raw).hexdigest(), historical_text_reproduction_evaluated=False))
        raise


def validate_capture(plan,role,result):
    p.require(result['role']==role and result['request_scope']==22,'Role/scope')
    expected=[(g,index)for g in p.groups()for index in range(g['count'])]
    p.require(len(result['requests'])==len(expected)==22,'Request count')
    seen=set()
    for row,(group,index)in zip(result['requests'],expected):
        p.require(row['group']==group['id'] and row['index']==index,'Request order')
        p.require(row['prompt_token_ids']==plan['prompts'][group['prompt']],'Engine prompt changed')
        p.require(type(row['token_ids'])is list and len(row['token_ids'])==group['tokens'] and
            all(type(x)is int and 0<=x<128256 for x in row['token_ids']),'Engine output IDs/budget')
        p.require(row['finish_reason']=='length' and row['model']=='smollm3-3b','Finish/model')
        p.require(row['streaming'] is group['stream'],'Stream mode')
        p.require(type(row['request_id'])is str and row['request_id'] not in seen,'Repeated/missing API ID');seen.add(row['request_id'])
        u=row['usage'];n=len(row['prompt_token_ids']);m=group['tokens']
        p.require(all(type(u.get(k))is int and u[k]==v for k,v in dict(prompt_tokens=n,completion_tokens=m,total_tokens=n+m).items()),'Usage closure')
        if group['id'].startswith('initial-')and not group['id'].endswith('warm8'):
            p.require(hashlib.sha256(row['text'].encode()).hexdigest()==plan['historical'][role]['initial_text_sha256'],'Historical initial text mismatch')
    target=result['requests'][-1]
    p.require(hashlib.sha256(target['text'].encode()).hexdigest()==plan['historical'][role]['short_first_single_text_sha256']and
        target['usage']==plan['historical'][role]['short_first_single_usage'],'Historical target mismatch')
    p.require(result['checks']==dict(initial_count=True,initial_texts_equal=True,target_text_equal=True,target_usage_equal=True),
              'Reported checks contradict independent reconstruction')
    p.require(result['passed']is True and result['complete']is True and result['full_model_qualified']is False and
              result['new_speedup_claim']is False,'Capture qualification fields')
    return True


def compare_captures(results,plan):
    """Pure post-drain metadata comparison; never manufactures v8 receipts."""
    p.require(set(results) == set(p.ROLES), 'Three complete fresh roles required')
    for role, result in results.items():
        validate_capture(plan,role,result)
    parser = parser_module()
    rows = {}
    for role in p.ROLES:
        item = results[role]['requests'][-1]
        p.require(item['group'] == 'short-round0-single' and len(item['token_ids']) == 512,
                  'Wrong target/budget')
        rows[role] = item
    p.require(all(row['prompt_token_ids'] == rows['reference']['prompt_token_ids'] for row in rows.values()),
              'Different engine prompts')
    first = parser.first_different_token(rows['reference']['token_ids'], rows['nki_write']['token_ids'])
    combined = parser.first_different_token(rows['reference']['token_ids'], rows['nki_read_write']['token_ids'])
    return dict(capture_complete=True, historical_texts_reproduced=True, write_first_difference=first,
        combined_first_difference=combined, three_roles_common_prefix=(
            first is not None and rows['nki_read_write']['token_ids'][:first['index']] == first['common_prefix']),
        token_positions_are_fresh_engine_positions=True, old_token_positions_recovered=False,
        graph_replay_or_cache_cause_proven=False, full_model_qualified=False, new_speedup_claim=False)


def capture(plan, role, directory, requester=request):
    rows = []
    for group in p.groups():
        p.performance_guard(); p.storage()
        if group['id'].endswith('warm8'):
            current = [requester(plan, role, group, 0, directory)]
        else:
            barrier = threading.Barrier(group['count'])
            def one(index):
                barrier.wait()
                return requester(plan, role, group, index, directory)
            with concurrent.futures.ThreadPoolExecutor(group['count']) as executor:
                current = list(executor.map(one, range(group['count'])))
        rows.extend(current)
        p.once(directory / (group['id'] + '-group.json'), dict(complete=True, group=group, requests=current))
    p.require(len(rows) == 22, 'Incomplete original prefix')
    historical = plan['historical'][role]
    initial = [row for row in rows if row['group'].startswith('initial-') and row['group'] != 'initial-warm8']
    target = rows[-1]
    checks = dict(initial_count=len(initial) == 19,
        initial_texts_equal=all(hashlib.sha256(row['text'].encode()).hexdigest() == historical['initial_text_sha256'] for row in initial),
        target_text_equal=hashlib.sha256(target['text'].encode()).hexdigest() == historical['short_first_single_text_sha256'],
        target_usage_equal=target['usage'] == historical['short_first_single_usage'])
    result = dict(passed=all(checks.values()), complete=True, role=role, checks=checks,
        requests=rows, historical_generated_ids_reconstructed=False,
        fresh_generated_token_origin='engine API return_token_ids',
        full_model_qualified=False, new_speedup_claim=False, request_scope=22,
        original76_requests_qualified=False, common_prefix_comparison_pending=True,
        observer_alignment_pending=True, observer_invariance_pending=True,
        failure_kind=None if all(checks.values()) else 'historical_text_or_usage_not_reproduced')
    # In contrast to frozen v9 compare(), a false result is always persisted first.
    p.once(directory / 'capture-result.json', result)
    p.require(result['passed'], 'Historical output did not reproduce; see retained capture-result.json')
    return result


def native_setup(plan, role, directory):
    # Only reached by authenticated --server/--client; source tests do not call.
    p.require(not any(x == 'vllm_neuron' or x.startswith('vllm_neuron.') for x in sys.modules), 'Package already imported')
    wheel = p.read(p.ROOT / 'smollm3-cache-ablation-native-v3/wheel.json')
    for name, expected in wheel['installed_files'].items():
        p.require(p.sha(p.TARGET / name) == expected, 'Installed package changed: ' + name)
    sys.path.insert(0, str(p.TARGET))
    for role_name, files in plan['artifact_sha256'].items():
        if role_name == role:
            for path, digest in files.items():
                p.require(p.sha(path) == digest, 'Retained artifact changed: ' + path)
    import vllm.plugins
    vllm.plugins.load_general_plugins()
    from vllm_neuron.models.smollm3.identity import build_identity
    from vllm_neuron.models.smollm3.artifacts import admit_existing
    from vllm_neuron.models.smollm3.modeling import NeuronSmolLM3ForCausalLM as cls
    case = plan['cases'][role]
    saved = p.read(Path(case['artifact']) / 'neuron_config.json'); saved['metadata'] = None
    identity = build_identity(case['model_path'], saved)
    p.require(identity['variant'] == role, 'Wrong cache variant')
    admit_existing(case['artifact'], identity)
    def forbidden(*args, **kwargs):
        raise RuntimeError('Token diagnostic must never compile or save an artifact')
    cls.compile = forbidden; cls.save = forbidden
    return cls


def tokenizer_proof(plan, role):
    import orchestrate as runner
    case = plan['cases'][role]
    contract = p.read(p.ROOT / 'research/smollm3-cache-ablation-request-inputs-20261106-v1.json')
    initial = runner.prompt_ids(case, contract['initial_prompt_text'])
    source = (p.ROOT / 'benchmark_workloads.py').read_text()
    node = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == 'inputs')
    namespace = {}; exec(compile(ast.Module(body=[node], type_ignores=[]), 'benchmark_workloads.py', 'exec'), namespace)
    inputs = namespace['inputs'](case)
    p.require(initial == plan['prompts']['initial'] == contract['initial_prompt_ids'], 'Official initial prompt changed')
    p.require(inputs[128] == plan['prompts']['short'] == contract['workload_inputs']['128'], 'Official short prompt changed')
    return dict(passed=True, initial_prompt_ids=initial, short_prompt_ids=inputs[128],
                native_template_date_utc='2026-10-31', output_retokenized=False)


def server(role, spawned=False):
    plan, admission = p.guard(role, 'server', spawned)
    directory = p.DEST / role
    cls = native_setup(plan, role, directory)
    import torch
    from vllm_neuron.worker.neuronx_distributed_model_runner import NeuronxDistributedModelRunner
    from smollm3_token_capture_observer_v4 import install
    install(role, directory, cls, NeuronxDistributedModelRunner, torch)
    if spawned:
        return
    import orchestrate as runner
    case = plan['cases'][role]
    sys.argv = [str(p.SCRIPT), '--model', case['model_path'], '--served-model-name', case['served_name'],
        '--tensor-parallel-size', '2', '--max-model-len', '2048', '--max-num-seqs', '4',
        '--no-enable-prefix-caching', '--host', '127.0.0.1', '--port', '8010',
        '--additional-config', json.dumps({'override_neuron_config': runner.overrides(case)})]
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__', alter_sys=False)


def client(role):
    plan, admission = p.guard(role, 'client')
    # Authenticate the live server before tokenizer/HTTP; never use an unidentified port.
    owner = p.read(p.DEST / role / 'server-owner.json'); live = p.properties(p.unit(role, 'server'))
    p.require(live['MainPID'] == str(owner['pid']) and live['InvocationID'] == owner['invocation_id'] and
              live['ControlGroup'] == owner['cgroup'] and live['ActiveState'] == 'active', 'Wrong live server')
    p.require(p.proc(owner['pid'])['start_ticks'] == str(owner['start_ticks']) and
              owner['attempt_id'] == admission['attempt_id'], 'Stale server owner')
    directory = p.DEST / role
    # Tokenizer lookup follows original orchestrate behavior; no model constructed here.
    p.once(directory / 'tokenizer-proof.json', tokenizer_proof(plan, role))
    return capture(plan, role, directory)


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true'); action.add_argument('--client', action='store_true'); action.add_argument('--server', action='store_true')
    parser.add_argument('--role', choices=p.ROLES)
    args = parser.parse_args()
    if args.check:
        plan = p.checked_plan()
        print(json.dumps(dict(passed=True, source_only=True, requests_per_role=22, roles=list(p.ROLES),
            sources=len(plan['source_sha256']), native_imports=False, payload_bytes_read=0,
            services_launched=False, runtime_admitted=False, root_supervisor_implemented=True)))
        return
    p.require(args.role is not None, 'Role required')
    try:
        return server(args.role) if args.server else client(args.role)
    except BaseException as error:
        path = p.DEST / args.role / ('server' if args.server else 'client')
        failure = Path(str(path) + '-failure.json')
        if failure.parent.is_dir() and not failure.exists():
            p.once(failure, dict(passed=False, error=repr(error), full_model_qualified=False))
        raise


if __name__ == '__mp_main__' and os.environ.get(p.SPAWN):
    server(os.environ[p.SPAWN].split(':', 1)[0], spawned=True)
elif __name__ == '__main__':
    main()
