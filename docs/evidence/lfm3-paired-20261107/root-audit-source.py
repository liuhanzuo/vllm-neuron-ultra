"""Independent, standard-library-only audit of completed LFM paired results.

No model, compiler, tokenizer, payload loading, service mutation, or canonical
write. This checks recorded observations; it does not rerun model inference.
"""
import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
D = R / 'lfm3-package-paired-full-v2'
OUT = R / 'research/lfm3-package-paired-v2-final-root-audit-20261107-v1.json'
evidence = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    evidence[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def guard():
    for name in ('smollm3-cache-ablation-full-v8', 'lfm3-package-paired-full-v2'):
        assert not json.loads((R / name / 'status.json').read_text())['performance_window']


def metrics(rows):
    assert rows
    for row in rows:
        assert all(math.isfinite(row[k]) for k in ('cosine', 'kl_reference_to_neuron', 'max_abs_error', 'reference_margin'))
        assert row['cosine'] >= .995 and row['kl_reference_to_neuron'] <= .02 and row['max_abs_error'] <= 2
        assert row['argmax_match'] == (row['reference_top1'] == row['neuron_top1'])
        assert row['reference_margin'] <= .5 or row['argmax_match']
    matches = sum(v['argmax_match'] for v in rows)
    assert matches / len(rows) >= .9
    return dict(rows=len(rows), top1_matches=matches, min_cosine=min(v['cosine'] for v in rows),
                max_kl=max(v['kl_reference_to_neuron'] for v in rows), max_abs=max(v['max_abs_error'] for v in rows),
                zero_max_abs_rows=sum(v['max_abs_error'] == 0 for v in rows))


def group_empty(group):
    path = Path('/sys/fs/cgroup' + group)
    if not path.exists():
        return dict(absent=True)
    rows = {}
    for file in [path / 'cgroup.procs', *path.rglob('*/cgroup.procs')]:
        if file.exists():
            value = file.read_text().strip()
            assert not value, (str(file), value)
            rows[str(file.relative_to(path))] = value
    for file in [path / 'cgroup.events', *path.rglob('*/cgroup.events')]:
        if file.exists():
            assert 'populated 1' not in file.read_text(), str(file)
    return dict(absent=False, recursive_process_lists=rows)


KEYS = 'LoadState,ActiveState,SubState,MainPID,ControlPID,Job,InvocationID,ControlGroup,Result,ExecMainPID,ExecMainCode,ExecMainStatus,ExecStart,KillMode,RemainAfterExit'


def properties(unit):
    raw = subprocess.check_output(['systemctl', '--user', 'show', '--all', unit, '--property=' + KEYS], text=True, timeout=15)
    return dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)


def terminal_audit():
    labels = ['reference-' + phase for phase in ('sampler', 'primitive-cpu', 'primitive-device', 'primitive-alias', 'inputs')]
    for role in ('reference', 'candidate'):
        labels += [role + '-' + phase for phase in ('compile', 'numeric', 'cold', 'long', 'quality-server', 'quality')]
    labels += ['reference-compare']
    for trial in range(3):
        for role in ('reference', 'candidate'):
            labels += [role + '-workloads-server-' + str(trial), role + '-workloads-' + str(trial)]
    assert len(labels) == len(set(labels)) == 30
    result = []
    for label in labels:
        launch = read(D / (label + '-launch.json'))
        owner = read(D / (label + '-owner.json'))
        receipt = read(D / (label + ('-drain.json' if launch['server'] else '-terminal.json')))
        name = 'trn1-latest-verify-lfm3-package-paired-v2-' + label
        group = '/user.slice/user-1000.slice/user@1000.service/app.slice/' + name + '.service'
        assert owner['unit'] == name and owner['cgroup'] == group
        assert owner['argv'] == launch['argv'] and owner['installed_argv'] == launch['installed_argv']
        assert '--unit=' + name in launch['command']
        assert 'LFM_PACKAGE_ATTEMPT=' + launch['nonce'] in launch['command']
        assert re.fullmatch('[0-9a-f]{32}', owner['invocation_id']) and re.fullmatch('[0-9]+', owner['start_ticks'])
        assert type(owner['pid']) is int and owner['pid'] > 0
        observations = []
        for i in range(2):
            value = properties(name)
            assert value['MainPID'] == value['ControlPID'] == '0' and not value['Job']
            assert value['ControlGroup'] in ('', group)
            assert value['ActiveState'] in ('inactive', 'failed') or (value['ActiveState'], value['SubState']) == ('active', 'exited')
            proc = Path('/proc') / str(owner['pid'])
            assert not proc.exists(), (name, owner['pid'])
            if value['LoadState'] == 'loaded':
                assert value['InvocationID'] == owner['invocation_id'] and value['ExecMainPID'] == str(owner['pid'])
                assert 'argv[]=' + owner['installed_argv'] + ' ;' in value['ExecStart']
            else:
                assert launch['server'] and value['LoadState'] == 'not-found' and not value['InvocationID']
            if not launch['server']:
                assert receipt['passed'] is True
                assert value['Result'] == 'success' and value['ExecMainCode'] == '1' and value['ExecMainStatus'] == '0'
                assert all(value.get(k) == receipt['properties'].get(k) for k in KEYS.split(','))
            observations.append(dict(properties=value, owner_pid_absent=True, recursive_group=group_empty(group)))
            if i == 0:
                time.sleep(.02)
        assert observations[0] == observations[1]
        result.append(dict(label=label, owner=owner, server=launch['server'], observations=observations,
                           execution_success_from_service_metadata=not launch['server'],
                           retired_server_absence_is_physical_drain_only=launch['server'] and observations[0]['properties']['LoadState'] == 'not-found'))
    assert not Path('/proc/1342036').exists()
    raw = subprocess.check_output(['systemctl', '--user', 'list-units', '--all', '--plain', '--no-legend', '--no-pager', 'trn1*'], text=True, timeout=15)
    live = []
    for line in raw.splitlines():
        parts = line.split()
        unit = next((v for v in parts if v.startswith('trn1') and v.endswith('.service')), None)
        if unit:
            value = properties(unit)
            if value['MainPID'] != '0' or value['ControlPID'] != '0' or value['Job']:
                live.append(dict(unit=unit, properties=value))
    assert not live, live
    return dict(units=result, units_checked=30, workers_successful=20, servers_physically_drained=10,
                standalone_controller=dict(pid=1342036, recorded_start_ticks='379124863', absent=True),
                other_active_trn1_user_units=live)


def main():
    guard()
    with (R / 'serial-enrollment.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard()
        assert not OUT.exists()
        qualification = read(D / 'qualification.json')
        plan_path = R / 'research/lfm3-package-paired-full-plan-20261107-v2.json'
        plan = read(plan_path)
        assert sha(plan_path) == '8958693311a80cdf98d20f2aa6e7be28b989e050524d5be0eb7179269373f9da'
        for path, digest in plan['source_sha256'].items():
            assert sha(path) == digest, path
        assert sha(R / 'run_lfm3_package_paired_full_v2.py') == '1e2967efda7d65fc8a34cab3eb31c19c4c8cc72402caf7c44354c47ca3f4ea40'
        repo = plan['repository']
        assert subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip() == plan['commit']
        assert not subprocess.check_output(['git', '-C', repo, 'status', '--porcelain'], text=True).strip()
        canonical_before = {name: sha(R / name) for name in ('continuation-state.json', 'roadmap.json', 'best-known-configurations.json')}
        admission = read(D / 'admission.json')
        assert canonical_before == admission['canonical_sha256']
        identities = {}
        for role in ('reference', 'candidate'):
            path = Path('/mnt/nvme/lfm3-package-paired-full-v2') / ('artifact-' + role) / 'lfm3-ultra-artifact.json'
            artifact = read(path)
            assert artifact['complete']
            assert artifact['identity']['variant'] == ('reference' if role == 'reference' else 'nki_predicated_decode_v4')
            identities[role] = dict(path=str(path), manifest=artifact, payloads_rehashed_in_this_audit=False)
        for k in ('configuration', 'model', 'scope', 'software', 'sources', 'environment', 'hardware'):
            assert identities['reference']['manifest']['identity'][k] == identities['candidate']['manifest']['identity'][k], k
        assert identities['reference']['manifest']['files']['model.pt'] != identities['candidate']['manifest']['files']['model.pt']
        reports = {role: [read(D / role / ('workloads-round-' + str(i) + '.json')) for i in range(3)] for role in ('reference', 'candidate')}
        specs = {'short_decode': (128, 512, {}), 'medium_prompt': (512, 128, {}), 'long_prompt': (1536, 128, {}),
                 'short_stochastic': (128, 512, {'temperature': .7, 'top_p': .9, 'top_k': 40})}
        rates = {}; pairs = []; usages = 0
        for role, rounds in reports.items():
            rates[role] = {}
            for trial, report in enumerate(rounds):
                assert report['complete'] and report['trial'] == trial
                assert report['inputs'] == reports['reference'][0]['inputs']
                for name, work in report['workloads'].items():
                    assert len(work['rounds']) == 1
                    mixed = name == 'mixed_lengths'
                    if mixed:
                        assert work['max_client_inflight'] == 4 and work['requests_per_round'] == 16 and work['sampling'] == specs['short_stochastic'][2]
                        groups = {'mixed': work['rounds'][0]}
                    else:
                        n, m, sampling = specs[name]
                        assert work['sampling'] == sampling and work['batch'] == 4 and work['output_tokens_per_request'] == m
                        assert work['prompt_inputs'] == [report['inputs'][str(n)]]
                        groups = {**work['rounds'][0], 'streaming': work['streaming']}
                    for kind, group in groups.items():
                        rows = group['requests']
                        expected = [(32, 32), (128, 128), (512, 256), (1536, 128)] * 4 if mixed else [(n, m)] * (1 if kind == 'single' else 4)
                        assert [(v['usage']['prompt_tokens'], v['usage']['completion_tokens']) for v in rows] == expected
                        assert all(v['usage']['total_tokens'] == v['usage']['prompt_tokens'] + v['usage']['completion_tokens'] for v in rows)
                        count = sum(v['usage']['completion_tokens'] for v in rows)
                        assert count == group['output_tokens'] and group['elapsed_s'] > 0 and math.isfinite(group['elapsed_s'])
                        rate = count / group['elapsed_s']
                        assert math.isclose(rate, group['throughput_tok_s'], rel_tol=1e-12)
                        rates[role].setdefault(name, {}).setdefault(kind, []).append(rate)
                        usages += len(rows)
                        if role == 'candidate' and name in ('short_decode', 'medium_prompt', 'long_prompt'):
                            rw = reports['reference'][trial]['workloads'][name]
                            refgroup = rw['streaming'] if kind == 'streaming' else rw['rounds'][0][kind]
                            for index, (a, b) in enumerate(zip(refgroup['requests'], rows)):
                                assert a['text'] == b['text'] and a['usage'] == b['usage'], (trial, name, kind, index)
                                pairs.append(dict(trial=trial, workload=name, mode=kind, request=index, text_equal=True, usage_equal=True))
        assert len(pairs) == 81 and usages == 312
        for role, workloads in rates.items():
            for name, kinds in workloads.items():
                for kind, values in kinds.items():
                    assert values == qualification['workloads']['raw_derived_rates'][role][name][kind]['raw_rates']
                    assert statistics.median(values) == qualification['workloads']['raw_derived_rates'][role][name][kind]['median_tok_s']
        order = [['reference', 'candidate'], ['candidate', 'reference'], ['reference', 'candidate']]
        starts = []
        for trial, roles in enumerate(order):
            for role in roles:
                owner = read(D / (role + '-workloads-' + str(trial) + '-owner.json'))
                starts.append(int(owner['start_ticks']))
        assert starts == sorted(starts) and len(set(starts)) == 6 and qualification['workloads']['round_order'] == order
        gates = {}
        for role in ('reference', 'candidate'):
            numeric = read(D / role / 'numeric.json'); long = read(D / role / 'long.json'); cold = read(D / role / 'cold.json')
            assert len(numeric['rows']) == 32 and len(long['rows']) == 84 and long['teacher_forced_decode_steps'] == 1529
            assert {'slot_reuse_prefill', 'decode_after_slot_reuse', 'partial_batch', 'resume_inactive'} <= {v['phase'] for v in long['rows']}
            assert [b['prefix'] for b in cold['buckets']] == [64, 160, 320, 768, 1280, 1536]
            coldrows = [v for bucket in cold['buckets'] for v in bucket['rows']]
            assert len(coldrows) == 96
            assert all(v['nan_elements'] == v['inf_elements'] == 0 and v['elements'] == 128000 for v in coldrows)
            assert all(v['nan_elements'] == v['inf_elements'] == 0 for bucket in cold['buckets'] for v in bucket['state_after_all_three_steps'])
            gates[role] = dict(numeric=metrics(numeric['rows']), long=metrics(long['rows']), cold_finite_rows=96,
                               cold_states_finite=True, long_steps=1529, long_phases=sorted({v['phase'] for v in long['rows']}))
            if role == 'candidate':
                gates[role]['cold_comparison'] = metrics([v['metrics_vs_original_neuron'] for v in coldrows])
                gates[role]['cold_bitwise_equal'] = sum(v['bitwise_equal_to_original'] for v in coldrows)
        paired = read(D / 'numeric-pair.json')
        pair_gates = {name: metrics(value['rows']) for name, value in paired.items()}
        quality = {role: read(D / role / 'quality.json') for role in ('reference', 'candidate')}
        qa, qb = quality['reference'], quality['candidate']
        assert qa['suite_sha256'] == qb['suite_sha256'] and qa['protocol'] == qb['protocol']
        fields = ('id', 'group', 'expected', 'prompt_ids', 'text', 'usage', 'finish_reason', 'correct', 'extracted', 'format_valid')
        assert len(qa['rows']) == len(qb['rows']) == 96
        for a, b in zip(qa['rows'], qb['rows']):
            assert all(a[k] == b[k] for k in fields)
        assert qa['correct'] == qb['correct'] == sum(v['correct'] for v in qa['rows']) == 63
        assert qa['groups'] == qb['groups'] and all(v['request_errors'] == 0 for v in qa['groups'].values())
        sampler = read(D / 'sampler.json')
        assert len(sampler['records']) == 90
        for row in sampler['records']:
            assert row['equal_tokens'] and row['equal_rng_state']
            if row['mode'] == 'topk40_p09':
                assert row['equal_shortlist'] is True and row['equal_upstream_reference'] is None
            else:
                assert row['mode'] in ('greedy', 'no_filter')
                assert row['equal_upstream_reference'] is True and row['equal_shortlist'] is None
        oracle = read(R / 'lfm3/long-decode-hf-reference.json')
        oracle_crosschecks = [v for fixture in oracle['fixtures'] for v in fixture['incremental_crosscheck']]
        assert all([v['position'] for v in fixture['incremental_crosscheck']] == [255, 256, 257] for fixture in oracle['fixtures'])
        oracle_metrics = metrics(oracle_crosschecks)
        cpu = read(D / 'primitive-cpu/validation.json'); checks = read(D / 'primitive-cpu/all-checks.json')
        assert len(checks) == 1648 and [v['id'] for v in checks] == list(range(1648))
        assert sum(v['kind'] == 'exhaustive' for v in checks) == 1560
        assert all(v['passed'] and v['bitwise_equal'] and v['inputs_unchanged'] for v in checks)
        assert cpu['extra_cases'] == checks[1560:]
        assert [i for batch in cpu['worker_batches'] for i in range(batch['start'], batch['stop'])] == list(range(1648))
        assert all(v['passed'] for v in cpu['worker_batches'])
        device = read(D / 'primitive-device/validation.json'); alias = read(D / 'primitive-alias/validation.json')
        assert len(device['graphs']) == 9 and sum(len(v['checks']) for v in device['graphs']) == 43
        assert all(v['bitwise_equal'] and v['cache_input_unchanged'] and v['updates_input_unchanged'] for g in device['graphs'] for v in g['checks'])
        assert len(alias['graphs']) == 12 and sum(len(v['checks']) for v in alias['graphs']) == 60
        assert all(v['attention_finite'] and v['cache_k_bitwise'] and v['cache_v_bitwise'] and v['input_k_bitwise'] and v['input_v_bitwise'] for g in alias['graphs'] for v in g['checks'])
        terminals = terminal_audit()
        guard()
        assert canonical_before == {name: sha(R / name) for name in canonical_before}
        out = dict(schema='lfm3_package_paired_v2_final_independent_root_audit_v1', observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   reviewer='/root', passed=True, scoped_package_qualification_accepted=True, whole_family_complete=False,
                   repository_commit=plan['commit'], source_pins_verified=len(plan['source_sha256']), repository_clean=True,
                   scope=plan['scope'], baseline='Packaged branch reference adapter with the same fast CPU sampler; not vanilla upstream vLLM-Neuron.',
                   workloads=qualification['workloads'], deterministic_request_pairs=pairs, deterministic_pairs_equal=81,
                   usage_records_checked=usages, independent_rates_recomputed=78, numeric_gates=gates, numeric_pairs=pair_gates,
                   original_full_causal_oracle_incremental_crosscheck=oracle_metrics,
                   quality=dict(identical_cases=96, correct_each=63, total=96, groups=qa['groups'], suite_sha256=qa['suite_sha256'],
                                custom_finite_suite=True, general_equivalence_claim=False),
                   sampler=dict(records_verified=90, scope=sampler['admitted_input_scope'],
                                historical_vs_packaged_tokens_and_rng_equal=90, upstream_reference_equal=60,
                                topk_shortlists_equal=30, topk_upstream_token_equivalence_claim=False,
                                explicit_request_generators_qualified=False, http_seed_support_qualified=False),
                   primitive_checks=dict(cpu=1648, device_graphs=9, device_checks=43, alias_graphs=12, alias_checks=60),
                   artifacts=identities, terminal_audit=terminals, evidence_sha256=evidence,
                   canonical_sha256_unchanged=canonical_before, new_runtime_started=False, inference_rerun_in_this_audit=False,
                   limitations=['Three alternating sequential rounds, no randomized experiment or statistical significance claim.',
                                'Deterministic workload comparison is decoded text and usage, not a token-ID trace.',
                                'Stochastic workloads have equal settings and budgets; their generated texts are not required equal.',
                                'Cold cache is finite and within the original tolerance; only 90/96 rows are bitwise equal.',
                                'Retired server absence establishes physical drain; successful serving is evidenced by completed request records.',
                                'Numeric rows and compiler manifests were reviewed; tensor payloads were not reloaded or rehashed in this audit.'])
        with OUT.open('x') as stream:
            json.dump(out, stream, indent=2, ensure_ascii=False); stream.write('\n')
        print(json.dumps(dict(path=str(OUT), sha256=sha(OUT), passed=True, deterministic_pairs=81, terminal_units=30)))


if __name__ == '__main__':
    main()
