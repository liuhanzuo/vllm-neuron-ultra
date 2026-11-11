"""Inert explicit-control matched writeback diagnostic successor; no CLI or admission producer.

The original scientific suite body and its fail-fast assertions are unchanged.
Original, narrowly substituted, and restored-original arms remain distinct.
No installed SDK file or original diagnostic module is modified.
"""
from lfm3_nki05_transport_kernel_v2 import *
import lfm3_nki05_transport_kernel_v2 as original
from lfm3_nki05_writeback_probe_v1 import WritebackProbe, host_snapshot, host_differences, require

_ACTIVE_PROBE = None


def bounded_error_trace(error):
    """Keep four innermost code locations within the original report budget."""
    frames = []; total = 0; cursor = error.__traceback__
    while cursor is not None:
        code = cursor.tb_frame.f_code
        frames.append(dict(filename=code.co_filename[-192:], function=code.co_name[:64], line=cursor.tb_lineno))
        frames = frames[-4:]; total += 1; cursor = cursor.tb_next
    filename = getattr(error, 'filename', None)
    return dict(error_type=type(error).__name__[:64], exception_filename=filename[-192:] if type(filename) is str else None,
                innermost_frames=frames, total_frames=total, omitted_outer_frames=total-len(frames), frame_limit=4,
                frame_locals_captured=False, source_text_read=False)


def run_case(nki, name, operation, cache, updates, slots):
    probe = _ACTIVE_PROBE
    require(probe is not None, 'Explicit admitted arm around every original case')
    if not probe.installed:
        probe.install(sys.modules['nki._simulator'], sys.modules['numpy'],
                      SDK / '_simulator.py', PINS[SDK / '_simulator.py'])
    operands = (cache, updates, slots)
    before = host_snapshot(operands)
    probe.label = [name, operation]
    try:
        row = original.run_case(nki, name, operation, *operands)
        row['host_input_differences'] = host_differences(before, operands)
        return row
    except BaseException as error:
        probe.case_errors.append(dict(name=name, operation=operation, error=repr(error)[:2048],
                                      host_input_differences=host_differences(before, operands),
                                      error_trace=bounded_error_trace(error)))
        raise
    finally:
        probe.label = None


def _suite(admit):
    """Execute bounded diagnostic after caller's existing serial admission.

    Caller must enforce inherited MemoryMax=10GiB/MemoryHigh=8GiB/swap=0,
    TasksMax=32, CPU affinity and wall timeout <=1800s. The established worker
    limits (12GiB address space, 1800 CPU seconds, 256KiB report) are retained.
    A passing return is ONLY this migrated simulator diagnostic's result.
    """
    assert callable(admit) and admit() is True, 'Existing serial admission required'
    assert sys.prefix == PREFIX and sys.version_info[:2] == (3, 12)
    assert sys.byteorder == 'little'
    for key, value in {'NKI_PRECISE_FP': '1', 'NEURON_PLATFORM_TARGET_OVERRIDE': 'trn1',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}.items():
        assert os.environ.get(key, value) == value, (key, 'conflicting environment')
        os.environ[key] = value
    for kind, cap in ((resource.RLIMIT_AS, 12 * 2**30), (resource.RLIMIT_CPU, 1800),
                      (resource.RLIMIT_FSIZE, 256 * 2**10)):
        soft, hard = resource.getrlimit(kind)
        resource.setrlimit(kind, (min(cap, soft) if soft != resource.RLIM_INFINITY else cap,
                                 min(cap, hard) if hard != resource.RLIM_INFINITY else cap))
    source_sha256 = {}
    for path, wanted in PINS.items():
        assert path.stat().st_size < 2**20
        source_sha256[str(path)] = sha(path.read_bytes())
        assert source_sha256[str(path)] == wanted, str(path)
    result = dict(schema='lfm3_nki05_transport_diagnostic_v1', complete=False, passed=False,
                  source_sha256=source_sha256, diagnostic_source_sha256=sha(Path(__file__).read_bytes()),
                  legacy_candidate_changed=False, legacy_full3716_passed=False,
                  device_behavior_proven=False, full_model_qualified=False, new_speedup_claim=False,
                  target='trn1', precise_fp=True, checks=[], negative_controls=negative_controls())
    began = time.monotonic()
    try:
        from importlib.metadata import version
        result['versions'] = {name: version(name) for name in ('nki', 'neuronx-cc', 'numpy', 'ml_dtypes')}
        assert result['versions']['nki'] == '0.5.0+28631259367.ga768afa6'
        assert result['versions']['neuronx-cc'] == '2.26.6360.0+6f180f47'
        assert result['versions']['ml_dtypes'] == '0.6.0'
        import numpy as np
        import ml_dtypes
        import nki
        from lfm3_batch_slots_extended_v1 import numpy_inputs

        def check(name, operation, cache, updates, slots):
            row = run_case(nki, name, operation, cache, updates, slots)
            result['checks'].append(row)
            assert row['passed'], (name, row['first_boundary_divergence'], row['returned_difference'])

        cache, updates, slots = numpy_inputs(CASE)
        expected = byte_oracle(cache.tobytes(), updates.tobytes(), CASE['slots'], 8, 256, 2)
        hashes = tuple(sha(x.tobytes()) for x in (cache, updates, slots)) + (sha(expected),)
        assert hashes == OLD_HASHES, ('Original fixture changed; no fallback or normalization', hashes)
        result['exact_fixture_hashes_verified'] = True
        result['historical_failed_actual_sha256'] = OLD_ACTUAL
        for operation in ('identity', 'copy', 'permutation', 'selector'):
            check('exact_v5_first_case', operation, cache.copy(), updates.copy(), slots.copy())

        # Each of the 65,536 encodings traverses every movement control. At most
        # 128 KiB per BF16 operand; kernels use 128-element SBUF tiles.
        raw = b''.join(struct.pack('<H', word) for word in range(65536))
        cache = np.frombuffer(raw, dtype=ml_dtypes.bfloat16).reshape(8, 8192).copy()
        updates = np.frombuffer(raw[::-1], dtype=ml_dtypes.bfloat16).reshape(8, 8192).copy()
        slots = np.asarray(list(reversed(range(8))), dtype=np.int32)
        for operation in ('identity', 'copy', 'permutation', 'selector'):
            check('all65536_bf16_encodings', operation, cache.copy(), updates.copy(), slots.copy())
        result['bf16_encoding_count'] = 65536
        result['bf16_encoding_source_sha256'] = sha(raw)

        for dtype in ('bfloat16', 'float32'):
            for capacity in (4, 8, 16):
                cases = {'inactive': [-1, capacity, -(2**31), 2**31 - 1],
                         'duplicate_last_writer': [0, 0, -1, 0],
                         'last_winner_before_inactive': [capacity - 1, capacity - 1, -1, capacity]}
                for name, mapping in cases.items():
                    case = dict(CASE, capacity=capacity, dtype=dtype, slots=mapping)
                    operands = numpy_inputs(case)
                    check(f'{dtype}_B{capacity}_{name}', 'selector', *operands)
                if capacity == 8:
                    for operation in ('identity', 'copy', 'permutation'):
                        check(dtype + '_movement_controls', operation, *numpy_inputs(case))

        def arithmetic(source):
            import nki.language as nl
            value = nl.load(source)
            out = nl.ndarray(source.shape, dtype=nl.float32, buffer=nl.shared_hbm)
            nl.store(out, value=nl.add(value, 1.0))
            return out

        def cast(source):
            import nki.language as nl
            import nki.isa as nisa
            value = nl.load(source)
            converted = nl.ndarray(source.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=converted, src=value)
            out = nl.ndarray(source.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
            nl.store(out, value=converted)
            return out

        source = np.frombuffer(struct.pack('<4f', -2, -1, 0, 2), dtype=np.float32).reshape(1, 4).copy()
        actual = nki.simulate(arithmetic)(source)
        result['arithmetic_control'] = strict_equal(actual.tobytes(), actual.dtype.name, actual.shape,
                                                   struct.pack('<4f', -1, 0, 1, 3), 'float32', (1, 4))
        source = np.frombuffer(struct.pack('<4I', 0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000),
                               dtype=np.float32).reshape(1, 4).copy()
        actual = nki.simulate(cast)(source)
        result['round_to_nearest_even_cast_control'] = strict_equal(
            actual.tobytes(), actual.dtype.name, actual.shape,
            struct.pack('<4H', 0x3f80, 0x3f82, 0xbf80, 0xbf82), 'bfloat16', (1, 4))
        assert result['arithmetic_control'] and result['round_to_nearest_even_cast_control']
        assert all(result['negative_controls'].values())
        result.update(complete=True, passed=True)
    except Exception as error:
        result['error'] = repr(error)
        result['error_trace'] = bounded_error_trace(error)
    result['seconds'] = time.monotonic() - began
    result['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert len(json.dumps(result, allow_nan=False).encode()) <= 256 * 2**10
    return result


def diagnose(admit):
    global _ACTIVE_PROBE
    require(_ACTIVE_PROBE is None, 'No nested or overlapping diagnostic arms')
    result = dict(schema='lfm3_nki05_writeback_diagnostic_v2', complete=False, passed=False,
                  original_suite_passed=False, candidate_suite_passed=False,
                  original_failure_preserved=True, restored_original_reproduced=False,
                  writeback_precision_explanation_supported=False,
                  legacy_candidate_changed=False, legacy_full3716_passed=False,
                  device_behavior_proven=False, full_model_qualified=False, new_speedup_claim=False,
                  installed_sdk_source_modified=False, source_sha256={})
    began = time.monotonic()
    def arm(name, mode):
        global _ACTIVE_PROBE
        require(_ACTIVE_PROBE is None, 'Original extractor restored before next arm')
        probe = WritebackProbe(mode)
        row = result[name] = {}
        _ACTIVE_PROBE = probe
        try:
            row['suite'] = _suite(admit)
        finally:
            try:
                probe.restore()
            finally:
                _ACTIVE_PROBE = None
                row['probe'] = probe.report()
        require(probe.installed and probe.restored, 'Actual probe installed and original callable restored after each suite')
        return row['suite']
    try:
        # Each arm runs the original admission/source/version/precision/resource
        # preflight before run_case installs a process-local observation wrapper.
        before = arm('original', 'original')
        require(before.get('complete') is before.get('passed') is False
                and len(before.get('checks', [])) == 1
                and before['checks'][0]['name'] == 'exact_v5_first_case'
                and before['checks'][0]['operation'] == 'identity'
                and before['checks'][0]['observer_coverage'] is True
                and before['checks'][0]['inputs_unchanged'] is False
                and before['checks'][0]['first_boundary_divergence'] is None
                and before['checks'][0]['returned_difference']['mismatched_words'] == 0,
                'Reproduce the original r2 identity/writeback failure before candidate interpretation')
        candidate = arm('candidate', 'precise_bf16_writeback')
        restored = arm('restored_original', 'original')
        require(before.get('exact_fixture_hashes_verified') is True
                and candidate.get('exact_fixture_hashes_verified') is True
                and restored.get('exact_fixture_hashes_verified') is True,
                'All arms retain the exact original fixture before comparison')
        require(before.get('checks') and candidate.get('checks') and restored.get('checks'),
                'All arms contain an actually executed original first case')
        require(before['checks'][0]['input_sha256'] == candidate['checks'][0]['input_sha256']
                == restored['checks'][0]['input_sha256'], 'Matched original input bytes across arms')
        result['original_suite_passed'] = before['passed']
        result['candidate_suite_passed'] = candidate['passed']
        result['restored_original_reproduced'] = before['checks'] == restored['checks'] and before['passed'] == restored['passed']
        require(result['restored_original_reproduced'], 'Removal restores the original observed behavior')
        require(result['original']['probe']['substitutions'] == result['restored_original']['probe']['substitutions'] == 0,
                'Original arms never substitute extraction precision')
        require(len(result['original']['probe']['extraction_calls']) == 4
                and len(result['restored_original']['probe']['extraction_calls']) == 4,
                'Both original identity arms observe three input extractions and one returned output')
        if candidate['passed']:
            require(candidate['complete'] is True and len(candidate['checks']) == 32
                    and all(row['passed'] and row['inputs_unchanged'] and row['observer_coverage']
                            for row in candidate['checks'])
                    and candidate['bf16_encoding_count'] == 65536
                    and candidate['arithmetic_control'] is candidate['round_to_nearest_even_cast_control'] is True
                    and candidate['negative_controls'] == original.negative_controls(),
                    'All original movement, exhaustive encoding, arithmetic/cast and negative gates')
            events = result['candidate']['probe']['extraction_calls']
            changed = [event for event in events if event['substitution']]
            require(len(events) == 132 and len(changed) == 40
                    and result['candidate']['probe']['substitutions'] == 40,
                    'All 32 cases plus arithmetic/cast: exactly 40 BF16 input writeback substitutions')
            require(all(event['writeback_call'] and event['requested_precise'] == 'omitted'
                        and event['effective_precise'] is True and event['extracted_dtype'] == 'bfloat16'
                        and event['storage_matches_host_before'] is True
                        and event['precision_controls']['true']['difference']['changed_words'] == 0
                        for event in changed), 'Precise copies preserve every real original input')
            baseline = [event for event in result['original']['probe']['extraction_calls']
                        if 'precision_controls' in event]
            require(len(baseline) == 2 and all(event['effective_precise'] is False
                        and event['extracted_dtype'] == 'float32'
                        and event['precision_controls']['false']['difference']['changed_words'] > 0
                        and event['precision_controls']['true']['difference']['changed_words'] == 0
                        for event in baseline), 'Matched original extraction/copy primitive isolates precision loss')
            result['writeback_precision_explanation_supported'] = True
        result['complete'] = True
        # Completion of a characterization is not a retroactive original pass.
        result['passed'] = bool(before['passed'] and candidate['passed'] and restored['passed'])
    except Exception as error:
        result['error'] = repr(error)[:4096]
        result['error_trace'] = bounded_error_trace(error)
    result['seconds'] = time.monotonic() - began
    result['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(_ACTIVE_PROBE is None, 'No probe survives diagnostic return')
    for path in (Path(__file__), ROOT / 'lfm3_nki05_writeback_probe_v1.py', Path(original.__file__),
                 ROOT / 'lfm3_nki05_transport_diagnostic_v1.py'):
        result['source_sha256'][str(path.resolve())] = sha(path.read_bytes())
    require(len(json.dumps(result, allow_nan=False).encode()) <= 256 * 2**10,
            'Complete three-arm evidence fits original 256 KiB cap; never truncate')
    return result
