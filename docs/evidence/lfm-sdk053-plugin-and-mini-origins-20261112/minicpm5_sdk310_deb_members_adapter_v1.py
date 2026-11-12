"""Eight fixed Mini package archives; no install or native closure claim."""
from pathlib import Path
import dory_serial_stage_contract_v2 as c
import minicpm5_sdk310_deb_input_binding_v1 as binding
import dory_deb_transport_v3 as io


def run(guard, emit, trust, out, mode, prior_ref=None):
    c.require(trust == binding.TRUST and mode in ('download','replay') and (prior_ref is None) == (mode == 'download'), 'Exact reviewed Mini archive mode')
    n, authority = binding.authenticated(guard); tool = authority['tool']; guard(); Path(out).mkdir(mode=0o700)
    prior = c.load(prior_ref, 8*c.MIB) if mode == 'replay' else None
    if prior is not None:
        binding.validate_result(prior, n, 'download', None)
    packages = []; stream_total = current = 0
    for index, row in enumerate(n['packages']):
        guard()
        if mode == 'download':
            archive = Path(out)/(row['package']+'.deb'); io.download(row, archive, guard, emit)
        else:
            previous = prior['packages'][index]
            archive = Path(previous['inspection']['archive']['path']); io.file_ref(archive, guard, row)
            emit(dict(event='archive_replay_start', package=row['package'], archive=previous['inspection']['archive']))
        inspection = io.inspect(row, archive, tool, guard, emit)
        stream_total += inspection['tar']['stream_bytes']; c.require(stream_total <= 512*c.MIB, 'Original aggregate decompressed stream cap')
        comparisons = []
        for target in row['targets']:
            guard(); member = inspection['tar']['targets'][target['observed_path']]; expected = dict(bytes=member['size'],sha256=member['sha256'])
            c.require(target['original_observed_reference'] is None, 'No invented original SDK member hash')
            observed, identity = io.file_ref(target['observed_path'], guard, expected, expected_nlink=1)
            saved_identity = target['current_lstat']
            c.require(identity == {k:saved_identity['st_'+k] for k in ('dev','ino','mode','nlink','size','mtime_ns','ctime_ns')}, 'Exact prior lstat identity plus new complete content hash')
            current += 1
            comparisons.append(dict(target=target, member=dict(bytes=member['size'],sha256=member['sha256'],terminal=member['terminal']), original_matches=None, current=observed, current_identity=identity, current_matches=True, new_observation_only=True))
        package = dict(package=row['package'], version=row['version'], necessity=row['necessity'], actual_loaded=None, inspection=inspection, comparisons=comparisons)
        if prior is not None: c.require(package == prior['packages'][index], 'Independent complete archive/member/current replay equals producer')
        packages.append(package)
        emit(dict(event='package_members_verified', package=row['package'], targets=len(comparisons), tar_sha256=inspection['tar']['stream_sha256']))
    c.require(current == 12 and len(packages) == 8, 'All twelve selected targets across eight packages')
    value = dict(schema=binding.RESULT_SCHEMA, complete=True, mode=mode, replay_from=prior_ref, inputs=trust, packages=packages, archive_total_bytes=n['archive_total_bytes'], decompressed_total_bytes=stream_total, target_count=12, original_match_count=0, current_match_count=current, required_package_count=3, candidate_package_count=5, package_content_authenticated=True, loader_selection_established=False, complete_native_dependency_closure_established=False, native_ABI_qualified=False, full_model_qualified=False, new_speedup_claim=False)
    binding.validate_result(value, n, mode, prior_ref)
    result = c.save(Path(out)/'sources-result.json', value, 8*c.MIB); guard()
    emit(dict(event='minicpm5_eight_packages_complete', mode=mode, result=result, packages=8, targets=12, original_matches=0, current_matches=12, required_packages=3, candidate_packages=5))
    return dict(result=result, packages=8, targets=12, mode=mode, replay_from=prior_ref)
