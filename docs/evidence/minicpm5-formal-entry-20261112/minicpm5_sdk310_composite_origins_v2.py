"""Original signed origins plus the two actually replayed stdlib members.

This is a saved-metadata verifier. Current ELF bytes, providers, maps and ABI
remain the separate SDK authenticator's job. No observation is promoted here.
"""
import copy
from research import minicpm5_sdk310_composite_origins_v1 as previous
from research import minicpm5_bootstrap_observation_v2 as discovery

require = previous.require
EXTRA_PATHS = frozenset((
    '/usr/lib/python3.10/lib-dynload/_contextvars.cpython-310-x86_64-linux-gnu.so',
    '/usr/lib/python3.10/lib-dynload/resource.cpython-310-x86_64-linux-gnu.so',
))


def replay_package_origins(reader, decision, pins):
    require(decision['schema'] == 'minicpm5_sdk310_native_source_review_v3',
            'Explicit supplemental member origin successor')
    origins = decision['package_origins']
    require(set(origins) == {'schema', 'old_libc6', 'new_eight', 'sdk310_supplemental_stdlib'}
            and origins['schema'] == 'minicpm5_sdk310_composite_package_origins_v1',
            'All original independent origins plus exact supplemental result')
    require(EXTRA_PATHS <= set(decision['native_files'])
            and not EXTRA_PATHS.intersection(decision['pure_sources']),
            'Both actually observed additional SDK extensions remain native')
    original = copy.deepcopy(decision)
    original['schema'] = 'minicpm5_sdk310_native_source_review_v2'
    original['package_origins'].pop('sdk310_supplemental_stdlib')
    for path in EXTRA_PATHS:
        original['native_files'].pop(path)
    expected = previous.replay_package_origins(reader, original, pins)
    ended, result = discovery.supplemental(reader.json)
    supplied = origins['sdk310_supplemental_stdlib']
    require(set(supplied) == {'result', 'actual_exit', 'recorder_actual_exit'}
            and previous.same(supplied['result'], ended['result'])
            and previous.same(supplied['actual_exit'], discovery.SUPPLEMENTAL_RECEIPTS['actual_exit'])
            and previous.same(supplied['recorder_actual_exit'], discovery.SUPPLEMENTAL_RECEIPTS['recorder_actual_exit']),
            'Exact actual producer and independent successful recorder')
    plan = reader.json(ended['plan'])
    review = reader.json(ended['review'])
    require(review['reviewer'] == '/root' and review['passed'] is True
            and review['blockers'] == [] and previous.same(review['plan'], ended['plan'])
            and review['source_sha256'] == plan['source_sha256']
            and all(pins.get(path) == sha for path, sha in plan['source_sha256'].items()),
            'All original supplemental executed source pins are retained')
    require({row['path'] for row in result['comparisons']} == EXTRA_PATHS,
            'Only the exact two completed additional members')
    for row in result['comparisons']:
        path = row['path']
        selected = decision['native_files'][path]
        required = dict(path=path, bytes=row['member']['bytes'], sha256=row['member']['sha256'],
                        package=row['package'], version=row['version'])
        require(selected == required and path not in expected,
                'Selected native member exactly matches independently audited complete replay')
        expected[path] = copy.deepcopy(selected)
    require(expected == {**decision['native_files'], **decision['pure_sources']},
            'Complete selected native and pure-source origin domains retained')
    return expected
