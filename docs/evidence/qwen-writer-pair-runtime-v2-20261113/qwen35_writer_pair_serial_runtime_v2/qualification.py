"""Mandatory already-executed int32/sliced-KV gate before the model pair.

This does not execute or self-certify the geometry leaf. The future independent
root must bind its actual geometry exit to the complete serial catalog first.
No existing int64 primitive result satisfies this successor schema.
"""
from pathlib import Path

from .contract import require, reference
from .io import read


def check_geometry_qualification(spec):
    from qwen35_writer_pair_validation_v1.geometry import check_complete_geometry, LAYOUTS
    qualification = read(spec['geometry_qualification'])
    require(set(qualification) == {'schema', 'reviewer', 'passed', 'blockers', 'reports', 'actual_exit',
                                  'source_sha256', 'full_model_qualified'}
            and qualification['schema'] == 'qwen35_writer_geometry_independent_qualification_v1'
            and qualification['reviewer'] == '/root' and qualification['passed'] is True
            and qualification['blockers'] == [] and qualification['full_model_qualified'] is False,
            'Independent actual int32/sliced-KV qualification is mandatory')
    require(set(qualification['reports']) == set(LAYOUTS), 'All eight geometry layouts required')
    references = [spec['geometry_qualification'], qualification['actual_exit'], *qualification['reports'].values()]
    for ref in references:
        reference(ref)
        bound = spec['input_extension'].get(ref['path'])
        require(bound is not None and bound['sha256'] == ref['sha256'], 'Geometry input outside immutable root binding')
    catalog = read(spec['current_serial_catalog'])
    require(sum(row['terminal'] == qualification['actual_exit'] for row in catalog['runs']) == 1,
            'Actual geometry exit must already occur once in complete serial history')
    actual = read(qualification['actual_exit'])
    require(actual['schema'] == 'qwen35_writer_geometry_root_actual_exit_v1'
            and actual['reviewer'] == '/root' and actual['actual_tool_exit_observed'] is True
            and actual['actual_tool']['exit_code'] == 0
            and actual['physically_drained'] is True and actual['owners_gone'] is True
            and actual['reports'] == qualification['reports']
            and actual['source_sha256'] == qualification['source_sha256']
            and actual['full_model_qualified'] is False,
            'Actual external successful exit, independent complete reports and drain required')
    require(qualification['source_sha256'] and all(spec['source_sha256'].get(path) == sha
            for path, sha in qualification['source_sha256'].items()), 'Geometry/source identity changed')
    reports = {name: read(ref) for name, ref in qualification['reports'].items()}
    result = check_complete_geometry(reports)
    require(all(report['source_sha256'] == qualification['source_sha256'] for report in reports.values()),
            'Geometry reports crossed executed source versions')
    return dict(qualification=spec['geometry_qualification'], actual_exit=qualification['actual_exit'],
                report_gate=result, full_model_qualified=False)
