"""One pure reservation calculation, sharing only same-call immutable results.

No cache survives this call. Frozen codec validation, exact wire bytes and both
original budget results are retained. No IO, runtime/native authority or guard
is supplied to the original two pure budget calculators; none is removed here.
"""
from research import minicpm5_formal_literal_codec_v2 as codec
from research import minicpm5_formal_literal_result_v2 as result


def combined_budget(admission_wire, argv, literal, output_root):
    """Return exactly the old runtime budget updated with the old stdout budget.

    Decode/validate the whole admission and construct the conservative terminal
    once. The exact immutable exit wire is shared by the two size calculations.
    All input validation still belongs to the frozen codec. No file/read receipt
    or native identity is reused, and changed inputs get a fresh computation.
    """
    terminal = codec.prospective_terminal(admission_wire, argv, literal)
    owner_wire = codec.encode('owner', terminal['owner'], literal)
    exit_wire = codec.encode('exit', terminal, literal)
    fallback = dict(event='sdk_import_diagnostic_exit_receipt_write_failed',
        actual_parent_observed_terminal=terminal, exit_receipt_saved=False,
        reporting_error_type='', diagnostic_passed=False)
    fallback_bytes = len(codec.encode('exit_fallback', fallback, literal))
    total = len(admission_wire) + len(owner_wire) + len(exit_wire) + codec.OBSERVER_RESERVE
    codec.require(total <= codec.RUNTIME_CAP and fallback_bytes <= codec.SUMMARY_CAP,
                  'Complete original runtime/failure envelope must fit before SDK spawn')
    budget = dict(schema='minicpm5_formal_literal_runtime_reservation_v1',
        admission_wire_bytes=len(admission_wire), owner_wire_bound=len(owner_wire),
        exit_wire_bound=len(exit_wire), observer_receipt_reserve=codec.OBSERVER_RESERVE,
        fallback_wire_without_error_name_bytes=fallback_bytes,
        fallback_error_name_serialized_bytes_available=codec.SUMMARY_CAP-fallback_bytes,
        total_runtime_bound=total, runtime_cap=codec.RUNTIME_CAP, summary_cap=codec.SUMMARY_CAP,
        actual_sdk_owner_observed=False)
    refs = {name: dict(path=output_root+'/'+name, sha256='f'*64, bytes=codec.FILE_CAP)
            for name in result.FILES}
    normal = dict(schema=result.SCHEMA, output_root=output_root, files=refs, has_summary=False,
        expanded_result_sha256='f'*64, expanded_result_bytes=codec.FILE_CAP+codec.SUMMARY_CAP+4096)
    failed = result._unavailable({'exit_code':255}, exit_wire, b'', refs)
    failed['observer_payload_bytes'] = 2**63-1
    sizes = dict(stdout_reference_bound=len(codec.encoded(normal)),
                 stdout_unavailable_bound=len(codec.encoded(failed)))
    codec.require(max(sizes.values()) <= codec.SUMMARY_CAP,
                  'Both whole stdout schemas must fit before SDK spawn')
    budget.update(sizes)
    return budget
