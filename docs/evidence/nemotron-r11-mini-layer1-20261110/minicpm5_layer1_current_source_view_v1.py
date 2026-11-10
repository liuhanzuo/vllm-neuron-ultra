"""Pass a reviewed caller's source context to inherited prerequisite checks.

This is an in-memory execution view. Original collector records are validated
separately and remain unchanged; the view is never saved as historical evidence.
The caller must authenticate its full source closure before constructing it.
"""
import copy


def current_source_view(historical_spec, authenticated_current_sources):
    historical = historical_spec['source_sha256']
    if not all(authenticated_current_sources.get(path) == digest
               for path, digest in historical.items()):
        raise RuntimeError('Every historical source must retain its reviewed digest')
    view = copy.deepcopy(historical_spec)
    view['source_sha256'] = dict(authenticated_current_sources)
    return view
