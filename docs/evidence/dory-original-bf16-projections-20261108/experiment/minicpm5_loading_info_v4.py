"""Strict JSON boundary for the pinned native MiniCPM5 loading report."""
SET_FIELDS = ('missing_keys', 'unexpected_keys', 'mismatched_keys')
FIELDS = (*SET_FIELDS, 'error_msgs')


def normalize_loading_info(loading):
    assert type(loading) is dict and set(loading) == set(FIELDS), 'Unexpected loading-info keys/type'
    for key in SET_FIELDS:
        assert type(loading[key]) is set and all(type(item) is str for item in loading[key]), (key, 'Expected set[str]')
    assert type(loading['error_msgs']) is list and all(type(item) is str for item in loading['error_msgs']), 'Expected error_msgs list[str]'
    assert not any(loading[key] for key in FIELDS), loading
    return {**{key: sorted(loading[key]) for key in SET_FIELDS}, 'error_msgs': list(loading['error_msgs'])}
