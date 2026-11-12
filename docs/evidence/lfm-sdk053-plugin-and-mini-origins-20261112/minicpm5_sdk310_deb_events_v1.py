"""Pure exact success journal validation for eight fixed packages."""
import dory_serial_stage_contract_v2 as c
MAX_EVENTS = 41


def expected_events(rows, mode):
    c.require(mode in ('download','replay') and len(rows) == 8 and len({r['package'] for r in rows}) == 8, 'Eight unique packages and one exact mode')
    events = []
    for row in rows:
        events += [(name,row['package']) for name in (('archive_download_start','archive_download_complete') if mode == 'download' else ('archive_replay_start',))]
        events += [(name,row['package']) for name in ('dpkg_deb_start','dpkg_deb_complete','package_members_verified')]
    events.append(('minicpm5_eight_packages_complete',None))
    c.require(len(events) == (41 if mode == 'download' else 33), 'Exact finite success event count')
    return events


def validate_success(events, rows, mode, result, result_ref):
    expected = expected_events(rows, mode)
    c.require(len(events) == len(expected) <= MAX_EVENTS, 'No omitted, duplicate or surplus events')
    c.require(len(result['packages']) == len(rows), 'Complete package result before event audit')
    cursor = 0
    for row, package in zip(rows,result['packages']):
        name = row['package']; inspection = package['inspection']; archive = inspection['archive']; decoder = inspection['decompressor']
        c.require(package['package'] == name and package['version'] == row['version'], 'Exact package identity for event association')
        local = expected[cursor:cursor+(5 if mode == 'download' else 4)]
        for kind, owner in local:
            event = events[cursor]; cursor += 1
            c.require(event.get('event') == kind, 'Exact ordered event kind')
            if kind != 'dpkg_deb_start': c.require(event.get('package') == owner, 'Exact event package; no cross-package evidence')
            if kind == 'archive_download_start': c.require(event['url'] == row['url'] and event['expected'] == dict(bytes=row['bytes'],sha256=row['sha256']), 'Exact signed download intent')
            elif kind == 'archive_download_complete': c.require(event['saved'] == archive and event['identity'] == inspection['archive_identity'], 'Full download receipt matches inspected archive')
            elif kind == 'archive_replay_start': c.require(event['archive'] == archive, 'Exact replay archive')
            elif kind == 'dpkg_deb_start': c.require(event['argv'] == decoder['argv'] == ['/usr/bin/dpkg-deb','--fsys-tarfile',archive['path']], 'Exact owned package decoder command')
            elif kind == 'dpkg_deb_complete': c.require(event['receipt'] == decoder and decoder['returncode'] == 0 and decoder['stdout_bytes'] == inspection['tar']['stream_bytes'], 'Full decoder EOF and actual success receipt')
            elif kind == 'package_members_verified': c.require(event['targets'] == len(row['targets']) and event['tar_sha256'] == inspection['tar']['stream_sha256'], 'Complete corresponding member stream')
    c.require(events[cursor] == dict(event='minicpm5_eight_packages_complete',mode=mode,result=result_ref,packages=8,targets=12,original_matches=0,current_matches=12,required_packages=3,candidate_packages=5), 'Final exact eight-package receipt')
    return dict(events=len(events), packages=8, ordered=True, mode=mode)
