"""Finite source-discovery API only: no network, process, filesystem or deb IO.

The caller supplies bounded HTTPS transport and an independently reviewed real
detached-signature verifier. A mock verifier is NOT cryptographic verification.
See dory-ubuntu-package-sources-interface-20261111-v1.md before integration.
"""
import datetime as dt
import email.utils
import hashlib
import io
import json
import lzma
import re

MIB = 2**20
PACKAGE_VERSIONS = {
    'libc6': '2.35-0ubuntu3.15',
    'libgcc-s1': '12.3.0-1ubuntu1~22.04.3',
    'libgomp1': '12.3.0-1ubuntu1~22.04.3',
    'libnuma1': '2.0.14-3ubuntu2',
    'libstdc++6': '12.3.0-1ubuntu1~22.04.3',
    'libtcmalloc-minimal4': '2.9.1-0ubuntu3',
}
PREFIX = '/usr/lib/x86_64-linux-gnu/'
TARGETS = {
    'libc6': tuple((PREFIX + n, 'lib/x86_64-linux-gnu/' + n) for n in
                  ('ld-linux-x86-64.so.2', 'libc.so.6', 'libdl.so.2', 'libm.so.6',
                   'libpthread.so.0', 'librt.so.1', 'libutil.so.1')),
    'libgcc-s1': ((PREFIX + 'libgcc_s.so.1', 'lib/x86_64-linux-gnu/libgcc_s.so.1'),),
    'libgomp1': ((PREFIX + 'libgomp.so.1.0.0', 'usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0'),),
    'libnuma1': ((PREFIX + 'libnuma.so.1.0.0', 'usr/lib/x86_64-linux-gnu/libnuma.so.1.0.0'),),
    'libstdc++6': ((PREFIX + 'libstdc++.so.6.0.30', 'usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30'),),
    'libtcmalloc-minimal4': ((PREFIX + 'libtcmalloc_minimal.so.4.5.9',
                            'usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.5.9'),),
}
SOURCES = (
    ('https://archive.ubuntu.com/ubuntu/', 'jammy'),
    ('https://archive.ubuntu.com/ubuntu/', 'jammy-updates'),
    ('https://security.ubuntu.com/ubuntu/', 'jammy-security'),
)
INPUT_PINS = {
    'next': ('dory-system-package-independent-bytes-next-inputs-20261111-v2.json',
             'aecccba2bc1b7dfa57a3a8f8e48e4c976df635a9da18fcf046ad0e91592b8e96'),
    'exit': ('dory-system-package-metadata-r1-actual-exit-20261111-v1.json',
             '934f2d0b9da075cf0d927557fcb302518d32a0ce99376e2ef6a18d245e21efe1'),
    'recorder': ('dory-system-package-metadata-r1-recorder-actual-exit-20261111-v1.json',
                 '91acd20ae8b661ec664713be3abaa708fe0129c7edfda1e2e0e5bfa8e873ded7'),
    'gaps': ('dory-inventory-r5-gap-classification-20261111-v1.json',
             'e902d47f1e0e580025480a0e32da7e75db3b603548e1915167722e113e9b1d03'),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    need(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'SHA256 syntax')
    return value


def count(value, cap):
    need(isinstance(value, str) and re.fullmatch('[0-9]{1,12}', value), 'Decimal size')
    value = int(value)
    need(0 < value <= cap, 'Positive bounded size')
    return value


def relative(value):
    need(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9+_.~/\-]*', value)
         and all(p not in ('', '.', '..') for p in value.split('/')), 'Safe relative metadata path')
    return value


def release_relative(value):
    """Release inventory names also include DEP-11 icon scale markers."""
    need(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9+_.~/@\-]*', value)
         and all(p not in ('', '.', '..') for p in value.split('/')), 'Safe relative Release metadata path')
    return value


def bind_inputs(raw_inputs):
    """Authenticate only four saved small JSON objects supplied as bytes."""
    need(set(raw_inputs) == set(INPUT_PINS), 'Exact saved input set')
    objects = {}
    for key, (_, expected) in INPUT_PINS.items():
        raw = raw_inputs[key]
        need(type(raw) is bytes and len(raw) <= MIB and sha(raw) == expected, 'Saved input SHA256: ' + key)
        objects[key] = json.loads(raw)
    nxt, ended, recorder, gaps = (objects[k] for k in ('next', 'exit', 'recorder', 'gaps'))
    need(nxt['independent_saved_audit_pending'] is False
         and nxt['independent_saved_audit']['sha256'] == INPUT_PINS['exit'][1]
         and nxt['independent_saved_audit_recorder_exit']['sha256'] == INPUT_PINS['recorder'][1], 'Completed next-input authority')
    need(ended['stage_success_observed'] and ended['actual_tool']['exit_code'] == 0
         and ended['parent_owner_gone'] and ended['child_owner_gone'] and ended['child_group_empty']
         and ended['complete_events_audit']['complete'], 'Completed metadata producer and audit')
    need(recorder['actual_tool']['exit_code'] == 0 and recorder['owner_gone']
         and recorder['independent_saved_audit_complete']
         and recorder['result']['sha256'] == INPUT_PINS['exit'][1], 'Completed independent recorder')
    need(nxt['source']['sha256'] == ended['evidence']['metadata-result.json']['sha256'], 'Metadata result cross-binding')
    packages = {p['package']: p for p in nxt['packages']}
    need(len(nxt['packages']) == 6 and set(packages) == set(PACKAGE_VERSIONS), 'Fixed six packages')
    observed = {r['path']: r['observed_reference'] for r in gaps['system_package_files']}
    all_targets = {path for targets in TARGETS.values() for path, _ in targets}
    need(set(observed) == all_targets - {PREFIX + 'libgomp.so.1.0.0'}, 'Eleven original payload references')
    for name, version in PACKAGE_VERSIONS.items():
        row = packages[name]
        need(row['version'] == version and row['architecture'] == 'amd64', 'Exact installed version/architecture')
        need(set(row['observed_targets']) == {path for path, _ in TARGETS[name]}, 'Fixed targets')
        need(all('/' + member in row['saved_package_names'] for _, member in TARGETS[name]), 'Saved member name mapping')
    return objects, observed


def stanzas(raw, cap, stanza_cap=256 * 1024):
    """Strict, finite deb822 parser; never accepts duplicate case-folded fields."""
    need(type(raw) is bytes and len(raw) <= cap and b'\x00' not in raw and b'\r' not in raw, 'Bounded LF metadata')
    row, previous, used, rows = {}, None, 0, 0
    for line in io.StringIO(raw.decode('utf-8')):
        line = line.removesuffix('\n')
        need(len(line) <= 65536, 'Metadata line cap')
        if not line:
            if row:
                yield row
                rows += 1
                need(rows <= 200000, 'Index stanza cap')
            row, previous, used = {}, None, 0
            continue
        used += len(line)
        need(used <= stanza_cap, 'Metadata stanza cap')
        if line[0] in ' \t':
            need(previous is not None, 'Continuation has field')
            row[previous] += '\n' + line[1:]
        else:
            name, separator, value = line.partition(':')
            need(separator and re.fullmatch('[A-Za-z0-9][A-Za-z0-9-]*', name), 'Field syntax')
            previous = name.lower()
            need(previous not in row, 'Duplicate metadata field')
            row[previous] = value.lstrip(' \t')
    if row:
        need(rows < 200000, 'Index stanza cap')
        yield row


def release_indexes(raw, suite, now):
    rows = list(stanzas(raw, 4 * MIB, 4 * MIB))
    need(len(rows) == 1, 'Single authenticated Release')
    r = rows[0]
    need(r.get('origin') == 'Ubuntu' and r.get('label') == 'Ubuntu'
         and r.get('suite') == suite and r.get('codename') == 'jammy'
         and 'amd64' in r.get('architectures', '').split()
         and {'main', 'universe'} <= set(r.get('components', '').split()), 'Exact Ubuntu suite/architecture/components')
    date = email.utils.parsedate_to_datetime(r['date'])
    need(date.utcoffset() is not None and date <= now + dt.timedelta(hours=24), 'Release date not in future')
    if 'valid-until' in r:
        until = email.utils.parsedate_to_datetime(r['valid-until'])
        need(until.utcoffset() is not None and date <= until and now <= until, 'Unexpired Release')
    selected = {}
    wanted = {c + '/binary-amd64/Packages' + suffix for c in ('main', 'universe') for suffix in ('', '.xz')}
    for line in r.get('sha256', '').splitlines():
        if not line.strip():
            continue
        fields = line.split()
        need(len(fields) == 3, 'Release SHA256 entry')
        h, size, path = fields
        digest(h)
        release_relative(path)
        if path in wanted:
            need(path not in selected, 'Duplicate Release index path')
            selected[path] = dict(sha256=h, bytes=count(size, 32 * MIB if path.endswith('.xz') else 128 * MIB))
    need(set(selected) == wanted, 'Both compressed and raw fixed indexes signed')
    return selected, dict(date=r['date'], valid_until=r.get('valid-until'))


# Exact 32-byte empty trailing XZ stream observed by root in the authenticated
# jammy index. No generic concatenation, alternative empty stream or padding.
EMPTY_XZ_TAIL = bytes.fromhex('fd377a585a000004e6d6b446000000001cdf44211fb6f37d010000000004595a')


def empty_xz_tail(raw):
    need(type(raw) is bytes, 'Exact decoder trailing bytes')
    if raw == b'':
        return True
    need(len(raw) == 32 and raw == EMPTY_XZ_TAIL, 'Only the exact reviewed empty trailing XZ stream')
    tail = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=64 * MIB)
    plain = tail.decompress(raw, max_length=1)
    need(plain == b'' and tail.eof and tail.unused_data == b'', 'Complete empty second XZ stream; no third stream')
    return True


def discover(raw_inputs, fetch, verify_detached, trust, now, guard):
    """Resolve six exact package addresses using at most twelve metadata requests.

    fetch(url, cap) -> {body: bytes, final_url: exact requested URL}.
    verify_detached(Release, Release.gpg, trust) -> documented crypto receipt.
    All callbacks are trusted injected dependencies, not evidence in themselves.
    """
    inputs, observed = bind_inputs(raw_inputs)
    need(isinstance(now, dt.datetime) and now.utcoffset() is not None, 'Explicit aware current time')
    need(type(trust) is dict and set(trust) == {'keyring_sha256', 'primary_fingerprints', 'authority_ref'}, 'Explicit trust anchor')
    digest(trust['keyring_sha256'])
    need(trust['authority_ref'].get('path') and digest(trust['authority_ref'].get('sha256')), 'Independently reviewed key authority')
    keys = trust['primary_fingerprints']
    need(type(keys) is list and 1 <= len(keys) <= 4 and len(set(keys)) == len(keys)
         and all(isinstance(k, str) and re.fullmatch('[0-9A-F]{40}|[0-9A-F]{64}', k) for k in keys), 'Finite full primary fingerprints')
    transferred = 0

    def get(url, cap):
        nonlocal transferred
        guard()
        cap = min(cap, 128 * MIB - transferred)
        need(cap > 0, 'Aggregate metadata transfer cap')
        response = fetch(url, cap)
        need(type(response['body']) is bytes and 0 < len(response['body']) <= cap
             and response['final_url'] == url, 'Bounded exact-URL response; redirects rejected')
        transferred += len(response['body'])
        guard()
        return response['body']

    found, provenance = {}, []
    for base, suite in SOURCES:
        release_url = base + 'dists/' + suite + '/Release'
        raw = get(release_url, 4 * MIB)
        signature = get(release_url + '.gpg', MIB)
        proof = verify_detached(raw, signature, trust)
        need(proof.get('verified') is True and proof.get('primary_fingerprint') in keys
             and re.fullmatch('[0-9A-F]{40}|[0-9A-F]{64}', proof.get('signing_fingerprint', ''))
             and proof.get('digest_algorithm') in ('SHA256', 'SHA384', 'SHA512')
             and proof.get('release_sha256') == sha(raw) and proof.get('signature_sha256') == sha(signature)
             and proof.get('keyring_sha256') == trust['keyring_sha256'], 'Bound genuine signature receipt')
        indexes, dates = release_indexes(raw, suite, now)
        chain = dict(release_url=release_url, release_sha256=sha(raw), signature_sha256=sha(signature),
                     signature_proof=proof, dates=dates, indexes=[])
        for component in ('main', 'universe'):
            name = component + '/binary-amd64/Packages'
            xz_ref, raw_ref = indexes[name + '.xz'], indexes[name]
            url = base + 'dists/' + suite + '/' + name + '.xz'
            compressed = get(url, xz_ref['bytes'])
            need(len(compressed) == xz_ref['bytes'] and sha(compressed) == xz_ref['sha256'], 'Signed compressed index bytes')
            decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=64 * MIB)
            plain = decoder.decompress(compressed, max_length=raw_ref['bytes'] + 1)
            need(decoder.eof and empty_xz_tail(decoder.unused_data) and len(plain) == raw_ref['bytes']
                 and sha(plain) == raw_ref['sha256'], 'Signed exact raw index; only the reviewed empty second stream')
            index_ref = dict(url=url, compressed=xz_ref, uncompressed=raw_ref)
            chain['indexes'].append(index_ref)
            seen = set()
            for ordinal, row in enumerate(stanzas(plain, 128 * MIB)):
                if ordinal % 128 == 0:
                    guard()
                package = row.get('package')
                if package not in PACKAGE_VERSIONS or row.get('version') != PACKAGE_VERSIONS[package] or row.get('architecture') != 'amd64':
                    continue
                need(package not in seen, 'Duplicate exact package stanza')
                seen.add(package)
                filename = relative(row['filename'])
                need(filename.startswith('pool/' + component + '/') and filename.endswith('.deb'), 'Package archive in signed component')
                candidate = dict(package=package, version=PACKAGE_VERSIONS[package], architecture='amd64',
                                 filename=filename, bytes=count(row['size'], 64 * MIB), sha256=digest(row['sha256']))
                source = dict(url=base + filename, index_url=url, release_url=release_url,
                              release_sha256=sha(raw), compressed_index_sha256=xz_ref['sha256'],
                              raw_index_sha256=raw_ref['sha256'])
                if package in found:
                    need(all(found[package][k] == v for k, v in candidate.items()), 'Conflicting exact packages across signed indexes')
                    found[package]['sources'].append(source)
                else:
                    found[package] = dict(candidate, sources=[source])
            del plain, compressed, decoder
        provenance.append(chain)
    rows = []
    for name in PACKAGE_VERSIONS:
        if name not in found:
            continue
        row = found[name]
        row['targets'] = [dict(observed_path=path, archive_member=member,
                               original_observed_reference=observed.get(path),
                               original_full_payload_hash_present=path in observed)
                          for path, member in TARGETS[name]]
        row['saved_package_names'] = next(p['saved_package_names'] for p in inputs['next']['packages'] if p['package'] == name)
        rows.append(row)
    missing = [dict(package=p, version=v, architecture='amd64') for p, v in PACKAGE_VERSIONS.items() if p not in found]
    need(sum(r['bytes'] for r in rows) <= 64 * MIB, 'Six archives total byte cap')
    return dict(schema='dory_ubuntu_exact_package_sources_v1', status='exact_addresses_ready' if not missing else 'exact_versions_missing',
                input_refs={k: dict(name=n, sha256=h) for k, (n, h) in INPUT_PINS.items()},
                prior_audit_pending_superseded_by_completed_exits=True, trust_anchor=trust,
                releases=provenance, packages=rows, missing=missing, metadata_transfer_bytes=transferred,
                archive_total_bytes=sum(r['bytes'] for r in rows), actual_archive_download=False,
                package_content_authenticated=False, unresolved_closed=0, native_ABI_qualified=False,
                new_speedup_claim=False)
