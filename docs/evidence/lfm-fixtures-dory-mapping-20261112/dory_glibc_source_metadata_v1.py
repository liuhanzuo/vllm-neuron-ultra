"""Exact historical Ubuntu metadata core; no filesystem, network or processes.

Replay the completed signed snapshot and the two complete saved Packages
indexes. Newly requested Sources indexes and DSC are metadata only. Source
archive references are returned, never fetched. No loader semantics are inferred.
"""
import copy
import hashlib
import json
import lzma
import re

import dory_ubuntu_package_sources_v4 as old

MIB = 2**20
Q = '/home/ubuntu/experiments/trn1-latest-models-20261031/research/'
SAVED = '/mnt/nvme/dory-serial-inventory-ubuntu-metadata-r5-20261111/signed-metadata/'
TARGET = dict(package='libc6', version='2.35-0ubuntu3.15', architecture='amd64',
              filename='pool/main/g/glibc/libc6_2.35-0ubuntu3.15_amd64.deb',
              bytes=3234380, sha256='79e35256227e16a607c154cdeb8d76ff12d20e31de286ea7fd9ad3b96fe0452d')
SOURCE = dict(package='glibc', version='2.35-0ubuntu3.15')
INPUTS = {
    'signed_result': dict(path=SAVED+'sources-result.json', bytes=25154,
        sha256='59ae44799afcb4504177f2627a78177c783b40c671749a909fde3cf7b8b06409'),
    'producer_exit': dict(path=Q+'dory-ubuntu-metadata-r5-actual-exit-20261111-v1.json', bytes=15187,
        sha256='46b910d3674f92c109b869a860d1701cd6e340ad36d130b4b48fd501e76c432e'),
    'recorder_exit': dict(path=Q+'dory-ubuntu-metadata-r5-recorder-actual-exit-20261111-v1.json', bytes=2649,
        sha256='16ae371eccb20c325b4bbaeea5670727e361e621bb02ff02f1bafd91f3c5a397'),
    'jammy-updates_release': dict(path=SAVED+'jammy-updates-Release', bytes=127167,
        sha256='8776e12b98b387ed0fd791bf2f9f733c39ace09b3435f2b470147f1e55c79ef9'),
    'jammy-security_release': dict(path=SAVED+'jammy-security-Release', bytes=128045,
        sha256='df8f969b9d421bd98407d98388c4fc6891d37e9f56effb833c51a3346421bb79'),
    'jammy-updates_packages': dict(path=SAVED+'jammy-updates-main-Packages.xz', bytes=3979464,
        sha256='8d6519d3da44040de05306a4607d63b541e8badededc193b08e45463bbd8952e'),
    'jammy-security_packages': dict(path=SAVED+'jammy-security-main-Packages.xz', bytes=3699964,
        sha256='fd67189ad0de0ceb0b71428061318c94cacc3243ee04d86b1d03ee2bd5e37d64'),
}
POCKETS = (('https://archive.ubuntu.com/ubuntu/', 'jammy-updates'),
           ('https://security.ubuntu.com/ubuntu/', 'jammy-security'))
# Derived from the authenticated saved Release text, not guessed addresses.
SOURCE_INDEXES = {
    'jammy-updates': dict(compressed=dict(bytes=611560,
        sha256='3d3031a77a1505f55f0ea61f3537c7de99d1d9b4a4a4517e5d1f44a8b8859bf8'),
        uncompressed=dict(bytes=2984362,
        sha256='de7360181ae9e72f54fd619be1e9b04c0dc51830e069a1232d5b683aeaa1357f')),
    'jammy-security': dict(compressed=dict(bytes=369460,
        sha256='127813d7a3d12123e6a3e7391f245907050d2889c5249392641473ddb2ccc70f'),
        uncompressed=dict(bytes=1914560,
        sha256='ddcfc76691c97513f28016f0eddc5a17d18b1e14a7516dbc0532587e4bab19d8')),
}
LIMITS = dict(saved_metadata_bytes=16*MIB, index_compressed_bytes=4*MIB,
              index_plain_bytes=32*MIB, dsc_bytes=MIB, transfer_bytes=4*MIB,
              max_requests=3, source_file_rows=8, result_bytes=4*MIB)
need, sha = old.need, old.sha


def request_table():
    return [dict(suite=suite, base=base,
        url=base+'dists/'+suite+'/main/source/by-hash/SHA256/'+SOURCE_INDEXES[suite]['compressed']['sha256'],
        release=copy.deepcopy(INPUTS[suite+'_release']),
        compressed=copy.deepcopy(SOURCE_INDEXES[suite]['compressed']),
        uncompressed=copy.deepcopy(SOURCE_INDEXES[suite]['uncompressed']),
        output_name=suite+'-main-Sources.xz') for base,suite in POCKETS]


def match(raw, ref, cap):
    need(type(raw) is bytes and type(ref['bytes']) is int
         and 0 < ref['bytes'] <= cap and len(raw) == ref['bytes']
         and sha(raw) == old.digest(ref['sha256']), 'Exact bounded metadata bytes')
    return raw


def decode_index(compressed, xz_ref, plain_ref):
    match(compressed, xz_ref, LIMITS['index_compressed_bytes'])
    need(type(plain_ref['bytes']) is int and 0 < plain_ref['bytes'] <= LIMITS['index_plain_bytes'],
         'Finite signed decompressed index bound')
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=64*MIB)
    plain = decoder.decompress(compressed, max_length=plain_ref['bytes']+1)
    need(decoder.eof and old.empty_xz_tail(decoder.unused_data), 'Complete reviewed XZ metadata stream')
    return match(plain, plain_ref, LIMITS['index_plain_bytes'])


def source_identity(binary):
    need(all(binary.get(k) == TARGET[k] for k in ('package', 'version', 'architecture', 'filename'))
         and binary.get('sha256') == TARGET['sha256']
         and old.count(binary.get('size'), 64*MIB) == TARGET['bytes'], 'Exact original binary stanza')
    value = binary.get('source')
    need(type(value) is str, 'Original complete binary Source field required')
    parsed = re.fullmatch(r'([a-z0-9][a-z0-9+.-]*)(?: \(([^()\s]+)\))?', value)
    need(parsed is not None, 'Exact Debian Source field syntax')
    identity = dict(package=parsed[1], version=parsed[2] or binary['version'])
    need(identity == SOURCE, 'Exact distro source package and patch version')
    return identity


def release_table(raw, suite):
    rows = list(old.stanzas(raw, 4*MIB, 4*MIB))
    need(len(rows) == 1, 'One pinned historical Release')
    row = rows[0]
    need(row.get('origin') == row.get('label') == 'Ubuntu' and row.get('suite') == suite
         and row.get('codename') == 'jammy' and row.get('acquire-by-hash') == 'yes'
         and 'amd64' in row.get('architectures', '').split()
         and 'main' in row.get('components', '').split(), 'Original pinned Ubuntu pocket/by-hash contract')
    wanted = {'main/'+kind+suffix for kind in ('source/Sources', 'binary-amd64/Packages')
              for suffix in ('', '.xz')}
    result = {}
    for line in row.get('sha256', '').splitlines():
        if not line.strip():
            continue
        fields = line.split(); need(len(fields) == 3, 'Release SHA256 triple')
        digest, size, path = fields; old.digest(digest); old.release_relative(path)
        if path in wanted:
            need(path not in result, 'Unique signed index path')
            cap = LIMITS['index_compressed_bytes'] if path.endswith('.xz') else LIMITS['index_plain_bytes']
            result[path] = dict(sha256=digest, bytes=old.count(size, cap))
    need(set(result) == wanted, 'Both complete Sources and Packages index identities signed')
    return result


def prepare(raw_inputs, guard):
    """Authenticate fixed saved metadata; no current history/physical inference."""
    need(set(raw_inputs) == set(INPUTS), 'Exactly seven fixed historical metadata files')
    need(sum(len(v) for v in raw_inputs.values()) <= LIMITS['saved_metadata_bytes'], 'Saved metadata total bound')
    for key, reference in INPUTS.items():
        guard(); match(raw_inputs[key], reference, LIMITS['saved_metadata_bytes'])
    saved, ended, recorder = (json.loads(raw_inputs[k]) for k in ('signed_result', 'producer_exit', 'recorder_exit'))
    need(ended['schema'] == 'dory_ubuntu_metadata_actual_exit_v1'
         and ended['actual_tool']['exit_code'] == 0 and ended['stage_success_observed'] is True
         and ended['complete_events_audit']['recomputed']['entire_discovery_reproduced'] is True
         and ended['complete_events_audit']['recomputed']['actual_signature_invocations_bound'] is True,
         'Original successful producer and independent complete metadata replay')
    need(recorder['schema'] == 'dory_ubuntu_metadata_recorder_actual_exit_v1'
         and recorder['actual_tool']['exit_code'] == 0 and recorder['actual_tool_exit_observed'] is True
         and recorder['owner_gone'] is True and recorder['result'] == {
             k: INPUTS['producer_exit'][k] for k in ('path', 'sha256')}, 'Original independently observed recorder exit')
    need(ended['evidence']['signed-metadata/sources-result.json'] == INPUTS['signed_result'],
         'Original successful evidence binds exact signed result')
    need(saved['schema'] == 'dory_ubuntu_exact_package_sources_v1' and saved['status'] == 'exact_addresses_ready',
         'Complete original signed metadata snapshot')
    original = [x for x in saved['packages'] if x['package'] == TARGET['package']]
    need(len(original) == 1 and all(original[0][k] == v for k, v in TARGET.items()), 'Same original libc6 archive identity')
    refs = {r['path']: r for r in saved['adapter']['metadata_files']}
    binary_evidence, requests = [], []
    for base, suite in POCKETS:
        guard(); release_ref = INPUTS[suite+'_release']; packages_ref = INPUTS[suite+'_packages']
        need(refs[release_ref['path']] == release_ref and refs[packages_ref['path']] == packages_ref,
             'Saved full Release and Packages copies bound by original result')
        for reference in (release_ref, packages_ref):
            name = 'signed-metadata/'+reference['path'].removeprefix(SAVED)
            need(ended['evidence'][name] == reference, 'Original completed independent audit binds saved bytes')
        url = base+'dists/'+suite+'/Release'
        release = [x for x in saved['releases'] if x['release_url'] == url]
        need(len(release) == 1 and release[0]['release_sha256'] == release_ref['sha256'], 'Exact signed Release chain')
        proof = release[0]['signature_proof']
        need(proof['verified'] is True and proof['process_returncode'] == 0
             and proof['release_sha256'] == release_ref['sha256']
             and proof['keyring_sha256'] == saved['trust_anchor']['keyring_sha256']
             and proof['primary_fingerprint'] in saved['trust_anchor']['primary_fingerprints'],
             'Saved successful signature receipt bound to reviewed trust')
        indexes = release_table(raw_inputs[suite+'_release'], suite)
        name = 'main/binary-amd64/Packages'
        need(indexes[name+'.xz'] == {k: packages_ref[k] for k in ('sha256','bytes')}, 'Same signed binary index')
        plain = decode_index(raw_inputs[suite+'_packages'], indexes[name+'.xz'], indexes[name])
        found = []
        for ordinal, row in enumerate(old.stanzas(plain, LIMITS['index_plain_bytes'])):
            if ordinal % 128 == 0: guard()
            if all(row.get(k) == TARGET[k] for k in ('package','version','architecture')):
                source_identity(row); found.append(dict(ordinal=ordinal, stanza=row))
        need(len(found) == 1, 'One complete exact binary stanza in each original index')
        binary_evidence.append(dict(suite=suite, binary_index=copy.deepcopy(packages_ref),
            raw_index=indexes[name], source_identity=SOURCE.copy(), **found[0]))
        name = 'main/source/Sources'
        need(dict(compressed=indexes[name+'.xz'],uncompressed=indexes[name]) == SOURCE_INDEXES[suite],
             'Exactly the reviewed source indexes from saved Release bytes')
        requests.append(dict(suite=suite, base=base,
            url=base+'dists/'+suite+'/main/source/by-hash/SHA256/'+indexes[name+'.xz']['sha256'],
            release=copy.deepcopy(release_ref), compressed=indexes[name+'.xz'], uncompressed=indexes[name],
            output_name=suite+'-main-Sources.xz'))
        del plain
    need(sum(x['compressed']['bytes'] for x in requests)+LIMITS['dsc_bytes'] <= LIMITS['transfer_bytes'],
         'Both known indexes plus maximum one DSC fit transfer cap')
    return dict(schema='dory_glibc_source_metadata_preparation_v1', source=SOURCE.copy(),
        binary=TARGET.copy(), inputs=copy.deepcopy(INPUTS), historical_signed_snapshot_only=True,
        new_signature_verification=False, latest_repository_snapshot_claimed=False,
        binary_Source_evidence=binary_evidence, requests=requests)


def checksum_rows(value):
    need(type(value) is str, 'Complete Checksums-Sha256 required')
    rows = []
    for line in value.splitlines():
        if not line.strip(): continue
        fields = line.split(); need(len(fields) == 3, 'Source checksum triple')
        digest, count, name = fields
        need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9+_.~-]*', name) and name not in ('.','..'),
             'Flat source member filename; no path or URL')
        rows.append(dict(name=name, sha256=old.digest(digest), bytes=old.count(count, 2**40)))
        need(len(rows) <= LIMITS['source_file_rows'], 'Finite complete source member table')
    need(rows and len({x['name'] for x in rows}) == len(rows), 'Nonempty unique source members')
    return sorted(rows, key=lambda x:x['name'])


def source_record(plain, request, guard):
    found = []
    for ordinal, row in enumerate(old.stanzas(plain, LIMITS['index_plain_bytes'])):
        if ordinal % 128 == 0: guard()
        if row.get('package') != SOURCE['package'] or row.get('version') != SOURCE['version']:
            continue
        need(row.get('directory') == 'pool/main/g/glibc', 'Exact original source directory')
        files = checksum_rows(row.get('checksums-sha256'))
        dsc = [x for x in files if x['name'].endswith('.dsc')]
        need(len(dsc) == 1 and dsc[0]['bytes'] <= LIMITS['dsc_bytes'], 'Exactly one bounded DSC metadata member')
        need(all(x is dsc[0] or re.search(r'\.(?:tar\.(?:xz|gz|bz2)|diff\.gz)$',x['name']) for x in files),
             'Only DSC and source archive references')
        found.append(dict(suite=request['suite'], index=request['url'], ordinal=ordinal,
                          stanza=row, files=files, dsc=dsc[0]))
    need(len(found) <= 1, 'No duplicate exact source stanza')
    return found[0] if found else None


def dsc_stanza(raw):
    need(type(raw) is bytes and 0 < len(raw) <= LIMITS['dsc_bytes']
         and b'\r' not in raw and b'\x00' not in raw, 'Bounded LF DSC metadata')
    text = raw.decode('utf-8')
    marker = '-----BEGIN PGP SIGNED MESSAGE-----\n'
    need(text.startswith(marker), 'Explicit clear-signed DSC envelope')
    headers, sep, rest = text[len(marker):].partition('\n\n')
    need(sep and re.fullmatch(r'Hash: (?:SHA256|SHA384|SHA512)(?:, ?(?:SHA256|SHA384|SHA512))*', headers),
         'Finite strong digest clear-sign header')
    body, sep, signature = rest.partition('\n-----BEGIN PGP SIGNATURE-----\n')
    need(sep and signature.endswith('-----END PGP SIGNATURE-----\n')
         and '-----BEGIN ' not in body and '-----BEGIN ' not in signature,
         'One complete DSC clear-sign envelope')
    lines = []
    for line in body.splitlines():
        if line.startswith('-'):
            need(line.startswith('- '), 'Correct clear-sign dash escaping'); line = line[2:]
        lines.append(line)
    rows = list(old.stanzas(('\n'.join(lines)+'\n').encode(), LIMITS['dsc_bytes']))
    need(len(rows) == 1, 'One authenticated DSC control stanza')
    return rows[0]


def collect(prepared, fetch, save, guard):
    """Injected metadata ports only. Caller must prepare from exact inputs first.

    No native authority, lock admission or cryptographic claim comes from a
    callback. The runtime wrapper separately binds these pure-source functions.
    """
    need(prepared['schema'] == 'dory_glibc_source_metadata_preparation_v1'
         and prepared['source'] == SOURCE and prepared['binary'] == TARGET
         and prepared['inputs'] == INPUTS, 'Exact authenticated preparation')
    need(prepared['requests'] == request_table(), 'Exactly two signed source indexes and by-hash URLs')
    transferred, requests, records, receipts, missing = 0, [], [], [], []
    def get(url, reference, name):
        nonlocal transferred
        guard(); need(url not in requests and len(requests) < LIMITS['max_requests'], 'No retries or extra requests')
        need(transferred+reference['bytes'] <= LIMITS['transfer_bytes'], 'Original finite total metadata transfer')
        response = fetch(url, reference['bytes'])
        need(type(response) is dict and set(response) == {'body','final_url'} and response['final_url'] == url,
             'Exact requested URL; no redirects')
        raw = match(response['body'], reference, LIMITS['index_compressed_bytes'])
        transferred += len(raw); requests.append(url); guard()
        receipt = save(name, raw)
        need(type(receipt) is dict and receipt.get('bytes') == len(raw) and receipt.get('sha256') == sha(raw),
             'Saved metadata receipt equals full authenticated bytes')
        receipts.append(receipt); return raw
    for request in prepared['requests']:
        compressed = get(request['url'], request['compressed'], request['output_name'])
        plain = decode_index(compressed, request['compressed'], request['uncompressed'])
        row = source_record(plain, request, guard)
        if row is None: missing.append(request['suite'])
        else: records.append(row)
        del compressed, plain
    source_files, dsc = [], None
    if records:
        expected = records[0]['files']
        need(all(r['files'] == expected for r in records), 'No conflicting exact source files across signed pockets')
        request = next(x for x in prepared['requests'] if x['suite'] == records[0]['suite'])
        directory = records[0]['stanza']['directory']; wanted = records[0]['dsc']
        raw = get(request['base']+directory+'/'+wanted['name'], wanted, 'glibc-exact.dsc')
        dsc = dsc_stanza(raw)
        need(dsc.get('source') == SOURCE['package'] and dsc.get('version') == SOURCE['version'],
             'DSC exact source patch version')
        source_files = [x for x in expected if x['name'] != wanted['name']]
        need(checksum_rows(dsc.get('checksums-sha256')) == source_files, 'Complete DSC/source-index file table equality')
    result = dict(schema='dory_glibc_exact_source_metadata_result_v1',
        status='exact_source_metadata_ready' if records else 'exact_source_version_missing',
        preparation=prepared, source_stanzas=records, source_index_missing_pockets=missing,
        dsc_parsed_stanza=dsc, source_archive_files=source_files, metadata_requests=requests,
        metadata_transfer_bytes=transferred, saved_metadata=receipts,
        dsc_hash_authenticated_by_signed_Sources=bool(records), DSC_signature_separately_verified=False,
        new_signature_verification=False, historical_signed_snapshot_only=True,
        actual_source_archive_download=False, source_archive_contents_authenticated=False,
        exact_loader_source_semantics_established=False, unresolved_loader_edges=15,
        native_ABI_qualified=False, full_model_qualified=False, new_speedup_claim=False)
    need(len(json.dumps(result,sort_keys=True).encode()) <= LIMITS['result_bytes'], 'Bounded complete metadata result')
    guard(); return result
