"""Real bounded HTTPS/gpgv adapter; v2 retains the owning worker process group.

No import-time IO beyond Python imports. This module does not acquire a lock,
admit itself, install anything, fetch a key, or download a .deb.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import ssl
import stat
import subprocess
import time
import urllib.error
import urllib.request

import dory_ubuntu_package_sources_v4 as discovery

MIB = 2**20
Q = Path(__file__).resolve().parent
DISCOVERY_SHA = '69c8a9c9d34df8a126fc67ec7fe621a9db79bfb4a5ca3352a684dc746e46fc43'
KEYRING = '/usr/share/keyrings/ubuntu-archive-keyring.gpg'
GPGV = '/usr/bin/gpgv'
UBUNTU_PRIMARY_KEYS = frozenset(('F6ECB3762474EDA9D21B7022871920D1991BC93C',
                               '790BC7277767219C42C86F933B4FE6ACC0B21F32'))
need = discovery.need
sha = discovery.sha
URL_FILES = {}
for _base, _suite in discovery.SOURCES:
    for _name, _suffix in (('Release', 'Release'), ('Release.gpg', 'Release.gpg'),
                           ('main/binary-amd64/Packages.xz', 'main-Packages.xz'),
                           ('universe/binary-amd64/Packages.xz', 'universe-Packages.xz')):
        URL_FILES[_base + 'dists/' + _suite + '/' + _name] = _suite + '-' + _suffix


def identity(s):
    return tuple(getattr(s, k) for k in ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def stable_read(path, cap, expected_sha, guard, expected_size=None):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path, 'Canonical input path')
    guard()
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'Bounded regular input')
    if expected_size is not None:
        need(type(expected_size) is int and before.st_size == expected_size, 'Exact input size')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as f:
        need(identity(os.fstat(f.fileno())) == identity(before), 'Stable opened input')
        raw = bytearray()
        while len(raw) < before.st_size:
            guard()
            chunk = f.read(min(65536, before.st_size - len(raw)))
            need(chunk, 'Complete input')
            raw.extend(chunk)
        need(not f.read(1) and identity(os.fstat(f.fileno())) == identity(before)
             and identity(path.lstat()) == identity(before), 'Stable full input and exact EOF')
    need(sha(raw) == expected_sha, 'Input SHA256')
    return bytes(raw)


class Output:
    def __init__(self, root, guard):
        root = Path(root)
        need(root.is_absolute() and root.parent.resolve() == root.parent
             and root.is_relative_to('/mnt/nvme') and root != Path('/mnt/nvme'), 'Fresh canonical NVMe namespace')
        guard()
        root.mkdir(mode=0o700)
        parent_fd = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        self.root, self.guard, self.used, self.refs = root, guard, 4096, []

    def save(self, name, raw):
        need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) and type(raw) is bytes
             and len(raw) <= 64 * MIB, 'Bounded fixed output')
        allocated = ((len(raw) + 4095) // 4096) * 4096
        need(self.used + allocated <= 128 * MIB, 'Metadata allocated output cap')
        self.used += allocated
        path = self.root / name
        self.guard()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as f:
            for offset in range(0, len(raw), 65536):
                self.guard()
                f.write(raw[offset:offset + 65536])
            f.flush()
            os.fsync(f.fileno())
        directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        ref = dict(path=str(path), bytes=len(raw), sha256=sha(raw))
        if raw:
            stable_read(path, 64 * MIB, ref['sha256'], self.guard, ref['bytes'])
        self.refs.append(ref)
        return ref

    def json(self, name, value):
        return self.save(name, (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode())


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, 'Redirect forbidden', headers, fp)


class Transport:
    def __init__(self, guard, emit, output):
        self.guard, self.emit, self.output = guard, emit, output
        self.seen, self.transferred = set(), 0
        # No proxy environment, implicit .netrc identity, or permissive TLS context.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
                                                  urllib.request.HTTPSHandler(context=ssl.create_default_context()))

    def __call__(self, url, cap):
        need(url in URL_FILES and url not in self.seen and len(self.seen) < 12, 'Fixed twelve single-attempt URLs')
        maximum = MIB if url.endswith('.gpg') else 4 * MIB if url.endswith('/Release') else 32 * MIB
        need(type(cap) is int and 0 < cap <= maximum, 'Fixed response cap')
        cap = min(cap, 128 * MIB - self.transferred)
        need(cap > 0, 'Aggregate metadata transfer cap')
        self.seen.add(url)
        self.guard()
        started = time.monotonic()
        request = urllib.request.Request(url, headers={'User-Agent': 'trn1-ubuntu-provenance/1.0', 'Accept-Encoding': 'identity'})
        with self.opener.open(request, timeout=5) as response:
            need(response.status == 200 and response.geturl() == url, 'Exact HTTPS response status/URL')
            need(response.headers.get('Content-Encoding', 'identity').lower() == 'identity', 'Unencoded HTTP body')
            lengths = response.headers.get_all('Content-Length', [])
            need(len(lengths) <= 1, 'Unique Content-Length')
            length = discovery.count(lengths[0], cap) if lengths else None
            chunks, total = [], 0
            while True:
                self.guard()
                need(time.monotonic() - started < 60, 'Metadata request wall deadline')
                # One underlying buffered/socket read per loop; repeated short
                # reads cannot hide a slow response from the wall deadline.
                chunk = response.read1(min(65536, cap - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                self.transferred += len(chunk)
                need(total <= cap and self.transferred <= 128 * MIB, 'Streaming response byte cap')
                chunks.append(chunk)
            need(total > 0 and (length is None or total == length), 'Complete HTTP Content-Length')
            raw = b''.join(chunks)
        self.guard()
        ref = self.output.save(URL_FILES[url], raw)
        self.emit(dict(event='ubuntu_metadata_download', url=url, saved=ref, total_transfer_bytes=self.transferred))
        return dict(body=raw, final_url=url)


from dory_ubuntu_gpgv_status_compliance_v1 import parse_status


def invoke_gpgv(argv, directory, guard):
    """No shell, no default keyring, bounded pipe reads; reap this owned process."""
    guard()
    started = time.monotonic()
    output = {'stdout': bytearray(), 'stderr': bytearray()}
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               cwd=directory, env={'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C'},
                               close_fds=True, start_new_session=False)
    try:
        with selectors.DefaultSelector() as selector:
            for name, stream in (('stdout', process.stdout), ('stderr', process.stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while selector.get_map() or process.poll() is None:
                guard()
                need(time.monotonic() - started < 30, 'gpgv wall deadline')
                for key, _ in selector.select(timeout=0.05):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    out = output[key.data]
                    need(len(out) + len(chunk) <= MIB, 'gpgv output cap')
                    out.extend(chunk)
            code = process.wait(timeout=1)
        guard()
        return code, bytes(output['stdout']), bytes(output['stderr'])
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


class Verifier:
    def __init__(self, guard, emit, output, trust, gpgv_ref):
        self.guard, self.emit, self.output, self.trust, self.gpgv_ref = guard, emit, output, trust, gpgv_ref
        self.calls = 0

    def __call__(self, release, signature, trust):
        need(trust == self.trust and self.calls < 3, 'Fixed trust and three signatures')
        self.calls += 1
        key = stable_read(self.output.root / 'ubuntu-archive-keyring.gpg', MIB, trust['keyring_sha256'], self.guard)
        del key
        stable_read(GPGV, 16 * MIB, self.gpgv_ref['sha256'], self.guard, self.gpgv_ref['bytes'])
        # These exact bytes were just fetched; new fixed per-call files bind gpgv inputs.
        release_ref = self.output.save('verify-' + str(self.calls) + '.Release', release)
        sig_ref = self.output.save('verify-' + str(self.calls) + '.gpg', signature)
        argv = [GPGV, '--homedir', str(self.output.root), '--keyring', str(self.output.root / 'ubuntu-archive-keyring.gpg'),
                '--status-fd', '1', '--', sig_ref['path'], release_ref['path']]
        self.emit(dict(event='ubuntu_gpgv_start', invocation=self.calls, argv=argv))
        code, stdout, stderr = invoke_gpgv(argv, str(self.output.root), self.guard)
        status_ref = self.output.save('verify-' + str(self.calls) + '-status.txt', stdout)
        stderr_ref = self.output.save('verify-' + str(self.calls) + '-stderr.txt', stderr)
        self.emit(dict(event='ubuntu_gpgv_exit', invocation=self.calls, returncode=code,
                       status=status_ref, stderr=stderr_ref))
        # Re-read the exact files and verifier after process exit, before claiming a chain.
        for ref in (release_ref, sig_ref, self.gpgv_ref):
            stable_read(ref['path'], 16 * MIB, ref['sha256'], self.guard, ref['bytes'])
        stable_read(self.output.root / 'ubuntu-archive-keyring.gpg', MIB, trust['keyring_sha256'], self.guard)
        verified = parse_status(stdout, code, trust, dt.datetime.now(dt.timezone.utc))
        return dict(verified=True, **verified[0], verified_signatures=verified,
                    release_sha256=sha(release), signature_sha256=sha(signature), keyring_sha256=trust['keyring_sha256'],
                    process_returncode=code, status_reference=status_ref, stderr_reference=stderr_ref)


def run(guard, emit, trust, outputroot):
    """Caller must already hold original serial descriptor and resource contract.

    outputroot is a fresh dedicated directory beneath /mnt/nvme, not an existing
    wrapper directory. Trust authority must match the documented fixed schema.
    """
    guard()
    stable_read(Q / 'dory_ubuntu_package_sources_v4.py', MIB, DISCOVERY_SHA, guard)
    need(set(trust) == {'keyring_sha256', 'primary_fingerprints', 'authority_ref'}, 'Explicit trust fields')
    need(type(trust['primary_fingerprints']) is list and 1 <= len(trust['primary_fingerprints']) <= 2
         and len(set(trust['primary_fingerprints'])) == len(trust['primary_fingerprints'])
         and set(trust['primary_fingerprints']) <= UBUNTU_PRIMARY_KEYS,
         'Fixed officially published Ubuntu 2012/2018 primary fingerprints')
    authority_raw = stable_read(trust['authority_ref']['path'], MIB, trust['authority_ref']['sha256'], guard)
    authority = json.loads(authority_raw)
    need(authority.get('schema') == 'dory_ubuntu_archive_key_authority_v1'
         and authority.get('independent_authority_verified') is True
         and authority['keyring']['path'] == KEYRING
         and authority['keyring']['sha256'] == trust['keyring_sha256']
         and authority['primary_fingerprints'] == trust['primary_fingerprints']
         and authority['gpgv']['path'] == GPGV and 1 <= len(authority['provenance']) <= 8,
         'Exact independently reviewed key/verifier authority')
    for ref in authority['provenance']:
        stable_read(ref['path'], 4 * MIB, ref['sha256'], guard, ref.get('bytes'))
    keyring = stable_read(KEYRING, MIB, trust['keyring_sha256'], guard, authority['keyring']['bytes'])
    stable_read(GPGV, 16 * MIB, authority['gpgv']['sha256'], guard, authority['gpgv']['bytes'])
    inputs = {key: stable_read(Q / name, MIB, pin, guard) for key, (name, pin) in discovery.INPUT_PINS.items()}
    discovery.bind_inputs(inputs)
    output = Output(outputroot, guard)
    output.save('ubuntu-archive-keyring.gpg', keyring)
    output.save('key-authority.json', authority_raw)
    transport = Transport(guard, emit, output)
    verifier = Verifier(guard, emit, output, trust, authority['gpgv'])
    result = discovery.discover(inputs, transport, verifier, trust, dt.datetime.now(dt.timezone.utc), guard)
    result['adapter'] = dict(schema='dory_ubuntu_metadata_adapter_receipt_v1', metadata_files=output.refs.copy(),
                             gpgv_reference=authority['gpgv'], requests=len(transport.seen), signature_invocations=verifier.calls,
                             adapter_output_allocated_bound=output.used, actual_cryptographic_verifier='gpgv',
                             runtime_admission_checked_here=False, caller_preconditions_required=True, process_exit_claimed=False)
    result_ref = output.json('sources-result.json', result)
    guard()
    emit(dict(event='ubuntu_metadata_discovery_complete', result=result_ref, status=result['status'],
              packages=len(result['packages']), missing=result['missing'], package_content_authenticated=False))
    return dict(result=result_ref, status=result['status'], packages=len(result['packages']),
                output_allocated_bound=output.used, package_content_authenticated=False,
                unresolved_closed=0, native_ABI_qualified=False, new_speedup_claim=False)


if __name__ == '__main__':
    raise RuntimeError('Root-admitted API only; no standalone execution')
