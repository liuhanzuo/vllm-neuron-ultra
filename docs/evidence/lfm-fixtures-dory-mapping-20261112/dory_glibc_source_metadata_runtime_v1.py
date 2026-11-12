"""Finite glibc metadata IO ports, called only inside the original serial lease.

No import-time IO, source archive downloads, signature subprocesses or native
operations. Actual external process exit remains an independent root fact.
"""
import copy
import json
import os
from pathlib import Path
import ssl
import time
import urllib.request

import dory_glibc_source_metadata_v1 as core
import collect_dory_glibc_source_metadata_v1 as collector
from dory_ubuntu_metadata_adapter_v6 import stable_read, Output as OriginalOutput, NoRedirect

SOCKET_TIMEOUT_SECONDS = 5
REQUEST_WALL_SECONDS = 60
need, sha = core.need, core.sha
OUTPUT_CAPS = {**{row['output_name']: row['compressed']['bytes'] for row in core.request_table()},
               'glibc-exact.dsc': core.LIMITS['dsc_bytes'], 'source-result.json': core.LIMITS['result_bytes']}


def preparation_digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def read_saved(reference, cap, guard):
    need(type(reference) is dict and reference in core.INPUTS.values()
         and cap == reference['bytes'], 'Exactly one of seven fixed saved metadata inputs')
    return stable_read(reference['path'], cap, reference['sha256'], guard, reference['bytes'])


class Output(OriginalOutput):
    """Original bounded durable writer plus exclusive atomic final publication."""
    def save(self, name, raw):
        need(name in OUTPUT_CAPS and type(raw) is bytes and 0 < len(raw) <= OUTPUT_CAPS[name],
             'Only four fixed bounded metadata output names')
        need(all(Path(row['path']).name != name for row in self.refs), 'Metadata output cannot be rewritten')
        temporary = 'publishing-' + name
        saved = super().save(temporary, raw)
        target = self.root / name
        self.guard()
        os.link(saved['path'], target, follow_symlinks=False)
        os.unlink(saved['path'])
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        reference = dict(saved, path=str(target))
        stable_read(target, OUTPUT_CAPS[name], reference['sha256'], self.guard, reference['bytes'])
        self.refs[-1] = reference
        return reference


class Transport:
    """Two ordered by-hash indexes; optional DSC derived from their exact bytes."""
    def __init__(self, guard, emit):
        self.guard, self.emit = guard, emit
        self.requests = core.request_table()
        self.seen, self.rows, self.receipts = [], [], []
        self.transferred = 0
        self.failed = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()))

    def expected(self, url, cap):
        index = len(self.seen)
        need(index < core.LIMITS['max_requests'] and url not in self.seen, 'At most three single-attempt metadata requests')
        if index < 2:
            request = self.requests[index]
            need(url == request['url'], 'Exact ordered by-hash source index URL')
            reference = request['compressed']
        else:
            records = [(request, row) for request, row in zip(self.requests, self.rows) if row is not None]
            need(len(self.rows) == 2 and records, 'Both exact source indexes authenticated before a DSC')
            request, selected = records[0]
            need(all(row['files'] == selected['files'] for _, row in records), 'Consistent complete source file tables')
            reference = selected['dsc']
            wanted = request['base'] + selected['stanza']['directory'] + '/' + reference['name']
            need(reference['name'].endswith('.dsc') and url == wanted, 'Only the single DSC selected by authenticated Sources')
        need(type(cap) is int and cap == reference['bytes']
             and self.transferred + cap <= core.LIMITS['transfer_bytes'], 'Exact per-file and total transfer bounds')
        return reference

    def __call__(self, url, cap):
        if self.failed is not None: raise self.failed
        try:
            self.guard()
            reference = self.expected(url, cap)
            self.seen.append(url)
            self.emit(dict(event='glibc_metadata_request', ordinal=len(self.seen), url=url,
                           expected_bytes=cap, expected_sha256=reference['sha256']))
            started = time.monotonic()
            def bounded():
                self.guard()
                need(time.monotonic() - started < REQUEST_WALL_SECONDS, 'Metadata request wall deadline')
            request = urllib.request.Request(url, headers={'User-Agent':'trn1-glibc-metadata/1.0',
                'Accept-Encoding':'identity', 'Connection':'close'})
            bounded()
            with self.opener.open(request, timeout=SOCKET_TIMEOUT_SECONDS) as response:
                bounded()
                need(response.status == 200 and response.geturl() == url, 'Exact HTTPS status and URL; no redirects')
                lengths = response.headers.get_all('Content-Length', [])
                need(len(lengths) == 1 and core.old.count(lengths[0], cap) == cap,
                     'Unique exact HTTP Content-Length required')
                encodings = response.headers.get_all('Content-Encoding', [])
                need(len(encodings) <= 1 and (not encodings or encodings[0].lower() == 'identity')
                     and not response.headers.get_all('Transfer-Encoding', [])
                     and not response.headers.get_all('Content-Range', []), 'Only complete identity HTTP bodies')
                chunks, total = [], 0
                while True:
                    bounded()
                    chunk = response.read1(min(65536, cap - total + 1))
                    bounded()
                    need(type(chunk) is bytes, 'Byte HTTP body')
                    if not chunk: break
                    total += len(chunk); self.transferred += len(chunk)
                    need(total <= cap and self.transferred <= core.LIMITS['transfer_bytes'], 'Streaming byte bounds')
                    chunks.append(chunk)
                need(total == cap, 'Complete exact HTTP body and EOF')
                body = core.match(b''.join(chunks), reference, core.LIMITS['index_compressed_bytes'])
            bounded()
            if len(self.seen) <= 2:
                request = self.requests[len(self.seen)-1]
                plain = core.decode_index(body, request['compressed'], request['uncompressed'])
                self.rows.append(core.source_record(plain, request, self.guard))
            receipt = dict(url=url, bytes=cap, sha256=sha(body), status=200,
                           content_length=cap, complete_body_EOF_observed=True,
                           redirected=False, retried=False)
            self.receipts.append(receipt)
            self.emit(dict(event='glibc_metadata_response', ordinal=len(self.receipts), receipt=receipt,
                           total_transfer_bytes=self.transferred))
            return dict(body=body, final_url=url)
        except BaseException as error:
            self.failed = error
            raise


def run(guard, emit, outputroot):
    """The admitted original worker supplies the retained-descriptor guard."""
    guard()
    output = Output(outputroot, guard)
    transport = Transport(guard, emit)
    result = collector.run(lambda reference, cap: read_saved(reference, cap, guard),
                           transport, output.save, guard)
    need(result['metadata_requests'] == transport.seen
         and result['metadata_transfer_bytes'] == transport.transferred
         and len(transport.receipts) == len(transport.seen), 'Complete actual transport receipts')
    result['runtime_transport'] = dict(requests=copy.deepcopy(transport.receipts),
        socket_timeout_seconds=SOCKET_TIMEOUT_SECONDS, request_wall_check_seconds=REQUEST_WALL_SECONDS,
        blocking_DNS_and_IO_also_bounded_by_original_parent_lifecycle=True,
        automatic_retry=False, mirror_fallback=False, redirects_followed=False)
    guard()
    reference = output.save('source-result.json', (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode())
    emit(dict(event='glibc_metadata_result', result=reference, status=result['status'],
              metadata_requests=len(transport.seen), actual_source_archive_download=False))
    guard()
    return dict(result=reference, status=result['status'], metadata_requests=len(transport.seen),
                metadata_transfer_bytes=transport.transferred)
