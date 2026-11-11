"""No real network, subprocess, keyring, proc, lock or output directory IO."""
import contextlib
import datetime as dt
import email.message
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.request

import dory_ubuntu_metadata_adapter_v6 as m

PRIMARY = 'F6ECB3762474EDA9D21B7022871920D1991BC93C'
SIGNING = 'A' * 40
NOW = dt.datetime(2026, 11, 11, 12, tzinfo=dt.timezone.utc)
TRUST = dict(primary_fingerprints=[PRIMARY], keyring_sha256='b' * 64,
             authority_ref=dict(path='/mock/authority.json', sha256='c' * 64))


def good_status(signing=SIGNING, primary=PRIMARY):
    timestamp = int(NOW.timestamp()) - 3600
    return (f'[GNUPG:] NEWSIG\n[GNUPG:] KEY_CONSIDERED {primary} 0\n'
            f'[GNUPG:] SIG_ID abcdef012345 2026-11-11 {timestamp}\n'
            f'[GNUPG:] GOODSIG {signing[-16:]} Ubuntu Archive Key\n'
            f'[GNUPG:] VALIDSIG {signing} 2026-11-11 {timestamp} 0 4 0 1 10 00 {primary}\n').encode()


class MemoryOutput:
    def __init__(self):
        self.root, self.files, self.refs, self.used = Path('/mnt/nvme/mock-source'), {}, [], 4096

    def save(self, name, raw):
        self.files[name] = raw
        ref = dict(path=str(self.root / name), bytes=len(raw), sha256=m.sha(raw))
        self.refs.append(ref)
        self.used += len(raw)
        return ref


class Response(io.BytesIO):
    def __init__(self, raw, url, status=200, headers=None):
        super().__init__(raw)
        self.url, self.status = url, status
        self.headers = email.message.Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value

    def geturl(self):
        return self.url


def transport(response):
    # Patch construction too: do not read the host TLS CA files in pure mocks.
    output, guard, emit = MemoryOutput(), Mock(), Mock()
    opener = Mock()
    opener.open.return_value = response
    with patch.object(m.ssl, 'create_default_context', return_value=object()), patch.object(m.urllib.request, 'build_opener', return_value=opener):
        obj = m.Transport(guard, emit, output)
    return obj, output, opener


class Tests(unittest.TestCase):
    def test_complete_status_with_primary_and_subkey_binding(self):
        for signing in (SIGNING, PRIMARY):
            proof = m.parse_status(good_status(signing=signing), 0, TRUST, NOW)
            self.assertEqual(proof[0]['primary_fingerprint'], PRIMARY)
            self.assertEqual(proof[0]['signing_fingerprint'], signing)
            self.assertEqual(proof[0]['digest_algorithm'], 'SHA512')

    def test_invalid_status_and_process_result_rejected(self):
        for tag in ('BADSIG', 'ERRSIG', 'NO_PUBKEY', 'EXPKEYSIG', 'EXPSIG', 'REVKEYSIG', 'NODATA', 'FAILURE', 'ERROR', 'UNREVIEWED'):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                m.parse_status(good_status() + f'[GNUPG:] {tag} bad\n'.encode(), 0, TRUST, NOW)
        for raw, code in ((good_status(), 1), (b'gpgv: Good signature\n', 0), (good_status()[:-1], 0),
                          (good_status().replace(b'[GNUPG:] NEWSIG\n', b''), 0),
                          (good_status().replace(b' 0 4 0 1 10 00 ', b' 0 4 0 1 2 00 '), 0),
                          (good_status(primary='F' * 40), 0),
                          (good_status().replace(b'GOODSIG ' + SIGNING[-16:].encode(), b'GOODSIG ' + b'0' * 16), 0)):
            with self.subTest(code=code, raw=raw[:25]), self.assertRaises(ValueError):
                m.parse_status(raw, code, TRUST, NOW)

    def test_duplicate_or_incomplete_signatures_rejected(self):
        for raw in (good_status() + good_status(), good_status() + b'[GNUPG:] NEWSIG\n',
                    good_status().replace(b'[GNUPG:] GOODSIG', b'[GNUPG:] NEWSIG\n[GNUPG:] GOODSIG')):
            with self.assertRaises(ValueError):
                m.parse_status(raw, 0, TRUST, NOW)

    def test_fixed_https_url_body_bounds_and_single_request(self):
        url = next(iter(m.URL_FILES))
        t, output, opener = transport(Response(b'abc', url, headers={'Content-Length': '3'}))
        result = t(url, 10)
        self.assertEqual(result, dict(body=b'abc', final_url=url))
        self.assertEqual(t.transferred, 3)
        self.assertEqual(list(output.files.values()), [b'abc'])
        self.assertEqual(opener.open.call_args.kwargs['timeout'], 5)
        with self.assertRaises(ValueError):
            t(url, 10)
        with self.assertRaises(ValueError):
            t('https://evil.example/Release', 10)

    def test_https_rejects_status_redirect_length_encoding_and_overflow(self):
        url = next(iter(m.URL_FILES))
        for response in (Response(b'abc', url, status=302), Response(b'abc', 'https://evil.example/'),
                         Response(b'abc', url, headers={'Content-Length': '2'}),
                         Response(b'abc', url, headers={'Content-Length': '99'}),
                         Response(b'abc', url, headers={'Content-Encoding': 'gzip'}), Response(b'abcd', url)):
            t, _, _ = transport(response)
            with self.assertRaises(ValueError):
                t(url, 3)
        handler = m.NoRedirect()
        with self.assertRaises(m.urllib.error.HTTPError):
            handler.redirect_request(urllib.request.Request(url), None, 302, 'Found', {}, 'https://evil.example/')

    def test_https_aggregate_and_wall_time_caps(self):
        url = next(iter(m.URL_FILES))
        t, _, _ = transport(Response(b'abc', url))
        t.transferred = 128 * m.MIB - 2
        with self.assertRaisesRegex(ValueError, 'byte cap'):
            t(url, 3)
        t, _, _ = transport(Response(b'abc', url))
        with patch.object(m.time, 'monotonic', side_effect=[0, 61]), self.assertRaisesRegex(ValueError, 'wall deadline'):
            t(url, 3)

    def test_real_verifier_fixed_argv_and_receipt_from_actual_status(self):
        output = MemoryOutput()
        ref = dict(path=m.GPGV, bytes=123, sha256='d' * 64)
        verifier = m.Verifier(Mock(), Mock(), output, TRUST, ref)
        with patch.object(m, 'stable_read', return_value=b'mock-only'), patch.object(m, 'invoke_gpgv', return_value=(0, good_status(), b'human diagnostic')) as invoke:
            receipt = verifier(b'Release', b'Signature', TRUST)
        argv, cwd, _ = invoke.call_args.args
        self.assertEqual(argv, [m.GPGV, '--homedir', str(output.root), '--keyring', str(output.root / 'ubuntu-archive-keyring.gpg'),
                                '--status-fd', '1', '--', str(output.root / 'verify-1.gpg'), str(output.root / 'verify-1.Release')])
        self.assertEqual(receipt['release_sha256'], m.sha(b'Release'))
        self.assertEqual(receipt['primary_fingerprint'], PRIMARY)
        self.assertEqual(receipt['process_returncode'], 0)
        self.assertIn('verify-1-status.txt', output.files)
        with patch.object(m, 'stable_read', return_value=b'mock-only'), patch.object(m, 'invoke_gpgv', return_value=(1, good_status(), b'bad')):
            with self.assertRaises(ValueError):
                verifier(b'Release', b'Signature', TRUST)

    def test_output_exclusive_creation_directory_fsync_and_cap(self):
        output = object.__new__(m.Output)
        output.root, output.guard, output.used, output.refs = Path('/mnt/nvme/mock'), Mock(), 4096, []
        buf = io.BytesIO()
        with patch.object(m.os, 'open', side_effect=[101, 102]) as opened, patch.object(m.os, 'fdopen', return_value=contextlib.nullcontext(buf)), \
             patch.object(m.os, 'fsync') as synced, patch.object(m.os, 'close') as closed, patch.object(m, 'stable_read', return_value=b'abc'):
            # BytesIO fileno is unsupported; supply a simple mock file wrapper.
            wrapped = Mock()
            wrapped.__enter__ = Mock(return_value=wrapped)
            wrapped.__exit__ = Mock(return_value=False)
            wrapped.fileno.return_value = 101
            with patch.object(m.os, 'fdopen', return_value=wrapped):
                output.save('fixed.txt', b'abc')
        self.assertTrue(opened.call_args_list[0].args[1] & m.os.O_EXCL)
        self.assertEqual([x.args[0] for x in synced.call_args_list], [101, 102])
        closed.assert_called_with(102)
        output.used = 128 * m.MIB
        with self.assertRaises(ValueError):
            output.save('full.txt', b'abc')

    def test_gpgv_stream_success_and_cleanup_without_real_process(self):
        process = Mock()
        process.stdout, process.stderr = Mock(), Mock()
        process.stdout.fileno.return_value, process.stderr.fileno.return_value = 21, 22
        process.poll.return_value = 0
        process.wait.return_value = 0
        selector = Mock()
        selector.__enter__ = Mock(return_value=selector)
        selector.__exit__ = Mock(return_value=False)
        selector.get_map.side_effect = [True, True, False]
        keys = [SimpleNamespace(fileobj=process.stdout, data='stdout'), SimpleNamespace(fileobj=process.stderr, data='stderr')]
        selector.select.side_effect = [[(keys[0], 1), (keys[1], 1)], [(keys[0], 1), (keys[1], 1)]]
        with patch.object(m.subprocess, 'Popen', return_value=process) as popen, patch.object(m.selectors, 'DefaultSelector', return_value=selector), \
             patch.object(m.os, 'set_blocking'), patch.object(m.os, 'read', side_effect=[good_status(), b'diagnostic', b'', b'']):
            code, out, err = m.invoke_gpgv([m.GPGV, 'mock-argv'], '/mnt/nvme/mock', Mock())
        self.assertEqual((code, out, err), (0, good_status(), b'diagnostic'))
        self.assertNotIn('shell', popen.call_args.kwargs)
        self.assertIs(popen.call_args.kwargs['start_new_session'], False)
        self.assertEqual(popen.call_args.kwargs['env'], {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C'})
        process.kill.assert_not_called()
        process.stdout.close.assert_called_once()

    def test_gpgv_deadline_kills_and_reaps_owned_process(self):
        process = Mock()
        process.stdout, process.stderr = Mock(), Mock()
        process.poll.return_value = None
        selector = Mock()
        selector.__enter__ = Mock(return_value=selector)
        selector.__exit__ = Mock(return_value=False)
        selector.get_map.return_value = True
        with patch.object(m.subprocess, 'Popen', return_value=process), patch.object(m.selectors, 'DefaultSelector', return_value=selector), \
             patch.object(m.os, 'set_blocking'), patch.object(m.time, 'monotonic', side_effect=[0, 31]):
            with self.assertRaisesRegex(ValueError, 'wall deadline'):
                m.invoke_gpgv([m.GPGV, 'mock-argv'], '/mnt/nvme/mock', Mock())
        process.kill.assert_called_once()
        process.wait.assert_called_with(timeout=5)
        process.stdout.close.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
