"""Synthetic ELF plus authenticated saved JSON/text only; no native file reads."""
import sys

FORBIDDEN = {'torch', 'numpy', 'ctypes', '_ctypes', 'transformers', 'vllm',
             'neuron', 'neuronxcc', 'neuronx_distributed',
             'neuronx_distributed_inference', 'nki'}


class RejectNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise AssertionError('No native import: ' + fullname)


sys.meta_path.insert(0, RejectNative())
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import unittest

Q = Path(__file__).parent
sys.path.insert(0, str(Q.parent))
from research import minicpm5_sdk_mapping_native_closure_v1 as old
from research import minicpm5_sdk_mapping_native_closure_v2 as core

BASE = Path('/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-bootstrap-r1-20261112')
SAVED = {
    'readelf': ('/mnt/nvme/dory-serial-inventory-r5-20261111/inventory/external-closure.json',
                '8e97c912e0cbcd75ee4eead45565a05d727c5bc043ef5bf0ea2eeb51002ffd6e'),
    'content': (str(BASE / 'candidate-content.json'),
                '38fdc4d462f5f0c038536306cc28e995d7c91eaed98cfb0309a8eccd46f89ab1'),
    'sdk': (str(BASE / 'sdk310/observation.json'),
            'cc013649c7a61f22f6dbab59c746be59a375e9d2507aadcd786168daab27dc4d'),
}


def saved_bytes(path, sha):
    path = Path(path)
    assert path.suffix in ('.json', '.maps'), path
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
            before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size,
                                   after.st_mtime_ns, after.st_ctime_ns)
    assert len(raw) <= 2**20 and hashlib.sha256(raw).hexdigest() == sha
    return raw


def saved_json(key):
    return json.loads(saved_bytes(*SAVED[key]))


def fixture(*, split=True, ffi=False):
    raw = bytearray(8192)
    raw[:16] = b'\x7fELF\x02\x01\x01' + bytes(9)
    phnum = 3 if split else 2
    struct.pack_into('<HHIQQQIHHHHHH', raw, 16, 3, 62, 1, 0, 64, 0, 0,
                     64, 56, phnum, 0, 0, 0)
    base, dynamic, string_at = 0x400000, 0x400, 0xff8
    strings = b'\0libalpha.so\0libsample.so\0'
    entries = [(1, 1), (14, 13), (5, base + string_at), (10, len(strings)), (0, 0)]
    struct.pack_into('<IIQQQQQQ', raw, 64, 1, 5, 0, base, 0,
                     0x1000 if split else len(raw), 0x1000 if split else len(raw), 4096)
    struct.pack_into('<IIQQQQQQ', raw, 120, 2, 4, dynamic, base + dynamic,
                     0, 16 * len(entries), 16 * len(entries), 8)
    if split:
        struct.pack_into('<IIQQQQQQ', raw, 176, 1, 4, 0x1000, base + 0x1000,
                         0, 0x1000, 0x1000, 4096)
    for i, entry in enumerate(entries):
        struct.pack_into('<qQ', raw, dynamic + 16 * i, *entry)
    raw[string_at:string_at + len(strings)] = strings
    if ffi:
        struct.pack_into('<Q', raw, 40, 0x1400)
        struct.pack_into('<HHH', raw, 58, 64, 4, 0)
        symbols = b'\0' + b''.join(name.encode() + b'\0' for name in sorted(core.FFI_FUNCTIONS))
        raw[0x1800:0x1800 + len(symbols)] = symbols
        struct.pack_into('<IIQQQQIIQQ', raw, 0x1440, 0, 11, 0, 0, 0x1600,
                         24 * (len(core.FFI_FUNCTIONS) + 1), 2, 0, 8, 24)
        struct.pack_into('<IIQQQQIIQQ', raw, 0x1480, 0, 3, 0, 0, 0x1800,
                         len(symbols), 0, 0, 1, 0)
        struct.pack_into('<IIQQQQIIQQ', raw, 0x14c0, 0, 1, 6, base + 0x200,
                         0x200, 0x100, 0, 0, 16, 0)
        for index, name in enumerate(sorted(core.FFI_FUNCTIONS), 1):
            struct.pack_into('<IBBHQQ', raw, 0x1600 + 24 * index,
                             symbols.index(name.encode() + b'\0'), 0x12, 0, 3,
                             base + 0x200 + 16 * index, 8)
    return raw


def saved_layout_fixture():
    """Reconstruct only ELF geometry from old readelf text, never native bytes.

    Dynamic strings and entries are synthetic. Their printed semantics and the
    complete LOAD geometry, STRTAB address/length and file size are preserved.
    """
    row = saved_json('readelf')['files'][0]
    assert row['reference']['sha256'] == '23bc148d2b48964b621c5561ff2b81c1d9c7b717967a7491efbf12f4304882b5'
    text = row['elf_command']['stdout']
    assert row['elf_command']['returncode'] == 0
    headers = []
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in ('LOAD', 'DYNAMIC', 'INTERP'):
            kind = {'LOAD': 1, 'DYNAMIC': 2, 'INTERP': 3}[fields[0]]
            values = [int(x, 16) for x in fields[1:6]]
            flags = ''.join(fields[6:-1])
            mask = (4 if 'R' in flags else 0) | (2 if 'W' in flags else 0) | (1 if 'E' in flags else 0)
            headers.append((kind, mask, *values, int(fields[-1], 16)))
    address = int(re.search(r'\(STRTAB\)\s+(0x[0-9a-f]+)', text)[1], 16)
    length = int(re.search(r'\(STRSZ\)\s+(\d+)', text)[1])
    assert (address, length) == (0x1ff588, 43612)
    raw = bytearray(row['reference']['bytes'])
    raw[:16] = b'\x7fELF\x02\x01\x01' + bytes(9)
    struct.pack_into('<HHIQQQIHHHHHH', raw, 16, 2, 62, 1, 0, 64, 0, 0,
                     64, 56, len(headers), 0, 0, 0)
    for i, header in enumerate(headers):
        struct.pack_into('<IIQQQQQQ', raw, 64 + 56 * i, *header)
    dynamic = next(h[2] for h in headers if h[0] == 2)
    interp = next(h for h in headers if h[0] == 3)
    raw[interp[2]:interp[2] + interp[5]] = row['elf']['interpreter'].encode() + b'\0'
    strings = bytearray(b'\0')
    entries = []
    for tag, value in [(15, row['elf']['search_paths'][0])] + [(1, name) for name in row['elf']['needed']]:
        entries.append((tag, len(strings)))
        strings.extend(value.encode() + b'\0')
    entries += [(30, 8), (0x6ffffffb, 1), (5, address), (10, length), (0, 0)]
    for i, entry in enumerate(entries):
        struct.pack_into('<qQ', raw, dynamic + 16 * i, *entry)
    # Both adjacent LOADs have the same virtual-to-file translation.
    offset = address - 0x1ff000
    raw[offset:offset + len(strings)] = strings
    return bytes(raw), row['elf']


def saved_sdk_closure():
    observation = saved_json('sdk')
    content = saved_json('content')['files']
    names = observation['map_summaries']['earliest']['selected_known_paths']
    metadata = {name: copy.deepcopy(content[name]['elf']) for name in names}
    identities = {name: {'device': [os.major(content[name]['identity']['st_dev']),
                                    os.minor(content[name]['identity']['st_dev'])],
                         'inode': content[name]['identity']['st_ino']} for name in names}
    ref = observation['maps']['earliest']
    maps = saved_bytes(ref['path'], ref['sha256']).decode('ascii')
    return metadata, identities, maps


class ELFSuccessorTests(unittest.TestCase):
    def test_original_single_segment_result_unchanged(self):
        raw = bytes(fixture(split=False))
        self.assertEqual(core.parse_elf64(raw), old.parse_elf64(raw))

    def test_saved_outer_geometry_crosses_two_contiguous_loads(self):
        raw, expected = saved_layout_fixture()
        with self.assertRaisesRegex(ValueError, 'STRTAB maps uniquely'):
            old.parse_elf64(raw)
        result = core.parse_elf64(raw)
        self.assertEqual(result['needed'], expected['needed'])
        self.assertEqual(result['search_tags'], [{'tag': 'RPATH', 'value': '$ORIGIN/../lib'}])
        self.assertEqual(result['interpreter'], expected['interpreter'])
        self.assertEqual(result['flags'], {'DT_FLAGS': 8, 'DT_FLAGS_1': 1})

    def test_adjacent_small_segments_preserve_dynamic_strings(self):
        self.assertEqual(core.parse_elf64(bytes(fixture())),
                         old.parse_elf64(bytes(fixture(split=False))))

    def test_virtual_gap_rejected(self):
        raw = fixture()
        struct.pack_into('<Q', raw, 176 + 16, 0x401001)
        with self.assertRaisesRegex(ValueError, 'uniquely'):
            core.parse_elf64(bytes(raw))

    def test_bss_cannot_supply_string_bytes(self):
        raw = fixture()
        struct.pack_into('<Q', raw, 64 + 32, 0xffa)
        with self.assertRaisesRegex(ValueError, 'uniquely'):
            core.parse_elf64(bytes(raw))

    def test_conflicting_and_identical_overlapping_loads_rejected(self):
        for first_offset in (0, 0x100):
            raw = fixture()
            struct.pack_into('<Q', raw, 64 + 8, first_offset)
            struct.pack_into('<Q', raw, 64 + 32, 0x1100)
            struct.pack_into('<Q', raw, 64 + 40, 0x1100)
            with self.assertRaisesRegex(ValueError, 'uniquely'):
                core.parse_elf64(bytes(raw))

    def test_discontinuous_file_offsets_rejected(self):
        raw = fixture()
        struct.pack_into('<Q', raw, 176 + 8, 0x1100)
        struct.pack_into('<Q', raw, 176 + 32, 0xf00)
        with self.assertRaisesRegex(ValueError, 'continuous file offsets'):
            core.parse_elf64(bytes(raw))

    def test_original_string_budget_remains_enforced(self):
        for length in (0, 2**20 + 1):
            raw = fixture()
            struct.pack_into('<Q', raw, 0x400 + 3 * 16 + 8, length)
            with self.assertRaisesRegex(ValueError, 'Bounded dynamic string table'):
                core.parse_elf64(bytes(raw))

    def test_positive_ffi_symbols_still_required_after_cross_segment_parse(self):
        proof = core.ffi_definitions(bytes(fixture(ffi=True)))
        self.assertEqual(set(proof['defined_functions']), core.FFI_FUNCTIONS)
        self.assertTrue(proof['positive_embedded_ffi_definitions'])
        with self.assertRaises(ValueError):
            core.ffi_definitions(bytes(fixture()))
        for offset, fmt, value in [(0x1600 + 24 + 6, '<H', 0),
                                   (0x1600 + 24 + 16, '<Q', 0),
                                   (0x1600 + 24 + 4, '<B', 0x11)]:
            raw = fixture(ffi=True)
            struct.pack_into(fmt, raw, offset, value)
            with self.assertRaises(ValueError):
                core.ffi_definitions(bytes(raw))

    def test_saved_libc_interp_is_not_another_kernel_interpreter_selection(self):
        metadata, identities, maps = saved_sdk_closure()
        executable, loader = '/usr/bin/python3.10', '/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2'
        self.assertEqual(metadata['/usr/lib/x86_64-linux-gnu/libc.so.6']['interpreter'],
                         '/lib64/ld-linux-x86-64.so.2')
        calls = []
        def resolve(path):
            calls.append(path)
            return loader
        with self.assertRaisesRegex(ValueError, 'Original-path actual ELF interpreter selection'):
            old.mapped_closure(metadata, identities, maps, executable, loader, resolve)
        calls.clear()
        result = core.mapped_closure(metadata, identities, maps, executable, loader, resolve)
        self.assertEqual(calls, ['/lib64/ld-linux-x86-64.so.2'])
        self.assertEqual(result['paths'], sorted(metadata))
        # This resolver is mocked. The saved-map replay is not fresh authority.

    def test_main_interpreter_wrong_alias_and_missing_main_interp_rejected(self):
        for case in ('path', 'alias', 'missing'):
            metadata, identities, maps = saved_sdk_closure()
            exe, loader = '/usr/bin/python3.10', '/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2'
            if case == 'path':
                metadata[exe]['interpreter'] = '/foreign/loader'
            if case == 'missing':
                metadata[exe]['interpreter'] = None
            with self.assertRaises(ValueError):
                core.mapped_closure(metadata, identities, maps, exe, loader,
                                    lambda path: '/wrong/loader' if case == 'alias' else loader)

    def test_map_identity_foreign_mapping_and_missing_dependency_still_rejected(self):
        for case in ('identity', 'foreign', 'dependency'):
            metadata, identities, maps = saved_sdk_closure()
            exe, loader = '/usr/bin/python3.10', '/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2'
            if case == 'identity':
                identities[exe]['inode'] += 1
            elif case == 'foreign':
                maps = maps.replace('/usr/bin/python3.10', '/foreign/python')
            else:
                metadata[exe]['needed'].append('unobserved-provider.so')
            with self.assertRaises(ValueError):
                core.mapped_closure(metadata, identities, maps, exe, loader, lambda path: loader)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(ELFSuccessorTests))
    assert not FORBIDDEN.intersection(sys.modules)
    print('Synthetic ELF and authenticated saved JSON/maps only; no binary/native/current-process access.')
    raise SystemExit(0 if result.wasSuccessful() else 1)
