"""Exact deb822 line-cap boundaries; synthetic in-memory metadata only."""
import ast
from pathlib import Path
import unittest
import dory_ubuntu_package_sources_v4 as candidate


def line(length,name=b'Provides'):
    prefix=name+b': '
    assert length>=len(prefix)
    return prefix+b'x'*(length-len(prefix))+b'\n'


class Tests(unittest.TestCase):
    def test_complete_source_diff_is_only_one_line_limit_expression(self):
        q=Path(__file__).resolve().parent
        old=(q/'dory_ubuntu_package_sources_v3.py').read_text()
        expected=old.replace("need(len(line) <= 65536, 'Metadata line cap')",
            "need(len(line) <= min(128 * 1024, stanza_cap), 'Metadata line cap')")
        self.assertNotEqual(expected,old)
        self.assertEqual(ast.dump(ast.parse(expected)),ast.dump(ast.parse((q/'dory_ubuntu_package_sources_v4.py').read_bytes())))

    def test_requested_long_provides_boundaries_and_one_over(self):
        for length in (65536,65537,70840,131072):
            raw=line(length)
            with self.subTest(length=length):
                rows=list(candidate.stanzas(raw,len(raw)))
                self.assertEqual(rows,[{'provides':'x'*(length-len(b'Provides: '))}])
        raw=line(131073)
        with self.assertRaisesRegex(ValueError,'Metadata line cap'):list(candidate.stanzas(raw,len(raw)))

    def test_lower_stanza_cap_is_also_a_stricter_per_line_cap(self):
        for cap in (16,65536,70840):
            with self.subTest(cap=cap):
                raw=line(cap);self.assertEqual(len(list(candidate.stanzas(raw,len(raw),cap))),1)
                raw=line(cap+1)
                with self.assertRaisesRegex(ValueError,'Metadata line cap'):list(candidate.stanzas(raw,len(raw),cap))

    def test_default_stanza_sum_and_raw_aggregate_caps_still_reject_overflow(self):
        exact=line(131072,b'Provides')+line(131072,b'Description')
        self.assertEqual(len(list(candidate.stanzas(exact,len(exact)))),1)
        over=exact+b'X: x\n'
        with self.assertRaisesRegex(ValueError,'Metadata stanza cap'):list(candidate.stanzas(over,len(over)))
        small=line(70840)
        with self.assertRaisesRegex(ValueError,'Bounded LF metadata'):list(candidate.stanzas(small,len(small)-1))
        combined=line(16,b'Provides')+line(16,b'Description')
        with self.assertRaisesRegex(ValueError,'Metadata stanza cap'):list(candidate.stanzas(combined,len(combined),31))

    def test_case_folded_duplicate_and_continuation_guards_unchanged(self):
        raw=line(70840)+b'pRoViDeS: duplicate\n'
        with self.assertRaisesRegex(ValueError,'Duplicate metadata field'):list(candidate.stanzas(raw,len(raw)))
        raw=b' '+b'x'*70839+b'\n'
        with self.assertRaisesRegex(ValueError,'Continuation has field'):list(candidate.stanzas(raw,len(raw)))
        raw=b'Provides: first\n'+b' '+b'x'*70839+b'\n'
        self.assertTrue(list(candidate.stanzas(raw,len(raw)))[0]['provides'].startswith('first\nx'))

    def test_release_stanza_cap_does_not_relax_new_line_ceiling(self):
        raw=line(131072);self.assertEqual(len(list(candidate.stanzas(raw,4*candidate.MIB,4*candidate.MIB))),1)
        raw=line(131073)
        with self.assertRaisesRegex(ValueError,'Metadata line cap'):list(candidate.stanzas(raw,4*candidate.MIB,4*candidate.MIB))
        for invalid in (b'Provides: a\x00b\n',b'Provides: a\rb\n'):
            with self.assertRaisesRegex(ValueError,'Bounded LF metadata'):list(candidate.stanzas(invalid,len(invalid)))


if __name__=='__main__':unittest.main(verbosity=2)
