"""Small parser-only examples: no gpgv, network, keyring, payload or lock IO."""
import datetime as dt
import unittest
from unittest.mock import patch
import dory_ubuntu_gpgv_status_compliance_v1 as m

PRIMARY='F6ECB3762474EDA9D21B7022871920D1991BC93C'
NOW=dt.datetime(2026,11,11,12,tzinfo=dt.timezone.utc)
TRUST={'primary_fingerprints':[PRIMARY]}
ANNOTATION=m.COMPLIANCE+b'\n'


def status(signing=PRIMARY):
    # The documented six-line shape supplied by root; not a read of saved data.
    return (f'[GNUPG:] NEWSIG\n'
        f'[GNUPG:] KEY_CONSIDERED {PRIMARY} 0\n'
        f'[GNUPG:] SIG_ID UJOdLzJmCV5QobCR5YILLcUILEI 2022-04-21 1650561399\n'
        f'[GNUPG:] GOODSIG {signing[-16:]} Ubuntu Archive Automatic Signing Key (2018)\n'
        f'[GNUPG:] VALIDSIG {signing} 2022-04-21 1650561399 0 4 0 1 10 00 {PRIMARY}\n').encode()


class Tests(unittest.TestCase):
    def test_exact_observed_shape_has_identical_old_signature_receipt(self):
        raw=status()+ANNOTATION;before=raw
        expected=m.original.parse_status(status(),0,TRUST,NOW)
        self.assertEqual(m.parse_status(raw,0,TRUST,NOW),expected)
        self.assertEqual(raw,before)
        self.assertEqual(expected[0]['digest_algorithm'],'SHA512')
        self.assertEqual(expected[0]['primary_fingerprint'],PRIMARY)
        with self.assertRaisesRegex(ValueError,'unreviewed gpgv status'):
            m.original.parse_status(raw,0,TRUST,NOW)

    def test_optional_annotation_and_one_per_distinct_signature(self):
        second=status('A'*40)
        for raw in (status(),status()+ANNOTATION+second,status()+second+ANNOTATION,
                    status()+ANNOTATION+second+ANNOTATION):
            expected=m.original.parse_status(raw.replace(ANNOTATION,b''),0,TRUST,NOW)
            self.assertEqual(m.parse_status(raw,0,TRUST,NOW),expected)

    def test_old_parser_gets_every_other_original_byte_once(self):
        raw=status()+ANNOTATION+b'[GNUPG:] ERROR retained exactly\n'
        original=m.original.parse_status
        with patch.object(m.original,'parse_status',wraps=original)as checked:
            with self.assertRaises(ValueError):m.parse_status(raw,0,TRUST,NOW)
        checked.assert_called_once_with(status()+b'[GNUPG:] ERROR retained exactly\n',0,TRUST,NOW)

    def test_malformed_unknown_flags_and_whitespace_variants_rejected(self):
        for line in (b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE ',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE 8',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE 8 23',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE 23 23',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE 24',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE 023',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE +23',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE 23 extra',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE  23',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE 23 ',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE\t23',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE \t23',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE 23\t',
                     b'[GNUPG:] VERIFICATION_COMPLIANCE_MODE -23'):
            with self.subTest(line=line),self.assertRaises(ValueError):
                m.parse_status(status()+line+b'\n',0,TRUST,NOW)

    def test_unscoped_early_duplicate_or_next_group_annotation_rejected(self):
        lines=status().splitlines(keepends=True)
        for raw in (ANNOTATION+status(),lines[0]+ANNOTATION+b''.join(lines[1:]),
                    b''.join(lines[:4])+ANNOTATION+lines[4],status()+ANNOTATION+ANNOTATION,
                    status()+ANNOTATION+b'[GNUPG:] NEWSIG\n'+ANNOTATION,
                    b''.join(lines[:3])+lines[4]+ANNOTATION+lines[3]):
            with self.subTest(raw=raw[:35]),self.assertRaises(ValueError):m.parse_status(raw,0,TRUST,NOW)

    def test_annotation_never_overrides_failure_status_or_nonzero_exit(self):
        for tag in ('BADSIG','ERRSIG','NO_PUBKEY','EXPKEYSIG','EXPSIG','REVKEYSIG',
                    'FAILURE','ERROR','NODATA','UNREVIEWED','DECRYPTION_COMPLIANCE_MODE'):
            bad=f'[GNUPG:] {tag} 23\n'.encode()
            for raw in (status()+bad+ANNOTATION,status()+ANNOTATION+bad):
                with self.subTest(tag=tag),self.assertRaises(ValueError):m.parse_status(raw,0,TRUST,NOW)
        for code in (1,2,137,-9):
            with self.subTest(code=code),self.assertRaises(ValueError):m.parse_status(status()+ANNOTATION,code,TRUST,NOW)

    def test_old_fingerprint_digest_date_and_complete_group_rules_remain(self):
        good=status()
        for raw in (good.replace(PRIMARY.encode(),b'F'*40),
                    good.replace(b'GOODSIG '+PRIMARY[-16:].encode(),b'GOODSIG '+b'0'*16),
                    good.replace(b' 0 4 0 1 10 00 ',b' 0 4 0 1 2 00 '),
                    good.replace(b' 1650561399 0 4 ',b' 1650561399 1650561400 4 '),
                    good.replace(b'VALIDSIG ',b'VALIDSIG incomplete '),
                    good+ANNOTATION+good,
                    good+ANNOTATION+b'[GNUPG:] NEWSIG\n'):
            with self.subTest(raw=raw[:30]),self.assertRaises(ValueError):m.parse_status(raw+ANNOTATION,0,TRUST,NOW)
        five=b''.join(status(hex(n)[2:].upper()*40)+ANNOTATION for n in range(10,15))
        with self.assertRaises(ValueError):m.parse_status(five,0,TRUST,NOW)

    def test_original_input_cap_and_full_line_integrity_checked_before_filter(self):
        good=status()+ANNOTATION
        for raw in (b'',good[:-1],good.replace(b'\n',b'\r\n'),good+b'\x00\n',
                    good.decode(),bytearray(good),good+ANNOTATION*(m.original.MIB//len(ANNOTATION)+1)):
            with self.subTest(kind=type(raw).__name__),self.assertRaises(ValueError):m.parse_status(raw,0,TRUST,NOW)


if __name__=='__main__':unittest.main(verbosity=2)
