"""Independently verify saved MiniCPM5 frame1 BF16 boundary bytes using stdlib.

Run only after the actual capture/replay process has ended and been drained.
This reads captures; it does not execute model math or prove process lifecycle.
"""
import argparse
from contextlib import ExitStack
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct

import review_minicpm5_frame1_metadata_v1 as meta

METADATA_SHA = '8d5243d29480870b97b3821a2a54bceee3228d05eb72817c61ad872500e38ce3'
require, same = meta.require, meta.same
ROLES, FIELDS = meta.ROLES, meta.FIELDS
REL_TOL, COS_ABS_TOL = 1e-11, 1e-12


def units(word):
    """Exact integer in units of 2**-133, including distinct signed-zero bits."""
    exponent = (word >> 7) & 255
    if exponent == 255:
        return None
    fraction = word & 127
    value = fraction if exponent == 0 else (128 + fraction) << (exponent - 1)
    return -value if word & 32768 else value


VALUES = tuple(units(i) for i in range(65536))


class Metric:
    def __init__(self):
        self.count = self.different = self.maximum = self.total = 0
        self.dot = self.square_a = self.square_b = 0

    def add(self, left, right):
        same(len(left), len(right), 'Same complete row width')
        for x, y in zip(left, right):
            a, b = VALUES[x], VALUES[y]
            require(a is not None and b is not None, 'Finite BF16 operands')
            delta = abs(a - b)
            self.count += 1
            self.different += x != y
            self.maximum = max(self.maximum, delta)
            self.total += delta
            self.dot += a * b
            self.square_a += a * a
            self.square_b += b * b

    def result(self):
        require(self.count > 0, 'Nonempty comparison')
        norm = math.sqrt(self.square_a) * math.sqrt(self.square_b)
        return dict(raw_exact=self.different == 0, different_elements=self.different,
                    compared_elements=self.count, nonfinite_elements=0,
                    max_abs=math.ldexp(float(self.maximum), -133),
                    mean_abs=math.ldexp(float(self.total) / self.count, -133),
                    cosine=float(self.dot) / norm if norm else None)


def compare_metric(actual, claimed):
    same(set(actual), set(claimed), 'All metric fields present')
    for key in ('raw_exact', 'different_elements', 'compared_elements', 'nonfinite_elements', 'max_abs'):
        same(actual[key], claimed[key], 'Recomputed exact metric: ' + key)
    for key in ('mean_abs', 'cosine'):
        a, b = actual[key], claimed[key]
        if a is None or b is None:
            same(a, b, 'Undefined zero-norm cosine')
        else:
            require(type(b) in (int, float) and math.isfinite(b), 'Finite reported metric')
            require(math.isclose(a, b, rel_tol=REL_TOL,
                                 abs_tol=COS_ABS_TOL if key == 'cosine' else 0),
                    'Independent double-precision reduction: ' + key)


def identity(path, info):
    return dict(realpath=str(path), device=info.st_dev, inode=info.st_ino,
                size=info.st_size, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)


class Payload:
    def __init__(self, row, directory):
        self.row = row
        self.path = Path(row['path'])
        same(self.path, directory / (row['name'] + '.bf16'), 'Exact field file')
        require(self.path.is_absolute() and self.path.resolve() == self.path and
                not any(p.is_symlink() for p in (self.path, *self.path.parents)),
                'Canonical capture path without symlinks')
        require(row['encoding'] == 'packed_bf16_le' and row['observed_not_recomputed'] is True,
                'Original packed logical-order BF16 capture')
        size = meta.layout(row['metadata'], row['metadata']['shape'])
        require(size == row['bytes'] == row['identity']['size'] and 0 < size <= 16 * 2**20,
                'Full bounded capture size')
        same(row['sha256'], row['metadata']['sha256'], 'Raw metadata digest link')
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(self.fd)
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.getuid(),
                    'Owned ordinary capture')
            same(identity(self.path, info), row['identity'], 'Original capture identity')
        except BaseException:
            os.close(self.fd)
            raise
        self.length, self.width = row['metadata']['shape'][1:]
        self.index = 0
        self.digest = hashlib.sha256()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        os.close(self.fd)

    def next_row(self):
        require(self.index < self.length, 'Read only complete captured rows')
        raw = os.pread(self.fd, 2 * self.width, 2 * self.width * self.index)
        same(len(raw), 2 * self.width, 'No truncated capture row')
        self.digest.update(raw)
        words = struct.unpack('<' + str(self.width) + 'H', raw)
        require(all(VALUES[x] is not None for x in words), 'Finite entire capture including padding')
        self.index += 1
        return words

    def finish(self):
        same(self.index, self.length, 'Read every captured row')
        same(self.digest.hexdigest(), self.row['sha256'], 'Full capture bytes authenticated')
        # Some filesystems coalesce timestamps for immediate same-size rewrites.
        # Re-read the bytes as well as checking stat identity.
        final_digest = hashlib.sha256()
        for offset in range(0, self.row['bytes'], 65536):
            wanted = min(65536, self.row['bytes'] - offset)
            part = os.pread(self.fd, wanted, offset)
            same(len(part), wanted, 'Complete final capture read')
            final_digest.update(part)
        same(final_digest.hexdigest(), self.row['sha256'], 'Capture content unchanged at completion')
        same(identity(self.path, os.fstat(self.fd)), self.row['identity'], 'Capture stable during read')
        same(identity(self.path, self.path.lstat()), self.row['identity'], 'Capture path unchanged')
        return dict(path=str(self.path), sha256=self.digest.hexdigest(), bytes=self.row['bytes'])


def boundary(streams, replicated):
    """Rank concatenation is per token, never whole-file concatenation."""
    native, rank0, rank1 = streams
    require(native.length <= rank0.length == rank1.length, 'Native prefix and equal CPU buckets')
    same(native.width, rank0.width if replicated else rank0.width + rank1.width, 'Complete boundary width')
    same(rank0.width, rank1.width, 'Equal original TP2 shards')
    real, prediction = Metric(), Metric()
    for index in range(rank0.length):
        a, b = rank0.next_row(), rank1.next_row()
        if replicated:
            same(a, b, 'Full CPU replicated boundary, including padding')
        if index < native.length:
            n = native.next_row()
            joined = a if replicated else a + b
            real.add(n, joined)
            if index == native.length - 1:
                prediction.add(n, joined)
    return dict(real_tokens=real.result(), prediction=prediction.result())


def audit_raw(root, captures, replay):
    references, metrics = [], {}
    for field in FIELDS:
        with ExitStack() as stack:
            streams = []
            for role in ROLES:
                rows = [r for r in captures[role]['captures'] if r['name'] == field]
                same(len(rows), 1, 'One field per original role')
                native = role == 'native'
                shape = [1, 1231 if native else 2048,
                         2048 if field in ('input', 'output') else (6144 if native else 3072)]
                meta.layout(rows[0]['metadata'], shape)
                streams.append(stack.enter_context(Payload(rows[0], root / role)))
            metrics[field] = boundary(streams, field in ('input', 'output'))
            references.extend(s.finish() for s in streams)
        for scope in ('real_tokens', 'prediction'):
            compare_metric(metrics[field][scope], replay['observed_native_vs_cpu_boundaries'][field][scope])
    same(len(references), 15, 'All 15 captures')
    same(sum(r['bytes'] for r in references), 164515840, 'Whole original payload budget')
    return dict(captures=references, observed_boundary_metrics=metrics,
                all15_full_payload_hashes_verified=True, all15_full_payloads_finite=True,
                both_cpu_replicated_boundaries_raw_exact=True,
                observed_boundary_metrics_recomputed=True,
                fixed_input_counterfactuals_recomputed=False,
                model_math_reexecuted=False, actual_exit_verified=False,
                physical_drain_verified=False, full_model_qualified=False, new_speedup_claim=False)


def main(args):
    same(hashlib.sha256(Path(meta.__file__).read_bytes()).hexdigest(), METADATA_SHA, 'Frozen metadata reviewer')
    reader = meta.Reader()
    review_ref = dict(path=args.metadata_review, sha256=args.metadata_review_sha256)
    review = reader.load(review_ref)
    same(review['schema'], 'minicpm5_frame1_independent_metadata_review_v1', 'Actual metadata review')
    same(review['reviewer'], '/root', 'Root metadata review')
    same(review['source_sha256'], METADATA_SHA, 'Original metadata reviewer source')
    spec = reader.load(review['spec'])
    same(spec['schema'], 'minicpm5_frame1_mlp_spec_v1', 'Actual frame1 spec')
    same(spec['base_plan'], meta.BASE, 'Frozen original plan')
    root = Path(spec['output_root'])
    require(root.parent == meta.R and meta.re.fullmatch('minicpm5-frame1-mlp-r[0-9]+-20261108-v[0-9]+', root.name),
            'Actual finite frame1 output namespace')
    same(review['result']['path'], str(root / 'result.json'), 'Actual original result path')
    result = reader.load(review['result'])
    same(result['report'], review['report'], 'Same original replay')
    same(review['report']['path'], str(root / 'replay.json'), 'Original replay path')
    replay = reader.load(review['report'])
    captures = {}
    for role in ROLES:
        ref = replay['captures'][role]
        same(ref['path'], str(root / role / 'capture.json'), 'Actual capture path')
        captures[role] = reader.load(ref)
    plan = reader.load(meta.BASE)
    same(meta.audit(plan, spec, replay, captures, reader.load), review['checks'],
         'Original metadata review fully reproduced')
    for path, ref in reader.references.items():
        if path != review_ref['path']:
            same(ref, review['metadata_references'][path], 'Every input bound by original review')
    checked = audit_raw(root, captures, replay)
    output = dict(schema='minicpm5_frame1_independent_raw_boundary_review_v1', reviewer='/root',
                  observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  metadata_review=review_ref, spec=review['spec'], result=review['result'],
                  checks=checked, metadata_references=reader.references,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  metric_method='Exact integer BF16 accumulation in units of 2**-133; final conversion to float64.',
                  reduction_tolerance=dict(relative=REL_TOL, cosine_absolute=COS_ABS_TOL,
                                           raw_equality_count_and_max_abs='exact'),
                  scope='Saved observed boundary bytes only; no model arithmetic, counterfactual replay, lifecycle or model qualification.')
    dest = Path(args.output)
    require(dest.parent == meta.R / 'research' and dest.suffix == '.json', 'New research JSON only')
    with dest.open('x') as f:
        f.write(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(dest),captures_verified=15,model_qualified=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata-review','metadata-review-sha256','output'):
        parser.add_argument('--' + name, required=True)
    main(parser.parse_args())
