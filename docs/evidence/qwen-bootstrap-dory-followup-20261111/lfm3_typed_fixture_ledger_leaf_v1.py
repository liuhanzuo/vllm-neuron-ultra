"""Inert fixture ledger leaf; imports no native library and starts no process.

The independently admitted caller provides the original numpy_inputs callable,
complete original case matrices, and authenticated runtime versions. This leaf
checks pinned source/code and exact case digests, then yields one fixture at a
time. It neither simulates a kernel nor writes files. Caller owns persistence,
original resource limits, dual-environment comparison and full-suite admission.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import CodeType, FunctionType

ROOT = Path(__file__).resolve().parent
GENERATOR = ROOT / 'run_lfm3_typed_writer_cpu_pilot_v3.py'
GENERATOR_SHA = 'f34462be2e8286db80e2dadf5915debfb14bf0af5dd285a4457e66241c03176a'
EXTENDED = ROOT / 'lfm3_batch_slots_bits_extended_v4.py'
EXTENDED_SHA = 'fdd5396b9309335800fda28fe529daaea6b70af39316bfbbd6fbc152fee4b4e8'
MATRICES = {
    'regression_cpu': (1648, 123, '68e6ead99c215943ecb8123cefb7d2d2eb37b4f616156929276e2958c64090dc'),
    'extended_cpu': (2068, 168, 'b08c2282d952d4ce2c33e7c969faac98ee45d7f190a225e0c5798fc21eee328c'),
}
OLD_FIRST_HASHES = (
    '5781e3d8081dc2fb9a8d4c3593c858dab77b09ed734bed5662d3d7d2a5dcb98d',
    '94555ec4d65509f60933cab1609a615e0d8e0b73a2c9a8a09c5657e5e6fe3f39',
    'df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119',
    'fd8dea2bf754093760877b1a8cdd6b455b320b268e09bff488826fef4482b4c5',
)
MAX_PAYLOAD_BYTES = 128 * 2**20
ROW_CAP_BYTES = 8192
BATCH_CAP_BYTES = 256 * 2**10


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def original_batches(values):
    """Same complete original batching; callers compare with original plan too."""
    result, begin = [], 0
    while begin < len(values):
        end = begin + 1
        if values[begin]['width'] < 524288:
            while end < len(values) and end-begin < 16 and values[end]['width'] < 524288:
                end += 1
        result.append([begin, end])
        begin = end
    return result


def validate_matrices(matrices):
    require(type(matrices) is dict and set(matrices) == set(MATRICES), 'Both original complete matrices')
    checked = copy.deepcopy(matrices)
    for phase, (count, batches, wanted) in MATRICES.items():
        values = checked[phase]
        require(type(values) is list and len(values) == count and digest(values) == wanted,
                'Every original case, order, seed, dtype, mapping and width is unchanged')
        require([value['id'] for value in values] == list(range(count))
                and len(original_batches(values)) == batches, 'Original identifiers and batch count')
    return checked


def authenticate_generator(generator):
    sources = {}
    for path, wanted in ((GENERATOR, GENERATOR_SHA), (EXTENDED, EXTENDED_SHA)):
        require(not path.is_symlink() and path.resolve() == path and path.stat().st_size < 2**20,
                'Bounded exact original generator source')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == wanted, 'Frozen original generator source')
        sources[str(path)] = wanted
        if path == GENERATOR:
            compiled = compile(raw, str(path), 'exec', dont_inherit=True, optimize=0)
            expected = next(c for c in compiled.co_consts
                            if isinstance(c, CodeType) and c.co_name == 'numpy_inputs')
    require(type(generator) is FunctionType and generator.__code__ == expected
            and generator.__defaults__ is None and generator.__kwdefaults__ is None
            and generator.__closure__ is None, 'Exact original input generator callable and defaults')
    protocol = generator.__globals__.get('p')
    require(protocol is not None and protocol.ROOT == ROOT, 'Original generator source root binding')
    # numpy_inputs uses these two globals when loading the extended definition;
    # the already-admitted caller owns import/runtime authentication separately.
    require(generator.__globals__.get('ast') is ast, 'Original extended-source AST parser')
    return sources


def describe_array(value, dtype, shape):
    require(value.dtype.name == dtype and tuple(value.shape) == tuple(shape), 'Exact fixture dtype and shape')
    itemsize = 2 if dtype == 'bfloat16' else 4
    count = 1
    for size in shape:
        require(type(size) is int and size > 0, 'Positive original shape')
        count *= size
    require(value.dtype.itemsize == itemsize and value.nbytes == count * itemsize <= MAX_PAYLOAD_BYTES
            and value.flags.c_contiguous is True and value.flags.writeable is True,
            'Original complete contiguous writable allocation, at most 128 MiB')
    raw = value.tobytes()
    require(type(raw) is bytes and len(raw) == value.nbytes, 'Hash all original bytes without conversion')
    return dict(dtype=dtype, shape=list(shape), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def fixture_record(case, phase, operands):
    """Original last-valid-row assignment oracle; holds no diagnostic clones."""
    require(phase in MATRICES and type(operands) is tuple and len(operands) == 3, 'Three original operands')
    cache, updates, slots = operands
    require(slots.tolist() == case['slots'], 'Actual int32 slots equal the original case mapping')
    dtype, capacity, width = case['dtype'], case['capacity'], case['width']
    before = [describe_array(cache, dtype, [capacity, width]),
              describe_array(updates, dtype, [len(case['slots']), width]),
              describe_array(slots, 'int32', [len(case['slots'])])]
    expected = cache.copy()
    for row, slot in enumerate(case['slots']):
        if 0 <= slot < case['capacity']:
            expected[slot] = updates[row]
    expected_row = describe_array(expected, dtype, [capacity, width])
    after = [describe_array(value, info['dtype'], info['shape']) for value, info in zip(operands, before)]
    require(before == after, 'Oracle construction never changes original inputs')
    record = dict(phase=phase, id=case['id'], case=copy.deepcopy(case), inputs=before, expected=expected_row)
    require(len(encoded(record)) <= ROW_CAP_BYTES, 'Fixed small fixture metadata only')
    return record, expected


def verify_same_fixture(original, current):
    """Old/new version metadata stays outside this exact scientific comparison."""
    require(original == current, 'All case fields and all four complete byte hashes match original environment')
    return True


class FixtureLedger:
    """Call only after admission; generate one unchanged original batch at a time."""
    def __init__(self, admit, generator, matrices, versions):
        require(callable(admit) and admit() is True, 'Caller owns serial runtime admission')
        require(sys.byteorder == 'little', 'Original little-endian input contract')
        require(versions in ({'numpy': '1.26.4', 'ml_dtypes': '0.5.4'},
                             {'numpy': '2.5.4', 'ml_dtypes': '0.6.0'}), 'Explicit original or migrated runtime versions')
        self.sources = authenticate_generator(generator)
        self.generator = generator
        self.matrices = validate_matrices(matrices)
        self.versions = copy.deepcopy(versions)
        self.admit = admit

    def batch(self, phase, begin, end):
        require(self.admit() is True, 'Revalidate existing admission before generating batch')
        validate_matrices(self.matrices)
        require(phase in MATRICES and type(begin) is type(end) is int
                and [begin, end] in original_batches(self.matrices[phase]), 'One complete original batch boundary')
        require(authenticate_generator(self.generator) == self.sources, 'Original generator binding still intact')
        # Yielded native arrays are consumed/dropped per case by the caller;
        # this leaf never accumulates payloads or writes them to disk.
        total_metadata_bytes = 0
        for case in self.matrices[phase][begin:end]:
            original = copy.deepcopy(case)
            operands = self.generator(copy.deepcopy(case), phase)
            require(case == original, 'No mutation of the pinned case matrix')
            record, expected = fixture_record(case, phase, operands)
            total_metadata_bytes += len(encoded(record))
            require(total_metadata_bytes <= BATCH_CAP_BYTES, 'Original 256 KiB batch-report budget')
            yield record, operands, expected


def validate_small_controls(arms):
    """Required receipt gate before any full CPU candidate batch is admitted.

    Caller must generate real records with fixture_record and actual strict
    output/input checks, close each CompactWriteback case, then restore it.
    These control receipts are additional to, never counted in, the 3,716.
    """
    require(type(arms) is list and len(arms) == 3, 'Original, candidate, restored-original controls')
    require([row['name'] for row in arms] == ['original', 'candidate', 'restored_original'], 'Exact control order')
    for index, row in enumerate(arms):
        fixture = row['fixture']
        require(fixture['phase'] == 'extended_cpu' and fixture['id'] == 0 and fixture['case'] == dict(id=0, seed=20261103,
                kind='selector_coverage', capacity=8, dtype='bfloat16', width=256, slots=[0]),
                'Exact original v5 first selector case')
        require(tuple(x['sha256'] for x in fixture['inputs']) + (fixture['expected']['sha256'],) == OLD_FIRST_HASHES,
                'Exact original first-case input and oracle bytes')
        descriptors = [dict(dtype=dtype, shape=shape, bytes=size, sha256=sha)
                       for dtype, shape, size, sha in zip(
                           ('bfloat16', 'bfloat16', 'int32', 'bfloat16'),
                           ([8, 256], [1, 256], [1], [8, 256]),
                           (4096, 512, 4, 4096), OLD_FIRST_HASHES)]
        require(fixture['inputs'] + [fixture['expected']] == descriptors,
                'Exact first-case metadata and full hashes, no substituted fixture descriptions')
        require(row['shape_equal'] is row['dtype_equal'] is row['bitwise_equal'] is True
                and row['actual_sha256'] == OLD_FIRST_HASHES[3], 'Selector output remains exactly correct')
        require(row['inputs_unchanged'] is (index == 1), 'Original/restored fail; only candidate inputs stay unchanged')
        after = row['input_after_sha256']
        require(type(after) is list and len(after) == 3
                and all(type(x) is str and len(x) == 64 and set(x) <= set('0123456789abcdef') for x in after),
                'Three complete post-call input hashes')
        require(after == list(OLD_FIRST_HASHES[:3]) if index == 1 else
                after[0] != OLD_FIRST_HASHES[0] and after[1] != OLD_FIRST_HASHES[1] and after[2] == OLD_FIRST_HASHES[2],
                'Both original BF16 inputs change; int32 slots never change')
        mode = 'precise_bf16_writeback' if index == 1 else 'original'
        probe, event = row['probe'], row['extraction']
        require(probe == dict(mode=mode, installed=True, restored=True, failed=False, completed_cases=1,
                             extraction_calls=4, substitutions=2 if index == 1 else 0,
                             installed_sdk_source_modified=False), 'Actual completed extraction and restoration receipt')
        require(event['phase'] == 'extended_cpu' and event['id'] == 0 and event['mode'] == mode
                and event['extraction_calls'] == 4 and event['substitutions'] == (2 if index == 1 else 0),
                'Exactly four bounded extractions per control')
        expected_events = []
        for position, (name, descriptor) in enumerate(zip(('cache', 'updates', 'slots', 'output'), descriptors)):
            changed = index == 1 and position < 2
            out_dtype = 'float32' if index != 1 and position < 2 else descriptor['dtype']
            expected_events.append(dict(operand=name, tensor_dtype=descriptor['dtype'], extracted_dtype=out_dtype,
                shape=descriptor['shape'], bytes=descriptor['bytes'] * (2 if index != 1 and position < 2 else 1),
                requested_precise=True if position == 3 else 'omitted',
                effective_precise=changed or position == 3, substitution=changed))
        require(event['events'] == expected_events, 'Full exact ordered extraction inventory')
    require(arms[0]['fixture'] == arms[1]['fixture'] == arms[2]['fixture'], 'Identical fixtures across all three controls')
    require(arms[0]['input_after_sha256'] == arms[2]['input_after_sha256']
            and arms[0]['extraction'] == arms[2]['extraction'], 'Removal reproduces original failure exactly')
    return dict(complete=True, original_failure_preserved=True, restored_original_reproduced=True,
                candidate_selector_control_passed=True, controls_sha256=digest(arms),
                full3716_passed=False, device_passed=False, new_speedup_claim=False)
