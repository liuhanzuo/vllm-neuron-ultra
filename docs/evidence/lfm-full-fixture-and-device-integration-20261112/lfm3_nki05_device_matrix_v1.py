"""Exact original pure matrices; bounded IO is supplied by an admitted context.

Generated as ordinary reviewed source during preparation, never dynamically
extracted/re-written at runtime. Original 2068 CPU cases are retained for count
comparison only; this module never runs the simulator.
"""
from pathlib import Path

WIDTHS = (128, 256, 384, 7168, 16256, 16384, 16512, 32768, 65536,
          131072, 262144, 524288, 2097152)


def mappings(capacity, rows, exhaustive_winners=False):
    """Finite explicit coverage, not exhaustive enumeration of B**B mappings."""
    assert capacity in (8, 16) and 1 <= rows <= capacity
    result = []
    def add(value):
        assert len(value) == rows
        if value not in result:
            result.append(value)
    for shift in range(capacity):
        add([(i + shift) % capacity for i in range(rows)])
    add([(capacity - 1 - i) % capacity for i in range(rows)])
    for missing in (-1, -9, capacity, capacity + 9, -(2**31), 2**31 - 1):
        add([missing] * rows)
    for destination in range(capacity):
        winners = range(rows) if exhaustive_winners else sorted({0, rows // 2, rows - 1})
        for winner in winners:
            single = [-1] * rows
            single[winner] = destination
            add(single)
            # Every possible last writer at every destination; later rows inactive.
            add([destination if i <= winner else -1 for i in range(rows)])
    add([i % capacity if i % 2 else -1 for i in range(rows)])
    add([i // 2 for i in range(rows)])
    return result


def cpu_cases():
    cases = []
    for capacity in (8, 16):
        for dtype in ('bfloat16', 'float32'):
            for rows in sorted({1, 4, capacity}):
                for slots in mappings(capacity, rows, True):
                    cases.append(dict(kind='selector_coverage', capacity=capacity,
                        dtype=dtype, width=256, slots=slots))
            for width in WIDTHS:
                for rows in (1, capacity):
                    choices = [[capacity - 1]] if rows == 1 else [
                        list(reversed(range(capacity))), [-1] * capacity,
                        [capacity - 1] * capacity,
                        [i if i % 2 else capacity + 5 for i in range(capacity)]]
                    for slots in choices:
                        cases.append(dict(kind='real_and_tile_boundary_width', capacity=capacity,
                            dtype=dtype, width=width, slots=slots))
    return [dict(id=i, seed=20261103 + i, **row) for i, row in enumerate(cases)]


def device_specs():
    specs = []
    for capacity in (8, 16):
        for kind, tail in [('selector', (256,)), ('conv', (1024, 7))] + [
                ('kv' + str(n), (2, n, 128)) for n in (128, 256, 512, 1024, 2048)]:
            for rows in sorted({1, 4, capacity}):
                specs.append(dict(name=f'b{capacity}-{kind}-r{rows}', capacity=capacity,
                    rows=rows, shape=[capacity, *tail], dtype='bfloat16'))
    return specs


def alias_specs():
    return [dict(name=f'b{capacity}-{kind}-{variant}', capacity=capacity, kind=kind,
                 length=length, variant=variant)
            for capacity in (8, 16)
            for kind, length in [('conv', 7)] + [('kv' + str(n), n) for n in (128,256,512,1024,2048)]
            for variant in ('reference', 'candidate')]


def alias_mappings(capacity):
    # All slots first populated, permuted, partially inactive, collided, reused,
    # and inactive together; every last-writer position is exercised in CPU cases.
    return [list(range(capacity)), list(reversed(range(capacity))),
            [(i + 1) % capacity for i in range(capacity)],
            [i if i % 2 else -1 for i in range(capacity)],
            [i // 2 for i in range(capacity)], [-1] * capacity,
            [capacity - 1] * capacity, list(range(capacity))]



class BoundMatrix:
    mappings=staticmethod(mappings)
    cpu_cases=staticmethod(cpu_cases)
    device_specs=staticmethod(device_specs)
    alias_specs=staticmethod(alias_specs)
    alias_mappings=staticmethod(alias_mappings)

    def __init__(self,context,phase):
        self.context,self.phase=context,phase
        self.DEST=Path(context.paths(phase)[1])

    def save(self,path,value):
        self.context.require_phase(self.phase)
        self.context.save(path,value,self.phase)

    def sha(self,path):
        self.context.require_phase(self.phase)
        return self.context.hash_artifact(path,self.phase)
