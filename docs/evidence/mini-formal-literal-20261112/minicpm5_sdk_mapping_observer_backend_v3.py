"""Original observer ports and Store semantics with three explicit wire encodings.

All native/wait operations are the frozen v2 implementations. Only persisted
admission, owner and exit representations change; full in-memory values do not.
"""
from research import minicpm5_sdk_mapping_observer_backend_v2 as previous
from research import minicpm5_formal_literal_codec_v1 as codec
from research import minicpm5_formal_literal_result_v1 as literal_result

ORIGINAL_NAMES = previous.ORIGINAL_NAMES
OBSERVER_NAMES = previous.OBSERVER_NAMES
make_ports = previous.make_ports
logic = previous.logic
require = previous.require


def make_store_class(original):
    base = previous.make_store_class(original)

    class LiteralObserverStore(base):
        KINDS = {'admission.json': 'admission', 'phase-1-owner.json': 'owner', 'phase-1-exit.json': 'exit'}

        def __init__(self, root, guard, *, literal):
            super().__init__(root, guard)
            codec.literal_bytes(literal, guard)
            self.literal = literal
            self.runtime_reservation = self.reserved_admission = self.reserved_argv = None

        def prospective_admission(self, value):
            raw = codec.encode('admission', value, self.literal, self.guard)
            return raw, dict(path=str(self.root / 'admission.json'), sha256=codec.digest(raw, self.guard))

        def reserve_runtime(self, value, reference, argv):
            require(self.runtime_reservation is None, 'One complete pre-spawn reservation')
            inventory = self.inventory()
            require(inventory['names'] == [], 'Reserve all original runtime artifacts before admission write')
            raw, expected = self.prospective_admission(value)
            require(reference == expected, 'Reservation binds actual proposed wire reference')
            budget = codec.future_runtime_budget(raw, argv, self.literal)
            budget.update(literal_result.future_stdout_budget(raw, argv, self.literal, str(self.root)))
            require(self.logical_metadata + budget['total_runtime_bound'] + logic.TRACE_CAP + logic.SUMMARY_CAP
                    <= original.serial.CAPS['metadata_bytes'], 'Original complete metadata reservation')
            # Preserve the original conservative 64KiB per-file allocation margin.
            require(self.allocated + budget['total_runtime_bound'] + logic.TRACE_CAP + logic.SUMMARY_CAP
                    + 6 * 65536 <= original.serial.CAPS['allocated_bytes'], 'Original complete allocation reservation')
            self.runtime_reservation = budget
            self.reserved_admission = dict(reference)
            self.reserved_argv = list(argv)
            return dict(budget)

        def before_spawn(self, admission, reference, argv):
            require(self.runtime_reservation is not None and self.reserved_admission == reference
                    and self.reserved_argv == argv, 'Complete actual reservation required before SDK Popen')
            raw, expected = self.prospective_admission(admission)
            require(expected == reference, 'In-memory admission unchanged since reservation')
            persisted = self.reference('admission.json')
            require({key: persisted[key] for key in ('path', 'sha256')} == reference
                    and self.read(persisted) == raw, 'Actual durable wire admission matches complete reservation')
            self.guard()

        def write(self, name, raw):
            if name in self.KINDS:
                require(self.runtime_reservation is not None, 'No boundary artifact before whole reservation')
                raw = codec.encode(self.KINDS[name], codec.parsed(raw), self.literal, self.guard)
                key = {'admission.json': 'admission_wire_bytes', 'phase-1-owner.json': 'owner_wire_bound',
                       'phase-1-exit.json': 'exit_wire_bound'}[name]
                require(len(raw) <= self.runtime_reservation[key], 'Actual whole boundary bytes exceed reservation')
            return super().write(name, raw)

        def load(self, name):
            reference = self.reference(name)
            raw = self.read(reference)  # Original complete readback/hash/identity/guard.
            if name in self.KINDS:
                value = codec.decode(self.KINDS[name], raw, self.literal,
                                     wire_sha256=reference['sha256'], check=self.guard)
            else:
                value = codec.parsed(raw)
            return value, reference

    return LiteralObserverStore
