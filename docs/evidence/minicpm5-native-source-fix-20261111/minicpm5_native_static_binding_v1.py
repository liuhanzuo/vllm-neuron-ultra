"""Source-only lifecycle binding: immutable catalog and live mutation checks.

No admission, model import, process action or automatic IO. The original full
binding runs at entry and terminal boundaries. Every lifecycle usage call still
checks all binding-consumed JSON containers and scalar values, including nested
mutation and equal-value container rebinding. Historical resolved branches not
consumed by the original binding are not copied or traversed here.
"""
import math

from research import minicpm5_layer1_operator_native_lifecycle_core_v2 as original
from research import minicpm5_native_source_phase_seal_v1 as sealed

require = original.require
RESOLVED_FIELDS = ('plan', 'plan_sha256', 'native_runtime_output_catalog',
                   'native_runtime_output_catalog_sha256', 'runtime_admitted',
                   'full_runtime_catalog_pending', 'runtime_integration_pending',
                   'physical_checks_performed')


def _fact(value):
    kind = type(value)
    require(kind in (dict, list, str, int, float, bool, type(None)), 'Exact supported static JSON type')
    if kind is float:
        require(math.isfinite(value), 'Finite static JSON number')
        return kind, value.hex()
    return kind, value


def _matches(value, fact):
    kind, expected = fact
    if type(value) is not kind:
        return False
    if kind in (dict, list):
        return value is expected
    if kind is float:
        return value.hex() == expected
    return value == expected


class JsonFacts:
    """Immutable facts reference original containers without a second JSON tree."""
    __slots__ = ('_records',)

    def __init__(self, values):
        records, visited, active = [], set(), set()
        def visit(value):
            fact = _fact(value)
            if type(value) not in (dict, list):
                return fact
            key = id(value)
            require(key not in active, 'No cyclic static JSON')
            if key in visited:
                return fact
            active.add(key)
            if type(value) is dict:
                require(all(type(key) is str for key in value), 'Exact static JSON keys')
                children = tuple((key, visit(child)) for key, child in value.items())
            else:
                children = tuple((index, visit(child)) for index, child in enumerate(value))
            records.append((value, type(value), children))
            active.remove(key)
            visited.add(key)
            return fact
        for value in values:
            visit(value)
        object.__setattr__(self, '_records', tuple(records))

    def __setattr__(self, name, value):
        raise AttributeError('Static JSON facts are immutable')

    def __delattr__(self, name):
        raise AttributeError('Static JSON facts are immutable')

    def check(self):
        for value, kind, children in self._records:
            require(type(value) is kind and len(value) == len(children), 'Unchanged static JSON container extent')
            if kind is dict:
                require(all(type(key) is str for key in value), 'Unchanged exact static JSON keys')
                valid = all(key in value and _matches(value[key], fact) for key, fact in children)
            else:
                valid = all(_matches(value[index], fact) for index, fact in children)
            require(valid, 'Unchanged static JSON members, scalar types and child identities')


class StaticBinding:
    """One authenticated lifecycle binding; caller keeps original mutable APIs."""
    __slots__ = ('_contract', '_resolved', '_roots', '_facts', '_catalog', '_output_root')

    def __init__(self, contract, resolved):
        require(type(contract) is dict and type(resolved) is dict, 'Exact original lifecycle roots')
        roots = tuple((key, _fact(resolved[key])) for key in RESOLVED_FIELDS)
        facts = JsonFacts((contract, *(resolved[key] for key in RESOLVED_FIELDS)))
        original.binding(contract, resolved)
        object.__setattr__(self, '_contract', contract)
        object.__setattr__(self, '_resolved', resolved)
        object.__setattr__(self, '_roots', roots)
        object.__setattr__(self, '_facts', facts)
        object.__setattr__(self, '_catalog', sealed.seal_json(resolved['native_runtime_output_catalog']))
        object.__setattr__(self, '_output_root', contract['output_root'])
        self.check(contract, resolved)

    def __setattr__(self, name, value):
        raise AttributeError('Authenticated lifecycle binding is immutable')

    def __delattr__(self, name):
        raise AttributeError('Authenticated lifecycle binding is immutable')

    def _check_roots(self, contract, resolved):
        require(contract is self._contract and resolved is self._resolved, 'Original bound lifecycle root identities')
        require(all(key in resolved and _matches(resolved[key], fact) for key, fact in self._roots),
                'Original bound plan/catalog/digest/flag identities and values')

    def check(self, contract, resolved):
        self._check_roots(contract, resolved)
        self._facts.check()

    def verify_full(self, contract, resolved):
        self.check(contract, resolved)
        original.binding(contract, resolved)

    def walk(self, contract, resolved, writers):
        # Exactly as before, full binding validation occurs in usage after the
        # live walk. Only immutable validated paths are consumed by the walker.
        # This also catches a mutation during IO at the original check point.
        self._check_roots(contract, resolved)
        return sealed.walk_sealed(self._output_root, self._catalog, writers)

    def inventory(self, contract, resolved, entries, *, successful_driver, writers):
        if successful_driver:
            self.verify_full(contract, resolved)
        else:
            self.check(contract, resolved)
        return sealed.inventory_sealed(self._catalog, entries, successful_driver=successful_driver, writers=writers)


class StaticPortsMixin:
    """Preserve original live Ports methods; replace only catalog rebuilding."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._static_binding = StaticBinding(self.contract, self.resolved)

    def __setattr__(self, name, value):
        if name in ('_static_binding', 'static_binding') and hasattr(self, '_static_binding'):
            raise AttributeError('Original lifecycle binding cannot be replaced')
        super().__setattr__(name, value)

    @property
    def static_binding(self):
        return self._static_binding

    def __delattr__(self, name):
        if name in ('_static_binding', 'static_binding'):
            raise AttributeError('Original lifecycle binding cannot be removed')
        super().__delattr__(name)

    def inventory(self):
        writers = dict(outer=self.admission['outer_owner']['pid'], coordinator=self.coordinator['pid'])
        if self.read_owner is not None:
            writers['native'] = self.read_owner['pid']
        return self.static_binding.walk(self.contract, self.resolved, writers)
