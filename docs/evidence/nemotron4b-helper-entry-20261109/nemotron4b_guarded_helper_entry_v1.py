"""Same-process execution core for a future authenticated helper bootstrap.

No CLI, source/import authorization or serial admission. The caller must bind
the fixed installed entrypoint, complete external imports and real library
identity, and call this inside the reviewed gated wrapper process. Generated
tests use synthetic source and a synthetic importer, never the installed SDK.
"""
from contextlib import redirect_stdout
import hashlib
import sys


def require(ok, message):
    if not ok:
        raise ValueError(message)


class BoundedText:
    def __init__(self, cap):
        require(type(cap) is int and 0 < cap <= 4096, 'Bounded helper Python stdout')
        self.cap, self.parts, self.size = cap, [], 0
        self.overflowed = False

    def write(self, value):
        require(type(value) is str, 'Helper stdout writes text')
        raw = value.encode('utf-8')
        if self.overflowed or self.size+len(raw) > self.cap:
            self.overflowed = True
            raise ValueError('Helper Python stdout cap exceeded')
        self.parts.append(raw)
        self.size += len(raw)
        return len(value)

    def flush(self):
        pass

    def bytes(self):
        require(not self.overflowed, 'Failed helper stdout cannot be accepted')
        return b''.join(self.parts)


def execute_source(raw, reference, *, expected_stdout, guard, import_context):
    """Execute exact already-authenticated bytes in a temporary __main__ scope.

    import_context is a reviewed context manager factory that authenticates and
    supplies full import provenance; a local-package-only graph is insufficient.
    Parent process still caps actual OS stdout/stderr, including native writes.
    """
    require(type(raw) is bytes and 0 < len(raw) <= 16384
            and set(reference) == {'path', 'sha256'}
            and type(reference['path']) is str and reference['path'].startswith('/')
            and hashlib.sha256(raw).hexdigest() == reference['sha256'],
            'Exact complete helper source bytes')
    require(type(expected_stdout) is bytes and 0 < len(expected_stdout) <= 4096,
            'Exact expected helper stdout bytes')
    guard()
    output = BoundedText(4096)
    namespace = dict(__name__='__main__', __file__=reference['path'],
                     __package__=None, __spec__=None, __cached__=None)
    original_argv = sys.argv
    try:
        sys.argv = [reference['path']]
        with import_context() as provenance:
            guard()
            with redirect_stdout(output):
                try:
                    exec(compile(raw, reference['path'], 'exec', dont_inherit=True), namespace)
                except SystemExit as exited:
                    require(exited.code is None or type(exited.code) is int and exited.code == 0,
                            'Original helper returned nonzero SystemExit')
            guard()
        guard()
        actual = output.bytes()
        require(actual == expected_stdout, 'Exact original helper stdout path and newline')
        return dict(stdout=actual, source=dict(reference), import_provenance=provenance,
                    Python_entrypoint_completed=True, operating_system_argv_was_not_replaced=True,
                    runtime_admitted=False, full_model_qualified=False, new_speedup_claim=False)
    finally:
        sys.argv = original_argv


def run(*, source_reference, read_source, expected_stdout, library_identity,
        expected_library_identity, guard, wait_for_parent, import_context):
    """Wait for actual parent release before any helper entrypoint execution."""
    require(not any(name == 'libneuronxla' or name.startswith('libneuronxla.')
                    for name in sys.modules), 'SDK helper imported before parent gate')
    guard()
    raw = read_source(dict(source_reference))
    require(hashlib.sha256(raw).hexdigest() == source_reference['sha256'], 'Helper source authenticated before gate')
    before = library_identity()
    require(before == expected_library_identity, 'Original library identity before gate')
    released = wait_for_parent()
    guard()
    require(library_identity() == before, 'Original library identity after gate')
    result = execute_source(raw, source_reference, expected_stdout=expected_stdout,
                            guard=guard, import_context=import_context)
    require(library_identity() == before, 'Original library identity after helper execution')
    guard()
    result.update(parent_release=released, library_before=before, library_after=before,
                  library_payload_digest_recomputed=False,
                  helper_ran_in_same_gated_process=True,
                  full_runtime_source_closure_proven_by_this_module=False)
    return result


if __name__ == '__main__':
    raise SystemExit('Source preparation only; requires exact authenticated bootstrap and complete serial lifecycle')
