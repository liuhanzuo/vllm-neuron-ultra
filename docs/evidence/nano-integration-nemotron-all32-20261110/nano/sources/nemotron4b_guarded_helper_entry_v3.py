"""Restore gated startup before switching argv for the original helper.

v2 changed sys.argv to the installed helper before entering import_context.
A phase startup policy expects the original wrapper argv and would reject that
order. v3 enters the caller's authenticated startup first, then executes the
same v2 bounded source core inside that still-active context. No SDK execution,
admission or native-file authentication happens at module import.
"""
from contextlib import nullcontext
import hashlib
import sys

from nemotron4b_guarded_helper_entry_v2 import execute_source
from nemotron4b_process_handshake_v1 import require


def run(*,source_reference,read_source,expected_stdout,library_identity,
        expected_library_identity,guard,wait_for_parent,startup_context):
    require(all(callable(x)for x in(read_source,library_identity,guard,wait_for_parent,startup_context)),
            'Explicit source, identity, guard, parent and startup callbacks')
    require(not {'site','torch','transformers','vllm','neuronx_distributed','libneuronxla'}&set(sys.modules)
            and not any(name.startswith('libneuronxla.')for name in sys.modules),
            'Helper startup must precede site and SDK imports before parent gate')
    original_argv=sys.argv;expected_argv=list(original_argv)
    guard();raw=read_source(dict(source_reference))
    require(type(raw)is bytes and hashlib.sha256(raw).hexdigest()==source_reference['sha256'],
            'Helper source authenticated before gate')
    before=library_identity()
    require(before==expected_library_identity,'Original library identity before gate')
    released=wait_for_parent()
    guard();require(library_identity()==before,'Original library identity after gate')
    require(sys.argv is original_argv and sys.argv==expected_argv,'Original wrapper argv before startup')
    with startup_context()as provenance:
        require(type(provenance)is dict,'Actual complete helper startup provenance')
        require(sys.argv is original_argv and sys.argv==expected_argv,
                'Original wrapper argv retained by startup')
        # The outer context authenticates imports throughout source execution.
        # nullcontext carries its same live provenance; it does not grant import
        # permission or replace the caller's installed dispatch/native policies.
        result=execute_source(raw,source_reference,expected_stdout=expected_stdout,
            guard=guard,import_context=lambda:nullcontext(provenance))
        require(sys.argv is original_argv and sys.argv==expected_argv,
                'Original wrapper argv restored before startup exit')
        require(library_identity()==before,'Original library identity after helper execution')
    require(provenance.get('complete')is True,'Actual helper startup context completed')
    guard();require(library_identity()==before,'Original library identity after startup exit')
    result.update(parent_release=released,library_before=before,library_after=before,
        startup=provenance,original_wrapper_argv_restored=True,
        library_payload_digest_recomputed=False,helper_ran_in_same_gated_process=True,
        full_runtime_source_closure_proven_by_this_module=False)
    return result


if __name__=='__main__':
    raise SystemExit('Requires actual reviewed bootstrap, parent gate and complete startup origins')
