"""Source-only retry controls with fake IO, including default-zero LAZY failure.

Extracted SDK function bodies are unchanged. No SDK/native import, compiler,
runtime metadata, graph/model read, real temporary directory or deletion occurs.
The actual runtime retry environment is intentionally not inspected or chosen.
"""
import ast
import hashlib
import io
import json
import logging
from pathlib import Path
import posixpath
import subprocess
import sys
from types import SimpleNamespace


SOURCE = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py')
EXPECTED_SOURCE_SHA256 = '33218bc87806d18277a47a2a7c5d9e7ddc78314d00f27c86816d2ec3731c961f'
DEDICATED = '/dedicated-capture-work/bucket'
FALLBACK = '/tmp/source-test-user/neuroncc_compile_workdir'


def audit(event, args):
    if event == 'open':
        path, mode, flags = args
        if str(path) != str(SOURCE) or (mode is not None and any(x in mode for x in 'wax+')):
            raise RuntimeError('Only the pinned Python source may be opened')
        if isinstance(flags, int) and flags & (1 | 2 | 64 | 512 | 1024):
            raise RuntimeError('No file writing in source-only diagnostic')
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.fork',
                 'os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.symlink', 'os.link'):
        raise RuntimeError('No process or filesystem mutation in source-only diagnostic')
    if event == 'import' and str(args[0]).split('.')[0] in (
        'torch', 'torch_neuronx', 'neuronx_distributed', 'neuronx_distributed_inference',
        'numpy', 'neuronxcc', 'libneuronxla'):
        raise RuntimeError('Native/SDK imports are forbidden')


def extracted(source):
    tree = ast.parse(source, filename=str(SOURCE))
    names = ('get_work_dir', 'compile_cache_entry')
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert tuple(node.name for node in funcs) == names
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name)
                              and target.id == 'NEURON_PARALLEL_COMPILE_MAX_RETRIES'
                              for target in node.targets))
    assert ast.unparse(assignment.value) == "int(os.environ.get('NEURON_PARALLEL_COMPILE_MAX_RETRIES', '0'))"
    return ast.Module(body=funcs, type_ignores=[])


def exercise(module, *, statuses, execution_mode='LAZY', maximum_retries=0,
             create_subdir=False, existing_paths=()):
    calls, downloads, uploads, directories, copies, removals = [], [], [], [], [], []
    statuses = list(statuses)
    existing_paths = set(existing_paths)

    class FakePath:
        def __init__(self, path): self.path = path
        def mkdir(self, **kwargs): directories.append((self.path, kwargs))

    class Entry:
        key = 'MODULE_SOURCE_TEST'
        neff_path = '/test-cache/model.neff'
        def log_exists(self): return False
        def download_hlo(self, target): downloads.append(('hlo', target))
        def download_compile_flags(self, target): downloads.append(('flags', target))
        def upload_neff(self, source): uploads.append(source)
        def upload_wrapped_neff(self, source): raise AssertionError('No wrapped file in this fixture')
        def put_log(self, text): self.error = text

    def compiler(work_dir, input_file, compile_flags, output_file, mode, **kwargs):
        calls.append(dict(work_dir=work_dir, input_file=input_file, output_file=output_file,
                          execution_mode=mode, options=kwargs))
        result = statuses.pop(0)
        if isinstance(result, BaseException): raise result
        # A real LAZY compiler failure raises; nonzero returns model EAGER only.
        assert mode == 'EAGER' or result == 0
        return result

    def fake_open(path, mode):
        assert mode == 'r' and path.endswith('.json')
        return io.StringIO('[]')

    noop = lambda *args, **kwargs: None
    scope = {
        'os': SimpleNamespace(path=SimpleNamespace(join=posixpath.join, exists=lambda p: p in existing_paths),
                              getenv=lambda key, default=None: 'source-test-user' if key == 'USER' else default),
        'uuid': SimpleNamespace(uuid4=lambda: 'source-test-uuid'),
        'Path': FakePath,
        'shutil': SimpleNamespace(rmtree=removals.append, copyfile=lambda a, b: copies.append((a, b))),
        'ExecutionMode': SimpleNamespace(LAZY='LAZY', EAGER='EAGER'),
        'NEURON_PARALLEL_COMPILE_MAX_RETRIES': maximum_retries,
        'WRAPPED_NEFF_FILE_NAME': 'wrapped_neff.hlo',
        'LOGGER': SimpleNamespace(debug=noop, info=noop, warning=noop, error=noop),
        'logger': SimpleNamespace(get_verbosity_from_flags=lambda _: logging.WARNING),
        'logging': logging, 'json': json, 'open': fake_open,
        'subprocess': SimpleNamespace(CalledProcessError=subprocess.CalledProcessError),
        'call_neuron_compiler': compiler,
    }
    exec(compile(module, str(SOURCE), 'exec'), scope)
    entry = Entry()
    error = None
    try:
        result = scope['compile_cache_entry'](
            '/temporary-output/model.neff', entry, work_dir=DEDICATED,
            execution_mode=execution_mode, create_subdir=create_subdir, framework='XLA', dump=None)
    except subprocess.CalledProcessError as exc:
        result, error = None, dict(returncode=exc.returncode, stderr=exc.stderr)
    assert not statuses
    return dict(result=result, error=error, calls=calls, downloads=downloads,
                uploads=uploads, fake_directories=directories, fake_removals=removals,
                copies=copies, maximum_retries=maximum_retries, execution_mode=execution_mode,
                cached_error=getattr(entry, 'error', None))


def failure():
    return subprocess.CalledProcessError(1, 'fake', stderr='synthetic')


def paths(case):
    return [call['work_dir'] for call in case['calls']]


def main():
    sys.addaudithook(audit)
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SOURCE_SHA256
    module = extracted(raw.decode())
    first = exercise(module, statuses=[0])
    assert paths(first) == [DEDICATED] and first['result'] == (True, 0)
    default_failure = exercise(module, statuses=[failure()])
    assert paths(default_failure) == [DEDICATED]
    assert default_failure['error'] == dict(returncode=1, stderr='synthetic')
    assert default_failure['cached_error'] == 'synthetic'
    retry = exercise(module, statuses=[failure(), 0], maximum_retries=2)
    assert paths(retry) == [DEDICATED, FALLBACK] and retry['result'] == (True, 1)
    eager_retry = exercise(module, statuses=[1, 0], execution_mode='EAGER', maximum_retries=2)
    assert paths(eager_retry) == [DEDICATED, FALLBACK] and eager_retry['result'] == (True, 1)
    subdir = exercise(module, statuses=[failure(), 0], maximum_retries=2, create_subdir=True)
    assert paths(subdir) == [path + '/source-test-uuid' for path in (DEDICATED, FALLBACK)]
    terminal = exercise(module, statuses=[failure(), failure(), failure()], maximum_retries=2)
    assert terminal['error'] == dict(returncode=1, stderr='synthetic')
    assert paths(terminal) == [DEDICATED, FALLBACK, FALLBACK]
    existing_fallback = exercise(module, statuses=[failure(), 0], maximum_retries=2,
                                 existing_paths=(FALLBACK,))
    assert paths(existing_fallback) == [DEDICATED, FALLBACK]
    assert existing_fallback['fake_removals'] == [FALLBACK]
    eager_zero = exercise(module, statuses=[1, 0], execution_mode='EAGER', maximum_retries=0)
    assert paths(eager_zero) == [DEDICATED, FALLBACK] and eager_zero['result'] == (True, 1)
    cases = dict(default_zero_first_success=first, default_zero_lazy_exception_no_retry=default_failure,
                 enabled_lazy_exception_retry=retry, enabled_eager_nonzero_retry=eager_retry,
                 enabled_lazy_subdirectory_retry=subdir, enabled_lazy_exhausted_exception_retry=terminal,
                 enabled_lazy_existing_fallback_removal=existing_fallback,
                 default_zero_eager_nonzero_still_recurses=eager_zero)
    assert all(call['options'] == dict(framework='XLA', dump=None)
               for case in cases.values() for call in case['calls'])
    assert all(not case['fake_removals'] for name, case in cases.items()
               if name != 'enabled_lazy_existing_fallback_removal')
    print(json.dumps(dict(
        schema='deepseek_compile_retry_source_reproduction_v2', passed=True, checks=len(cases),
        source=dict(path=str(SOURCE), sha256=EXPECTED_SOURCE_SHA256),
        extracted_functions=[node.name for node in module.body],
        original_function_bodies_unchanged=True, cases=cases,
        retry_default_verified_from_source_ast=0, runtime_retry_environment_inspected=False,
        actual_native_imports=False, actual_compilation=False,
        actual_runtime_metadata_or_payload_read=False, actual_tmp_writes=False,
        actual_current_process_or_lock_inspection=False, actual_deletion=False,
        new_speedup_claim=False), indent=2, sort_keys=True))


if __name__ == '__main__': main()
