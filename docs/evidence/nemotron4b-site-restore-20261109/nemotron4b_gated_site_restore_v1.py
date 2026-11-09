"""Restore reviewed Python site startup inside the guarded helper context.

Source preparation only. A real caller must supply the exact startup policy
and full authenticated interpreter/import/native context. This module does not
create admission, launch children, authenticate all import origins, or remove
the observable sys.flags.no_site difference caused by the required -S gate.
"""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import sys
import time

import nemotron4b_helper_dependency_catalog_v1 as catalog
from nemotron4b_guarded_helper_entry_v2 import BoundedText, require

USED = False


def startup_inputs(policy):
    """Check the exact configured startup files and .pth set without running them."""
    from pathlib import Path
    require(set(policy) == {
        'helper_script', 'site_module', 'venv_config', 'site_packages',
        'files', 'pth_files', 'pre_path', 'post_path', 'prefix', 'base_prefix',
    }, 'Exact site startup policy fields')
    require(policy['helper_script'].startswith('/') and policy['pre_path']
            and policy['post_path'] and all(type(x) is str and x.startswith('/')
            for x in policy['pre_path']+policy['post_path']), 'Exact absolute startup paths')
    folder = Path(policy['site_packages'])
    require(folder.is_dir() and folder.resolve() == folder
            and not any(p.is_symlink() for p in (folder, *folder.parents)), 'Original site directory')
    names = sorted(str(p) for p in folder.glob('*.pth'))
    require(names == policy['pth_files'], 'Exact complete startup .pth set')
    required = {policy['site_module'], policy['venv_config'], *names}
    require(required <= set(policy['files']), 'All site/config/pth source pins required')
    verified = {}
    for path, row in policy['files'].items():
        require(set(row) == {'sha256', 'bytes', 'fullstat'}, 'Exact startup source record')
        raw, actual = catalog.source(path, row['sha256'])
        require(len(raw) == row['bytes'] and actual['fullstat'] == row['fullstat'],
                'Startup source identity and complete bytes unchanged')
        verified[path] = dict(row)
    return dict(files=verified, pth_files=names)


@contextmanager
def restore(policy, *, guard, origin_context, provenance):
    """Enter after the real parent gate and before the original helper body.

    Intended as the import_context used by guarded_helper_entry_v2.run. That
    caller sets the original helper argv before entering this context. The
    outer origin_context must authenticate the real site module and everything
    it imports. Startup side effects remain in this dedicated process; failure
    requires process exit, not a retry or pretend rollback.
    """
    global USED
    require(not USED, 'Site startup restoration is one-use per helper process')
    USED = True
    require(type(provenance) is dict and not provenance, 'Fresh mutable startup provenance')
    provenance.update(schema='nemotron4b_gated_site_restore_v1', complete=False,
        runtime_admitted=False, full_runtime_dependency_closure_proven=False,
        no_site_flag_equivalence_proven=False, new_speedup_claim=False)
    require(sys.flags.no_site == 1 and sys.flags.no_user_site == 1
            and sys.dont_write_bytecode and __debug__, 'Pre-site -S, no-user-site, -B and no optimization')
    require(sys.argv == [policy['helper_script']] and sys.path == policy['pre_path'],
            'Original helper argv and exact pre-site path')
    require(sys.prefix == sys.exec_prefix == policy['base_prefix']
            and sys.base_prefix == sys.base_exec_prefix == policy['base_prefix'],
            'Exact original pre-site base prefixes')
    require('site' not in sys.modules
            and not any(x == 'libneuronxla' or x.startswith('libneuronxla.') for x in sys.modules),
            'No site initialization or SDK import before restoration')
    guard()
    before = startup_inputs(policy)
    provenance['inputs_before'] = before
    stdout, stderr = BoundedText(4096), BoundedText(4096)
    try:
        with origin_context() as imports:
            require(type(imports) is dict, 'Caller authenticated import context returns provenance')
            guard()
            # Match the original installed console script's search directory.
            sys.path[0] = policy['helper_script'].rsplit('/', 1)[0]
            provenance['restore_start_ns'] = time.monotonic_ns()
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    import site
                    require(site.__file__ == policy['site_module'],
                            'Exact original site module origin')
                    site.main()
            finally:
                provenance['startup_stdout'] = b''.join(stdout.parts).decode('utf-8', errors='replace')
                provenance['startup_stderr'] = b''.join(stderr.parts).decode('utf-8', errors='replace')
                provenance['startup_output_overflow'] = stdout.overflowed or stderr.overflowed
            guard()
            require(stdout.bytes() == b'' and stderr.bytes() == b'',
                    'Silent startup required: site may swallow .pth/customization errors')
            require(sys.path == policy['post_path'], 'Exact restored helper import path')
            require(sys.prefix == sys.exec_prefix == policy['prefix']
                    and sys.base_prefix == sys.base_exec_prefix == policy['base_prefix'],
                    'Exact restored venv/base prefixes')
            require(site.ENABLE_USER_SITE is False and 'usercustomize' not in sys.modules,
                    'Original user-site exclusion retained')
            after = startup_inputs(policy)
            require(after == before, 'Startup source identities unchanged through initialization')
            provenance.update(inputs_after=after, actual_prefix=sys.prefix,
                actual_path=list(sys.path), no_site_flag_remains=sys.flags.no_site,
                startup_complete_ns=time.monotonic_ns(), startup_complete=True,
                import_context=imports)
            guard()
            yield provenance
            guard()
            require(startup_inputs(policy) == before, 'Startup source identities unchanged after helper')
            require(sys.path == policy['post_path']
                    and sys.prefix == sys.exec_prefix == policy['prefix']
                    and sys.base_prefix == sys.base_exec_prefix == policy['base_prefix']
                    and site.ENABLE_USER_SITE is False,
                    'Original restored environment unchanged after helper')
        guard()
        provenance['complete'] = True
    except BaseException as error:
        provenance['failure'] = dict(type=type(error).__name__, message=str(error)[:2048])
        # Preserve SystemExit and all original nonzero/failure behavior.
        raise


if __name__ == '__main__':
    raise SystemExit('Source preparation only; requires exact reviewed helper bootstrap and runtime context')
