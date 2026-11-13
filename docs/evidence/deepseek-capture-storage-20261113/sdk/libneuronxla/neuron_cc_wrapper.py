# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
from enum import Enum
import json
import logging
import os
from pathlib import Path
import tempfile
import shlex
import shutil
import subprocess
import uuid
import warnings

import libneuronxla.logger as logger
from libneuronxla.neuron_cc_cache import CacheUrl, create_compile_cache, NEURON_COMPILE_CACHE_URL_INFO
from typing import List, Union


# eager mode configuration
NEURONCC_DAEMON_INACTIVITY_TIMEOUT = 300
# compilation error handling
NEFF_ERROR = "ERROR"
NEURON_PARALLEL_COMPILE_MAX_RETRIES = int(
    os.environ.get("NEURON_PARALLEL_COMPILE_MAX_RETRIES", "0"))
# info
NEURONCC_WRAPPER_ENV_INFO = f"""
    {NEURON_COMPILE_CACHE_URL_INFO}
    NEURON_PARALLEL_COMPILE_MAX_RETRIES: By default the utility would not retry a
     failed compilation. To retry, you can set the NEURON_PARALLEL_COMPILE_MAX_RETRIES
     env variable to an integer value."""

LOGGER = logger.get_logger("NEURON_CC_WRAPPER")

# Wrapped Neff file name for WLO
WRAPPED_NEFF_FILE_NAME = "wrapped_neff.hlo"


class ExecutionMode(Enum):
    LAZY = 0
    EAGER = 1


class GlobalCounter:
    _counter = 0

    def __call__(self):
        count = GlobalCounter._counter
        GlobalCounter._counter += 1
        return count


def call_neuron_compiler(work_dir, input_file, compile_flags,
                         output_file, execution_mode=ExecutionMode.LAZY,
                         framework="XLA", dump=None):
    cmd = ["neuronx-cc",
           "compile",
           f"--framework={framework}",
           input_file,
           "--output",
           output_file, ] + compile_flags

    with tempfile.TemporaryDirectory() as tmpdir:
        if dump is not None:
            tmpdir = os.path.abspath(dump)
            tmpdir = os.path.join(
                tmpdir, f'pid{os.getpid()}-program{GlobalCounter()()}')
            os.makedirs(tmpdir, exist_ok=True)
            cmd.extend(['--pipeline', 'compile', 'SaveTemps'])
            ver_cmd = ['neuronx-cc', '--version']
            ncc_version = subprocess.check_output(
                ver_cmd, stderr=subprocess.STDOUT).decode()
            ncc_version, *_ = ncc_version.split('\n')
            *_, ncc_version = ncc_version.split('version ')
            with open(os.path.join(tmpdir, 'neuronx_cc_metadata.json'), 'w') as fp:
                json.dump([ncc_version, cmd], fp)

    if execution_mode == ExecutionMode.EAGER:
        # TODO: move this import to the top of the file once the daemon change
        # in neuroncc reaches beta.
        try:
            from neuronxcc.cli.Client import Client
        except ImportError:
            from neuroncc.cli.Client import Client

        client = Client(
            pid_file=os.path.join(work_dir, "neuroncc_daemon_logs.txt"),
            log_file=os.path.join(work_dir, "neuron.pid"),
            inactivity_timeout=NEURONCC_DAEMON_INACTIVITY_TIMEOUT,
        )

        cmd = [f"--framework={framework}",
               input_file,
               "--output",
               output_file,
               ] + compile_flags + ["-O0",
                                    "--internal-tensorizer-opt-level=eager"]

        LOGGER.info(f"Call compiler client with cmd: {' '.join(cmd)}")
        LOGGER.debug("Compile: Start compilation as eager mode")
        exit_code = client.run(cmd)
        return exit_code
    else:
        # Modify the environment to remove libtcmalloc from LD_PRELOAD
        env = os.environ.copy()
        ld_preload = env.get('LD_PRELOAD', '')
        if 'libtcmalloc' in ld_preload:
            # Remove libtcmalloc
            updated_ld_preload = ':'.join(
                path for path in ld_preload.split(':') if 'libtcmalloc' not in path
            )
            env['LD_PRELOAD'] = updated_ld_preload
        save_compile_command(cmd, work_dir)
        res = subprocess.run(cmd, stderr=subprocess.PIPE, cwd=work_dir, env=env)
        if res.returncode != 0:
            from libneuronxla.libncc import _clean_stderr_message
            clean_error = _clean_stderr_message(res.stderr.decode())
            error_with_cmd = f"Failed compilation with {cmd}:\n{clean_error}"
            LOGGER.error(error_with_cmd)
            raise subprocess.CalledProcessError(res.returncode, cmd, stderr=error_with_cmd)
        return res.returncode


def save_compile_command(command: List[str], workdir: Union[str, Path]):
    """
    Saves the compiler command as command.txt in the workdir.
    """
    command_filename = os.path.join(workdir, "command.txt")
    with open(command_filename, "w") as f:
        # Strip file paths from the command.
        stripped = []
        for item in command:
            item = str(item)
            if "=" in item:
                item_parts = item.split("=", 1)
                stripped.append(item_parts[0] + "=" + os.path.basename(item_parts[1]))
            else:
                stripped.append(os.path.basename(item))
        f.write(shlex.join(stripped))


def get_work_dir(work_dir, create_subdir=True):
    """
    This function creates a subdirectory by default in work_dir to avoid multiple processes
    accessing the same folder. Creation of subdirectory can be skipped by setting create_subdir
    as `False` and in that case, it is the resposibility of caller to ensure unique work_dir is
    passed to this function for every compilation.
    """
    if work_dir is None:
        work_dir = f"/tmp/{os.getenv('USER', 'no-user')}/neuroncc_compile_workdir"
    if create_subdir:
        work_dir = os.path.join(work_dir, str(uuid.uuid4()))
    if os.path.exists(work_dir):
        LOGGER.debug("Removing existing workdir {}".format(work_dir))
        shutil.rmtree(work_dir)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    return work_dir


def compile_cache_entry(output_neff, entry, execution_mode=ExecutionMode.LAZY, retry=0,
                        platform_target="trn1", framework="XLA", dump=None, work_dir=None,
                        retry_failed_compilation=False, create_subdir=True,
                        output_wrapped_neff=None):
    work_dir = get_work_dir(work_dir, create_subdir)

    # First see if there's a cached failure
    if entry.log_exists():
        error_log = entry.get_log()
        if retry_failed_compilation:
            entry.remove_log()
            LOGGER.info(
                f"Got a cached failed neff at {entry.neff_path}. With eror log: [{error_log}]. \
                Retrying...")
        else:
            LOGGER.error(f"Got a cached failed neff at {entry.neff_path}. Will skip compilation, please set "
                         f"--retry_failed_compilation for recompilation: \n {error_log}.")
            # raise exception to upper level
            raise (subprocess.CalledProcessError(-1, "", stderr=error_log))

    tmp_model_path = os.path.join(work_dir, f"model.{entry.key}.hlo_module.pb")
    tmp_compile_flags_path = os.path.join(work_dir, f"compile_flags.{entry.key}.json")
    tmp_neff_path = os.path.join(work_dir, f"model.{entry.key}.neff")
    # This file is required for Weight Layout Optimization
    tmp_wrapped_neff_path = os.path.join(work_dir, WRAPPED_NEFF_FILE_NAME)

    # Download graph and flags
    entry.download_hlo(tmp_model_path)
    entry.download_compile_flags(tmp_compile_flags_path)

    with open(tmp_compile_flags_path, 'r') as f:
        compile_flags = json.loads(f.read())

    # Display progress dots rather lots of compiler warnings
    found_user_verbose = False
    for arg in compile_flags:
        if "verbose" in arg:
            found_user_verbose = True
    if not found_user_verbose and execution_mode != ExecutionMode.EAGER:
        # Compiler in eager mode doesn't accept verbose flag
        compile_flags.append("--verbose=35")

    LOGGER.debug(f"Compilation starts for {tmp_model_path}.")
    try:
        ret = call_neuron_compiler(
            work_dir,
            tmp_model_path,
            compile_flags,
            tmp_neff_path,
            execution_mode, framework=framework, dump=dump)
        if ret == 0:
            verbosity_level = logger.get_verbosity_from_flags(compile_flags)
            # Only log success if verbosity allows INFO level (INFO=20, WARNING=30, ERROR=40)
            # Lower values = more verbose, so INFO and below should show the message
            if verbosity_level <= logging.WARNING:
                LOGGER.info(
                    f"Compilation Successfully Completed for {tmp_model_path.split('/')[-1]}")
            # populate cache
            entry.upload_neff(tmp_neff_path)
            if tmp_neff_path != output_neff:
                shutil.copyfile(tmp_neff_path, output_neff)
            if os.path.exists(tmp_wrapped_neff_path):
                entry.upload_wrapped_neff(tmp_wrapped_neff_path)
                output_wrapped_hlo = output_wrapped_neff
                if tmp_wrapped_neff_path != output_wrapped_hlo:
                    shutil.copyfile(tmp_wrapped_neff_path, output_wrapped_hlo)
            return True, retry
    except subprocess.CalledProcessError as e:
        if retry == NEURON_PARALLEL_COMPILE_MAX_RETRIES:
            # save compilation error
            entry.put_log(e.stderr)
            # raise exception to upper layer
            raise (e)

    # retry compilation
    return compile_cache_entry(output_neff, entry, execution_mode=execution_mode, retry=retry + 1,
                               platform_target=platform_target, framework=framework, dump=dump,
                               create_subdir=create_subdir, output_wrapped_neff=output_wrapped_neff)


def neuron_xla_compile(module_bytes, compiler_flags, input_format="hlo", platform_target="trn1",
                       cache_key=None, retry_failed_compilation=False,
                       lazy=True, use_cache=True, cache_dir=None, dump=None,
                       work_dir=f"/tmp/{os.getenv('USER', 'no-user')}/neuroncc_compile_workdir",
                       create_subdir=True):
    if isinstance(compiler_flags, str):
        warnings.warn(SyntaxWarning(
            "str format compiler_flags is discouraged as its handling involves repeated joining and splitting, which "
            "can easily make mistakes if something is quoted or escaped. Use list[str] instead. Refer to documentation "
            "of the Python subprocess module for details."))
        compiler_flags = shlex.split(compiler_flags)
    with tempfile.TemporaryDirectory() as tmpdirname:
        input_file = os.path.join(tmpdirname, "model.hlo")

        with open(input_file, 'wb') as f:
            f.write(module_bytes)

        output = os.path.join(tmpdirname, "model.neff")
        neuron_xla_compile_impl(
            input_file, compiler_flags, output,
            cache_key=cache_key, retry_failed_compilation=retry_failed_compilation,
            lazy=lazy, use_cache=use_cache, cache_dir=cache_dir, platform_target=platform_target,
            dump=dump, work_dir=work_dir, create_subdir=create_subdir,
        )
        if os.path.isfile(output):
            with open(output, 'rb') as f:
                return f.read()

    return b""


def neuron_xla_wlo_compile(module_bytes, compiler_flags, input_format="hlo", platform_target="trn1",
                           cache_key=None, retry_failed_compilation=False,
                           lazy=True, use_cache=True, cache_dir=None, dump=None,
                           work_dir=f"/tmp/{os.getenv('USER', 'no-user')}/neuroncc_compile_workdir",
                           create_subdir=False):
    """
    This API will be used for Weight Layout Optimization (WLO) compilations. There is an additional file,
    `wrapped_neff.hlo` for WLO compilations which will not be returned from neuron_xla_compile(). This file
    is used to generate layouts from the compilation outputs and required to optimize the layouts during model build.
    Another difference here is, create_subdir is False which avoids creation of subdirectories inside the work_dir.
    So, it is the resposibility of caller to ensure there are unique work_dir passed to this function.
    """
    if isinstance(compiler_flags, str):
        warnings.warn(SyntaxWarning(
            "str format compiler_flags is discouraged as its handling involves repeated joining and splitting, which "
            "can easily make mistakes if something is quoted or escaped. Use list[str] instead. Refer to documentation "
            "of the Python subprocess module for details."))
        compiler_flags = shlex.split(compiler_flags)
    with tempfile.TemporaryDirectory() as tmpdirname:
        input_file = os.path.join(tmpdirname, "model.hlo")

        with open(input_file, 'wb') as f:
            f.write(module_bytes)

        output = os.path.join(tmpdirname, "model.neff")
        wrapped_neff_path = os.path.join(tmpdirname, WRAPPED_NEFF_FILE_NAME)
        neuron_xla_compile_impl(
            input_file, compiler_flags, output,
            cache_key=cache_key, retry_failed_compilation=retry_failed_compilation,
            lazy=lazy, use_cache=use_cache, cache_dir=cache_dir, platform_target=platform_target,
            dump=dump, work_dir=work_dir, create_subdir=create_subdir,
            output_wrapped_neff=wrapped_neff_path
        )
        if not os.path.isfile(output):
            raise FileNotFoundError(f"File {output} not found")
        neff_bytes = b""
        wrapped_neff_bytes = b""
        with open(output, 'rb') as f:
            neff_bytes = f.read()
        if os.path.isfile(wrapped_neff_path):
            with open(wrapped_neff_path, 'rb') as f:
                wrapped_neff_bytes = f.read()
        else:
            LOGGER.debug(f"{wrapped_neff_path} file not found, "
                         "weights are already in optimal layout")

        return neff_bytes, wrapped_neff_bytes

    return b"", b""


def neuron_xla_compile_impl(input_file, compiler_flags, output, cache_key=None, retry_failed_compilation=False,
                            lazy=True, use_cache=True, cache_dir=None, platform_target="trn1", framework="XLA",
                            dump=None, work_dir=None, create_subdir=True, output_wrapped_neff=None):
    execution_mode = ExecutionMode.EAGER
    if lazy:
        execution_mode = ExecutionMode.LAZY

    # Ensure --target flag is set by user via NEURON_CC_FLAGS (higher precedence)
    # or automatically based on runtime platform_version code (pjrt/neuron/nrt_adaptor.cc)
    LOGGER.debug("Original compiler flags:", compiler_flags)

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--target')
    target_flag, compiler_flags = parser.parse_known_args(compiler_flags)
    if not target_flag.target:
        target_flag.target = platform_target
    # To ensure consistent persistent cache hash key, move target flag to beginning
    compiler_flags.insert(0, f"--target={target_flag.target}")

    LOGGER.debug("Compiler flags with target platform:", compiler_flags)

    # Check if the cache is switched on, if no, we run the neuronx-cc command
    # as is and return the exit code.
    if not use_cache:
        return call_neuron_compiler(work_dir, input_file, compiler_flags, output, execution_mode,
                                    framework=framework, dump=dump)

    # cache_key has be provided for cache
    assert cache_key is not None, "No cache key provided."

    # create compile cache
    cache_url = CacheUrl.get_cache_url(cache_dir=cache_dir)
    compile_cache = create_compile_cache(cache_url)

    # Look in the cache
    compile_flags_str = json.dumps(compiler_flags)
    entry = compile_cache.lookup(cache_key, compiler_flags)

    # Skip compiling if we're just extracting graphs for neuron_parallel_compile
    extract_graphs_only = os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", "0") == "1"
    if extract_graphs_only:
        if not entry.exists:
            # It's ok to do this without holding the entry lock, since both
            # implementations of `upload_inputs()` are atomic.
            entry.upload_inputs(input_file, compile_flags_str)

        LOGGER.info(
            f"Extracting graphs ({entry.hlo_path}) for ahead-of-time parallel compilation."
            " No compilation was done.")
        return 0

    # Otherwise normal cache logic applies: we use the entry as a context
    # manager to take a lock if required, and get either an existing NEFF or
    # the responsibility to create one.
    with entry:
        if entry.exists:
            entry.download_neff(output)
            LOGGER.info(f"Using a cached neff at {entry.neff_path}")
            if output_wrapped_neff:
                entry.download_wrapped_neff(output_wrapped_neff)
            return 0

        # This graph doesn't have a NEFF in the cache yet, and we're holding the lock for it
        # First make sure the inputs are in the cache
        entry.upload_inputs(input_file, compile_flags_str)

        # It's go time
        return compile_cache_entry(output, entry, execution_mode,
                                   platform_target=platform_target,
                                   framework=framework, dump=dump,
                                   work_dir=work_dir,
                                   retry_failed_compilation=retry_failed_compilation,
                                   create_subdir=create_subdir,
                                   output_wrapped_neff=output_wrapped_neff)
