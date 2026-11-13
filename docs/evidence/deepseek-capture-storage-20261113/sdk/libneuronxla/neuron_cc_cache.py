from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from urllib.parse import urlparse
import boto3
import botocore
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import time
import random

import libneuronxla.logger as logger

DEFAULT_FS_CACHE_PATH = "/var/tmp/neuron-compile-cache"
CACHE_STRUCTURE_INFO = """
    The cache dir would have the following structure:
    <cache_url>/
        neuronxcc_<neuronxcc_version>/
                MODULE_<hlo_hash>+<flag_hash>/
                    model.hlo_module.pb
                    compile_flags.json
                    model.neff
                    model.done
"""
NEURON_COMPILE_CACHE_URL_INFO = f"""NEURON_COMPILE_CACHE_URL:
     If not set, the default cache dir is {DEFAULT_FS_CACHE_PATH}.
     If url starts with s3://, it will try to s3 as cache backend, and local file system otherwise."""

LOGGER = logger.get_logger("NEURON_CACHE")


@contextmanager
def os_umask_zero():
    oldmask = os.umask(000)
    try:
        yield
    finally:
        os.umask(oldmask)


def file_opener(path, flags):
    return os.open(path, flags, 0o666)


def get_cache_version_dir():
    try:
        import neuronxcc
    except ImportError:
        import neuroncc as neuronxcc
    neuronxcc_version = neuronxcc.__version__
    # We do not need the FAL version to define the uniqueness. If the
    # HLO modules are same, their hash would be same in this case.
    # fal_version = _get_fal_version(FAL)
    return f"neuronxcc-{neuronxcc_version}"


class CacheUrl:
    def __init__(self, url, url_type="s3"):
        self.url = url
        self.url_type = url_type

    def is_fs(self):
        return self.url_type == "fs"

    def is_s3(self):
        return self.url_type == "s3"

    @classmethod
    def get_cache_url(cls, cache_dir=None):
        compile_cache_path = cache_dir

        if compile_cache_path is None:
            compile_cache_path = os.environ.get("NEURON_COMPILE_CACHE_URL", None)

        url_type = "fs"

        if compile_cache_path is None:
            compile_cache_path = DEFAULT_FS_CACHE_PATH
        if compile_cache_path.startswith("s3://"):
            url_type = "s3"
            compile_cache_path = compile_cache_path.strip('/')

        LOGGER.debug(f"Compile cache path: {compile_cache_path}")

        # Save the compile cache path for Neuron Parallel Compile
        parallel_compile_cache_path = os.environ.get(
            "NEURON_PARALLEL_COMPILE_CACHE_PATH_LOG", None)
        if parallel_compile_cache_path:
            try:
                # parallel_compile_cache_path is under per user workdir
                # no need to apply os_umask_zero
                with open(parallel_compile_cache_path, 'w') as f:
                    f.write(compile_cache_path)
            except BaseException:
                pass

        cache_url = cls(compile_cache_path, url_type=url_type)
        return cache_url


class CacheEntry:
    def __init__(self, cache, key, locked=False):
        self.cache = cache
        self.key = key
        self.locked = locked
        self.dir = cache.get_cache_dir_with_cache_key(key)
        self.hlo_path = f"{self.dir}/model.hlo_module.pb"
        self.wrapped_neff = f"{self.dir}/wrapped_neff.hlo"
        self.neff_path = f"{self.dir}/model.neff"
        self.compile_flags_path = f"{self.dir}/compile_flags.json"
        self.log_path = f"{self.dir}/model.log"
        self.done_path = f"{self.dir}/model.done"

        self.exists = cache.exists(self.done_path)

    def _wait_for_lock(self):
        total_waited_time_s = 0
        sleep_time_s = 5
        log_every_s = 60
        while True:
            if self.cache.hlo_acquire_lock(self.hlo_path):
                self.locked = True
                return
            time.sleep(sleep_time_s)
            total_waited_time_s += sleep_time_s
            # Don't print message at time 0
            if total_waited_time_s % log_every_s == 0:
                LOGGER.info(
                    f"Another process must be compiling {self.hlo_path}, "
                    f"been waiting for: {total_waited_time_s/60} minutes")

    def __enter__(self):
        if self.exists:
            return self
        if self.locked:
            return self
        # acquire lock if neff doesn't already exist
        LOGGER.debug(f"Acquiring lock on {self.hlo_path}")
        self._wait_for_lock()
        # See if another process successfully compiled while we were waiting for the lock
        self.exists = self.cache.exists(self.done_path)
        if self.exists:
            self.cache.hlo_release_lock(self.hlo_path)
            self.locked = False
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.locked:
            self.cache.hlo_release_lock(self.hlo_path)

    def download_neff(self, dst_path):
        self.cache.download_file(self.neff_path, dst_path)

    def upload_neff(self, src_path):
        # Note: upload_file() args are the reverse order you'd expect
        self.cache.upload_file(self.neff_path, src_path)
        self.cache.upload_string_to_file(self.done_path, '')

    def download_wrapped_neff(self, dst_path):
        # If the weights are already in optimal layout,
        # then wrapped_neff.hlo file is not generated.
        if self.cache.exists(self.wrapped_neff):
            self.cache.download_file(self.wrapped_neff, dst_path)

    def upload_wrapped_neff(self, src_path):
        # Note: upload_file() args are the reverse order you'd expect
        self.cache.upload_file(self.wrapped_neff, src_path)

    def upload_inputs(self, model_path, compile_flags_str):
        self.cache.upload_file(self.hlo_path, model_path)
        self.cache.upload_string_to_file(self.compile_flags_path, compile_flags_str)

    def download_hlo(self, dst_path):
        self.cache.download_file(self.hlo_path, dst_path)

    def download_compile_flags(self, dst_path):
        self.cache.download_file(self.compile_flags_path, dst_path)

    def log_exists(self):
        return self.cache.exists(self.log_path)

    def get_log(self):
        self.cache.download_file_to_string(self.log_path)

    def put_log(self, log_str):
        self.cache.upload_string_to_file(self.log_path, log_str)

    def remove_log(self):
        self.cache.remove(self.log_path)


class CompileCache:
    def __init__(self, cache_url):
        self.cache_url = cache_url

    def lookup(self, model_hash, compiler_flags):
        """Lookup a model in the cache (with the given compiler flags).

        Returns a context manager in one of two possible states:
         * Either the NEFF already exists (has previously been successfully compiled),
            and you can retreive it by calling the download_neff() method on the context manager
         * Or the NEFF does not exist (either a previously unseen model or an earlier compilation failed),
            and now the context manager holds an exclusive lock on the model. You can access the log of
            previous compilation attempts with get_log(), and/or populate the cache with upload_neff()
            or put_log().

        The context manager will automatically release the lock held in the second case.
        """
        # Ensure the directory exists
        self.get_cache_dir(model_hash, compiler_flags)
        key = CompileCache.get_cache_key(model_hash, compiler_flags)
        return CacheEntry(self, key)

    def try_lookup_for_compile(self, model_path):
        """Look for a model in the cache at the given path. It it's already done
        or locked, return None, otherwise return the cache entry for it.
        """
        if self.is_done(model_path):
            return None
        if self.hlo_acquire_lock(model_path):
            cache_key = self._get_cache_key_from_model_path(model_path)
            return CacheEntry(self, cache_key, locked=True)
        return None

    def _get_cache_key_from_model_path(self, model_path):
        cache_key = re.sub("/model.hlo_module.pb", "", model_path)
        cache_key = re.sub(".*/", "", cache_key)
        return cache_key

    def get_cache_dir_with_cache_key(self, cache_key):
        version_dir_name = get_cache_version_dir()
        return f"{self.cache_path}/{version_dir_name}/{cache_key}"

    def generate_lock_file_name(self, name):
        return name + ".lock"

    def is_done(self, hlo):
        return self.exists(hlo.replace(".hlo_module.pb", ".done"))

    @staticmethod
    def get_compiler_flags_hash(compiler_flags):
        compile_flags_str = json.dumps(compiler_flags)
        return str(hashlib.md5(compile_flags_str.encode()).hexdigest())[:8]

    @staticmethod
    def get_cache_key(model_hash, compiler_flags):
        return f'MODULE_{model_hash}+{CompileCache.get_compiler_flags_hash(compiler_flags)}'

    def _get_cache_dir(self, model_hash, compiler_flags):
        cache_key = CompileCache.get_cache_key(model_hash, compiler_flags)
        output_parent_dir = self.get_cache_dir_with_cache_key(cache_key)
        LOGGER.debug(f"Compile cache path: {output_parent_dir}")
        return output_parent_dir, cache_key

    def clear_locks(self):
        self.clear_with_suffix(".lock")

    def clear_neffs(self):
        self.clear_with_suffix(".neff")

    def download_artifacts(self, parent_dir, output_dir):
        """HACK: download cache artifacts from a given parent_dir to output_dir

        This method should not be used, and only exists to maintain compatibility
        with released versions of TNX.
        """
        files = ["model.hlo_module.pb", "model.neff", "model.log", "compile_flags.json"]
        for file in files:
            src_path = os.path.join(parent_dir, file)
            if self.exists(src_path):
                self.download_file(src_path, os.path.join(output_dir, file))


class CompileCacheFs(CompileCache):
    cache_type = "fs"

    def __init__(self, cache_url):
        super().__init__(cache_url)
        self.cache_path = os.path.abspath(cache_url.url)
        with os_umask_zero():
            Path(self.cache_path).mkdir(parents=True, exist_ok=True)

    def get_cache_dir(self, model_hash, compiler_flags):
        output_parent_dir, cache_key = self._get_cache_dir(model_hash, compiler_flags)
        with os_umask_zero():
            Path(output_parent_dir).mkdir(parents=True, exist_ok=True)
        return output_parent_dir, cache_key

    def clean(self):
        shutil.rmtree(self.cache_path)

    def clear_with_suffix(self, suffix):
        prefix = self.cache_path + '/' + get_cache_version_dir()
        files = glob.glob(prefix + f'/*/*{suffix}')
        for file in files:
            self.remove(file)

    def get_hlos(self, failed_neff_str=None):
        prefix = self.cache_path + '/' + get_cache_version_dir()
        files = glob.glob(prefix + '/*/model.*')

        hlos = set()
        locked = set()
        done = set()
        failed = set()
        for f in files:
            if f.endswith(".hlo_module.pb"):
                hlos.add(f)
            if f.endswith(".hlo_module.pb.lock"):
                hlo_path = f.replace(".hlo_module.pb.lock", ".hlo_module.pb")
                locked.add(hlo_path)
            if f.endswith(".done"):
                hlo_path = f.replace(".done", ".hlo_module.pb")
                done.add(hlo_path)
            if f.endswith(".log"):
                hlo_path = f.replace(".log", ".hlo_module.pb")
                failed.add(hlo_path)

        filtered_hlos = hlos - done
        filtered_hlos = filtered_hlos - locked
        filtered_hlos = filtered_hlos - failed

        LOGGER.info(f"Current remaining items are {len(filtered_hlos)}, locked are {len(locked)}, "
                    f"failed are {len(failed)}, done are {len(done)}, total is {len(hlos)}")

        return filtered_hlos, locked, done, failed

    def hlo_acquire_lock(self, h):
        lock_file_name = self.generate_lock_file_name(h)
        with os_umask_zero():
            try:
                f = open(lock_file_name, 'x')
                f.close()
            except FileExistsError:
                return False
        return True

    def hlo_release_lock(self, h):
        lock_file_name = self.generate_lock_file_name(h)
        os.remove(lock_file_name)
        return True

    def remove(self, path):
        return os.remove(path)

    def exists(self, path):
        return os.path.exists(path)

    def download_file(self, cache_path, dst_path):
        max_retries = int(os.environ.get("NEURON_COMPILE_CACHE_CP_MAX_RETRIES", "5"))
        backoff_in_seconds = 1
        for attempt in range(max_retries):
            # get from cache
            try:
                shutil.copy(cache_path, dst_path)
            except OSError as e:
                # P210377908: retry copy to work-around "No data available" when cache is on FSx mount
                if e.errno == 61 and attempt < max_retries - 1:
                    wait_time = backoff_in_seconds * (2 ** attempt)
                    jitter = random.uniform(0, wait_time)
                    total_wait = wait_time + jitter

                    LOGGER.warning("WARNING: Hit OSError(61) when copying file from cache "
                                   f"(usually with FSx). Attempt {attempt + 1}/{max_retries}. "
                                   f"Retrying {cache_path} -> {dst_path} "
                                   f"waiting {total_wait:.2f} seconds")
                    time.sleep(total_wait)
                else:
                    raise e

    @cached_property
    def _unique_suffix(self):
        import socket
        hostname = socket.gethostname()
        pid = os.getpid()
        return f'{hostname}-{pid}'

    def upload_file(self, cache_path, src_path):
        # atomic put into cache
        tempname = f'{cache_path}-{self._unique_suffix}'
        shutil.copyfile(src_path, tempname)
        os.chmod(tempname, 0o666)
        os.replace(tempname, cache_path)

    def upload_string_to_file(self, cache_path, data):
        with os_umask_zero():
            with open(cache_path, 'w', opener=file_opener) as f:
                f.write(data)

    def download_file_to_string(self, cache_path, limit=None):
        with os_umask_zero():
            with open(cache_path, 'rb', opener=file_opener) as f:
                if limit is None:
                    s = f.read().decode(errors='replace')
                else:
                    s = f.read(limit).decode(errors='replace')
        return s


class CompileCacheS3(CompileCache):
    cache_type = "s3"

    def __init__(self, cache_url):
        super().__init__(cache_url)
        self.cache_path = cache_url.url
        self.s3_bucket, self.s3_key = self.s3_path_to_bucket_and_key(
            cache_url.url)

        self.s3 = boto3.client('s3')
        self.bucket = boto3.resource('s3').Bucket(self.s3_bucket)
        logging.getLogger('boto3').setLevel(logging.CRITICAL)

    def get_cache_dir(self, cache_key, compile_flags_str):
        return self._get_cache_dir(cache_key, compile_flags_str)

    def clean(self):
        s3_objects = self.bucket.objects.filter(Prefix=self.s3_key)
        s3_objects.delete()

    def clear_with_suffix(self, suffix):
        if self.s3_key == '':
            prefix = get_cache_version_dir()
        else:
            prefix = self.s3_key + '/' + get_cache_version_dir()

        for obj in self.bucket.objects.filter(Prefix=prefix):
            key = obj.key
            if key.endswith(suffix):
                s3_lock_path = "s3://" + self.s3_bucket + '/' + key
                self.remove(s3_lock_path)

    def get_hlos(self, failed_neff_str=None):
        if self.s3_key == '':
            prefix = get_cache_version_dir()
        else:
            prefix = self.s3_key + '/' + get_cache_version_dir()

        hlos = set()
        locked = set()
        done = set()
        failed = set()
        for obj in self.bucket.objects.filter(Prefix=prefix):
            key = obj.key
            if key.endswith(".hlo_module.pb"):
                s3_hlo_path = "s3://" + self.s3_bucket + '/' + key
                hlos.add(s3_hlo_path)
            if key.endswith(".hlo_module.pb.lock"):
                s3_lock_path = "s3://" + self.s3_bucket + '/' + key
                s3_hlo_path = s3_lock_path.replace(".hlo_module.pb.lock", ".hlo_module.pb")
                locked.add(s3_hlo_path)
            if key.endswith(".done"):
                s3_done_path = "s3://" + self.s3_bucket + '/' + key
                s3_hlo_path = s3_done_path.replace(".done", ".hlo_module.pb")
                done.add(s3_hlo_path)
            if key.endswith(".log"):
                s3_log_path = "s3://" + self.s3_bucket + '/' + key
                s3_hlo_path = s3_log_path.replace(".log", ".hlo_module.pb")
                failed.add(s3_hlo_path)

        filtered_hlos = hlos - done
        filtered_hlos = filtered_hlos - locked
        filtered_hlos = filtered_hlos - failed

        LOGGER.info(f"Current remaining items are {len(filtered_hlos)}, locked are {len(locked)}, "
                    f"failed are {len(failed)}, done are {len(done)} total is {len(hlos)}")

        return filtered_hlos, locked, done, failed

    def hlo_acquire_lock(self, h):
        lock_file_name = self.generate_lock_file_name(h)
        exists = self.exists(lock_file_name)

        if exists:
            return False

        # FIXME: this is racy - another worker could upload the lock file at the
        # same time as we have no way to tell. S3 doesn't really offer any way
        # to do locking correctly. Ultimately, this shouldn't cause any actual
        # problems at runtime since S3 puts are atomic - no reader will ever see
        # a partial NEFF or HLO.
        self.upload_string_to_file(lock_file_name, "")
        return True

    def hlo_release_lock(self, h):
        lock_file_name = self.generate_lock_file_name(h)
        s3_bucket, s3_key = self.s3_path_to_bucket_and_key(lock_file_name)
        self.bucket.objects.filter(Prefix=s3_key).delete()
        return True

    def s3_path_to_bucket_and_key(self, s3_path):
        o = urlparse(s3_path, allow_fragments=False)
        s3_bucket = o.netloc
        s3_key = o.path.lstrip('/')
        return s3_bucket, s3_key

    def remove(self, path):
        s3_bucket, s3_key = self.s3_path_to_bucket_and_key(path)
        self.s3.delete_object(Bucket=s3_bucket, Key=s3_key)

    def exists(self, path):
        s3_bucket, s3_key = self.s3_path_to_bucket_and_key(path)
        try:
            self.s3.head_object(Bucket=s3_bucket, Key=s3_key)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                # The object does not exist.
                return False
            elif e.response['Error']['Code'] == "503":
                # Connection refused but model might exist in s3 cache.
                # Let downstream code handle the error.
                raise ConnectionError(f"Connection failed to {s3_bucket}: error code 503 Service Unavailable")
            else:
                # Something else has gone wrong.
                LOGGER.error(
                    "Something went wrong while checking if NEFF exists in cache.")
                raise

        # The object does exist.
        return True

    def download_file(self, s3_path, dst_path):
        _, s3_key = self.s3_path_to_bucket_and_key(s3_path)
        try:
            self.bucket.download_file(s3_key, dst_path)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                # The object does not exist.
                raise FileNotFoundError(f"{s3_key} not found.")
            else:
                # Something else has gone wrong.
                LOGGER.error(
                    f"Something went wrong downloading {s3_key} to {dst_path} from the cache.")
                raise

    def upload_file(self, s3_path, src_path):
        s3_bucket, s3_key = self.s3_path_to_bucket_and_key(s3_path)
        self.bucket.upload_file(src_path, s3_key)

    def upload_string_to_file(self, s3_path, data):
        s3_bucket, s3_key = self.s3_path_to_bucket_and_key(s3_path)
        self.s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=data)

    def download_file_to_string(self, s3_path, limit=None):
        s3_bucket, s3_key = self.s3_path_to_bucket_and_key(s3_path)
        obj = self.s3.get_object(Bucket=s3_bucket, Key=s3_key)
        try:
            if limit is None:
                return obj['Body'].read().decode(errors='replace')
            return obj['Body'].read(limit).decode(errors='replace')
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                # The object does not exist.
                raise FileNotFoundError(f"{s3_key} not found.")
            else:
                # Something else has gone wrong.
                LOGGER.error(
                    f"Something went wrong downloading {s3_key} from the cache.")
                raise


def create_compile_cache(cache_url):
    if cache_url.is_s3():
        return CompileCacheS3(cache_url)
    else:
        return CompileCacheFs(cache_url)
