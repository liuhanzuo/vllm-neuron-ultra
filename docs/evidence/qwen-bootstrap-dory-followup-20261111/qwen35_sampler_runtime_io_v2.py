"""Bounded stdlib IO/process observations for the new, unbound sampler runtime.

Nothing reads proc, acquires a lease, imports native libraries or starts a
service on import. A future explicitly reviewed CLI calls these operations.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

from qwen35_sampler_runtime_contract_v2 import (LIMITS, LOCK_PATH, LOCK_IDENTITY, ROOT, Q, PHASES,
                                               budget, control_root, digest, reference, require)


def process_api():
    import minicpm5_layer0_execution_protocol_v4
    return minicpm5_layer0_execution_protocol_v4


def identity(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path
            and not any(x.is_symlink() for x in (path, *path.parents)), "canonical metadata path required")
    for attempt in range(8):
        try:
            s = observed_stat(path)
            break
        except PublicationRace:
            if attempt == 7: raise ValueError("authenticated atomic metadata publisher did not settle")
            time.sleep(.01)
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, "singleton regular metadata file required")
    return dict(realpath=str(path), device=s.st_dev, inode=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def raw(path, cap=LIMITS["metadata_bytes"]):
    before = identity(path)
    require(before["size"] <= cap, "bounded metadata read")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        s = os.fstat(fd)
        require((s.st_dev, s.st_ino) == (before["device"], before["inode"]), "file open identity changed")
        chunks, count = [], 0
        while count <= cap:
            block = os.read(fd, min(1024**2, cap + 1 - count))
            if not block: break
            chunks.append(block); count += len(block)
        require(count == before["size"] and identity(path) == before, "metadata changed while reading")
        return b"".join(chunks)
    finally:
        os.close(fd)


def ref(path, cap=LIMITS["metadata_bytes"]):
    return {"path": str(path), "sha256": hashlib.sha256(raw(path, cap)).hexdigest()}


def read(ref_value, cap=LIMITS["metadata_bytes"]):
    reference(ref_value)
    value = raw(ref_value["path"], cap)
    require(hashlib.sha256(value).hexdigest() == ref_value["sha256"], "authenticated metadata digest changed")
    return json.loads(value)


def once(path, value):
    """Exclusive final path; publishes atomically only after fsync."""
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not path.exists(), "new exact receipt path required")
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    require(len(payload) <= LIMITS["metadata_bytes"], "bounded output receipt")
    temporary = path.with_name(path.name + ".publishing")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    # link is exclusive even if another producer has already published a path.
    os.link(temporary, path, follow_symlinks=False)
    temporary.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try: os.fsync(fd)
    finally: os.close(fd)
    return ref(path)


def descriptor(owner, fd):
    from research import descriptor_serial_lock_proof_v1 as proof
    for path in LOCK_PATH.parents:
        require(stat.S_ISDIR(path.lstat().st_mode), "original lock ancestry is nonsymlink")
    p = process_api()
    return proof.prove(owner, LOCK_IDENTITY, fd, proc=p.proc, open_binary=lambda x: Path(x).open("rb"),
                       lstat=lambda x: Path(x).lstat(), stat=lambda x: Path(x).stat(), readlink=os.readlink)


class RetainedLock:
    """Only the outer owns this FD; it is never inherited or reacquired."""
    def __init__(self):
        self.fd = None; self.owner = None; self.initial = None

    def acquire(self):
        require(self.fd is None, "single acquisition only")
        p = process_api(); p.user_bus()
        s = LOCK_PATH.lstat()
        require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1
                and {"device": s.st_dev, "inode": s.st_ino} == LOCK_IDENTITY, "original singleton lock inode required")
        self.fd = os.open(LOCK_PATH, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.owner = p.proc(os.getpid())
            self.initial = self.check()
        except BaseException:
            os.close(self.fd); self.fd = None; raise
        return self.initial

    def check(self):
        require(self.fd is not None and not os.get_inheritable(self.fd), "parent-only retained CLOEXEC descriptor")
        return descriptor(self.owner, self.fd)

    def release(self, completely_drained):
        require(completely_drained is True and self.fd is not None, "never release before all owned units/processes drain")
        self.check()
        fcntl.flock(self.fd, fcntl.LOCK_UN); os.close(self.fd); self.fd = None


def wait_until(probe, guard, seconds, *, message, interval=.1):
    deadline = time.monotonic() + seconds
    while True:
        guard()
        result = probe()
        if result is not None: return result
        require(time.monotonic() < deadline, message)
        time.sleep(min(interval, max(0., deadline - time.monotonic())))


def read_if_present(path):
    return read(ref(path)) if Path(path).exists() else None


def temporary_roots(roots):
    temp_roots = [root / "tmp" for root in roots]
    temp_roots.extend(roots[1] / (arm + "-" + name) / "tmp" for arm, name in PHASES)
    return temp_roots


def storage_class(path, roots):
    temp_roots = temporary_roots(roots)
    if any(root in path.parents for root in temp_roots): return "temporary_and_directory_bytes"
    if path.suffix == ".pt": return "model_payload_bytes"
    if path.suffix == ".bin": return "capture_payload_bytes"
    if path.suffix == ".log": return "all_logs_bytes"
    if path.suffix == ".json": return "all_metadata_bytes"
    if path.name.endswith((".json.tmp", ".json.publishing")):
        return "temporary_and_directory_bytes"
    raise ValueError("undeclared runtime output type: " + str(path))


class PublicationRace(Exception):
    """Only a known .json.tmp/.publishing atomic publisher may cause a retry."""


def transient(path):
    return path.name.endswith((".json.tmp", ".json.publishing"))


def observed_stat(path):
    try: s = path.lstat()
    except FileNotFoundError:
        if transient(path): raise PublicationRace("known atomic temporary disappeared")
        raise
    if stat.S_ISREG(s.st_mode) and s.st_nlink == 2:
        peer = (path.with_name(path.name.removesuffix(".publishing")) if path.name.endswith(".json.publishing")
                else path.with_name(path.name + ".publishing") if path.suffix == ".json" else None)
        require(peer is not None, "unexplained runtime hardlink")
        try: other = peer.lstat()
        except FileNotFoundError:
            # The known publisher may already have unlinked its temporary.
            try: fresh = path.lstat()
            except FileNotFoundError:
                if transient(path): raise PublicationRace("known temporary unlinked after hardlink observation")
                raise
            require(fresh.st_nlink == 1 and (fresh.st_dev, fresh.st_ino) == (s.st_dev, s.st_ino),
                    "unexplained hardlink after publication")
            raise PublicationRace("known publisher completed during observation")
        require(stat.S_ISREG(other.st_mode) and (other.st_dev, other.st_ino) == (s.st_dev, s.st_ino), "unknown hardlink peer")
        if other.st_nlink == 1 and path.name.endswith(".json.publishing"):
            try: path.lstat()
            except FileNotFoundError: raise PublicationRace("known temporary unlinked before final peer observation")
            raise ValueError("inconsistent singleton peer while publishing path still exists")
        require(other.st_nlink == 2, "unknown hardlink peer count")
        raise PublicationRace("known exclusive atomic publisher has not unlinked temporary yet")
    return s


def storage_scan(spec):
    caps = budget(); usage = dict.fromkeys(caps, 0); entries = 0
    roots = (Path(spec["output_root"]), control_root(spec))
    inventory = []
    for root in roots:
        if not root.exists(): continue
        pending = [root]
        while pending:
            path = pending.pop()
            try: s = observed_stat(path)
            except FileNotFoundError:
                if any(tmp in path.parents for tmp in temporary_roots(roots)):
                    raise PublicationRace("declared native temporary disappeared before stat")
                raise
            entries += 1
            inventory.append((str(path), s.st_dev, s.st_ino, s.st_size, s.st_blocks, s.st_mtime_ns, s.st_ctime_ns))
            require(entries <= LIMITS["max_output_entries"] and not stat.S_ISLNK(s.st_mode), "finite nonsymlink output tree")
            if stat.S_ISDIR(s.st_mode):
                usage["temporary_and_directory_bytes"] += s.st_blocks * 512
                try: pending.extend(path.iterdir())
                except FileNotFoundError:
                    if any(tmp in path.parents for tmp in temporary_roots(roots)):
                        raise PublicationRace("declared native temporary directory disappeared before scan")
                    raise
                continue
            require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, "runtime output must be a singleton regular file")
            kind = storage_class(path, roots)
            cap = (LIMITS["model_single_file_bytes"] if kind == "model_payload_bytes"
                   else LIMITS["metadata_bytes"] if kind == "all_metadata_bytes" else LIMITS["single_file_bytes"])
            require(s.st_size <= cap, "single output size exceeded: " + str(path))
            if kind == "all_logs_bytes":
                cap = (LIMITS["coordinator_log_bytes"] if path.name == "coordinator.log"
                       else LIMITS["server_log_bytes"] if path.name.endswith("-server.log") else LIMITS["leaf_log_bytes"])
                require(s.st_size <= cap, "individual finite log cap exceeded")
            usage[kind] += s.st_blocks * 512
    require(all(usage[k] <= v for k, v in caps.items()), "per-class allocated output limit exceeded")
    require(sum(usage.values()) <= LIMITS["total_allocated_output_bytes"], "total allocated output limit exceeded")
    return {"classes": usage, "allocated_bytes": sum(usage.values()), "entries": entries}, sorted(inventory)


def storage(spec, *, terminal=False):
    # Running log sizes need not be static. Only known atomic publication
    # races get finite retries; arbitrary disappearance/hardlinks fail closed.
    for attempt in range(8):
        try:
            value, inventory = storage_scan(spec)
            if terminal:
                second, again = storage_scan(spec)
                require(value == second and inventory == again, "terminal output inventory is not stable")
            return value
        except PublicationRace:
            if attempt == 7: raise ValueError("known atomic publisher did not settle within finite storage retries")
            time.sleep(.01)


def resources(spec, *, reserve=False):
    usage = storage(spec)
    free = {}
    for path, key in ((ROOT, "root_floor_bytes"), (Path("/mnt/nvme"), "nvme_floor_bytes")):
        s = os.statvfs(path); free[str(path)] = s.f_bavail * s.f_frsize
        remaining = max(0, LIMITS["total_allocated_output_bytes"] - usage["allocated_bytes"]) if reserve and path.name == "nvme" else 0
        require(free[str(path)] >= LIMITS[key] + remaining, "actual storage floor/reservation failed")
    with Path("/proc/meminfo").open() as stream: text = stream.read(65537)
    require(len(text) <= 65536, "bounded memory observation")
    values = {line.split(":", 1)[0]: line.split(":", 1)[1].strip().split() for line in text.splitlines()}
    available = int(values["MemAvailable"][0]) * 1024
    require(available >= LIMITS["memory_available_floor_bytes"], "MemAvailable floor failed")
    return dict(storage=usage, available_disk_bytes=free, available_memory_bytes=available)


class Metadata:
    """Preserves complete inherited transports and authenticates new small refs."""
    def __init__(self, guard):
        self.guard = guard; self.identities = {}

    def read(self, value):
        from research import minicpm5_layer1_operator_sdk_sources_v1 as sources
        self.guard(); sources.reference(value)
        # Inherited metadata can include reviewed aliases/large envelopes. Its
        # original reader and identity rules remain authoritative.
        from minicpm5_layer0_runtime_auth_v1 import identity as historical_identity
        before = historical_identity(value["path"])
        result = sources.load(value)
        require(historical_identity(value["path"]) == before, "historical metadata changed")
        require(value["path"] not in self.identities or self.identities[value["path"]] == before, "metadata identity rebound")
        self.identities[value["path"]] = before
        self.guard(); return result

    def unchanged(self):
        from minicpm5_layer0_runtime_auth_v1 import identity as historical_identity
        require(all(historical_identity(path) == row for path, row in self.identities.items()), "complete inherited metadata identities changed")


def loaded_local_sources(pins):
    """Verify each actually imported experiment source, never binary payloads."""
    checked = {}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and str(Path(path).resolve()).startswith(str(ROOT) + "/"):
            path = str(Path(path).resolve())
            require(path in pins and path.endswith(".py"), "unreviewed loaded local source: " + path)
            require(hashlib.sha256(raw(path, 4 * 1024**2)).hexdigest() == pins[path], "loaded local source pin changed")
            checked[path] = pins[path]
    return checked


def service_owner(unit, owner, props):
    require(type(owner) is dict and owner["pid"] > 0 and str(owner["start_ticks"]).isdigit()
            and len(props["InvocationID"]) == 32, "real service owner required")
    return {"unit": unit, "pid": owner["pid"], "start_ticks": int(owner["start_ticks"]), "invocation_id": props["InvocationID"]}
