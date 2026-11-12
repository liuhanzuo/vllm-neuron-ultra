"""Bounded stdlib IO/process observations for the new, unbound sampler runtime.

Nothing reads proc, acquires a lease, imports native libraries or starts a
service on import. A future explicitly reviewed CLI calls these operations.
"""
import copy
import fcntl
import hashlib
import inspect
import json
import os
import re
from pathlib import Path
import stat
import sys
import time

from .contract import (LIMITS, LOCK_PATH, LOCK_IDENTITY, ROOT, Q, PHASES,
                                               budget, control_root, digest, reference, require, server_temporary_roots)


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
    temp_roots.extend(roots[0] / arm / phase / name for arm, phase in PHASES
                      if phase in ("capture", "quality", "http") for name in ("tmp", "rpc"))
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


def ipc_socket_path(path, spec):
    """Only UUID4 sockets directly inside the six declared server temp roots."""
    return (path.parent in server_temporary_roots(spec)
            and re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                             path.name) is not None)


def storage_scan(spec):
    from .budget import roots as all_roots, reviewed_budget, unchanged_inputs, check_view_link, CLASSES
    reviewed = unchanged_inputs(spec, reviewed_budget(spec))
    caps = {**budget(), **{key: row["allocated_bytes"] for key, row in reviewed["classes"].items()}}
    usage = dict.fromkeys(caps, 0); counts = dict.fromkeys(caps, 0); entries = 0
    roots = all_roots(spec); inventory = []; links = []
    science_roots = roots[:2]
    tree_classes = (None, None, "view", "artifact", "compiler_cache", "compiler_workdir")
    for root, tree_class in zip(roots, tree_classes):
        require(not root.is_symlink(), "root symlink forbidden")
        if not root.exists(): continue
        pending = [root]
        while pending:
            path = pending.pop()
            try: s = observed_stat(path)
            except FileNotFoundError:
                if any(tmp in path.parents for tmp in temporary_roots(science_roots)):
                    raise PublicationRace("declared native temporary disappeared before stat")
                raise
            entries += 1
            inventory.append((str(path), s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_blocks, s.st_mtime_ns, s.st_ctime_ns))
            require(entries <= LIMITS["max_output_entries"] + sum(row["entries"] for row in reviewed["classes"].values()), "global entry cap")
            kind = tree_class
            if kind is None:
                if any(path == stage or stage in path.parents for stage in
                       (control_root(spec) / "shared-prepare", control_root(spec) / "shared-build")):
                    kind = "build_auxiliary"
                elif stat.S_ISDIR(s.st_mode) or stat.S_ISSOCK(s.st_mode): kind = "temporary_and_directory_bytes"
                else: kind = storage_class(path, science_roots)
            usage[kind] += s.st_blocks * 512; counts[kind] += 1
            if stat.S_ISLNK(s.st_mode):
                require(tree_class == "view", "symlinks forbidden outside exact prepared view")
                require(s.st_nlink == 1 and s.st_size <= reviewed["classes"]["view"]["single_file_bytes"]
                        and s.st_blocks * 512 <= reviewed["classes"]["view"]["single_file_bytes"], "bounded singleton view link")
                links.append(check_view_link(path, spec, reviewed)); continue
            if stat.S_ISDIR(s.st_mode):
                require(path.resolve() == path, "nonsymlink directory ancestry")
                try: pending.extend(path.iterdir())
                except FileNotFoundError:
                    if any(tmp in path.parents for tmp in temporary_roots(science_roots)):
                        raise PublicationRace("declared native temporary directory disappeared before scan")
                    raise
                continue
            if stat.S_ISSOCK(s.st_mode):
                require(s.st_nlink == 1 and ipc_socket_path(path, spec) and s.st_size <= LIMITS["single_file_bytes"], "only declared UUID4 RPC sockets")
                continue
            require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, "singleton regular output required")
            cap = (reviewed["classes"][kind]["single_file_bytes"] if kind in CLASSES
                   else LIMITS["model_single_file_bytes"] if kind == "model_payload_bytes"
                   else LIMITS["metadata_bytes"] if kind == "all_metadata_bytes" else LIMITS["single_file_bytes"])
            require(s.st_size <= cap and s.st_blocks * 512 <= cap, "single logical/allocated file cap exceeded: " + str(path))
            if path.suffix == ".log":
                log_cap = (LIMITS["coordinator_log_bytes"] if path.name == "coordinator.log" else
                    LIMITS["server_log_bytes"] if path.name.endswith("-server.log") else LIMITS["leaf_log_bytes"])
                require(s.st_size <= log_cap, "original finite log cap exceeded")
    require(sum(counts[k] for k in budget()) <= LIMITS["max_output_entries"], "original science entry cap")
    require(all(counts[k] <= reviewed["classes"][k]["entries"] for k in CLASSES), "per-class entry cap exceeded")
    require(all(usage[k] <= cap for k, cap in caps.items()), "per-class allocated storage cap exceeded")
    require(sum(usage[k] for k in budget()) <= LIMITS["total_allocated_output_bytes"], "original science 2 GiB cap")
    require(sum(usage.values()) <= sum(caps.values()), "global allocated storage cap")
    return {"classes": usage, "allocated_bytes": sum(usage.values()), "entries": entries,
            "view_links": sorted(links, key=lambda row: row["path"])}, sorted(inventory)


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
    from .budget import reviewed_budget, remaining_reserve
    reserved = remaining_reserve(spec, usage, reviewed_budget(spec))
    free = {}
    for path, key in ((ROOT, "root_floor_bytes"), (Path("/mnt/nvme"), "nvme_floor_bytes")):
        s = os.statvfs(path); free[str(path)] = s.f_bavail * s.f_frsize
        remaining = reserved["nvme" if path.name == "nvme" else "root"]
        required = LIMITS[key] + remaining
        if not free[str(path)] >= required:
            # All filesystem fields below belong to this one statvfs call.
            # f_fsid identifies the filesystem; it is deliberately not st_dev.
            failure = dict(path=str(path),
                device=dict(kind="statvfs_fsid", value=s.f_fsid),
                available_bytes=free[str(path)], floor_bytes=LIMITS[key],
                remaining_reservation_bytes=remaining, required_bytes=required,
                shortfall_bytes=required - free[str(path)],
                failed_constraints=(["floor"] if free[str(path)] < LIMITS[key] else [])
                    + (["reservation"] if remaining and free[str(path)] < required else []),
                reserve=reserve)
            raise ValueError("actual storage floor/reservation failed: "
                             + json.dumps(failure, sort_keys=True, separators=(",", ":")))
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
    from qwen35_sampler_torch_namespace_sources_v1 import known_namespace
    checked = {}
    modules = dict(sys.modules)
    for name, module in modules.items():
        if known_namespace(name, module, modules, pins, lambda path: raw(path, 4 * 1024**2)):
            continue
        path = inspect.getattr_static(module, "__file__", None)
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
