"""Narrow, pure ELF metadata/candidate-graph helpers for Dory's 27-file table.

No filesystem reads, subprocesses, native imports, loader calls, or admission.
The caller supplies either authenticated saved readelf text or authenticated
bytes of exactly the three missing objects. Candidate edges never mean selected
providers; lookup outside the finite named table remains uncovered.
"""
from copy import deepcopy
from pathlib import PurePosixPath
import posixpath
import re
import struct

MAX_FILE = 4 * 2**20
MAX_NEEDED = 64
STRING_TAGS = {1: "NEEDED", 14: "SONAME", 15: "RPATH", 29: "RUNPATH",
               0x6FFFFEFA: "CONFIG", 0x6FFFFEFB: "DEPAUDIT",
               0x6FFFFEFC: "AUDIT", 0x7FFFFFFD: "AUXILIARY", 0x7FFFFFFF: "FILTER"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def normalized(strings, interpreter, flags):
    result = dict(needed=[], soname=None, search_tags=[], interpreter=interpreter,
                  extra_loader_directives=[], flags=flags)
    for tag, value in strings:
        require(type(value) is str and value.isascii() and "\0" not in value
                and len(value) <= 4096, "Bounded ASCII loader string")
        if tag == "NEEDED":
            require(value and "/" not in value and len(result["needed"]) < MAX_NEEDED,
                    "Simple finite NEEDED name; path-valued needs require a new review")
            require(value not in result["needed"], "Unique NEEDED entries")
            result["needed"].append(value)
        elif tag == "SONAME":
            require(result["soname"] is None and value and "/" not in value, "One simple SONAME")
            result["soname"] = value
        elif tag in ("RPATH", "RUNPATH"):
            require(not any(row["tag"] == tag for row in result["search_tags"]), "Unique path tag")
            result["search_tags"].append(dict(tag=tag, value=value))
        else:
            result["extra_loader_directives"].append(dict(tag=tag, value=value))
    require(interpreter is None or interpreter.startswith("/") and len(interpreter) <= 4096,
            "Absolute finite interpreter path")
    return result


def parse_saved_readelf(text):
    """Parse only already saved, hash-authenticated readelf -lW -dW output."""
    require(type(text) is str and len(text.encode()) <= 2**20, "Bounded saved ELF text")
    strings = []
    wanted = set(STRING_TAGS.values())
    for line in text.splitlines():
        match = re.match(r"\s*0x[0-9a-f]+\s+\(([A-Z_0-9]+)\)\s+(.*)$", line)
        if match and match[1] in wanted:
            value = re.search(r"\[([^\]]*)\]", match[2])
            require(value is not None, "Saved loader string must be complete")
            strings.append((match[1], value[1]))
    interpreter = re.findall(r"Requesting program interpreter:\s*([^\]]+)\]", text)
    require(len(interpreter) <= 1, "One saved interpreter")
    return normalized(strings, interpreter[0] if interpreter else None, None)


def parse_elf64(raw):
    """Read an in-memory ELF64 little-endian x86-64 object, never execute it.

    Program headers map virtual DT_STRTAB to file bytes. Section headers, symbol
    lookup, relocation processing and ELF constructors are deliberately unused.
    Unsupported/malformed forms fail closed instead of guessing dependencies.
    """
    require(type(raw) is bytes and 64 <= len(raw) <= MAX_FILE, "Finite ELF payload")
    require(raw[:7] == b"\x7fELF\x02\x01\x01", "ELF64 little-endian current version")
    header = struct.unpack_from("<HHIQQQIHHHHHH", raw, 16)
    kind, machine, version, _, phoff, _, _, ehsize, phsize, phnum, _, _, _ = header
    require(kind in (2, 3) and machine == 62 and version == 1 and ehsize == 64,
            "Only ordinary x86-64 EXEC/DYN ELF")
    require(phsize == 56 and 1 <= phnum <= 128 and phoff >= 64
            and phoff + phsize * phnum <= len(raw), "Bounded complete program headers")
    loads = []
    dynamic = []
    interpreter = []
    for i in range(phnum):
        typ, _, offset, vaddr, _, filesz, memsz, _ = struct.unpack_from("<IIQQQQQQ", raw, phoff + i * 56)
        require(offset <= len(raw) and filesz <= len(raw) - offset, "Segment file range in bounds")
        if typ == 1:
            require(filesz <= memsz, "LOAD file bytes fit memory range")
            loads.append((vaddr, offset, filesz))
        elif typ == 2:
            dynamic.append((offset, filesz))
        elif typ == 3:
            require(1 < filesz <= 4096, "Bounded INTERP")
            value = raw[offset:offset + filesz]
            require(value.endswith(b"\0") and b"\0" not in value[:-1], "Complete INTERP string")
            interpreter.append(value[:-1].decode("ascii"))
    require(loads and len(dynamic) == 1 and len(interpreter) <= 1, "One dynamic table and optional interpreter")
    offset, size = dynamic[0]
    require(16 <= size <= 65536 and size % 16 == 0, "Bounded aligned dynamic table")
    values = {}
    string_offsets = []
    ended = False
    for at in range(offset, offset + size, 16):
        tag, value = struct.unpack_from("<qQ", raw, at)
        if tag == 0:
            ended = True
            break
        if tag in STRING_TAGS:
            string_offsets.append((STRING_TAGS[tag], value))
        elif tag in (5, 10, 30, 0x6FFFFFFB):
            require(tag not in values, "Unique string-table/flags tag")
            values[tag] = value
    require(ended and 5 in values and 10 in values, "Terminated table with explicit STRTAB/STRSZ")
    address, length = values[5], values[10]
    require(1 <= length <= 2**20, "Bounded dynamic string table")
    matches = [base + address - start for start, base, count in loads
               if start <= address and address + length <= start + count]
    require(len(matches) == 1, "STRTAB maps uniquely to file-backed LOAD bytes")
    strings = raw[matches[0]:matches[0] + length]
    decoded = []
    for name, pos in string_offsets:
        require(pos < length, "String offset in STRSZ")
        end = strings.find(b"\0", pos, min(length, pos + 4097))
        require(end >= 0, "Bounded terminated dynamic string")
        decoded.append((name, strings[pos:end].decode("ascii")))
    return normalized(decoded, interpreter[0] if interpreter else None,
                      {"DT_FLAGS": values.get(30), "DT_FLAGS_1": values.get(0x6FFFFFFB)})


def candidate_graph(table, metadata):
    """Recurse inside the exact 27 named content candidates, never a filesystem.

    SONAME matches are conservative possible content providers, not evidence of
    visibility, search order, selected objects or symbol resolution. All nodes
    are roots so optional CPU/native branches remain represented without imports.
    """
    rows = table["candidates"]
    require(len(rows) == 27, "Exact 27-candidate finite table")
    named = {row["reference"]["path"]: row for row in rows}
    require(len(named) == 27 and set(metadata) <= set(named), "Unique named metadata scope")
    aliases = {name: name for name in named}
    for row in rows:
        link = row.get("saved_link_observation")
        if link:
            old = aliases.setdefault(link["requested"], row["reference"]["path"])
            require(old == row["reference"]["path"], "No conflicting saved alias")
    saved = table["saved_dependency_candidates"]
    queue = list(named)
    visited = set()
    edges = []
    uncovered = []
    while queue:
        path = queue.pop(0)
        if path in visited:
            continue
        visited.add(path)
        obj = metadata.get(path)
        if obj is None:
            uncovered.append(dict(path=path, reason="ELF_metadata_not_collected"))
            continue
        for directive in obj["extra_loader_directives"]:
            uncovered.append(dict(path=path, reason="additional_loader_directive", directive=directive))
        requests = [(name, "DT_NEEDED") for name in obj["needed"]]
        if obj["interpreter"]:
            requests.append((obj["interpreter"], "PT_INTERP"))
        for name, kind in requests:
            candidates = set()
            sources = []
            if kind == "PT_INTERP":
                if name in aliases:
                    candidates.add(aliases[name]); sources.append("saved_interpreter_alias")
                else:
                    uncovered.append(dict(path=path, needed=name, reason="interpreter_outside_named_table"))
            else:
                for other, meta in metadata.items():
                    if meta["soname"] == name:
                        candidates.add(other)
                if candidates:
                    sources.append("SONAME_matches_only_no_visibility_or_selection_claim")
                for old in saved.get(path, []):
                    if old["name"] == name:
                        for target in old["resolved_candidates"]:
                            if target in named:
                                candidates.add(target)
                            else:
                                uncovered.append(dict(path=path, needed=name, target=target,
                                                      reason="saved_candidate_without_content_authority_in_table"))
                        sources.append("saved_static_candidates_not_current_search")
                        for unknown in old.get("unresolved_search_paths", []):
                            uncovered.append(dict(path=path, needed=name, search=unknown,
                                                  reason="saved_unresolved_search_path"))
                for tag in obj["search_tags"]:
                    for directory in tag["value"].split(":"):
                        directory = directory.replace("${ORIGIN}", str(PurePosixPath(path).parent))
                        directory = directory.replace("$ORIGIN", str(PurePosixPath(path).parent))
                        if not directory.startswith("/") or "$" in directory:
                            uncovered.append(dict(path=path, needed=name, search=directory,
                                                  reason="unexpanded_or_relative_search_path"))
                            continue
                        target = posixpath.normpath(directory + "/" + name)
                        if target in aliases:
                            candidates.add(aliases[target]); sources.append("named_path_expansion")
                        else:
                            uncovered.append(dict(path=path, needed=name, target=target,
                                                  reason="search_path_candidate_existence_and_authority_unobserved"))
            if not candidates:
                uncovered.append(dict(path=path, needed=name, reason="no_named_content_candidate"))
            for candidate in sorted(candidates):
                if candidate not in visited:
                    queue.append(candidate)
            edges.append(dict(requester=path, request_kind=kind, needed=name,
                              content_candidates=sorted(candidates), evidence_kinds=sorted(set(sources)),
                              actual_selected_path=None))
            require(len(edges) <= 27 * 65 and len(uncovered) <= 10000, "Finite graph/output bounds")
    return dict(named_candidate_count=27, visited_named_candidates=len(visited),
                named_ELF_metadata_complete=set(metadata) == set(named),
                edges=edges, uncovered=uncovered,
                candidate_content_table=deepcopy(rows),
                external_loader_search_complete=False, runtime_dlopen_closure_complete=False,
                preexec_native_content_authority_accepted=False,
                actual_loader_selection_established=False, native_ABI_qualified=False,
                full_model_qualified=False, new_speedup_claim=False)
