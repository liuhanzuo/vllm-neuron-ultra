"""Synthetic in-memory ELF tests and saved-JSON/source checks only.

No native file, archive, gzip, process, service, lock, catalog or network access.
The future collector is parsed as source and never imported or invoked.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import unittest

import dory_loader_candidate_elf_v1 as core

Q = Path(__file__).parent


def fixture(strings=None, *, duplicate_load=False):
    strings = strings if strings is not None else [(1, "libalpha.so"), (14, "libsample.so")]
    data = bytearray(8192)
    data[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
    phnum = 3 if duplicate_load else 2
    struct.pack_into("<HHIQQQIHHHHHH", data, 16, 3, 62, 1, 0, 64, 0, 0,
                     64, 56, phnum, 0, 0, 0)
    dynamic, string_at, base = 0x400, 0x1000, 0x400000
    string_data = bytearray(b"\0")
    entries = []
    for tag, value in strings:
        entries.append((tag, len(string_data)))
        string_data.extend(value.encode() + b"\0")
    entries += [(5, base + string_at), (10, len(string_data)), (0, 0)]
    struct.pack_into("<IIQQQQQQ", data, 64, 1, 4, 0, base, 0, len(data), len(data), 4096)
    struct.pack_into("<IIQQQQQQ", data, 120, 2, 4, dynamic, base + dynamic, 0,
                     16 * len(entries), 16 * len(entries), 8)
    if duplicate_load:
        struct.pack_into("<IIQQQQQQ", data, 176, 1, 4, 0, base, 0, len(data), len(data), 4096)
    for i, entry in enumerate(entries):
        struct.pack_into("<qQ", data, dynamic + i * 16, *entry)
    data[string_at:string_at + len(string_data)] = string_data
    return data, entries


def graph_fixture():
    paths = ["/candidate/obj%d.so" % i for i in range(27)]
    table = dict(candidates=[dict(reference=dict(path=p), saved_link_observation=None) for p in paths],
                 saved_dependency_candidates={})
    metadata = {p: core.normalized([("SONAME", "obj%d.so" % i)], None, {}) for i, p in enumerate(paths)}
    return table, metadata, paths


class ELFChecks(unittest.TestCase):
    def test_minimal_valid_dynamic_object(self):
        raw, _ = fixture()
        result = core.parse_elf64(bytes(raw))
        self.assertEqual(result["needed"], ["libalpha.so"])
        self.assertEqual(result["soname"], "libsample.so")

    def test_distinguishes_rpath_runpath_and_records_extra_loader_directives(self):
        raw, _ = fixture([(15, "$ORIGIN/../lib"), (29, "$ORIGIN"),
                          (0x7FFFFFFF, "libfilter.so"), (0x6FFFFEFC, "audit.so")])
        result = core.parse_elf64(bytes(raw))
        self.assertEqual([r["tag"] for r in result["search_tags"]], ["RPATH", "RUNPATH"])
        self.assertEqual([r["tag"] for r in result["extra_loader_directives"]], ["FILTER", "AUDIT"])

    def test_wrong_machine_class_endianness_rejected(self):
        for offset, value in [(4, 1), (5, 2), (18, 3)]:
            with self.subTest(offset=offset):
                raw, _ = fixture(); raw[offset] = value
                with self.assertRaises(ValueError): core.parse_elf64(bytes(raw))

    def test_truncated_header_or_segment_rejected(self):
        raw, _ = fixture()
        for length in [0, 63, 100, 8191]:
            with self.subTest(length=length), self.assertRaises(ValueError):
                core.parse_elf64(bytes(raw[:length]))

    def test_out_of_range_program_headers_rejected(self):
        raw, _ = fixture(); struct.pack_into("<Q", raw, 32, 8192)
        with self.assertRaises(ValueError): core.parse_elf64(bytes(raw))

    def test_missing_dynamic_terminator_rejected(self):
        raw, entries = fixture(); struct.pack_into("<qQ", raw, 0x400 + 16 * (len(entries)-1), 1, 0)
        with self.assertRaises(ValueError): core.parse_elf64(bytes(raw))

    def test_string_table_must_have_unique_file_backed_mapping(self):
        raw, _ = fixture(duplicate_load=True)
        with self.assertRaises(ValueError): core.parse_elf64(bytes(raw))
        raw, _ = fixture(); struct.pack_into("<Q", raw, 0x400 + 2 * 16 + 8, 0x900000)
        with self.assertRaises(ValueError): core.parse_elf64(bytes(raw))

    def test_oversized_or_out_of_range_strings_rejected(self):
        raw, _ = fixture(); struct.pack_into("<Q", raw, 0x400 + 3 * 16 + 8, 2**20 + 1)
        with self.assertRaises(ValueError): core.parse_elf64(bytes(raw))
        raw, _ = fixture(); struct.pack_into("<Q", raw, 0x408, 8192)
        with self.assertRaises(ValueError): core.parse_elf64(bytes(raw))

    def test_duplicate_needed_or_path_valued_needed_rejected(self):
        for values in [[(1, "x.so"), (1, "x.so")], [(1, "/unreviewed/x.so")]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                core.parse_elf64(bytes(fixture(values)[0]))

    def test_missing_string_terminator_rejected(self):
        raw, entries = fixture(); size = entries[-2][1]
        raw[0x1000:0x1000+size] = b"x" * size
        with self.assertRaises(ValueError): core.parse_elf64(bytes(raw))

    def test_needed_count_is_bounded(self):
        raw, _ = fixture([(1, "lib%d.so" % i) for i in range(65)])
        with self.assertRaises(ValueError): core.parse_elf64(bytes(raw))


class GraphChecks(unittest.TestCase):
    def test_two_same_soname_candidates_never_become_selected_provider(self):
        table, metadata, paths = graph_fixture()
        metadata[paths[0]]["needed"] = ["shared.so"]
        metadata[paths[1]]["soname"] = metadata[paths[2]]["soname"] = "shared.so"
        result = core.candidate_graph(table, metadata)
        self.assertEqual(result["edges"][0]["content_candidates"], paths[1:3])
        self.assertIsNone(result["edges"][0]["actual_selected_path"])
        self.assertFalse(result["actual_loader_selection_established"])

    def test_single_candidate_is_also_not_loader_selection(self):
        table, metadata, paths = graph_fixture(); metadata[paths[0]]["needed"] = ["obj1.so"]
        result = core.candidate_graph(table, metadata)
        self.assertEqual(result["edges"][0]["content_candidates"], [paths[1]])
        self.assertIsNone(result["edges"][0]["actual_selected_path"])

    def test_external_saved_candidate_stays_uncovered(self):
        table, metadata, paths = graph_fixture(); metadata[paths[0]]["needed"] = ["missing.so"]
        table["saved_dependency_candidates"][paths[0]] = [dict(name="missing.so",
            resolved_candidates=["/not-admitted/missing.so"], unresolved_search_paths=[])]
        result = core.candidate_graph(table, metadata)
        self.assertTrue(any(r["reason"] == "saved_candidate_without_content_authority_in_table" for r in result["uncovered"]))
        self.assertEqual(result["visited_named_candidates"], 27)

    def test_cycles_terminate_without_scope_expansion(self):
        table, metadata, paths = graph_fixture()
        metadata[paths[0]]["needed"] = ["obj1.so"]; metadata[paths[1]]["needed"] = ["obj0.so"]
        result = core.candidate_graph(table, metadata)
        self.assertEqual(result["visited_named_candidates"], 27); self.assertEqual(len(result["edges"]), 2)

    def test_unknown_origin_tokens_and_unobserved_paths_are_not_assumed_absent(self):
        table, metadata, paths = graph_fixture(); metadata[paths[0]]["needed"] = ["libc.so.6"]
        metadata[paths[0]]["search_tags"] = [dict(tag="RUNPATH", value="$ORIGIN/../other:$LIB:")]
        result = core.candidate_graph(table, metadata)
        reasons = {x["reason"] for x in result["uncovered"]}
        self.assertIn("unexpanded_or_relative_search_path", reasons)
        self.assertIn("search_path_candidate_existence_and_authority_unobserved", reasons)

    def test_new_directives_remain_uncovered(self):
        table, metadata, paths = graph_fixture()
        metadata[paths[0]]["extra_loader_directives"] = [dict(tag="AUDIT", value="other.so")]
        self.assertTrue(core.candidate_graph(table, metadata)["uncovered"])

    def test_missing_metadata_does_not_complete_closure(self):
        table, metadata, paths = graph_fixture(); del metadata[paths[-1]]
        result = core.candidate_graph(table, metadata)
        self.assertFalse(result["named_ELF_metadata_complete"])
        self.assertFalse(result["preexec_native_content_authority_accepted"])


class SavedSourceChecks(unittest.TestCase):
    def test_exact_saved_table_and_three_refs(self):
        raw = (Q / "dory-loader-candidate-elf-inputs-20261111-v1.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "db254b660a0f29effc9e4bb1ee0784a7dbf66177a632a94b2e600ab713123fe6")
        table = json.loads(raw)
        self.assertEqual(len(table["new_native_reads"]), 3)
        self.assertEqual(sum(x["bytes"] for x in table["new_native_reads"]), 2422457)
        self.assertFalse(table["policy"]["all_fifteen_runtime_selections_must_be_resolved"])
        preview = core.candidate_graph(table, table["saved_elf_metadata"])
        self.assertEqual(sum(x["reason"] == "ELF_metadata_not_collected" for x in preview["uncovered"]), 3)
        self.assertTrue(all(x["actual_selected_path"] is None for x in preview["edges"]))

    def test_collector_reuses_existing_guard_handoff_and_empty_child_release(self):
        old = ast.parse((Q / "collect_dory_loader_route_sources_v2.py").read_text())
        new = ast.parse((Q / "collect_dory_loader_candidate_elf_v1.py").read_text())
        def nested(tree, name):
            return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        for name in ["guard", "current_handoff"]:
            self.assertEqual(ast.dump(nested(old, name)), ast.dump(nested(new, name)))
        releases = [x for x in ast.walk(new) if isinstance(x, ast.Call)
                    and isinstance(x.func, ast.Attribute) and x.func.attr == "release_after_drain"]
        self.assertEqual(len(releases), 1)
        self.assertEqual([x.value for x in releases[0].args[:2]], [None, None])

    def test_collector_has_no_native_commands_or_imports(self):
        tree = ast.parse((Q / "collect_dory_loader_candidate_elf_v1.py").read_text())
        forbidden = {"torch", "vllm", "numpy", "ctypes", "subprocess", "gzip"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse({x.name.split('.')[0] for x in node.names} & forbidden)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module.split('.')[0], forbidden)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"command", "Popen", "CDLL", "gzip_records", "native_authority"})


if __name__ == "__main__":
    unittest.main()
