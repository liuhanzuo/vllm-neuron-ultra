# SPDX-License-Identifier: Apache-2.0
"""Source-only dispatch checks; tensor equivalence needs serial native validation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class CacheVariantDispatchTests(unittest.TestCase):
    def test_independent_read_and_write_paths(self):
        path=Path(__file__).parents[2]/"vllm_neuron/models/smollm3/cache_attention.py"
        tree=ast.parse(path.read_text())
        classes=[node for node in tree.body if isinstance(node,ast.ClassDef)]
        native_read=lambda *args: "nki-read"
        native_write=lambda *args: "nki-write"
        reference_read=lambda *args: "reference-read"
        reference_write=lambda *args: "reference-write"
        forward=lambda *args: None
        env=dict(Attention=type("Attention",(),{}),attention_forward=forward,
                 read_cache=native_read,write_cache=native_write,
                 read_reference=reference_read,store_slots=reference_write)
        exec(compile(ast.Module(body=classes,type_ignores=[]),str(path),"exec"),env)
        for name,read,write in [("NkiAttention",native_read,native_write),
                               ("NkiReadAttention",native_read,reference_write),
                               ("NkiWriteAttention",reference_read,native_write)]:
            instance=env[name]()
            self.assertIs(instance.read_cache,read)
            self.assertIs(instance.write_cache,write)
            self.assertIs(type(instance).forward,forward)

    def test_prefill_returns_through_original_before_decode(self):
        path=Path(__file__).parents[2]/"vllm_neuron/models/smollm3/cache_attention.py"
        tree=ast.parse(path.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="attention_forward")
        seen=[]
        original=lambda *args: seen.append(args) or "original-prefill"
        env=dict(Attention=SimpleNamespace(forward=original))
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),"exec"),env)
        values=[object() for _ in range(6)]
        self.assertEqual(env["attention_forward"](*values,True),"original-prefill")
        self.assertEqual(seen,[(*values,True)])
