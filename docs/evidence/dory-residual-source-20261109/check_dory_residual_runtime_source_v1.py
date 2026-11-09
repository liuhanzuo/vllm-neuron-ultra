"""Source/setup/admission checks only, no native model execution."""
import ast
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import dory_residual_replay_runtime_v1 as runtime


Q = Path(__file__).resolve().parent


class SourceIntegration(unittest.TestCase):
    def test_original_model_setup_is_unchanged_except_selected_replay_import(self):
        old = Q / "dory_first5_worker_preparation_v1.py"
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),
                         "b9cbb6ef95c7c274d93e33624e983d3d9d748fffeb73dde4b29e45537216ec28")
        new = Q / "dory_residual_worker_preparation_v1.py"
        old_main = next(n for n in ast.parse(old.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "main")
        new_main = next(n for n in ast.parse(new.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "main")
        old_main = copy.deepcopy(old_main)
        changed = 0
        for node in ast.walk(old_main):
            if isinstance(node, ast.ImportFrom) and node.module == "dory_first5_replay_runtime_v2":
                node.module = "dory_residual_replay_runtime_v1"
                changed += 1
        self.assertEqual(changed, 1)
        self.assertEqual(ast.dump(old_main, include_attributes=False),
                         ast.dump(new_main, include_attributes=False))

    def rejected(self, spec):
        guard = Mock()
        env = dict(torch=None, official=None, spec=spec, guard=guard,
                   raw=None, save=None, out=None)
        with patch.object(runtime, "OriginalEventStream") as stream:
            with self.assertRaisesRegex(RuntimeError, "Separate finalized"):
                runtime.run(env)
            stream.assert_not_called()
        guard.assert_not_called()

    def test_unadmitted_spec_cannot_start_metadata_or_model_work(self):
        self.rejected(dict(schema="dory_residual_complete_replay_diagnostic_v1",
                           diagnostic_only=True, runtime_admitted=False))

    def test_old_first_five_admission_cannot_start_complete_replay(self):
        self.rejected(dict(schema="dory_first5_propagation_diagnostic_v1",
                           diagnostic_only=True, runtime_admitted=True))

    def test_standalone_runtime_entrypoint_is_disabled(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(Q / "dory_residual_replay_runtime_v1.py")],
            capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, "")
        self.assertIn("Source preparation only: complete native worker", completed.stderr)

    def test_source_imports_do_not_load_numerical_or_model_modules(self):
        self.assertFalse(any(name.split(".")[0] in {"torch", "numpy", "vllm", "neuronxcc", "torch_xla"}
                             for name in sys.modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
