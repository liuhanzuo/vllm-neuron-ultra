"""Pure checks of the new diagnostic proposal; never execute target sources."""
import ast
import hashlib
from pathlib import Path, PurePosixPath
import unittest

import dory_loader_observation_environment_v1 as proposal


def original_constants():
    """Evaluate only the narrow literal/path expressions in two assignments."""
    path = Path(__file__).parent / "dory_stage_a_native_resolver_v4.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != proposal.RESOLVER_SHA256:
        raise ValueError("Original resolver source changed")
    tree = ast.parse(raw)
    names = {"BASE": PurePosixPath(proposal.BASE), "ENV": PurePosixPath(proposal.ENV)}

    def literal(node):
        if isinstance(node, ast.Constant) and type(node.value) is str:
            return node.value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return literal(node.left) / literal(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return literal(node.left) + literal(node.right)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "str" and len(node.args) == 1 and not node.keywords):
            return str(literal(node.args[0]))
        raise ValueError("Unsupported AST expression; target source is never executed")

    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        name = getattr(node.targets[0], "id", None)
        if name == "MODULES":
            found[name] = ast.literal_eval(node.value)
        elif name == "ENVIRONMENT":
            call = node.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "dict" and not call.args):
                raise ValueError("Original environment declaration changed")
            found[name] = {entry.arg: literal(entry.value) for entry in call.keywords}
    return found


class ProposalChecks(unittest.TestCase):
    def test_original_twenty_settings_and_eleven_imports_retained(self):
        original = original_constants()
        self.assertEqual(len(original["ENVIRONMENT"]), 20)
        environment = proposal.diagnostic_environment()
        for key, value in original["ENVIRONMENT"].items():
            self.assertEqual(environment[key], value, key)
        self.assertEqual(proposal.IMPORT_PREFIX, original["MODULES"])

    def test_zero_string_cannot_accidentally_enable_global_torch(self):
        environment = proposal.diagnostic_environment()
        self.assertNotIn("TORCH_USE_RTLD_GLOBAL", environment)
        environment["TORCH_USE_RTLD_GLOBAL"] = "0"
        with self.assertRaises(ValueError):
            proposal.validate_proposal(environment, proposal.IMPORT_PREFIX)

    def test_unknown_inheritance_rejected(self):
        for name in proposal.EXPLICITLY_ABSENT + ("SOME_UNREVIEWED_FLAG",):
            with self.subTest(name=name):
                environment = proposal.diagnostic_environment()
                self.assertNotIn(name, environment)
                environment[name] = ""
                with self.assertRaises(ValueError):
                    proposal.validate_proposal(environment, proposal.IMPORT_PREFIX)

    def test_each_declared_setting_must_be_present_and_unchanged(self):
        for key in proposal.diagnostic_environment():
            with self.subTest(key=key):
                environment = proposal.diagnostic_environment()
                del environment[key]
                with self.assertRaises(ValueError):
                    proposal.validate_proposal(environment, proposal.IMPORT_PREFIX)
                environment = proposal.diagnostic_environment()
                environment[key] += "unreviewed"
                with self.assertRaises(ValueError):
                    proposal.validate_proposal(environment, proposal.IMPORT_PREFIX)

    def test_optional_native_imports_and_model_work_cannot_be_added(self):
        for name in ("vllm.spinloop", "vllm.fs_io_C", "vllm._C", "vllm._C_AVX2",
                     "vllm._C_AVX512", "tests.models.test_dory"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                proposal.validate_proposal(
                    proposal.diagnostic_environment(), proposal.IMPORT_PREFIX + (name,)
                )

    def test_torch_before_vllm_order_cannot_change(self):
        order = list(proposal.IMPORT_PREFIX)
        order[0], order[4] = order[4], order[0]
        with self.assertRaises(ValueError):
            proposal.validate_proposal(proposal.diagnostic_environment(), order)

    def test_proposal_does_not_mint_execution_or_historical_claims(self):
        result = proposal.validate_proposal(
            proposal.diagnostic_environment(), proposal.IMPORT_PREFIX
        )
        self.assertTrue(result.pop("proposal_consistent"))
        self.assertEqual(result.pop("environment_kind"), "new_explicit_controlled_diagnostic")
        self.assertTrue(all(value is False for value in result.values()))

    def test_environment_returns_independent_values(self):
        one = proposal.diagnostic_environment()
        one["LD_PRELOAD"] = "changed"
        self.assertNotEqual(one, proposal.diagnostic_environment())

    def test_nonstring_environment_rejected(self):
        environment = proposal.diagnostic_environment()
        environment["VLLM_USE_SPINLOOP_EXT"] = 0
        with self.assertRaises(ValueError):
            proposal.validate_proposal(environment, proposal.IMPORT_PREFIX)


if __name__ == "__main__":
    unittest.main()
