"""Unchanged bounded output inventory with exact r9 generated-link routing."""
from types import FunctionType
import nemotron9b_package_storage_inventory_candidate_v5 as original
import nemotron9b_compiler_tiled_links_v1 as compiler_links

inventory = FunctionType(original.inventory.__code__, dict(vars(original), compiler_links=compiler_links),
                         'inventory', original.inventory.__defaults__)
inventory.__kwdefaults__ = original.inventory.__kwdefaults__
