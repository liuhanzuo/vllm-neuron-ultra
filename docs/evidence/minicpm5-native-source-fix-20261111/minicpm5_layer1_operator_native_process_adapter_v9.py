"""Source-only successor: original live callbacks with static catalog binding.

Loaded Python coverage/diagnostics remain v8. No patch of executed sources.
"""
from research.minicpm5_layer1_operator_native_process_adapter_v8 import *
from research import minicpm5_layer1_operator_native_process_adapter_v8 as previous
from research import minicpm5_layer1_operator_native_lifecycle_core_v3 as life
from research.minicpm5_native_static_binding_v1 import StaticPortsMixin


class Ports(StaticPortsMixin, previous.Ports):
    pass
