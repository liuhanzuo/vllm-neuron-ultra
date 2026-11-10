"""Observe original phase frozen dispatcher definitions; no SDK/native reads."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from observe_nemotron4b_interpreter_policy_v1 import describe
from nemotron4b_bootstrap_dispatch_origin_v1 import description
from nemotron4b_process_handshake_v1 import require

if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1]in('native-meta','nxd-meta'),'One original phase')
    candidate=describe(sys.argv[1])
    print(json.dumps(dict(schema='nemotron4b_bootstrap_dispatch_manifest_candidate_v1',
        original_interpreter_observation=candidate,dispatch_manifest=description(),
        actual_interpreter_authenticated=False,explicit_native_payload_file_bytes_read=0,
        actual_model_or_SDK_executed=False,complete_preloaded_source_provenance=False,
        runtime_admitted=False,new_speedup_claim=False),sort_keys=True))
