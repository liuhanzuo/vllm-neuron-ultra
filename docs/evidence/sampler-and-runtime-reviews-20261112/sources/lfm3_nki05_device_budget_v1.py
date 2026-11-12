"""Finite device experiment ceilings, estimates rather than measured NKI0.5 peaks."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
GIB=2**30;MIB=2**20
METADATA_BYTES=8*MIB
ARTIFACT_BYTES=256*MIB
COMPILER_FILE_BYTES=2*GIB
OUTPUT_BYTES=24*GIB
MAX_FILES=32768
ROOT_FLOOR=28*GIB
NVME_FLOOR=30*GIB
WORKER_SECONDS=43200
PHASE_SECONDS=14400
CHILD_SECONDS=1800
SDK_VERIFY_SECONDS=3600
SDK_FILE_BYTES=8*GIB
SDK_TOTAL_BYTES=64*GIB
SDK_FILE_COUNT=131072
PROPS=dict(MemoryMax=str(16*GIB),MemoryHigh=str(14*GIB),MemorySwapMax='0',TasksMax='256',
    AllowedCPUs='6-7',CPUAffinity='6-7',Nice='19',KillMode='control-group',RemainAfterExit='yes',
    SendSIGKILL='yes',WorkingDirectory=str(ROOT),LimitFSIZE=str(COMPILER_FILE_BYTES),
    LimitAS=str(24*GIB),LimitCPU=str(WORKER_SECONDS),RuntimeMaxSec=str(WORKER_SECONDS),TimeoutStopSec='30')


def manifest():
    return dict(properties=PROPS,metadata_bytes=METADATA_BYTES,artifact_bytes=ARTIFACT_BYTES,
        compiler_file_bytes=COMPILER_FILE_BYTES,total_output_bytes=OUTPUT_BYTES,max_files=MAX_FILES,
        root_floor_bytes=ROOT_FLOOR,nvme_floor_bytes=NVME_FLOOR,whole_seconds=WORKER_SECONDS,
        phase_seconds=PHASE_SECONDS,individual_graph_child_seconds=CHILD_SECONDS,
        bulk_original_matrix_child_seconds=PHASE_SECONDS,sdk_verify_seconds=SDK_VERIFY_SECONDS,
        sdk_file_bytes=SDK_FILE_BYTES,sdk_total_bytes=SDK_TOTAL_BYTES,sdk_file_count=SDK_FILE_COUNT,
        measured_new_compiler_peak_bytes=None,measured_new_artifact_bytes=None,
        estimates_are_launch_caps_not_success_predictions=True)
