"""Enable unchanged original Dory setup only after actual parent/child ACK."""
import argparse
import json
import os
from pathlib import Path
import select
import sys

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
from dory_residual_serial_contract_v3 import require, validate_child


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--admission', required=True)
    a = parser.parse_args()
    require(sys.flags.optimize == 0, 'Original assertions enabled')
    spec = json.loads(Path(a.spec).read_text())
    admission = json.loads(Path(a.admission).read_text())
    import minicpm5_layer0_execution_protocol_v4 as low
    owner = validate_child(a.spec, spec, admission, low.proc, low.lock_proof)
    fd = int(os.environ.pop('DORY_RESIDUAL_READY_FD'))
    try:
        require(select.select([fd], [], [], 30)[0] and os.read(fd, 1) == b'R',
                'Parent records actual child identity before native imports')
    finally:
        os.close(fd)
    require(owner == low.proc(os.getpid()) == json.loads((Path(spec['output'])/'child-owner.json').read_text()),
            'Exact actual child start identity after ACK')
    from dory_residual_worker_preparation_v1 import main as original
    original()


if __name__ == '__main__':
    main()
